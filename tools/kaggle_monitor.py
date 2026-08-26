#!/usr/bin/env python3
"""Live monitor for the TZ-SAD jobs running on the remote Kaggle session.

Reads GPU utilisation, per-job progress and the newest log lines off the Kaggle
kernel and renders a dashboard. Run it from this machine:

    tools/watch                 # live dashboard, refreshes every 20s
    tools/watch --once          # one snapshot, then exit
    tools/watch -n 60           # custom refresh interval
    tools/watch --logs qwen     # tail one job's log instead

The session URL lives in tools/kaggle_url.txt (gitignored, holds a session token).
When the Kaggle session expires, paste the new proxy URL into that file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field

try:
    import requests
    import websocket
except ImportError:
    sys.exit("missing deps; run:  tools/.venv/bin/pip install requests websocket-client\n"
             "or use the wrapper:  tools/watch")

HERE = os.path.dirname(os.path.abspath(__file__))
URL_FILE = os.path.join(HERE, "kaggle_url.txt")

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, RED, BLUE, GREY = ("\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[90m")


@dataclass
class Job:
    """One tracked background job on the Kaggle box."""

    name: str
    proc_pattern: str
    log: str
    progress_re: str = ""          # regex with groups (done, total)
    done_marker: str = ""          # a log line meaning "finished successfully"
    elapsed: str = ""
    running: bool = False
    done: bool = False
    cur: int = 0
    total: int = 0
    last_line: str = ""
    rate_hist: list = field(default_factory=list)


JOBS = [
    Job("qwen", "run_qwen.py", "/kaggle/working/qwen.log",
        r"qwen scored (\d+)/(\d+) images", "POOLED AUROC"),
    Job("corruption", "run_corruption.py", "/kaggle/working/corruption.log",
        r"", "silent-failure cells"),
    Job("ensemble", "run_clip.py", "/kaggle/working/ensemble.log",
        r"", "POOLED AUROC"),
    Job("bakeoff", "run_bakeoff.py", "/kaggle/working/bakeoff.log",
        r"", "bake-off written"),
    Job("synthetic", "make_synthetic.py", "", r"", ""),
]

_ANSI = re.compile(r"\033\[[0-9;]*m")


def vlen(text: str) -> int:
    """Visible length: ANSI colour codes take no columns but len() counts them."""
    return len(_ANSI.sub("", text))


def pad(text: str, width: int) -> str:
    """Left-align to a visible width, ignoring colour codes."""
    return text + " " * max(0, width - vlen(text))

# Number of (corruption, severity) cells the sweep will run: 1 clean + 6 x 5.
CORRUPTION_CELLS = 31

PROBE = r"""
import subprocess, json, os, re
def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
out = {}
out['gpu'] = sh("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")
out['load'] = sh("cat /proc/loadavg")
out['ncpu'] = sh("nproc")
out['disk'] = sh("df -h /kaggle/working | tail -1 | awk '{print $3\" / \"$2}'")
out['ps'] = sh("ps -eo etime,pcpu,cmd | grep -E 'run_qwen|run_corruption|run_clip|run_bakeoff|make_synthetic' | grep -v grep")
logs = {}
for name, path in %(LOGPATHS)s.items():
    if os.path.exists(path):
        logs[name] = sh("tail -c 60000 " + path)
    else:
        logs[name] = ""
out['logs'] = logs
print("@@JSON@@" + json.dumps(out))
"""


def read_base() -> str:
    if not os.path.exists(URL_FILE):
        sys.exit(f"no session URL. Put the Kaggle proxy URL in:\n  {URL_FILE}")
    return open(URL_FILE).read().strip().rstrip("/")


def kernel_exec(base: str, code: str, timeout: int = 90) -> str:
    """Run code on the remote kernel and return its stdout."""
    r = requests.get(base + "/api/kernels", timeout=20)
    r.raise_for_status()
    kernels = r.json()
    if not kernels:
        raise RuntimeError("no kernel running on the Kaggle session")
    kid = kernels[0]["id"]
    ws = websocket.create_connection(
        base.replace("https://", "wss://") + f"/api/kernels/{kid}/channels",
        timeout=timeout, header={"User-Agent": "Mozilla/5.0"})
    msg_id = uuid.uuid4().hex
    ws.send(json.dumps({
        "header": {"msg_id": msg_id, "username": "watch", "session": uuid.uuid4().hex,
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": False,
                    "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
        "channel": "shell"}))
    chunks, deadline = [], time.time() + timeout
    try:
        while time.time() < deadline:
            ws.settimeout(max(1, deadline - time.time()))
            m = json.loads(ws.recv())
            if m.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            if m["msg_type"] == "stream":
                chunks.append(m["content"]["text"])
            elif m["msg_type"] == "error":
                raise RuntimeError("\n".join(m["content"].get("traceback", []))[-500:])
            elif m["msg_type"] == "status" and m["content"].get("execution_state") == "idle":
                break
    finally:
        ws.close()
    return "".join(chunks)


def bar(pct: float, width: int = 18) -> str:
    pct = max(0.0, min(1.0, pct))
    filled = int(round(pct * width))
    color = GREEN if pct > 0.6 else (YELLOW if pct > 0.2 else RED)
    return f"{color}{'█' * filled}{GREY}{'░' * (width - filled)}{RESET}"


def fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds or seconds > 86400 * 2:
        return "--:--"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_elapsed(etime: str) -> float:
    """ps etime -> seconds."""
    parts = etime.strip().split("-")
    days = int(parts[0]) if len(parts) == 2 else 0
    hms = (parts[-1]).split(":")
    hms = [0] * (3 - len(hms)) + [int(x) for x in hms]
    return days * 86400 + hms[0] * 3600 + hms[1] * 60 + hms[2]


def collect(base: str) -> dict:
    logpaths = {j.name: j.log for j in JOBS}
    code = PROBE % {"LOGPATHS": repr(logpaths)}
    out = kernel_exec(base, code)
    marker = out.rfind("@@JSON@@")
    if marker < 0:
        raise RuntimeError("unexpected probe output: " + out[-300:])
    return json.loads(out[marker + 8:])


def update_jobs(data: dict, jobs: list[Job]) -> None:
    ps = data.get("ps", "")
    for job in jobs:
        job.running = False
        job.elapsed = ""
        for line in ps.splitlines():
            if job.proc_pattern in line:
                fields = line.split(None, 2)
                if len(fields) >= 3:
                    job.elapsed = fields[0]
                job.running = True
                break
        log = data.get("logs", {}).get(job.name, "") or ""
        lines = [ln for ln in log.splitlines() if ln.strip()]
        job.last_line = lines[-1][-160:] if lines else ""
        job.done = bool(job.done_marker) and job.done_marker in log

        if job.progress_re:
            hits = re.findall(job.progress_re, log)
            if hits:
                job.cur, job.total = int(hits[-1][0]), int(hits[-1][1])
        elif job.name == "corruption":
            job.cur = len(re.findall(r"s=\d+\s+AUROC", log)) + log.count("clean AUROC")
            job.total = CORRUPTION_CELLS
        elif job.name == "ensemble":
            job.cur = len(re.findall(r"embedded \S+\s+\S+\s+\d+ images", log))
            job.total = 90       # 3 backbones x 15 categories x 2 splits


def render(data: dict, jobs: list[Job], interval: int) -> str:
    L = []
    now = time.strftime("%H:%M:%S")
    L.append(f"{BOLD}TZ-SAD · Kaggle monitor{RESET}   {DIM}{now}  refresh {interval}s  ctrl-c to quit{RESET}")
    L.append(GREY + "─" * 78 + RESET)

    for row in (data.get("gpu") or "").splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 5:
            continue
        idx, name, util, used, total = parts[:5]
        pct = float(util) / 100.0
        gb_used, gb_total = float(used) / 1024, float(total) / 1024
        L.append(f"  GPU {idx}  " + pad(name, 11) + bar(pct)
                 + f" {int(float(util)):3d}%   {gb_used:4.1f} / {gb_total:.1f} GB")

    load = (data.get("load") or "0 0 0").split()[:3]
    ncpu = data.get("ncpu") or "?"
    try:
        cpu_pct = float(load[0]) / float(ncpu)
    except (ValueError, ZeroDivisionError):
        cpu_pct = 0.0
    L.append("  CPU    " + pad("load", 11) + bar(cpu_pct)
             + f" {load[0]:>4} / {ncpu} cores    disk {data.get('disk', '?')}")
    L.append("")
    L.append("  " + BOLD + pad("JOB", 12) + pad("", 3) + pad("ELAPSED", 11)
             + pad("PROGRESS", 28) + "ETA" + RESET)

    for job in jobs:
        if job.done and not job.running:
            dot, state = f"{GREEN}✓{RESET}", f"{GREEN}done{RESET}"
        elif job.running:
            dot, state = f"{GREEN}●{RESET}", ""
        elif job.last_line:
            dot, state = f"{RED}✗{RESET}", f"{RED}stopped{RESET}"
        else:
            dot, state = f"{GREY}○{RESET}", f"{GREY}queued{RESET}"

        eta, prog = "", state
        if job.running and job.total:
            frac = job.cur / job.total
            prog = f"{bar(frac, 10)} {job.cur}/{job.total} ({frac * 100:.0f}%)"
            secs = parse_elapsed(job.elapsed) if job.elapsed else 0
            if job.cur > 0 and secs > 0:
                eta = fmt_eta(secs / job.cur * (job.total - job.cur))
        elif job.running:
            prog = f"{BLUE}working{RESET}"
        L.append("  " + pad(job.name, 12) + pad(dot, 3) + pad(job.elapsed or "-", 11)
                 + pad(prog, 28) + eta)

    L.append("")
    for job in jobs:
        if job.last_line and (job.running or job.done):
            trimmed = re.sub(r"^\d{4}-\d\d-\d\d ", "", job.last_line)
            L.append(f"  {GREY}{job.name:>10}{RESET} {DIM}{trimmed[:64]}{RESET}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--interval", type=int, default=20, help="refresh seconds")
    ap.add_argument("--once", action="store_true", help="one snapshot then exit")
    ap.add_argument("--logs", metavar="JOB", help="tail one job's log instead")
    args = ap.parse_args()

    base = read_base()
    jobs = JOBS

    if args.logs:
        job = next((j for j in jobs if j.name == args.logs), None)
        if job is None:
            sys.exit(f"unknown job {args.logs!r}; pick from {[j.name for j in jobs]}")
        data = collect(base)
        print(data.get("logs", {}).get(job.name, "")[-8000:])
        return 0

    while True:
        try:
            data = collect(base)
            update_jobs(data, jobs)
            screen = render(data, jobs, args.interval)
        except requests.RequestException as exc:
            screen = f"{RED}cannot reach the Kaggle session{RESET}\n  {exc}\n" \
                     f"  {DIM}the token in {URL_FILE} may have expired{RESET}"
        except Exception as exc:  # noqa: BLE001 - a monitor must not die on one bad poll
            screen = f"{YELLOW}poll failed:{RESET} {exc}"
        if not args.once:
            print("\033[2J\033[H", end="")
        print(screen, flush=True)
        if args.once:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print()
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
