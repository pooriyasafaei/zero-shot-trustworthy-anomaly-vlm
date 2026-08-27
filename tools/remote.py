#!/usr/bin/env python3
"""Drive a remote GPU session (Kaggle/Colab Jupyter kernel) from this machine.

Everything needed to move between environments lives here, in the repo, because
scratch directories do not survive a session change:

    tools/remote status                 # what is on the box, what is running
    tools/remote deploy                 # push the working tree to the session
    tools/remote exec 'print(1)'        # run python on the kernel
    tools/remote sh 'nvidia-smi'        # run a shell command
    tools/remote run --gpu 1 --name qwen scripts/run_qwen.py --config configs/qwen_subset.yaml
    tools/remote pull                   # copy results back here (durable storage)
    tools/remote logs qwen              # tail a job log

The session URL is read from tools/kaggle_url.txt (gitignored; it holds a token).

Design note: `deploy` never deletes the remote tree. Overwriting in place keeps
running processes alive, and their working directory is the thing being replaced.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tarfile
import time
import uuid
from pathlib import Path

try:
    import requests
    import websocket
except ImportError:
    sys.exit("run through the wrapper: tools/remote (it uses tools/.venv)")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
URL_FILE = HERE / "kaggle_url.txt"
REMOTE_ROOT = "/kaggle/working/tzsad"
REMOTE_RESULTS = "/kaggle/working/results"
CHUNK = 400_000

EXCLUDE_DIRS = {".git", "notebooks", "sources", "results", "__pycache__",
                ".venv", ".pytest_cache", "tools"}


def base_url() -> str:
    if not URL_FILE.exists():
        sys.exit(f"no session URL; put the proxy URL in {URL_FILE}")
    return URL_FILE.read_text().strip().rstrip("/")


def kernel_exec(code: str, timeout: int = 600, quiet: bool = False) -> tuple[str, str]:
    """Execute python on the remote kernel. Returns (status, combined output)."""
    base = base_url()
    try:
        r = requests.get(base + "/api/kernels", timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        sys.exit(f"cannot reach the session ({exc}). The token may have expired; "
                 f"paste a fresh proxy URL into {URL_FILE}")
    kernels = r.json()
    if not kernels:
        sys.exit("no kernel running on the session; open the notebook once to start one")
    ws = websocket.create_connection(
        base.replace("https://", "wss://") + f"/api/kernels/{kernels[0]['id']}/channels",
        timeout=timeout, header={"User-Agent": "Mozilla/5.0"})
    msg_id = uuid.uuid4().hex
    ws.send(json.dumps({
        "header": {"msg_id": msg_id, "username": "remote", "session": uuid.uuid4().hex,
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": False,
                    "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
        "channel": "shell"}))
    chunks, status, deadline = [], "ok", time.time() + timeout
    try:
        while time.time() < deadline:
            ws.settimeout(max(1, deadline - time.time()))
            try:
                m = json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:
                chunks.append("\n[remote] timed out waiting for the kernel\n")
                status = "timeout"
                break
            if m.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            kind, content = m["msg_type"], m["content"]
            if kind == "stream":
                chunks.append(content["text"])
                if not quiet:
                    sys.stdout.write(content["text"])
                    sys.stdout.flush()
            elif kind in ("execute_result", "display_data"):
                text = content.get("data", {}).get("text/plain", "")
                chunks.append(text + "\n")
                if not quiet:
                    print(text)
            elif kind == "error":
                status = "error"
                text = "\n".join(content.get("traceback", []))
                chunks.append(text)
                if not quiet:
                    print(text)
            elif kind == "status" and content.get("execution_state") == "idle":
                break
    finally:
        ws.close()
    return status, "".join(chunks)


def shell(cmd: str, timeout: int = 600, quiet: bool = False) -> tuple[str, str]:
    """Run a shell command on the remote box."""
    return kernel_exec(
        "import subprocess;"
        f"r = subprocess.run({cmd!r}, shell=True, capture_output=True, text=True);"
        "print(r.stdout + r.stderr)", timeout=timeout, quiet=quiet)


# -- deploy ----------------------------------------------------------------
def build_archive() -> bytes:
    """Tar the working tree, excluding heavyweight and environment-specific paths."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(REPO.rglob("*")):
            rel = path.relative_to(REPO)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            if path.is_file() and not path.name.endswith((".pyc", ".ipynb")):
                tar.add(path, arcname=str(rel))
    return buf.getvalue()


def cmd_deploy(args: argparse.Namespace) -> int:
    blob = base64.b64encode(build_archive()).decode()
    parts = [blob[i:i + CHUNK] for i in range(0, len(blob), CHUNK)]
    print(f"uploading {len(blob) / 1e3:.0f} KB in {len(parts)} chunk(s)")
    kernel_exec("open('/kaggle/working/_up.b64','w').close()", quiet=True)
    for i, part in enumerate(parts, 1):
        st, out = kernel_exec(f"open('/kaggle/working/_up.b64','a').write({part!r})", quiet=True)
        if st != "ok":
            print(out[-500:])
            return 1
        print(f"  chunk {i}/{len(parts)}")
    st, out = kernel_exec(f"""
import base64, pathlib, tarfile, io, os
raw = base64.b64decode(pathlib.Path('/kaggle/working/_up.b64').read_text())
os.makedirs({REMOTE_ROOT!r}, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw)) as t:
    t.extractall({REMOTE_ROOT!r})       # overwrite in place; never rmtree a live tree
os.remove('/kaggle/working/_up.b64')
print('deployed', len(raw), 'bytes ->', {REMOTE_ROOT!r})
""")
    return 0 if st == "ok" else 1


# -- run / status / logs ---------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    """Launch a script detached on the box so it survives this connection."""
    script = args.script
    extra = " ".join(a for a in args.args if a != "--")
    name = args.name or Path(script).stem
    gpu = f"CUDA_VISIBLE_DEVICES={args.gpu} " if args.gpu is not None else ""
    log = f"/kaggle/working/{name}.log"
    cmd = (f"cd {REMOTE_ROOT} && {gpu}nohup python {script} {extra} > {log} 2>&1 &")
    kernel_exec(f"import subprocess; subprocess.Popen({cmd!r}, shell=True)", quiet=True)
    print(f"launched {name} (gpu={args.gpu if args.gpu is not None else 'all'}) -> {log}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    _, out = shell(
        "echo '--- gpu ---'; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader; echo '--- load ---'; uptime; "
        "echo '--- running ---'; ps -eo etime,pcpu,cmd | grep -E 'scripts/run_' | grep -v grep; "
        f"echo '--- results ---'; ls {REMOTE_RESULTS} 2>/dev/null | head -20; "
        f"echo '--- disk ---'; df -h /kaggle/working | tail -1", quiet=True)
    print(out)
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    _, out = shell(f"tail -n {args.lines} /kaggle/working/{args.name}.log", quiet=True)
    print(out)
    return 0


def cmd_exec(args: argparse.Namespace) -> int:
    st, _ = kernel_exec(args.code, timeout=args.timeout)
    return 0 if st == "ok" else 1


def cmd_sh(args: argparse.Namespace) -> int:
    st, _ = shell(args.cmd, timeout=args.timeout)
    return 0 if st == "ok" else 1


# -- pull ------------------------------------------------------------------
def cmd_pull(args: argparse.Namespace) -> int:
    """Copy result artefacts back here, so they survive the session.

    Embedding caches are excluded by default: they are hundreds of MB and are
    rebuildable from the dataset, whereas records, reports and figures are the
    outputs that must not be lost.
    """
    dest = REPO / args.dest
    dest.mkdir(parents=True, exist_ok=True)
    include = "-o -name '*.csv' -o -name '*.parquet' -o -name '*.json' -o -name '*.jsonl' " \
              "-o -name '*.pdf' -o -name '*.png' -o -name '*.log'"
    st, out = kernel_exec(f"""
import base64, io, os, tarfile
root = {REMOTE_RESULTS!r}
buf = io.BytesIO()
kept = 0
with tarfile.open(fileobj=buf, mode='w:gz') as t:
    for dirpath, dirnames, filenames in os.walk(root):
        if 'cache' in dirpath.split(os.sep) and not {args.include_cache!r}:
            continue
        for fn in filenames:
            if not fn.endswith(('.csv','.parquet','.json','.jsonl','.pdf','.png','.log','.npz')):
                continue
            full = os.path.join(dirpath, fn)
            if not {args.include_cache!r} and full.endswith('.npz') and 'cache' in full:
                continue
            if os.path.getsize(full) > {args.max_mb} * 1e6:
                continue
            t.add(full, arcname=os.path.relpath(full, root))
            kept += 1
data = base64.b64encode(buf.getvalue()).decode()
print('FILES', kept, 'BYTES', len(data))
open('/kaggle/working/_down.b64','w').write(data)
""", quiet=True)
    if st != "ok":
        print(out[-800:])
        return 1
    print(out.strip().splitlines()[-1])
    _, size_out = kernel_exec("print(os.path.getsize('/kaggle/working/_down.b64'))", quiet=True)
    total = int(size_out.strip().splitlines()[-1])
    parts = []
    for off in range(0, total, CHUNK):
        _, chunk = kernel_exec(
            f"print(open('/kaggle/working/_down.b64').read()[{off}:{off + CHUNK}], end='')",
            quiet=True)
        parts.append(chunk)
        print(f"  {min(off + CHUNK, total) / 1e3:.0f}/{total / 1e3:.0f} KB", end="\r")
    raw = base64.b64decode("".join(parts))
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        t.extractall(dest)
    kernel_exec("import os; os.remove('/kaggle/working/_down.b64')", quiet=True)
    n = sum(1 for _ in dest.rglob("*") if _.is_file())
    print(f"\npulled into {dest} ({n} files)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("deploy", help="push the working tree to the session").set_defaults(fn=cmd_deploy)
    sub.add_parser("status", help="GPUs, running jobs, results").set_defaults(fn=cmd_status)

    # argparse.REMAINDER swallows everything after the script, so --gpu/--name must
    # be declared (and passed) before it.
    p = sub.add_parser("run", help="launch a script detached on the box",
                       usage="tools/remote run [--gpu N] [--name NAME] SCRIPT [ARGS...]")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--name", default=None, help="log name (default: script stem)")
    p.add_argument("script")
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("logs", help="tail a job log")
    p.add_argument("name")
    p.add_argument("-n", "--lines", type=int, default=40)
    p.set_defaults(fn=cmd_logs)

    p = sub.add_parser("exec", help="run python on the kernel")
    p.add_argument("code")
    p.add_argument("-t", "--timeout", type=int, default=600)
    p.set_defaults(fn=cmd_exec)

    p = sub.add_parser("sh", help="run a shell command on the box")
    p.add_argument("cmd")
    p.add_argument("-t", "--timeout", type=int, default=600)
    p.set_defaults(fn=cmd_sh)

    p = sub.add_parser("pull", help="copy results back to this machine")
    p.add_argument("--dest", default="results_remote")
    p.add_argument("--include-cache", action="store_true", help="also pull embedding caches")
    p.add_argument("--max-mb", type=float, default=25.0)
    p.set_defaults(fn=cmd_pull)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
