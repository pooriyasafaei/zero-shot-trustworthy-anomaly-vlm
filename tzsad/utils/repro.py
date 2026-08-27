"""Determinism helpers: seeding, provenance capture, run directories."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> int:
    """Seed ``random``, ``numpy`` and (if importable) ``torch``.

    Returns the seed so callers can log it. ``deterministic_torch`` also fixes
    cuDNN into deterministic mode, which costs throughput but makes patch-level
    convolutions bit-reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def git_sha(repo_root: Path | str | None = None) -> str:
    """Return the current git SHA, or ``"nogit"`` when not in a repository."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "nogit"


def environment_fingerprint() -> dict[str, Any]:
    """Capture library versions and hardware so a number can be traced later."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for mod in ("torch", "transformers", "open_clip", "sklearn", "pandas"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - optional dependency probing
            info[mod] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        info["cuda_available"] = False
        info["gpus"] = []
    return info


@dataclass
class RunContext:
    """A results directory plus the provenance written into it."""

    run_dir: Path
    seed: int
    config: Mapping[str, Any]

    def write_manifest(self, extra: Mapping[str, Any] | None = None) -> Path:
        """Write ``manifest.json`` (config + seed + git SHA + env) into the run dir."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "seed": self.seed,
            "git_sha": git_sha(),
            "env": environment_fingerprint(),
            "config": _jsonable(self.config),
            "argv": sys.argv,
        }
        if extra:
            manifest["extra"] = _jsonable(extra)
        path = self.run_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return path


#: Config fields whose change means a re-run covers different data, not just
#: different settings - overwriting then silently replaces results at one scale
#: with results at another.
_SCOPE_FIELDS = (("data", "categories"), ("data", "subset"), ("data", "cal_subset"))


def start_run(run_dir: Path | str, config: Mapping[str, Any], seed: int) -> RunContext:
    """Seed everything, create ``run_dir`` and drop a provenance manifest in it.

    Warns loudly when an existing run in the same directory covered a different
    data scope: a small test run reusing a run_name will otherwise overwrite
    full-scale records with a handful of rows, and nothing about the resulting
    files says that happened.
    """
    seed_everything(seed)
    _warn_on_scope_change(Path(run_dir), config)
    ctx = RunContext(Path(run_dir), seed, config)
    ctx.write_manifest()
    return ctx


def _warn_on_scope_change(run_dir: Path, config: Mapping[str, Any]) -> None:
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return
    try:
        previous = json.loads(manifest.read_text()).get("config", {})
    except (OSError, json.JSONDecodeError):
        return
    changed = []
    for section, field in _SCOPE_FIELDS:
        was = (previous.get(section) or {}).get(field)
        now = (dict(config).get(section) or {}).get(field)
        if was != now:
            changed.append(f"{section}.{field}: {was!r} -> {now!r}")
    if changed:
        logging.getLogger("tzsad.repro").warning(
            "run directory %s already holds a run over a DIFFERENT data scope and is "
            "about to be overwritten (%s). Use a distinct paths.run_name, or a separate "
            "paths.results, if you did not mean to replace it.",
            run_dir, "; ".join(changed))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
