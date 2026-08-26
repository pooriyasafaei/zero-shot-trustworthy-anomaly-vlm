"""Determinism helpers: seeding, provenance capture, run directories."""
from __future__ import annotations

import json
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


def start_run(run_dir: Path | str, config: Mapping[str, Any], seed: int) -> RunContext:
    """Seed everything, create ``run_dir`` and drop a provenance manifest in it."""
    seed_everything(seed)
    ctx = RunContext(Path(run_dir), seed, config)
    ctx.write_manifest()
    return ctx


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
