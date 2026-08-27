"""Environment detection, so the same config runs on Kaggle, Colab or a laptop.

The dataset root and the results directory are the only things that genuinely
differ between machines. Hard-coding them into configs means every launch needs a
pile of overrides and a config checked into git is not reproducible anywhere but
the machine it was written on. This module resolves them from the environment,
and any explicit value in the config still wins.
"""
from __future__ import annotations

import os
from pathlib import Path

from .logging import get_logger

log = get_logger("utils.env")

#: Marker that a config value should be filled in from the environment.
AUTO = "auto"

_DATA_CANDIDATES = (
    "/kaggle/input/datasets/ipythonx/mvtec-ad",
    "/kaggle/input/mvtec-ad",
    "/kaggle/input/mvtec_ad",
    "/content/mvtec_ad",
    "~/data/mvtec_ad",
    "./data/mvtec_ad",
)

_RESULTS_CANDIDATES = (
    ("/kaggle/working", "/kaggle/working/results"),
    ("/content", "/content/results"),
)


def detect_environment() -> str:
    """``kaggle`` | ``colab`` | ``local``."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content").is_dir():
        return "colab"
    return "local"


def resolve_data_root(value: str | None = None) -> str:
    """Find the dataset root, preferring an explicit value or ``TZSAD_DATA_ROOT``.

    Fails loudly with the list of places searched rather than letting an empty
    index surface as a confusing error much later.
    """
    if value and value != AUTO:
        return str(Path(value).expanduser())
    env = os.environ.get("TZSAD_DATA_ROOT")
    if env:
        return str(Path(env).expanduser())
    for candidate in _DATA_CANDIDATES:
        path = Path(candidate).expanduser()
        if path.is_dir():
            log.info("resolved data root -> %s", path)
            return str(path)
    raise FileNotFoundError(
        "could not locate the MVTec-AD root. Set data.root in the config, export "
        f"TZSAD_DATA_ROOT, or place it at one of: {', '.join(_DATA_CANDIDATES)}"
    )


def resolve_results_root(value: str | None = None) -> str:
    """Where run outputs go. Prefers writable session storage on hosted runtimes."""
    if value and value != AUTO:
        return str(Path(value).expanduser())
    env = os.environ.get("TZSAD_RESULTS")
    if env:
        return str(Path(env).expanduser())
    for marker, results in _RESULTS_CANDIDATES:
        if Path(marker).is_dir():
            return results
    return "results"


def apply_to_config(cfg) -> None:
    """Fill ``auto`` placeholders in a loaded config, in place.

    Explicit values are never overwritten, so a launch can still pin any path.
    """
    if str(cfg.data.get("root", AUTO)) in (AUTO, "", "None"):
        cfg.data.root = resolve_data_root()
    results = str(cfg.paths.get("results", AUTO))
    if results in (AUTO, "", "None"):
        cfg.paths.results = resolve_results_root()
    if str(cfg.paths.get("cache", AUTO)) in (AUTO, "", "None"):
        cfg.paths.cache = str(Path(cfg.paths.results) / "cache")
    log.info("environment=%s data.root=%s results=%s", detect_environment(),
             cfg.data.root, cfg.paths.results)
