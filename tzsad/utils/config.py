"""Config loading. Every experiment is fully specified by one yaml file."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a yaml config, resolve a ``defaults:`` parent chain, apply CLI overrides.

    ``defaults`` is a list of config file names (relative to ``configs/``) that are
    merged left-to-right before the current file's own keys, so a config can say
    ``defaults: [base.yaml]`` and override a handful of fields.
    """
    path = Path(path)
    if not path.exists():
        candidate = CONFIG_ROOT / path
        if not candidate.exists():
            raise FileNotFoundError(
                f"config not found: {path} (also tried {candidate}). "
                "Every experiment must be specified by a config file."
            )
        path = candidate

    cfg = _load_with_defaults(path, seen=set())
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg  # type: ignore[return-value]


def _load_with_defaults(path: Path, seen: set[Path]) -> DictConfig:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular config defaults involving {path}")
    seen.add(path)
    node = OmegaConf.load(path)
    parents = node.pop("defaults", []) if "defaults" in node else []
    merged = OmegaConf.create({})
    for parent in parents:
        parent_path = (path.parent / parent) if (path.parent / parent).exists() else CONFIG_ROOT / parent
        merged = OmegaConf.merge(merged, _load_with_defaults(parent_path, seen))
    return OmegaConf.merge(merged, node)  # type: ignore[return-value]


def to_dict(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Container-ify a config for JSON serialisation."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return copy.deepcopy(dict(cfg))
