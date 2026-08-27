"""Environment resolution: the same config must run anywhere."""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from tzsad.utils import env


def test_explicit_paths_always_win(tmp_path, monkeypatch):
    monkeypatch.setenv("TZSAD_DATA_ROOT", str(tmp_path / "ignored"))
    assert env.resolve_data_root(str(tmp_path)) == str(tmp_path)
    assert env.resolve_results_root("/somewhere/else") == "/somewhere/else"


def test_env_var_is_used_when_auto(tmp_path, monkeypatch):
    monkeypatch.setenv("TZSAD_DATA_ROOT", str(tmp_path))
    assert env.resolve_data_root(env.AUTO) == str(tmp_path)
    monkeypatch.setenv("TZSAD_RESULTS", str(tmp_path / "out"))
    assert env.resolve_results_root(env.AUTO) == str(tmp_path / "out")


def test_missing_dataset_fails_loudly_and_lists_where_it_looked(monkeypatch):
    monkeypatch.delenv("TZSAD_DATA_ROOT", raising=False)
    monkeypatch.setattr(env, "_DATA_CANDIDATES", ("/nonexistent/a", "/nonexistent/b"))
    with pytest.raises(FileNotFoundError, match="nonexistent/a"):
        env.resolve_data_root(env.AUTO)


def test_apply_to_config_fills_only_auto_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("TZSAD_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("TZSAD_RESULTS", str(tmp_path / "res"))
    cfg = OmegaConf.create({"data": {"root": "auto"},
                            "paths": {"results": "auto", "cache": "auto"}})
    env.apply_to_config(cfg)
    assert cfg.data.root == str(tmp_path)
    assert cfg.paths.results == str(tmp_path / "res")
    assert cfg.paths.cache == str(tmp_path / "res" / "cache")

    pinned = OmegaConf.create({"data": {"root": "/pinned"},
                               "paths": {"results": "/out", "cache": "/c"}})
    env.apply_to_config(pinned)
    assert (pinned.data.root, pinned.paths.results, pinned.paths.cache) == ("/pinned", "/out", "/c")


def test_detect_environment_returns_a_known_value():
    assert env.detect_environment() in {"kaggle", "colab", "local"}
