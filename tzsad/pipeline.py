"""Pipeline glue: the stages the CLI scripts compose.

Each stage reads and writes files under a run directory, so any stage can be
re-run on its own. Only :func:`stage_embed` and the Qwen scorer touch a GPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .calibration.conformal import MondrianConformal, coverage_report, n_cal_sensitivity
from .data.mvtec import SubsetSpec, build_index
from .features.cache import EmbeddingCache
from .features.clip_embedder import BackboneSpec, ClipEmbedder, WindowSpec, embed_index
from .features.heatmap import bank_maps, load_windows, save_maps, text_maps
from .records import read_records, write_records
from .scorers.base import ScoringContext
from .scorers.clip_scorer import ClipScorer, ClipScorerConfig
from .uncertainty.image_side import (BackboneEnsemble, NormalManifoldDistance,
                                     NormalizedPromptEnsemble, PromptEnsembleStd, TTAVariance)
from .utils.logging import get_logger

log = get_logger("pipeline")

TTA_VIEWS = {
    "hflip": lambda im: im.transpose(0),                     # PIL.Image.FLIP_LEFT_RIGHT
    "vflip": lambda im: im.transpose(1),                     # PIL.Image.FLIP_TOP_BOTTOM
    "crop90": lambda im: _center_crop(im, 0.90),
    "crop80": lambda im: _center_crop(im, 0.80),
    "jitter": lambda im: _photometric(im, 1.15, 0.9),
}


def _center_crop(img, frac: float):
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def _photometric(img, brightness: float, contrast: float):
    from PIL import ImageEnhance

    return ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(brightness)).enhance(contrast)


def run_dir_for(cfg: DictConfig) -> Path:
    """The results directory for this config."""
    return Path(cfg.paths.results) / str(cfg.paths.run_name)


def backbones_of(cfg: DictConfig) -> list[BackboneSpec]:
    """Backbone specs listed in the config."""
    return [BackboneSpec(name=b["name"], pretrained=b["pretrained"]) for b in cfg.clip.backbones]


def window_spec_of(cfg: DictConfig) -> WindowSpec | None:
    """Window grid, or ``None`` when patch scoring is disabled."""
    w = cfg.clip.windows
    if not w.enabled:
        return None
    return WindowSpec(tuple(w.scales), float(w.stride_frac), int(w.image_size))


def stage_index(cfg: DictConfig) -> pd.DataFrame:
    """Build and persist the shared image index (identical across every scorer)."""
    subset = SubsetSpec.parse(str(cfg.data.subset), seed=int(cfg.seed))
    cal_subset = SubsetSpec.parse(str(cfg.data.get("cal_subset", "full")), seed=int(cfg.seed))
    cats = list(cfg.data.categories) if cfg.data.categories else None
    index = build_index(cfg.data.root, cats, tuple(cfg.data.splits), subset, cal_subset)
    out = run_dir_for(cfg) / "index.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(out, index=False)
    return index


def stage_embed(cfg: DictConfig, index: pd.DataFrame, corruption: str = "none",
                severity: int = 0) -> dict[str, ClipEmbedder]:
    """The GPU pass: cache image (and window, and TTA-view) embeddings per backbone."""
    cache = EmbeddingCache(cfg.paths.cache)
    win = window_spec_of(cfg)
    embedders: dict[str, ClipEmbedder] = {}
    for spec in backbones_of(cfg):
        emb = ClipEmbedder(spec, batch_size=int(cfg.clip.batch_size),
                           allow_download=bool(cfg.clip.allow_download))
        embed_index(emb, index, cache, corruption, severity, int(cfg.seed),
                    windows=win if corruption == "none" else None)
        embedders[spec.cache_tag] = emb
    if cfg.uncertainty.tta.enabled and corruption == "none":
        _embed_tta_views(cfg, index, cache, embedders)
    return embedders


def _embed_tta_views(cfg: DictConfig, index: pd.DataFrame, cache: EmbeddingCache,
                     embedders: Mapping[str, ClipEmbedder]) -> None:
    """Cache one extra shard per TTA view; scoring them is then free."""
    tag = backbones_of(cfg)[0].cache_tag
    embedder = embedders[tag]
    for view in cfg.uncertainty.tta.views:
        fn = TTA_VIEWS[view]
        for (category, split), group in index.groupby(["category", "split"], sort=True):
            keys = dict(model_tag=tag, category=category, split=split, kind="image",
                        corruption="none", severity=0, extra=f"tta-{view}")
            if cache.has(**keys):
                continue
            group = group.sort_values("image_id")
            cache.save(group["image_id"].tolist(),
                       embedder.encode_images(group["path"].tolist(), fn),
                       meta={"tta_view": view}, **keys)
        log.info("cached TTA view %s", view)


def stage_score_clip(cfg: DictConfig, index: pd.DataFrame, embedders: Mapping[str, ClipEmbedder],
                     corruption: str = "none", severity: int = 0, return_text: bool = False):
    """Score every backbone offline from the cache.

    Returns ``{backbone_tag: records}``, or that plus the primary backbone's text
    embeddings when ``return_text`` is set (the heatmap stage needs them).
    """
    cache = EmbeddingCache(cfg.paths.cache)
    sc_cfg = ClipScorerConfig(prompt_set=str(cfg.clip.prompt_set), tau=float(cfg.clip.tau),
                              aggregate=str(cfg.clip.aggregate))
    ctx = ScoringContext(run_id=str(cfg.paths.run_name),
                         subset_tag=SubsetSpec.parse(str(cfg.data.subset)).tag,
                         corruption=corruption, severity=severity)
    out: dict[str, pd.DataFrame] = {}
    primary_text = None
    for i, spec in enumerate(backbones_of(cfg)):
        scorer = ClipScorer(spec, cache, sc_cfg)
        scorer.ensure_text(sorted(index["category"].unique()), embedders[spec.cache_tag])
        scorer.save_text(run_dir_for(cfg) / f"text_{spec.cache_tag}.npz")
        out[spec.cache_tag] = scorer.score_index(index, ctx)
        if i == 0:
            primary_text = scorer.text_embeddings
    return (out, primary_text) if return_text else out


def stage_tta_scores(cfg: DictConfig, index: pd.DataFrame,
                     embedders: Mapping[str, ClipEmbedder]) -> list[pd.DataFrame]:
    """Score each cached TTA view with the primary backbone."""
    cache = EmbeddingCache(cfg.paths.cache)
    spec = backbones_of(cfg)[0]
    sc_cfg = ClipScorerConfig(prompt_set=str(cfg.clip.prompt_set), tau=float(cfg.clip.tau),
                              aggregate=str(cfg.clip.aggregate), keep_per_template=False)
    frames: list[pd.DataFrame] = []
    for view in cfg.uncertainty.tta.views:
        scorer = _ViewScorer(spec, cache, sc_cfg, f"tta-{view}")
        scorer.ensure_text(sorted(index["category"].unique()), embedders[spec.cache_tag])
        frames.append(scorer.score_index(index, ScoringContext(run_id=f"tta-{view}")))
    return frames


class _ViewScorer(ClipScorer):
    """A ClipScorer that reads a named extra cache shard (a TTA view)."""

    def __init__(self, spec, cache, config, extra: str) -> None:
        super().__init__(spec, cache, config)
        self.extra = extra

    def score_index(self, index: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
        frames = []
        for (category, split), group in index.groupby(["category", "split"], sort=True):
            group = group.sort_values("image_id")
            ids, emb, _ = self.cache.load(model_tag=self.spec.cache_tag, category=category,
                                          split=split, kind="image", corruption="none",
                                          severity=0, extra=self.extra)
            lookup = {i: k for k, i in enumerate(ids)}
            rows = emb[[lookup[i] for i in group["image_id"]]]
            frames.append(self._score_block(rows, group, category))
        return self.finalize(pd.concat(frames, ignore_index=True), ctx)


def stage_heatmaps(cfg: DictConfig, index: pd.DataFrame,
                   embedders: Mapping[str, ClipEmbedder],
                   text_embeddings: Mapping[str, tuple] | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Compute and cache both patch maps for test images.

    Returns ``{"bank": {...}, "text": {...}}``. The ``bank`` map (kNN to the
    positionally-matched ``train/good`` window bank) is what the hallucination
    module trusts; the ``text`` map is the WinCLIP-style one usually reported for
    pixel AUROC. Both come from the same cached windows, so neither costs a pass.
    """
    win = window_spec_of(cfg)
    if win is None:
        return {}
    cache = EmbeddingCache(cfg.paths.cache)
    spec = backbones_of(cfg)[0]
    map_size = int(cfg.clip.windows.map_size)
    tau = float(cfg.clip.tau)
    out_dir = run_dir_for(cfg) / "maps"
    maps: dict[str, dict[str, np.ndarray]] = {"bank": {}, "text": {}}

    for category in sorted(index["category"].unique()):
        try:
            _, bank, _ = load_windows(cache, spec.cache_tag, category, "train")
            ids, test_win, meta = load_windows(cache, spec.cache_tag, category, "test")
        except FileNotFoundError:
            log.warning("no window cache for %s; skipping heatmaps", category)
            continue
        boxes = [tuple(b) for b in meta["boxes"]]
        image_size = int(meta["image_size"])

        m = bank_maps(test_win, bank, boxes, image_size, k=int(cfg.clip.windows.bank_k),
                      out_size=map_size, position_aware=bool(cfg.clip.windows.position_aware))
        save_maps(out_dir / f"{category}__bank.npz", ids, m)
        maps["bank"].update(dict(zip(ids, m)))

        if text_embeddings and category in text_embeddings:
            t_norm, t_anom = text_embeddings[category]
            tm = text_maps(test_win, boxes, image_size, t_norm, t_anom, tau, out_size=map_size)
            save_maps(out_dir / f"{category}__text.npz", ids, tm)
            maps["text"].update(dict(zip(ids, tm)))
    return maps


def stage_pixel_eval(cfg: DictConfig, index: pd.DataFrame,
                     maps: Mapping[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """Pixel AUROC and PRO per category, for each available map type."""
    from .eval.pixel import load_mask, pixel_auroc, pro_score

    map_size = int(cfg.clip.windows.map_size)
    test = index[index["split"] == "test"]
    masks: dict[str, np.ndarray] = {}
    for _, r in test.iterrows():
        mp = str(r["mask_path"] or "")
        masks[r["image_id"]] = (load_mask(mp, map_size) if mp
                               else np.zeros((map_size, map_size), dtype=np.uint8))

    rows = []
    for kind, by_id in maps.items():
        if not by_id:
            continue
        for category, g in test.groupby("category", sort=True):
            sel = {i: by_id[i] for i in g["image_id"] if i in by_id}
            if not sel:
                continue
            sub_masks = {i: masks[i] for i in sel}
            rows.append({"map": kind, "category": category, "n": len(sel),
                         "pixel_auroc": pixel_auroc(sel, sub_masks),
                         "pro": pro_score(sel, sub_masks)})
        pooled = {i: by_id[i] for i in test["image_id"] if i in by_id}
        rows.append({"map": kind, "category": "POOLED", "n": len(pooled),
                     "pixel_auroc": pixel_auroc(pooled, {i: masks[i] for i in pooled}),
                     "pro": pro_score(pooled, {i: masks[i] for i in pooled})})
    return pd.DataFrame(rows)


def stage_calibrate(cfg: DictConfig, records: pd.DataFrame, delta: float | None = None,
                    score_col: str = "anomaly_score") -> tuple[pd.DataFrame, MondrianConformal, pd.DataFrame]:
    """Fit Mondrian conformal on ``train/good`` and transform the test records.

    ``score_col`` selects the non-conformity score. Pass ``"s_adj"`` to calibrate on
    the CMCS-attenuated score (proposal Eq. 5), which is what §4.2 of the brief
    specifies once the hallucination module is in play.
    """
    delta = float(cfg.conformal.delta if delta is None else delta)
    cal = records[(records["split"] == "train") & (records["label"] == 0)]
    test = records[records["split"] == "test"].reset_index(drop=True)
    if cal.empty:
        raise ValueError(
            "no train/good records to calibrate on. Include the 'train' split in "
            "data.splits - normal-only calibration is what keeps this zero-shot."
        )
    calib = MondrianConformal(delta=delta, randomized=bool(cfg.conformal.randomized),
                              seed=int(cfg.seed)).fit(
        cal, score_col=score_col, n_cal=cfg.conformal.n_cal if cfg.conformal.n_cal else None)
    out = calib.transform(test, score_col=score_col,
                          uncertainty_mode=str(cfg.conformal.uncertainty_mode))
    out["correct"] = (out["conformal_pred"].to_numpy() == out["label"].to_numpy())
    cov = coverage_report(out, calib, score_col=score_col, n_boot=int(cfg.eval.n_boot))
    return out, calib, cov


def stage_conformal_sweeps(cfg: DictConfig, records: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Sweep delta and calibration-set size; write both tables."""
    cal = records[(records["split"] == "train") & (records["label"] == 0)]
    test = records[records["split"] == "test"].reset_index(drop=True)
    rows = []
    for delta in cfg.conformal.deltas:
        calib = MondrianConformal(delta=float(delta), seed=int(cfg.seed)).fit(cal)
        rows.append(coverage_report(test, calib, n_boot=int(cfg.eval.n_boot)).assign(delta=float(delta)))
    cov = pd.concat(rows, ignore_index=True)
    ncal = n_cal_sensitivity(cal, test, deltas=[float(cfg.conformal.delta)],
                             n_cals=[None if v in (None, "null") else int(v)
                                     for v in cfg.conformal.n_cal_sweep], seed=int(cfg.seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"coverage_sweep": out_dir / "coverage_sweep.csv", "n_cal": out_dir / "n_cal_sensitivity.csv"}
    cov.to_csv(paths["coverage_sweep"], index=False)
    ncal.to_csv(paths["n_cal"], index=False)
    return paths


def stage_uncertainty_clip(cfg: DictConfig, test_records: pd.DataFrame,
                           per_backbone: Mapping[str, pd.DataFrame],
                           tta_frames: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    """Attach every configured image-side uncertainty column to the test records."""
    cache = EmbeddingCache(cfg.paths.cache)
    primary = backbones_of(cfg)[0].cache_tag
    out = test_records
    wanted = set(cfg.uncertainty.estimators)

    if "prompt_std" in wanted:
        out = PromptEnsembleStd().attach(out)
    if "prompt_std_z" in wanted:
        out = NormalizedPromptEnsemble().attach(out)
    if "manifold_knn" in wanted:
        out = NormalManifoldDistance(cache, primary, k=int(cfg.uncertainty.manifold_k)).attach(out)
    if "backbone_ens" in wanted and len(per_backbone) >= 2:
        test_only = {t: f[f["split"] == "test"] for t, f in per_backbone.items()}
        out = BackboneEnsemble().attach(out, score_frames=test_only)
    if "tta_var" in wanted and tta_frames:
        views = [f[f["split"] == "test"] for f in tta_frames]
        out = TTAVariance().attach(out, view_frames=views)
    return out


def save_stage(df: pd.DataFrame, path: Path) -> Path:
    """Write a stage output, preferring parquet."""
    return write_records(df, path)


def load_stage(path: Path) -> pd.DataFrame:
    """Read a stage output back."""
    return read_records(path)


def config_summary(cfg: DictConfig) -> dict[str, Any]:
    """Container form of the config, for the run manifest."""
    return OmegaConf.to_container(cfg, resolve=True)
