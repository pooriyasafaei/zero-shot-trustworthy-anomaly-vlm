"""CERM-AD Stage I+ orchestration: bidirectional, artifact-matched causal verification.

Per anomalous target ``I`` (mask ``M``, normal donor ``J``), every intervention is
built from pixels first - all control selection happens here, with no model in
scope - and only then scored:

    necessity   original, defect_removed (shared by arms A-C)
                A random      K translated masks, uniform over admissible positions
                B context     the Stage-I context-matched placebos, reproduced exactly
                C artifact    context + artifact matched placebos (new)
                D occlusion   mean-colour fill of the defect and of K controls
    sufficiency recipient J, J_defect, K artifact-matched normal-patch shams

The Stage-I module is untouched; arm B calls it with Stage-I parameters, so the
v1 effect is recomputed on the same targets in the same run and pairs exactly.

All images of a target go through :func:`~tzsad.causal.scoring.score_images`, which
scores each distinct image once (byte-identical images in different arms share a
forward pass) and caches every score as it is produced. CPU construction of the next
target overlaps GPU scoring of the current one.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..scorers.prompts import class_name_for
from ..utils.budget import TimeBudget
from ..utils.checkpoint import RecordCheckpoint
from ..utils.logging import get_logger
from PIL import Image

from .artifact_matching import (ArmPlan, MaskTemplate, artifact_matched_placebos,
                                context_arm_from_stage1, occlusion_fill, random_placebos)
from .audit import CausalConfig, ScoreCache, assign_shards, select_targets
from .donor import DonorBank, aligned_donor_pixels, load_mask, load_rgb
from .effects import compute_effects
from .placebo import feasible_candidates, generate_placebos
from .reconstruction import Intervention, apply_intervention, identity_intervention
from .scoring import score_images
from .sufficiency import ShamPlan, SourceBank, select_shams, source_array

log = get_logger("causal.audit_v2")

NECESSITY_ARMS = ("random", "context", "artifact")
ALL_ARMS = ("random", "context", "artifact", "occlusion", "sufficiency")

#: Roles whose image is written to disk for every target (casebook + structured P1).
SAVED_ROLES = ("defect_removed", "context_placebo_0", "random_placebo_0",
               "artifact_placebo_best", "occluded_defect", "recipient",
               "defect_inserted", "sham_best")


@dataclass
class V2Config:
    """Stage I+ settings on top of the Stage-I :class:`CausalConfig`."""

    base: CausalConfig = field(default_factory=CausalConfig)
    k_random: int = 19
    k_context: int = 19
    k_artifact: int = 19
    artifact_max_candidates: int = 1000
    artifact_match_quantile: float = 0.3
    k_occlusion: int = 9
    k_sham: int = 9
    max_sham_sources: int = 60
    sham_match_quantile: float = 0.3
    batch_size: int = 6
    max_pixels: int = 0          # scoring resolution and dtype enter the cache key
    dtype: str = ""

    def stage1(self) -> CausalConfig:
        """The Stage-I config with arm B's K, for the exact reproduction."""
        from dataclasses import replace
        return replace(self.base, k_placebos=int(self.k_context))


@dataclass
class Item:
    """One image to score, plus where it came from."""

    role: str                 # original | defect_removed | placebo | occluded_defect | recipient | defect_inserted | sham
    arm: str                  # shared | necessity | random | context | artifact | occlusion | sufficiency | sham
    iv: Intervention
    extra: dict = field(default_factory=dict)
    save_as: str = ""


@dataclass
class BuiltTarget:
    """Everything constructed for one target before any model call."""

    base: dict
    class_name: str
    items: list[Item] = field(default_factory=list)
    plans: dict = field(default_factory=dict)
    sham_plan: ShamPlan | None = None
    donor: dict = field(default_factory=dict)
    saved: dict = field(default_factory=dict)
    build_seconds: float = 0.0
    fast_path_drift: float = 0.0
    error: str = ""


def quadrant(cee_n: float, cee_s: float) -> str:
    """Descriptive quadrant of the (necessity, sufficiency) plane. Not a score."""
    if not (np.isfinite(cee_n) and np.isfinite(cee_s)):
        return "undetermined"
    if cee_n > 0 and cee_s > 0:
        return "bidirectional"
    if cee_n > 0:
        return "necessity_only"
    if cee_s > 0:
        return "sufficiency_only"
    return "neither"


def _safe(image_id: str) -> str:
    return image_id.replace("/", "__")


def build_target(row: pd.Series, bank: DonorBank, sources: SourceBank, cfg: V2Config,
                 images_dir: Path | None = None) -> BuiltTarget:
    """Construct every intervention for one target. Pixels only - no model here."""
    from scipy import ndimage

    t0 = time.monotonic()
    c = cfg.base
    image_id = str(row["image_id"])
    category = str(row["category"])
    original = load_rgb(row["path"])
    mask = load_mask(row["mask_path"], original.shape[:2])
    base = {"image_id": image_id, "category": category,
            "defect_type": str(row["defect_type"]), "path": str(row["path"]),
            "mask_path": str(row["mask_path"]), "mask_area": int(mask.sum()),
            "mask_fraction": float(mask.mean()), "mask_source": c.mask_source}
    built = BuiltTarget(base, class_name_for(category))
    if mask.sum() == 0:
        built.error = "empty_mask"
        return built

    # ---- shared: the Stage-I donor and the defect-removal image -------------
    exclude = ndimage.binary_dilation(mask, iterations=max(c.dilate_for_donor, 1))
    donor = bank.select(str(row["path"]), category, exclude, c.max_donor_candidates)
    donor_px = aligned_donor_pixels(donor.donor_path, original.shape[:2])
    built.donor = donor.as_dict()
    tmpl = MaskTemplate(mask, c.feather_px)
    defect_iv = tmpl.apply(original, donor_px, "defect_removed")
    built.items.append(Item("original", "shared", identity_intervention(original)))
    built.items.append(Item("defect_removed", "necessity", defect_iv, save_as="defect_removed"))

    # ---- B: Stage-I context arm, called exactly as Stage I calls it ----------
    s1 = cfg.stage1()
    stage1_plan = generate_placebos(
        original, mask, s1.k_placebos, s1.seed, image_id, max_iou=s1.max_iou,
        ring_width=s1.ring_width, min_separation=s1.min_separation,
        candidate_stride=s1.candidate_stride, match_quantile=s1.match_quantile,
        min_pool=s1.min_pool, use_gradient=s1.use_gradient, max_candidates=s1.max_candidates)
    plans: dict[str, ArmPlan] = {"context": context_arm_from_stage1(stage1_plan, int(mask.sum()))}

    # ---- A and C share one admissible-translation sweep ----------------------
    feasible, _ = feasible_candidates(
        original, mask, max_iou=c.max_iou, ring_width=c.ring_width,
        candidate_stride=c.candidate_stride, use_gradient=c.use_gradient,
        max_candidates=cfg.artifact_max_candidates)
    plans["random"] = random_placebos(feasible, cfg.k_random, c.seed, image_id,
                                      int(mask.sum()), c.min_separation)
    plans["artifact"] = artifact_matched_placebos(
        original, donor_px, mask, feasible, defect_iv.edit_energy, defect_iv.seam_score,
        cfg.k_artifact, c.seed, image_id, cfg.artifact_match_quantile, c.min_separation,
        c.feather_px, tmpl)

    best_artifact = (min(plans["artifact"].placebos, key=lambda p: p.artifact_match_distance).index
                     if plans["artifact"].placebos else -1)
    for arm in NECESSITY_ARMS:
        for p in plans[arm].placebos:
            iv = tmpl.apply(original, donor_px, "placebo", p.dy, p.dx)
            save_as = ""
            if arm in ("context", "random") and p.index == 0:
                save_as = f"{arm}_placebo_0"
            elif arm == "artifact" and p.index == best_artifact:
                save_as = "artifact_placebo_best"
            built.items.append(Item("placebo", arm, iv, p.as_dict(), save_as))

    # ---- D: occlusion-style baseline at the defect + the first K context positions
    fill = occlusion_fill(original)
    built.items.append(Item("occluded_defect", "occlusion",
                            tmpl.apply(original, fill, "occluded_defect"),
                            save_as="occluded_defect"))
    occ_positions = plans["context"].placebos[:cfg.k_occlusion]
    occ = ArmPlan("occlusion", n_requested=cfg.k_occlusion, n_feasible=plans["context"].n_feasible,
                  n_pool=len(plans["context"].placebos))
    for p in occ_positions:
        iv = tmpl.apply(original, fill, "occlusion_control", p.dy, p.dx)
        occ.placebos.append(p)
        built.items.append(Item("placebo", "occlusion", iv, {**p.as_dict(), "arm": "occlusion"}))
    if occ.n_valid < cfg.k_occlusion:
        occ.quality = "few_feasible" if occ.n_valid else "degenerate"
        occ.reason = f"{occ.n_valid} context positions available for k={cfg.k_occlusion}"
    plans["occlusion"] = occ

    # ---- sufficiency: transplant the defect into the donor, plus shams --------
    recipient = donor_px
    insert_iv = tmpl.apply(recipient, original, "defect_inserted")
    built.items.append(Item("recipient", "sufficiency", identity_intervention(recipient),
                            save_as="recipient"))
    built.items.append(Item("defect_inserted", "sufficiency", insert_iv, save_as="defect_inserted"))
    srcs = sources.sources(category, original.shape[:2])
    sham_plan = select_shams(recipient, mask, insert_iv.edit_energy, insert_iv.seam_score,
                             srcs, donor.donor_image_id, cfg.k_sham, c.seed, image_id,
                             cfg.sham_match_quantile, c.feather_px, tmpl)
    best_sham = (min(sham_plan.shams, key=lambda s: s.artifact_match_distance).index
                 if sham_plan.shams else -1)
    for s in sham_plan.shams:
        iv = tmpl.apply(recipient, source_array(srcs, s.source_image_id), "sham")
        built.items.append(Item("sham", "sufficiency", iv, s.as_dict(),
                                "sham_best" if s.index == best_sham else ""))
    built.plans, built.sham_plan = plans, sham_plan
    built.fast_path_drift = _fast_path_drift(built, original, donor_px, recipient, mask, c)

    if images_dir is not None and c.save_images:
        target_dir = Path(images_dir) / _safe(image_id)
        for item in built.items:
            if item.save_as:
                target_dir.mkdir(parents=True, exist_ok=True)
                path = target_dir / f"{item.save_as}.png"
                Image.fromarray(item.iv.image).save(path, compress_level=1)
                built.saved[item.save_as] = str(path)
    built.build_seconds = time.monotonic() - t0
    return built


def _fast_path_drift(built: BuiltTarget, original, donor_px, recipient, mask, c) -> float:
    """Recompute a few interventions with the reference Stage-I implementation.

    The window fast path is exact by construction; this keeps it honest on real
    data. Returns the largest metric difference, and raises if any image differs.
    """
    checks = [(it, original, donor_px) for it in built.items if it.role == "defect_removed"]
    checks += [(it, original, donor_px) for it in built.items
               if it.role == "placebo" and it.arm == "artifact"][:1]
    checks += [(it, recipient, original) for it in built.items if it.role == "defect_inserted"]
    drift = 0.0
    for it, base_img, fill in checks:
        ref = apply_intervention(base_img, fill, it.iv.mask, it.iv.kind, c.feather_px)
        if not np.array_equal(ref.image, it.iv.image):
            raise AssertionError(f"fast-path image differs from reference for {it.role}")
        drift = max(drift, abs(ref.edit_energy - it.iv.edit_energy),
                    abs(ref.seam_score - it.iv.seam_score))
    return float(drift)


def _effect_columns(arm: str, delta: float, controls: list[float]) -> dict:
    eff = compute_effects(delta, controls).as_dict()
    return {f"{arm}_{k}": v for k, v in eff.items()}


def _balance_columns(arm: str, defect: Intervention, controls: list[Intervention]) -> dict:
    """Artifact balance of one arm: defect-arm intervention vs its controls' mean."""
    p = f"{arm}_"
    if not controls:
        return {p + "seam_ratio": float("nan"), p + "edit_ratio": float("nan"),
                p + "artifact_imbalance": float("nan")}
    seam = float(np.mean([iv.seam_score for iv in controls]))
    edit = float(np.mean([iv.edit_energy for iv in controls]))
    seam_ratio = defect.seam_score / seam if seam > 0 else float("nan")
    edit_ratio = defect.edit_energy / edit if edit > 0 else float("nan")
    return {p + "defect_seam": defect.seam_score, p + "control_seam_mean": seam,
            p + "defect_edit": defect.edit_energy, p + "control_edit_mean": edit,
            p + "seam_ratio": seam_ratio, p + "edit_ratio": edit_ratio,
            p + "artifact_imbalance": float(abs(np.log(seam_ratio)) + abs(np.log(edit_ratio)))
            if seam_ratio > 0 and edit_ratio > 0 else float("nan")}


def score_target(built: BuiltTarget, probe, cache: ScoreCache | None,
                 batch_size: int = 1) -> tuple[dict, list[dict]]:
    """Score a built target and assemble its causal record and intervention rows."""
    scores, digests, hits = score_images([it.iv.image for it in built.items],
                                         built.class_name, probe, cache, batch_size)
    z = [float(s.get("audit_logit", float("nan"))) for s in scores]
    rows = []
    for i, (it, s) in enumerate(zip(built.items, scores)):
        rows.append({**built.base, **it.iv.as_dict(), **built.donor, **it.extra,
                     "role": it.role, "arm": it.extra.get("arm", it.arm),
                     "audit_p_yes": s.get("audit_p_yes"), "audit_logit": s.get("audit_logit"),
                     "probe_version": s.get("probe_version", ""), "image_digest": digests[i],
                     "cache_hit": hits[i], "image_output_path": built.saved.get(it.save_as, ""),
                     "saved_role": it.save_as})

    def pick(role: str, arm: str | None = None) -> list[int]:
        return [i for i, it in enumerate(built.items)
                if it.role == role and (arm is None or (it.extra.get("arm", it.arm) == arm))]

    i0, idef = pick("original")[0], pick("defect_removed")[0]
    z0 = z[i0]
    rec: dict = {**built.base, "status": "ok", **built.donor,
                 "audit_p_yes_original": scores[i0].get("audit_p_yes"),
                 "audit_logit_original": z0,
                 "audit_p_yes_defect_removed": scores[idef].get("audit_p_yes"),
                 "audit_logit_defect_removed": z[idef],
                 "original_direct_prediction": "YES" if z0 >= 0 else "NO",
                 "original_is_correct_for_anomaly": bool(z0 >= 0),
                 "build_seconds": built.build_seconds}
    delta_defect = z0 - z[idef]
    rec["delta_defect"] = delta_defect
    for arm in NECESSITY_ARMS:
        idx = pick("placebo", arm)
        rec.update(_effect_columns(arm, delta_defect, [z0 - z[i] for i in idx]))
        rec.update(_balance_columns(arm, built.items[idef].iv, [built.items[i].iv for i in idx]))
        rec.update(built.plans[arm].as_dict())
        rec[f"{arm}_mean_context_distance"] = (
            float(np.mean([built.items[i].extra.get("context_distance", np.nan) for i in idx]))
            if idx else float("nan"))

    iocc = pick("occluded_defect")[0]
    occ_idx = pick("placebo", "occlusion")
    rec["occlusion_delta_defect_raw"] = z0 - z[iocc]
    rec.update(_effect_columns("occlusion", z0 - z[iocc], [z0 - z[i] for i in occ_idx]))
    rec.update(_balance_columns("occlusion", built.items[iocc].iv,
                                [built.items[i].iv for i in occ_idx]))
    rec.update(built.plans["occlusion"].as_dict())

    irec, iins = pick("recipient")[0], pick("defect_inserted")[0]
    sham_idx = pick("sham")
    z_rec = z[irec]
    delta_insert = z[iins] - z_rec
    rec.update({"audit_p_yes_recipient": scores[irec].get("audit_p_yes"),
                "audit_logit_recipient": z_rec,
                "audit_p_yes_defect_inserted": scores[iins].get("audit_p_yes"),
                "audit_logit_defect_inserted": z[iins],
                "recipient_direct_prediction": "YES" if z_rec >= 0 else "NO",
                "delta_insert_defect": delta_insert})
    rec.update(_effect_columns("sufficiency", delta_insert, [z[i] - z_rec for i in sham_idx]))
    rec.update(_balance_columns("sufficiency", built.items[iins].iv,
                                [built.items[i].iv for i in sham_idx]))
    rec.update(built.sham_plan.as_dict() if built.sham_plan else {})

    # Headline aliases, named as in docs/CERM_AD_V2.md.
    rec.update({
        "cee_n_random": rec["random_cee"], "cee_n_context": rec["context_cee"],
        "cee_n_artifact": rec["artifact_cee"], "cee_n_occlusion": rec["occlusion_cee"],
        "robust_cee_n_z": rec["artifact_robust_cee_z"],
        "cee_s": rec["sufficiency_cee"], "robust_cee_s_z": rec["sufficiency_robust_cee_z"],
        "empirical_sham_tail": rec["sufficiency_empirical_placebo_p"],
        "defect_insert_beats_all_shams": rec["sufficiency_defect_beats_all_placebos"],
        "quadrant": quadrant(rec["artifact_cee"], rec["sufficiency_cee"]),
        "max_outside_change": float(max(it.iv.outside_change_max for it in built.items)),
        "n_forward_images": len(built.items),
        "n_unique_images": len(set(digests)),
        "saved_images": json.dumps(built.saved, sort_keys=True),
        "probe_version": scores[i0].get("probe_version", ""),
    })
    # Pixel metrics used for matching must equal the full-image metrics of the
    # chosen interventions; any drift would mean the matching optimised the wrong thing.
    drift = [abs(float(it.extra["candidate_edit_energy"]) - it.iv.edit_energy)
             + abs(float(it.extra["candidate_seam_score"]) - it.iv.seam_score)
             for it in built.items if "candidate_edit_energy" in it.extra
             and np.isfinite(float(it.extra["candidate_edit_energy"]))]
    rec["matching_metric_drift_max"] = float(max(drift)) if drift else 0.0
    rec["fast_path_drift"] = built.fast_path_drift
    return rec, rows


def run_audit_v2(index: pd.DataFrame, probe, cfg: V2Config, run_dir: Path,
                 shard: int = 0, num_shards: int = 1,
                 budget: TimeBudget | None = None) -> None:
    """Audit this worker's shard of targets, resuming from checkpoints."""
    targets = select_targets(index, cfg.base)
    if num_shards > 1:
        targets = targets[assign_shards(targets["image_id"].tolist(), num_shards) == shard]
    targets = targets.reset_index(drop=True)
    suffix = "" if num_shards <= 1 else f".shard{shard}"
    bank = DonorBank(index)
    sources = SourceBank(bank, cfg.max_sham_sources)
    cache = ScoreCache(run_dir / f"score_cache{suffix}.jsonl", probe_signature(cfg), "")
    rec_ckpt = RecordCheckpoint(run_dir / f"causal_records_v2{suffix}.jsonl")
    iv_ckpt = RecordCheckpoint(run_dir / f"interventions_v2{suffix}.jsonl", key="_row_key")
    images_dir = run_dir / "intervention_images" if cfg.base.save_images else None

    done: set[str] = set()
    for path in Path(run_dir).glob("causal_records_v2*.jsonl"):
        done |= RecordCheckpoint(path).done_ids()
    todo = targets[~targets["image_id"].astype(str).isin(done)].reset_index(drop=True)
    log.info("shard %d/%d: %d targets, %d already done, %d to go", shard, num_shards,
             len(targets), len(targets) - len(todo), len(todo))
    if not len(todo):
        return

    t_start = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=1) as pool:
        def submit(i):
            return pool.submit(build_target, todo.iloc[i], bank, sources, cfg, images_dir)
        future = submit(0)
        for n in range(len(todo)):
            row = todo.iloc[n]
            try:
                built = future.result()
            except Exception as exc:                          # noqa: BLE001
                log.exception("building %s failed", row["image_id"])
                built = BuiltTarget({"image_id": str(row["image_id"]),
                                     "category": str(row["category"]),
                                     "defect_type": str(row["defect_type"])}, "",
                                    error=f"failed: {type(exc).__name__}: {exc}")
            stop = budget is not None and not budget.has_room()
            future = submit(n + 1) if (n + 1 < len(todo) and not stop) else None
            if built.error:
                rec, rows = {**built.base, "status": built.error}, []
            else:
                try:
                    rec, rows = score_target(built, probe, cache, cfg.batch_size)
                except Exception as exc:                      # noqa: BLE001
                    log.exception("scoring %s failed", row["image_id"])
                    rec, rows = {**built.base, "status": f"failed: {type(exc).__name__}: {exc}"}, []
            rec_ckpt.append(rec)
            for i, r in enumerate(rows):
                iv_ckpt.append({**r, "_row_key": f"{r['image_id']}#{i}"})
            rec_ckpt.flush()
            iv_ckpt.flush()
            elapsed = time.monotonic() - t_start
            log.info("[%d/%d] %s | %s | build %.1fs | %d imgs (%d unique) | cache hits %d, "
                     "scored %d | %.1fs/target", n + 1, len(todo), row["image_id"],
                     rec.get("status"), rec.get("build_seconds", float("nan")),
                     rec.get("n_forward_images", 0), rec.get("n_unique_images", 0),
                     cache.hits, cache.misses, elapsed / (n + 1))
            if stop:
                log.warning("time budget reached after %d/%d targets", n + 1, len(todo))
                break


def probe_signature(cfg: V2Config) -> str:
    """Cache signature for direct-verdict scores: model + probe, not geometry."""
    from .scoring import scoring_signature
    return scoring_signature(cfg.base.model_id, cfg.base.score_mode,
                             PROBE_VERSION_HINT.get(cfg.base.score_mode, ""),
                             cfg.max_pixels, cfg.dtype)


#: Probe versions by mode, mirrored from QwenScorer so the signature needs no torch.
PROBE_VERSION_HINT = {"direct_verdict": "direct-verdict-v1",
                      "generated_verdict": "generated-verdict-v1"}
