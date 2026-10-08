"""Presentation casebook: deterministic selection of real examples, and their figures.

Examples are chosen by **documented rules over measured values**, never by eye, so
the same run always yields the same casebook and the selection can be stated on a
slide. They illustrate the population results; they are never used to estimate them.

Every figure here reproduces MVTec images, so casebook output goes into the run
directory and the presentation bundle only - never into the tracked public summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..utils.logging import get_logger
from .donor import load_mask, load_rgb

log = get_logger("causal.casebook")

#: Slot -> (rule as stated in the manifest). Order is the order of the casebook.
SLOTS = {
    "best_bidirectional": "among targets whose necessity AND sufficiency artifact imbalance "
                          "are at or below the run median, maximise min(robust CEE_N z, "
                          "robust CEE_S z), preferring CEE_N>0 and CEE_S>0",
    "necessity_not_sufficiency": "CEE_N>0 and CEE_S<=0, maximise robust CEE_N z - robust CEE_S z "
                                 "(fallback: maximise CEE_N - CEE_S)",
    "failure_ungrounded": "minimise max(robust CEE_N z, robust CEE_S z): weakest evidence "
                          "in both directions",
    "missed_anomaly": "original direct verdict NO; maximise |robust CEE_N z|",
    "category_strong": "category with the highest median CEE_N; its target with the highest CEE_N",
    "category_weak": "category with the lowest median CEE_N; its target with the lowest CEE_N",
    "artifact_control": "maximise the paired reduction in artifact imbalance "
                        "(context-only minus artifact-matched)",
}

CATEGORY_COLORS = {"bottle": "#0072B2", "carpet": "#D55E00", "pill": "#009E73",
                   "screw": "#CC79A7"}
DEFECT_COLOR = "#D55E00"
PLACEBO_COLOR = "#56B4E9"


def category_color(category: str) -> str:
    """Fixed colour per category, so a category looks the same in every figure."""
    if category in CATEGORY_COLORS:
        return CATEGORY_COLORS[category]
    extra = ("#E69F00", "#000000", "#999999", "#F0E442")
    return extra[sum(map(ord, category)) % len(extra)]


def _pick(frame: pd.DataFrame, score: pd.Series, taken: set[str], largest: bool = True):
    order = score.sort_values(ascending=not largest, kind="stable")
    for idx in order.index:
        if not np.isfinite(order[idx]):
            continue
        image_id = str(frame.loc[idx, "image_id"])
        if image_id not in taken:
            return idx, float(order[idx])
    return None, float("nan")


def select_cases(records: pd.DataFrame, n_p1: int = 10) -> pd.DataFrame:
    """Deterministic casebook. Returns one row per slot plus padding rows for P1.

    Ties break on ``image_id`` order (records are sorted first), so the selection is
    a pure function of the records.
    """
    ok = records[records.get("status", "ok") == "ok"] if "status" in records else records
    ok = ok[np.isfinite(ok["cee_n_artifact"].astype(float))
            & np.isfinite(ok["cee_s"].astype(float))]
    ok = ok.sort_values("image_id").reset_index(drop=True)
    rows: list[dict] = []
    taken: set[str] = set()
    if not len(ok):
        return pd.DataFrame(columns=["slot", "image_id", "category", "rule", "criterion_value"])

    zn = ok["robust_cee_n_z"].astype(float).clip(-50, 50)
    zs = ok["robust_cee_s_z"].astype(float).clip(-50, 50)

    def add(slot: str, idx, value: float) -> None:
        if idx is None:
            log.warning("casebook slot %s: no eligible target", slot)
            return
        image_id = str(ok.loc[idx, "image_id"])
        taken.add(image_id)
        rows.append({"slot": slot, "image_id": image_id,
                     "category": str(ok.loc[idx, "category"]), "rule": SLOTS.get(slot, ""),
                     "criterion_value": value})

    balanced = ok[(ok["artifact_artifact_imbalance"] <= ok["artifact_artifact_imbalance"].median())
                  & (ok["sufficiency_artifact_imbalance"]
                     <= ok["sufficiency_artifact_imbalance"].median())]
    pool = balanced if len(balanced) else ok
    both = pool[(pool["cee_n_artifact"] > 0) & (pool["cee_s"] > 0)]
    pool = both if len(both) else pool
    add("best_bidirectional", *_pick(ok, np.minimum(zn, zs)[pool.index], taken))

    nn = ok[(ok["cee_n_artifact"] > 0) & (ok["cee_s"] <= 0)]
    if len(nn):
        add("necessity_not_sufficiency", *_pick(ok, (zn - zs)[nn.index], taken))
    else:
        add("necessity_not_sufficiency",
            *_pick(ok, ok["cee_n_artifact"] - ok["cee_s"], taken))

    add("failure_ungrounded", *_pick(ok, np.maximum(zn, zs), taken, largest=False))

    missed = ok[ok["original_direct_prediction"] == "NO"]
    if len(missed):
        add("missed_anomaly", *_pick(ok, zn.abs()[missed.index], taken))

    med = ok.groupby("category")["cee_n_artifact"].median().sort_values(kind="stable")
    if len(med) >= 2:
        strong, weak = med.index[-1], med.index[0]
        g = ok[ok["category"] == strong]
        add("category_strong", *_pick(ok, g["cee_n_artifact"], taken))
        g = ok[ok["category"] == weak]
        add("category_weak", *_pick(ok, g["cee_n_artifact"], taken, largest=False))

    add("artifact_control", *_pick(
        ok, ok["context_artifact_imbalance"] - ok["artifact_artifact_imbalance"], taken))

    # P1 padding: spread over the CEE_N distribution, deterministic quantiles.
    for q in (0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8):
        if len(rows) >= n_p1:
            break
        target = ok["cee_n_artifact"].quantile(q)
        slot = f"spread_q{int(q * 100):02d}"
        SLOTS.setdefault(slot, f"P1 padding: target whose CEE_N is closest to its "
                               f"{int(q * 100)}th percentile")
        add(slot, *_pick(ok, -(ok["cee_n_artifact"] - target).abs(), taken))
    return pd.DataFrame(rows[:max(n_p1, len(SLOTS))])


def p1_items(records: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """Structured-generation items for casebook targets: six roles each."""
    from .structured import P1_ROLES

    rows = []
    rec = records.set_index("image_id")
    for image_id in dict.fromkeys(cases["image_id"].astype(str)):
        if image_id not in rec.index:
            continue
        r = rec.loc[image_id]
        saved = json.loads(r["saved_images"]) if isinstance(r.get("saved_images"), str) else {}
        paths = {"original": str(r["path"]), "recipient": saved.get("recipient", ""),
                 "defect_removed": saved.get("defect_removed", ""),
                 "artifact_placebo_best": saved.get("artifact_placebo_best", ""),
                 "defect_inserted": saved.get("defect_inserted", ""),
                 "sham_best": saved.get("sham_best", "")}
        for role in P1_ROLES:
            if paths.get(role):
                rows.append({"item_key": f"{image_id}#{role}", "image_id": image_id,
                             "category": str(r["category"]), "role": role,
                             "image_path": paths[role]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def _style() -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "figure.dpi": 110, "savefig.dpi": 200})


def _translated(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    out[ys + int(dy), xs + int(dx)] = True
    return out


def _outline(ax, mask: np.ndarray, color: str, lw: float = 2.0, ls: str = "-") -> None:
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw,
                   linestyles=ls)


def _img_axis(ax, image, title: str, caption: str = "") -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12, pad=6)
    if caption:
        ax.set_xlabel(caption, fontsize=10.5, family="monospace", labelpad=6)


def _iv_row(iv: pd.DataFrame, image_id: str, **match) -> pd.Series | None:
    sub = iv[iv["image_id"] == image_id]
    for k, v in match.items():
        sub = sub[sub[k] == v]
    return sub.iloc[0] if len(sub) else None


def _load(path) -> np.ndarray | None:
    try:
        return load_rgb(path) if path and Path(str(path)).exists() else None
    except OSError:
        return None


def _cap(p: float, z: float, delta: float | None = None, dlabel: str = "Δ") -> str:
    s = f"P(YES)={p:.3f}  z={z:+.2f}"
    if delta is not None and np.isfinite(delta):
        s += f"\n{dlabel}={delta:+.2f}"
    return s


def _missing(ax, title: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, f"{title}\n(not available)", ha="center", va="center", fontsize=11)


def case_figure(rec: pd.Series, iv: pd.DataFrame, out: Path, title: str = "") -> Path | None:
    """Two-row figure: necessity (top) and sufficiency (bottom) for one target."""
    _style()
    image_id = str(rec["image_id"])
    original = _load(rec["path"])
    if original is None:
        return None
    mask = load_mask(rec["mask_path"], original.shape[:2])
    z0 = float(rec["audit_logit_original"])
    zj = float(rec["audit_logit_recipient"])
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 9.2))

    _img_axis(axes[0, 0], original, "[A] Original",
              _cap(rec["audit_p_yes_original"], z0))
    _img_axis(axes[0, 1], original, "[B] GT defect mask (oracle)")
    _outline(axes[0, 1], mask, DEFECT_COLOR, 2.5)

    r = _iv_row(iv, image_id, role="defect_removed")
    img = _load(r["image_output_path"]) if r is not None else None
    if img is not None:
        _img_axis(axes[0, 2], img, "[C] Defect removed",
                  _cap(r["audit_p_yes"], r["audit_logit"], z0 - r["audit_logit"], "Δ removal"))
    else:
        _missing(axes[0, 2], "[C] Defect removed")

    r = _iv_row(iv, image_id, saved_role="artifact_placebo_best")
    img = _load(r["image_output_path"]) if r is not None else None
    if img is not None:
        _img_axis(axes[0, 3], img, "[D] Matched placebo removal",
                  _cap(r["audit_p_yes"], r["audit_logit"], z0 - r["audit_logit"], "Δ removal"))
        _outline(axes[0, 3], _translated(mask, r["dy"], r["dx"]), PLACEBO_COLOR, 2.0, "--")
    else:
        _missing(axes[0, 3], "[D] Matched placebo")

    r = _iv_row(iv, image_id, role="recipient")
    img = _load(r["image_output_path"]) if r is not None else None
    if img is not None:
        _img_axis(axes[1, 0], img, "[E] Normal donor J",
                  _cap(r["audit_p_yes"], r["audit_logit"]))
    else:
        _missing(axes[1, 0], "[E] Normal donor")
    r = _iv_row(iv, image_id, role="defect_inserted")
    img = _load(r["image_output_path"]) if r is not None else None
    if img is not None:
        _img_axis(axes[1, 1], img, "[F] Defect transplanted into J",
                  _cap(r["audit_p_yes"], r["audit_logit"], r["audit_logit"] - zj, "Δ insertion"))
        _outline(axes[1, 1], mask, DEFECT_COLOR, 1.5, ":")
    else:
        _missing(axes[1, 1], "[F] Defect transplanted")
    r = _iv_row(iv, image_id, saved_role="sham_best")
    img = _load(r["image_output_path"]) if r is not None else None
    if img is not None:
        _img_axis(axes[1, 2], img, "[G] Matched normal sham insertion",
                  _cap(r["audit_p_yes"], r["audit_logit"], r["audit_logit"] - zj, "Δ insertion"))
        _outline(axes[1, 2], mask, PLACEBO_COLOR, 1.5, ":")
    else:
        _missing(axes[1, 2], "[G] Sham insertion")

    ax = axes[1, 3]
    ax.axis("off")
    lines = [
        f"{rec['category']} / {rec['defect_type']}",
        f"original direct verdict: {rec['original_direct_prediction']}",
        "",
        f"CEE_N (artifact-matched) {float(rec['cee_n_artifact']):+.3f}",
        f"  robust z               {float(rec['robust_cee_n_z']):+.2f}",
        f"CEE_N (context-only, v1) {float(rec['cee_n_context']):+.3f}",
        f"CEE_S (sham-matched)     {float(rec['cee_s']):+.3f}",
        f"  robust z               {float(rec['robust_cee_s_z']):+.2f}",
        f"quadrant: {rec['quadrant']}",
        "",
        "removal artifact ratio (defect/placebo)",
        f"  seam {float(rec['artifact_seam_ratio']):.2f}   edit {float(rec['artifact_edit_ratio']):.2f}",
        "insertion artifact ratio (defect/sham)",
        f"  seam {float(rec['sufficiency_seam_ratio']):.2f}   "
        f"edit {float(rec['sufficiency_edit_ratio']):.2f}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11.5,
            family="monospace", transform=ax.transAxes)
    fig.suptitle(title or image_id, fontsize=14, y=0.995)
    fig.text(0.5, 0.005, "z = log P(YES)/P(NO) from the direct-verdict probe.  "
             "Orange outline: GT defect. Blue dashed: translated placebo mask.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out


def mask_and_placebos_figure(rec: pd.Series, iv: pd.DataFrame, out: Path,
                             n_show: int = 5) -> Path | None:
    """The methodology slide: GT mask and translated matched placebo masks."""
    _style()
    original = _load(rec["path"])
    if original is None:
        return None
    mask = load_mask(rec["mask_path"], original.shape[:2])
    sub = iv[(iv["image_id"] == rec["image_id"]) & (iv["role"] == "placebo")]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.0))
    for ax, arm, label in ((axes[0], "context", "context-matched placebos (v1)"),
                           (axes[1], "artifact", "context + artifact matched placebos (v2)")):
        _img_axis(ax, original, label)
        _outline(ax, mask, DEFECT_COLOR, 3.0)
        arm_rows = sub[sub["arm"] == arm].sort_values("placebo_index").head(n_show)
        for _, r in arm_rows.iterrows():
            pm = _translated(mask, r["dy"], r["dx"])
            _outline(ax, pm, PLACEBO_COLOR, 2.0, "--")
            ys, xs = np.nonzero(pm)
            ax.text(xs.mean(), ys.mean(), str(int(r["placebo_index"]) + 1), color="white",
                    fontsize=12, ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.2", fc=PLACEBO_COLOR, ec="none"))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=DEFECT_COLOR, lw=3, label="true defect (GT mask)"),
               Line2D([], [], color=PLACEBO_COLOR, lw=2, ls="--",
                      label=f"translated placebo masks (first {n_show} of K)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11.5)
    fig.suptitle(f"Same shape, same area, different place - {rec['image_id']}", fontsize=13.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out


def artifact_matching_figure(rec: pd.Series, iv: pd.DataFrame, out: Path) -> Path | None:
    """Context-only placebo vs artifact-matched placebo for the same target."""
    _style()
    original = _load(rec["path"])
    if original is None:
        return None
    mask = load_mask(rec["mask_path"], original.shape[:2])
    image_id = str(rec["image_id"])
    z0 = float(rec["audit_logit_original"])
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 5.6))
    _img_axis(axes[0], original, "Original + GT mask", _cap(rec["audit_p_yes_original"], z0))
    _outline(axes[0], mask, DEFECT_COLOR, 2.5)
    for ax, title, match in ((axes[1], "Defect removed", {"role": "defect_removed"}),
                             (axes[2], "Context-only placebo (v1)",
                              {"saved_role": "context_placebo_0"}),
                             (axes[3], "Artifact-matched placebo (v2)",
                              {"saved_role": "artifact_placebo_best"})):
        r = _iv_row(iv, image_id, **match)
        img = _load(r["image_output_path"]) if r is not None else None
        if img is None:
            _missing(ax, title)
            continue
        cap = (f"edit={r['edit_energy']:.1f}  seam={r['seam_score']:.2f}\n"
               f"Δ removal={z0 - r['audit_logit']:+.2f}")
        _img_axis(ax, img, title, cap)
        if r["role"] == "placebo":
            _outline(ax, _translated(mask, r["dy"], r["dx"]), PLACEBO_COLOR, 2.0, "--")
    fig.suptitle(
        f"{image_id}: defect/placebo artifact ratio  "
        f"v1 seam {float(rec['context_seam_ratio']):.2f}, edit {float(rec['context_edit_ratio']):.2f}"
        f"   ->   v2 seam {float(rec['artifact_seam_ratio']):.2f}, "
        f"edit {float(rec['artifact_edit_ratio']):.2f}  (1.00 = balanced)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out


def contrast_figure(strong: pd.Series, weak: pd.Series, iv: pd.DataFrame, out: Path) -> Path | None:
    """Strong-category case vs weak-category case, necessity and sufficiency rows."""
    _style()
    fig, axes = plt.subplots(2, 5, figsize=(19.0, 9.0))
    for row_i, (rec, label) in enumerate(((strong, "strongest category"),
                                          (weak, "weakest category"))):
        original = _load(rec["path"])
        if original is None:
            for ax in axes[row_i]:
                ax.axis("off")
            continue
        mask = load_mask(rec["mask_path"], original.shape[:2])
        image_id = str(rec["image_id"])
        z0, zj = float(rec["audit_logit_original"]), float(rec["audit_logit_recipient"])
        _img_axis(axes[row_i, 0], original, f"{rec['category']} ({label})",
                  _cap(rec["audit_p_yes_original"], z0))
        _outline(axes[row_i, 0], mask, DEFECT_COLOR, 2.5)
        specs = ((1, "Defect removed", {"role": "defect_removed"}, "rem"),
                 (2, "Matched placebo", {"saved_role": "artifact_placebo_best"}, "rem"),
                 (3, "Defect -> normal J", {"role": "defect_inserted"}, "ins"),
                 (4, "Matched sham", {"saved_role": "sham_best"}, "ins"))
        for col, title, match, kind in specs:
            r = _iv_row(iv, image_id, **match)
            img = _load(r["image_output_path"]) if r is not None else None
            if img is None:
                _missing(axes[row_i, col], title)
                continue
            d = (z0 - r["audit_logit"]) if kind == "rem" else (r["audit_logit"] - zj)
            _img_axis(axes[row_i, col], img, title,
                      _cap(r["audit_p_yes"], r["audit_logit"], d,
                           "Δ removal" if kind == "rem" else "Δ insertion"))
        axes[row_i, 4].text(1.04, 0.5, f"CEE_N {float(rec['cee_n_artifact']):+.2f}\n"
                            f"CEE_S {float(rec['cee_s']):+.2f}", transform=axes[row_i, 4].transAxes,
                            fontsize=12.5, family="monospace", va="center")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out


def build_casebook(records: pd.DataFrame, iv: pd.DataFrame, cases: pd.DataFrame,
                   run_dir: Path) -> list[str]:
    """Every qualitative figure. A missing optional example never fails the run."""
    figs = Path(run_dir) / "figures"
    case_dir = Path(run_dir) / "casebook"
    rec = records.set_index("image_id", drop=False)
    made: list[str] = []

    def slot(name: str):
        hit = cases[cases["slot"] == name]
        if not len(hit) or hit.iloc[0]["image_id"] not in rec.index:
            return None
        return rec.loc[hit.iloc[0]["image_id"]]

    jobs = []
    for _, c in cases.iterrows():
        if c["image_id"] in rec.index:
            jobs.append((f"case_{c['slot']}", lambda c=c: case_figure(
                rec.loc[c["image_id"]], iv, case_dir / f"case_{c['slot']}.png",
                f"{c['slot'].replace('_', ' ')}: {c['image_id']}")))
    best = slot("best_bidirectional")
    if best is not None:
        jobs.append(("example_bidirectional_best", lambda: case_figure(
            best, iv, figs / "example_bidirectional_best.png",
            f"Best bidirectional case: {best['image_id']}")))
        jobs.append(("example_mask_and_placebos", lambda: mask_and_placebos_figure(
            best, iv, figs / "example_mask_and_placebos.png")))
    fail = slot("failure_ungrounded")
    if fail is not None:
        jobs.append(("example_failure_case", lambda: case_figure(
            fail, iv, figs / "example_failure_case.png",
            f"Failure / ungrounded case: {fail['image_id']}")))
    strong, weak = slot("category_strong"), slot("category_weak")
    if strong is not None and weak is not None:
        jobs.append(("example_category_contrast", lambda: contrast_figure(
            strong, weak, iv, figs / "example_category_contrast.png")))
    art = slot("artifact_control")
    if art is not None:
        jobs.append(("example_artifact_matching", lambda: artifact_matching_figure(
            art, iv, figs / "example_artifact_matching.png")))
    for name, fn in jobs:
        try:
            if fn() is not None:
                made.append(name)
        except Exception:                                  # noqa: BLE001
            log.exception("casebook figure %s failed; continuing", name)
    return made


def structured_examples_md(cases: pd.DataFrame, records: pd.DataFrame,
                           structured: pd.DataFrame, out: Path) -> Path:
    """Exact structured generations for casebook targets (original/removed/inserted)."""
    lines = ["# Structured generations for the casebook targets", "",
             "Exact output of the repository's structured scorer (Qwen2.5-VL-3B, greedy) "
             "for selected targets. **This is a qualitative illustration, not rationale "
             "mediation** - Stage II (teacher-forced mediation) is not implemented. It "
             "shows whether the stated observation changes when the visual evidence "
             "changes.", ""]
    st = structured.copy()
    if "structured_status" in st:
        st = st[st["structured_status"].isna()]
    rec = records.set_index("image_id")
    for _, c in cases.iterrows():
        image_id = str(c["image_id"])
        lines += [f"## {c['slot']}: `{image_id}`", "", f"Selection rule: {c['rule']}", ""]
        if image_id in rec.index:
            r = rec.loc[image_id]
            lines += [f"Direct probe: CEE_N(artifact) {float(r['cee_n_artifact']):+.3f}, "
                      f"CEE_S {float(r['cee_s']):+.3f}, original verdict "
                      f"{r['original_direct_prediction']}.", ""]
        for role, label in (("original", "Original"), ("defect_removed", "Defect removed"),
                            ("artifact_placebo_best", "Matched placebo removal"),
                            ("recipient", "Normal donor J"),
                            ("defect_inserted", "Defect transplanted into J"),
                            ("sham_best", "Matched normal sham insertion")):
            hit = st[(st["image_id"] == image_id) & (st["role"] == role)]
            if not len(hit):
                continue
            h = hit.iloc[-1]
            p = h.get("structured_audit_p_yes")
            lines += [f"### {label}", "",
                      f"- **ANSWER**: {_vote(h.get('structured_generated_vote'))}  "
                      f"(P(YES) at verdict token = {_num(p)})",
                      f"- **OBSERVATION**: {h.get('structured_observation', '')}",
                      f"- **DEFECT_TYPE**: {h.get('structured_predicted_defect', '')}",
                      f"- **DEFECT_LOCATION**: {h.get('structured_predicted_location', '')}",
                      "", "<details><summary>raw generation</summary>", "", "```",
                      str(h.get("structured_generation_text", "")).strip(), "```", "",
                      "</details>", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def _vote(v) -> str:
    try:
        return {1: "YES", 0: "NO"}.get(int(float(v)), "unparsed")
    except (TypeError, ValueError):
        return "unparsed"


def _num(v) -> str:
    try:
        f = float(v)
        return f"{f:.3f}" if np.isfinite(f) else "n/a"
    except (TypeError, ValueError):
        return "n/a"
