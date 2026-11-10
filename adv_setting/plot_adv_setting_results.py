from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# Ensure this directory is importable (for adv_setting_common)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import adv_setting_common as C


# ===================== Global font sizes =====================
FONT_SIZE = 18
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.labelsize"] = FONT_SIZE
plt.rcParams["xtick.labelsize"] = FONT_SIZE
plt.rcParams["ytick.labelsize"] = FONT_SIZE
plt.rcParams["legend.fontsize"] = FONT_SIZE


# ===================== Figure sizes =====================
EPS_FIGSIZE = (14.5, 5.2)                 # reference (2 subplots)
EPS_SUBPLOT_W = EPS_FIGSIZE[0] / 2.0       # per-subplot width reference
EPS_SUBPLOT_H = EPS_FIGSIZE[1]


# ===================== Algorithm legend name mapping =====================
# NOTE: If your matplotlib mathtext does not support \text{...}, switch \text -> \mathrm.
ALG_DISPLAY_NAME: Dict[str, str] = {
    "BDOO": "BDOO",
    "PBDOO-cor1": "PBDOO",
    "PBDOO-strong": "PBDOO",
    "PTF-PBDOO-cor1": r"P$\text{B}^2$D-FTRL",
    "PTF-PBDOO-strong": r"P$\text{B}^2$D-FTAL",
    "PTF-PBDOO-fixed-block-cor1": r"P$\text{B}^2$D-FTRL (fixed block)",
    "PTF-PBDOO-fixed-block-strong": r"P$\text{B}^2$D-FTAL (fixed block)",
}


def _alg_display(raw_alg: str) -> str:
    raw_alg = str(raw_alg)
    return ALG_DISPLAY_NAME.get(raw_alg, raw_alg)


def _unique_in_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _summary_path(output_dir: Path, experiment: str, alpha_label: str) -> Path:
    return output_dir / f"{experiment}_{alpha_label}_summary.csv"


def _runs_path(output_dir: Path, experiment: str, alpha_label: str) -> Path:
    return output_dir / f"{experiment}_{alpha_label}_runs.csv"


def _load_summary_rows(
    output_dir: Path, experiment: str, alpha_label: str, recompute: bool,
    require_runs: int = 10,
) -> List[Dict[str, object]]:
    runs_path = _runs_path(output_dir, experiment, alpha_label)
    summary_path = _summary_path(output_dir, experiment, alpha_label)

    if require_runs > 0:
        if not runs_path.is_file():
            raise RuntimeError(f"Missing raw run file; refusing to plot: {runs_path}")
        raw_rows = C.read_csv(runs_path)
        validation_errors = C.validate_run_rows(raw_rows, expected_runs=require_runs)
        if validation_errors:
            preview = "\n".join(validation_errors[:20])
            raise RuntimeError(
                f"Refusing to plot invalid R1-9 run records in {runs_path}:\n{preview}"
            )

    if recompute and runs_path.exists():
        C.recompute_summary_only(runs_path, summary_path)

    rows = C.read_csv(summary_path)
    if not rows:
        raise RuntimeError(f"Missing or empty summary file; refusing to plot: {summary_path}")
    if require_runs > 0:
        incomplete = [
            (r.get("algorithm", ""), r.get("T", ""), r.get("epsilon", ""), r.get("topology", ""), int(float(r.get("num_runs", 0) or 0)))
            for r in rows
            if int(float(r.get("num_runs", 0) or 0)) != require_runs
        ]
        if incomplete:
            preview = "; ".join(
                f"alg={a}, T={t}, eps={e}, topology={topo}: runs={nr}"
                for a, t, e, topo, nr in incomplete[:8]
            )
            raise RuntimeError(
                f"Refusing to plot incomplete R1-9 results: every point must have exactly "
                f"{require_runs} independent runs. Found {len(incomplete)} incomplete/mismatched points. {preview}"
            )
    return rows  # type: ignore[return-value]


def _canon(field: str, v: object) -> str:
    return C.canonicalize_for_key(field, v)


def _print_regret_ratios(
    *,
    experiment: str,
    alpha_label: str,
    summary_rows: List[Dict[str, object]],
    num_learners: float,   # NEW: match plotting normalization
) -> None:
    """
    Print regret ratios using EXACTLY the same normalization as the figure:
      normalized_regret = avg_regret_mean / num_learners

    The metric helper returns sum_j Reg_j(T)/(nT). Dividing by n here takes
    the mean over learners while retaining the paper's nT normalization;
    this is intentional, not a second division of an already learner-averaged
    metric.
    """
    def _alg_norm(raw: str) -> str:
        mapping = {
            "BDOO-convex": "BDOO",
            "PBDOO-cor1": "PBDOO-cor1",
            "PBDOO-strong": "PBDOO-strong",
            "PTF-PBDOO-cor1": "PTF-PBDOO-cor1",
            "PTF-PBDOO-strong": "PTF-PBDOO-strong",
            "BDOO": "BDOO",
        }
        return mapping.get(raw, raw)

    key_fields = ("T", "epsilon", "num_nodes", "topology")
    key_noeps_fields = ("T", "num_nodes", "topology")

    def _key(r: Dict[str, object]) -> Tuple[str, str, str, str]:
        return tuple(_canon(f, r.get(f, "")) for f in key_fields)  # type: ignore[return-value]

    def _key_noeps(r: Dict[str, object]) -> Tuple[str, str, str]:
        return tuple(_canon(f, r.get(f, "")) for f in key_noeps_fields)  # type: ignore[return-value]

    by_alg: Dict[str, Dict[Tuple[str, str, str, str], Dict[str, object]]] = {}
    by_alg_noeps: Dict[str, Dict[Tuple[str, str, str], Dict[str, object]]] = {}
    for r in summary_rows:
        alg = _alg_norm(str(r.get("algorithm", "")))
        by_alg.setdefault(alg, {})[_key(r)] = r
        by_alg_noeps.setdefault(alg, {})[_key_noeps(r)] = r

    if alpha_label == "alpha0":
        ftrl = "PTF-PBDOO-cor1"
        ftal = "PTF-PBDOO-strong"
        pbdoo = "PBDOO-cor1"
    else:
        ftrl = "PTF-PBDOO-strong"
        ftal = "PTF-PBDOO-strong"
        pbdoo = "PBDOO-strong"
    bdoo = "BDOO"

    def _ratio(a: float, b: float) -> float:
        return a / b if b != 0 else float("inf")

    def _norm_regret(row: Dict[str, object]) -> float:
        return float(row.get("avg_regret_mean", 0.0)) / float(num_learners)

    def _emit(name_a: str, name_b: str, label: str) -> None:
        if name_a == "BDOO" and name_a not in by_alg_noeps:
            return
        if name_a != "BDOO" and name_a not in by_alg:
            return
        if name_b == "BDOO" and name_b not in by_alg_noeps:
            return
        if name_b != "BDOO" and name_b not in by_alg:
            return

        print(f"[{experiment}][{alpha_label}] {label} ratios (normalized by num_learners={num_learners}):")

        if name_a == "BDOO" and name_b != "BDOO":
            # For each BDOO key, compare against all eps of name_b with same T,n,topo.
            for (t, n, topo), row_a in by_alg_noeps[name_a].items():
                ra = _norm_regret(row_a)
                for (tb, epsb, nb, topob), row_b in by_alg[name_b].items():
                    if tb == t and nb == n and topob == topo:
                        rb = _norm_regret(row_b)
                        print(f"  T={t}, eps={epsb}, n={n}, topo={topo} -> {ra:.6g}/{rb:.6g} = {_ratio(ra, rb):.6g}")
            return

        items_a = by_alg_noeps[name_a].items() if name_a == "BDOO" else by_alg[name_a].items()
        for k, row_a in items_a:
            if name_a == "BDOO":
                t, n, topo = k
                row_b = by_alg_noeps[name_b].get((t, n, topo)) if name_b == "BDOO" else None
                eps = ""
            else:
                t, eps, n, topo = k
                row_b = by_alg_noeps[name_b].get((t, n, topo)) if name_b == "BDOO" else by_alg[name_b].get(k)
            if not row_b:
                continue

            ra = _norm_regret(row_a)
            rb = _norm_regret(row_b)
            print(f"  T={t}, eps={eps}, n={n}, topo={topo} -> {ra:.6g}/{rb:.6g} = {_ratio(ra, rb):.6g}")

    _emit(ftrl, bdoo, "P^2D-FTRL / BDOO")
    _emit(ftrl, pbdoo, "P^2D-FTRL / PBDOO")
    _emit(ftal, bdoo, "P^2D-FTAL / BDOO")
    _emit(ftal, pbdoo, "P^2D-FTAL / PBDOO")


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)


# ===================== Auto legend placement (no manual trial) =====================
def _legend_size_frac(fig: plt.Figure, leg: plt.Legend, renderer) -> Tuple[float, float]:
    bbox = leg.get_window_extent(renderer=renderer)
    return (bbox.width / fig.bbox.width, bbox.height / fig.bbox.height)


def _axes_span(fig_axes: List[plt.Axes]) -> Tuple[float, float, float]:
    poses = [ax.get_position() for ax in fig_axes]
    span_x0 = min(p.x0 for p in poses)
    span_x1 = max(p.x1 for p in poses)
    top_y = max(p.y1 for p in poses)
    return span_x0, span_x1, top_y


def _gap_center_for_two_axes(ax0: plt.Axes, ax1: plt.Axes) -> float:
    p0, p1 = ax0.get_position(), ax1.get_position()
    return 0.5 * (p0.x1 + p1.x0)


def _place_legends_above_axes_auto(
    fig: plt.Figure,
    axes: List[plt.Axes],
    legends: List[plt.Legend],
    *,
    prefer_gap_center: bool = False,
    pad_top: float = 0.01,
    pad_between: float = 0.01,
    side_gap: float = 0.01,
    stack_gap: float = 0.008,
    wspace: Optional[float] = None,
) -> None:
    if not legends:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    sizes = [_legend_size_frac(fig, lg, renderer) for lg in legends]
    widths = [w for (w, _h) in sizes]
    heights = [h for (_w, h) in sizes]

    span_x0, span_x1, _ = _axes_span(axes)
    span_w = span_x1 - span_x0

    if prefer_gap_center and len(axes) == 2:
        x_center = _gap_center_for_two_axes(axes[0], axes[1])
    else:
        x_center = 0.5 * (span_x0 + span_x1)

    layout = "single"
    required_h = heights[0]
    if len(legends) == 2:
        total_w = widths[0] + widths[1] + side_gap
        if total_w <= span_w + 1e-9:
            layout = "side_by_side"
            required_h = max(heights[0], heights[1])
        else:
            layout = "stacked"
            required_h = heights[0] + heights[1] + stack_gap

    axes_top_target = 1.0 - (required_h + pad_top + pad_between)
    axes_top_target = max(0.05, min(0.95, axes_top_target))

    if wspace is None:
        fig.subplots_adjust(top=axes_top_target)
    else:
        fig.subplots_adjust(top=axes_top_target, wspace=wspace)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    span_x0, span_x1, top_y = _axes_span(axes)
    span_w = span_x1 - span_x0

    if prefer_gap_center and len(axes) == 2:
        x_center = _gap_center_for_two_axes(axes[0], axes[1])
    else:
        x_center = 0.5 * (span_x0 + span_x1)

    sizes = [_legend_size_frac(fig, lg, renderer) for lg in legends]
    widths = [w for (w, _h) in sizes]
    heights = [h for (_w, h) in sizes]

    y0 = top_y + pad_between

    if len(legends) == 1:
        legends[0].set_bbox_to_anchor((x_center, y0), transform=fig.transFigure)
        return

    if layout == "side_by_side":
        total_w = widths[0] + widths[1] + side_gap
        left_x0 = x_center - 0.5 * total_w
        if left_x0 < span_x0:
            left_x0 = span_x0
        if left_x0 + total_w > span_x1:
            left_x0 = span_x1 - total_w

        x0 = left_x0 + 0.5 * widths[0]
        x1 = left_x0 + widths[0] + side_gap + 0.5 * widths[1]
        legends[0].set_bbox_to_anchor((x0, y0), transform=fig.transFigure)
        legends[1].set_bbox_to_anchor((x1, y0), transform=fig.transFigure)
        return

    legends[0].set_bbox_to_anchor((x_center, y0), transform=fig.transFigure)
    legends[1].set_bbox_to_anchor((x_center, y0 + heights[0] + stack_gap), transform=fig.transFigure)


# ===================== Panel captions under x-label (auto) =====================
def _add_panel_caption_below_xlabel(fig: plt.Figure, ax: plt.Axes, caption: str, pad_px: int = 6) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    axpos = ax.get_position()
    xcenter = 0.5 * (axpos.x0 + axpos.x1)

    xlabel = ax.xaxis.label
    bbox = xlabel.get_window_extent(renderer=renderer)
    y = (bbox.y0 - pad_px) / fig.bbox.height

    fig.text(xcenter, y, caption, ha="center", va="top", fontsize=FONT_SIZE)


# ===================== Markers / Colors =====================
def _make_marker_map(display_alg_names: List[str]) -> Dict[str, str]:
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "+", "x"]
    return {alg: marker_cycle[i % len(marker_cycle)] for i, alg in enumerate(display_alg_names)}


def _make_alg_rgb_color_map(display_alg_names: List[str]) -> Dict[str, str]:
    """
    Red/Yellow/Blue for the 3 algorithms (by DISPLAY name):
      - BDOO -> red
      - PBDOO -> yellow
      - P B^2 D-* -> blue
    """
    out: Dict[str, str] = {}
    for a in display_alg_names:
        if a == "BDOO":
            out[a] = "red"
        elif a == "PBDOO":
            out[a] = "gold"
        elif "(fixed block)" in a:
            out[a] = "black"
        else:
            out[a] = "blue"
    return out


# ===================== Core plot helper (fixed color per algorithm) =====================
def _plot_lines_on_ax_fixed_alg_color(
    ax: plt.Axes,
    *,
    summary_rows: List[Dict[str, object]],
    x_key: str,
    x_label: str,
    log_x: bool,
    marker_map: Dict[str, str],
    alg_color_map: Dict[str, str],
    y_label: str = "Normalized regret loss",
    num_learners=9
) -> None:
    if not summary_rows:
        return

    raw_alg_names = sorted({str(r.get("algorithm", "")) for r in summary_rows})

    for alg in raw_alg_names:
        disp_alg = _alg_display(alg)
        rows = [r for r in summary_rows if str(r.get("algorithm", "")) == alg]
        if not rows:
            continue

        xs = [float(r[x_key]) for r in rows]
        ys = [float(r["avg_regret_mean"]) / num_learners for r in rows]
        # The stored regret is sum_j Reg_j(T)/(nT); dividing both mean and
        # sample std by n gives the learner-averaged regret normalized by nT.
        # Error bars are +/- one sample standard deviation across independent runs.
        yerr = [float(r["avg_regret_std"]) / num_learners for r in rows]

        order_idx = np.argsort(xs)
        xs_sorted = [xs[i] for i in order_idx]
        ys_sorted = [ys[i] for i in order_idx]
        yerr_sorted = [yerr[i] for i in order_idx]

        ax.errorbar(
            xs_sorted,
            ys_sorted,
            yerr=yerr_sorted,
            marker=marker_map.get(disp_alg, "o"),
            linestyle="--" if "(fixed block)" in disp_alg else "-",
            color=alg_color_map.get(disp_alg, "black"),
            capsize=3,
        )

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(x_label, fontsize=FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)


# ===================== NEW: Epsilon (3 subplots per alpha, no epsilon legend) =====================
def plot_impact_epsilon_three_subplots_per_alpha(
    *,
    output_dir: Path,
    summary_rows: List[Dict[str, object]],
    filename: str,
) -> None:
    """
    For a fixed alpha:
      - three epsilons -> three subplots in one figure
      - SAME subplot: algorithms colored red/yellow/blue
      - NO epsilon legend (only algorithm legend, shared above)
    """
    if not summary_rows:
        return

    # epsilon values (exclude BDOO placeholder "")
    eps_vals_raw = sorted({_canon("epsilon", r.get("epsilon", "")) for r in summary_rows})
    eps_vals = [e for e in eps_vals_raw if e != ""]
    if not eps_vals:
        return

    # numeric-sort eps if possible, then take first 3
    def _eps_key(e: str) -> Tuple[int, float, str]:
        try:
            return (0, float(e), e)
        except Exception:
            return (1, 0.0, e)

    eps_vals = sorted(eps_vals, key=_eps_key)[:3]

    # shared algorithms/markers/colors
    raw_alg_names = sorted({str(r.get("algorithm", "")) for r in summary_rows})
    display_alg_names = _unique_in_order([_alg_display(a) for a in raw_alg_names])
    marker_map = _make_marker_map(display_alg_names)
    alg_color_map = _make_alg_rgb_color_map(display_alg_names)

    # figure size: 3 subplots, each same size as previous epsilon subplot
    fig_w = EPS_SUBPLOT_W * 3.0
    fig_h = EPS_SUBPLOT_H
    fig, axes = plt.subplots(1, len(eps_vals), figsize=(fig_w, fig_h), sharey=False)
    axes_list = [axes] if len(eps_vals) == 1 else list(axes)

    panel_letters = ["(a)", "(b)", "(c)"]

    for i, (eps, ax) in enumerate(zip(eps_vals, axes_list)):
        # include BDOO rows regardless of eps (BDOO uses epsilon="")
        rows_eps = [
            r for r in summary_rows
            if (_canon("epsilon", r.get("epsilon", "")) == eps)
            or (str(r.get("algorithm", "")) == "BDOO" and _canon("epsilon", r.get("epsilon", "")) == "")
        ]

        _plot_lines_on_ax_fixed_alg_color(
            ax,
            summary_rows=rows_eps,
            x_key="T",
            x_label="Learning time T",
            log_x=True,
            marker_map=marker_map,
            alg_color_map=alg_color_map,
            y_label="Normalized regret loss",
            num_learners=9
        )

        letter = panel_letters[i] if i < len(panel_letters) else f"({i+1})"
        _add_panel_caption_below_xlabel(fig, ax, rf"{letter} $\epsilon$={eps}")

    # shared algorithm legend above all subplots (RGB colors)
    alg_handles = [
        Line2D([0], [0], color=alg_color_map.get(a, "black"), marker=marker_map[a], linestyle="--" if "(fixed block)" in a else "-", label=a)
        for a in display_alg_names
    ]
    leg_alg = fig.legend(
        handles=alg_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(alg_handles), 4),
        frameon=True,
        fontsize=FONT_SIZE,
        borderaxespad=0.0,
    )

    _place_legends_above_axes_auto(
        fig,
        axes=axes_list,
        legends=[leg_alg],
        prefer_gap_center=False,
        wspace=0.25,
    )

    _save_fig(fig, output_dir / filename)
    plt.close(fig)


# ===================== Topology (kept): 3 subplots per alpha, captions under, alg RGB =====================
def _topology_pretty_and_key(topo: str) -> Tuple[int, str]:
    s = str(topo)
    low = s.lower()
    if "cycle" in low:
        return (0, "Cycle")
    if "grid" in low:
        return (1, "Grid")
    if "clique" in low or "complete" in low or "all" in low:
        return (2, "All-connected")
    return (3, s)


def plot_impact_topology_three_subplots_per_alpha(
    *,
    output_dir: Path,
    summary_rows: List[Dict[str, object]],
    filename: str,
) -> None:
    if not summary_rows:
        return

    topo_vals_raw = sorted({_canon("topology", r.get("topology", "")) for r in summary_rows})
    topo_vals_raw = [t for t in topo_vals_raw if t != ""]
    if not topo_vals_raw:
        return

    topo_vals = sorted(topo_vals_raw, key=lambda t: _topology_pretty_and_key(t)[0])
    topo_pretty = [_topology_pretty_and_key(t)[1] for t in topo_vals]

    fig_w = EPS_SUBPLOT_W * 3.0
    fig_h = EPS_SUBPLOT_H
    fig, axes = plt.subplots(1, len(topo_vals), figsize=(fig_w, fig_h), sharey=True)
    axes_list = [axes] if len(topo_vals) == 1 else list(axes)

    raw_alg_names = sorted({str(r.get("algorithm", "")) for r in summary_rows})
    display_alg_names = _unique_in_order([_alg_display(a) for a in raw_alg_names])
    marker_map = _make_marker_map(display_alg_names)
    alg_color_map = _make_alg_rgb_color_map(display_alg_names)

    panel_letters = ["(a)", "(b)", "(c)"]

    for j, (topo, topo_name) in enumerate(zip(topo_vals, topo_pretty)):
        rows_t = [r for r in summary_rows if _canon("topology", r.get("topology", "")) == topo]
        _plot_lines_on_ax_fixed_alg_color(
            axes_list[j],
            summary_rows=rows_t,
            x_key="T",
            x_label="Learning time T",
            log_x=True,
            marker_map=marker_map,
            alg_color_map=alg_color_map,
            y_label="Normalized regret loss",
            num_learners=36
        )
        letter = panel_letters[j] if j < len(panel_letters) else f"({j+1})"
        _add_panel_caption_below_xlabel(fig, axes_list[j], f"{letter} {topo_name}")

    # shared algorithm legend above
    alg_handles = [
        Line2D([0], [0], color=alg_color_map.get(a, "black"), marker=marker_map[a], linestyle="--" if "(fixed block)" in a else "-", label=a)
        for a in display_alg_names
    ]
    leg = fig.legend(
        handles=alg_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(alg_handles), 4),
        frameon=True,
        fontsize=FONT_SIZE,
        borderaxespad=0.0,
    )
    _place_legends_above_axes_auto(fig, axes=axes_list, legends=[leg], wspace=0.25)

    _save_fig(fig, output_dir / filename)
    plt.close(fig)


# ===================== main =====================
def main() -> None:
    ap = argparse.ArgumentParser(description="Plot adv_setting results from CSVs (can run independently).")
    ap.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. Default: env PDOM_OUTPUT_DIR or adv_setting/experiment_results",
    )
    ap.add_argument(
        "--experiments",
        type=str,
        default="impact_epsilon_and_T,impact_topology",
        help="Comma-separated: impact_epsilon_and_T, impact_topology",
    )
    ap.add_argument(
        "--alphas",
        type=str,
        default="alpha0,alpha1",
        help="Comma-separated alpha labels to plot: alpha0,alpha1",
    )
    ap.add_argument(
        "--recompute_summary",
        action="store_true",
        help="Recompute summary.csv from runs.csv before plotting (recommended).",
    )
    ap.add_argument(
        "--require_runs", type=int, default=10,
        help="Require exactly this many independent runs per plotted point; use 0 to disable the completeness check.",
    )
    args = ap.parse_args()

    output_root = (args.output_dir or os.environ.get("PDOM_OUTPUT_DIR", "")).strip()
    if output_root:
        output_dir = Path(output_root)
    else:
        output_dir = Path(__file__).resolve().parent / "experiment_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    alphas = [a.strip() for a in args.alphas.split(",") if a.strip()]
    alpha_set = set(alphas)

    # ---- epsilon: alpha0 -> 3 subplots, alpha1 -> 3 subplots ----
    if "impact_epsilon_and_T" in experiments:
        for a in ["alpha0", "alpha1"]:
            if a in alpha_set:
                rows = _load_summary_rows(output_dir, "impact_epsilon_and_T", a, recompute=args.recompute_summary, require_runs=args.require_runs)

                # IMPORTANT: print values using the SAME normalization as the plot (num_learners=9 here)
                _print_regret_ratios(
                    experiment="impact_epsilon_and_T",
                    alpha_label=a,
                    summary_rows=rows,
                    num_learners=9,
                )

                plot_impact_epsilon_three_subplots_per_alpha(
                    output_dir=output_dir,
                    summary_rows=rows,
                    filename=f"impact_epsilon_and_T_{a}_3subplots.png",
                )

    # ---- topology: alpha0 & alpha1 each -> 3 subplots + shared legend ----
    if "impact_topology" in experiments:
        for a in ["alpha0", "alpha1"]:
            if a in alpha_set:
                rows = _load_summary_rows(output_dir, "impact_topology", a, recompute=args.recompute_summary, require_runs=args.require_runs)

                # IMPORTANT: print values using the SAME normalization as the plot (num_learners=36 here)
                _print_regret_ratios(
                    experiment="impact_topology",
                    alpha_label=a,
                    summary_rows=rows,
                    num_learners=36,
                )

                plot_impact_topology_three_subplots_per_alpha(
                    output_dir=output_dir,
                    summary_rows=rows,
                    filename=f"impact_topology_{a}_3subplots.png",
                )


if __name__ == "__main__":
    main()
