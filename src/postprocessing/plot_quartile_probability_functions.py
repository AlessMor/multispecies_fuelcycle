import numpy as np
import pandas as pd
import os, re

# Headless by default
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from src.postprocessing.plot_utils_functions import (
    drop_near_constant,
    ensure_registry,
    get_param_label,
    quartile_bins,
    quartile_colors,
    resolve_outdir_and_stem,
    select_scalar_numeric,
)

# ---- MathText: always render $...$ instead of showing literally ----
try:
    mpl.rcParams["text.usetex"] = False
    if "text.parse_math" in mpl.rcParams:
        mpl.rcParams["text.parse_math"] = True  # Matplotlib ≥3.8
    mpl.rcParams["mathtext.fontset"] = "dejavusans"
except Exception:
    pass

# ---- robust “per Joule” detector (matches 1/J, J^-1, per J, etc.; case-insensitive) ----
PER_J_RE = re.compile(r'(?i)(?:^|[^a-z])(?:1\s*/\s*j|j\s*(?:\^|-)?\s*-?1|per\s*j)(?:$|[^a-z])')


def _norm_unit(u: str) -> str:
    """Normalize a unit string for matching (lower; map 'joule'->'j'; strip brackets/odd chars)."""
    return re.sub(r"[^a-z0-9/^\-\s]", "", str(u or "").lower().replace("joule", "j"))


def _clean_symbol_label(s: str | None) -> str:
    """
    Make registry-provided symbols safe for Matplotlib mathtext:
      - Strip optional r/u/ur prefix + surrounding quotes (r"...", '...').
      - Trim whitespace.
      - Leave $...$ intact so mathtext renders.
    """
    if not s:
        return ""
    s = str(s).strip()
    m = re.fullmatch(r'(?is)\s*(?:ur|ru|r|u)?\s*([\'"])(.*)\1\s*', s)
    return m.group(2) if m else s


def _strip_trailing_unit(label: str) -> str:
    """Remove a trailing ' [ ... ]' unit suffix from a label if present."""
    return re.sub(r"\s*\[[^\]]*\]\s*$", "", label or "")


def _escape_dollars(s: str | None) -> str:
    """Escape $ so units like $/kWh don't break mathtext parsing."""
    return "" if not s else s.replace("$", r"\$")

def quartile_probability_plot(
    *,
    df: pd.DataFrame,
    target: str,
    inputs=None,
    input_parameters=None,
    target_unit: str | None = None,
    output_dir=None,
    outputs_dir=None,
    file_type: str = "",
    plot_name_prefix: str | None = None,
    plot_name: str | None = None,
    registry=None,
    failed_counts_summary: dict | None = None,  # Lightweight summary of failed rows
    include_failed_in_count: bool = True,  # Include failed in denominator for normalization
    display_failed_in_plot: bool = False,  # Display grey FAILED line/bar
    PLOT_STYLE: str = "line",              # "line" | "bar"
    min_per_bin: int = 1,
    MAX_POINTS: int = 12,                  # target number of x points to display
    RESCALE: bool = True,                  # auto-rescale y-axis per subplot based on data spread
    show_titles: bool = True,
    **_,
):
    if df is None or len(df) == 0:
        print("   No data. Skipping quartile-probability plot.")
        return

    _inputs = input_parameters if inputs is None else inputs
    outdir, stem = resolve_outdir_and_stem(
        output_dir=output_dir,
        outputs_dir=outputs_dir,
        plot_name_prefix=plot_name_prefix,
        plot_name=plot_name,
        default_stem="quartile_probs",
        suffix="__quartile_probs" if plot_name_prefix else None,
    )

    PLOT_STYLE = (PLOT_STYLE or "line").lower()
    if PLOT_STYLE not in {"line", "bar"}:
        PLOT_STYLE = "line"

    registry = ensure_registry(registry)

    # ---------- target quartiles on successes ----------
    # df now contains only successful cases (no _is_failed=True)
    t = pd.to_numeric(df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    succ = t.notna()
    
    if not succ.any():
        print(f"   No successful finite '{target}'. Skipping.")
        return

    qbins, qlabels = quartile_bins(t[succ])
    qcodes = qbins.cat.codes.to_numpy()
    kq = len(qlabels)

    # ---------- select scalar, varying inputs ----------
    cand = [c for c in list(_inputs or []) if c != "sol_success" and c != "_is_failed"]
    scalar_inputs = select_scalar_numeric(df, cand)
    varying = drop_near_constant(df, scalar_inputs)
    if not varying:
        print("   No varying scalar inputs to plot. Skipping.")
        return

    # ---------- colors & legend order ----------
    quart_colors = quartile_colors(kq)  # Q1..Q4
    FAILED_COLOR, FAILED_ALPHA = "#808080", 0.3

    if PLOT_STYLE == "bar":
        quart_legend_proxies = [
            Patch(facecolor=quart_colors[i], edgecolor="none", label=qlabels[i])
            for i in range(kq)
        ]
        fail_legend_proxy = Patch(
            facecolor=FAILED_COLOR, edgecolor="none", alpha=FAILED_ALPHA, label="FAILED"
        )
    else:
        quart_legend_proxies = [
            Line2D([0], [0], color=quart_colors[i], marker="o", linewidth=1.5, label=qlabels[i])
            for i in range(kq)
        ]
        fail_legend_proxy = Line2D(
            [0], [0], color=FAILED_COLOR, alpha=FAILED_ALPHA, marker="o", linewidth=1.5, label="FAILED"
        )

    # track if FAILED ever actually appears
    any_failed_plotted = False
    
    # Get failed dataframe from summary if available
    df_failed = failed_counts_summary.get("_failed_df") if failed_counts_summary else None

    def _fmt(v: float) -> str:
        if not np.isfinite(v) or v == 0:
            return "0"
        a = abs(v)
        return f"{v:.2e}" if (a >= 1e6 or a < 1e-3) else f"{v:.3g}"

    # ---------- transforms (t_startup→days, 1/J→$/kWh; label cleaning) ----------
    FORCE_KWH_NAMES = {"price_of_electricity", "c_kwh", "ckwh", "c_kwhn"}

    def _x_transform(name: str, unit_str: str | None, label_str: str | None, x: np.ndarray) -> tuple[np.ndarray, str | None]:
        lname = (name or "").lower()
        utok  = _norm_unit(unit_str or "")
        ltok  = (label_str or "").lower()

        # t_startup -> days
        if lname in {"t_startup", "tstartup"} or "t_startup" in lname:
            return x / 86400.0, "days"

        # name/label hints for kWh
        if (lname in FORCE_KWH_NAMES) or ("kwh" in ltok):
            if PER_J_RE.search(utok):  # stored as $/J
                return x * 3.6e6, "$/kWh"
            return x, "$/kWh"

        # otherwise: infer from unit
        if "kwh" in utok:
            return (x * 3.6e6, "$/kWh") if PER_J_RE.search(utok) else (x, "$/kWh")
        if PER_J_RE.search(utok):
            return x * 3.6e6, "$/kWh"

        return x, None

    # ---------- layout ----------
    n = len(varying)
    ncols = min(3, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 4.0 * nrows + 1.5),
    )
    axes = np.atleast_1d(axes).ravel()

    csv_rows = []
    for i, param in enumerate(varying):
        ax = axes[i]

        # get & clean label and unit up front
        raw_label = get_param_label(param, registry=registry) or ""
        label0    = _clean_symbol_label(raw_label)  # keep $...$ for mathtext
        unit0     = getattr(registry, "get_unit",  lambda n, **k: None)(param) or ""

        # numeric vectors
        x_all  = pd.to_numeric(df[param], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64, copy=False)
        x_succ = pd.to_numeric(df.loc[succ, param], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64, copy=False)

        # primary transform
        x_all,  unit_override = _x_transform(param, unit0, label0, x_all)
        x_succ, _             = _x_transform(param, unit0, label0, x_succ)

        # FORCE $/kWh for electricity price (by name/label/unit/value scale)
        name_l  = (param or "").lower()
        label_l = (label0 or "").lower()
        unit_l  = _norm_unit(unit0)
        looks_kwh   = ("kwh" in name_l) or ("kwh" in label_l)
        looks_per_j = PER_J_RE.search(unit_l) is not None or ("[1/j]" in label_l.replace(" ", "")) or ("1/j" in unit_l)
        must_be_kwh = looks_kwh or looks_per_j or (name_l in FORCE_KWH_NAMES)
        if must_be_kwh:
            need_scale = looks_per_j or (np.nanmedian(np.abs(x_all)) < 1e-3 and np.nanmax(np.abs(x_all)) < 1.0)
            if need_scale:
                x_all  = x_all  * 3.6e6
                x_succ = x_succ * 3.6e6
            unit_override = "$/kWh"

        # compute after any scaling
        finite_all = np.isfinite(x_all)
        xa = x_all[finite_all]
        if xa.size == 0:
            ax.set_visible(False)
            continue

        # clean label to avoid trailing "[...]" duplication and keep mathtext
        p_label = _strip_trailing_unit(label0)
        p_unit  = _escape_dollars(unit_override or unit0)  # ESCAPE $ IN UNIT ONLY

        # ---------- choose representative x positions ----------
        uniq_vals = np.sort(np.unique(x_all[np.isfinite(x_all)]))
        if uniq_vals.size == 0:
            ax.set_visible(False)
            continue
        if uniq_vals.size <= MAX_POINTS:
            x_pos = uniq_vals
        else:
            targets = np.linspace(float(uniq_vals[0]), float(uniq_vals[-1]), MAX_POINTS)
            mapped = []
            for tval in targets:
                idx = np.searchsorted(uniq_vals, tval)
                candidates = []
                if idx < uniq_vals.size:
                    candidates.append(uniq_vals[idx])
                if idx > 0:
                    candidates.append(uniq_vals[idx - 1])
                mapped.append(min(candidates, key=lambda v: abs(v - tval)) if candidates else tval)
            x_pos = np.array(sorted(set(mapped)), dtype=float)

        # bins at midpoints between chosen values
        if x_pos.size == 1:
            edges = np.array([x_pos[0] - 1, x_pos[0] + 1], dtype=float)
        else:
            mids = 0.5 * (x_pos[:-1] + x_pos[1:])
            edges = np.concatenate(([-np.inf], mids, [np.inf]))

        # Bin successful data by quartile
        finite_succ = np.isfinite(x_succ)
        xs = x_succ[finite_succ]
        qs = qcodes[finite_succ].astype(np.int32, copy=False)
        idx_s = np.digitize(xs, edges, right=True) - 1
        idx_s[idx_s < 0] = 0
        idx_s[idx_s >= x_pos.size] = x_pos.size - 1

        counts_q = np.zeros((x_pos.size, kq), dtype=np.int64)
        for q in range(kq):
            sel = (qs == q)
            if sel.any():
                np.add.at(counts_q[:, q], idx_s[sel], 1)

        # Bin failed data separately
        if include_failed_in_count and df_failed is not None and param in df_failed.columns:
            x_failed = pd.to_numeric(df_failed[param], errors="coerce").values
            finite_failed = np.isfinite(x_failed)
            xf = x_failed[finite_failed]
            idx_fail = np.digitize(xf, edges, right=True) - 1
            idx_fail[idx_fail < 0] = 0
            idx_fail[idx_fail >= x_pos.size] = x_pos.size - 1
            fails = np.bincount(idx_fail, minlength=x_pos.size)
        else:
            fails = np.zeros(x_pos.size, dtype=np.int64)

        # Total counts per bin = successful + failed (if counting failed)
        totals = counts_q.sum(axis=1) + (fails if include_failed_in_count else 0)

        keep = totals >= max(1, min_per_bin)
        if not np.any(keep):
            keep = np.ones_like(totals, dtype=bool)

        x_pos   = x_pos[keep]
        totals  = totals[keep]
        counts_q = counts_q[keep, :]
        fails   = fails[keep] if include_failed_in_count else None

        # ---------- probabilities (per parameter) ----------
        denom = totals.astype(np.float64)
        denom[denom == 0] = np.nan
        probs_q = np.nan_to_num(counts_q / denom[:, None], nan=0.0)
        prob_failed = (
            np.nan_to_num(fails / denom, nan=0.0)
            if (include_failed_in_count and fails is not None)
            else None
        )
        # Only display failed if requested AND there are actual failures to show
        show_failed = display_failed_in_plot and (prob_failed is not None) and np.nanmax(prob_failed) > 0.0
        if show_failed:
            any_failed_plotted = True

        # Decide plotting positions: direct when few points, normalized otherwise
        direct_plot = x_pos.size <= MAX_POINTS
        if direct_plot:
            x_plot = x_pos
        else:
            x_plot = np.linspace(0.0, 1.0, x_pos.size)

        # ---------- draw (per parameter) ----------
        if PLOT_STYLE == "bar":
            xpos = x_plot
            bottom = np.zeros(x_pos.size, float)
            width = (x_plot[1] - x_plot[0]) * 0.8 if x_pos.size > 1 else 0.5
            for q, lab in enumerate(qlabels):
                ax.bar(
                    xpos,
                    probs_q[:, q],
                    bottom=bottom,
                    width=width,
                    align="center",
                    edgecolor="none",
                    color=quart_colors[q],
                    label=lab,
                )
                bottom += probs_q[:, q]
            if show_failed and prob_failed is not None:
                ax.bar(
                    xpos,
                    prob_failed,
                    bottom=bottom,
                    width=width,
                    align="center",
                    edgecolor="none",
                    color=FAILED_COLOR,
                    alpha=FAILED_ALPHA,
                    label="FAILED",
                )
            tick_pos = xpos
            tick_labels = [_fmt(v) for v in x_pos]
        else:
            for q, lab in enumerate(qlabels):
                ax.plot(
                    x_plot,
                    probs_q[:, q],
                    marker="o",
                    linestyle="-",
                    linewidth=1.5,
                    markersize=4,
                    color=quart_colors[q],
                    label=lab,
                )
            if show_failed and prob_failed is not None:
                ax.plot(
                    x_plot,
                    prob_failed,
                    marker="o",
                    linestyle="-",
                    linewidth=1.5,
                    markersize=4,
                    color=FAILED_COLOR,
                    alpha=FAILED_ALPHA,
                    label="FAILED",
                )
            tick_pos = x_plot
            tick_labels = [_fmt(v) for v in x_pos]

        # ticks: direct scale for few points, normalized otherwise
        if direct_plot:
            xmin, xmax = float(np.min(x_pos)), float(np.max(x_pos))
            if xmax == xmin:
                pad = (abs(xmin) + 1.0) * 1e-3
                xmin -= pad; xmax += pad
            else:
                pad = (xmax - xmin) * 0.05
                xmin -= pad; xmax += pad
            ax.set_xlim(xmin, xmax)
            ax.set_xticks(x_pos)

            # scientific formatting with exponent only if |order| >= 2
            xformatter = mticker.ScalarFormatter(useMathText=True)
            xformatter.set_scientific(True)
            xformatter.set_powerlimits((-2, 2))  # no sci for ~10^1 or ~10^-1
            ax.xaxis.set_major_formatter(xformatter)
        else:
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=12)

        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha("right")
            lbl.set_fontsize(12)

        if RESCALE:
            # Only include prob_failed in rescaling if it's being displayed
            vals = [probs_q]
            if show_failed and prob_failed is not None:
                vals.append(prob_failed)
            vals = np.concatenate([v.ravel() for v in vals if v is not None])
            if vals.size == 0 or not np.isfinite(vals).any():
                y_min, y_max = 0.0, 1.0
            else:
                min_prob = float(np.nanmin(vals))
                max_prob = float(np.nanmax(vals))
                spread = max_prob - min_prob
                pad = spread * 0.1
                if pad == 0:
                    pad = max_prob * 0.05 if max_prob > 0 else 0.05
                y_min = max(min_prob - pad, 0.0)
                y_max = max_prob + pad
                if y_max <= y_min:
                    y_min, y_max = 0.0, 1.0
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(0, 1.0)
        ax.set_ylabel("Probability", fontsize=11)

        # y-axis scientific formatting (if needed)
        yformatter = mticker.ScalarFormatter(useMathText=True)
        yformatter.set_scientific(True)
        yformatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(yformatter)

        # per-subplot label: tiny legend below the x-axis instead of title
        label_text = p_label if not p_unit else f"{p_label} [{p_unit}]"
        dummy = [Line2D([], [], color="none", marker="", linestyle="")]
        ax.legend(
            dummy,
            [label_text],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            frameon=False,
            handlelength=0,
            handletextpad=0.0,
            fontsize=11,
        )

        # CSV rows for this parameter
        for j in range(x_pos.size):
            row = {
                "parameter": param,
                "bin_index": int(j),
                "x_value_or_mean": float(x_pos[j]),
                "n_total_bin": int(totals[j]),
            }
            for q, lab in enumerate(qlabels):
                row[f"prob_{lab}"] = float(probs_q[j, q])
            if include_failed_in_count and prob_failed is not None:
                row["prob_FAILED"] = float(prob_failed[j])
            csv_rows.append(row)

    # hide unused axes
    for j in range(len(varying), len(axes)):
        axes[j].set_visible(False)

    # figure title: clean mathtext for the target symbol; escape dollars in unit only
    raw_tlabel = get_param_label(target, registry=registry)
    t_label = _strip_trailing_unit(_clean_symbol_label(raw_tlabel))
    t_unit  = _escape_dollars(target_unit or getattr(registry, "get_unit", lambda n, **k: None)(target) or "")
    title = f"P(quartile | parameter value) wrt {t_label if not t_unit else f'{t_label} [{t_unit}]'}"
    if file_type:
        title += f" • {file_type}"
    has_title = show_titles and bool(title.strip())
    if has_title:
        fig.suptitle(title, fontsize=14)

    # ---------- global legend (omit FAILED if never plotted) ----------
    legend_labels = list(qlabels)
    legend_proxies = list(quart_legend_proxies)
    if any_failed_plotted:
        legend_labels.append("FAILED")
        legend_proxies.append(fail_legend_proxy)

    legend_title = t_label if not t_unit else f"{t_label} [{t_unit}]"

    # layout: no top blank if no title; legend in bottom band with less extra white space
    top = 0.90 if has_title else 0.99   # almost no blank when no title
    bottom = 0.20                       # axes region bottom
    left = 0.08
    right = 0.98

    # legend centered in the band below axes; small gap to bottom
    legend_y = 0.06

    fig.legend(
        legend_proxies,
        legend_labels,
        loc="lower center",
        ncol=max(2, (len(legend_labels) + 1) // 2),
        bbox_to_anchor=(0.5, legend_y),
        title=legend_title,
        title_fontsize=13,
    )

    fig.tight_layout(rect=[left, bottom, right, top])
    fig.subplots_adjust(wspace=0.4, hspace=0.6, top=top, bottom=bottom, left=left, right=right)

    out_png = outdir / f"{stem}__{target}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"   Quartile-probability plot saved: {out_png.name}")

    if csv_rows:
        out_csv = outdir / f"{stem}__{target}.csv"
        pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
        print(f"   Quartile-probability CSV saved: {out_csv.name}")
