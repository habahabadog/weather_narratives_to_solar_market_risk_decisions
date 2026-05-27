from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_baselines import (
    build_hybrid_residual_scenarios,
    settlement_revenue,
    solve_cvar_bids,
)


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}


LABELS = {
    "Ridge-text stochastic LP (S7)": "S7 ridge-text LP",
    "Ridge-text stochastic LP (S7; 10 seeds)": "S7 ridge-text LP",
    "CAISO public solar forecast (S0)": "S0 CAISO forecast",
    "MLP-text deterministic anchor (S11)": "MLP rule-text anchor",
    "MLP-text residual quantile bid (S25)": "S25 residual-quantile",
    "Ridge rule+LLM stochastic LP (S22)": "S22 ridge rule+LLM LP",
    "Rule-text hybrid blend": "Rule-text hybrid",
    "Matched text-anchor LLM hybrid": "Matched rule-text-anchor LLM",
    "Pure LLM hybrid LP (w=1.00)": "Pure LLM hybrid LP",
    "LLM hybrid blend": "LLM hybrid",
}


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def prepare_tradeoff_data(bidding_table: pd.DataFrame) -> pd.DataFrame:
    out = bidding_table.copy()
    out["plot_label"] = out["paper_label"].map(LABELS).fillna(out["paper_label"])
    out["revenue_musd"] = out["total_revenue_mean"].astype(float) / 1_000_000.0
    out["cvar_loss_kusd"] = out["cvar_95_loss_mean"].astype(float) / 1_000.0
    out["highlight"] = out["paper_label"].astype(str).eq("LLM hybrid blend")
    return out


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, aliases: tuple[str, ...] = ()) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for figure_stem in (stem, *aliases):
        fig.savefig(output_dir / f"{figure_stem}.pdf", bbox_inches="tight")
        fig.savefig(output_dir / f"{figure_stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def case_study_stem(case_date: str) -> str:
    return f"fig_extreme_weather_case_study_{case_date.replace('-', '_')}"


def add_pipeline_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        edgecolor="0.35",
        facecolor=color,
        alpha=0.16,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=7.2,
        linespacing=1.18,
    )


def plot_pipeline_schematic(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        ((0.025, 0.58), 0.205, 0.27, "Public inputs\nCAISO solar\nSP15 prices\n5-point HRRR\n5-office AFD", OKABE_ITO["sky_blue"]),
        ((0.275, 0.58), 0.205, 0.27, "Text extraction\nkeyword risks\nbounded LLM JSON\none-day shift", OKABE_ITO["orange"]),
        ((0.525, 0.58), 0.195, 0.27, "Forecast models\nPV generation\nreal-time price\nresidual pools", OKABE_ITO["bluish_green"]),
        ((0.770, 0.58), 0.205, 0.27, "Proxy decision\n20 scenarios\nCVaR LP\nanchor blend", OKABE_ITO["vermillion"]),
        ((0.275, 0.16), 0.205, 0.22, "Audit controls\npublic sources\nas-of alignment\nquality checks", OKABE_ITO["reddish_purple"]),
        ((0.525, 0.16), 0.195, 0.22, "2024 test\nforecast errors\npaired blocks\nweather slices", OKABE_ITO["blue"]),
        ((0.770, 0.16), 0.205, 0.22, "Outputs\nproxy value\nCVaR95 loss\nimbalance", OKABE_ITO["yellow"]),
    ]
    for xy, width, height, text, color in boxes:
        add_pipeline_box(ax, xy, width, height, text, color)

    arrow_kw = {"arrowstyle": "->", "lw": 1.2, "color": "0.28", "shrinkA": 2, "shrinkB": 2}
    for start, end in [
        ((0.230, 0.715), (0.275, 0.715)),
        ((0.480, 0.715), (0.525, 0.715)),
        ((0.720, 0.715), (0.770, 0.715)),
        ((0.623, 0.58), (0.623, 0.38)),
        ((0.720, 0.27), (0.770, 0.27)),
        ((0.378, 0.38), (0.378, 0.58)),
    ]:
        ax.annotate("", xy=end, xytext=start, arrowprops=arrow_kw)

    ax.text(0.025, 0.935, "Controlled public-data backtest", fontsize=10.5, fontweight="bold", ha="left")
    ax.text(
        0.025,
        0.055,
        "Scope: system-level CAISO solar and SP15 price proxy; public-data settlement backtest, not a deployable market-trading system.",
        fontsize=7.0,
        ha="left",
        color="0.28",
    )
    save_figure(fig, output_dir, "fig_pipeline_schematic")


def reconstruct_hybrid_blend_bid_series(
    preds: pd.DataFrame,
    train_residuals: pd.DataFrame,
    rated_capacity: float,
    seed: int = 71_000,
    pv_model_name: str = "mlp_llm",
    rt_model_name: str = "transformer_llm",
    anchor_model_name: str = "mlp_text",
    lp_weight: float = 0.5,
    residual_scale: float = 1.0,
    cvar_gamma: float = 0.25,
    deviation_penalty: float = 50.0,
    scenario_count: int = 20,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    lp_bid_series = pd.Series(index=preds.index, dtype=float)
    for _, day in preds.groupby("local_date", sort=True):
        ordered = day.sort_values("timestamp_utc")
        pv_scenarios, rt_scenarios = build_hybrid_residual_scenarios(
            ordered,
            train_residuals,
            pv_model_name=pv_model_name,
            rt_model_name=rt_model_name,
            rated_capacity=rated_capacity,
            scenario_count=scenario_count,
            rng=rng,
            residual_scale=residual_scale,
        )
        try:
            q_day = solve_cvar_bids(
                pv_scenarios=pv_scenarios,
                rt_scenarios=rt_scenarios,
                da_lmp=ordered["da_lmp"].to_numpy(dtype=float),
                rated_capacity=rated_capacity,
                gamma=cvar_gamma,
                shortage_penalty=deviation_penalty,
                surplus_penalty=deviation_penalty,
            )
        except RuntimeError:
            q_day = ordered[f"pv_{pv_model_name}"].to_numpy(dtype=float)
        lp_bid_series.loc[ordered.index] = q_day

    anchor_bid = np.clip(
        np.nan_to_num(preds[f"pv_{anchor_model_name}"].to_numpy(dtype=float), nan=0.0),
        0.0,
        rated_capacity,
    )
    lp_bid = np.clip(np.nan_to_num(lp_bid_series.to_numpy(dtype=float), nan=0.0), 0.0, rated_capacity)
    blended = np.clip(float(lp_weight) * lp_bid + (1.0 - float(lp_weight)) * anchor_bid, 0.0, rated_capacity)
    return pd.Series(blended, index=preds.index, name="llm_hybrid_bid_mw")


def prepare_case_study_data(
    day: pd.DataFrame,
    hybrid_bid: list[float] | np.ndarray | pd.Series,
    deviation_penalty: float = 50.0,
    anchor_col: str = "pv_mlp_text",
) -> pd.DataFrame:
    out = day.sort_values("timestamp_utc").copy()
    out["anchor_bid_mw"] = np.clip(np.nan_to_num(out[anchor_col].to_numpy(dtype=float), nan=0.0), 0.0, np.inf)
    out["llm_hybrid_bid_mw"] = np.clip(np.asarray(hybrid_bid, dtype=float), 0.0, np.inf)
    actual = out["pv_mw"].to_numpy(dtype=float)
    da = out["da_lmp"].to_numpy(dtype=float)
    rt = out["rt_lmp"].to_numpy(dtype=float)
    anchor_bid = out["anchor_bid_mw"].to_numpy(dtype=float)
    llm_bid = out["llm_hybrid_bid_mw"].to_numpy(dtype=float)

    out["anchor_signed_imbalance_mw"] = anchor_bid - actual
    out["llm_signed_imbalance_mw"] = llm_bid - actual
    out["anchor_abs_imbalance_mw"] = np.abs(out["anchor_signed_imbalance_mw"])
    out["llm_abs_imbalance_mw"] = np.abs(out["llm_signed_imbalance_mw"])
    out["anchor_revenue_usd"] = settlement_revenue(
        actual,
        anchor_bid,
        da,
        rt,
        shortage_penalty=deviation_penalty,
        surplus_penalty=deviation_penalty,
    )
    out["llm_hybrid_revenue_usd"] = settlement_revenue(
        actual,
        llm_bid,
        da,
        rt,
        shortage_penalty=deviation_penalty,
        surplus_penalty=deviation_penalty,
    )
    out["revenue_delta_usd"] = out["llm_hybrid_revenue_usd"] - out["anchor_revenue_usd"]
    out["cumulative_revenue_delta_usd"] = out["revenue_delta_usd"].cumsum()
    return out


def plot_revenue_cvar_tradeoff(bidding_table: pd.DataFrame, output_dir: Path) -> None:
    data = prepare_tradeoff_data(bidding_table)
    hidden_labels = {
        "S7 ridge-text LP",
        "S0 CAISO forecast",
    }
    data = data.loc[~data["plot_label"].isin(hidden_labels)].copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.35))
    style_by_label = {
        "MLP rule-text anchor": ("s", OKABE_ITO["sky_blue"]),
        "S25 residual-quantile": ("P", OKABE_ITO["yellow"]),
        "S22 ridge rule+LLM LP": ("^", OKABE_ITO["reddish_purple"]),
        "Rule-text hybrid": ("D", OKABE_ITO["bluish_green"]),
        "Matched rule-text-anchor LLM": ("d", OKABE_ITO["orange"]),
        "Pure LLM hybrid LP": ("X", OKABE_ITO["orange"]),
        "LLM hybrid": ("*", OKABE_ITO["vermillion"]),
    }

    for _, row in data.iterrows():
        label = str(row["plot_label"])
        marker, color = style_by_label.get(label, ("o", OKABE_ITO["blue"]))
        is_highlight = bool(row["highlight"])
        ax.scatter(
            row["cvar_loss_kusd"],
            row["revenue_musd"],
            s=170 if is_highlight else 70,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=4 if is_highlight else 3,
            label=label,
        )

    rule = data[data["paper_label"].eq("Rule-text hybrid blend")]
    matched = data[data["paper_label"].eq("Matched text-anchor LLM hybrid")]
    llm = data[data["paper_label"].eq("LLM hybrid blend")]
    if not rule.empty and not llm.empty:
        target_row = matched.iloc[0] if not matched.empty else llm.iloc[0]
        llm_row = llm.iloc[0]
        ax.annotate(
            "LLM hybrid\n(w=0.50)",
            xy=(target_row["cvar_loss_kusd"], target_row["revenue_musd"]),
            xytext=(548.0, 379.5),
            arrowprops={
                "arrowstyle": "->",
                "color": OKABE_ITO["vermillion"],
                "lw": 1.1,
                "connectionstyle": "arc3,rad=-0.10",
            },
            color=OKABE_ITO["vermillion"],
            fontsize=8,
            ha="left",
            va="center",
            linespacing=1.05,
        )
        ax.annotate(
            "Rule-text hybrid",
            xy=(rule.iloc[0]["cvar_loss_kusd"], rule.iloc[0]["revenue_musd"]),
            xytext=(538.0, 369.0),
            textcoords="data",
            arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 0.8, "shrinkA": 2, "shrinkB": 3},
            fontsize=8,
            ha="left",
            va="top",
        )

    ax.set_title("2024 proxy value and tail-risk trade-off", pad=8)
    ax.set_xlabel("CVaR95 loss (thousand USD per hour)")
    ax.set_ylabel("Total proxy value (million USD)")
    ax.set_xlim(data["cvar_loss_kusd"].min() - 12, data["cvar_loss_kusd"].max() + 12)
    ax.set_ylim(data["revenue_musd"].min() - 5, data["revenue_musd"].max() + 7)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        columnspacing=1.8,
        handletextpad=0.7,
    )
    fig.subplots_adjust(bottom=0.32, top=0.92, left=0.12, right=0.98)
    save_figure(
        fig,
        output_dir,
        "fig_revenue_cvar_tradeoff",
        aliases=("fig_settlement_proxy_cvar_tradeoff",),
    )


def _read_seed_result(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"seed", "total_revenue", "cvar_95_loss", "imbalance_mwh_proxy"}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def build_scenario_replacement_data(results_dir: Path) -> pd.DataFrame:
    current_rule = results_dir / "bidding_hybrid_blended_mlp_text_transformer_text_10seed_penalty50.csv"
    current_llm = results_dir / "bidding_hybrid_blended_mlp_llm_transformer_llm_anchor_mlp_text_10seed_penalty50.csv"
    if current_rule.exists() and current_llm.exists():
        specs = [
            (
                "MLP rule-text anchor",
                current_rule.name,
                current_llm.name,
            )
        ]
    else:
        specs = [
            (
                "MLP rule-text anchor",
                "bidding_fixedw_mlp_text_transformer_text_anchor_mlp_text.csv",
                "bidding_fixedw_mlp_llm_rule_transformer_llm_anchor_mlp_text.csv",
            ),
            (
                "MLP LLM-rule anchor",
                "bidding_fixedw_mlp_text_transformer_text_anchor_mlp_llm_rule.csv",
                "bidding_fixedw_mlp_llm_rule_transformer_llm_anchor_mlp_llm_rule.csv",
            ),
        ]
    rows: list[dict[str, float | int | str]] = []
    for anchor, rule_name, llm_name in specs:
        rule = _read_seed_result(results_dir / rule_name)
        llm = _read_seed_result(results_dir / llm_name)
        if rule.empty or llm.empty:
            continue
        if "lp_weight" in rule.columns:
            rule = rule[np.isclose(pd.to_numeric(rule["lp_weight"], errors="coerce"), 0.5)].copy()
        if "lp_weight" in llm.columns:
            llm = llm[np.isclose(pd.to_numeric(llm["lp_weight"], errors="coerce"), 0.5)].copy()
        merged = rule.merge(llm, on="seed", suffixes=("_rule", "_llm"))
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "anchor": anchor,
                    "seed": int(row.seed),
                    "value_delta_musd": (float(row.total_revenue_llm) - float(row.total_revenue_rule)) / 1_000_000.0,
                    "cvar95_loss_reduction_kusd_h": (float(row.cvar_95_loss_rule) - float(row.cvar_95_loss_llm)) / 1_000.0,
                    "imbalance_reduction_gwh": (
                        float(row.imbalance_mwh_proxy_rule) - float(row.imbalance_mwh_proxy_llm)
                    )
                    / 1_000.0,
                }
            )
    return pd.DataFrame(rows)


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, mean, mean
    half_width = float(stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr)))
    return mean, mean - half_width, mean + half_width


def plot_scenario_replacement_effects(data: pd.DataFrame, output_dir: Path) -> None:
    if data.empty:
        return
    metrics = [
        ("value_delta_musd", "Value gain\n(M USD)", OKABE_ITO["orange"]),
        ("cvar95_loss_reduction_kusd_h", "CVaR95 loss reduction\n(k USD/h)", OKABE_ITO["bluish_green"]),
        ("imbalance_reduction_gwh", "Imbalance reduction\n(GWh)", OKABE_ITO["blue"]),
    ]
    anchors = [anchor for anchor in ["MLP rule-text anchor", "MLP LLM-rule anchor"] if anchor in set(data["anchor"].astype(str))]
    fig_height = 2.55 if len(anchors) == 1 else 3.05
    fig, axes = plt.subplots(1, 3, figsize=(7.25, fig_height), sharey=True)
    rng = np.random.default_rng(4)

    for ax, (metric, xlabel, color) in zip(axes, metrics):
        for y_pos, anchor in enumerate(anchors):
            subset = data[data["anchor"].astype(str).eq(anchor)].copy()
            if subset.empty:
                continue
            x = pd.to_numeric(subset[metric], errors="coerce").to_numpy(dtype=float)
            jitter = rng.normal(0.0, 0.035, len(x))
            ax.scatter(x, y_pos + jitter, s=18, color=color, alpha=0.60, edgecolor="none")
            mean, low, high = _mean_ci(subset[metric])
            ax.errorbar(
                mean,
                y_pos,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                color="black",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=5,
                elinewidth=1.1,
                capsize=3,
                zorder=4,
            )
        ax.axvline(0, color="0.35", lw=0.8, ls=":")
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.5, len(anchors) - 0.5)
        ax.invert_yaxis()
    axes[0].set_yticks(range(len(anchors)))
    axes[0].set_yticklabels(anchors)
    fig.suptitle("Seed-paired effect of replacing rule-text scenarios with LLM scenarios", y=1.04)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_scenario_replacement_effects")


def build_paired_day_evidence_data(ci: pd.DataFrame) -> pd.DataFrame:
    by_metric = {str(row["metric"]): row for _, row in ci.iterrows()}
    rows = []
    if "annual_proxy_revenue_delta_usd" in by_metric:
        row = by_metric["annual_proxy_revenue_delta_usd"]
        rows.append(
            {
                "metric": "Annual value gain",
                "value": float(row["estimate"]) / 1_000_000.0,
                "low": float(row["ci_low"]) / 1_000_000.0,
                "high": float(row["ci_high"]) / 1_000_000.0,
                "unit": "M USD",
                "color": OKABE_ITO["orange"],
            }
        )
    if "mean_daily_cvar95_loss_reduction_usd" in by_metric:
        row = by_metric["mean_daily_cvar95_loss_reduction_usd"]
        rows.append(
            {
                "metric": "Daily CVaR95 loss reduction",
                "value": float(row["estimate"]) / 1_000.0,
                "low": float(row["ci_low"]) / 1_000.0,
                "high": float(row["ci_high"]) / 1_000.0,
                "unit": "k USD",
                "color": OKABE_ITO["bluish_green"],
            }
        )
    if "mean_daily_imbalance_delta_mwh" in by_metric:
        row = by_metric["mean_daily_imbalance_delta_mwh"]
        rows.append(
            {
                "metric": "Daily imbalance reduction",
                "value": -float(row["estimate"]),
                "low": -float(row["ci_high"]),
                "high": -float(row["ci_low"]),
                "unit": "MWh",
                "color": OKABE_ITO["blue"],
            }
        )
    return pd.DataFrame(rows)


def plot_paired_day_evidence(data: pd.DataFrame, output_dir: Path) -> None:
    if data.empty:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(7.25, 2.65))
    if len(data) == 1:
        axes = [axes]
    for ax, row in zip(axes, data.itertuples(index=False)):
        value = float(row.value)
        low = float(row.low)
        high = float(row.high)
        ax.errorbar(
            value,
            0,
            xerr=[[value - low], [high - value]],
            fmt="o",
            color="black",
            markerfacecolor=str(row.color),
            markeredgecolor="black",
            markersize=6,
            elinewidth=1.3,
            capsize=4,
        )
        ax.axvline(0, color="0.35", lw=0.8, ls=":")
        ax.set_yticks([])
        ax.set_xlabel(str(row.unit))
        ax.set_title(str(row.metric), fontsize=8.5)
    fig.suptitle("Paired day-level uncertainty for the LLM hybrid", y=1.05)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_paired_day_evidence")


def plot_lp_weight_sensitivity(lp_table: pd.DataFrame, output_dir: Path) -> None:
    data = lp_table.sort_values("lp_weight").copy()
    x = data["lp_weight"].astype(float).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), sharex=True)

    axes[0].plot(
        x,
        data["total_revenue_mean_vs_S7_pct"].astype(float),
        marker="o",
        lw=1.7,
        color=OKABE_ITO["blue"],
        label="Mean revenue",
    )
    axes[0].plot(
        x,
        data["total_revenue_min_vs_S7_pct"].astype(float),
        marker="s",
        lw=1.5,
        color=OKABE_ITO["sky_blue"],
        label="Worst-seed revenue",
    )
    axes[0].axhline(0, color="0.45", lw=0.7, ls=":", alpha=0.75, zorder=0)
    axes[0].axvline(0.5, color=OKABE_ITO["vermillion"], lw=1.0, ls="--")
    axes[0].set_title("Revenue improvement")
    axes[0].set_xlabel("LP weight")
    axes[0].set_ylabel("Improvement vs 10-seed S7 (%)")
    axes[0].legend(frameon=True, facecolor="white", edgecolor="white", framealpha=0.9, loc="lower left")

    axes[1].plot(
        x,
        data["cvar_95_loss_mean_vs_S7_pct"].astype(float),
        marker="o",
        lw=1.7,
        color=OKABE_ITO["bluish_green"],
        label="Mean CVaR95 loss",
    )
    axes[1].plot(
        x,
        data["cvar_95_loss_max_vs_S7_pct"].astype(float),
        marker="s",
        lw=1.5,
        color=OKABE_ITO["orange"],
        label="Worst-seed CVaR95 loss",
    )
    axes[1].axhline(0, color="0.45", lw=0.7, ls=":", alpha=0.75, zorder=0)
    axes[1].axvline(0.5, color=OKABE_ITO["vermillion"], lw=1.0, ls="--", label="Validation-selected")
    axes[1].set_title("Tail-risk reduction")
    axes[1].set_xlabel("LP weight")
    axes[1].set_ylabel("Loss reduction vs 10-seed S7 (%)")
    axes[1].legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:.2f}" for value in x])

    fig.suptitle("LLM hybrid blend sensitivity to the LP-anchor mixing weight", y=1.03)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_lp_weight_sensitivity")


def plot_same_penalty_sensitivity(penalty_table: pd.DataFrame, output_dir: Path) -> None:
    data = penalty_table.sort_values("deviation_penalty").copy()
    x = data["deviation_penalty"].astype(float).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), sharex=True)

    axes[0].plot(
        x,
        data["total_revenue_mean_vs_ridge_same_penalty_pct"].astype(float),
        marker="o",
        lw=1.7,
        color=OKABE_ITO["blue"],
        label="Mean revenue",
    )
    axes[0].plot(
        x,
        data["total_revenue_min_vs_ridge_same_penalty_pct"].astype(float),
        marker="s",
        lw=1.5,
        color=OKABE_ITO["sky_blue"],
        label="Worst-seed revenue",
    )
    axes[0].axhline(0, color="0.45", lw=0.7, ls=":", alpha=0.75, zorder=0)
    axes[0].set_title("Revenue robustness")
    axes[0].set_xlabel("Deviation penalty (USD/MWh)")
    axes[0].set_ylabel("Improvement vs same-penalty ridge LP (%)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_ylim(bottom=0)

    axes[1].plot(
        x,
        data["cvar_95_loss_mean_vs_ridge_same_penalty_pct"].astype(float),
        marker="o",
        lw=1.7,
        color=OKABE_ITO["bluish_green"],
        label="Mean CVaR95 loss",
    )
    axes[1].plot(
        x,
        data["cvar_95_loss_max_vs_ridge_same_penalty_pct"].astype(float),
        marker="s",
        lw=1.5,
        color=OKABE_ITO["orange"],
        label="Worst-seed CVaR95 loss",
    )
    axes[1].axhline(0, color="0.45", lw=0.7, ls=":", alpha=0.75, zorder=0)
    axes[1].set_title("Tail-risk robustness")
    axes[1].set_xlabel("Deviation penalty (USD/MWh)")
    axes[1].set_ylabel("Loss reduction vs same-penalty ridge LP (%)")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].set_ylim(bottom=0)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(value)}" for value in x])

    fig.suptitle("LLM hybrid blend robustness across deviation-penalty settings", y=1.03)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_same_penalty_sensitivity")


def plot_extreme_weather_case_study(case_data: pd.DataFrame, output_dir: Path, case_date: str) -> None:
    data = case_data.sort_values("timestamp_utc").copy()
    hours = data["hour"].astype(int).to_numpy()
    event_text = "Heavy rain and flood event"
    if "event_types" in data.columns:
        events = str(data["event_types"].dropna().iloc[0]) if not data["event_types"].dropna().empty else ""
        if events:
            event_text = events.replace("|", ", ")

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.45), sharex=True, gridspec_kw={"hspace": 0.28})

    axes[0].plot(hours, data["pv_mw"], color=OKABE_ITO["black"], lw=1.8, label="Actual PV")
    axes[0].plot(hours, data["anchor_bid_mw"], color=OKABE_ITO["blue"], lw=1.4, ls="--", label="Deterministic anchor bid")
    axes[0].plot(
        hours,
        data["llm_hybrid_bid_mw"],
        color=OKABE_ITO["vermillion"],
        lw=1.8,
        label="LLM hybrid bid",
    )
    axes[0].set_ylabel("PV or bid (MW)")
    axes[0].set_ylim(-800, 17500)
    axes[0].set_title(f"{case_date}: {event_text}", loc="left", fontsize=9, pad=5)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")

    axes[1].plot(hours, data["da_lmp"], color=OKABE_ITO["orange"], lw=1.6, label="Day-ahead price")
    axes[1].plot(hours, data["rt_lmp"], color=OKABE_ITO["sky_blue"], lw=1.6, label="Real-time price")
    axes[1].fill_between(
        hours,
        data["da_lmp"].astype(float),
        data["rt_lmp"].astype(float),
        color="0.8",
        alpha=0.35,
        label="DA-RT spread",
    )
    axes[1].set_ylabel("Price (USD/MWh)")
    axes[1].legend(frameon=False, ncol=3, loc="upper left")

    axes[2].plot(
        hours,
        data["anchor_abs_imbalance_mw"],
        color=OKABE_ITO["blue"],
        lw=1.4,
        ls="--",
        label="Anchor absolute imbalance",
    )
    axes[2].plot(
        hours,
        data["llm_abs_imbalance_mw"],
        color=OKABE_ITO["vermillion"],
        lw=1.8,
        label="LLM hybrid absolute imbalance",
    )
    axes[2].set_ylabel("Absolute imbalance (MW)")
    axes[2].legend(frameon=False, ncol=2, loc="upper left")

    colors = np.where(data["revenue_delta_usd"].to_numpy(dtype=float) >= 0, OKABE_ITO["bluish_green"], OKABE_ITO["vermillion"])
    axes[3].bar(hours, data["revenue_delta_usd"] / 1_000.0, color=colors, alpha=0.82, label="Hourly revenue delta")
    axes[3].plot(
        hours,
        data["cumulative_revenue_delta_usd"] / 1_000.0,
        color=OKABE_ITO["black"],
        lw=1.5,
        marker="o",
        ms=3,
        label="Cumulative revenue delta",
    )
    axes[3].axhline(0, color="0.3", lw=0.8)
    axes[3].set_ylabel("Revenue delta\n(thousand USD)")
    axes[3].set_xlabel("Local hour")
    axes[3].legend(frameon=False, ncol=2, loc="upper left")

    for idx, ax in enumerate(axes):
        ax.text(
            -0.055,
            1.02,
            chr(ord("a") + idx),
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
    axes[-1].set_xticks(np.arange(0, 24, 2))
    axes[-1].set_xlim(-0.5, 23.5)
    fig.subplots_adjust(top=0.965, left=0.14, right=0.985, bottom=0.075)
    save_figure(fig, output_dir, case_study_stem(case_date))


def plot_case_screening_summary(funnel_table: pd.DataFrame, top_days: pd.DataFrame, output_dir: Path) -> None:
    funnel = funnel_table.copy()
    top = top_days.copy()
    criterion_col = "Criterion" if "Criterion" in funnel.columns else "criterion"
    days_col = "Days" if "Days" in funnel.columns else "day_count"
    share_col = "Share of test days (%)" if "Share of test days (%)" in funnel.columns else "share_of_test_days_pct"
    funnel[days_col] = funnel[days_col].astype(int)
    funnel[share_col] = funnel[share_col].astype(float)
    top["revenue_delta_mean_kusd"] = top["revenue_delta_mean_kusd"].astype(float)
    top["imbalance_delta_mean_mwh"] = top["imbalance_delta_mean_mwh"].astype(float)

    short_labels = [
        "All days",
        "Mean revenue win",
        ">=80% seeds win",
        "All-seed win",
        "All-seed + lower imbalance",
        "Extreme clean wins",
    ]
    if len(funnel) == len(short_labels):
        funnel["plot_label"] = short_labels
    else:
        funnel["plot_label"] = funnel[criterion_col].astype(str)

    top = top.sort_values("revenue_delta_mean_kusd", ascending=True).tail(5)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.5), gridspec_kw={"width_ratios": [1.08, 1.0]})

    y = np.arange(len(funnel))
    colors = ["0.68"] * (len(funnel) - 1) + [OKABE_ITO["bluish_green"]]
    axes[0].barh(y, funnel[days_col], color=colors, edgecolor="0.25", linewidth=0.4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(funnel["plot_label"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Days in 2024")
    axes[0].set_title("Fixed screening funnel")
    xmax = float(funnel[days_col].max())
    for idx, row in funnel.iterrows():
        axes[0].text(
            float(row[days_col]) + xmax * 0.015,
            idx,
            f"{int(row[days_col])} ({row[share_col]:.1f}%)",
            va="center",
            fontsize=7,
        )
    axes[0].set_xlim(0, xmax * 1.34)

    y2 = np.arange(len(top))
    axes[1].barh(y2, top["revenue_delta_mean_kusd"], color=OKABE_ITO["bluish_green"], alpha=0.90)
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels(pd.to_datetime(top["local_date"]).dt.strftime("%b %d"))
    axes[1].set_xlabel("Mean daily revenue gain (thousand USD)")
    axes[1].set_title("Top extreme-weather clean wins")
    for idx, row in enumerate(top.itertuples(index=False)):
        imbalance = getattr(row, "imbalance_delta_mean_mwh")
        axes[1].text(
            getattr(row, "revenue_delta_mean_kusd") + 4,
            idx,
            f"imb. {imbalance:.0f} MWh",
            va="center",
            fontsize=7,
        )
    axes[1].set_xlim(0, float(top["revenue_delta_mean_kusd"].max()) * 1.42)

    for idx, ax in enumerate(axes):
        ax.text(
            -0.10,
            1.04,
            chr(ord("A") + idx),
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
    fig.suptitle("Extreme-weather clean-win screening on the 2024 test year", y=1.04)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_case_screening_summary")


def build_paper_figures(
    table_dir: Path,
    output_dir: Path,
    results_dir: Path = Path("results"),
    case_date: str = "2024-07-14",
) -> None:
    apply_publication_style()
    plot_pipeline_schematic(output_dir)
    bidding = pd.read_csv(table_dir / "paper_table_bidding_main.csv")
    plot_revenue_cvar_tradeoff(bidding, output_dir)
    figure_rows = [
        {
            "figure": "fig_pipeline_schematic",
            "description": "Public-data evaluation pipeline from CAISO, HRRR, and NWS inputs to forecasts, residual scenarios, and proxy decision outputs.",
        },
        {
            "figure": "fig_revenue_cvar_tradeoff",
            "description": "Revenue-CVaR95 trade-off for baseline and hybrid bidding strategies.",
        },
        {
            "figure": "fig_settlement_proxy_cvar_tradeoff",
            "description": "Manuscript alias for the revenue-CVaR95 trade-off figure.",
        }
    ]

    scenario_replacement = build_scenario_replacement_data(results_dir)
    if not scenario_replacement.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        scenario_replacement.to_csv(output_dir / "fig_scenario_replacement_effects_data.csv", index=False)
        plot_scenario_replacement_effects(scenario_replacement, output_dir)
        figure_rows.append(
            {
                "figure": "fig_scenario_replacement_effects",
                "description": "Seed-paired decision effects from replacing rule-text scenarios with LLM scenarios under the matched MLP rule-text deterministic PV anchor.",
            }
        )

    paired_ci_path = results_dir / "supplementary" / "supp_daily_paired_block_bootstrap_ci.csv"
    if not paired_ci_path.exists():
        paired_ci_path = results_dir / "latest_llm_rule_daily_uncertainty" / "latest_llm_rule_daily_paired_block_bootstrap_ci.csv"
    if paired_ci_path.exists():
        paired_day = build_paired_day_evidence_data(pd.read_csv(paired_ci_path))
        output_dir.mkdir(parents=True, exist_ok=True)
        paired_day.to_csv(output_dir / "fig_paired_day_evidence_current_data.csv", index=False)
        plot_paired_day_evidence(paired_day, output_dir)
        figure_rows.append(
            {
                "figure": "fig_paired_day_evidence",
                "description": "Paired day-level block-bootstrap uncertainty for value, CVaR95 loss, and imbalance.",
            }
        )

    lp_weight_path = table_dir / "paper_table_llm_hybrid_lp_weight_sensitivity.csv"
    if lp_weight_path.exists():
        lp_weight = pd.read_csv(lp_weight_path)
        plot_lp_weight_sensitivity(lp_weight, output_dir)
        figure_rows.append(
            {
                "figure": "fig_lp_weight_sensitivity",
                "description": "LLM hybrid blend sensitivity to the LP-anchor mixing weight.",
            }
        )

    penalty_path = table_dir / "paper_table_llm_hybrid_same_penalty.csv"
    if penalty_path.exists():
        penalty = pd.read_csv(penalty_path)
        plot_same_penalty_sensitivity(penalty, output_dir)
        figure_rows.append(
            {
                "figure": "fig_same_penalty_sensitivity",
                "description": "LLM hybrid blend robustness across deviation-penalty settings.",
            }
        )

    funnel_path = table_dir / "paper_table_case_screening_funnel.csv"
    top_extreme_path = table_dir / "paper_table_case_screening_top_extreme_clean_wins.csv"
    if funnel_path.exists() and top_extreme_path.exists():
        funnel = pd.read_csv(funnel_path)
        top_extreme = pd.read_csv(top_extreme_path)
        plot_case_screening_summary(funnel, top_extreme, output_dir)
        figure_rows.append(
            {
                "figure": "fig_case_screening_summary",
                "description": "Fixed case-screening funnel and top extreme-weather clean-win days.",
            }
        )

    preds = pd.read_csv(results_dir / "test_predictions.csv")
    train_residuals = pd.read_csv(results_dir / "train_residuals.csv")
    audit = pd.read_csv(results_dir / "data_audit.csv")
    rated_capacity = float(audit.iloc[0]["rated_capacity_mw"])
    hybrid_bid = reconstruct_hybrid_blend_bid_series(
        preds=preds,
        train_residuals=train_residuals,
        rated_capacity=rated_capacity,
    )
    case_mask = preds["local_date"].astype(str).eq(case_date)
    case_day = preds.loc[case_mask].copy()
    if not case_day.empty:
        case_data = prepare_case_study_data(
            case_day,
            hybrid_bid.loc[case_mask],
            deviation_penalty=50.0,
            anchor_col="pv_mlp_text",
        )
        stem = case_study_stem(case_date)
        output_dir.mkdir(parents=True, exist_ok=True)
        case_data.to_csv(output_dir / f"{stem}_data.csv", index=False)
        plot_extreme_weather_case_study(case_data, output_dir, case_date=case_date)
        figure_rows.append(
            {
                "figure": stem,
                "description": "Extreme-weather case study comparing actual PV, anchor bids, LLM hybrid bids, prices, imbalance, and revenue deltas.",
            }
        )
    lines = ["# Paper Figures", ""]
    for row in figure_rows:
        lines.extend(
            [
                f"## {row['figure']}",
                "",
                row["description"],
                "",
                f"- PDF: `{row['figure']}.pdf`",
                f"- PNG: `{row['figure']}.png`",
                "",
            ]
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready English figures as PDF and PNG.")
    parser.add_argument("--table-dir", type=Path, default=Path("results/paper_tables"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper_figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--case-date", default="2024-07-14")
    args = parser.parse_args()
    build_paper_figures(args.table_dir, args.output_dir, args.results_dir, args.case_date)


if __name__ == "__main__":
    main()
