from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SELECTED_WEIGHT = 0.25

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


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def reset_outputs() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def copy_pipeline_overview() -> None:
    source = ASSETS / "fig_pipeline_method_overview.png"
    if source.exists():
        shutil.copy2(source, FIGURES / "fig_pipeline_method_overview.png")


def write_table(name: str, table: pd.DataFrame) -> None:
    table.to_csv(TABLES / f"{name}.csv", index=False)
    table.to_latex(TABLES / f"{name}.tex", index=False)


def build_main_forecast_table(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary[
        [
            "target",
            "model",
            "feature_role",
            "seed_count",
            "rmse_mean",
            "rmse_sd",
            "mae_mean",
            "mae_sd",
        ]
    ].rename(
        columns={
            "target": "Target",
            "model": "Model",
            "feature_role": "Feature role",
            "seed_count": "Seeds",
            "rmse_mean": "RMSE mean",
            "rmse_sd": "RMSE SD",
            "mae_mean": "MAE mean",
            "mae_sd": "MAE SD",
        }
    )
    return out.round({"RMSE mean": 2, "RMSE SD": 2, "MAE mean": 2, "MAE SD": 2})


def build_forecast_seed_table(seed_summary: pd.DataFrame) -> pd.DataFrame:
    out = seed_summary[
        [
            "comparison_label",
            "metric",
            "mean_delta",
            "mean_improvement_pct",
            "ci95_low",
            "ci95_high",
            "paired_t_p_value",
        ]
    ].rename(
        columns={
            "comparison_label": "Comparison",
            "metric": "Metric",
            "mean_delta": "Mean improvement",
            "mean_improvement_pct": "Mean improvement (%)",
            "ci95_low": "95% CI low",
            "ci95_high": "95% CI high",
            "paired_t_p_value": "Paired t-test p",
        }
    )
    return out.round(
        {
            "Mean improvement": 3,
            "Mean improvement (%)": 3,
            "95% CI low": 3,
            "95% CI high": 3,
            "Paired t-test p": 4,
        }
    )


def build_decision_table(decision: pd.DataFrame) -> pd.DataFrame:
    out = decision[
        [
            "strategy",
            "pv_scenario",
            "rt_scenario",
            "anchor",
            "lp_weight",
            "seed_count",
            "value_musd",
            "cvar95_loss_kusd_h",
            "imbalance_gwh",
        ]
    ].rename(
        columns={
            "strategy": "Strategy",
            "pv_scenario": "PV scenario",
            "rt_scenario": "RT scenario",
            "anchor": "Anchor",
            "lp_weight": "w",
            "seed_count": "Seeds",
            "value_musd": "Value (M USD)",
            "cvar95_loss_kusd_h": "CVaR95 loss (k USD/h)",
            "imbalance_gwh": "Imbalance (GWh)",
        }
    )
    return out.round(
        {
            "w": 2,
            "Value (M USD)": 2,
            "CVaR95 loss (k USD/h)": 2,
            "Imbalance (GWh)": 1,
        }
    )


def build_validation_weight_table(weights: pd.DataFrame) -> pd.DataFrame:
    out = weights[
        [
            "w",
            "mean_value_musd",
            "min_value_musd",
            "mean_cvar95_loss_kusd_h",
            "max_cvar95_loss_kusd_h",
            "mean_imbalance_gwh",
            "max_imbalance_gwh",
            "score",
            "rank",
        ]
    ].rename(
        columns={
            "w": "w",
            "mean_value_musd": "Mean value (M USD)",
            "min_value_musd": "Minimum value (M USD)",
            "mean_cvar95_loss_kusd_h": "Mean CVaR95 loss (k USD/h)",
            "max_cvar95_loss_kusd_h": "Maximum CVaR95 loss (k USD/h)",
            "mean_imbalance_gwh": "Mean imbalance (GWh)",
            "max_imbalance_gwh": "Maximum imbalance (GWh)",
            "score": "Robust score",
            "rank": "Rank",
        }
    )
    return out.round(
        {
            "w": 2,
            "Mean value (M USD)": 2,
            "Minimum value (M USD)": 2,
            "Mean CVaR95 loss (k USD/h)": 2,
            "Maximum CVaR95 loss (k USD/h)": 2,
            "Mean imbalance (GWh)": 1,
            "Maximum imbalance (GWh)": 1,
            "Robust score": 3,
        }
    )


def build_paired_seed_table(seed_summary: pd.DataFrame) -> pd.DataFrame:
    out = seed_summary[
        [
            "metric",
            "mean",
            "ci95_low",
            "ci95_high",
            "unit",
            "seed_count",
            "paired_t_p_value",
        ]
    ].rename(
        columns={
            "metric": "Metric",
            "mean": "Mean",
            "ci95_low": "95% CI low",
            "ci95_high": "95% CI high",
            "unit": "Unit",
            "seed_count": "Seeds",
            "paired_t_p_value": "Paired t-test p",
        }
    )
    return out.round({"Mean": 3, "95% CI low": 3, "95% CI high": 3, "Paired t-test p": 4})


def build_forecast_slice_table(slices: pd.DataFrame) -> pd.DataFrame:
    return slices.round(
        {
            "reference_rmse": 2,
            "llm_rmse": 2,
            "rmse_improvement_pct": 2,
            "reference_mae": 2,
            "llm_mae": 2,
            "mae_improvement_pct": 2,
        }
    )


def build_event_day_table(event_days: pd.DataFrame) -> pd.DataFrame:
    out = event_days[
        [
            "local_date",
            "event_types",
            "reference",
            "value_delta_kusd",
            "absolute_imbalance_reduction_mwh",
            "pv_rmse_improvement_pct",
        ]
    ].rename(
        columns={
            "local_date": "Date",
            "event_types": "Event labels",
            "reference": "Reference",
            "value_delta_kusd": "Value delta (k USD)",
            "absolute_imbalance_reduction_mwh": "Imbalance reduction (MWh)",
            "pv_rmse_improvement_pct": "PV RMSE improvement (%)",
        }
    )
    out["Event labels"] = out["Event labels"].astype(str).str.replace("|", "; ", regex=False)
    return out.round(
        {
            "Value delta (k USD)": 1,
            "Imbalance reduction (MWh)": 0,
            "PV RMSE improvement (%)": 1,
        }
    )


def write_tables() -> None:
    write_table("main_forecast_table", build_main_forecast_table(pd.read_csv(DATA / "forecast_main.csv")))
    write_table("forecast_seed_summary", build_forecast_seed_table(pd.read_csv(DATA / "forecast_seed_summary.csv")))
    write_table("forecast_slice_metrics", build_forecast_slice_table(pd.read_csv(DATA / "forecast_slice_metrics.csv")))
    write_table("validation_weight_selection", build_validation_weight_table(pd.read_csv(DATA / "validation_weight_selection.csv")))
    write_table("main_decision_table", build_decision_table(pd.read_csv(DATA / "decision_table.csv")))
    write_table("decision_frontier_points", pd.read_csv(DATA / "decision_frontier_points.csv").round(3))
    write_table("paired_seed_deltas", pd.read_csv(DATA / "paired_seed_deltas.csv").round(3))
    write_table("paired_seed_summary", build_paired_seed_table(pd.read_csv(DATA / "paired_seed_summary.csv")))
    write_table("event_day_examples", build_event_day_table(pd.read_csv(DATA / "event_day_examples.csv")))


def plot_forecast_signal() -> None:
    data = pd.read_csv(DATA / "forecast_seed_summary.csv")
    data["metric"] = data["metric"].astype(str).str.upper()
    panels = [
        (
            "PV generation",
            ["PV LLM cloud-rule vs no-text", "PV LLM cloud-rule vs rule-core"],
            "Error reduction (MW)",
            OKABE_ITO["blue"],
        ),
        (
            "Real-time price",
            ["RT LLM-rule vs no-text", "RT LLM-rule vs rule-text"],
            "Error reduction (USD/MWh)",
            OKABE_ITO["orange"],
        ),
    ]
    metric_style = {"RMSE": ("o", OKABE_ITO["bluish_green"]), "MAE": ("s", OKABE_ITO["sky_blue"])}
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25))
    for panel_idx, (ax, (title, comparisons, xlabel, title_color)) in enumerate(zip(axes, panels)):
        subset = data[data["comparison_label"].isin(comparisons)].copy()
        subset["comparison_label"] = pd.Categorical(subset["comparison_label"], comparisons, ordered=True)
        subset = subset.sort_values(["comparison_label", "metric"])
        labels = [label.replace("PV LLM cloud-rule ", "").replace("RT LLM-rule ", "") for label in comparisons]
        y_base = np.arange(len(comparisons), dtype=float)
        for metric, offset in [("RMSE", -0.12), ("MAE", 0.12)]:
            rows = subset[subset["metric"].eq(metric)].set_index("comparison_label")
            xs, lows, highs, ys = [], [], [], []
            for i, comparison in enumerate(comparisons):
                if comparison not in rows.index:
                    continue
                row = rows.loc[comparison]
                mean = float(row["mean_delta"])
                low = float(row["ci95_low"])
                high = float(row["ci95_high"])
                xs.append(mean)
                lows.append(mean - low)
                highs.append(high - mean)
                ys.append(y_base[i] + offset)
            if xs:
                marker, color = metric_style[metric]
                ax.errorbar(
                    xs,
                    ys,
                    xerr=[lows, highs],
                    fmt=marker,
                    color=color,
                    markeredgecolor="black",
                    markeredgewidth=0.45,
                    markersize=5.3,
                    elinewidth=1.1,
                    capsize=3,
                    label=metric,
                    zorder=3,
                )
        ax.axvline(0.0, color="0.35", lw=0.8, ls=":")
        ax.set_yticks(y_base)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, color=title_color, fontsize=9.3)
        ax.grid(True, axis="x", alpha=0.18)
        ax.grid(False, axis="y")
        x_low = float(np.nanmin(subset["ci95_low"]))
        x_high = float(np.nanmax(subset["ci95_high"]))
        x_span = max(x_high - x_low, 1e-6)
        ax.set_xlim(min(0.0, x_low - 0.10 * x_span), x_high + 0.12 * x_span)
        ax.text(-0.13, 1.04, chr(ord("a") + panel_idx), transform=ax.transAxes, fontsize=10, fontweight="bold")
    axes[1].legend(frameon=False, loc="lower right", title="Metric", title_fontsize=7.5)
    fig.tight_layout()
    save_figure(fig, "fig_selected_cloud_rule_forecast_signal")


def row_at_weight(df: pd.DataFrame, weight: float) -> pd.Series:
    rows = df[np.isclose(pd.to_numeric(df["lp_weight"], errors="coerce"), weight)]
    if rows.empty:
        raise ValueError(f"No frontier row found for lp_weight={weight}")
    return rows.iloc[0]


def annotate_weight(ax: plt.Axes, row: pd.Series, dx: float, dy: float, label: str | None = None) -> None:
    text = label or f"w={float(row['lp_weight']):.2f}"
    ax.annotate(
        text,
        xy=(float(row["cvar_kusd_h"]), float(row["value_musd"])),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=6.7,
        ha="center",
        va="center",
        color="0.18",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "0.84", "linewidth": 0.3, "alpha": 0.88},
    )


def plot_decision_frontier() -> None:
    data = pd.read_csv(DATA / "decision_frontier_points.csv")
    no_text = data[data["path"].eq("No-text LP-anchor path")].sort_values("lp_weight")
    llm = data[data["path"].eq("LLM cloud-rule LP-anchor path")].sort_values("lp_weight")
    selected = row_at_weight(llm, SELECTED_WEIGHT)

    fig, ax = plt.subplots(figsize=(4.85, 3.85))
    polygon_x = np.r_[llm["cvar_kusd_h"].to_numpy(), no_text["cvar_kusd_h"].to_numpy()[::-1]]
    polygon_y = np.r_[llm["value_musd"].to_numpy(), no_text["value_musd"].to_numpy()[::-1]]
    ax.fill(polygon_x, polygon_y, color=OKABE_ITO["vermillion"], alpha=0.08, zorder=0)
    ax.plot(
        no_text["cvar_kusd_h"],
        no_text["value_musd"],
        color="0.35",
        marker="o",
        markersize=4.4,
        markerfacecolor="white",
        markeredgewidth=1.0,
        lw=1.2,
        ls="--",
        label="No-text LP-anchor path",
        zorder=2,
    )
    ax.plot(
        llm["cvar_kusd_h"],
        llm["value_musd"],
        color=OKABE_ITO["vermillion"],
        marker="o",
        markersize=4.8,
        markerfacecolor=OKABE_ITO["vermillion"],
        markeredgecolor="black",
        markeredgewidth=0.55,
        lw=1.55,
        label="LLM cloud-rule LP-anchor path",
        zorder=3,
    )
    ax.scatter(
        float(selected["cvar_kusd_h"]),
        float(selected["value_musd"]),
        s=138,
        marker="*",
        color=OKABE_ITO["orange"],
        edgecolor="black",
        linewidth=0.85,
        zorder=5,
    )
    anchor = row_at_weight(llm, 0.0)
    ax.scatter(float(anchor["cvar_kusd_h"]), float(anchor["value_musd"]), s=60, marker="D", color="white", edgecolor="black", linewidth=0.85, zorder=5)

    for _, row in llm.iterrows():
        w = float(row["lp_weight"])
        if np.isclose(w, SELECTED_WEIGHT):
            annotate_weight(ax, row, 16, 11, f"selected w={w:.2f}")
        elif np.isclose(w, 0.0):
            annotate_weight(ax, row, -2, -20, "anchor only\nw=0")
        elif np.isclose(w, 1.0):
            annotate_weight(ax, row, -25, -12, "pure LLM LP\nw=1")
        elif np.isclose(w, 0.10) or np.isclose(w, 0.75):
            annotate_weight(ax, row, 0, 11)
    annotate_weight(ax, row_at_weight(no_text, 1.0), -7, 16, "pure no-text LP\nw=1")

    x_min = min(float(no_text["cvar_kusd_h"].min()), float(llm["cvar_kusd_h"].min())) - 6.0
    x_max = max(float(no_text["cvar_kusd_h"].max()), float(llm["cvar_kusd_h"].max())) + 7.0
    y_min = min(float(no_text["value_musd"].min()), float(llm["value_musd"].min())) - 5.0
    y_max = max(float(no_text["value_musd"].max()), float(llm["value_musd"].max())) + 5.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("CVaR95 loss (k USD/h)")
    ax.set_ylabel("Proxy value (M USD)")
    ax.set_title("LP-anchor weight path", fontsize=9.4)
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False, fontsize=7.3, loc="lower left")
    save_figure(fig, "fig_selected_cloud_rule_decision_frontier")


def plot_case_study() -> None:
    data = pd.read_csv(DATA / "case_2025_03_07.csv").sort_values("timestamp_utc").copy()
    hours = data["hour"].astype(int).to_numpy()
    case_date = str(data["local_date"].dropna().iloc[0])
    event_text = str(data["event_types"].dropna().iloc[0]).replace("|", ", ")

    fig, axes = plt.subplots(4, 1, figsize=(7.25, 7.45), sharex=True, gridspec_kw={"hspace": 0.30})
    axes[0].plot(hours, data["pv_mw"], color=OKABE_ITO["black"], lw=1.8, label="Actual PV")
    axes[0].plot(hours, data["common_anchor_bid_mw"], color=OKABE_ITO["blue"], lw=1.4, ls="--", label="MLP no-text anchor")
    axes[0].plot(
        hours,
        data["llm_common_anchor_hybrid_bid_mw"],
        color=OKABE_ITO["vermillion"],
        lw=1.8,
        label="LLM LP-anchor hybrid (w=.25)",
    )
    axes[0].set_ylabel("PV or quantity (MW)")
    pv_top = float(data[["pv_mw", "common_anchor_bid_mw", "llm_common_anchor_hybrid_bid_mw"]].max().max())
    axes[0].set_ylim(0.0, pv_top * 1.28)
    axes[0].set_title(f"{case_date}: {event_text}, common no-text anchor", loc="left", fontsize=9, pad=5)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    risk_text = (
        f"Narrative: rule cloud/rain={float(data['wx_prior_cloud_score'].iloc[0]):.1f}/{float(data['wx_prior_rain_score'].iloc[0]):.1f}\n"
        f"LLM irradiance/conf.={float(data['llm_prior_irradiance_reduction_risk'].iloc[0]):.1f}/"
        f"{float(data['llm_prior_confidence'].iloc[0]):.2f}"
    )
    axes[0].text(
        0.995,
        0.92,
        risk_text,
        transform=axes[0].transAxes,
        fontsize=7.0,
        ha="right",
        va="top",
        color="0.28",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "0.86", "linewidth": 0.3, "alpha": 0.88},
    )

    axes[1].plot(hours, data["da_lmp"], color=OKABE_ITO["orange"], lw=1.6, label="Day-ahead price")
    axes[1].plot(hours, data["rt_lmp"], color=OKABE_ITO["sky_blue"], lw=1.6, label="Real-time price")
    axes[1].fill_between(hours, data["da_lmp"].astype(float), data["rt_lmp"].astype(float), color="0.80", alpha=0.35, label="DA-RT spread")
    axes[1].set_ylabel("Price (USD/MWh)")
    price_min = float(data[["da_lmp", "rt_lmp"]].min().min())
    price_max = float(data[["da_lmp", "rt_lmp"]].max().max())
    price_span = max(price_max - price_min, 1.0)
    axes[1].set_ylim(price_min - 0.08 * price_span, price_max + 0.28 * price_span)
    axes[1].legend(frameon=False, ncol=3, loc="upper left")

    axes[2].plot(hours, data["common_anchor_abs_imbalance_mw"], color=OKABE_ITO["blue"], lw=1.4, ls="--", label="MLP no-text anchor")
    axes[2].plot(
        hours,
        data["llm_common_anchor_abs_imbalance_mw"],
        color=OKABE_ITO["vermillion"],
        lw=1.8,
        label="LLM LP-anchor hybrid (w=.25)",
    )
    imb_top = float(data[["common_anchor_abs_imbalance_mw", "llm_common_anchor_abs_imbalance_mw"]].max().max())
    axes[2].set_ylim(0.0, imb_top * 1.24)
    axes[2].set_ylabel("Absolute imbalance (MW)")
    axes[2].legend(frameon=False, ncol=2, loc="upper left")

    delta_kusd = data["common_anchor_value_delta_usd"] / 1_000.0
    cumulative_kusd = data["common_anchor_cumulative_value_delta_usd"] / 1_000.0
    bar_colors = np.where(delta_kusd.to_numpy(dtype=float) >= 0.0, OKABE_ITO["bluish_green"], OKABE_ITO["vermillion"])
    axes[3].bar(hours, delta_kusd, color=bar_colors, alpha=0.84, label="Hourly value delta")
    axes[3].plot(hours, cumulative_kusd, color=OKABE_ITO["black"], lw=1.5, marker="o", ms=3, label="Cumulative value delta")
    axes[3].axhline(0.0, color="0.30", lw=0.8)
    axes[3].set_ylabel("Value delta\n(thousand USD)")
    axes[3].set_xlabel("Local hour")
    axes[3].legend(frameon=False, ncol=2, loc="upper left")
    final_delta = float(cumulative_kusd.iloc[-1])
    lower = float(min(delta_kusd.min(), cumulative_kusd.min(), 0.0))
    upper = float(max(delta_kusd.max(), cumulative_kusd.max(), 0.0))
    y_span = max(upper - lower, 1.0)
    axes[3].set_ylim(lower - 0.10 * y_span, upper + 0.20 * y_span)
    axes[3].annotate(f"+{final_delta:.1f}k USD", xy=(float(hours[-1]), final_delta), xytext=(7, 0), textcoords="offset points", fontsize=7.3, ha="left", va="center", color="0.20", clip_on=False)

    for idx, ax in enumerate(axes):
        ax.axvspan(6, 19, color=OKABE_ITO["yellow"], alpha=0.08, zorder=0)
        ax.text(-0.055, 1.02, chr(ord("a") + idx), transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
        ax.grid(True, alpha=0.18)
    axes[-1].set_xticks(np.arange(0, 24, 2))
    axes[-1].set_xlim(-0.5, 24.8)
    fig.subplots_adjust(top=0.925, left=0.14, right=0.985, bottom=0.075)
    save_figure(fig, "fig_selected_cloud_rule_case_2025_03_07")


def main() -> None:
    reset_outputs()
    apply_style()
    copy_pipeline_overview()
    write_tables()
    plot_forecast_signal()
    plot_decision_frontier()
    plot_case_study()
    print(f"Wrote tables to {TABLES}")
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
