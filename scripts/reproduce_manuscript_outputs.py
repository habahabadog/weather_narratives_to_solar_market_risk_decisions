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
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def copy_pipeline_overview() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    source = ASSETS / "fig_pipeline_method_overview.png"
    if source.exists():
        shutil.copy2(source, FIGURES / "fig_pipeline_method_overview.png")


def reset_outputs() -> None:
    for path in (TABLES, FIGURES):
        if path.exists():
            shutil.rmtree(path)


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
    keep = seed_summary[~seed_summary["comparison_label"].eq("RT LLM-rule vs rule-text") | ~seed_summary["metric"].eq("mae")]
    out = keep[
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


def build_decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "CAISO public forecast anchor": "CAISO public forecast anchor",
        "No-text hybrid blend": "No-text hybrid",
        "LLM cloud-rule no-text-anchor hybrid blend": "LLM no-text-anchor hybrid",
        "Rule-core hybrid blend": "Rule-core hybrid reference",
        "Pure LLM cloud-rule LP": "Pure LLM cloud-rule LP",
        "LLM cloud-rule hybrid blend": "LLM rule-core-anchor hybrid",
    }
    data = summary[summary["comparison"].astype(str).eq("validation_selected")].copy()
    data["Strategy"] = data["strategy"].map(labels).fillna(data["strategy"])
    keep = [
        "Strategy",
        "pv_scenario",
        "rt_scenario",
        "anchor",
        "lp_weight",
        "seed_count",
        "value_musd",
        "cvar95_loss_kusd_h",
        "imbalance_gwh",
    ]
    out = data[[col for col in keep if col in data.columns]].rename(
        columns={
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


def build_paired_seed_table(seed_summary: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "metric",
        "mean",
        "ci95_low",
        "ci95_high",
        "unit",
        "paired_t_p_value",
    ]
    out = seed_summary[[col for col in keep if col in seed_summary.columns]].copy()
    return out.round({"mean": 3, "ci95_low": 3, "ci95_high": 3})


def build_event_day_table(event_days: pd.DataFrame) -> pd.DataFrame:
    out = event_days[
        [
            "local_date",
            "event_types",
            "value_delta_kusd",
            "absolute_imbalance_reduction_mwh",
            "pv_rmse_improvement_pct",
        ]
    ].rename(
        columns={
            "local_date": "Date",
            "event_types": "Event labels",
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
    TABLES.mkdir(parents=True, exist_ok=True)
    forecast_main = pd.read_csv(DATA / "forecast_main.csv")
    forecast_seed = pd.read_csv(DATA / "forecast_seed_summary.csv")
    decision_summary = pd.read_csv(DATA / "decision_summary.csv")
    seed_summary = pd.read_csv(DATA / "paired_seed_summary.csv")
    seed_summary_vs_no_text_path = DATA / "paired_seed_summary_vs_no_text.csv"
    slices = pd.read_csv(DATA / "forecast_slice_metrics.csv")
    event_days = pd.read_csv(DATA / "event_day_examples.csv")

    tables = {
        "main_forecast_table": build_main_forecast_table(forecast_main),
        "forecast_seed_summary": build_forecast_seed_table(forecast_seed),
        "main_decision_table": build_decision_table(decision_summary),
        "paired_seed_summary": build_paired_seed_table(seed_summary),
        "forecast_slice_metrics": slices.round(
            {
                "reference_rmse": 2,
                "llm_rmse": 2,
                "rmse_improvement_pct": 2,
                "reference_mae": 2,
                "llm_mae": 2,
                "mae_improvement_pct": 2,
            }
        ),
        "event_day_examples": build_event_day_table(event_days),
    }
    if seed_summary_vs_no_text_path.exists():
        tables["paired_seed_summary_vs_no_text"] = build_paired_seed_table(pd.read_csv(seed_summary_vs_no_text_path))

    for name, table in tables.items():
        table.to_csv(TABLES / f"{name}.csv", index=False)
        table.to_latex(TABLES / f"{name}.tex", index=False)


def plot_forecast_slices() -> None:
    metrics = pd.read_csv(DATA / "forecast_slice_metrics.csv")
    plot_data = metrics[
        metrics["slice"].isin(["All test hours", "Solar hours", "Extreme-event hours", "Extreme solar hours"])
    ].copy()
    targets = ["PV generation", "Real-time price"]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), sharey=False)
    for ax, target in zip(axes, targets):
        subset = plot_data[plot_data["target"].eq(target)].copy()
        subset["slice"] = pd.Categorical(
            subset["slice"],
            ["All test hours", "Solar hours", "Extreme-event hours", "Extreme solar hours"],
            ordered=True,
        )
        subset = subset.sort_values("slice")
        y = np.arange(len(subset))
        values = subset["rmse_improvement_pct"].astype(float).to_numpy()
        colors = np.where(values >= 0.0, OKABE_ITO["bluish_green"], OKABE_ITO["vermillion"])
        ax.barh(y, values, color=colors, alpha=0.88)
        ax.axvline(0.0, color="0.25", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(subset["slice"].astype(str))
        ax.invert_yaxis()
        ax.set_xlabel("RMSE improvement (%)")
        ax.set_title(target)
    axes[0].text(-0.15, 1.04, "a", transform=axes[0].transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
    axes[1].text(-0.15, 1.04, "b", transform=axes[1].transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
    fig.tight_layout()
    save_figure(fig, "fig_forecast_slices")


def plot_forecast_signal() -> None:
    data = pd.read_csv(DATA / "forecast_seed_summary.csv")
    data["metric"] = data["metric"].astype(str).str.upper()
    panels = [
        ("PV generation", ["PV LLM cloud-rule vs no-text", "PV LLM cloud-rule vs rule-core"], "Error reduction (MW)", OKABE_ITO["blue"]),
        ("Real-time price", ["RT LLM-rule vs no-text", "RT LLM-rule vs rule-text"], "Error reduction (USD/MWh)", OKABE_ITO["orange"]),
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
                ax.errorbar(xs, ys, xerr=[lows, highs], fmt=marker, color=color, markeredgecolor="black", markeredgewidth=0.45, markersize=5.3, elinewidth=1.1, capsize=3, label=metric, zorder=3)
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
        ax.text(-0.13, 1.04, chr(ord("a") + panel_idx), transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
    axes[1].legend(frameon=False, loc="lower right", title="Metric", title_fontsize=7.5)
    fig.tight_layout()
    save_figure(fig, "fig_forecast_signal")


def _format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value == 0.0:
        return "<1e-12"
    if value < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3f}"


def _draw_value_cvar_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    style = {
        "CAISO public forecast anchor": ("CAISO\npublic", "0.45", "D", 58),
        "No-text hybrid blend": ("No-text\nhybrid", OKABE_ITO["bluish_green"], "o", 66),
        "LLM cloud-rule no-text-anchor hybrid blend": ("LLM no-text\nanchor", OKABE_ITO["vermillion"], "*", 132),
        "Rule-core hybrid blend": ("Rule-core\nreference", OKABE_ITO["blue"], "s", 72),
        "Pure LLM cloud-rule LP": ("Pure LLM LP", OKABE_ITO["orange"], "^", 82),
        "LLM cloud-rule hybrid blend": ("LLM rule-core", OKABE_ITO["reddish_purple"], "P", 78),
    }
    rule = data[data["strategy"].eq("Rule-core hybrid blend")].iloc[0]
    ax.axvline(float(rule["cvar95_loss_kusd_h"]), color="0.55", lw=0.75, ls="--", alpha=0.35, zorder=0)
    ax.axhline(float(rule["value_musd"]), color="0.55", lw=0.75, ls="--", alpha=0.35, zorder=0)
    no_text = data[data["strategy"].eq("No-text hybrid blend")]
    if not no_text.empty:
        ax.axvline(float(no_text.iloc[0]["cvar95_loss_kusd_h"]), color=OKABE_ITO["bluish_green"], lw=0.70, ls=":", alpha=0.28, zorder=0)
        ax.axhline(float(no_text.iloc[0]["value_musd"]), color=OKABE_ITO["bluish_green"], lw=0.70, ls=":", alpha=0.28, zorder=0)
    offsets = {
        "CAISO public forecast anchor": (7, -6, "left", "top"),
        "No-text hybrid blend": (-8, 9, "right", "bottom"),
        "LLM cloud-rule no-text-anchor hybrid blend": (8, -18, "left", "top"),
        "Rule-core hybrid blend": (7, 8, "left", "bottom"),
        "Pure LLM cloud-rule LP": (-8, -9, "right", "top"),
        "LLM cloud-rule hybrid blend": (7, 6, "left", "bottom"),
    }
    for _, row in data.iterrows():
        label, color, marker, size = style.get(str(row["strategy"]), (str(row["strategy"]), OKABE_ITO["black"], "o", 70))
        x_value = float(row["cvar95_loss_kusd_h"])
        y_value = float(row["value_musd"])
        ax.scatter(x_value, y_value, s=size, color=color, marker=marker, edgecolor="black", linewidth=0.75, zorder=3)
        dx, dy, ha, va = offsets.get(str(row["strategy"]), (5, 4, "left", "bottom"))
        ax.annotate(
            label,
            xy=(x_value, y_value),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.0,
            ha=ha,
            va=va,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "0.86", "linewidth": 0.30, "alpha": 0.92},
        )
    fused = data[data["strategy"].eq("LLM cloud-rule no-text-anchor hybrid blend")]
    if not no_text.empty and not fused.empty:
        ax.annotate("", xy=(float(fused.iloc[0]["cvar95_loss_kusd_h"]), float(fused.iloc[0]["value_musd"])), xytext=(float(no_text.iloc[0]["cvar95_loss_kusd_h"]), float(no_text.iloc[0]["value_musd"])), arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "0.25", "shrinkA": 8, "shrinkB": 10})
    ax.set_xlim(float(data["cvar95_loss_kusd_h"].min()) - 5.5, float(data["cvar95_loss_kusd_h"].max()) + 5.8)
    ax.set_ylim(float(data["value_musd"].min()) - 7.0, float(data["value_musd"].max()) + 8.0)
    ax.set_xlabel("CVaR95 loss (k USD/h)")
    ax.set_ylabel("Proxy value (M USD)")
    ax.grid(True, alpha=0.18)


def plot_decision_evidence() -> None:
    decision = pd.read_csv(DATA / "decision_summary.csv")
    fused_deltas = DATA / "paired_seed_deltas_fused_vs_no_text.csv"
    fused_summary = DATA / "paired_seed_summary_fused_vs_no_text.csv"
    deltas = pd.read_csv(fused_deltas if fused_deltas.exists() else DATA / "paired_seed_deltas.csv")
    summary = pd.read_csv(fused_summary if fused_summary.exists() else DATA / "paired_seed_summary.csv")
    metrics = [
        ("Value gain", "value_delta_musd", "Value gain (M USD)", OKABE_ITO["orange"]),
        ("CVaR95 loss reduction", "cvar95_loss_reduction_kusd_h", "CVaR95 loss reduction (k USD/h)", OKABE_ITO["bluish_green"]),
        ("Imbalance reduction", "imbalance_reduction_gwh", "Imbalance reduction (GWh)", OKABE_ITO["blue"]),
    ]
    summary_by_metric = {str(row["metric"]): row for _, row in summary.iterrows()}
    fig = plt.figure(figsize=(7.25, 5.65))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.22], hspace=0.58, wspace=0.42)
    trade_ax = fig.add_subplot(gs[:, 0])
    _draw_value_cvar_panel(trade_ax, decision)
    trade_ax.text(-0.18, 1.02, "a", transform=trade_ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
    trade_ax.set_title("Value-risk position", fontsize=9.2)
    rng = np.random.default_rng(27)
    for idx, (metric_name, column, xlabel, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx, 1])
        values = deltas[column].astype(float).to_numpy()
        row = summary_by_metric[metric_name]
        mean = float(row["mean"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        jitter = rng.normal(0.0, 0.035, len(values))
        ax.scatter(values, jitter, s=28, color=color, alpha=0.82, edgecolor="white", linewidth=0.35, zorder=3)
        ax.errorbar(mean, 0.0, xerr=[[mean - low], [high - mean]], fmt="o", color="black", markerfacecolor=color, markeredgecolor="black", markersize=5.2, elinewidth=1.15, capsize=3.5, zorder=4)
        x_min = min(float(values.min()), low)
        x_max = max(float(values.max()), high)
        span = max(x_max - x_min, 1e-6)
        ax.set_xlim(x_min - 0.18 * span, x_max + 0.36 * span)
        ax.set_ylim(-0.15, 0.15)
        ax.set_yticks([])
        ax.set_xlabel(xlabel)
        ax.set_title(metric_name, fontsize=8.8, loc="left", pad=2)
        ax.grid(True, axis="x", alpha=0.18)
        ax.grid(False, axis="y")
        ax.text(-0.12, 1.04, chr(ord("b") + idx), transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
        ax.text(0.98, 0.90, f"paired t p={_format_p_value(float(row['paired_t_p_value']))}", transform=ax.transAxes, fontsize=7.0, ha="right", va="top", color="0.25")
    save_figure(fig, "fig_decision_evidence")


def plot_value_cvar() -> None:
    data = pd.read_csv(DATA / "decision_summary.csv")
    style = {
        "CAISO public forecast anchor": ("CAISO public\nanchor", "0.45", "D", 82),
        "No-text hybrid blend": ("No-text\nhybrid", OKABE_ITO["bluish_green"], "o", 94),
        "LLM cloud-rule no-text-anchor hybrid blend": ("LLM no-text\nanchor", OKABE_ITO["vermillion"], "*", 165),
        "Rule-core hybrid blend": ("Rule-core hybrid\nreference", OKABE_ITO["blue"], "s", 96),
        "Pure LLM cloud-rule LP": ("Pure LLM\ncloud-rule LP", OKABE_ITO["orange"], "^", 105),
        "LLM cloud-rule hybrid blend": ("LLM rule-core\nanchor", OKABE_ITO["reddish_purple"], "P", 112),
    }
    fig, ax = plt.subplots(figsize=(6.35, 4.05))
    rule = data[data["strategy"].eq("Rule-core hybrid blend")].iloc[0]
    llm = data[data["strategy"].eq("LLM cloud-rule no-text-anchor hybrid blend")].iloc[0]
    x_min = float(data["cvar95_loss_kusd_h"].min()) - 5.5
    x_max = float(data["cvar95_loss_kusd_h"].max()) + 5.8
    y_min = float(data["value_musd"].min()) - 7.0
    y_max = float(data["value_musd"].max()) + 8.0
    ax.axvline(float(rule["cvar95_loss_kusd_h"]), color="0.55", lw=0.8, ls="--", alpha=0.38, zorder=0)
    ax.axhline(float(rule["value_musd"]), color="0.55", lw=0.8, ls="--", alpha=0.38, zorder=0)
    no_text = data[data["strategy"].eq("No-text hybrid blend")]
    if not no_text.empty:
        ax.axvline(float(no_text.iloc[0]["cvar95_loss_kusd_h"]), color=OKABE_ITO["bluish_green"], lw=0.75, ls=":", alpha=0.30, zorder=0)
        ax.axhline(float(no_text.iloc[0]["value_musd"]), color=OKABE_ITO["bluish_green"], lw=0.75, ls=":", alpha=0.30, zorder=0)
    for _, row in data.iterrows():
        label, color, marker, size = style.get(str(row["strategy"]), (str(row["strategy"]), OKABE_ITO["black"], "o", 88))
        x = float(row["cvar95_loss_kusd_h"])
        y = float(row["value_musd"])
        ax.scatter(x, y, s=size, color=color, marker=marker, edgecolor="black", linewidth=0.8, zorder=3)
        offsets = {
            "CAISO public forecast anchor": (9, -7, "left", "top"),
            "No-text hybrid blend": (-8, 10, "right", "bottom"),
            "LLM cloud-rule no-text-anchor hybrid blend": (9, -18, "left", "top"),
            "Rule-core hybrid blend": (9, 9, "left", "bottom"),
            "Pure LLM cloud-rule LP": (-9, -10, "right", "top"),
            "LLM cloud-rule hybrid blend": (9, 7, "left", "bottom"),
        }
        dx, dy, ha, va = offsets.get(str(row["strategy"]), (5, 4, "left", "bottom"))
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.6,
            ha=ha,
            va=va,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.82", "linewidth": 0.45, "alpha": 0.96},
        )

    reference = no_text.iloc[0] if not no_text.empty else rule
    rule_x = float(reference["cvar95_loss_kusd_h"])
    rule_y = float(reference["value_musd"])
    llm_x = float(llm["cvar95_loss_kusd_h"])
    llm_y = float(llm["value_musd"])
    ax.annotate(
        "",
        xy=(llm_x, llm_y),
        xytext=(rule_x, rule_y),
        arrowprops={"arrowstyle": "->", "lw": 1.25, "color": "0.25", "shrinkA": 10, "shrinkB": 12},
    )
    value_gain = llm_y - rule_y
    cvar_gain = rule_x - llm_x
    ax.annotate(
        f"+{value_gain:.2f} M USD\n-{cvar_gain:.2f} k USD/h CVaR",
        xy=((rule_x + llm_x) / 2.0, (rule_y + llm_y) / 2.0),
        xytext=(18, 0),
        textcoords="offset points",
        fontsize=7.4,
        ha="left",
        va="center",
        color="0.25",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.84", "linewidth": 0.45, "alpha": 0.96},
    )
    ax.set_xlabel("CVaR95 loss (k USD/h, lower is better)")
    ax.set_ylabel("Test-period proxy value (M USD, higher is better)")
    ax.grid(True, alpha=0.18)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    save_figure(fig, "fig_value_cvar_tradeoff")


def plot_paired_seed_effects() -> None:
    deltas = pd.read_csv(DATA / "paired_seed_deltas.csv")
    summary = pd.read_csv(DATA / "paired_seed_summary.csv")
    metrics = [
        ("Value gain", "value_delta_musd", "Value gain\n(M USD)", OKABE_ITO["orange"]),
        ("CVaR95 loss reduction", "cvar95_loss_reduction_kusd_h", "CVaR95 loss reduction\n(k USD/h)", OKABE_ITO["bluish_green"]),
        ("Imbalance reduction", "imbalance_reduction_gwh", "Imbalance reduction\n(GWh)", OKABE_ITO["blue"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.15))
    rng = np.random.default_rng(27)
    for panel_idx, (ax, (metric_name, column, xlabel, color)) in enumerate(zip(axes, metrics)):
        values = deltas[column].astype(float).to_numpy()
        jitter = rng.normal(0.0, 0.035, len(values))
        ax.scatter(values, jitter, s=30, color=color, alpha=0.66, edgecolor="white", linewidth=0.35, zorder=3)
        row = summary[summary["metric"].eq(metric_name)].iloc[0]
        mean = float(row["mean"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        ax.errorbar(mean, 0.0, xerr=[[mean - low], [high - mean]], fmt="o", color="black", mfc=color, mec="black", ms=5, capsize=5, lw=1.3, zorder=4)
        ax.axvspan(low, high, color=color, alpha=0.08, zorder=0)
        ax.axvline(0.0, color="0.25", lw=0.8, ls=":")
        ax.set_xlabel(xlabel)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.18)
        ax.grid(False, axis="y")
        x_min = min(float(values.min()), low)
        x_max = max(float(values.max()), high)
        span = max(x_max - x_min, 1e-6)
        if column == "imbalance_reduction_gwh":
            ax.set_xlim(x_min - 0.18 * span, x_max + 0.12 * span)
        else:
            ax.set_xlim(min(0.0, x_min - 0.12 * span), x_max + 0.12 * span)
        ax.set_ylim(-0.15, 0.15)
        ax.text(-0.12, 1.03, chr(ord("a") + panel_idx), transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
    fig.tight_layout()
    save_figure(fig, "fig_paired_seed_effects")


def plot_case_study() -> None:
    data = pd.read_csv(DATA / "case_2025_03_07.csv").sort_values("timestamp_utc").copy()
    hours = data["hour"].astype(int).to_numpy()
    case_date = str(data["local_date"].dropna().iloc[0])
    event_text = str(data["event_types"].dropna().iloc[0]).replace("|", ", ")

    fig, axes = plt.subplots(4, 1, figsize=(7.25, 7.55), sharex=True, gridspec_kw={"hspace": 0.30})

    axes[0].plot(hours, data["pv_mw"], color=OKABE_ITO["black"], lw=1.8, label="Actual PV")
    axes[0].plot(hours, data["anchor_bid_mw"], color=OKABE_ITO["blue"], lw=1.4, ls="--", label="Rule-core anchor")
    axes[0].plot(hours, data["llm_hybrid_bid_mw"], color=OKABE_ITO["vermillion"], lw=1.8, label="LLM hybrid")
    axes[0].set_ylabel("PV or bid (MW)")
    pv_top = float(data[["pv_mw", "anchor_bid_mw", "llm_hybrid_bid_mw"]].max().max())
    axes[0].set_ylim(0.0, pv_top * 1.30)
    axes[0].set_title(f"{case_date}: {event_text}", loc="left", fontsize=9, pad=5)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    risk_text = (
        "Narrative features: "
        f"rule cloud/rain={float(data['wx_prior_cloud_score'].iloc[0]):.1f}/{float(data['wx_prior_rain_score'].iloc[0]):.1f}; "
        f"LLM irradiance/confidence={float(data['llm_prior_irradiance_reduction_risk'].iloc[0]):.1f}/{float(data['llm_prior_confidence'].iloc[0]):.2f}"
    )
    axes[0].text(1.0, 1.10, risk_text, transform=axes[0].transAxes, fontsize=7.0, ha="right", va="bottom", color="0.28", clip_on=False)

    axes[1].plot(hours, data["da_lmp"], color=OKABE_ITO["orange"], lw=1.6, label="Day-ahead price")
    axes[1].plot(hours, data["rt_lmp"], color=OKABE_ITO["sky_blue"], lw=1.6, label="Real-time price")
    axes[1].fill_between(
        hours,
        data["da_lmp"].astype(float),
        data["rt_lmp"].astype(float),
        color="0.80",
        alpha=0.35,
        label="DA-RT spread",
    )
    axes[1].set_ylabel("Price (USD/MWh)")
    price_min = float(data[["da_lmp", "rt_lmp"]].min().min())
    price_max = float(data[["da_lmp", "rt_lmp"]].max().max())
    price_span = max(price_max - price_min, 1.0)
    axes[1].set_ylim(price_min - 0.08 * price_span, price_max + 0.28 * price_span)
    axes[1].legend(frameon=False, ncol=3, loc="upper left")

    axes[2].plot(hours, data["anchor_abs_imbalance_mw"], color=OKABE_ITO["blue"], lw=1.4, ls="--", label="Rule-core anchor")
    axes[2].plot(hours, data["llm_abs_imbalance_mw"], color=OKABE_ITO["vermillion"], lw=1.8, label="LLM hybrid")
    imb_top = float(data[["anchor_abs_imbalance_mw", "llm_abs_imbalance_mw"]].max().max())
    axes[2].set_ylim(0.0, imb_top * 1.26)
    axes[2].set_ylabel("Absolute imbalance (MW)")
    axes[2].legend(frameon=False, ncol=2, loc="upper left")

    bar_colors = np.where(data["revenue_delta_usd"].to_numpy(dtype=float) >= 0.0, OKABE_ITO["bluish_green"], OKABE_ITO["vermillion"])
    axes[3].bar(hours, data["revenue_delta_usd"] / 1_000.0, color=bar_colors, alpha=0.84, label="Hourly value delta")
    axes[3].plot(
        hours,
        data["cumulative_revenue_delta_usd"] / 1_000.0,
        color=OKABE_ITO["black"],
        lw=1.5,
        marker="o",
        ms=3,
        label="Cumulative value delta",
    )
    axes[3].axhline(0.0, color="0.30", lw=0.8)
    axes[3].set_ylabel("Value delta\n(thousand USD)")
    axes[3].set_xlabel("Local hour")
    axes[3].legend(frameon=False, ncol=2, loc="upper left")
    final_delta = float(data["cumulative_revenue_delta_usd"].iloc[-1]) / 1_000.0
    lower = float(min((data["revenue_delta_usd"] / 1_000.0).min(), (data["cumulative_revenue_delta_usd"] / 1_000.0).min(), 0.0))
    upper = float(max((data["revenue_delta_usd"] / 1_000.0).max(), (data["cumulative_revenue_delta_usd"] / 1_000.0).max(), 0.0))
    y_span = max(upper - lower, 1.0)
    axes[3].set_ylim(lower - 0.10 * y_span, upper + 0.18 * y_span)
    axes[3].annotate(
        f"+{final_delta:.1f}k USD",
        xy=(float(hours[-1]), final_delta),
        xytext=(7, 0),
        textcoords="offset points",
        fontsize=7.3,
        ha="left",
        va="center",
        color="0.20",
        clip_on=False,
    )

    for idx, ax in enumerate(axes):
        ax.axvspan(6, 19, color=OKABE_ITO["yellow"], alpha=0.08, zorder=0)
        ax.text(-0.055, 1.02, chr(ord("a") + idx), transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
        ax.grid(True, alpha=0.18)
    axes[-1].set_xticks(np.arange(0, 24, 2))
    axes[-1].set_xlim(-0.5, 24.8)
    fig.subplots_adjust(top=0.925, left=0.14, right=0.985, bottom=0.075)
    save_figure(fig, "fig_case_2025_03_07")


def main() -> None:
    reset_outputs()
    apply_style()
    copy_pipeline_overview()
    write_tables()
    plot_forecast_slices()
    plot_forecast_signal()
    plot_value_cvar()
    plot_paired_seed_effects()
    plot_decision_evidence()
    plot_case_study()
    print(f"Wrote tables to {TABLES}")
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
