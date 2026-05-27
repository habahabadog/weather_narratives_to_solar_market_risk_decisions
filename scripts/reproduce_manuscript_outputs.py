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


def metric_value(matrix: pd.DataFrame, target: str, model: str, metric: str, column: str) -> float:
    row = matrix[
        matrix["target"].eq(target)
        & matrix["model"].eq(model)
        & matrix["metric"].eq(metric)
    ]
    if row.empty:
        raise ValueError(f"Missing metric row: {target} {model} {metric}")
    return float(row.iloc[0][column])


def build_main_forecast_table(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "Target": "PV",
            "Model": "MLP",
            "Feature role": "No text",
            "RMSE": metric_value(matrix, "PV", "MLP", "RMSE", "no_text_test"),
            "MAE": metric_value(matrix, "PV", "MLP", "MAE", "no_text_test"),
        },
        {
            "Target": "PV",
            "Model": "MLP",
            "Feature role": "Rule-core text features",
            "RMSE": metric_value(matrix, "PV", "MLP", "RMSE", "rule_test"),
            "MAE": metric_value(matrix, "PV", "MLP", "MAE", "rule_test"),
        },
        {
            "Target": "PV",
            "Model": "MLP",
            "Feature role": "Selected LLM cloud-rule feature",
            "RMSE": metric_value(matrix, "PV", "MLP", "RMSE", "llm_test"),
            "MAE": metric_value(matrix, "PV", "MLP", "MAE", "llm_test"),
        },
        {
            "Target": "RT price",
            "Model": "Transformer",
            "Feature role": "No text",
            "RMSE": metric_value(matrix, "RT price", "Transformer", "RMSE", "no_text_test"),
            "MAE": metric_value(matrix, "RT price", "Transformer", "MAE", "no_text_test"),
        },
        {
            "Target": "RT price",
            "Model": "Transformer",
            "Feature role": "Rule-text weather scores",
            "RMSE": metric_value(matrix, "RT price", "Transformer", "RMSE", "rule_test"),
            "MAE": metric_value(matrix, "RT price", "Transformer", "MAE", "rule_test"),
        },
        {
            "Target": "RT price",
            "Model": "Transformer",
            "Feature role": "LLM rule-equivalent weather scores",
            "RMSE": metric_value(matrix, "RT price", "Transformer", "RMSE", "llm_test"),
            "MAE": metric_value(matrix, "RT price", "Transformer", "MAE", "llm_test"),
        },
    ]
    return pd.DataFrame(rows).round({"RMSE": 2, "MAE": 2})


def build_decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary[summary["comparison"].eq("validation_selected")].copy()
    labels = {
        "Rule-core hybrid blend": "Rule-core hybrid reference",
        "Pure LLM cloud-rule LP": "Pure LLM cloud-rule LP",
        "LLM cloud-rule hybrid blend": "LLM cloud-rule hybrid",
    }
    data["Strategy"] = data["strategy"].map(labels).fillna(data["strategy"])
    out = data[
        [
            "Strategy",
            "lp_weight",
            "value_musd",
            "cvar95_loss_kusd_h",
            "imbalance_gwh",
            "value_delta_musd_vs_rule",
            "cvar_delta_kusd_h_vs_rule",
            "imbalance_delta_gwh_vs_rule",
        ]
    ].rename(
        columns={
            "lp_weight": "w",
            "value_musd": "Value (M USD)",
            "cvar95_loss_kusd_h": "CVaR95 loss (k USD/h)",
            "imbalance_gwh": "Imbalance (GWh)",
            "value_delta_musd_vs_rule": "Value delta (M USD)",
            "cvar_delta_kusd_h_vs_rule": "CVaR delta (k USD/h)",
            "imbalance_delta_gwh_vs_rule": "Imbalance delta (GWh)",
        }
    )
    return out.round(
        {
            "w": 2,
            "Value (M USD)": 2,
            "CVaR95 loss (k USD/h)": 2,
            "Imbalance (GWh)": 1,
            "Value delta (M USD)": 2,
            "CVaR delta (k USD/h)": 2,
            "Imbalance delta (GWh)": 1,
        }
    )


def build_case_screening_table(case_screen: pd.DataFrame) -> pd.DataFrame:
    data = case_screen[
        case_screen["has_extreme_event"].eq(1)
        & case_screen["value_delta_kusd"].gt(0)
        & case_screen["absolute_imbalance_reduction_mwh"].gt(0)
        & case_screen["pv_rmse_improvement_pct"].gt(0)
    ].copy()
    data = data.sort_values("value_delta_kusd", ascending=False).head(5)
    out = data[
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
    matrix = pd.read_csv(DATA / "forecast_matrix.csv")
    summary = pd.read_csv(DATA / "decision_summary.csv")
    seed_summary = pd.read_csv(DATA / "paired_seed_summary.csv")
    slices = pd.read_csv(DATA / "forecast_slice_metrics.csv")
    case_screen = pd.read_csv(DATA / "case_screening.csv")

    tables = {
        "main_forecast_table": build_main_forecast_table(matrix),
        "main_decision_table": build_decision_table(summary),
        "paired_seed_summary": seed_summary.round(
            {"mean": 2, "ci95_low": 2, "ci95_high": 2, "paired_t_p_value": 4}
        ),
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
        "case_screening_top5": build_case_screening_table(case_screen),
    }

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


def plot_value_cvar() -> None:
    summary = pd.read_csv(DATA / "decision_summary.csv")
    data = summary[summary["comparison"].eq("validation_selected")].copy()
    style = {
        "Rule-core hybrid blend": ("Rule-core hybrid\nreference", OKABE_ITO["blue"], "s"),
        "Pure LLM cloud-rule LP": ("Pure LLM\ncloud-rule LP", OKABE_ITO["orange"], "^"),
        "LLM cloud-rule hybrid blend": ("LLM cloud-rule\nhybrid", OKABE_ITO["vermillion"], "*"),
    }
    fig, ax = plt.subplots(figsize=(5.55, 3.55))
    rule = data[data["strategy"].eq("Rule-core hybrid blend")].iloc[0]
    llm = data[data["strategy"].eq("LLM cloud-rule hybrid blend")].iloc[0]
    ax.axvline(float(rule["cvar95_loss_kusd_h"]), color="0.55", lw=0.85, ls="--", alpha=0.7, zorder=0)
    ax.axhline(float(rule["value_musd"]), color="0.55", lw=0.85, ls="--", alpha=0.7, zorder=0)
    for _, row in data.iterrows():
        label, color, marker = style[str(row["strategy"])]
        size = 145 if "LLM cloud-rule hybrid" in str(row["strategy"]) else 70
        x = float(row["cvar95_loss_kusd_h"])
        y = float(row["value_musd"])
        ax.scatter(x, y, s=size, color=color, marker=marker, edgecolor="black", linewidth=0.8, zorder=3)
        offsets = {
            "Rule-core hybrid blend": (7, 7, "left", "bottom"),
            "Pure LLM cloud-rule LP": (8, 0, "left", "center"),
            "LLM cloud-rule hybrid blend": (8, 8, "left", "bottom"),
        }
        dx, dy, ha, va = offsets[str(row["strategy"])]
        ax.annotate(label, xy=(x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.4, ha=ha, va=va)

    ax.annotate(
        "",
        xy=(float(llm["cvar95_loss_kusd_h"]), float(llm["value_musd"])),
        xytext=(float(rule["cvar95_loss_kusd_h"]), float(rule["value_musd"])),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "0.25", "shrinkA": 7, "shrinkB": 7},
    )
    value_gain = float(llm["value_musd"]) - float(rule["value_musd"])
    cvar_gain = float(rule["cvar95_loss_kusd_h"]) - float(llm["cvar95_loss_kusd_h"])
    ax.text(
        0.03,
        0.06,
        f"Hybrid gain vs reference\n+{value_gain:.2f} M USD\n+{cvar_gain:.2f} k USD/h CVaR reduction",
        transform=ax.transAxes,
        fontsize=7.2,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.82", "linewidth": 0.6},
    )
    ax.set_xlabel("CVaR95 loss (k USD/h, lower is better)")
    ax.set_ylabel("Annual proxy value (M USD, higher is better)")
    ax.grid(True, alpha=0.18)
    ax.set_xlim(float(data["cvar95_loss_kusd_h"].min()) - 10, float(data["cvar95_loss_kusd_h"].max()) + 13)
    ax.set_ylim(float(data["value_musd"].min()) - 2.5, float(data["value_musd"].max()) + 2.2)
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
        ax.text(0.98, 0.93, f"{int(np.sum(values > 0))}/{len(values)} positive", transform=ax.transAxes, fontsize=7.4, ha="right", va="top", color="0.25")
    fig.tight_layout()
    save_figure(fig, "fig_paired_seed_effects")


def plot_case_study() -> None:
    data = pd.read_csv(DATA / "case_2024_03_13.csv").sort_values("timestamp_utc").copy()
    hours = data["hour"].astype(int).to_numpy()
    event_text = str(data["event_types"].dropna().iloc[0]).replace("|", ", ")

    fig, axes = plt.subplots(4, 1, figsize=(7.25, 7.35), sharex=True, gridspec_kw={"hspace": 0.24})

    axes[0].plot(hours, data["pv_mw"], color=OKABE_ITO["black"], lw=1.8, label="Actual PV")
    axes[0].plot(hours, data["anchor_bid_mw"], color=OKABE_ITO["blue"], lw=1.4, ls="--", label="Rule-core anchor bid")
    axes[0].plot(hours, data["llm_hybrid_bid_mw"], color=OKABE_ITO["vermillion"], lw=1.8, label="LLM cloud-rule hybrid bid")
    axes[0].set_ylabel("PV or bid (MW)")
    pv_top = float(data[["pv_mw", "anchor_bid_mw", "llm_hybrid_bid_mw"]].max().max())
    axes[0].set_ylim(-800.0, pv_top * 1.26)
    axes[0].set_title(f"2024-03-13: {event_text}", loc="left", fontsize=9, pad=5)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")

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

    axes[2].plot(hours, data["anchor_abs_imbalance_mw"], color=OKABE_ITO["blue"], lw=1.4, ls="--", label="Rule-core anchor imbalance")
    axes[2].plot(hours, data["llm_abs_imbalance_mw"], color=OKABE_ITO["vermillion"], lw=1.8, label="LLM cloud-rule hybrid imbalance")
    imb_top = float(data[["anchor_abs_imbalance_mw", "llm_abs_imbalance_mw"]].max().max())
    axes[2].set_ylim(-120.0, imb_top * 1.24)
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
    axes[3].annotate(
        f"+{final_delta:.1f}k USD",
        xy=(float(hours[-1]), final_delta),
        xytext=(-42, 10),
        textcoords="offset points",
        fontsize=7.5,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.25"},
        ha="right",
        va="bottom",
    )

    for idx, ax in enumerate(axes):
        ax.axvspan(6, 19, color=OKABE_ITO["yellow"], alpha=0.08, zorder=0)
        ax.text(-0.055, 1.02, chr(ord("a") + idx), transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
        ax.grid(True, alpha=0.18)
    axes[-1].set_xticks(np.arange(0, 24, 2))
    axes[-1].set_xlim(-0.5, 23.5)
    fig.subplots_adjust(top=0.965, left=0.14, right=0.985, bottom=0.075)
    save_figure(fig, "fig_case_2024_03_13")


def main() -> None:
    apply_style()
    copy_pipeline_overview()
    write_tables()
    plot_forecast_slices()
    plot_value_cvar()
    plot_paired_seed_effects()
    plot_case_study()
    print(f"Wrote tables to {TABLES}")
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
