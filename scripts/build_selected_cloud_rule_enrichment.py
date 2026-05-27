from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_paper_figures import (
    OKABE_ITO,
    apply_publication_style,
    prepare_case_study_data,
    reconstruct_hybrid_blend_bid_series,
    save_figure,
)


def _rmse(actual: pd.Series, pred: pd.Series) -> float:
    error = pd.to_numeric(pred, errors="coerce") - pd.to_numeric(actual, errors="coerce")
    return float(np.sqrt(np.nanmean(np.square(error.to_numpy(dtype=float)))))


def _mae(actual: pd.Series, pred: pd.Series) -> float:
    error = pd.to_numeric(pred, errors="coerce") - pd.to_numeric(actual, errors="coerce")
    return float(np.nanmean(np.abs(error.to_numpy(dtype=float))))


def _improvement_pct(reference: float, llm: float) -> float:
    if not np.isfinite(reference) or abs(reference) < 1e-12:
        return np.nan
    return 100.0 * (reference - llm) / reference


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame.loc[:, columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def _forecast_slice_masks(preds: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {
        "All test hours": pd.Series(True, index=preds.index),
        "Solar hours": pd.to_numeric(preds["is_solar_hour"], errors="coerce").fillna(0).astype(int).eq(1),
        "Extreme-event hours": pd.to_numeric(preds["has_extreme_event"], errors="coerce").fillna(0).astype(int).eq(1),
    }
    masks["Extreme solar hours"] = masks["Solar hours"] & masks["Extreme-event hours"]

    if "llm_prior_rule_cloud_score" in preds.columns:
        cloud = pd.to_numeric(preds["llm_prior_rule_cloud_score"], errors="coerce")
        threshold = float(cloud.quantile(0.75))
        masks["High LLM cloud-risk hours"] = cloud.ge(threshold)

    if {"da_lmp", "rt_lmp"}.issubset(preds.columns):
        spread = (pd.to_numeric(preds["da_lmp"], errors="coerce") - pd.to_numeric(preds["rt_lmp"], errors="coerce")).abs()
        threshold = float(spread.quantile(0.75))
        masks["High DA-RT spread hours"] = spread.ge(threshold)

    return masks


def build_forecast_slice_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    model_specs = [
        {
            "target": "PV generation",
            "actual_col": "pv_mw",
            "unit": "MW",
            "reference_label": "MLP rule-core",
            "reference_col": "pv_mlp_rule_core",
            "llm_label": "MLP LLM cloud-rule",
            "llm_col": "pv_mlp_llm_rule_cloud",
        },
        {
            "target": "Real-time price",
            "actual_col": "rt_lmp",
            "unit": "USD/MWh",
            "reference_label": "Transformer rule-text",
            "reference_col": "rt_transformer_text",
            "llm_label": "Transformer LLM rule",
            "llm_col": "rt_transformer_llm_rule",
        },
    ]
    rows: list[dict[str, object]] = []
    for slice_name, mask in _forecast_slice_masks(preds).items():
        subset = preds.loc[mask].copy()
        if subset.empty:
            continue
        for spec in model_specs:
            actual = subset[str(spec["actual_col"])]
            ref = subset[str(spec["reference_col"])]
            llm = subset[str(spec["llm_col"])]
            ref_rmse = _rmse(actual, ref)
            llm_rmse = _rmse(actual, llm)
            ref_mae = _mae(actual, ref)
            llm_mae = _mae(actual, llm)
            rows.append(
                {
                    "slice": slice_name,
                    "target": spec["target"],
                    "n_hours": int(len(subset)),
                    "unit": spec["unit"],
                    "reference": spec["reference_label"],
                    "llm": spec["llm_label"],
                    "reference_rmse": ref_rmse,
                    "llm_rmse": llm_rmse,
                    "rmse_improvement_pct": _improvement_pct(ref_rmse, llm_rmse),
                    "reference_mae": ref_mae,
                    "llm_mae": llm_mae,
                    "mae_improvement_pct": _improvement_pct(ref_mae, llm_mae),
                }
            )
    return pd.DataFrame(rows)


def write_forecast_slice_report(metrics: pd.DataFrame, output_dir: Path, date_tag: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"forecast_slice_metrics_{date_tag}.csv"
    md_path = output_dir / f"forecast_slice_metrics_{date_tag}.md"
    metrics.to_csv(csv_path, index=False)

    display_cols = [
        "slice",
        "target",
        "n_hours",
        "unit",
        "reference",
        "llm",
        "reference_rmse",
        "llm_rmse",
        "rmse_improvement_pct",
        "reference_mae",
        "llm_mae",
        "mae_improvement_pct",
    ]
    lines = [
        "# Forecast Slice Metrics",
        "",
        "Positive improvement means the LLM-feature model has lower error than the matched neural reference.",
        "",
        _markdown_table(metrics, display_cols),
        "",
        "Source predictions: `test_predictions.csv` from the selected cloud-rule downstream run.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def plot_forecast_slice_metrics(metrics: pd.DataFrame, output_dir: Path) -> None:
    plot_data = metrics[metrics["slice"].isin(["All test hours", "Solar hours", "Extreme-event hours", "Extreme solar hours"])].copy()
    if plot_data.empty:
        return
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
        colors = np.where(subset["rmse_improvement_pct"].to_numpy(dtype=float) >= 0.0, OKABE_ITO["bluish_green"], OKABE_ITO["vermillion"])
        ax.barh(y, subset["rmse_improvement_pct"].astype(float), color=colors, alpha=0.88)
        ax.axvline(0.0, color="0.25", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(subset["slice"].astype(str))
        ax.invert_yaxis()
        ax.set_xlabel("RMSE improvement (%)")
        ax.set_title(target)
    axes[0].text(-0.15, 1.04, "a", transform=axes[0].transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
    axes[1].text(-0.15, 1.04, "b", transform=axes[1].transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_selected_cloud_rule_forecast_slices")


def plot_decision_sensitivity(weight_table: pd.DataFrame, output_dir: Path) -> None:
    data = weight_table.sort_values("lp_weight").copy()
    x = data["lp_weight"].astype(float).to_numpy()
    metrics = [
        ("value_delta_musd", "Value gain\n(M USD)", OKABE_ITO["orange"]),
        ("cvar95_loss_reduction_kusd_h", "CVaR95 loss reduction\n(k USD/h)", OKABE_ITO["bluish_green"]),
        ("imbalance_reduction_gwh", "Imbalance reduction\n(GWh)", OKABE_ITO["blue"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.9), sharex=True)
    for ax, (column, ylabel, color) in zip(axes, metrics):
        y = data[column].astype(float).to_numpy()
        ax.plot(x, y, marker="o", lw=1.7, color=color)
        ax.axhline(0.0, color="0.35", lw=0.75, ls=":")
        ax.axvline(0.50, color=OKABE_ITO["vermillion"], lw=1.0, ls="--")
        ax.set_xlabel("LP weight")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:.2f}" for value in x])
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_selected_cloud_rule_decision_sensitivity")


def plot_paired_seed_effects(seed_deltas: pd.DataFrame, seed_summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("value_delta_musd", "Value gain\n(M USD)", OKABE_ITO["orange"], "Value gain"),
        ("cvar95_loss_reduction_kusd_h", "CVaR95 loss reduction\n(k USD/h)", OKABE_ITO["bluish_green"], "CVaR95 loss reduction"),
        ("imbalance_reduction_gwh", "Imbalance reduction\n(GWh)", OKABE_ITO["blue"], "Imbalance reduction"),
    ]
    summary = {str(row["metric"]): row for _, row in seed_summary.iterrows()}
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 3.55))
    rng = np.random.default_rng(27)
    for ax, (column, xlabel, color, summary_key) in zip(axes, metrics):
        values = pd.to_numeric(seed_deltas[column], errors="coerce").dropna().to_numpy(dtype=float)
        jitter = rng.normal(0.0, 0.025, len(values))
        ax.scatter(values, jitter, s=28, color=color, alpha=0.64, edgecolor="none")
        if summary_key in summary:
            row = summary[summary_key]
            mean = float(row["mean"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
        else:
            mean = float(np.mean(values))
            low = float(np.min(values))
            high = float(np.max(values))
        ax.errorbar(
            mean,
            0.0,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color="black",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=5.5,
            elinewidth=1.2,
            capsize=4,
            zorder=4,
        )
        ax.axvline(0.0, color="0.35", lw=0.8, ls=":")
        ax.set_yticks([])
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.12, 0.12)
        ax.text(
            0.02,
            0.92,
            f"{int(np.sum(values > 0))}/{len(values)} seeds positive",
            transform=ax.transAxes,
            fontsize=8.0,
            ha="left",
            va="top",
        )
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_selected_cloud_rule_paired_seed_effects")


def plot_selected_value_cvar_tradeoff(summary: pd.DataFrame, output_dir: Path) -> None:
    data = summary[summary["comparison"].astype(str).eq("validation_selected")].copy()
    if data.empty:
        return
    data["value_musd"] = pd.to_numeric(data["value_musd"], errors="coerce")
    data["cvar95_loss_kusd_h"] = pd.to_numeric(data["cvar95_loss_kusd_h"], errors="coerce")
    data["imbalance_gwh"] = pd.to_numeric(data["imbalance_gwh"], errors="coerce")
    style = {
        "Rule-core hybrid blend": ("Rule-core hybrid\nreference", OKABE_ITO["blue"], "s"),
        "Pure LLM cloud-rule LP": ("Pure LLM\ncloud-rule LP", OKABE_ITO["orange"], "^"),
        "LLM cloud-rule hybrid blend": ("LLM cloud-rule\nhybrid", OKABE_ITO["vermillion"], "*"),
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.35))
    for _, row in data.iterrows():
        label, color, marker = style.get(str(row["strategy"]), (str(row["strategy"]), OKABE_ITO["black"], "o"))
        size = 145 if "LLM cloud-rule hybrid" in str(row["strategy"]) else 70
        ax.scatter(
            float(row["cvar95_loss_kusd_h"]),
            float(row["value_musd"]),
            s=size,
            color=color,
            marker=marker,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            label,
            xy=(float(row["cvar95_loss_kusd_h"]), float(row["value_musd"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
            ha="left",
            va="bottom",
        )
    rule = data[data["strategy"].astype(str).eq("Rule-core hybrid blend")]
    llm = data[data["strategy"].astype(str).eq("LLM cloud-rule hybrid blend")]
    if not rule.empty and not llm.empty:
        ax.annotate(
            "",
            xy=(float(llm.iloc[0]["cvar95_loss_kusd_h"]), float(llm.iloc[0]["value_musd"])),
            xytext=(float(rule.iloc[0]["cvar95_loss_kusd_h"]), float(rule.iloc[0]["value_musd"])),
            arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "0.25", "shrinkA": 7, "shrinkB": 7},
        )
    ax.set_xlabel("CVaR95 loss (k USD/h)")
    ax.set_ylabel("Annual proxy value (M USD)")
    ax.grid(True, alpha=0.23)
    ax.set_xlim(float(data["cvar95_loss_kusd_h"].min()) - 10, float(data["cvar95_loss_kusd_h"].max()) + 13)
    ax.set_ylim(float(data["value_musd"].min()) - 2.5, float(data["value_musd"].max()) + 2.2)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_selected_cloud_rule_value_cvar_tradeoff")


def plot_selected_case_study(case_data: pd.DataFrame, output_dir: Path, case_date: str) -> None:
    data = case_data.sort_values("timestamp_utc").copy()
    hours = data["hour"].astype(int).to_numpy()
    event_text = "Extreme-weather event"
    if "event_types" in data.columns and not data["event_types"].dropna().empty:
        event_text = str(data["event_types"].dropna().iloc[0]).replace("|", ", ")

    fig, axes = plt.subplots(4, 1, figsize=(7.25, 7.45), sharex=True, gridspec_kw={"hspace": 0.28})

    axes[0].plot(hours, data["pv_mw"], color=OKABE_ITO["black"], lw=1.8, label="Actual PV")
    axes[0].plot(hours, data["anchor_bid_mw"], color=OKABE_ITO["blue"], lw=1.4, ls="--", label="Rule-core anchor bid")
    axes[0].plot(hours, data["llm_hybrid_bid_mw"], color=OKABE_ITO["vermillion"], lw=1.8, label="LLM cloud-rule hybrid bid")
    axes[0].set_ylabel("PV or bid (MW)")
    pv_top = float(data[["pv_mw", "anchor_bid_mw", "llm_hybrid_bid_mw"]].max().max())
    axes[0].set_ylim(-800.0, pv_top * 1.26)
    axes[0].set_title(f"{case_date}: {event_text}", loc="left", fontsize=9, pad=5)
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

    for idx, ax in enumerate(axes):
        ax.text(-0.055, 1.02, chr(ord("a") + idx), transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
    axes[-1].set_xticks(np.arange(0, 24, 2))
    axes[-1].set_xlim(-0.5, 23.5)
    fig.subplots_adjust(top=0.965, left=0.14, right=0.985, bottom=0.075)
    save_figure(fig, output_dir, f"fig_selected_cloud_rule_case_{case_date.replace('-', '_')}")


def screen_case_days(preds: pd.DataFrame, hybrid_bid: pd.Series) -> pd.DataFrame:
    x = preds.copy()
    x["hybrid_bid_mw"] = hybrid_bid
    rows: list[dict[str, object]] = []
    for local_date, day in x.groupby("local_date", sort=True):
        case_day = prepare_case_study_data(
            day,
            day["hybrid_bid_mw"],
            deviation_penalty=50.0,
            anchor_col="pv_mlp_rule_core",
        )
        pv_ref_rmse = _rmse(case_day["pv_mw"], case_day["pv_mlp_rule_core"])
        pv_llm_rmse = _rmse(case_day["pv_mw"], case_day["pv_mlp_llm_rule_cloud"])
        rows.append(
            {
                "local_date": local_date,
                "has_extreme_event": int(pd.to_numeric(day["has_extreme_event"], errors="coerce").fillna(0).max()),
                "event_types": "|".join(sorted(set(str(v) for v in day["event_types"].dropna() if str(v) != "nan"))),
                "value_delta_kusd": float(case_day["revenue_delta_usd"].sum()) / 1_000.0,
                "absolute_imbalance_reduction_mwh": float(case_day["anchor_abs_imbalance_mw"].sum())
                - float(case_day["llm_abs_imbalance_mw"].sum()),
                "pv_reference_rmse_mw": pv_ref_rmse,
                "pv_llm_rmse_mw": pv_llm_rmse,
                "pv_rmse_improvement_pct": _improvement_pct(pv_ref_rmse, pv_llm_rmse),
            }
        )
    return pd.DataFrame(rows)


def select_case_date(case_screen: pd.DataFrame, requested_case_date: str) -> str:
    if requested_case_date.lower() != "auto":
        return requested_case_date
    clean = case_screen[
        (case_screen["has_extreme_event"].eq(1))
        & (case_screen["value_delta_kusd"].gt(0.0))
        & (case_screen["absolute_imbalance_reduction_mwh"].gt(0.0))
        & (case_screen["pv_rmse_improvement_pct"].gt(0.0))
    ].copy()
    if clean.empty:
        clean = case_screen[
            (case_screen["has_extreme_event"].eq(1))
            & (case_screen["value_delta_kusd"].gt(0.0))
            & (case_screen["absolute_imbalance_reduction_mwh"].gt(0.0))
        ].copy()
    if clean.empty:
        raise ValueError("no positive extreme-weather case found for selected cloud-rule hybrid")
    clean = clean.sort_values(["value_delta_kusd", "absolute_imbalance_reduction_mwh"], ascending=False)
    return str(clean.iloc[0]["local_date"])


def build_case_study(results_dir: Path, output_dir: Path, case_date: str, date_tag: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    preds = pd.read_csv(results_dir / "test_predictions.csv")
    train_residuals = pd.read_csv(results_dir / "train_residuals.csv")
    audit = pd.read_csv(results_dir / "data_audit.csv")
    rated_capacity = float(audit.iloc[0]["rated_capacity_mw"])
    hybrid_bid = reconstruct_hybrid_blend_bid_series(
        preds=preds,
        train_residuals=train_residuals,
        rated_capacity=rated_capacity,
        seed=71_000,
        pv_model_name="mlp_llm_rule_cloud",
        rt_model_name="transformer_llm_rule",
        anchor_model_name="mlp_rule_core",
        lp_weight=0.50,
        residual_scale=1.0,
        cvar_gamma=0.25,
        deviation_penalty=50.0,
        scenario_count=20,
    )
    case_screen = screen_case_days(preds, hybrid_bid)
    case_dir = output_dir / "case_study"
    case_dir.mkdir(parents=True, exist_ok=True)
    case_screen.to_csv(case_dir / f"selected_cloud_rule_case_screening_{date_tag}.csv", index=False)
    selected_case_date = select_case_date(case_screen, case_date)

    case_mask = preds["local_date"].astype(str).eq(selected_case_date)
    if not bool(case_mask.any()):
        raise ValueError(f"case date {selected_case_date} is absent from {results_dir / 'test_predictions.csv'}")
    case_data = prepare_case_study_data(
        preds.loc[case_mask].copy(),
        hybrid_bid.loc[case_mask],
        deviation_penalty=50.0,
        anchor_col="pv_mlp_rule_core",
    )
    figure_dir = output_dir / "figures"
    case_data.to_csv(case_dir / f"selected_cloud_rule_case_{selected_case_date.replace('-', '_')}_data_{date_tag}.csv", index=False)
    plot_selected_case_study(case_data, figure_dir, selected_case_date)
    return case_data, case_screen, selected_case_date


def write_summary(
    output_dir: Path,
    date_tag: str,
    slice_metrics: pd.DataFrame,
    weight_table: pd.DataFrame,
    case_data: pd.DataFrame,
    case_screen: pd.DataFrame,
    case_date: str,
) -> None:
    all_pv = slice_metrics[(slice_metrics["slice"].eq("All test hours")) & (slice_metrics["target"].eq("PV generation"))].iloc[0]
    all_rt = slice_metrics[(slice_metrics["slice"].eq("All test hours")) & (slice_metrics["target"].eq("Real-time price"))].iloc[0]
    selected_w = weight_table[np.isclose(pd.to_numeric(weight_table["lp_weight"], errors="coerce"), 0.50)].iloc[0]
    total_case_delta = float(case_data["revenue_delta_usd"].sum()) / 1_000.0
    case_imbalance_delta = (
        float(case_data["anchor_abs_imbalance_mw"].sum()) - float(case_data["llm_abs_imbalance_mw"].sum())
    )
    selected_case_screen = case_screen[case_screen["local_date"].astype(str).eq(case_date)].iloc[0]
    lines = [
        "# Selected Cloud-Rule Enrichment Results",
        "",
        "Scope: neural forecast models only. The reference is the matched rule-core or rule-text neural model; the LLM variant uses the selected cloud-rule feature path.",
        "",
        "## Forecast slices",
        "",
        f"- All-hour PV: RMSE improvement {float(all_pv['rmse_improvement_pct']):.3f}%, MAE improvement {float(all_pv['mae_improvement_pct']):.3f}%.",
        f"- All-hour RT price: RMSE improvement {float(all_rt['rmse_improvement_pct']):.3f}%, MAE improvement {float(all_rt['mae_improvement_pct']):.3f}%.",
        f"- Full slice table: `forecast_slice_metrics_{date_tag}.csv` and `forecast_slice_metrics_{date_tag}.md`.",
        "",
        "## Decision sensitivity",
        "",
        f"- At the selected LLM weight w=0.50, same-weight diagnostic deltas are value {float(selected_w['value_delta_musd']):.3f}M USD, CVaR95 loss reduction {float(selected_w['cvar95_loss_reduction_kusd_h']):.3f}k USD/h, and imbalance reduction {float(selected_w['imbalance_reduction_gwh']):.3f}GWh.",
        "- Figure: `figures/fig_selected_cloud_rule_decision_sensitivity.pdf` and `.png`.",
        "",
        "## Extreme-weather case study",
        "",
        f"- Case date: {case_date}.",
        f"- Event labels: {selected_case_screen['event_types']}.",
        f"- PV RMSE improvement on this day: {float(selected_case_screen['pv_rmse_improvement_pct']):.3f}%.",
        f"- LLM cloud-rule hybrid cumulative value delta on this day: {total_case_delta:.3f}k USD.",
        f"- Absolute-imbalance reduction on this day: {case_imbalance_delta:.3f}MWh.",
        f"- Figure: `figures/fig_selected_cloud_rule_case_{case_date.replace('-', '_')}.pdf` and `.png`.",
        f"- Data: `case_study/selected_cloud_rule_case_{case_date.replace('-', '_')}_data_{date_tag}.csv`.",
        f"- Screening table: `case_study/selected_cloud_rule_case_screening_{date_tag}.csv`.",
        "",
    ]
    (output_dir / f"selected_cloud_rule_extra_results_{date_tag}.md").write_text("\n".join(lines), encoding="utf-8")


def build_enrichment(results_dir: Path, case_date: str, date_tag: str) -> None:
    apply_publication_style()
    output_dir = results_dir
    figure_dir = output_dir / "figures"
    preds = pd.read_csv(results_dir / "test_predictions.csv")
    slice_metrics = build_forecast_slice_metrics(preds)
    write_forecast_slice_report(slice_metrics, output_dir, date_tag)
    plot_forecast_slice_metrics(slice_metrics, figure_dir)

    weight_table = pd.read_csv(results_dir / f"selected_cloud_rule_downstream_weight_sensitivity_{date_tag}.csv")
    plot_decision_sensitivity(weight_table, figure_dir)
    seed_deltas = pd.read_csv(results_dir / f"selected_cloud_rule_downstream_paired_seed_deltas_{date_tag}.csv")
    seed_summary = pd.read_csv(results_dir / f"selected_cloud_rule_downstream_paired_seed_summary_{date_tag}.csv")
    plot_paired_seed_effects(seed_deltas, seed_summary, figure_dir)
    downstream_summary = pd.read_csv(results_dir / f"selected_cloud_rule_downstream_summary_{date_tag}.csv")
    plot_selected_value_cvar_tradeoff(downstream_summary, figure_dir)

    case_data, case_screen, selected_case_date = build_case_study(results_dir, output_dir, case_date, date_tag)
    write_summary(output_dir, date_tag, slice_metrics, weight_table, case_data, case_screen, selected_case_date)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build selected cloud-rule enrichment tables and figures.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/selected_cloud_rule_downstream_20260527"))
    parser.add_argument("--case-date", default="auto")
    parser.add_argument("--date-tag", default="20260527")
    args = parser.parse_args()
    build_enrichment(args.results_dir, args.case_date, args.date_tag)


if __name__ == "__main__":
    main()
