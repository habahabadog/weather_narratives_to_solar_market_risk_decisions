from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_baselines import (
    evaluate_hybrid_blended_scenario_seed_robustness,
    evaluate_validation_selected_hybrid_blend,
    fit_hybrid_blend_split_predictions,
)


SELECTED_WEIGHT = 0.25


def _resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else ROOT / path


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _row_for_weight(summary: pd.DataFrame, lp_weight: float) -> pd.Series:
    weights = pd.to_numeric(summary["lp_weight"], errors="coerce")
    rows = summary[np.isclose(weights, lp_weight)]
    if rows.empty:
        raise ValueError(f"summary has no row for lp_weight={lp_weight:.2f}")
    return rows.iloc[0]


def _metric_row(
    strategy: str,
    pv_scenario: str,
    rt_scenario: str,
    anchor: str,
    row: pd.Series,
) -> dict[str, float | int | str]:
    return {
        "strategy": strategy,
        "pv_scenario": pv_scenario,
        "rt_scenario": rt_scenario,
        "anchor": anchor,
        "lp_weight": float(row["lp_weight"]),
        "seed_count": int(row.get("seed_count", 0)),
        "value_musd": float(row["total_revenue_mean"]) / 1_000_000.0,
        "cvar95_loss_kusd_h": float(row["cvar_95_loss_mean"]) / 1_000.0,
        "imbalance_gwh": float(row["imbalance_mwh_proxy_mean"]) / 1_000.0,
    }


def build_decision_table(no_text_summary: pd.DataFrame, llm_summary: pd.DataFrame) -> pd.DataFrame:
    no_text_anchor = _row_for_weight(no_text_summary, 0.0)
    no_text_hybrid = _row_for_weight(no_text_summary, SELECTED_WEIGHT)
    llm_hybrid = _row_for_weight(llm_summary, SELECTED_WEIGHT)
    pure_no_text = _row_for_weight(no_text_summary, 1.0)
    pure_llm = _row_for_weight(llm_summary, 1.0)
    return pd.DataFrame(
        [
            _metric_row("MLP no-text deterministic anchor", "--", "--", "MLP no-text", no_text_anchor),
            _metric_row(
                "No-text LP-anchor hybrid",
                "MLP no-text",
                "Transformer no-text",
                "MLP no-text",
                no_text_hybrid,
            ),
            _metric_row(
                "LLM cloud-rule LP-anchor hybrid",
                "MLP LLM cloud-rule",
                "Transformer LLM-rule",
                "MLP no-text",
                llm_hybrid,
            ),
            _metric_row("Pure no-text LP", "MLP no-text", "Transformer no-text", "none", pure_no_text),
            _metric_row("Pure LLM cloud-rule LP", "MLP LLM cloud-rule", "Transformer LLM-rule", "none", pure_llm),
        ]
    )


def build_frontier_points(no_text_summary: pd.DataFrame, llm_summary: pd.DataFrame) -> pd.DataFrame:
    def convert(frame: pd.DataFrame, path_label: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "lp_weight": pd.to_numeric(frame["lp_weight"], errors="coerce"),
                "value_musd": pd.to_numeric(frame["total_revenue_mean"], errors="coerce") / 1_000_000.0,
                "cvar_kusd_h": pd.to_numeric(frame["cvar_95_loss_mean"], errors="coerce") / 1_000.0,
                "imbalance_gwh": pd.to_numeric(frame["imbalance_mwh_proxy_mean"], errors="coerce") / 1_000.0,
                "path": path_label,
            }
        ).sort_values("lp_weight")

    return pd.concat(
        [
            convert(no_text_summary, "No-text LP-anchor path"),
            convert(llm_summary, "LLM cloud-rule LP-anchor path"),
        ],
        ignore_index=True,
    )


def build_paired_seed_deltas(no_text_seed: pd.DataFrame, llm_seed: pd.DataFrame, weight: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    no_text = no_text_seed[np.isclose(pd.to_numeric(no_text_seed["lp_weight"], errors="coerce"), weight)].copy()
    llm = llm_seed[np.isclose(pd.to_numeric(llm_seed["lp_weight"], errors="coerce"), weight)].copy()
    merged = no_text.merge(llm, on="seed", suffixes=("_no_text", "_llm"))
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    deltas = pd.DataFrame(
        {
            "seed": merged["seed"],
            "value_delta_musd": (merged["total_revenue_llm"] - merged["total_revenue_no_text"]) / 1_000_000.0,
            "cvar95_loss_reduction_kusd_h": (merged["cvar_95_loss_no_text"] - merged["cvar_95_loss_llm"]) / 1_000.0,
            "imbalance_reduction_gwh": (
                merged["imbalance_mwh_proxy_no_text"] - merged["imbalance_mwh_proxy_llm"]
            )
            / 1_000.0,
        }
    ).sort_values("seed")
    rows: list[dict[str, float | int | str]] = []
    for metric, column, unit in [
        ("Value gain", "value_delta_musd", "M USD"),
        ("CVaR95 loss reduction", "cvar95_loss_reduction_kusd_h", "k USD/h"),
        ("Imbalance reduction", "imbalance_reduction_gwh", "GWh"),
    ]:
        values = deltas[column].to_numpy(dtype=float)
        n = int(len(values))
        mean = float(np.mean(values))
        if n > 1:
            margin = float(stats.t.ppf(0.975, n - 1) * stats.sem(values))
            _, p_value = stats.ttest_1samp(values, popmean=0.0)
            p_value = float(p_value)
        else:
            margin = 0.0
            p_value = np.nan
        rows.append(
            {
                "metric": metric,
                "column": column,
                "mean": mean,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
                "unit": unit,
                "seed_count": n,
                "positive_seed_count": int(np.sum(values > 0)),
                "paired_t_p_value": p_value,
            }
        )
    return deltas, pd.DataFrame(rows)


def write_markdown(
    path: Path,
    decision_table: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Selected Cloud-Rule Downstream Results",
        "",
        "Scope: common no-text-anchor downstream experiment for the reported manuscript.",
        "",
        "## Main Decision Table",
        "",
        decision_table.to_markdown(index=False),
        "",
        "## Paired Seed Summary",
        "",
        paired_summary.to_markdown(index=False) if not paired_summary.empty else "No paired seed rows generated.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_validation_artifacts(
    output_dir: Path,
    stem: str,
    validation_meta: dict[str, float | int | str],
    validation_preds: pd.DataFrame,
    validation_train_residuals: pd.DataFrame,
    validation_seed_rows: pd.DataFrame,
    validation_summary: pd.DataFrame,
) -> None:
    pd.DataFrame([validation_meta]).to_csv(output_dir / f"{stem}_validation_meta.csv", index=False)
    validation_preds.to_csv(output_dir / f"{stem}_validation_predictions.csv", index=False)
    validation_train_residuals.to_csv(output_dir / f"{stem}_validation_train_residuals.csv", index=False)
    validation_seed_rows.to_csv(output_dir / f"{stem}_validation_seed.csv", index=False)
    validation_summary.to_csv(output_dir / f"{stem}_validation_summary.csv", index=False)


def run(
    results_dir: Path,
    master_path: Path,
    *,
    output_dir: Path | None = None,
    date_tag: str = "20260529",
    validation_train_end: str = "2023-10-01",
    validation_start: str = "2023-10-01",
    validation_end: str = "2024-01-01",
    seeds: tuple[int, ...] = (71_000,),
    lp_weights: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50, 0.75, 1.00),
    selected_weight: float = SELECTED_WEIGHT,
    residual_scale: float = 1.0,
    cvar_gamma: float = 0.25,
    deviation_penalty: float = 50.0,
    scenario_count: int = 20,
    selection_objective: str = "balanced_revenue_cvar_imbalance",
) -> None:
    results_dir = _resolve_path(results_dir) or results_dir
    output_dir = _resolve_path(output_dir) if output_dir is not None else results_dir
    master_path = _resolve_path(master_path) or master_path
    output_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(master_path)
    test_preds = pd.read_csv(results_dir / "test_predictions.csv")
    test_train_residuals = pd.read_csv(results_dir / "train_residuals.csv")
    audit = pd.read_csv(results_dir / "data_audit.csv")
    rated_capacity = float(audit.iloc[0]["rated_capacity_mw"])

    no_text_validation_preds, no_text_validation_train_residuals, no_text_validation_meta = fit_hybrid_blend_split_predictions(
        master=master,
        train_end=validation_train_end,
        eval_start=validation_start,
        eval_end=validation_end,
        pv_model_name="mlp",
        rt_model_name="transformer_none",
        anchor_model_name="mlp",
    )
    llm_validation_preds, llm_validation_train_residuals, llm_validation_meta = fit_hybrid_blend_split_predictions(
        master=master,
        train_end=validation_train_end,
        eval_start=validation_start,
        eval_end=validation_end,
        pv_model_name="mlp_llm_rule_cloud",
        rt_model_name="transformer_llm_rule",
        anchor_model_name="mlp",
    )

    no_text_validation_seed, no_text_validation_summary, _, _ = evaluate_validation_selected_hybrid_blend(
        validation_preds=no_text_validation_preds,
        validation_train_residuals=no_text_validation_train_residuals,
        test_preds=test_preds,
        test_train_residuals=test_train_residuals,
        rated_capacity=rated_capacity,
        seeds=seeds,
        pv_model_name="mlp",
        rt_model_name="transformer_none",
        anchor_model_name="mlp",
        lp_weights=lp_weights,
        residual_scale=residual_scale,
        cvar_gamma=cvar_gamma,
        deviation_penalty=deviation_penalty,
        scenario_count=scenario_count,
        selection_objective=selection_objective,
    )
    llm_validation_seed, llm_validation_summary, _, _ = evaluate_validation_selected_hybrid_blend(
        validation_preds=llm_validation_preds,
        validation_train_residuals=llm_validation_train_residuals,
        test_preds=test_preds,
        test_train_residuals=test_train_residuals,
        rated_capacity=rated_capacity,
        seeds=seeds,
        pv_model_name="mlp_llm_rule_cloud",
        rt_model_name="transformer_llm_rule",
        anchor_model_name="mlp",
        lp_weights=lp_weights,
        residual_scale=residual_scale,
        cvar_gamma=cvar_gamma,
        deviation_penalty=deviation_penalty,
        scenario_count=scenario_count,
        selection_objective=selection_objective,
    )

    no_text_weight_seed, no_text_weight_summary = evaluate_hybrid_blended_scenario_seed_robustness(
        preds=test_preds,
        train_residuals=test_train_residuals,
        rated_capacity=rated_capacity,
        seeds=seeds,
        pv_model_name="mlp",
        rt_model_name="transformer_none",
        anchor_model_name="mlp",
        lp_weights=lp_weights,
        residual_scale=residual_scale,
        cvar_gamma=cvar_gamma,
        deviation_penalty=deviation_penalty,
        scenario_count=scenario_count,
    )
    llm_weight_seed, llm_weight_summary = evaluate_hybrid_blended_scenario_seed_robustness(
        preds=test_preds,
        train_residuals=test_train_residuals,
        rated_capacity=rated_capacity,
        seeds=seeds,
        pv_model_name="mlp_llm_rule_cloud",
        rt_model_name="transformer_llm_rule",
        anchor_model_name="mlp",
        lp_weights=lp_weights,
        residual_scale=residual_scale,
        cvar_gamma=cvar_gamma,
        deviation_penalty=deviation_penalty,
        scenario_count=scenario_count,
    )

    decision_table = build_decision_table(no_text_weight_summary, llm_weight_summary)
    frontier_points = build_frontier_points(no_text_weight_summary, llm_weight_summary)
    paired_deltas, paired_summary = build_paired_seed_deltas(no_text_weight_seed, llm_weight_seed, selected_weight)

    stem = f"selected_cloud_rule_downstream_{date_tag}"
    _write_validation_artifacts(
        output_dir,
        f"{stem}_no_text_common_anchor",
        no_text_validation_meta,
        no_text_validation_preds,
        no_text_validation_train_residuals,
        no_text_validation_seed,
        no_text_validation_summary,
    )
    _write_validation_artifacts(
        output_dir,
        f"{stem}_llm_common_anchor",
        llm_validation_meta,
        llm_validation_preds,
        llm_validation_train_residuals,
        llm_validation_seed,
        llm_validation_summary,
    )
    no_text_weight_seed.to_csv(output_dir / f"{stem}_no_text_common_anchor_weight_seed.csv", index=False)
    no_text_weight_summary.to_csv(output_dir / f"{stem}_no_text_common_anchor_weight_summary.csv", index=False)
    llm_weight_seed.to_csv(output_dir / f"{stem}_llm_common_anchor_weight_seed.csv", index=False)
    llm_weight_summary.to_csv(output_dir / f"{stem}_llm_common_anchor_weight_summary.csv", index=False)
    decision_table.to_csv(output_dir / f"{stem}_decision_table.csv", index=False)
    frontier_points.to_csv(output_dir / f"{stem}_decision_frontier_points.csv", index=False)
    paired_deltas.to_csv(output_dir / f"{stem}_paired_seed_deltas.csv", index=False)
    paired_summary.to_csv(output_dir / f"{stem}_paired_seed_summary.csv", index=False)
    write_markdown(output_dir / f"{stem}_summary.md", decision_table, paired_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the common no-text-anchor downstream decision evaluation.")
    parser.add_argument("--results-dir", type=Path, default=Path("results_selected_cloud_rule_2022_2025_test_2024_2025"))
    parser.add_argument(
        "--master-path",
        type=Path,
        default=Path("data_multi_weather_2022_2025/processed/master_hourly_caiso_noaa_2022-01-01_2026-01-01.csv"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--date-tag", default="20260529")
    parser.add_argument("--validation-train-end", default="2023-10-01")
    parser.add_argument("--validation-start", default="2023-10-01")
    parser.add_argument("--validation-end", default="2024-01-01")
    parser.add_argument("--seeds", default="71000,71001,71011,71021,71031,71041,71051,71061,71071,71081")
    parser.add_argument("--lp-weights", default="0.00,0.10,0.25,0.50,0.75,1.00")
    parser.add_argument("--selected-weight", type=float, default=SELECTED_WEIGHT)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--cvar-gamma", type=float, default=0.25)
    parser.add_argument("--deviation-penalty", type=float, default=50.0)
    parser.add_argument("--scenario-count", type=int, default=20)
    parser.add_argument("--selection-objective", default="balanced_revenue_cvar_imbalance")
    args = parser.parse_args()
    run(
        results_dir=args.results_dir,
        master_path=args.master_path,
        output_dir=args.output_dir,
        date_tag=args.date_tag,
        validation_train_end=args.validation_train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        seeds=_parse_int_tuple(args.seeds),
        lp_weights=_parse_float_tuple(args.lp_weights),
        selected_weight=args.selected_weight,
        residual_scale=args.residual_scale,
        cvar_gamma=args.cvar_gamma,
        deviation_penalty=args.deviation_penalty,
        scenario_count=args.scenario_count,
        selection_objective=args.selection_objective,
    )


if __name__ == "__main__":
    main()
