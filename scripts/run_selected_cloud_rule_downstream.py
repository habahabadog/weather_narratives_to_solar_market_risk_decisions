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


def _resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else ROOT / path


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _row_for_weight(summary: pd.DataFrame, lp_weight: float | None = None) -> pd.Series:
    if summary.empty:
        raise ValueError("summary is empty")
    if lp_weight is None:
        return summary.iloc[0]
    weights = pd.to_numeric(summary["lp_weight"], errors="coerce")
    match = summary[np.isclose(weights, float(lp_weight))]
    if match.empty:
        raise ValueError(f"summary has no row for lp_weight={lp_weight:.2f}")
    return match.iloc[0]


def _metric_row(
    comparison: str,
    strategy: str,
    scenario_source: str,
    row: pd.Series,
    rule_reference: pd.Series | None = None,
) -> dict[str, float | int | str]:
    value_musd = float(row["total_revenue_mean"]) / 1_000_000.0
    cvar_kusd_h = float(row["cvar_95_loss_mean"]) / 1_000.0
    imbalance_gwh = float(row["imbalance_mwh_proxy_mean"]) / 1_000.0
    if rule_reference is None:
        value_delta = 0.0
        cvar_delta = 0.0
        imbalance_delta = 0.0
    else:
        rule_value_musd = float(rule_reference["total_revenue_mean"]) / 1_000_000.0
        rule_cvar_kusd_h = float(rule_reference["cvar_95_loss_mean"]) / 1_000.0
        rule_imbalance_gwh = float(rule_reference["imbalance_mwh_proxy_mean"]) / 1_000.0
        value_delta = value_musd - rule_value_musd
        cvar_delta = rule_cvar_kusd_h - cvar_kusd_h
        imbalance_delta = rule_imbalance_gwh - imbalance_gwh
    return {
        "comparison": comparison,
        "strategy": strategy,
        "scenario_source": scenario_source,
        "lp_weight": float(row["lp_weight"]),
        "seed_count": int(row.get("seed_count", 0)),
        "value_musd": value_musd,
        "cvar95_loss_kusd_h": cvar_kusd_h,
        "imbalance_gwh": imbalance_gwh,
        "value_delta_musd_vs_rule": value_delta,
        "cvar_delta_kusd_h_vs_rule": cvar_delta,
        "imbalance_delta_gwh_vs_rule": imbalance_delta,
    }


def build_main_downstream_summary(
    rule_selected_summary: pd.DataFrame,
    llm_selected_summary: pd.DataFrame,
    pure_llm_summary: pd.DataFrame,
    *,
    fixed_rule_summary: pd.DataFrame | None = None,
    fixed_llm_summary: pd.DataFrame | None = None,
    fixed_weight: float = 0.50,
) -> pd.DataFrame:
    rule_selected = _row_for_weight(rule_selected_summary)
    llm_selected = _row_for_weight(llm_selected_summary)
    pure_llm = _row_for_weight(pure_llm_summary, 1.0)
    rows = [
        _metric_row(
            "validation_selected",
            "Rule-core hybrid blend",
            "PV MLP rule-core + RT Transformer rule-text",
            rule_selected,
        ),
        _metric_row(
            "validation_selected",
            "Pure LLM cloud-rule LP",
            "PV MLP LLM rule-cloud + RT Transformer LLM-rule",
            pure_llm,
            rule_selected,
        ),
        _metric_row(
            "validation_selected",
            "LLM cloud-rule hybrid blend",
            "PV MLP LLM rule-cloud + RT Transformer LLM-rule",
            llm_selected,
            rule_selected,
        ),
    ]
    if fixed_rule_summary is not None and fixed_llm_summary is not None:
        fixed_rule = _row_for_weight(fixed_rule_summary, fixed_weight)
        fixed_llm = _row_for_weight(fixed_llm_summary, fixed_weight)
        rows.extend(
            [
                _metric_row(
                    f"fixed_w_{fixed_weight:.2f}",
                    "Rule-core hybrid blend",
                    "PV MLP rule-core + RT Transformer rule-text",
                    fixed_rule,
                ),
                _metric_row(
                    f"fixed_w_{fixed_weight:.2f}",
                    "LLM cloud-rule hybrid blend",
                    "PV MLP LLM rule-cloud + RT Transformer LLM-rule",
                    fixed_llm,
                    fixed_rule,
                ),
            ]
        )
    return pd.DataFrame(rows)


def build_weight_sensitivity(rule_summary: pd.DataFrame, llm_summary: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "lp_weight",
        "total_revenue_mean",
        "cvar_95_loss_mean",
        "imbalance_mwh_proxy_mean",
    ]
    merged = rule_summary[value_cols].merge(
        llm_summary[value_cols],
        on="lp_weight",
        suffixes=("_rule", "_llm"),
    )
    merged = merged.sort_values("lp_weight").reset_index(drop=True)
    merged["value_delta_musd"] = (
        merged["total_revenue_mean_llm"] - merged["total_revenue_mean_rule"]
    ) / 1_000_000.0
    merged["cvar95_loss_reduction_kusd_h"] = (
        merged["cvar_95_loss_mean_rule"] - merged["cvar_95_loss_mean_llm"]
    ) / 1_000.0
    merged["imbalance_reduction_gwh"] = (
        merged["imbalance_mwh_proxy_mean_rule"] - merged["imbalance_mwh_proxy_mean_llm"]
    ) / 1_000.0
    return merged


def build_paired_seed_effects(rule_seed: pd.DataFrame, llm_seed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rule_seed.empty or llm_seed.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = rule_seed.merge(llm_seed, on="seed", suffixes=("_rule", "_llm"))
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    deltas = pd.DataFrame(
        {
            "seed": merged["seed"],
            "value_delta_musd": (merged["total_revenue_llm"] - merged["total_revenue_rule"]) / 1_000_000.0,
            "cvar95_loss_reduction_kusd_h": (merged["cvar_95_loss_rule"] - merged["cvar_95_loss_llm"]) / 1_000.0,
            "imbalance_reduction_gwh": (
                merged["imbalance_mwh_proxy_rule"] - merged["imbalance_mwh_proxy_llm"]
            )
            / 1_000.0,
        }
    )
    metric_specs = [
        ("Value gain", "value_delta_musd", "M USD"),
        ("CVaR95 loss reduction", "cvar95_loss_reduction_kusd_h", "k USD/h"),
        ("Imbalance reduction", "imbalance_reduction_gwh", "GWh"),
    ]
    rows: list[dict[str, float | int | str]] = []
    for label, col, unit in metric_specs:
        values = deltas[col].to_numpy(dtype=float)
        n = int(len(values))
        mean = float(np.mean(values))
        if n > 1:
            sem = float(stats.sem(values))
            margin = float(stats.t.ppf(0.975, n - 1) * sem)
            _, p_value = stats.ttest_1samp(values, popmean=0.0)
            p_value = float(p_value)
        else:
            margin = 0.0
            p_value = np.nan
        rows.append(
            {
                "metric": label,
                "mean": mean,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
                "unit": unit,
                "improved_seed_count": int(np.sum(values > 0.0)),
                "seed_count": n,
                "paired_t_p_value": p_value,
            }
        )
    return deltas, pd.DataFrame(rows)


def write_markdown(
    path: Path,
    main_summary: pd.DataFrame,
    weight_sensitivity: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> None:
    columns = [
        "comparison",
        "strategy",
        "lp_weight",
        "value_musd",
        "cvar95_loss_kusd_h",
        "imbalance_gwh",
        "value_delta_musd_vs_rule",
        "cvar_delta_kusd_h_vs_rule",
        "imbalance_delta_gwh_vs_rule",
    ]
    lines = [
        "# Selected Cloud-Rule Downstream Results",
        "",
        "Scope: neural selected cloud-rule downstream only; no ridge/GBR or legacy bidding baseline is used.",
        "",
        "## Main Downstream Summary",
        "",
        main_summary[columns].to_markdown(index=False),
        "",
        "## LP-Weight Sensitivity",
        "",
        weight_sensitivity.to_markdown(index=False) if not weight_sensitivity.empty else "No weight rows generated.",
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
    test_seed_rows: pd.DataFrame,
    test_summary: pd.DataFrame,
) -> None:
    pd.DataFrame([validation_meta]).to_csv(output_dir / f"{stem}_validation_meta.csv", index=False)
    validation_preds.to_csv(output_dir / f"{stem}_validation_predictions.csv", index=False)
    validation_train_residuals.to_csv(output_dir / f"{stem}_validation_train_residuals.csv", index=False)
    validation_seed_rows.to_csv(output_dir / f"{stem}_validation_seed.csv", index=False)
    validation_summary.to_csv(output_dir / f"{stem}_validation_summary.csv", index=False)
    test_seed_rows.to_csv(output_dir / f"{stem}_test_seed.csv", index=False)
    test_summary.to_csv(output_dir / f"{stem}_test_summary.csv", index=False)


def run(
    results_dir: Path,
    master_path: Path,
    *,
    output_dir: Path | None = None,
    date_tag: str = "20260528",
    validation_train_end: str = "2023-10-01",
    validation_start: str = "2023-10-01",
    validation_end: str = "2024-01-01",
    seeds: tuple[int, ...] = (71_000,),
    lp_weights: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
    fixed_weight: float = 0.50,
    residual_scale: float = 1.0,
    cvar_gamma: float = 0.25,
    deviation_penalty: float = 50.0,
    scenario_count: int = 20,
    selection_objective: str = "balanced_revenue_cvar",
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

    rule_validation_preds, rule_validation_train_residuals, rule_validation_meta = fit_hybrid_blend_split_predictions(
        master=master,
        train_end=validation_train_end,
        eval_start=validation_start,
        eval_end=validation_end,
        pv_model_name="mlp_rule_core",
        rt_model_name="transformer_text",
        anchor_model_name="mlp_rule_core",
    )
    llm_validation_preds, llm_validation_train_residuals, llm_validation_meta = fit_hybrid_blend_split_predictions(
        master=master,
        train_end=validation_train_end,
        eval_start=validation_start,
        eval_end=validation_end,
        pv_model_name="mlp_llm_rule_cloud",
        rt_model_name="transformer_llm_rule",
        anchor_model_name="mlp_rule_core",
    )

    rule_validation_seed, rule_validation_summary, rule_test_seed, rule_test_summary = (
        evaluate_validation_selected_hybrid_blend(
            validation_preds=rule_validation_preds,
            validation_train_residuals=rule_validation_train_residuals,
            test_preds=test_preds,
            test_train_residuals=test_train_residuals,
            rated_capacity=rated_capacity,
            seeds=seeds,
            pv_model_name="mlp_rule_core",
            rt_model_name="transformer_text",
            anchor_model_name="mlp_rule_core",
            lp_weights=lp_weights,
            residual_scale=residual_scale,
            cvar_gamma=cvar_gamma,
            deviation_penalty=deviation_penalty,
            scenario_count=scenario_count,
            selection_objective=selection_objective,
        )
    )
    llm_validation_seed, llm_validation_summary, llm_test_seed, llm_test_summary = (
        evaluate_validation_selected_hybrid_blend(
            validation_preds=llm_validation_preds,
            validation_train_residuals=llm_validation_train_residuals,
            test_preds=test_preds,
            test_train_residuals=test_train_residuals,
            rated_capacity=rated_capacity,
            seeds=seeds,
            pv_model_name="mlp_llm_rule_cloud",
            rt_model_name="transformer_llm_rule",
            anchor_model_name="mlp_rule_core",
            lp_weights=lp_weights,
            residual_scale=residual_scale,
            cvar_gamma=cvar_gamma,
            deviation_penalty=deviation_penalty,
            scenario_count=scenario_count,
            selection_objective=selection_objective,
        )
    )
    pure_llm_seed, pure_llm_summary = evaluate_hybrid_blended_scenario_seed_robustness(
        preds=test_preds,
        train_residuals=test_train_residuals,
        rated_capacity=rated_capacity,
        seeds=seeds,
        pv_model_name="mlp_llm_rule_cloud",
        rt_model_name="transformer_llm_rule",
        anchor_model_name="mlp_rule_core",
        lp_weights=(1.0,),
        residual_scale=residual_scale,
        cvar_gamma=cvar_gamma,
        deviation_penalty=deviation_penalty,
        scenario_count=scenario_count,
    )
    rule_weight_seed, rule_weight_summary = evaluate_hybrid_blended_scenario_seed_robustness(
        preds=test_preds,
        train_residuals=test_train_residuals,
        rated_capacity=rated_capacity,
        seeds=seeds,
        pv_model_name="mlp_rule_core",
        rt_model_name="transformer_text",
        anchor_model_name="mlp_rule_core",
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
        anchor_model_name="mlp_rule_core",
        lp_weights=lp_weights,
        residual_scale=residual_scale,
        cvar_gamma=cvar_gamma,
        deviation_penalty=deviation_penalty,
        scenario_count=scenario_count,
    )

    stem = f"selected_cloud_rule_downstream_{date_tag}"
    _write_validation_artifacts(
        output_dir,
        f"{stem}_rule_core_validation_selected",
        rule_validation_meta,
        rule_validation_preds,
        rule_validation_train_residuals,
        rule_validation_seed,
        rule_validation_summary,
        rule_test_seed,
        rule_test_summary,
    )
    _write_validation_artifacts(
        output_dir,
        f"{stem}_llm_cloud_rule_validation_selected",
        llm_validation_meta,
        llm_validation_preds,
        llm_validation_train_residuals,
        llm_validation_seed,
        llm_validation_summary,
        llm_test_seed,
        llm_test_summary,
    )
    pure_llm_seed.to_csv(output_dir / f"{stem}_pure_llm_lp_seed.csv", index=False)
    pure_llm_summary.to_csv(output_dir / f"{stem}_pure_llm_lp_summary.csv", index=False)
    rule_weight_seed.to_csv(output_dir / f"{stem}_rule_core_weight_sensitivity_seed.csv", index=False)
    rule_weight_summary.to_csv(output_dir / f"{stem}_rule_core_weight_sensitivity_summary.csv", index=False)
    llm_weight_seed.to_csv(output_dir / f"{stem}_llm_cloud_rule_weight_sensitivity_seed.csv", index=False)
    llm_weight_summary.to_csv(output_dir / f"{stem}_llm_cloud_rule_weight_sensitivity_summary.csv", index=False)

    main_summary = build_main_downstream_summary(
        rule_test_summary,
        llm_test_summary,
        pure_llm_summary,
        fixed_rule_summary=rule_weight_summary,
        fixed_llm_summary=llm_weight_summary,
        fixed_weight=fixed_weight,
    )
    weight_sensitivity = build_weight_sensitivity(rule_weight_summary, llm_weight_summary)
    paired_deltas, paired_summary = build_paired_seed_effects(rule_test_seed, llm_test_seed)
    main_summary.to_csv(output_dir / f"{stem}_summary.csv", index=False)
    weight_sensitivity.to_csv(output_dir / f"{stem}_weight_sensitivity.csv", index=False)
    paired_deltas.to_csv(output_dir / f"{stem}_paired_seed_deltas.csv", index=False)
    paired_summary.to_csv(output_dir / f"{stem}_paired_seed_summary.csv", index=False)
    write_markdown(output_dir / f"{stem}_summary.md", main_summary, weight_sensitivity, paired_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected cloud-rule downstream decision evaluation.")
    parser.add_argument("--results-dir", type=Path, default=Path("results_selected_cloud_rule_2022_2025_test_2024_2025"))
    parser.add_argument(
        "--master-path",
        type=Path,
        default=Path("data_multi_weather_2022_2025/processed/master_hourly_caiso_noaa_2022-01-01_2026-01-01.csv"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--date-tag", default="20260528")
    parser.add_argument("--validation-train-end", default="2023-10-01")
    parser.add_argument("--validation-start", default="2023-10-01")
    parser.add_argument("--validation-end", default="2024-01-01")
    parser.add_argument("--seeds", default="71000")
    parser.add_argument("--lp-weights", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--fixed-weight", type=float, default=0.50)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--cvar-gamma", type=float, default=0.25)
    parser.add_argument("--deviation-penalty", type=float, default=50.0)
    parser.add_argument("--scenario-count", type=int, default=20)
    parser.add_argument("--selection-objective", default="balanced_revenue_cvar")
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
        fixed_weight=args.fixed_weight,
        residual_scale=args.residual_scale,
        cvar_gamma=args.cvar_gamma,
        deviation_penalty=args.deviation_penalty,
        scenario_count=args.scenario_count,
        selection_objective=args.selection_objective,
    )


if __name__ == "__main__":
    main()
