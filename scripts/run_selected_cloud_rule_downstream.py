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
    cvar_loss,
    evaluate_hybrid_blended_scenario_seed_robustness,
    evaluate_validation_selected_hybrid_blend,
    fit_hybrid_blend_split_predictions,
    settlement_revenue,
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
    *,
    pv_scenario: str = "",
    rt_scenario: str = "",
    anchor: str = "",
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
        "pv_scenario": pv_scenario,
        "rt_scenario": rt_scenario,
        "anchor": anchor,
        "lp_weight": float(row["lp_weight"]),
        "seed_count": int(row.get("seed_count", 0)),
        "value_musd": value_musd,
        "cvar95_loss_kusd_h": cvar_kusd_h,
        "imbalance_gwh": imbalance_gwh,
        "value_delta_musd_vs_rule": value_delta,
        "cvar_delta_kusd_h_vs_rule": cvar_delta,
        "imbalance_delta_gwh_vs_rule": imbalance_delta,
    }


def _add_no_text_reference_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    selected = out[out["comparison"].astype(str).eq("validation_selected")]
    no_text = selected[selected["strategy"].astype(str).eq("No-text hybrid blend")]
    if no_text.empty:
        out["value_delta_musd_vs_no_text"] = np.nan
        out["cvar_delta_kusd_h_vs_no_text"] = np.nan
        out["imbalance_delta_gwh_vs_no_text"] = np.nan
        return out
    ref = no_text.iloc[0]
    out["value_delta_musd_vs_no_text"] = out["value_musd"] - float(ref["value_musd"])
    out["cvar_delta_kusd_h_vs_no_text"] = float(ref["cvar95_loss_kusd_h"]) - out["cvar95_loss_kusd_h"]
    out["imbalance_delta_gwh_vs_no_text"] = float(ref["imbalance_gwh"]) - out["imbalance_gwh"]
    return out


def _merge_public_forecast(test_preds: pd.DataFrame, master: pd.DataFrame, rated_capacity: float) -> pd.DataFrame:
    if "pv_caiso_solar_dam" in test_preds.columns:
        out = test_preds.copy()
        out["pv_caiso_solar_dam"] = out["pv_caiso_solar_dam"].clip(lower=0.0, upper=rated_capacity)
        return out
    required = {"timestamp_utc", "caiso_solar_dam_forecast_mw"}
    if not required.issubset(master.columns):
        missing = ", ".join(sorted(required - set(master.columns)))
        raise ValueError(f"master data are missing public forecast columns: {missing}")
    out = test_preds.copy()
    public = master[["timestamp_utc", "caiso_solar_dam_forecast_mw"]].copy()
    out["_timestamp_key"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    public["_timestamp_key"] = pd.to_datetime(public["timestamp_utc"], utc=True)
    public = public.drop_duplicates("_timestamp_key")
    out = out.merge(public[["_timestamp_key", "caiso_solar_dam_forecast_mw"]], on="_timestamp_key", how="left")
    missing_frac = float(out["caiso_solar_dam_forecast_mw"].isna().mean())
    if missing_frac > 0.0:
        raise ValueError(f"CAISO public solar forecast is missing for {missing_frac:.2%} of test rows")
    out["pv_caiso_solar_dam"] = out["caiso_solar_dam_forecast_mw"].clip(lower=0.0, upper=rated_capacity)
    return out.drop(columns=["_timestamp_key"])


def evaluate_deterministic_anchor(
    preds: pd.DataFrame,
    *,
    quantity_col: str,
    rated_capacity: float,
    deviation_penalty: float,
    strategy_name: str,
) -> pd.Series:
    required = {"pv_mw", "da_lmp", "rt_lmp", quantity_col}
    if not required.issubset(preds.columns):
        missing = ", ".join(sorted(required - set(preds.columns)))
        raise ValueError(f"deterministic anchor input is missing columns: {missing}")
    actual = preds["pv_mw"].to_numpy(dtype=float)
    da = preds["da_lmp"].to_numpy(dtype=float)
    rt = preds["rt_lmp"].to_numpy(dtype=float)
    q = np.clip(np.nan_to_num(preds[quantity_col].to_numpy(dtype=float), nan=0.0), 0.0, rated_capacity)
    revenue = settlement_revenue(
        actual,
        q,
        da,
        rt,
        shortage_penalty=deviation_penalty,
        surplus_penalty=deviation_penalty,
    )
    shortage = np.maximum(q - actual, 0.0)
    surplus = np.maximum(actual - q, 0.0)
    return pd.Series(
        {
            "strategy": strategy_name,
            "pv_model": quantity_col,
            "rt_model": "settlement_baseline",
            "anchor_model": quantity_col,
            "lp_weight": np.nan,
            "seed_count": 1,
            "hours": int(len(q)),
            "total_revenue_mean": float(np.sum(revenue)),
            "cvar_95_loss_mean": cvar_loss(-revenue, alpha=0.95),
            "imbalance_mwh_proxy_mean": float(np.sum(shortage + surplus)),
            "shortage_mwh_proxy_mean": float(np.sum(shortage)),
            "surplus_mwh_proxy_mean": float(np.sum(surplus)),
        }
    )


def build_main_downstream_summary(
    rule_selected_summary: pd.DataFrame,
    llm_selected_summary: pd.DataFrame,
    pure_llm_summary: pd.DataFrame,
    *,
    no_text_selected_summary: pd.DataFrame | None = None,
    llm_no_text_anchor_selected_summary: pd.DataFrame | None = None,
    public_anchor_summary: pd.Series | None = None,
    fixed_rule_summary: pd.DataFrame | None = None,
    fixed_llm_summary: pd.DataFrame | None = None,
    fixed_weight: float = 0.50,
) -> pd.DataFrame:
    rule_selected = _row_for_weight(rule_selected_summary)
    llm_selected = _row_for_weight(llm_selected_summary)
    pure_llm = _row_for_weight(pure_llm_summary, 1.0)
    rows = []
    if public_anchor_summary is not None:
        rows.append(
            _metric_row(
                "validation_selected",
                "CAISO public forecast anchor",
                "CAISO public PV forecast + settlement price baseline",
                public_anchor_summary,
                rule_selected,
                pv_scenario="CAISO public forecast",
                rt_scenario="Settlement price baseline",
                anchor="CAISO public PV forecast",
            )
        )
    if no_text_selected_summary is not None and not no_text_selected_summary.empty:
        no_text_selected = _row_for_weight(no_text_selected_summary)
        rows.append(
            _metric_row(
                "validation_selected",
                "No-text hybrid blend",
                "PV MLP no-text + RT Transformer no-text",
                no_text_selected,
                rule_selected,
                pv_scenario="MLP no-text",
                rt_scenario="Transformer no-text",
                anchor="MLP rule-core",
            )
        )
    if llm_no_text_anchor_selected_summary is not None and not llm_no_text_anchor_selected_summary.empty:
        llm_no_text_anchor_selected = _row_for_weight(llm_no_text_anchor_selected_summary)
        rows.append(
            _metric_row(
                "validation_selected",
                "LLM cloud-rule no-text-anchor hybrid blend",
                "PV MLP LLM rule-cloud + RT Transformer LLM-rule + no-text PV anchor",
                llm_no_text_anchor_selected,
                rule_selected,
                pv_scenario="MLP LLM cloud-rule",
                rt_scenario="Transformer LLM-rule",
                anchor="MLP no-text",
            )
        )
    rows.extend(
        [
        _metric_row(
            "validation_selected",
            "Rule-core hybrid blend",
            "PV MLP rule-core + RT Transformer rule-text",
            rule_selected,
            pv_scenario="MLP rule-core",
            rt_scenario="Transformer rule-text",
            anchor="MLP rule-core",
        ),
        _metric_row(
            "validation_selected",
            "Pure LLM cloud-rule LP",
            "PV MLP LLM rule-cloud + RT Transformer LLM-rule",
            pure_llm,
            rule_selected,
            pv_scenario="MLP LLM cloud-rule",
            rt_scenario="Transformer LLM-rule",
            anchor="None",
        ),
        _metric_row(
            "validation_selected",
            "LLM cloud-rule hybrid blend",
            "PV MLP LLM rule-cloud + RT Transformer LLM-rule",
            llm_selected,
            rule_selected,
            pv_scenario="MLP LLM cloud-rule",
            rt_scenario="Transformer LLM-rule",
            anchor="MLP rule-core",
        ),
        ]
    )
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
                    pv_scenario="MLP rule-core",
                    rt_scenario="Transformer rule-text",
                    anchor="MLP rule-core",
                ),
                _metric_row(
                    f"fixed_w_{fixed_weight:.2f}",
                    "LLM cloud-rule hybrid blend",
                    "PV MLP LLM rule-cloud + RT Transformer LLM-rule",
                    fixed_llm,
                    fixed_rule,
                    pv_scenario="MLP LLM cloud-rule",
                    rt_scenario="Transformer LLM-rule",
                    anchor="MLP rule-core",
                ),
            ]
        )
    return _add_no_text_reference_deltas(pd.DataFrame(rows))


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
        "pv_scenario",
        "rt_scenario",
        "anchor",
        "lp_weight",
        "value_musd",
        "cvar95_loss_kusd_h",
        "imbalance_gwh",
        "value_delta_musd_vs_rule",
        "cvar_delta_kusd_h_vs_rule",
        "imbalance_delta_gwh_vs_rule",
        "value_delta_musd_vs_no_text",
        "cvar_delta_kusd_h_vs_no_text",
        "imbalance_delta_gwh_vs_no_text",
    ]
    columns = [col for col in columns if col in main_summary.columns]
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
    test_preds = _merge_public_forecast(test_preds, master, rated_capacity)
    fusion_lp_weights = tuple(sorted(set(lp_weights + (0.0, 0.10))))

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
    no_text_validation_preds, no_text_validation_train_residuals, no_text_validation_meta = fit_hybrid_blend_split_predictions(
        master=master,
        train_end=validation_train_end,
        eval_start=validation_start,
        eval_end=validation_end,
        pv_model_name="mlp",
        rt_model_name="transformer_none",
        anchor_model_name="mlp_rule_core",
    )
    llm_no_text_anchor_validation_preds, llm_no_text_anchor_validation_train_residuals, llm_no_text_anchor_validation_meta = fit_hybrid_blend_split_predictions(
        master=master,
        train_end=validation_train_end,
        eval_start=validation_start,
        eval_end=validation_end,
        pv_model_name="mlp_llm_rule_cloud",
        rt_model_name="transformer_llm_rule",
        anchor_model_name="mlp",
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
    no_text_validation_seed, no_text_validation_summary, no_text_test_seed, no_text_test_summary = (
        evaluate_validation_selected_hybrid_blend(
            validation_preds=no_text_validation_preds,
            validation_train_residuals=no_text_validation_train_residuals,
            test_preds=test_preds,
            test_train_residuals=test_train_residuals,
            rated_capacity=rated_capacity,
            seeds=seeds,
            pv_model_name="mlp",
            rt_model_name="transformer_none",
            anchor_model_name="mlp_rule_core",
            lp_weights=lp_weights,
            residual_scale=residual_scale,
            cvar_gamma=cvar_gamma,
            deviation_penalty=deviation_penalty,
            scenario_count=scenario_count,
            selection_objective=selection_objective,
        )
    )
    (
        llm_no_text_anchor_validation_seed,
        llm_no_text_anchor_validation_summary,
        llm_no_text_anchor_test_seed,
        llm_no_text_anchor_test_summary,
    ) = evaluate_validation_selected_hybrid_blend(
        validation_preds=llm_no_text_anchor_validation_preds,
        validation_train_residuals=llm_no_text_anchor_validation_train_residuals,
        test_preds=test_preds,
        test_train_residuals=test_train_residuals,
        rated_capacity=rated_capacity,
        seeds=seeds,
        pv_model_name="mlp_llm_rule_cloud",
        rt_model_name="transformer_llm_rule",
        anchor_model_name="mlp",
        lp_weights=fusion_lp_weights,
        residual_scale=residual_scale,
        cvar_gamma=cvar_gamma,
        deviation_penalty=deviation_penalty,
        scenario_count=scenario_count,
        selection_objective="balanced_revenue_cvar_imbalance",
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
    public_anchor_summary = evaluate_deterministic_anchor(
        test_preds,
        quantity_col="pv_caiso_solar_dam",
        rated_capacity=rated_capacity,
        deviation_penalty=deviation_penalty,
        strategy_name="CAISO_public_forecast_anchor",
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
    _write_validation_artifacts(
        output_dir,
        f"{stem}_no_text_validation_selected",
        no_text_validation_meta,
        no_text_validation_preds,
        no_text_validation_train_residuals,
        no_text_validation_seed,
        no_text_validation_summary,
        no_text_test_seed,
        no_text_test_summary,
    )
    _write_validation_artifacts(
        output_dir,
        f"{stem}_llm_no_text_anchor_validation_selected",
        llm_no_text_anchor_validation_meta,
        llm_no_text_anchor_validation_preds,
        llm_no_text_anchor_validation_train_residuals,
        llm_no_text_anchor_validation_seed,
        llm_no_text_anchor_validation_summary,
        llm_no_text_anchor_test_seed,
        llm_no_text_anchor_test_summary,
    )
    pure_llm_seed.to_csv(output_dir / f"{stem}_pure_llm_lp_seed.csv", index=False)
    pure_llm_summary.to_csv(output_dir / f"{stem}_pure_llm_lp_summary.csv", index=False)
    pd.DataFrame([public_anchor_summary]).to_csv(output_dir / f"{stem}_public_forecast_anchor_summary.csv", index=False)
    rule_weight_seed.to_csv(output_dir / f"{stem}_rule_core_weight_sensitivity_seed.csv", index=False)
    rule_weight_summary.to_csv(output_dir / f"{stem}_rule_core_weight_sensitivity_summary.csv", index=False)
    llm_weight_seed.to_csv(output_dir / f"{stem}_llm_cloud_rule_weight_sensitivity_seed.csv", index=False)
    llm_weight_summary.to_csv(output_dir / f"{stem}_llm_cloud_rule_weight_sensitivity_summary.csv", index=False)

    main_summary = build_main_downstream_summary(
        rule_test_summary,
        llm_test_summary,
        pure_llm_summary,
        no_text_selected_summary=no_text_test_summary,
        llm_no_text_anchor_selected_summary=llm_no_text_anchor_test_summary,
        public_anchor_summary=public_anchor_summary,
        fixed_rule_summary=rule_weight_summary,
        fixed_llm_summary=llm_weight_summary,
        fixed_weight=fixed_weight,
    )
    weight_sensitivity = build_weight_sensitivity(rule_weight_summary, llm_weight_summary)
    paired_deltas, paired_summary = build_paired_seed_effects(rule_test_seed, llm_test_seed)
    paired_deltas_vs_no_text, paired_summary_vs_no_text = build_paired_seed_effects(no_text_test_seed, llm_test_seed)
    paired_deltas_fused_vs_no_text, paired_summary_fused_vs_no_text = build_paired_seed_effects(
        no_text_test_seed,
        llm_no_text_anchor_test_seed,
    )
    main_summary.to_csv(output_dir / f"{stem}_summary.csv", index=False)
    weight_sensitivity.to_csv(output_dir / f"{stem}_weight_sensitivity.csv", index=False)
    paired_deltas.to_csv(output_dir / f"{stem}_paired_seed_deltas.csv", index=False)
    paired_summary.to_csv(output_dir / f"{stem}_paired_seed_summary.csv", index=False)
    paired_deltas_vs_no_text.to_csv(output_dir / f"{stem}_paired_seed_deltas_vs_no_text.csv", index=False)
    paired_summary_vs_no_text.to_csv(output_dir / f"{stem}_paired_seed_summary_vs_no_text.csv", index=False)
    paired_deltas_fused_vs_no_text.to_csv(output_dir / f"{stem}_paired_seed_deltas_fused_vs_no_text.csv", index=False)
    paired_summary_fused_vs_no_text.to_csv(output_dir / f"{stem}_paired_seed_summary_fused_vs_no_text.csv", index=False)
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
