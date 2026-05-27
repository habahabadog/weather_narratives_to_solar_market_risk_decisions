from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_baselines import (
    build_master_table,
    fit_hybrid_blend_split_predictions,
    regression_metrics,
)


DEFAULT_SUFFIX = "2023-01-01_2025-01-01"
DEFAULT_MASTER_NAME = "master_hourly_caiso_noaa_2023-01-19_2024-12-31.csv"


MODEL_RUNS = [
    {
        "config_label": "rule_core_reference",
        "pv_model": "mlp_rule_core",
        "rt_model": "transformer_text",
        "anchor_model": "mlp_rule_core",
    },
    {
        "config_label": "llm_cloud_rule",
        "pv_model": "mlp_llm_rule_cloud",
        "rt_model": "transformer_llm_rule",
        "anchor_model": "mlp_rule_core",
    },
]


FORECAST_SPECS = [
    ("pv", "PV generation", "pv_mw", "pv_mlp_rule_core", "MLP rule-core", True),
    ("pv", "PV generation", "pv_mw", "pv_mlp_llm_rule_cloud", "MLP LLM cloud-rule", True),
    ("rt_price", "Real-time price", "rt_lmp", "rt_transformer_text", "Transformer rule-text", False),
    ("rt_price", "Real-time price", "rt_lmp", "rt_transformer_llm_rule", "Transformer LLM rule", False),
]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _read_or_build_master(processed_dir: Path, data_suffix: str, master_path: Path | None) -> tuple[pd.DataFrame, dict[str, object], Path]:
    processed_dir = _resolve(processed_dir)
    if master_path is not None:
        resolved_master_path = _resolve(master_path)
        master = pd.read_csv(resolved_master_path)
        return master, {"master_source": str(resolved_master_path), "data_suffix": data_suffix}, resolved_master_path

    master, audit = build_master_table(processed_dir, data_suffix=data_suffix)
    out_path = processed_dir / DEFAULT_MASTER_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False)
    audit["master_source"] = str(out_path)
    return master, audit, out_path


def _merge_columns(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if len(base) != len(extra):
        raise ValueError("cannot merge prediction frames with different row counts")
    out = base.copy()
    for column in extra.columns:
        if column not in out.columns:
            out[column] = extra[column].to_numpy()
    return out


def _forecast_masks(preds: pd.DataFrame) -> dict[str, pd.Series]:
    solar = pd.to_numeric(preds["is_solar_hour"], errors="coerce").fillna(0).astype(int).eq(1)
    extreme = pd.to_numeric(preds["has_extreme_event"], errors="coerce").fillna(0).astype(int).eq(1)
    return {
        "all": pd.Series(True, index=preds.index),
        "solar_hours": solar,
        "extreme": extreme,
        "extreme_solar_hours": solar & extreme,
    }


def _build_forecast_metrics(preds: pd.DataFrame, rated_capacity: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    masks = _forecast_masks(preds)
    for subset, mask in masks.items():
        for target, target_label, actual_col, pred_col, model_label, is_pv in FORECAST_SPECS:
            if pred_col not in preds.columns:
                continue
            subset_frame = preds.loc[mask, [actual_col, pred_col]].dropna()
            metrics = regression_metrics(
                subset_frame[actual_col],
                subset_frame[pred_col],
                rated_capacity=rated_capacity if is_pv else None,
            )
            rows.append(
                {
                    "target": target,
                    "target_label": target_label,
                    "subset": subset,
                    "model": pred_col,
                    "model_label": model_label,
                    "n_hours": int(len(subset_frame)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _write_placeholder_bidding_metrics(path: Path) -> None:
    pd.DataFrame(
        {
            "strategy": [
                "S7_ridge_text_stochastic_LP",
                "S11_mlp_text_deterministic",
                "S22_ridge_text_llm_stochastic_LP",
            ],
            "total_revenue": [np.nan, np.nan, np.nan],
            "cvar_95_loss": [np.nan, np.nan, np.nan],
        }
    ).to_csv(path, index=False)


def run(
    processed_dir: Path,
    data_suffix: str,
    output_dir: Path,
    master_path: Path | None,
    train_end: str,
    eval_start: str,
    eval_end: str | None,
) -> None:
    os.environ.setdefault("NWS_LLM_FEATURE_MODE", "cache")
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    master, audit, written_master_path = _read_or_build_master(processed_dir, data_suffix, master_path)
    prediction_frames: list[pd.DataFrame] = []
    residual_frames: list[pd.DataFrame] = []
    meta_rows: list[dict[str, object]] = []

    for spec in MODEL_RUNS:
        preds, train_residuals, meta = fit_hybrid_blend_split_predictions(
            master=master,
            train_end=train_end,
            eval_start=eval_start,
            eval_end=eval_end,
            pv_model_name=str(spec["pv_model"]),
            rt_model_name=str(spec["rt_model"]),
            anchor_model_name=str(spec["anchor_model"]),
        )
        meta["config_label"] = spec["config_label"]
        prediction_frames.append(preds)
        residual_frames.append(train_residuals)
        meta_rows.append(meta)

    preds = prediction_frames[0]
    train_residuals = residual_frames[0]
    for extra_preds, extra_residuals in zip(prediction_frames[1:], residual_frames[1:]):
        preds = _merge_columns(preds, extra_preds)
        train_residuals = _merge_columns(train_residuals, extra_residuals)

    rated_capacity = float(meta_rows[0]["rated_capacity_mw"])
    audit.update(
        {
            "rated_capacity_mw": rated_capacity,
            "train_end": train_end,
            "eval_start": eval_start,
            "eval_end": eval_end or "",
            "master_path": str(written_master_path),
            "selected_workflow": "PV MLP rule-core/LLM cloud-rule plus RT Transformer rule-text/LLM rule",
        }
    )

    pd.DataFrame([audit]).to_csv(output_dir / "data_audit.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(output_dir / "selected_cloud_rule_prediction_meta.csv", index=False)
    preds.to_csv(output_dir / "test_predictions.csv", index=False)
    train_residuals.to_csv(output_dir / "train_residuals.csv", index=False)
    _build_forecast_metrics(preds, rated_capacity).to_csv(output_dir / "forecast_metrics.csv", index=False)
    _write_placeholder_bidding_metrics(output_dir / "bidding_metrics.csv")

    print(f"Wrote selected predictions to {output_dir}")
    print(f"Wrote or used master table at {written_master_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the retained neural selected-model predictions from public-data processed inputs."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data_multi_weather_2023_2024/processed"))
    parser.add_argument("--data-suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/selected_cloud_rule_downstream"))
    parser.add_argument("--master-path", type=Path, help="Optional prebuilt master hourly CSV.")
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--eval-start", default="2024-01-01")
    parser.add_argument("--eval-end", default=None)
    args = parser.parse_args()
    run(
        processed_dir=args.processed_dir,
        data_suffix=args.data_suffix,
        output_dir=args.output_dir,
        master_path=args.master_path,
        train_end=args.train_end,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
    )


if __name__ == "__main__":
    main()
