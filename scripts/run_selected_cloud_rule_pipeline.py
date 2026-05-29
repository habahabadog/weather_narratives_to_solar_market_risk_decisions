from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_baselines import (
    _feature_group_for_model,
    _fit_named_split_prediction,
    _fit_predict_torch_sequence,
    _make_named_tabular_model,
    _mlp_model,
    _sequence_epochs,
    _sequence_model_kind,
    build_master_table,
    feature_columns,
    make_train_test_masks,
    regression_metrics,
)


@dataclass(frozen=True)
class SelectedForecastSpec:
    model_name: str
    target_col: str
    prefix: str
    capped: bool


SELECTED_FORECAST_SPECS = [
    SelectedForecastSpec("mlp", "pv_mw", "pv", True),
    SelectedForecastSpec("mlp_rule_core", "pv_mw", "pv", True),
    SelectedForecastSpec("mlp_llm_rule_cloud", "pv_mw", "pv", True),
    SelectedForecastSpec("transformer_none", "rt_lmp", "rt", False),
    SelectedForecastSpec("transformer_text", "rt_lmp", "rt", False),
    SelectedForecastSpec("transformer_llm_rule", "rt_lmp", "rt", False),
]

SELECTED_SEQUENCE_OVERRIDES = {
    "transformer_llm_rule": {"seq_len": 48, "hidden_size": 32, "epochs": 6},
}


def _resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else ROOT / path


def _base_prediction_frame(test: pd.DataFrame) -> pd.DataFrame:
    pred_base_cols = [
        "timestamp_utc",
        "local_date",
        "hour",
        "is_solar_hour",
        "has_extreme_event",
        "event_types",
        "pv_mw",
        "rt_lmp",
        "da_lmp",
    ]
    pred_base_cols.extend([col for col in test.columns if col.startswith("wx_prior_")])
    pred_base_cols.extend([col for col in test.columns if col.startswith("llm_prior_")])
    return test[[col for col in pred_base_cols if col in test.columns]].copy()


def build_selected_forecast_metrics(preds: pd.DataFrame, rated_capacity: float) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    subsets = [
        ("all", preds),
        ("solar_hours", preds[preds["is_solar_hour"] == 1]),
        ("extreme", preds[preds["has_extreme_event"] == 1]),
        ("extreme_solar_hours", preds[(preds["has_extreme_event"] == 1) & (preds["is_solar_hour"] == 1)]),
    ]
    for spec in SELECTED_FORECAST_SPECS:
        pred_col = f"{spec.prefix}_{spec.model_name}"
        target = "pv" if spec.prefix == "pv" else "rt_price"
        capacity = rated_capacity if spec.target_col == "pv_mw" else None
        for subset_name, subset in subsets:
            metrics = regression_metrics(subset[spec.target_col], subset[pred_col], capacity)
            rows.append(
                {
                    "target": target,
                    "subset": subset_name,
                    "model": pred_col,
                    "n": int(len(subset)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _seeded_named_tabular_model(model_name: str, target_col: str, random_state: int) -> object:
    if model_name == "mlp_llm_rule_cloud":
        return _mlp_model(random_state=random_state, alpha=1e-3, learning_rate_init=5e-4, max_iter=340)
    if model_name == "mlp_rule_core":
        return _mlp_model(random_state=random_state, alpha=1e-3, learning_rate_init=1e-3)
    if model_name.startswith("mlp"):
        return _mlp_model(random_state=random_state)
    return _make_named_tabular_model(model_name, target_col)


def _fit_selected_split_prediction(
    model_df: pd.DataFrame,
    train_mask: pd.Series,
    eval_mask: pd.Series,
    model_name: str,
    target_col: str,
    rated_capacity: float | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if random_state is None:
        return _fit_named_split_prediction(
            model_df=model_df,
            train_mask=train_mask,
            eval_mask=eval_mask,
            model_name=model_name,
            target_col=target_col,
            rated_capacity=rated_capacity,
        )

    feature_group = _feature_group_for_model(model_name)
    train = model_df.loc[train_mask]
    feature_cols = [col for col in feature_columns(model_df, text_group=feature_group) if train[col].notna().any()]
    if not feature_cols:
        raise ValueError(f"no usable feature columns for {model_name}")

    sequence_kind = _sequence_model_kind(model_name)
    if sequence_kind is not None:
        sequence_kwargs = {
            "epochs": _sequence_epochs(sequence_kind, feature_group),
            **SELECTED_SEQUENCE_OVERRIDES.get(model_name, {}),
        }
        train_pred_full, eval_pred = _fit_predict_torch_sequence(
            model_df=model_df,
            train_mask=train_mask,
            test_mask=eval_mask,
            feature_cols=feature_cols,
            target_col=target_col,
            model_kind=sequence_kind,
            random_state=random_state,
            **sequence_kwargs,
        )
    else:
        model = _seeded_named_tabular_model(model_name, target_col, random_state=random_state)
        model.fit(train[feature_cols], train[target_col])
        train_pred_full = np.full(len(model_df), np.nan)
        train_pred_full[train_mask.to_numpy()] = model.predict(train[feature_cols])
        eval_pred = model.predict(model_df.loc[eval_mask, feature_cols])

    if target_col == "pv_mw":
        upper = rated_capacity if rated_capacity is not None else math.inf
        train_pred_full = np.clip(train_pred_full, 0.0, upper)
        eval_pred = np.clip(eval_pred, 0.0, upper)
    return train_pred_full, eval_pred


def fit_selected_cloud_rule_models(
    master: pd.DataFrame,
    output_dir: Path,
    *,
    train_end: str = "2024-01-01",
    test_start: str = "2024-01-01",
    test_end: str | None = "2026-01-01",
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_df = (
        master.dropna(subset=["pv_mw", "rt_lmp", "pv_mw_lag_24", "rt_lmp_lag_24"])
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    train_mask, test_mask = make_train_test_masks(
        model_df,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )
    if not train_mask.any() or not test_mask.any():
        raise ValueError("split must contain non-empty train and test rows")

    train = model_df.loc[train_mask].copy()
    test = model_df.loc[test_mask].copy()
    rated_capacity = float(model_df["pv_mw"].max())
    preds = _base_prediction_frame(test)
    train_residuals = train[["timestamp_utc", "local_date", "hour"]].copy()

    for spec in SELECTED_FORECAST_SPECS:
        pred_col = f"{spec.prefix}_{spec.model_name}"
        print(f"fit {pred_col}", flush=True)
        train_pred_full, test_pred = _fit_selected_split_prediction(
            model_df=model_df,
            train_mask=train_mask,
            eval_mask=test_mask,
            model_name=spec.model_name,
            target_col=spec.target_col,
            rated_capacity=rated_capacity if spec.capped else None,
            random_state=random_state,
        )
        train_pred = train_pred_full[train_mask.to_numpy()]
        preds[pred_col] = test_pred
        train_residuals[f"{pred_col}_residual"] = train[spec.target_col].to_numpy() - train_pred
        preds.to_csv(output_dir / "test_predictions.partial.csv", index=False)
        train_residuals.to_csv(output_dir / "train_residuals.partial.csv", index=False)

    metrics = build_selected_forecast_metrics(preds, rated_capacity)
    meta = {
        "rated_capacity_mw": rated_capacity,
        "numeric_feature_count": len(feature_columns(model_df, text_group="none")),
        "rule_core_feature_count": len(feature_columns(model_df, text_group="rule_core")),
        "llm_rule_cloud_feature_count": len(feature_columns(model_df, text_group="llm_rule_cloud")),
        "rule_text_feature_count": len(feature_columns(model_df, text_group="rule")),
        "llm_rule_feature_count": len(feature_columns(model_df, text_group="llm_rule")),
        "selected_cloud_rule_model_count": len(SELECTED_FORECAST_SPECS),
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end or "",
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
    }
    if random_state is not None:
        meta["random_state"] = int(random_state)
    return preds, metrics, train_residuals, meta


def write_selected_summary(path: Path, audit: dict[str, float | int | str], metrics: pd.DataFrame) -> None:
    all_metrics = metrics[metrics["subset"].eq("all")].copy()
    visible_cols = ["target", "model", "n", "mae", "rmse", "nrmse", "mape"]
    display = all_metrics[visible_cols].copy()
    for col in ["mae", "rmse", "nrmse", "mape"]:
        display[col] = pd.to_numeric(display[col], errors="coerce").map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    lines = [
        "# Selected Cloud-Rule Forecast Pipeline",
        "",
        "Scope: neural forecast models used in the manuscript.",
        "",
        "## Split",
        "",
        f"- Train end, exclusive: `{audit['train_end']}`",
        f"- Test window: `{audit['test_start']}` to `{audit['test_end'] or 'end of data'}`",
        f"- Train rows: {audit['train_rows']}",
        f"- Test rows: {audit['test_rows']}",
        "",
        "## All-Hour Forecast Metrics",
        "",
        display.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    processed_dir: Path,
    output_dir: Path,
    data_suffix: str | None = None,
    train_end: str = "2024-01-01",
    test_start: str = "2024-01-01",
    test_end: str | None = "2026-01-01",
    master_path: Path | None = None,
    master_output: Path | None = None,
    random_state: int | None = None,
) -> None:
    output_dir = _resolve_path(output_dir) or output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_master_path = _resolve_path(master_path)
    if resolved_master_path is not None:
        master = pd.read_csv(resolved_master_path)
        audit: dict[str, float | int | str] = {
            "data_suffix": data_suffix or resolved_master_path.stem.removeprefix("master_hourly_caiso_noaa_"),
            "master_path": str(resolved_master_path),
        }
    else:
        resolved_processed_dir = _resolve_path(processed_dir) or processed_dir
        master, audit = build_master_table(resolved_processed_dir, data_suffix=data_suffix)
        data_token = str(audit["data_suffix"]).removesuffix(".csv")
        resolved_master_output = _resolve_path(master_output) if master_output is not None else None
        target_master_path = resolved_master_output or resolved_processed_dir / f"master_hourly_caiso_noaa_{data_token}.csv"
        master.to_csv(target_master_path, index=False)
        audit["master_path"] = str(target_master_path)

    preds, metrics, train_residuals, meta = fit_selected_cloud_rule_models(
        master,
        output_dir,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        random_state=random_state,
    )
    audit.update(meta)
    pd.DataFrame([audit]).to_csv(output_dir / "data_audit.csv", index=False)
    preds.to_csv(output_dir / "test_predictions.csv", index=False)
    train_residuals.to_csv(output_dir / "train_residuals.csv", index=False)
    metrics.to_csv(output_dir / "forecast_metrics.csv", index=False)
    write_selected_summary(output_dir / "selected_cloud_rule_forecast_summary.md", audit, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the selected cloud-rule neural forecast pipeline.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data_multi_weather_2022_2025/processed"))
    parser.add_argument("--output", type=Path, default=Path("results_selected_cloud_rule_2022_2025_test_2024_2025"))
    parser.add_argument("--data-suffix", default="2022-01-01_2026-01-01")
    parser.add_argument("--train-end", default="2024-01-01", help="Final model training end date, exclusive.")
    parser.add_argument("--test-start", default="2024-01-01", help="Test start date, inclusive.")
    parser.add_argument("--test-end", default="2026-01-01", help="Optional test end date, exclusive.")
    parser.add_argument("--master-path", type=Path, help="Use an existing master hourly CSV instead of rebuilding it.")
    parser.add_argument("--master-output", type=Path, help="Optional path for the rebuilt master hourly CSV.")
    parser.add_argument("--random-state", type=int, help="Optional fixed seed for every selected model in this run.")
    args = parser.parse_args()
    run(
        args.processed_dir,
        args.output,
        data_suffix=args.data_suffix,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        master_path=args.master_path,
        master_output=args.master_output,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
