from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_baselines import (
    HRRR_FEATURE_COLUMNS,
    _fit_predict_torch_sequence,
    build_master_table,
    feature_columns,
    regression_metrics,
)


ModelFactory = Callable[[], object]

HRRR_SUBSETS = {
    "hrrr_cloud": ["tcdc_entire_atmosphere", "prate_surface"],
    "hrrr_solar": ["tcdc_entire_atmosphere", "prate_surface", "tmp_2m", "dpt_2m"],
    "hrrr_temp_cloud": ["tcdc_entire_atmosphere", "tmp_2m", "dpt_2m"],
    "hrrr_wind": ["ugrd_10m", "vgrd_10m", "gust_surface"],
}

RULE_SOLAR_COLUMNS = [
    "wx_prior_cloud_score",
    "wx_prior_rain_score",
    "wx_prior_fog_visibility_score",
    "wx_prior_heat_score",
    "wx_prior_fire_smoke_score",
]
RULE_CORE_COLUMNS = [
    "wx_prior_cloud_score",
    "wx_prior_rain_score",
    "wx_prior_fog_visibility_score",
]
LLM_RULE_SOLAR_COLUMNS = [
    "llm_prior_rule_cloud_score",
    "llm_prior_rule_rain_score",
    "llm_prior_rule_fog_visibility_score",
    "llm_prior_rule_heat_score",
    "llm_prior_rule_fire_smoke_score",
]
LLM_RULE_CORE_COLUMNS = [
    "llm_prior_rule_cloud_score",
    "llm_prior_rule_rain_score",
    "llm_prior_rule_fog_visibility_score",
]
LLM_CORE_SOLAR_COLUMNS = [
    "llm_prior_cloud_severity",
    "llm_prior_irradiance_reduction_risk",
    "llm_prior_rain_risk",
    "llm_prior_fog_visibility_risk",
    "llm_prior_heat_risk",
    "llm_prior_smoke_dust_risk",
]
LLM_CORE_CLOUD_COLUMNS = [
    "llm_prior_cloud_severity",
    "llm_prior_irradiance_reduction_risk",
]


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_csv(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_hidden_layer_grid(value: str) -> list[tuple[int, ...]]:
    specs: list[tuple[int, ...]] = []
    for raw_spec in value.replace(",", ";").split(";"):
        raw_spec = raw_spec.strip()
        if not raw_spec:
            continue
        layers = tuple(int(part.strip()) for part in raw_spec.replace("-", "x").split("x") if part.strip())
        if not layers:
            raise ValueError(f"empty hidden-layer spec: {raw_spec}")
        specs.append(layers)
    return specs


def parse_sequence_grid(value: str) -> list[tuple[int, int, int]]:
    specs: list[tuple[int, int, int]] = []
    for raw_spec in value.replace(",", ";").split(";"):
        raw_spec = raw_spec.strip()
        if not raw_spec:
            continue
        parts = [int(part.strip()) for part in raw_spec.replace("-", "x").split("x") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"sequence spec must be seq_len x hidden_size x epochs: {raw_spec}")
        specs.append((parts[0], parts[1], parts[2]))
    return specs


def _numeric_existing(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


def select_search_feature_columns(df: pd.DataFrame, feature_group: str, base_variant: str = "full") -> list[str]:
    base = feature_columns(df, text_group="none")
    if base_variant == "no_hrrr":
        base = [col for col in base if col not in HRRR_FEATURE_COLUMNS]
    elif base_variant in HRRR_SUBSETS:
        no_hrrr_base = [col for col in base if col not in HRRR_FEATURE_COLUMNS]
        base = no_hrrr_base + HRRR_SUBSETS[base_variant]
    elif base_variant == "market_lags":
        keep_prefixes = (
            "hour",
            "month",
            "dayofweek",
            "is_weekend",
            "is_solar_hour",
            "pv_mw_lag",
            "rt_lmp_lag",
            "da_lmp_lag",
            "demand_da_forecast_mw_lag",
        )
        base = [col for col in base if col.startswith(keep_prefixes)]
    elif base_variant != "full":
        raise ValueError(f"unknown base_variant: {base_variant}")

    if feature_group == "none":
        text_cols: list[str] = []
    elif feature_group == "rule":
        text_cols = [col for col in df.columns if col.startswith("wx_prior_")]
    elif feature_group == "rule_solar":
        text_cols = RULE_SOLAR_COLUMNS
    elif feature_group == "rule_core":
        text_cols = RULE_CORE_COLUMNS
    elif feature_group == "rule_cloud":
        text_cols = ["wx_prior_cloud_score"]
    elif feature_group == "llm_core":
        text_cols = [col for col in df.columns if col.startswith("llm_prior_") and not col.startswith("llm_prior_rule_")]
    elif feature_group == "llm_core_solar":
        text_cols = LLM_CORE_SOLAR_COLUMNS
    elif feature_group == "llm_core_cloud":
        text_cols = LLM_CORE_CLOUD_COLUMNS
    elif feature_group == "llm_rule":
        text_cols = [col for col in df.columns if col.startswith("llm_prior_rule_")]
    elif feature_group == "llm_rule_solar":
        text_cols = LLM_RULE_SOLAR_COLUMNS
    elif feature_group == "llm_rule_core":
        text_cols = LLM_RULE_CORE_COLUMNS
    elif feature_group == "llm_rule_cloud":
        text_cols = ["llm_prior_rule_cloud_score"]
    elif feature_group == "llm_all":
        text_cols = [col for col in df.columns if col.startswith("llm_prior_")]
    elif feature_group == "all_text":
        text_cols = [col for col in df.columns if col.startswith(("wx_prior_", "llm_prior_"))]
    else:
        raise ValueError(f"unknown feature_group: {feature_group}")
    return _numeric_existing(df, base + text_cols)


def _ridge_factory(alpha: float) -> ModelFactory:
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def _gbr_factory(n_estimators: int, learning_rate: float, max_depth: int, random_state: int) -> ModelFactory:
    return lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        ),
    )


def _mlp_factory(
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    batch_size: int,
    random_state: int,
    max_iter: int,
) -> ModelFactory:
    return lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=18,
            random_state=random_state,
        ),
    )


def _fit_predict_torch_tabular(
    model_df: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    feature_cols: list[str],
    target_col: str,
    hidden_layer_sizes: tuple[int, ...],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    dropout: float,
    random_state: int,
) -> np.ndarray:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc

    torch.manual_seed(random_state)
    torch.set_num_threads(1)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_features_raw = model_df.loc[train_mask, feature_cols]
    imputer.fit(train_features_raw)
    scaler.fit(imputer.transform(train_features_raw))
    features = scaler.transform(imputer.transform(model_df[feature_cols])).astype(np.float32)

    target_values = model_df[target_col].to_numpy(dtype=np.float32)
    target_train = target_values[train_mask.to_numpy()]
    target_mean = float(np.nanmean(target_train))
    target_std = float(np.nanstd(target_train))
    if not np.isfinite(target_std) or target_std < 1e-6:
        target_std = 1.0
    scaled_targets = ((target_values - target_mean) / target_std).astype(np.float32)

    train_x = features[train_mask.to_numpy()]
    train_y = scaled_targets[train_mask.to_numpy()]
    pred_x = features[test_mask.to_numpy()]
    if len(train_x) == 0 or len(pred_x) == 0:
        return np.full(int(test_mask.sum()), np.nan)

    layers: list[nn.Module] = []
    input_size = train_x.shape[1]
    for hidden_size in hidden_layer_sizes:
        layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU(), nn.LayerNorm(hidden_size)])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        input_size = hidden_size
    layers.append(nn.Linear(input_size, 1))

    device = torch.device("cpu")
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x).squeeze(-1), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pred_x), 512):
            batch = torch.from_numpy(pred_x[start : start + 512]).to(device)
            preds.append(model(batch).squeeze(-1).cpu().numpy())
    return np.concatenate(preds) * target_std + target_mean


def candidate_factories(
    model_families: list[str],
    *,
    random_state: int,
    random_state_offsets: list[int] | None = None,
    max_mlp_iter: int,
    quick: bool,
    sequence_epochs: int = 12,
    mlp_hidden_layer_sizes: list[tuple[int, ...]] | None = None,
    mlp_alphas: list[float] | None = None,
    mlp_learning_rates: list[float] | None = None,
    mlp_batch_sizes: list[int] | None = None,
    ridge_alphas: list[float] | None = None,
    sequence_specs: list[tuple[int, int, int]] | None = None,
    torch_epochs: list[int] | None = None,
    torch_dropouts: list[float] | None = None,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    random_states = [random_state + offset for offset in (random_state_offsets or [0])]
    if "ridge" in model_families:
        alpha_grid = ridge_alphas or ([1.0, 10.0, 100.0] if quick else [0.1, 1.0, 10.0, 100.0])
        for alpha in alpha_grid:
            candidates.append(
                {
                    "model_type": "tabular",
                    "model_family": "ridge",
                    "model_spec": {"alpha": alpha},
                    "make_model": _ridge_factory(alpha),
                }
            )
    if "gbr" in model_families:
        grid = [(150, 0.05, 2), (250, 0.05, 3)] if quick else [(150, 0.03, 2), (250, 0.03, 2), (250, 0.05, 3)]
        for seed in random_states:
            for n_estimators, learning_rate, max_depth in grid:
                candidates.append(
                    {
                        "model_type": "tabular",
                        "model_family": "gbr",
                        "model_spec": {
                            "n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "max_depth": max_depth,
                            "random_state": seed,
                        },
                        "make_model": _gbr_factory(n_estimators, learning_rate, max_depth, random_state=seed),
                    }
                )
    if "mlp" in model_families:
        if mlp_hidden_layer_sizes or mlp_alphas or mlp_learning_rates or mlp_batch_sizes:
            hidden_grid = mlp_hidden_layer_sizes or [(64,), (64, 32)]
            alpha_grid = mlp_alphas or [1e-4]
            learning_rate_grid = mlp_learning_rates or [8e-4]
            batch_size_grid = mlp_batch_sizes or [256]
            grid = [
                (hidden_layer_sizes, alpha, learning_rate_init, batch_size)
                for hidden_layer_sizes in hidden_grid
                for alpha in alpha_grid
                for learning_rate_init in learning_rate_grid
                for batch_size in batch_size_grid
            ]
        else:
            default_grid = [((64,), 1e-4, 8e-4), ((64, 32), 1e-4, 8e-4)] if quick else [
                ((32,), 1e-4, 8e-4),
                ((64,), 1e-4, 8e-4),
                ((64, 32), 1e-4, 8e-4),
                ((128, 64), 5e-5, 8e-4),
                ((64, 32), 1e-3, 5e-4),
                ((128,), 1e-4, 1e-3),
            ]
            grid = [(hidden_layer_sizes, alpha, learning_rate_init, 256) for hidden_layer_sizes, alpha, learning_rate_init in default_grid]
        for seed in random_states:
            for hidden_layer_sizes, alpha, learning_rate_init, batch_size in grid:
                candidates.append(
                    {
                        "model_type": "tabular",
                        "model_family": "mlp",
                        "model_spec": {
                            "hidden_layer_sizes": hidden_layer_sizes,
                            "alpha": alpha,
                            "learning_rate_init": learning_rate_init,
                            "batch_size": batch_size,
                            "max_iter": max_mlp_iter,
                            "random_state": seed,
                        },
                        "make_model": _mlp_factory(
                            hidden_layer_sizes=hidden_layer_sizes,
                            alpha=alpha,
                            learning_rate_init=learning_rate_init,
                            batch_size=batch_size,
                            random_state=seed,
                            max_iter=max_mlp_iter,
                        ),
                    }
                )
    if "torch_mlp" in model_families:
        hidden_grid = mlp_hidden_layer_sizes or ([(64,), (64, 32)] if quick else [(32,), (64,), (64, 32), (128,), (128, 64)])
        weight_decay_grid = mlp_alphas or ([1e-4] if quick else [1e-5, 1e-4, 1e-3])
        learning_rate_grid = mlp_learning_rates or ([1e-3] if quick else [3e-4, 5e-4, 1e-3])
        batch_size_grid = mlp_batch_sizes or [256]
        epoch_grid = torch_epochs or ([30] if quick else [40, 80])
        dropout_grid = torch_dropouts or ([0.0] if quick else [0.0, 0.1])
        for seed in random_states:
            for hidden_layer_sizes in hidden_grid:
                for weight_decay in weight_decay_grid:
                    for learning_rate in learning_rate_grid:
                        for batch_size in batch_size_grid:
                            for epochs in epoch_grid:
                                for dropout in dropout_grid:
                                    candidates.append(
                                        {
                                            "model_type": "torch_tabular",
                                            "model_family": "torch_mlp",
                                            "model_spec": {
                                                "hidden_layer_sizes": hidden_layer_sizes,
                                                "learning_rate": learning_rate,
                                                "weight_decay": weight_decay,
                                                "batch_size": batch_size,
                                                "epochs": epochs,
                                                "dropout": dropout,
                                                "random_state": seed,
                                            },
                                        }
                                    )
    for model_kind in ("lstm", "gru", "transformer", "cnn"):
        if model_kind not in model_families:
            continue
        grid = sequence_specs or (
            [(24, 32, sequence_epochs)]
            if quick
            else [(24, 32, sequence_epochs), (48, 32, sequence_epochs), (24, 64, sequence_epochs)]
        )
        for seed in random_states:
            sequence_seed = seed + {"lstm": 1000, "gru": 2000, "transformer": 3000, "cnn": 4000}[model_kind]
            for seq_len, hidden_size, epochs in grid:
                candidates.append(
                    {
                        "model_type": "sequence",
                        "model_family": model_kind,
                        "model_spec": {
                            "seq_len": seq_len,
                            "hidden_size": hidden_size,
                            "epochs": epochs,
                            "random_state": sequence_seed,
                        },
                        "sequence_kind": model_kind,
                        "random_state": sequence_seed,
                    }
                )
    return candidates


def _subset_mask(frame: pd.DataFrame, subset: str) -> pd.Series:
    if subset == "all":
        return pd.Series(True, index=frame.index)
    if subset == "solar_hours":
        return frame["is_solar_hour"].astype(int).eq(1)
    if subset == "extreme":
        return frame["has_extreme_event"].astype(int).eq(1)
    if subset == "extreme_solar_hours":
        return frame["has_extreme_event"].astype(int).eq(1) & frame["is_solar_hour"].astype(int).eq(1)
    raise ValueError(f"unknown subset: {subset}")


def _metric_value(actual: pd.Series, pred: np.ndarray, metric: str, rated_capacity: float | None) -> float:
    metrics = regression_metrics(actual, pred, rated_capacity=rated_capacity)
    return float(metrics[metric])


def _candidate_name(model_family: str, feature_group: str, base_variant: str, model_spec: dict[str, object]) -> str:
    spec = ",".join(f"{key}={value}" for key, value in model_spec.items())
    return f"{model_family}|base={base_variant}|features={feature_group}|{spec}"


def evaluate_search(
    master: pd.DataFrame,
    *,
    validation_start: str,
    test_start: str,
    targets: list[str],
    feature_groups: list[str],
    base_variants: list[str],
    model_families: list[str],
    subsets: list[str],
    metric: str,
    quick: bool,
    max_mlp_iter: int,
    random_state_offsets: list[int] | None = None,
    sequence_epochs: int = 12,
    mlp_hidden_layer_sizes: list[tuple[int, ...]] | None = None,
    mlp_alphas: list[float] | None = None,
    mlp_learning_rates: list[float] | None = None,
    mlp_batch_sizes: list[int] | None = None,
    ridge_alphas: list[float] | None = None,
    sequence_specs: list[tuple[int, int, int]] | None = None,
    torch_epochs: list[int] | None = None,
    torch_dropouts: list[float] | None = None,
) -> pd.DataFrame:
    model_df = master.dropna(subset=["pv_mw_lag_24", "rt_lmp_lag_24"]).sort_values("timestamp_utc").reset_index(drop=True)
    dates = model_df["local_date"].astype(str)
    fit_mask = dates < validation_start
    validation_mask = (dates >= validation_start) & (dates < test_start)
    train_mask = dates < test_start
    test_mask = dates >= test_start
    if not fit_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("fit, validation, and test splits must all be non-empty")

    target_map = {
        "pv": ("pv_mw", float(model_df["pv_mw"].max()), 23),
        "rt_price": ("rt_lmp", None, 29),
    }
    rows: list[dict[str, object]] = []

    for target in targets:
        target_col, rated_capacity, random_state = target_map[target]
        factories = candidate_factories(
            model_families,
            random_state=random_state,
            random_state_offsets=random_state_offsets,
            max_mlp_iter=max_mlp_iter,
            quick=quick,
            sequence_epochs=sequence_epochs,
            mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
            mlp_alphas=mlp_alphas,
            mlp_learning_rates=mlp_learning_rates,
            mlp_batch_sizes=mlp_batch_sizes,
            ridge_alphas=ridge_alphas,
            sequence_specs=sequence_specs,
            torch_epochs=torch_epochs,
            torch_dropouts=torch_dropouts,
        )
        for base_variant in base_variants:
            for feature_group in feature_groups:
                feature_cols = select_search_feature_columns(model_df, feature_group, base_variant=base_variant)
                feature_cols = [col for col in feature_cols if model_df.loc[fit_mask, col].notna().any()]
                if not feature_cols:
                    continue
                for candidate in factories:
                    model_family = str(candidate["model_family"])
                    model_spec = dict(candidate["model_spec"])
                    candidate_name = _candidate_name(model_family, feature_group, base_variant, model_spec)
                    if candidate.get("model_type") == "sequence":
                        _, validation_pred = _fit_predict_torch_sequence(
                            model_df=model_df,
                            train_mask=fit_mask,
                            test_mask=validation_mask,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            model_kind=str(candidate["sequence_kind"]),
                            random_state=int(candidate["random_state"]),
                            seq_len=int(model_spec["seq_len"]),
                            hidden_size=int(model_spec["hidden_size"]),
                            epochs=int(model_spec["epochs"]),
                        )
                        _, test_pred = _fit_predict_torch_sequence(
                            model_df=model_df,
                            train_mask=train_mask,
                            test_mask=test_mask,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            model_kind=str(candidate["sequence_kind"]),
                            random_state=int(candidate["random_state"]),
                            seq_len=int(model_spec["seq_len"]),
                            hidden_size=int(model_spec["hidden_size"]),
                            epochs=int(model_spec["epochs"]),
                        )
                    elif candidate.get("model_type") == "torch_tabular":
                        validation_pred = _fit_predict_torch_tabular(
                            model_df=model_df,
                            train_mask=fit_mask,
                            test_mask=validation_mask,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            hidden_layer_sizes=tuple(model_spec["hidden_layer_sizes"]),
                            learning_rate=float(model_spec["learning_rate"]),
                            weight_decay=float(model_spec["weight_decay"]),
                            batch_size=int(model_spec["batch_size"]),
                            epochs=int(model_spec["epochs"]),
                            dropout=float(model_spec["dropout"]),
                            random_state=int(model_spec["random_state"]),
                        )
                        test_pred = _fit_predict_torch_tabular(
                            model_df=model_df,
                            train_mask=train_mask,
                            test_mask=test_mask,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            hidden_layer_sizes=tuple(model_spec["hidden_layer_sizes"]),
                            learning_rate=float(model_spec["learning_rate"]),
                            weight_decay=float(model_spec["weight_decay"]),
                            batch_size=int(model_spec["batch_size"]),
                            epochs=int(model_spec["epochs"]),
                            dropout=float(model_spec["dropout"]),
                            random_state=int(model_spec["random_state"]),
                        )
                    else:
                        make_model = candidate["make_model"]
                        validation_model = make_model()
                        validation_model.fit(model_df.loc[fit_mask, feature_cols], model_df.loc[fit_mask, target_col])
                        validation_pred = np.asarray(
                            validation_model.predict(model_df.loc[validation_mask, feature_cols]),
                            dtype=float,
                        )

                        final_model = make_model()
                        final_model.fit(model_df.loc[train_mask, feature_cols], model_df.loc[train_mask, target_col])
                        test_pred = np.asarray(final_model.predict(model_df.loc[test_mask, feature_cols]), dtype=float)
                    if target == "pv":
                        validation_pred = np.clip(validation_pred, 0.0, rated_capacity)
                        test_pred = np.clip(test_pred, 0.0, rated_capacity)

                    validation_frame = model_df.loc[validation_mask].copy()
                    test_frame = model_df.loc[test_mask].copy()
                    for subset in subsets:
                        val_subset = _subset_mask(validation_frame, subset)
                        test_subset = _subset_mask(test_frame, subset)
                        if not val_subset.any() or not test_subset.any():
                            continue
                        rows.append(
                            {
                                "target": target,
                                "target_col": target_col,
                                "subset": subset,
                                "metric": metric,
                                "model_family": model_family,
                                "feature_group": feature_group,
                                "base_variant": base_variant,
                                "candidate_name": candidate_name,
                                "model_spec": json.dumps(model_spec, sort_keys=True),
                                "feature_count": len(feature_cols),
                                "validation_metric": _metric_value(
                                    validation_frame.loc[val_subset, target_col],
                                    validation_pred[val_subset.to_numpy()],
                                    metric,
                                    rated_capacity,
                                ),
                                "test_metric": _metric_value(
                                    test_frame.loc[test_subset, target_col],
                                    test_pred[test_subset.to_numpy()],
                                    metric,
                                    rated_capacity,
                                ),
                                "validation_n": int(val_subset.sum()),
                                "test_n": int(test_subset.sum()),
                            }
                        )
    return pd.DataFrame(rows)


def _best_by_validation(frame: pd.DataFrame) -> pd.Series | None:
    finite = frame[np.isfinite(pd.to_numeric(frame["validation_metric"], errors="coerce"))].copy()
    if finite.empty:
        return None
    return finite.sort_values(["validation_metric", "test_metric"], ascending=[True, True]).iloc[0]


def _is_llm_feature_group(value: object) -> bool:
    name = str(value)
    return name == "all_text" or name.startswith("llm_")


def summarize_selected_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    results = results.copy()
    if "base_variant" not in results.columns:
        results["base_variant"] = "full"
    rows: list[dict[str, object]] = []
    group_cols = ["target", "subset", "metric", "model_family", "base_variant"]
    for keys, group in results.groupby(group_cols, sort=True):
        none = _best_by_validation(group[group["feature_group"].astype(str).eq("none")])
        llm = _best_by_validation(group[group["feature_group"].map(_is_llm_feature_group)])
        if none is None or llm is None:
            continue
        none_test = float(none["test_metric"])
        llm_test = float(llm["test_metric"])
        improvement = (none_test - llm_test) / abs(none_test) * 100.0 if abs(none_test) > 1e-12 else math.nan
        rows.append(
            {
                "target": keys[0],
                "subset": keys[1],
                "metric": keys[2],
                "model_family": keys[3],
                "base_variant": keys[4],
                "selected_none_candidate": none["candidate_name"],
                "selected_none_validation_metric": float(none["validation_metric"]),
                "selected_none_test_metric": none_test,
                "selected_llm_candidate": llm["candidate_name"],
                "selected_llm_feature_group": llm["feature_group"],
                "selected_llm_validation_metric": float(llm["validation_metric"]),
                "selected_llm_test_metric": llm_test,
                "test_improvement_pct": improvement,
                "llm_wins_on_test": bool(improvement > 0),
            }
        )
    return pd.DataFrame(rows)


def write_summary(output_dir: Path, comparisons: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    wins = comparisons[comparisons["llm_wins_on_test"].astype(bool)].copy() if not comparisons.empty else pd.DataFrame()
    lines = [
        "# Validation-selected text feature search",
        "",
        "Selection rule: choose candidates by the validation metric only, then report the locked 2024 test metric.",
        "",
    ]
    if wins.empty:
        lines.append("No validation-selected LLM/text candidate beat the matched no-text candidate on the locked test metric.")
    else:
        visible = [
            "target",
            "subset",
            "model_family",
            "base_variant",
            "selected_llm_feature_group",
            "selected_none_test_metric",
            "selected_llm_test_metric",
            "test_improvement_pct",
        ]
        lines.extend(["## LLM wins on locked test", "", wins[visible].to_markdown(index=False), ""])
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.master_path is not None:
        master = pd.read_csv(args.master_path)
    else:
        master, _ = build_master_table(args.processed_dir, data_suffix=args.data_suffix)
    results = evaluate_search(
        master,
        validation_start=args.validation_start,
        test_start=args.test_start,
        targets=_parse_csv(args.targets),
        feature_groups=_parse_csv(args.feature_groups),
        base_variants=_parse_csv(args.base_variants),
        model_families=_parse_csv(args.model_families),
        subsets=_parse_csv(args.subsets),
        metric=args.metric,
        quick=args.quick,
        max_mlp_iter=args.max_mlp_iter,
        random_state_offsets=_parse_int_csv(args.random_state_offsets),
        sequence_epochs=args.sequence_epochs,
        mlp_hidden_layer_sizes=parse_hidden_layer_grid(args.mlp_hidden_layers),
        mlp_alphas=_parse_float_csv(args.mlp_alphas),
        mlp_learning_rates=_parse_float_csv(args.mlp_learning_rates),
        mlp_batch_sizes=_parse_int_csv(args.mlp_batch_sizes),
        ridge_alphas=_parse_float_csv(args.ridge_alphas),
        sequence_specs=parse_sequence_grid(args.sequence_grid),
        torch_epochs=_parse_int_csv(args.torch_epochs),
        torch_dropouts=_parse_float_csv(args.torch_dropouts),
    )
    comparisons = summarize_selected_comparisons(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "candidate_results.csv", index=False)
    comparisons.to_csv(args.output_dir / "selected_comparisons.csv", index=False)
    if not comparisons.empty:
        comparisons[comparisons["llm_wins_on_test"].astype(bool)].to_csv(args.output_dir / "llm_wins.csv", index=False)
    write_summary(args.output_dir, comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-selected search for LLM text-feature forecast settings.")
    parser.add_argument("--master-path", type=Path, help="Optional prebuilt master hourly CSV to avoid rebuilding inputs.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data_multi_weather_2023_2024/processed"))
    parser.add_argument("--data-suffix", default="2023-01-01_2025-01-01")
    parser.add_argument("--output-dir", type=Path, default=Path("results_validation_text_feature_search"))
    parser.add_argument("--validation-start", default="2023-10-01")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--targets", default="pv,rt_price")
    parser.add_argument("--feature-groups", default="none,rule,llm_core,llm_rule,llm_all,all_text")
    parser.add_argument("--base-variants", default="full,no_hrrr")
    parser.add_argument("--model-families", default="mlp,cnn,gru,lstm,transformer")
    parser.add_argument("--subsets", default="all,solar_hours,extreme,extreme_solar_hours")
    parser.add_argument("--metric", default="rmse", choices=["rmse", "mae"])
    parser.add_argument("--max-mlp-iter", type=int, default=220)
    parser.add_argument("--random-state-offsets", default="0")
    parser.add_argument("--sequence-epochs", type=int, default=12)
    parser.add_argument("--mlp-hidden-layers", default="", help="Semicolon-separated layer specs, e.g. 32;64x32;128x64.")
    parser.add_argument("--mlp-alphas", default="", help="Comma-separated MLP L2 penalties.")
    parser.add_argument("--mlp-learning-rates", default="", help="Comma-separated MLP initial learning rates.")
    parser.add_argument("--mlp-batch-sizes", default="", help="Comma-separated MLP batch sizes.")
    parser.add_argument("--ridge-alphas", default="", help="Comma-separated Ridge L2 penalties.")
    parser.add_argument("--sequence-grid", default="", help="Semicolon-separated seq_len x hidden_size x epochs specs, e.g. 24x32x8;48x64x12.")
    parser.add_argument("--torch-epochs", default="", help="Comma-separated epoch counts for torch_mlp candidates.")
    parser.add_argument("--torch-dropouts", default="", help="Comma-separated dropout rates for torch_mlp candidates.")
    parser.add_argument("--quick", action="store_true", help="Use a small grid for smoke tests.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
