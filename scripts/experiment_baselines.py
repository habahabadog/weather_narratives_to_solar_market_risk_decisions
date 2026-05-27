from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PACIFIC_TZ = "America/Los_Angeles"
WEATHER_SCORE_COLUMNS = [
    "wx_cloud_score",
    "wx_storm_score",
    "wx_rain_score",
    "wx_wind_score",
    "wx_fog_visibility_score",
    "wx_heat_score",
    "wx_fire_smoke_score",
]
LLM_WEATHER_RISK_COLUMNS = [
    "llm_cloud_severity",
    "llm_irradiance_reduction_risk",
    "llm_storm_risk",
    "llm_rain_risk",
    "llm_fog_visibility_risk",
    "llm_wind_risk",
    "llm_heat_risk",
    "llm_smoke_dust_risk",
]
LLM_WEATHER_TIME_COLUMNS = [
    "llm_morning_risk",
    "llm_afternoon_risk",
    "llm_evening_risk",
    "llm_all_day_risk",
]
LLM_RULE_EQUIVALENT_COLUMNS = [
    "llm_rule_cloud_score",
    "llm_rule_storm_score",
    "llm_rule_rain_score",
    "llm_rule_wind_score",
    "llm_rule_fog_visibility_score",
    "llm_rule_heat_score",
    "llm_rule_fire_smoke_score",
]
LLM_WEATHER_NUMERIC_COLUMNS = [
    *LLM_WEATHER_RISK_COLUMNS,
    *LLM_WEATHER_TIME_COLUMNS,
    *LLM_RULE_EQUIVALENT_COLUMNS,
    "llm_confidence",
    "llm_overall_risk_score",
    "llm_text_product_count",
]
LLM_WEATHER_OUTPUT_COLUMNS = [
    "issue_local_date",
    *LLM_WEATHER_NUMERIC_COLUMNS,
    "llm_affected_hours",
    "llm_short_reason",
    "llm_feature_source",
]
HRRR_FEATURE_COLUMNS = [
    "tmp_2m",
    "dpt_2m",
    "ugrd_10m",
    "vgrd_10m",
    "gust_surface",
    "tcdc_entire_atmosphere",
    "prate_surface",
]
WEATHER_KEYWORDS = {
    "wx_cloud_score": [
        "cloud",
        "cloudy",
        "overcast",
        "stratus",
        "stratocu",
        "marine layer",
        "low clouds",
        "mostly cloudy",
        "partly cloudy",
    ],
    "wx_storm_score": [
        "thunderstorm",
        "thunderstorms",
        "convective",
        "convection",
        "lightning",
        "monsoon",
        "instability",
        "severe",
    ],
    "wx_rain_score": [
        "rain",
        "rainfall",
        "showers",
        "precipitation",
        "drizzle",
        "atmospheric river",
        "snow",
    ],
    "wx_wind_score": [
        "wind",
        "winds",
        "gust",
        "gusty",
        "advisory",
        "warning",
        "santa ana",
        "offshore",
    ],
    "wx_fog_visibility_score": [
        "fog",
        "dense fog",
        "visibility",
        "haze",
        "mist",
        "low visibility",
    ],
    "wx_heat_score": [
        "heat",
        "hot",
        "excessive heat",
        "heat advisory",
        "very warm",
        "triple digits",
    ],
    "wx_fire_smoke_score": [
        "smoke",
        "wildfire",
        "fire weather",
        "red flag",
        "ash",
        "dust",
        "blowing dust",
    ],
}
MONTH_NUMBERS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def regression_metrics(actual: Iterable[float], pred: Iterable[float], rated_capacity: float | None = None) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not mask.any():
        return {"mae": math.nan, "rmse": math.nan, "nrmse": math.nan, "mape": math.nan}
    err = yhat[mask] - y[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    nonzero = np.abs(y[mask]) > 1e-6
    mape = float(np.mean(np.abs(err[nonzero] / y[mask][nonzero]))) if nonzero.any() else math.nan
    if rated_capacity is None or rated_capacity <= 0:
        nrmse = math.nan
    else:
        nrmse = float(rmse / rated_capacity)
    return {"mae": mae, "rmse": rmse, "nrmse": nrmse, "mape": mape}


def settlement_revenue(
    actual_pv: Iterable[float],
    bid_q: Iterable[float],
    da_lmp: Iterable[float],
    rt_lmp: Iterable[float],
    shortage_penalty: float = 50.0,
    surplus_penalty: float = 50.0,
) -> np.ndarray:
    actual = np.asarray(actual_pv, dtype=float)
    bid = np.asarray(bid_q, dtype=float)
    da = np.asarray(da_lmp, dtype=float)
    rt = np.asarray(rt_lmp, dtype=float)
    surplus = np.maximum(actual - bid, 0.0)
    shortage = np.maximum(bid - actual, 0.0)
    return da * bid + rt * (actual - bid) - surplus_penalty * surplus - shortage_penalty * shortage


def quantile_residual_bid(
    preds: pd.DataFrame,
    train_residuals: pd.DataFrame,
    *,
    pred_col: str,
    residual_col: str,
    quantile: float,
    rated_capacity: float,
) -> np.ndarray:
    residuals = pd.to_numeric(train_residuals[residual_col], errors="coerce")
    by_hour = (
        train_residuals.assign(_residual=residuals)
        .dropna(subset=["hour", "_residual"])
        .groupby("hour")["_residual"]
        .quantile(quantile)
    )
    finite_residuals = residuals[np.isfinite(residuals)]
    if not by_hour.empty:
        fallback = float(by_hour.median())
    elif not finite_residuals.empty:
        fallback = float(finite_residuals.quantile(quantile))
    else:
        fallback = 0.0
    shifts = preds["hour"].map(by_hour).fillna(fallback).to_numpy(dtype=float)
    point_forecast = pd.to_numeric(preds[pred_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.clip(point_forecast + shifts, 0.0, rated_capacity)


def _run_linprog(**kwargs):
    try:
        return linprog(method="highs", **kwargs)
    except (TypeError, ValueError):
        return linprog(**kwargs)


def solve_cvar_bids(
    pv_scenarios: np.ndarray,
    rt_scenarios: np.ndarray,
    da_lmp: np.ndarray,
    rated_capacity: float,
    scenario_prob: np.ndarray | None = None,
    gamma: float = 0.0,
    alpha: float = 0.95,
    shortage_penalty: float = 50.0,
    surplus_penalty: float = 50.0,
) -> np.ndarray:
    pv = np.asarray(pv_scenarios, dtype=float)
    rt = np.asarray(rt_scenarios, dtype=float)
    da = np.asarray(da_lmp, dtype=float)
    if pv.ndim != 2 or rt.shape != pv.shape:
        raise ValueError("pv_scenarios and rt_scenarios must have the same S x H shape")
    scenario_count, horizon = pv.shape
    if da.shape != (horizon,):
        raise ValueError("da_lmp must have one value per horizon interval")
    if scenario_prob is None:
        prob = np.full(scenario_count, 1.0 / scenario_count)
    else:
        prob = np.asarray(scenario_prob, dtype=float)
        prob = prob / prob.sum()

    q_offset = 0
    up_offset = horizon
    um_offset = up_offset + scenario_count * horizon
    use_cvar = gamma > 0.0
    eta_index = um_offset + scenario_count * horizon if use_cvar else None
    xi_offset = eta_index + 1 if use_cvar else None
    variable_count = (xi_offset + scenario_count) if use_cvar else (um_offset + scenario_count * horizon)

    def up_idx(scenario: int, hour: int) -> int:
        return up_offset + scenario * horizon + hour

    def um_idx(scenario: int, hour: int) -> int:
        return um_offset + scenario * horizon + hour

    c = np.zeros(variable_count)
    c[q_offset : q_offset + horizon] = -da
    for scenario in range(scenario_count):
        for hour in range(horizon):
            c[up_idx(scenario, hour)] = -prob[scenario] * (rt[scenario, hour] - surplus_penalty)
            c[um_idx(scenario, hour)] = prob[scenario] * (rt[scenario, hour] + shortage_penalty)
    if use_cvar:
        c[eta_index] = gamma
        c[xi_offset : xi_offset + scenario_count] = gamma * prob / (1.0 - alpha)

    a_eq = []
    b_eq = []
    for scenario in range(scenario_count):
        for hour in range(horizon):
            row = np.zeros(variable_count)
            row[q_offset + hour] = 1.0
            row[up_idx(scenario, hour)] = 1.0
            row[um_idx(scenario, hour)] = -1.0
            a_eq.append(row)
            b_eq.append(pv[scenario, hour])

    a_ub = []
    b_ub = []
    if use_cvar:
        for scenario in range(scenario_count):
            row = np.zeros(variable_count)
            row[q_offset : q_offset + horizon] = -da
            for hour in range(horizon):
                row[up_idx(scenario, hour)] = surplus_penalty - rt[scenario, hour]
                row[um_idx(scenario, hour)] = rt[scenario, hour] + shortage_penalty
            row[eta_index] = -1.0
            row[xi_offset + scenario] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)

    bounds = [(0.0, rated_capacity)] * horizon
    bounds.extend([(0.0, None)] * (2 * scenario_count * horizon))
    if use_cvar:
        bounds.append((None, None))
        bounds.extend([(0.0, None)] * scenario_count)

    result = _run_linprog(
        c=c,
        A_ub=np.asarray(a_ub) if a_ub else None,
        b_ub=np.asarray(b_ub) if b_ub else None,
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
    )
    if not result.success:
        raise RuntimeError(f"CVaR bidding LP failed: {result.message}")
    return np.asarray(result.x[q_offset : q_offset + horizon])


def cvar_loss(losses: Iterable[float], alpha: float = 0.95) -> float:
    values = np.asarray(losses, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    tail_count = max(1, int(math.ceil((1.0 - alpha) * len(values))))
    return float(np.mean(np.sort(values)[-tail_count:]))


def interval_coverage(actual: Iterable[float], lower: Iterable[float], upper: Iterable[float]) -> float:
    y = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not mask.any():
        return math.nan
    return float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])))


def percent_improvement(baseline_metric: float, candidate_metric: float) -> float:
    if not np.isfinite(baseline_metric) or abs(baseline_metric) < 1e-12:
        return math.nan
    return float((baseline_metric - candidate_metric) / baseline_metric * 100.0)


def apply_residual_correction(
    base_prediction: Iterable[float],
    residual_prediction: Iterable[float],
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    corrected = np.asarray(base_prediction, dtype=float) + np.asarray(residual_prediction, dtype=float)
    if lower is not None or upper is not None:
        corrected = np.clip(
            corrected,
            -np.inf if lower is None else lower,
            np.inf if upper is None else upper,
        )
    return corrected


def make_sequence_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    eligible_rows: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    row_indices = []
    for idx in range(seq_len - 1, len(features)):
        if not eligible_rows[idx] or not np.isfinite(targets[idx]):
            continue
        window = features[idx - seq_len + 1 : idx + 1]
        if not np.isfinite(window).all():
            continue
        x_rows.append(window)
        y_rows.append(targets[idx])
        row_indices.append(idx)
    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.float32),
        np.asarray(row_indices, dtype=int),
    )


def _keyword_count_lowered(lowered_text: str, keyword: str) -> int:
    pattern = r"(?<![A-Za-z])" + re.escape(keyword.lower()) + r"(?![A-Za-z])"
    return len(re.findall(pattern, lowered_text))


def score_weather_text(text: str) -> dict[str, float]:
    lowered = text.lower()
    scores: dict[str, float] = {}
    for col, keywords in WEATHER_KEYWORDS.items():
        count = sum(_keyword_count_lowered(lowered, keyword) for keyword in keywords)
        scores[col] = min(1.0, count / 5.0)
    return scores


def _clip01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return float(min(1.0, max(0.0, numeric)))


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def normalize_llm_weather_payload(
    payload: dict[str, Any],
    text_product_count: int = 1,
    source: str = "openai_structured_output",
) -> dict[str, float | str]:
    normalized: dict[str, float | str] = {}
    for col in LLM_WEATHER_RISK_COLUMNS + LLM_WEATHER_TIME_COLUMNS + LLM_RULE_EQUIVALENT_COLUMNS + ["llm_confidence"]:
        normalized[col] = _clip01(payload.get(col.removeprefix("llm_"), payload.get(col, 0.0)))
    normalized["llm_overall_risk_score"] = max(
        float(normalized[col]) for col in LLM_WEATHER_RISK_COLUMNS + LLM_WEATHER_TIME_COLUMNS + LLM_RULE_EQUIVALENT_COLUMNS
    )
    normalized["llm_text_product_count"] = float(max(0, int(text_product_count)))
    affected = str(payload.get("affected_hours", payload.get("llm_affected_hours", "all_day"))).strip().lower()
    if affected not in {"morning", "afternoon", "evening", "all_day"}:
        affected = "all_day"
    normalized["llm_affected_hours"] = affected
    normalized["llm_short_reason"] = str(payload.get("short_reason", payload.get("llm_short_reason", "")))[:240]
    normalized["llm_feature_source"] = source
    return normalized


def resolve_llm_api_config(mode: str, provider: str | None = None, model: str | None = None) -> dict[str, str]:
    normalized_mode = mode.strip().lower()
    normalized_provider = (provider or "").strip().lower()
    if not normalized_provider:
        normalized_provider = "deepseek" if "deepseek" in normalized_mode else "openai"

    if normalized_provider == "deepseek":
        selected_model = (model or "").strip() or "deepseek-v4-pro"
        source_model = re.sub(r"[^a-z0-9]+", "_", selected_model.lower()).strip("_")
        return {
            "provider": "deepseek",
            "model": selected_model,
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
            "source": f"{source_model}_json_output",
        }
    if normalized_provider == "openai":
        return {
            "provider": "openai",
            "model": (model or "").strip() or "gpt-4o-mini",
            "base_url": "",
            "api_key_env": "OPENAI_API_KEY",
            "source": "openai_structured_output",
        }
    raise ValueError(f"unsupported NWS_LLM_PROVIDER: {provider}")


def build_llm_weather_messages(text: str) -> list[dict[str, str]]:
    example = {
        "cloud_severity": 0.0,
        "irradiance_reduction_risk": 0.0,
        "storm_risk": 0.0,
        "rain_risk": 0.0,
        "fog_visibility_risk": 0.0,
        "wind_risk": 0.0,
        "heat_risk": 0.0,
        "smoke_dust_risk": 0.0,
        "morning_risk": 0.0,
        "afternoon_risk": 0.0,
        "evening_risk": 0.0,
        "all_day_risk": 0.0,
        "rule_cloud_score": 0.0,
        "rule_storm_score": 0.0,
        "rule_rain_score": 0.0,
        "rule_wind_score": 0.0,
        "rule_fog_visibility_score": 0.0,
        "rule_heat_score": 0.0,
        "rule_fire_smoke_score": 0.0,
        "affected_hours": "all_day",
        "confidence": 0.0,
        "short_reason": "brief reason",
    }
    system_prompt = (
        "You extract photovoltaic-relevant weather risk features from NWS Area Forecast Discussion text. "
        "Return one valid json object only. Scores must be numbers in [0,1]. Use low values for clear, dry, "
        "low-impact weather. Use high irradiance_reduction_risk for clouds, smoke, fog, precipitation, "
        "or storm language that can reduce solar generation. affected_hours must be one of morning, "
        "afternoon, evening, all_day. Also return rule_* scores that emulate a careful keyword/rule-based "
        "weather-text extractor: cloud, storm, rain, wind, fog/visibility, heat, and fire/smoke. These rule_* "
        "fields should preserve explicit mentions even when the overall photovoltaic impact is low. "
        "Example json output:\n"
        f"{json.dumps(example, indent=2)}"
    )
    user_prompt = (
        "Extract the json weather-risk object for CAISO solar forecasting from this NWS discussion:\n\n"
        f"{compact_nws_text_for_llm(text)}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def compact_nws_text_for_llm(text: str, max_chars: int = 2500) -> str:
    wanted = ("SYNOPSIS", "SHORT TERM", "LONG TERM", "HAZARD POTENTIAL OUTLOOK", "DISCUSSION")
    stop_prefixes = ("AVIATION", "MARINE", "BEACHES", "LOX WATCHES", "FIRE WEATHER")
    lines = []
    keep = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        section = line.strip(".").upper()
        if any(section.startswith(prefix) for prefix in wanted):
            keep = True
        elif any(section.startswith(prefix) for prefix in stop_prefixes):
            keep = False
        if keep and line:
            lines.append(line)
    compact = "\n".join(lines).strip()
    if not compact:
        compact = text.strip()
    return compact[:max_chars]


def parse_llm_weather_json(content: str, source: str) -> dict[str, float | str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("LLM JSON response must be an object")
    return normalize_llm_weather_payload(payload, source=source)


def extract_llm_weather_features(text: str) -> dict[str, float | str]:
    rule_scores = score_weather_text(text)
    cloud = rule_scores["wx_cloud_score"]
    storm = rule_scores["wx_storm_score"]
    rain = rule_scores["wx_rain_score"]
    wind = rule_scores["wx_wind_score"]
    fog = rule_scores["wx_fog_visibility_score"]
    heat = rule_scores["wx_heat_score"]
    smoke = rule_scores["wx_fire_smoke_score"]
    irradiance = max(cloud * 0.95, rain * 0.85, fog * 0.75, smoke * 0.70, storm * 0.60)
    morning_terms = ("morning", "am ", " a.m.", "sunrise", "marine layer", "low clouds", "fog")
    afternoon_terms = ("afternoon", "pm ", " p.m.", "heating", "convection", "thunderstorm", "hot")
    evening_terms = ("evening", "tonight", "overnight", "after sunset", "late day")

    morning_risk = max(cloud, fog, rain * 0.6) if _contains_any(text, morning_terms) else 0.0
    afternoon_risk = max(storm, heat, wind, smoke, rain * 0.7) if _contains_any(text, afternoon_terms) else 0.0
    evening_risk = max(wind, rain, storm, fog * 0.7) if _contains_any(text, evening_terms) else 0.0
    all_day_risk = max(cloud, rain, smoke, wind) * 0.7 if _contains_any(text, ("through the day", "all day", "period")) else 0.0
    time_risks = {
        "morning": morning_risk,
        "afternoon": afternoon_risk,
        "evening": evening_risk,
        "all_day": all_day_risk,
    }
    affected_hours = max(time_risks, key=time_risks.get)
    if time_risks[affected_hours] <= 0:
        affected_hours = "all_day"
    overall = max(cloud, irradiance, storm, rain, fog, wind, heat, smoke, *time_risks.values())
    confidence = min(1.0, 0.45 + 0.25 * overall + 0.05 * sum(score > 0 for score in rule_scores.values()))
    payload = {
        "cloud_severity": cloud,
        "irradiance_reduction_risk": irradiance,
        "storm_risk": storm,
        "rain_risk": rain,
        "fog_visibility_risk": fog,
        "wind_risk": wind,
        "heat_risk": heat,
        "smoke_dust_risk": smoke,
        "morning_risk": morning_risk,
        "afternoon_risk": afternoon_risk,
        "evening_risk": evening_risk,
        "all_day_risk": all_day_risk,
        "rule_cloud_score": cloud,
        "rule_storm_score": storm,
        "rule_rain_score": rain,
        "rule_wind_score": wind,
        "rule_fog_visibility_score": fog,
        "rule_heat_score": heat,
        "rule_fire_smoke_score": smoke,
        "confidence": confidence,
        "affected_hours": affected_hours,
        "short_reason": "structured proxy from weather-risk language",
    }
    return normalize_llm_weather_payload(payload, source="heuristic_structured_proxy")


def extract_openai_llm_weather_features(text: str, model: str) -> dict[str, float | str]:
    try:
        from openai import OpenAI
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("OpenAI structured weather extraction requires openai and pydantic") from exc

    class WeatherNarrativeRisk(BaseModel):
        cloud_severity: float = Field(ge=0.0, le=1.0)
        irradiance_reduction_risk: float = Field(ge=0.0, le=1.0)
        storm_risk: float = Field(ge=0.0, le=1.0)
        rain_risk: float = Field(ge=0.0, le=1.0)
        fog_visibility_risk: float = Field(ge=0.0, le=1.0)
        wind_risk: float = Field(ge=0.0, le=1.0)
        heat_risk: float = Field(ge=0.0, le=1.0)
        smoke_dust_risk: float = Field(ge=0.0, le=1.0)
        morning_risk: float = Field(ge=0.0, le=1.0)
        afternoon_risk: float = Field(ge=0.0, le=1.0)
        evening_risk: float = Field(ge=0.0, le=1.0)
        all_day_risk: float = Field(ge=0.0, le=1.0)
        rule_cloud_score: float = Field(ge=0.0, le=1.0)
        rule_storm_score: float = Field(ge=0.0, le=1.0)
        rule_rain_score: float = Field(ge=0.0, le=1.0)
        rule_wind_score: float = Field(ge=0.0, le=1.0)
        rule_fog_visibility_score: float = Field(ge=0.0, le=1.0)
        rule_heat_score: float = Field(ge=0.0, le=1.0)
        rule_fire_smoke_score: float = Field(ge=0.0, le=1.0)
        affected_hours: str
        confidence: float = Field(ge=0.0, le=1.0)
        short_reason: str

    prompt = (
        "Extract photovoltaic-relevant weather risk features from this NWS Area Forecast Discussion. "
        "Return risk scores in [0,1]. Focus on cloud cover, irradiance reduction, precipitation, "
        "fog/visibility, wind, heat, smoke/dust, and likely affected daylight periods. "
        "Use low values when the discussion says dry, clear, or low impact. Also return rule_* scores "
        "that emulate a careful keyword/rule-based weather-text extractor for cloud, storm, rain, wind, "
        "fog/visibility, heat, and fire/smoke, preserving explicit mentions even when overall PV impact is low.\n\n"
        f"{text[:12000]}"
    )
    client = OpenAI()
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You extract structured weather-risk features for CAISO solar forecasting.",
            },
            {"role": "user", "content": prompt},
        ],
        text_format=WeatherNarrativeRisk,
    )
    for output in response.output:
        if output.type != "message":
            continue
        for item in output.content:
            if getattr(item, "type", None) == "refusal":
                raise RuntimeError(f"OpenAI refusal while extracting weather features: {item.refusal}")
            parsed = getattr(item, "parsed", None)
            if parsed is not None:
                payload = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed.dict()
                return normalize_llm_weather_payload(payload, source="openai_structured_output")
    raise RuntimeError("OpenAI response did not contain parsed structured weather features")


def extract_chat_json_llm_weather_features(text: str, config: dict[str, str]) -> dict[str, float | str]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Chat JSON weather extraction requires the openai package") from exc

    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise RuntimeError(f"{config['api_key_env']} is required for {config['provider']} weather extraction")
    client_kwargs = {"api_key": api_key}
    if config["base_url"]:
        client_kwargs["base_url"] = config["base_url"]
    client = OpenAI(timeout=60.0, max_retries=1, **client_kwargs)
    messages = build_llm_weather_messages(text)
    last_error: Exception | None = None
    for _ in range(3):
        response = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1600,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content or ""
        if not content.strip():
            last_error = RuntimeError("LLM returned empty JSON content")
            continue
        try:
            return parse_llm_weather_json(content, source=config["source"])
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError("LLM JSON extraction failed after retries") from last_error


def extract_api_llm_weather_features(text: str, config: dict[str, str]) -> dict[str, float | str]:
    if config["provider"] == "openai":
        return extract_openai_llm_weather_features(text, model=config["model"])
    return extract_chat_json_llm_weather_features(text, config=config)


def _parse_nws_issue_local_date(product_text: str) -> str | None:
    match = re.search(
        r"^\d{3,4}\s+[AP]M\s+[A-Z]{3}\s+\w{3}\s+([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{4})\s*$",
        product_text,
        flags=re.MULTILINE,
    )
    if not match:
        return None
    month_name, day, year = match.groups()
    month = MONTH_NUMBERS.get(month_name)
    if month is None:
        return None
    return f"{int(year):04d}-{month:02d}-{int(day):02d}"


def load_nws_weather_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["issue_local_date", "wx_text_product_count", *WEATHER_SCORE_COLUMNS])

    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    for product in _split_nws_products(text):
        issue_date = _parse_nws_issue_local_date(product)
        if issue_date is None:
            continue
        rows.append({"issue_local_date": issue_date, **score_weather_text(product)})

    if not rows:
        return pd.DataFrame(columns=["issue_local_date", "wx_text_product_count", *WEATHER_SCORE_COLUMNS])

    products_df = pd.DataFrame(rows)
    daily = products_df.groupby("issue_local_date", as_index=False)[WEATHER_SCORE_COLUMNS].max()
    daily["wx_text_product_count"] = products_df.groupby("issue_local_date").size().reindex(daily["issue_local_date"]).to_numpy()
    daily["wx_overall_risk_score"] = daily[WEATHER_SCORE_COLUMNS].max(axis=1)
    return daily


def load_nws_weather_features_from_paths(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [load_nws_weather_features(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["issue_local_date", "wx_text_product_count", *WEATHER_SCORE_COLUMNS])

    combined = pd.concat(frames, ignore_index=True)
    for col in WEATHER_SCORE_COLUMNS:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)
    combined["wx_text_product_count"] = pd.to_numeric(
        combined["wx_text_product_count"], errors="coerce"
    ).fillna(0).astype(int)

    daily = combined.groupby("issue_local_date", as_index=False)[WEATHER_SCORE_COLUMNS].max()
    counts = combined.groupby("issue_local_date")["wx_text_product_count"].sum()
    daily["wx_text_product_count"] = counts.reindex(daily["issue_local_date"]).fillna(0).astype(int).to_numpy()
    daily["wx_overall_risk_score"] = daily[WEATHER_SCORE_COLUMNS].max(axis=1)
    return daily


def _empty_llm_weather_features() -> pd.DataFrame:
    return pd.DataFrame(columns=LLM_WEATHER_OUTPUT_COLUMNS)


def load_nws_llm_weather_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_llm_weather_features()
    try:
        daily = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return _empty_llm_weather_features()
    for col in LLM_WEATHER_OUTPUT_COLUMNS:
        if col not in daily.columns:
            daily[col] = "" if col in {"issue_local_date", "llm_affected_hours", "llm_short_reason", "llm_feature_source"} else 0.0
    for col in LLM_WEATHER_NUMERIC_COLUMNS:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)
        if col != "llm_text_product_count":
            daily[col] = daily[col].clip(lower=0.0, upper=1.0)
    return daily[LLM_WEATHER_OUTPUT_COLUMNS].copy()


def _split_nws_products(text: str) -> list[str]:
    return [part for part in re.split(r"(?=^FXUS\d{2}\s+K[A-Z]{3}\s+\d{6})", text, flags=re.MULTILINE) if part.strip()]


def build_nws_llm_weather_features(
    nws_path: Path,
    mode: str = "heuristic",
    model: str = "gpt-4o-mini",
    api_config: dict[str, str] | None = None,
) -> pd.DataFrame:
    if not nws_path.exists():
        return _empty_llm_weather_features()

    text = nws_path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    for product in _split_nws_products(text):
        issue_date = _parse_nws_issue_local_date(product)
        if issue_date is None:
            continue
        rows.append({"issue_local_date": issue_date, "product_text": product})
    if not rows:
        return _empty_llm_weather_features()

    products_df = pd.DataFrame(rows)
    daily_rows = []
    use_api = mode in {"api", "openai", "deepseek", "refresh_openai", "refresh_deepseek"}
    for issue_date, group in products_df.groupby("issue_local_date", sort=True):
        combined_text = "\n\n".join(group["product_text"].astype(str).tail(3))
        if use_api:
            config = api_config or resolve_llm_api_config(mode=mode, model=model)
            features = extract_api_llm_weather_features(combined_text, config=config)
        else:
            features = extract_llm_weather_features(combined_text)
        normalized = normalize_llm_weather_payload(
            features,
            text_product_count=int(len(group)),
            source=str(features.get("llm_feature_source", "heuristic_structured_proxy")),
        )
        daily_rows.append({"issue_local_date": issue_date, **normalized})

    daily = pd.DataFrame(daily_rows)
    return load_nws_llm_weather_features_from_frame(daily)


def load_nws_llm_weather_features_from_frame(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return _empty_llm_weather_features()
    out = daily.copy()
    for col in LLM_WEATHER_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col in {"issue_local_date", "llm_affected_hours", "llm_short_reason", "llm_feature_source"} else 0.0
    for col in LLM_WEATHER_NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        if col != "llm_text_product_count":
            out[col] = out[col].clip(lower=0.0, upper=1.0)
    return out[LLM_WEATHER_OUTPUT_COLUMNS].copy()


def load_or_build_nws_llm_weather_features(nws_path: Path, cache_path: Path) -> pd.DataFrame:
    mode = os.environ.get("NWS_LLM_FEATURE_MODE", "cache_or_heuristic").strip().lower()
    model = os.environ.get("NWS_LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    cache_modes = {"cache", "cache_or_heuristic", "cache_or_openai", "cache_or_deepseek"}
    if cache_path.exists() and mode in cache_modes:
        return load_nws_llm_weather_features(cache_path)
    if mode == "cache":
        return _empty_llm_weather_features()

    api_modes = {"openai", "deepseek", "cache_or_openai", "cache_or_deepseek", "refresh_openai", "refresh_deepseek"}
    if mode in api_modes:
        config = resolve_llm_api_config(
            mode=mode,
            provider=os.environ.get("NWS_LLM_PROVIDER", ""),
            model=os.environ.get("NWS_LLM_MODEL", ""),
        )
        daily = build_nws_llm_weather_features(nws_path, mode="api", model=config["model"], api_config=config)
    else:
        daily = build_nws_llm_weather_features(nws_path, mode="heuristic", model=model)
    if not daily.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(cache_path, index=False)
    return daily


def aggregate_nws_llm_weather_features(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return _empty_llm_weather_features()

    combined = pd.concat(frames, ignore_index=True)
    for col in LLM_WEATHER_NUMERIC_COLUMNS:
        combined[col] = pd.to_numeric(combined.get(col, 0.0), errors="coerce").fillna(0.0)

    aggregations: dict[str, str] = {
        col: "max" for col in LLM_WEATHER_NUMERIC_COLUMNS if col not in {"llm_text_product_count", "llm_confidence"}
    }
    aggregations["llm_text_product_count"] = "sum"
    aggregations["llm_confidence"] = "mean"
    daily = combined.groupby("issue_local_date", as_index=False).agg(aggregations)

    for col in ["llm_affected_hours", "llm_short_reason", "llm_feature_source"]:
        if col in combined:
            text_values = (
                combined.groupby("issue_local_date")[col]
                .apply(lambda values: "; ".join(sorted({str(value) for value in values if str(value).strip()})))
                .reindex(daily["issue_local_date"])
                .fillna("")
                .to_numpy()
            )
            daily[col] = text_values
    return load_nws_llm_weather_features_from_frame(daily)


def load_or_build_multi_nws_llm_weather_features(nws_paths: Iterable[Path], cache_path: Path) -> pd.DataFrame:
    paths = [path for path in nws_paths if path.exists()]
    if not paths:
        return _empty_llm_weather_features()
    if len(paths) == 1:
        return load_or_build_nws_llm_weather_features(paths[0], cache_path)

    mode = os.environ.get("NWS_LLM_FEATURE_MODE", "cache_or_heuristic").strip().lower()
    model = os.environ.get("NWS_LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    cache_modes = {"cache", "cache_or_heuristic", "cache_or_openai", "cache_or_deepseek"}
    if cache_path.exists() and mode in cache_modes:
        return load_nws_llm_weather_features(cache_path)
    if mode == "cache":
        return _empty_llm_weather_features()

    api_modes = {"openai", "deepseek", "cache_or_openai", "cache_or_deepseek", "refresh_openai", "refresh_deepseek"}
    api_config = None
    build_mode = "heuristic"
    if mode in api_modes:
        api_config = resolve_llm_api_config(
            mode=mode,
            provider=os.environ.get("NWS_LLM_PROVIDER", ""),
            model=os.environ.get("NWS_LLM_MODEL", ""),
        )
        build_mode = "api"

    daily = aggregate_nws_llm_weather_features(
        build_nws_llm_weather_features(path, mode=build_mode, model=model, api_config=api_config)
        for path in paths
    )
    if not daily.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(cache_path, index=False)
    return daily


def make_prior_weather_features(daily_features: pd.DataFrame) -> pd.DataFrame:
    if daily_features.empty:
        return pd.DataFrame(columns=["local_date"])

    shifted = daily_features.copy()
    shifted["local_date"] = (
        pd.to_datetime(shifted["issue_local_date"], errors="coerce") + timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")
    shifted = shifted.dropna(subset=["local_date"]).drop(columns=["issue_local_date"])
    rename = {col: f"wx_prior_{col.removeprefix('wx_')}" for col in shifted.columns if col != "local_date"}
    return shifted.rename(columns=rename)


def make_prior_llm_weather_features(daily_features: pd.DataFrame) -> pd.DataFrame:
    if daily_features.empty:
        return pd.DataFrame(columns=["local_date"])

    shifted = daily_features.copy()
    shifted["local_date"] = (
        pd.to_datetime(shifted["issue_local_date"], errors="coerce") + timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")
    shifted = shifted.dropna(subset=["local_date"])
    numeric_cols = [
        col
        for col in shifted.columns
        if col.startswith("llm_") and col in LLM_WEATHER_NUMERIC_COLUMNS
    ]
    out = shifted[["local_date", *numeric_cols]].copy()
    rename = {col: f"llm_prior_{col.removeprefix('llm_')}" for col in numeric_cols}
    return out.rename(columns=rename)


def add_lag_features(df: pd.DataFrame, columns: list[str], lags: tuple[int, ...] = (24, 168)) -> pd.DataFrame:
    out = df.sort_values("timestamp_utc").reset_index(drop=True).copy()
    for col in columns:
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
    return out


def _timestamp_hour(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.floor("h")


def _local_date(timestamp_utc: pd.Series) -> pd.Series:
    return timestamp_utc.dt.tz_convert(PACIFIC_TZ).dt.date.astype(str)


def _local_hour(timestamp_utc: pd.Series) -> pd.Series:
    return timestamp_utc.dt.tz_convert(PACIFIC_TZ).dt.hour


def load_actual_solar(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["interval_start_utc", "Solar"])
    df["timestamp_utc"] = _timestamp_hour(df["interval_start_utc"])
    df["Solar"] = pd.to_numeric(df["Solar"], errors="coerce").clip(lower=0.0)
    return df.groupby("timestamp_utc", as_index=False)["Solar"].mean().rename(columns={"Solar": "pv_mw"})


def load_demand(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["interval_start_utc", "Day ahead forecast", "Hour ahead forecast", "Current demand"],
    )
    df["timestamp_utc"] = _timestamp_hour(df["interval_start_utc"])
    for col in ["Day ahead forecast", "Hour ahead forecast", "Current demand"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = df.groupby("timestamp_utc", as_index=False)[
        ["Day ahead forecast", "Hour ahead forecast", "Current demand"]
    ].mean()
    return grouped.rename(
        columns={
            "Day ahead forecast": "demand_da_forecast_mw",
            "Hour ahead forecast": "demand_ha_forecast_mw",
            "Current demand": "demand_mw",
        }
    )


def load_net_demand(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["interval_start_utc", "Net demand", "Net demand forecast"])
    df["timestamp_utc"] = _timestamp_hour(df["interval_start_utc"])
    for col in ["Net demand", "Net demand forecast"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = df.groupby("timestamp_utc", as_index=False)[["Net demand", "Net demand forecast"]].mean()
    return grouped.rename(columns={"Net demand": "net_demand_mw", "Net demand forecast": "net_demand_forecast_mw"})


def load_caiso_solar_forecast(
    path: Path,
    output_col: str = "caiso_solar_dam_forecast_mw",
) -> pd.DataFrame:
    paths = sorted(path.rglob("*.csv")) if path.is_dir() else [path]
    frames = [
        pd.read_csv(
            item,
            usecols=lambda col: col in {"INTERVALSTARTTIME_GMT", "RENEWABLE_TYPE", "TRADING_HUB", "MW"},
        )
        for item in paths
    ]
    if not frames:
        return pd.DataFrame(columns=["timestamp_utc", output_col])
    df = pd.concat(frames, ignore_index=True)
    if "RENEWABLE_TYPE" in df.columns:
        df = df[df["RENEWABLE_TYPE"].astype(str).str.lower() == "solar"].copy()
    df["interval_start_utc"] = pd.to_datetime(df["INTERVALSTARTTIME_GMT"], utc=True)
    df["timestamp_utc"] = df["interval_start_utc"].dt.floor("h")
    df["MW"] = pd.to_numeric(df["MW"], errors="coerce").clip(lower=0.0)
    if "TRADING_HUB" in df.columns:
        by_interval_hub = df.groupby(["interval_start_utc", "TRADING_HUB"], as_index=False)["MW"].mean()
        by_interval = by_interval_hub.groupby("interval_start_utc", as_index=False)["MW"].sum()
    else:
        by_interval = df.groupby("interval_start_utc", as_index=False)["MW"].mean()
    by_interval["timestamp_utc"] = by_interval["interval_start_utc"].dt.floor("h")
    return by_interval.groupby("timestamp_utc", as_index=False)["MW"].mean().rename(columns={"MW": output_col})


def load_da_lmp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["INTERVALSTARTTIME_GMT", "LMP_TYPE", "MW"])
    df = df[df["LMP_TYPE"] == "LMP"].copy()
    df["timestamp_utc"] = _timestamp_hour(df["INTERVALSTARTTIME_GMT"])
    df["MW"] = pd.to_numeric(df["MW"], errors="coerce")
    return df.groupby("timestamp_utc", as_index=False)["MW"].mean().rename(columns={"MW": "da_lmp"})


def load_rt_lmp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["INTERVALSTARTTIME_GMT", "LMP_TYPE", "VALUE"])
    df = df[df["LMP_TYPE"] == "LMP"].copy()
    df["timestamp_utc"] = _timestamp_hour(df["INTERVALSTARTTIME_GMT"])
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    return df.groupby("timestamp_utc", as_index=False)["VALUE"].mean().rename(columns={"VALUE": "rt_lmp"})


def load_hrrr(path: Path, variable_names: Iterable[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["valid_time_utc", "variable_name", "value", "status"])
    df = df[df["status"] == "ok"].copy()
    keep = set(HRRR_FEATURE_COLUMNS if variable_names is None else variable_names)
    df = df[df["variable_name"].isin(keep)].copy()
    df["timestamp_utc"] = _timestamp_hour(df["valid_time_utc"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    grouped = df.groupby(["timestamp_utc", "variable_name"], as_index=False)["value"].mean()
    pivot = grouped.pivot(index="timestamp_utc", columns="variable_name", values="value").reset_index()
    pivot.columns.name = None
    return pivot


def load_extreme_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["has_extreme_event"] = pd.to_numeric(df["has_extreme_event"], errors="coerce").fillna(0).astype(int)
    return df.rename(columns={"date": "local_date"})


def normalize_data_suffix(value: str) -> str:
    return value if value.endswith(".csv") else f"{value}.csv"


def infer_processed_suffix(processed_dir: Path) -> str:
    prefix = "caiso_todays_outlook_fuelsource_"
    suffixes = sorted(
        path.name.removeprefix(prefix)
        for path in processed_dir.glob(f"{prefix}*.csv")
        if path.name.startswith(prefix)
    )
    if not suffixes:
        raise FileNotFoundError(f"No {prefix}*.csv file found in {processed_dir}")
    preferred = "2023-01-01_2025-01-01.csv"
    if preferred in suffixes:
        return preferred
    if len(suffixes) == 1:
        return suffixes[0]
    raise ValueError(f"Multiple data suffixes found in {processed_dir}: {', '.join(suffixes)}")


def nws_text_paths_for_suffix(processed_dir: Path, text_suffix: str) -> list[Path]:
    nws_dir = processed_dir.parent / "raw" / "nws_text"
    paths = sorted(nws_dir.glob(f"AFD*_{text_suffix}"))
    if paths:
        return paths
    return [nws_dir / f"AFDLOX_{text_suffix}"]


def build_master_table(processed_dir: Path, data_suffix: str | None = None) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    suffix = normalize_data_suffix(data_suffix) if data_suffix else infer_processed_suffix(processed_dir)
    text_suffix = suffix.removesuffix(".csv") + ".txt"
    master = load_actual_solar(processed_dir / f"caiso_todays_outlook_fuelsource_{suffix}")
    caiso_solar_forecast_path = processed_dir.parent / "raw" / "caiso_oasis" / "solar_dam"
    if not caiso_solar_forecast_path.exists():
        caiso_solar_forecast_path = processed_dir / f"caiso_solar_dam_SP15_{suffix}"
    for table in [
        load_demand(processed_dir / f"caiso_todays_outlook_demand_{suffix}"),
        load_net_demand(processed_dir / f"caiso_todays_outlook_netdemand_{suffix}"),
        load_caiso_solar_forecast(caiso_solar_forecast_path),
        load_da_lmp(processed_dir / f"caiso_lmp_da_TH_SP15_GEN-APND_{suffix}"),
        load_rt_lmp(processed_dir / f"caiso_lmp_rt_5min_TH_SP15_GEN-APND_{suffix}"),
        load_hrrr(processed_dir / f"hrrr_zarr_point_{suffix}"),
    ]:
        master = master.merge(table, on="timestamp_utc", how="left")

    master["local_date"] = _local_date(master["timestamp_utc"])
    master["hour"] = _local_hour(master["timestamp_utc"])
    master = master.merge(load_extreme_labels(processed_dir / f"extreme_weather_labels_{suffix}"), on="local_date", how="left")
    nws_paths = nws_text_paths_for_suffix(processed_dir, text_suffix)
    weather_features = make_prior_weather_features(load_nws_weather_features_from_paths(nws_paths))
    master = master.merge(weather_features, on="local_date", how="left")
    wx_prior_cols = [col for col in master.columns if col.startswith("wx_prior_")]
    if wx_prior_cols:
        master[wx_prior_cols] = master[wx_prior_cols].fillna(0.0)
    llm_cache_prefix = "nws_llm_weather_features_multi_nws" if len([path for path in nws_paths if path.exists()]) > 1 else "nws_llm_weather_features"
    llm_daily_path = processed_dir / f"{llm_cache_prefix}_{suffix}"
    llm_daily = load_or_build_multi_nws_llm_weather_features(nws_paths, llm_daily_path)
    llm_weather_features = make_prior_llm_weather_features(llm_daily)
    master = master.merge(llm_weather_features, on="local_date", how="left")
    llm_prior_cols = [col for col in master.columns if col.startswith("llm_prior_")]
    if llm_prior_cols:
        master[llm_prior_cols] = master[llm_prior_cols].fillna(0.0)
    master["has_extreme_event"] = master["has_extreme_event"].fillna(0).astype(int)
    master["event_types"] = master["event_types"].fillna("")
    master["month"] = master["timestamp_utc"].dt.tz_convert(PACIFIC_TZ).dt.month
    master["dayofweek"] = master["timestamp_utc"].dt.tz_convert(PACIFIC_TZ).dt.dayofweek
    master["is_weekend"] = (master["dayofweek"] >= 5).astype(int)
    master["is_solar_hour"] = master["hour"].between(6, 19).astype(int)

    master = add_lag_features(master, ["pv_mw", "rt_lmp", "da_lmp", "demand_da_forecast_mw"], lags=(24, 168))
    master = master.sort_values("timestamp_utc").reset_index(drop=True)

    required = ["pv_mw", "da_lmp", "rt_lmp"]
    audit = {
        "data_suffix": suffix,
        "rows_before_required_drop": int(len(master)),
        "price_start_utc": str(master.loc[master["da_lmp"].notna() & master["rt_lmp"].notna(), "timestamp_utc"].min()),
        "price_end_utc": str(master.loc[master["da_lmp"].notna() & master["rt_lmp"].notna(), "timestamp_utc"].max()),
        "da_lmp_missing_frac": float(master["da_lmp"].isna().mean()),
        "rt_lmp_missing_frac": float(master["rt_lmp"].isna().mean()),
        "caiso_solar_dam_forecast_missing_frac": float(master["caiso_solar_dam_forecast_mw"].isna().mean())
        if "caiso_solar_dam_forecast_mw" in master
        else math.nan,
        "hrrr_missing_frac_mean": float(master[[col for col in HRRR_FEATURE_COLUMNS if col in master]].isna().mean().mean()),
        "nws_text_files": ";".join(path.name for path in nws_paths if path.exists()),
        "nws_office_count": int(len([path for path in nws_paths if path.exists()])),
        "nws_text_feature_days": int(weather_features["local_date"].nunique()) if "local_date" in weather_features else 0,
        "nws_llm_feature_days": int(llm_weather_features["local_date"].nunique())
        if "local_date" in llm_weather_features
        else 0,
        "nws_llm_feature_source": ";".join(sorted(set(llm_daily["llm_feature_source"].dropna().astype(str))))
        if "llm_feature_source" in llm_daily and not llm_daily.empty
        else "",
    }
    master = master.dropna(subset=required).copy()
    audit["rows_after_required_drop"] = int(len(master))
    audit["extreme_hours_after_drop"] = int(master["has_extreme_event"].sum())
    audit["rated_capacity_p99_5"] = float(master["pv_mw"].quantile(0.995))
    audit["rated_capacity_max"] = float(master["pv_mw"].max())
    return master, audit


def feature_columns(df: pd.DataFrame, text_group: str = "rule") -> list[str]:
    base = [
        "hour",
        "month",
        "dayofweek",
        "is_weekend",
        "is_solar_hour",
        "demand_da_forecast_mw",
        "net_demand_forecast_mw",
        "pv_mw_lag_24",
        "pv_mw_lag_168",
        "rt_lmp_lag_24",
        "rt_lmp_lag_168",
        "da_lmp_lag_24",
        "da_lmp_lag_168",
        "demand_da_forecast_mw_lag_24",
        "demand_da_forecast_mw_lag_168",
    ]
    hrrr = [col for col in HRRR_FEATURE_COLUMNS if col in df.columns]
    if text_group == "none":
        weather_text: list[str] = []
    elif text_group == "rule":
        weather_text = [col for col in df.columns if col.startswith("wx_prior_")]
    elif text_group == "rule_core":
        weather_text = [
            "wx_prior_cloud_score",
            "wx_prior_rain_score",
            "wx_prior_fog_visibility_score",
        ]
    elif text_group == "llm":
        weather_text = [col for col in df.columns if col.startswith("llm_prior_")]
    elif text_group == "llm_rule":
        weather_text = [col for col in df.columns if col.startswith("llm_prior_rule_")]
    elif text_group == "llm_rule_cloud":
        weather_text = ["llm_prior_rule_cloud_score"]
    elif text_group == "all":
        weather_text = [col for col in df.columns if col.startswith(("wx_prior_", "llm_prior_"))]
    else:
        raise ValueError(f"unknown text_group: {text_group}")
    cols = [col for col in base + hrrr + weather_text if col in df.columns]
    return [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]


def _feature_columns(df: pd.DataFrame, include_weather_text: bool = True) -> list[str]:
    return feature_columns(df, text_group="rule" if include_weather_text else "none")


def _ridge_model() -> object:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))


def _gbr_model(random_state: int) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=random_state),
    )


def _mlp_model(
    random_state: int,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    alpha: float = 5e-5,
    learning_rate_init: float = 8e-4,
    max_iter: int = 220,
    n_iter_no_change: int = 18,
) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=256,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
        ),
    )


def fit_best_model_by_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidates: list[dict[str, object]],
    target_col: str,
    validation_start: str,
    lower: float | None = None,
    upper: float | None = None,
    rated_capacity: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    fit_part = train[train["local_date"] < validation_start]
    validation = train[train["local_date"] >= validation_start]
    if fit_part.empty or validation.empty:
        selected = candidates[0]
        validation_rmse = math.nan
    else:
        best_score = math.inf
        selected = candidates[0]
        validation_rmse = math.nan
        for candidate in candidates:
            feature_cols = candidate["feature_cols"]
            make_model = candidate["make_model"]
            model = make_model()
            model.fit(fit_part[feature_cols], fit_part[target_col])
            pred = np.asarray(model.predict(validation[feature_cols]), dtype=float)
            if lower is not None or upper is not None:
                pred = np.clip(
                    pred,
                    -np.inf if lower is None else lower,
                    np.inf if upper is None else upper,
                )
            score = regression_metrics(validation[target_col], pred, rated_capacity)["rmse"]
            if np.isfinite(score) and score < best_score:
                best_score = score
                selected = candidate
                validation_rmse = score

    feature_cols = selected["feature_cols"]
    final_model = selected["make_model"]()
    final_model.fit(train[feature_cols], train[target_col])
    train_pred = np.asarray(final_model.predict(train[feature_cols]), dtype=float)
    test_pred = np.asarray(final_model.predict(test[feature_cols]), dtype=float)
    if lower is not None or upper is not None:
        train_pred = np.clip(train_pred, -np.inf if lower is None else lower, np.inf if upper is None else upper)
        test_pred = np.clip(test_pred, -np.inf if lower is None else lower, np.inf if upper is None else upper)
    return (
        train_pred,
        test_pred,
        {
            "selected_candidate": str(selected["name"]),
            "validation_rmse": float(validation_rmse),
            "selected_feature_count": int(len(feature_cols)),
        },
    )


def fit_predict_residual_mlp(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    base_train_prediction: np.ndarray,
    base_test_prediction: np.ndarray,
    random_state: int,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    train_x = train[feature_cols].copy()
    test_x = test[feature_cols].copy()
    train_x["base_prediction"] = base_train_prediction
    test_x["base_prediction"] = base_test_prediction
    residual_target = train[target_col].to_numpy(dtype=float) - np.asarray(base_train_prediction, dtype=float)
    model = _mlp_model(random_state=random_state)
    model.fit(train_x, residual_target)
    train_pred = apply_residual_correction(
        base_train_prediction,
        model.predict(train_x),
        lower=lower,
        upper=upper,
    )
    test_pred = apply_residual_correction(
        base_test_prediction,
        model.predict(test_x),
        lower=lower,
        upper=upper,
    )
    return train_pred, test_pred


def _fit_predict_torch_sequence(
    model_df: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    feature_cols: list[str],
    target_col: str,
    model_kind: str,
    random_state: int,
    seq_len: int = 24,
    hidden_size: int = 32,
    epochs: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
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

    train_x, train_y, train_rows = make_sequence_dataset(features, scaled_targets, train_mask.to_numpy(), seq_len)
    pred_x, _, pred_rows = make_sequence_dataset(features, scaled_targets, test_mask.to_numpy(), seq_len)
    if len(train_x) == 0 or len(pred_x) == 0:
        train_pred = np.full(len(model_df), np.nan)
        test_pred = np.full(int(test_mask.sum()), np.nan)
        return train_pred, test_pred

    class SequenceRegressor(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.model_kind = model_kind
            if model_kind == "transformer":
                self.input_projection = nn.Linear(input_size, hidden_size)
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=4,
                    dim_feedforward=hidden_size * 2,
                    dropout=0.1,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            elif model_kind == "cnn":
                self.conv = nn.Sequential(
                    nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            else:
                recurrent_cls = nn.GRU if model_kind == "gru" else nn.LSTM
                self.recurrent = recurrent_cls(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

        def forward(self, x):
            if self.model_kind == "transformer":
                out = self.encoder(self.input_projection(x))
            elif self.model_kind == "cnn":
                out = self.conv(x.transpose(1, 2)).transpose(1, 2)
            else:
                out, _ = self.recurrent(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    device = torch.device("cpu")
    model = SequenceRegressor(input_size=train_x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=256,
        shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    def predict_array(x: np.ndarray) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(x), 512):
                batch = torch.from_numpy(x[start : start + 512]).to(device)
                preds.append(model(batch).cpu().numpy())
        return np.concatenate(preds) * target_std + target_mean

    train_pred_full = np.full(len(model_df), np.nan)
    train_pred_full[train_rows] = predict_array(train_x)
    pred_full = np.full(len(model_df), np.nan)
    pred_full[pred_rows] = predict_array(pred_x)
    test_positions = np.flatnonzero(test_mask.to_numpy())
    test_pred = pred_full[test_positions]
    return train_pred_full, test_pred


def fit_predict_models(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model_df = master.dropna(subset=["pv_mw_lag_24", "rt_lmp_lag_24"]).copy()
    train = model_df[model_df["local_date"] < "2024-01-01"].copy()
    test = model_df[model_df["local_date"] >= "2024-01-01"].copy()
    rated_capacity = float(model_df["pv_mw"].max())
    numeric_feature_cols = [col for col in feature_columns(model_df, text_group="none") if train[col].notna().any()]
    text_feature_cols = [col for col in feature_columns(model_df, text_group="rule") if train[col].notna().any()]
    llm_rule_feature_cols = [
        col for col in feature_columns(model_df, text_group="llm_rule") if train[col].notna().any()
    ]
    llm_feature_cols = [col for col in feature_columns(model_df, text_group="llm") if train[col].notna().any()]
    combined_feature_cols = [col for col in feature_columns(model_df, text_group="all") if train[col].notna().any()]
    has_llm_rule_features = len(llm_rule_feature_cols) > len(numeric_feature_cols)
    has_llm_features = len(llm_feature_cols) > len(numeric_feature_cols)
    has_combined_features = len(combined_feature_cols) > len(text_feature_cols)
    model_specs = [
        ("ridge", "numeric", _ridge_model, _ridge_model),
        ("ridge_text", "text", _ridge_model, _ridge_model),
        ("gbr", "numeric", lambda: _gbr_model(random_state=7), lambda: _gbr_model(random_state=11)),
        ("gbr_text", "text", lambda: _gbr_model(random_state=7), lambda: _gbr_model(random_state=11)),
        ("mlp", "numeric", lambda: _mlp_model(random_state=7), lambda: _mlp_model(random_state=11)),
        ("mlp_text", "text", lambda: _mlp_model(random_state=7), lambda: _mlp_model(random_state=11)),
    ]
    if has_llm_features:
        model_specs.extend(
            [
                ("ridge_llm", "llm", _ridge_model, _ridge_model),
                ("mlp_llm", "llm", lambda: _mlp_model(random_state=23), lambda: _mlp_model(random_state=29)),
            ]
        )
    if has_llm_rule_features:
        model_specs.extend(
            [
                ("ridge_llm_rule", "llm_rule", _ridge_model, _ridge_model),
                ("mlp_llm_rule", "llm_rule", lambda: _mlp_model(random_state=23), lambda: _mlp_model(random_state=29)),
            ]
        )
    if has_combined_features:
        model_specs.extend(
            [
                ("ridge_text_llm", "all", _ridge_model, _ridge_model),
                ("mlp_text_llm", "all", lambda: _mlp_model(random_state=31), lambda: _mlp_model(random_state=37)),
            ]
        )

    pred_base_cols = [
        "timestamp_utc",
        "local_date",
        "hour",
        "is_solar_hour",
        "has_extreme_event",
        "event_types",
        "pv_mw",
        "caiso_solar_dam_forecast_mw",
        "rt_lmp",
        "da_lmp",
    ]
    pred_base_cols.extend([col for col in test.columns if col.startswith("wx_prior_")])
    pred_base_cols.extend([col for col in test.columns if col.startswith("llm_prior_")])
    preds = test[[col for col in pred_base_cols if col in test.columns]].copy()
    preds["pv_persistence"] = test["pv_mw_lag_24"].clip(lower=0.0)
    if "caiso_solar_dam_forecast_mw" in test.columns:
        preds["pv_caiso_solar_dam"] = test["caiso_solar_dam_forecast_mw"].clip(lower=0.0, upper=rated_capacity)
    preds["rt_persistence"] = test["rt_lmp_lag_24"]
    train_residuals = train[["timestamp_utc", "local_date", "hour"]].copy()
    if "caiso_solar_dam_forecast_mw" in train.columns:
        train_residuals["pv_caiso_solar_dam_residual"] = (
            train["pv_mw"].to_numpy()
            - train["caiso_solar_dam_forecast_mw"].clip(lower=0.0, upper=rated_capacity).to_numpy()
        )

    for name, feature_set, make_pv_model, make_price_model in model_specs:
        if feature_set == "text":
            feature_cols = text_feature_cols
        elif feature_set == "llm":
            feature_cols = llm_feature_cols
        elif feature_set == "llm_rule":
            feature_cols = llm_rule_feature_cols
        elif feature_set == "all":
            feature_cols = combined_feature_cols
        else:
            feature_cols = numeric_feature_cols
        pv_model = make_pv_model()
        pv_model.fit(train[feature_cols], train["pv_mw"])
        train_residuals[f"pv_{name}_residual"] = train["pv_mw"].to_numpy() - np.maximum(
            pv_model.predict(train[feature_cols]), 0.0
        )
        preds[f"pv_{name}"] = np.maximum(pv_model.predict(test[feature_cols]), 0.0)

        price_model = make_price_model()
        price_model.fit(train[feature_cols], train["rt_lmp"])
        train_residuals[f"rt_{name}_residual"] = train["rt_lmp"].to_numpy() - price_model.predict(train[feature_cols])
        preds[f"rt_{name}"] = price_model.predict(test[feature_cols])

    tuned_meta: dict[str, float | str | int] = {}
    if has_combined_features:
        validation_start = "2023-10-01"

        def make_tuned_candidates(seed: int) -> list[dict[str, object]]:
            return [
                {
                    "name": "rule_default",
                    "feature_cols": text_feature_cols,
                    "make_model": lambda seed=seed: _mlp_model(random_state=seed),
                },
                {
                    "name": "llm_default",
                    "feature_cols": llm_feature_cols,
                    "make_model": lambda seed=seed: _mlp_model(random_state=seed),
                },
                {
                    "name": "combined_regularized_small",
                    "feature_cols": combined_feature_cols,
                    "make_model": lambda seed=seed: _mlp_model(
                        random_state=seed,
                        hidden_layer_sizes=(64, 32),
                        alpha=5e-4,
                        learning_rate_init=5e-4,
                        n_iter_no_change=12,
                    ),
                },
                {
                    "name": "combined_strong_regularized_small",
                    "feature_cols": combined_feature_cols,
                    "make_model": lambda seed=seed: _mlp_model(
                        random_state=seed,
                        hidden_layer_sizes=(64, 32),
                        alpha=2e-3,
                        learning_rate_init=5e-4,
                        n_iter_no_change=12,
                    ),
                },
            ]

        pv_tuned_train_pred, pv_tuned_test_pred, pv_tuned_meta = fit_best_model_by_validation(
            train=train,
            test=test,
            candidates=make_tuned_candidates(41),
            target_col="pv_mw",
            validation_start=validation_start,
            lower=0.0,
            upper=rated_capacity,
            rated_capacity=rated_capacity,
        )
        train_residuals["pv_mlp_text_llm_tuned_residual"] = train["pv_mw"].to_numpy() - pv_tuned_train_pred
        preds["pv_mlp_text_llm_tuned"] = pv_tuned_test_pred
        tuned_meta.update(
            {
                "pv_mlp_text_llm_tuned_candidate": pv_tuned_meta["selected_candidate"],
                "pv_mlp_text_llm_tuned_validation_rmse": pv_tuned_meta["validation_rmse"],
                "pv_mlp_text_llm_tuned_feature_count": pv_tuned_meta["selected_feature_count"],
            }
        )

        rt_tuned_train_pred, rt_tuned_test_pred, rt_tuned_meta = fit_best_model_by_validation(
            train=train,
            test=test,
            candidates=make_tuned_candidates(43),
            target_col="rt_lmp",
            validation_start=validation_start,
        )
        train_residuals["rt_mlp_text_llm_tuned_residual"] = train["rt_lmp"].to_numpy() - rt_tuned_train_pred
        preds["rt_mlp_text_llm_tuned"] = rt_tuned_test_pred
        tuned_meta.update(
            {
                "rt_mlp_text_llm_tuned_candidate": rt_tuned_meta["selected_candidate"],
                "rt_mlp_text_llm_tuned_validation_rmse": rt_tuned_meta["validation_rmse"],
                "rt_mlp_text_llm_tuned_feature_count": rt_tuned_meta["selected_feature_count"],
            }
        )

    pv_ridge_text_train_pred = train["pv_mw"].to_numpy() - train_residuals["pv_ridge_text_residual"].to_numpy()
    pv_resid_train_pred, pv_resid_test_pred = fit_predict_residual_mlp(
        train=train,
        test=test,
        feature_cols=text_feature_cols,
        target_col="pv_mw",
        base_train_prediction=pv_ridge_text_train_pred,
        base_test_prediction=preds["pv_ridge_text"].to_numpy(),
        random_state=17,
        lower=0.0,
        upper=rated_capacity,
    )
    train_residuals["pv_mlp_resid_text_residual"] = train["pv_mw"].to_numpy() - pv_resid_train_pred
    preds["pv_mlp_resid_text"] = pv_resid_test_pred

    rt_ridge_text_train_pred = train["rt_lmp"].to_numpy() - train_residuals["rt_ridge_text_residual"].to_numpy()
    rt_resid_train_pred, rt_resid_test_pred = fit_predict_residual_mlp(
        train=train,
        test=test,
        feature_cols=text_feature_cols,
        target_col="rt_lmp",
        base_train_prediction=rt_ridge_text_train_pred,
        base_test_prediction=preds["rt_ridge_text"].to_numpy(),
        random_state=19,
    )
    train_residuals["rt_mlp_resid_text_residual"] = train["rt_lmp"].to_numpy() - rt_resid_train_pred
    preds["rt_mlp_resid_text"] = rt_resid_test_pred

    torch_sequence_models = 0
    train_mask = model_df["local_date"] < "2024-01-01"
    test_mask = model_df["local_date"] >= "2024-01-01"
    for model_kind in ["lstm", "gru", "transformer"]:
        name = f"{model_kind}_text"
        try:
            pv_train_pred, pv_test_pred = _fit_predict_torch_sequence(
                model_df=model_df,
                train_mask=train_mask,
                test_mask=test_mask,
                feature_cols=text_feature_cols,
                target_col="pv_mw",
                model_kind=model_kind,
                random_state={"lstm": 101, "gru": 103, "transformer": 105}[model_kind],
            )
            rt_train_pred, rt_test_pred = _fit_predict_torch_sequence(
                model_df=model_df,
                train_mask=train_mask,
                test_mask=test_mask,
                feature_cols=text_feature_cols,
                target_col="rt_lmp",
                model_kind=model_kind,
                random_state={"lstm": 201, "gru": 203, "transformer": 205}[model_kind],
            )
        except RuntimeError:
            continue
        train_residuals[f"pv_{name}_residual"] = train["pv_mw"].to_numpy() - np.maximum(
            pv_train_pred[train_mask.to_numpy()], 0.0
        )
        train_residuals[f"rt_{name}_residual"] = train["rt_lmp"].to_numpy() - rt_train_pred[train_mask.to_numpy()]
        preds[f"pv_{name}"] = np.maximum(pv_test_pred, 0.0)
        preds[f"rt_{name}"] = rt_test_pred
        torch_sequence_models += 1

    if has_llm_features:
        try:
            pv_train_pred, pv_test_pred = _fit_predict_torch_sequence(
                model_df=model_df,
                train_mask=train_mask,
                test_mask=test_mask,
                feature_cols=llm_feature_cols,
                target_col="pv_mw",
                model_kind="transformer",
                random_state=305,
            )
            rt_train_pred, rt_test_pred = _fit_predict_torch_sequence(
                model_df=model_df,
                train_mask=train_mask,
                test_mask=test_mask,
                feature_cols=llm_feature_cols,
                target_col="rt_lmp",
                model_kind="transformer",
                random_state=405,
            )
        except RuntimeError:
            pass
        else:
            train_residuals["pv_transformer_llm_residual"] = train["pv_mw"].to_numpy() - np.maximum(
                pv_train_pred[train_mask.to_numpy()], 0.0
            )
            train_residuals["rt_transformer_llm_residual"] = train["rt_lmp"].to_numpy() - rt_train_pred[
                train_mask.to_numpy()
            ]
            preds["pv_transformer_llm"] = np.maximum(pv_test_pred, 0.0)
            preds["rt_transformer_llm"] = rt_test_pred
            torch_sequence_models += 1

    if has_combined_features:
        try:
            pv_train_pred, pv_test_pred = _fit_predict_torch_sequence(
                model_df=model_df,
                train_mask=train_mask,
                test_mask=test_mask,
                feature_cols=combined_feature_cols,
                target_col="pv_mw",
                model_kind="transformer",
                random_state=505,
            )
            rt_train_pred, rt_test_pred = _fit_predict_torch_sequence(
                model_df=model_df,
                train_mask=train_mask,
                test_mask=test_mask,
                feature_cols=combined_feature_cols,
                target_col="rt_lmp",
                model_kind="transformer",
                random_state=605,
            )
        except RuntimeError:
            pass
        else:
            train_residuals["pv_transformer_text_llm_residual"] = train["pv_mw"].to_numpy() - np.maximum(
                pv_train_pred[train_mask.to_numpy()], 0.0
            )
            train_residuals["rt_transformer_text_llm_residual"] = train["rt_lmp"].to_numpy() - rt_train_pred[
                train_mask.to_numpy()
            ]
            preds["pv_transformer_text_llm"] = np.maximum(pv_test_pred, 0.0)
            preds["rt_transformer_text_llm"] = rt_test_pred
            torch_sequence_models += 1

    metrics_rows: list[dict[str, float | str | int]] = []
    for target, actual_col, capacity in [("pv", "pv_mw", rated_capacity), ("rt_price", "rt_lmp", None)]:
        pred_cols = [
            col for col in preds.columns if col.startswith("pv_" if target == "pv" else "rt_") and col != actual_col
        ]
        subsets = [
            ("all", preds),
            ("solar_hours", preds[preds["is_solar_hour"] == 1]),
            ("extreme", preds[preds["has_extreme_event"] == 1]),
            ("extreme_solar_hours", preds[(preds["has_extreme_event"] == 1) & (preds["is_solar_hour"] == 1)]),
        ]
        for subset_name, subset in subsets:
            for pred_col in pred_cols:
                metrics = regression_metrics(subset[actual_col], subset[pred_col], capacity)
                metrics_rows.append(
                    {
                        "target": target,
                        "subset": subset_name,
                        "model": pred_col,
                        "n": int(len(subset)),
                        **metrics,
                    }
                )

    spike_threshold = float(train["rt_lmp"].quantile(0.95))
    y_spike = (preds["rt_lmp"] > spike_threshold).astype(int)
    for pred_col in [col for col in preds.columns if col.startswith("rt_") and col != "rt_lmp"]:
        y_hat = (preds[pred_col] > spike_threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_spike, y_hat, average="binary", zero_division=0)
        try:
            auc = float(roc_auc_score(y_spike, preds[pred_col]))
        except ValueError:
            auc = math.nan
        metrics_rows.append(
            {
                "target": "rt_spike",
                "subset": "all",
                "model": pred_col,
                "n": int(len(preds)),
                "mae": float(precision),
                "rmse": float(recall),
                "nrmse": float(f1),
                "mape": auc,
            }
        )

    meta = {
        "rated_capacity_mw": rated_capacity,
        "spike_threshold": spike_threshold,
        "numeric_feature_count": len(numeric_feature_cols),
        "text_feature_count": len(text_feature_cols),
        "rule_text_feature_count": max(0, len(text_feature_cols) - len(numeric_feature_cols)),
        "llm_feature_count": len(llm_feature_cols),
        "llm_text_feature_count": max(0, len(llm_feature_cols) - len(numeric_feature_cols)),
        "llm_rule_feature_count": len(llm_rule_feature_cols),
        "llm_rule_text_feature_count": max(0, len(llm_rule_feature_cols) - len(numeric_feature_cols)),
        "combined_feature_count": len(combined_feature_cols),
        "combined_text_feature_count": max(0, len(combined_feature_cols) - len(numeric_feature_cols)),
        "torch_sequence_models": torch_sequence_models,
    }
    meta.update(tuned_meta)
    return preds, pd.DataFrame(metrics_rows), train_residuals, meta


def _feature_group_for_model(model_name: str) -> str:
    if model_name.endswith("_text_llm"):
        return "all"
    if model_name.endswith("_llm_rule_cloud"):
        return "llm_rule_cloud"
    if model_name.endswith("_llm_rule"):
        return "llm_rule"
    if model_name.endswith("_llm"):
        return "llm"
    if model_name.endswith("_rule_core"):
        return "rule_core"
    if model_name.endswith("_text"):
        return "rule"
    return "none"


def _sequence_model_kind(model_name: str) -> str | None:
    for kind in ("lstm", "gru", "transformer"):
        if model_name.startswith(f"{kind}_"):
            return kind
    return None


def _make_named_tabular_model(model_name: str, target_col: str) -> object:
    if model_name.startswith("ridge"):
        return _ridge_model()
    if model_name.startswith("gbr"):
        return _gbr_model(random_state=7 if target_col == "pv_mw" else 11)
    if model_name.startswith("mlp"):
        if model_name == "mlp_llm_rule_cloud":
            return _mlp_model(
                random_state=23 if target_col == "pv_mw" else 29,
                alpha=1e-5,
                learning_rate_init=1e-3,
            )
        if model_name == "mlp_rule_core":
            return _mlp_model(
                random_state=23 if target_col == "pv_mw" else 29,
                alpha=1e-3,
                learning_rate_init=1e-3,
            )
        if model_name in {"mlp_llm", "mlp_llm_rule"}:
            return _mlp_model(random_state=23 if target_col == "pv_mw" else 29)
        if model_name == "mlp_text_llm":
            return _mlp_model(random_state=31 if target_col == "pv_mw" else 37)
        return _mlp_model(random_state=7 if target_col == "pv_mw" else 11)
    raise ValueError(f"unsupported tabular model for split prediction: {model_name}")


def _sequence_random_state(model_kind: str, feature_group: str, target_col: str) -> int:
    if feature_group in {"rule", "llm_rule"} and model_kind == "transformer":
        return 3023 if target_col == "pv_mw" else 3029
    if feature_group == "llm" and model_kind == "transformer":
        return 305 if target_col == "pv_mw" else 405
    if feature_group == "all" and model_kind == "transformer":
        return 505 if target_col == "pv_mw" else 605
    if target_col == "pv_mw":
        return {"lstm": 101, "gru": 103, "transformer": 105}[model_kind]
    return {"lstm": 201, "gru": 203, "transformer": 205}[model_kind]


def _sequence_epochs(model_kind: str, feature_group: str) -> int:
    if model_kind == "transformer" and feature_group in {"rule", "llm_rule"}:
        return 4
    return 20


def _fit_named_split_prediction(
    model_df: pd.DataFrame,
    train_mask: pd.Series,
    eval_mask: pd.Series,
    model_name: str,
    target_col: str,
    rated_capacity: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feature_group = _feature_group_for_model(model_name)
    train = model_df.loc[train_mask]
    feature_cols = [col for col in feature_columns(model_df, text_group=feature_group) if train[col].notna().any()]
    if not feature_cols:
        raise ValueError(f"no usable feature columns for {model_name}")

    sequence_kind = _sequence_model_kind(model_name)
    if sequence_kind is not None:
        train_pred_full, eval_pred = _fit_predict_torch_sequence(
            model_df=model_df,
            train_mask=train_mask,
            test_mask=eval_mask,
            feature_cols=feature_cols,
            target_col=target_col,
            model_kind=sequence_kind,
            random_state=_sequence_random_state(sequence_kind, feature_group, target_col),
            epochs=_sequence_epochs(sequence_kind, feature_group),
        )
    else:
        model = _make_named_tabular_model(model_name, target_col)
        model.fit(train[feature_cols], train[target_col])
        train_pred_full = np.full(len(model_df), np.nan)
        train_pred_full[train_mask.to_numpy()] = model.predict(train[feature_cols])
        eval_pred = model.predict(model_df.loc[eval_mask, feature_cols])

    if target_col == "pv_mw":
        upper = rated_capacity if rated_capacity is not None else math.inf
        train_pred_full = np.clip(train_pred_full, 0.0, upper)
        eval_pred = np.clip(eval_pred, 0.0, upper)
    return train_pred_full, eval_pred


def fit_hybrid_blend_split_predictions(
    master: pd.DataFrame,
    train_end: str,
    eval_start: str,
    eval_end: str | None = None,
    pv_model_name: str = "mlp_text",
    rt_model_name: str = "transformer_text",
    anchor_model_name: str = "mlp_text",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    model_df = master.dropna(subset=["pv_mw_lag_24", "rt_lmp_lag_24"]).copy()
    model_df = model_df.sort_values("timestamp_utc").reset_index(drop=True)
    local_dates = model_df["local_date"].astype(str)
    train_mask = local_dates < train_end
    eval_mask = local_dates >= eval_start
    if eval_end is not None:
        eval_mask &= local_dates < eval_end
    if not train_mask.any() or not eval_mask.any():
        raise ValueError("split must contain non-empty train and evaluation rows")

    rated_capacity = float(model_df["pv_mw"].max())
    eval_df = model_df.loc[eval_mask].copy()
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
    pred_base_cols.extend([col for col in eval_df.columns if col.startswith("wx_prior_")])
    pred_base_cols.extend([col for col in eval_df.columns if col.startswith("llm_prior_")])
    preds = eval_df[[col for col in pred_base_cols if col in eval_df.columns]].copy()
    train_residuals = model_df.loc[train_mask, ["timestamp_utc", "local_date", "hour"]].copy()

    pv_train_pred, pv_eval_pred = _fit_named_split_prediction(
        model_df=model_df,
        train_mask=train_mask,
        eval_mask=eval_mask,
        model_name=pv_model_name,
        target_col="pv_mw",
        rated_capacity=rated_capacity,
    )
    preds[f"pv_{pv_model_name}"] = pv_eval_pred
    train_residuals[f"pv_{pv_model_name}_residual"] = (
        model_df.loc[train_mask, "pv_mw"].to_numpy(dtype=float) - pv_train_pred[train_mask.to_numpy()]
    )

    if anchor_model_name == pv_model_name:
        preds[f"pv_{anchor_model_name}"] = preds[f"pv_{pv_model_name}"]
    else:
        _, anchor_eval_pred = _fit_named_split_prediction(
            model_df=model_df,
            train_mask=train_mask,
            eval_mask=eval_mask,
            model_name=anchor_model_name,
            target_col="pv_mw",
            rated_capacity=rated_capacity,
        )
        preds[f"pv_{anchor_model_name}"] = anchor_eval_pred

    rt_train_pred, rt_eval_pred = _fit_named_split_prediction(
        model_df=model_df,
        train_mask=train_mask,
        eval_mask=eval_mask,
        model_name=rt_model_name,
        target_col="rt_lmp",
    )
    preds[f"rt_{rt_model_name}"] = rt_eval_pred
    train_residuals[f"rt_{rt_model_name}_residual"] = (
        model_df.loc[train_mask, "rt_lmp"].to_numpy(dtype=float) - rt_train_pred[train_mask.to_numpy()]
    )

    meta: dict[str, float | int | str] = {
        "rated_capacity_mw": rated_capacity,
        "train_end": train_end,
        "eval_start": eval_start,
        "eval_end": eval_end or "",
        "train_rows": int(train_mask.sum()),
        "eval_rows": int(eval_mask.sum()),
        "pv_model": pv_model_name,
        "rt_model": rt_model_name,
        "anchor_model": anchor_model_name,
    }
    return preds, train_residuals, meta


def build_residual_scenarios(
    day: pd.DataFrame,
    train_residuals: pd.DataFrame,
    model_name: str,
    rated_capacity: float,
    scenario_count: int,
    rng: np.random.Generator,
    residual_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = day.sort_values("timestamp_utc")
    pv_pred = ordered[f"pv_{model_name}"].to_numpy(dtype=float)
    rt_pred = ordered[f"rt_{model_name}"].to_numpy(dtype=float)
    hours = ordered["hour"].to_numpy(dtype=int)
    pv_resid_col = f"pv_{model_name}_residual"
    rt_resid_col = f"rt_{model_name}_residual"

    pv_scenarios = np.zeros((scenario_count, len(ordered)))
    rt_scenarios = np.zeros((scenario_count, len(ordered)))
    for idx, hour in enumerate(hours):
        pool = train_residuals[train_residuals["hour"] == hour]
        if pool.empty:
            pool = train_residuals
        pv_residuals = np.nan_to_num(pool[pv_resid_col].to_numpy(dtype=float), nan=0.0)
        rt_residuals = np.nan_to_num(pool[rt_resid_col].to_numpy(dtype=float), nan=0.0)
        choices = rng.integers(0, len(pool), size=scenario_count)
        pv_scenarios[:, idx] = np.clip(pv_pred[idx] + residual_scale * pv_residuals[choices], 0.0, rated_capacity)
        rt_scenarios[:, idx] = rt_pred[idx] + residual_scale * rt_residuals[choices]
    return pv_scenarios, rt_scenarios


def build_hybrid_residual_scenarios(
    day: pd.DataFrame,
    train_residuals: pd.DataFrame,
    pv_model_name: str,
    rt_model_name: str,
    rated_capacity: float,
    scenario_count: int,
    rng: np.random.Generator,
    residual_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = day.sort_values("timestamp_utc")
    pv_pred = ordered[f"pv_{pv_model_name}"].to_numpy(dtype=float)
    rt_pred = ordered[f"rt_{rt_model_name}"].to_numpy(dtype=float)
    hours = ordered["hour"].to_numpy(dtype=int)
    pv_resid_col = f"pv_{pv_model_name}_residual"
    rt_resid_col = f"rt_{rt_model_name}_residual"

    pv_scenarios = np.zeros((scenario_count, len(ordered)))
    rt_scenarios = np.zeros((scenario_count, len(ordered)))
    for idx, hour in enumerate(hours):
        pool = train_residuals[train_residuals["hour"] == hour]
        if pool.empty:
            pool = train_residuals
        pv_residuals = np.nan_to_num(pool[pv_resid_col].to_numpy(dtype=float), nan=0.0)
        rt_residuals = np.nan_to_num(pool[rt_resid_col].to_numpy(dtype=float), nan=0.0)
        choices = rng.integers(0, len(pool), size=scenario_count)
        pv_scenarios[:, idx] = np.clip(pv_pred[idx] + residual_scale * pv_residuals[choices], 0.0, rated_capacity)
        rt_scenarios[:, idx] = rt_pred[idx] + residual_scale * rt_residuals[choices]
    return pv_scenarios, rt_scenarios


def _safe_corr(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return math.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def evaluate_scenario_quality(
    preds: pd.DataFrame,
    train_residuals: pd.DataFrame,
    rated_capacity: float,
    model_names: tuple[str, ...] = (
        "ridge_text",
        "mlp_text",
        "transformer_text",
        "ridge_llm",
        "mlp_llm",
        "transformer_llm",
        "ridge_text_llm",
        "mlp_text_llm",
        "transformer_text_llm",
    ),
    scenario_counts: tuple[int, ...] = (10, 20, 50),
) -> pd.DataFrame:
    rows = []
    for model_name in model_names:
        required = {f"pv_{model_name}", f"rt_{model_name}"}
        residual_required = {f"pv_{model_name}_residual", f"rt_{model_name}_residual"}
        if not required.issubset(preds.columns) or not residual_required.issubset(train_residuals.columns):
            continue

        for scenario_count in scenario_counts:
            rng = np.random.default_rng(10_000 + scenario_count + len(model_name))
            pv_actual_parts = []
            rt_actual_parts = []
            pv_mean_parts = []
            rt_mean_parts = []
            pv_p05_parts = []
            pv_p10_parts = []
            pv_p90_parts = []
            pv_p95_parts = []
            rt_p05_parts = []
            rt_p10_parts = []
            rt_p90_parts = []
            rt_p95_parts = []
            corr_errors = []
            for _, day in preds.groupby("local_date", sort=True):
                ordered = day.sort_values("timestamp_utc")
                pv_scenarios, rt_scenarios = build_residual_scenarios(
                    ordered,
                    train_residuals,
                    model_name=model_name,
                    rated_capacity=rated_capacity,
                    scenario_count=scenario_count,
                    rng=rng,
                )
                pv_actual = ordered["pv_mw"].to_numpy(dtype=float)
                rt_actual = ordered["rt_lmp"].to_numpy(dtype=float)
                pv_mean = pv_scenarios.mean(axis=0)
                rt_mean = rt_scenarios.mean(axis=0)

                pv_actual_parts.append(pv_actual)
                rt_actual_parts.append(rt_actual)
                pv_mean_parts.append(pv_mean)
                rt_mean_parts.append(rt_mean)
                pv_p05_parts.append(np.quantile(pv_scenarios, 0.05, axis=0))
                pv_p10_parts.append(np.quantile(pv_scenarios, 0.10, axis=0))
                pv_p90_parts.append(np.quantile(pv_scenarios, 0.90, axis=0))
                pv_p95_parts.append(np.quantile(pv_scenarios, 0.95, axis=0))
                rt_p05_parts.append(np.quantile(rt_scenarios, 0.05, axis=0))
                rt_p10_parts.append(np.quantile(rt_scenarios, 0.10, axis=0))
                rt_p90_parts.append(np.quantile(rt_scenarios, 0.90, axis=0))
                rt_p95_parts.append(np.quantile(rt_scenarios, 0.95, axis=0))

                actual_corr = _safe_corr(pv_actual, rt_actual)
                scenario_corr = _safe_corr(pv_mean, rt_mean)
                if np.isfinite(actual_corr) and np.isfinite(scenario_corr):
                    corr_errors.append(abs(actual_corr - scenario_corr))

            pv_actual_all = np.concatenate(pv_actual_parts)
            rt_actual_all = np.concatenate(rt_actual_parts)
            pv_mean_all = np.concatenate(pv_mean_parts)
            rt_mean_all = np.concatenate(rt_mean_parts)
            pv_p05 = np.concatenate(pv_p05_parts)
            pv_p10 = np.concatenate(pv_p10_parts)
            pv_p90 = np.concatenate(pv_p90_parts)
            pv_p95 = np.concatenate(pv_p95_parts)
            rt_p05 = np.concatenate(rt_p05_parts)
            rt_p10 = np.concatenate(rt_p10_parts)
            rt_p90 = np.concatenate(rt_p90_parts)
            rt_p95 = np.concatenate(rt_p95_parts)

            rows.append(
                {
                    "model": model_name,
                    "scenario_count": scenario_count,
                    "hours": int(len(pv_actual_all)),
                    "pv_mean_rmse": regression_metrics(pv_actual_all, pv_mean_all, rated_capacity)["rmse"],
                    "rt_mean_rmse": regression_metrics(rt_actual_all, rt_mean_all)["rmse"],
                    "pv_80pct_interval_coverage": interval_coverage(pv_actual_all, pv_p10, pv_p90),
                    "pv_90pct_interval_coverage": interval_coverage(pv_actual_all, pv_p05, pv_p95),
                    "rt_80pct_interval_coverage": interval_coverage(rt_actual_all, rt_p10, rt_p90),
                    "rt_90pct_interval_coverage": interval_coverage(rt_actual_all, rt_p05, rt_p95),
                    "pv_80pct_interval_width_mean": float(np.mean(pv_p90 - pv_p10)),
                    "rt_80pct_interval_width_mean": float(np.mean(rt_p90 - rt_p10)),
                    "pv_below_p10_frac": float(np.mean(pv_actual_all < pv_p10)),
                    "rt_above_p90_frac": float(np.mean(rt_actual_all > rt_p90)),
                    "pv_rt_daily_corr_abs_error": float(np.mean(corr_errors)) if corr_errors else math.nan,
                }
            )
    return pd.DataFrame(rows)


def calibrate_ridge_text_llm_scenarios(
    preds: pd.DataFrame,
    train_residuals: pd.DataFrame,
    rated_capacity: float,
    residual_scales: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0),
    cvar_gammas: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0),
    deviation_penalties: tuple[float, ...] = (25.0, 50.0, 75.0, 100.0),
    scenario_count: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_name = "ridge_text_llm"
    required_pred = {
        "timestamp_utc",
        "local_date",
        "hour",
        "pv_mw",
        "da_lmp",
        "rt_lmp",
        f"pv_{model_name}",
        f"rt_{model_name}",
    }
    required_residual = {f"pv_{model_name}_residual", f"rt_{model_name}_residual", "hour"}
    if not required_pred.issubset(preds.columns) or not required_residual.issubset(train_residuals.columns):
        return pd.DataFrame(), pd.DataFrame()

    scenario_rows = []
    scenario_cache: dict[float, list[tuple[pd.Index, pd.DataFrame, np.ndarray, np.ndarray]]] = {}
    for scale in residual_scales:
        rng = np.random.default_rng(70_000 + int(round(scale * 1000)))
        cache_parts = []
        pv_actual_parts = []
        rt_actual_parts = []
        pv_mean_parts = []
        rt_mean_parts = []
        pv_p05_parts = []
        pv_p10_parts = []
        pv_p90_parts = []
        pv_p95_parts = []
        rt_p05_parts = []
        rt_p10_parts = []
        rt_p90_parts = []
        rt_p95_parts = []
        corr_errors = []
        for _, day in preds.groupby("local_date", sort=True):
            ordered = day.sort_values("timestamp_utc")
            pv_scenarios, rt_scenarios = build_residual_scenarios(
                ordered,
                train_residuals,
                model_name=model_name,
                rated_capacity=rated_capacity,
                scenario_count=scenario_count,
                rng=rng,
                residual_scale=scale,
            )
            cache_parts.append((ordered.index, ordered, pv_scenarios, rt_scenarios))
            pv_actual = ordered["pv_mw"].to_numpy(dtype=float)
            rt_actual = ordered["rt_lmp"].to_numpy(dtype=float)
            pv_mean = pv_scenarios.mean(axis=0)
            rt_mean = rt_scenarios.mean(axis=0)

            pv_actual_parts.append(pv_actual)
            rt_actual_parts.append(rt_actual)
            pv_mean_parts.append(pv_mean)
            rt_mean_parts.append(rt_mean)
            pv_p05_parts.append(np.quantile(pv_scenarios, 0.05, axis=0))
            pv_p10_parts.append(np.quantile(pv_scenarios, 0.10, axis=0))
            pv_p90_parts.append(np.quantile(pv_scenarios, 0.90, axis=0))
            pv_p95_parts.append(np.quantile(pv_scenarios, 0.95, axis=0))
            rt_p05_parts.append(np.quantile(rt_scenarios, 0.05, axis=0))
            rt_p10_parts.append(np.quantile(rt_scenarios, 0.10, axis=0))
            rt_p90_parts.append(np.quantile(rt_scenarios, 0.90, axis=0))
            rt_p95_parts.append(np.quantile(rt_scenarios, 0.95, axis=0))

            actual_corr = _safe_corr(pv_actual, rt_actual)
            scenario_corr = _safe_corr(pv_mean, rt_mean)
            if np.isfinite(actual_corr) and np.isfinite(scenario_corr):
                corr_errors.append(abs(actual_corr - scenario_corr))

        scenario_cache[scale] = cache_parts
        pv_actual_all = np.concatenate(pv_actual_parts)
        rt_actual_all = np.concatenate(rt_actual_parts)
        pv_mean_all = np.concatenate(pv_mean_parts)
        rt_mean_all = np.concatenate(rt_mean_parts)
        pv_p05 = np.concatenate(pv_p05_parts)
        pv_p10 = np.concatenate(pv_p10_parts)
        pv_p90 = np.concatenate(pv_p90_parts)
        pv_p95 = np.concatenate(pv_p95_parts)
        rt_p05 = np.concatenate(rt_p05_parts)
        rt_p10 = np.concatenate(rt_p10_parts)
        rt_p90 = np.concatenate(rt_p90_parts)
        rt_p95 = np.concatenate(rt_p95_parts)

        scenario_rows.append(
            {
                "model": model_name,
                "scenario_count": int(scenario_count),
                "residual_scale": float(scale),
                "hours": int(len(pv_actual_all)),
                "pv_mean_rmse": regression_metrics(pv_actual_all, pv_mean_all, rated_capacity)["rmse"],
                "rt_mean_rmse": regression_metrics(rt_actual_all, rt_mean_all)["rmse"],
                "pv_80pct_interval_coverage": interval_coverage(pv_actual_all, pv_p10, pv_p90),
                "pv_90pct_interval_coverage": interval_coverage(pv_actual_all, pv_p05, pv_p95),
                "rt_80pct_interval_coverage": interval_coverage(rt_actual_all, rt_p10, rt_p90),
                "rt_90pct_interval_coverage": interval_coverage(rt_actual_all, rt_p05, rt_p95),
                "pv_80pct_interval_width_mean": float(np.mean(pv_p90 - pv_p10)),
                "rt_80pct_interval_width_mean": float(np.mean(rt_p90 - rt_p10)),
                "pv_below_p10_frac": float(np.mean(pv_actual_all < pv_p10)),
                "rt_above_p90_frac": float(np.mean(rt_actual_all > rt_p90)),
                "pv_rt_daily_corr_abs_error": float(np.mean(corr_errors)) if corr_errors else math.nan,
            }
        )

    actual = preds["pv_mw"].to_numpy(dtype=float)
    da = preds["da_lmp"].to_numpy(dtype=float)
    rt = preds["rt_lmp"].to_numpy(dtype=float)
    bidding_rows = []
    for scale in residual_scales:
        for gamma in cvar_gammas:
            for penalty in deviation_penalties:
                bid_series = pd.Series(index=preds.index, dtype=float)
                failures = 0
                solve_time = 0.0
                for ordered_index, ordered, pv_scenarios, rt_scenarios in scenario_cache[scale]:
                    start = time.perf_counter()
                    try:
                        q_day = solve_cvar_bids(
                            pv_scenarios=pv_scenarios,
                            rt_scenarios=rt_scenarios,
                            da_lmp=ordered["da_lmp"].to_numpy(dtype=float),
                            rated_capacity=rated_capacity,
                            gamma=gamma,
                            shortage_penalty=penalty,
                            surplus_penalty=penalty,
                        )
                    except RuntimeError:
                        failures += 1
                        q_day = ordered[f"pv_{model_name}"].to_numpy(dtype=float)
                    solve_time += time.perf_counter() - start
                    bid_series.loc[ordered_index] = q_day

                q = np.clip(np.nan_to_num(bid_series.to_numpy(dtype=float), nan=0.0), 0.0, rated_capacity)
                revenue = settlement_revenue(
                    actual,
                    q,
                    da,
                    rt,
                    shortage_penalty=penalty,
                    surplus_penalty=penalty,
                )
                shortage = np.maximum(q - actual, 0.0)
                surplus = np.maximum(actual - q, 0.0)
                bidding_rows.append(
                    {
                        "strategy": f"S22_calibrated_scale_{scale:.2f}_gamma_{gamma:.2f}_penalty_{penalty:g}",
                        "model": model_name,
                        "scenario_count": int(scenario_count),
                        "residual_scale": float(scale),
                        "cvar_gamma": float(gamma),
                        "deviation_penalty": float(penalty),
                        "hours": int(len(q)),
                        "total_revenue": float(np.sum(revenue)),
                        "avg_revenue_per_hour": float(np.mean(revenue)),
                        "imbalance_mwh_proxy": float(np.sum(shortage + surplus)),
                        "shortage_mwh_proxy": float(np.sum(shortage)),
                        "surplus_mwh_proxy": float(np.sum(surplus)),
                        "worst_5pct_avg_revenue": float(
                            np.mean(np.sort(revenue)[: max(1, math.ceil(0.05 * len(revenue)))])
                        ),
                        "cvar_95_loss": cvar_loss(-revenue, alpha=0.95),
                        "solve_time_sec": float(solve_time),
                        "optimization_failures": int(failures),
                    }
                )

    return pd.DataFrame(scenario_rows), pd.DataFrame(bidding_rows)


def evaluate_calibrated_s22_seed_robustness(
    preds: pd.DataFrame,
    train_residuals: pd.DataFrame,
    rated_capacity: float,
    seeds: tuple[int, ...] = (71_000, 71_001, 71_011, 71_021, 71_031),
    residual_scale: float = 1.0,
    cvar_gamma: float = 0.25,
    deviation_penalty: float = 50.0,
    scenario_count: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_name = "ridge_text_llm"
    strategy_name = (
        f"S22_calibrated_scale_{residual_scale:.2f}_gamma_{cvar_gamma:.2f}_penalty_{deviation_penalty:g}"
    )
    required_pred = {
        "timestamp_utc",
        "local_date",
        "hour",
        "pv_mw",
        "da_lmp",
        "rt_lmp",
        f"pv_{model_name}",
        f"rt_{model_name}",
    }
    required_residual = {f"pv_{model_name}_residual", f"rt_{model_name}_residual", "hour"}
    if not required_pred.issubset(preds.columns) or not required_residual.issubset(train_residuals.columns):
        return pd.DataFrame(), pd.DataFrame()

    actual = preds["pv_mw"].to_numpy(dtype=float)
    da = preds["da_lmp"].to_numpy(dtype=float)
    rt = preds["rt_lmp"].to_numpy(dtype=float)
    seed_rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        bid_series = pd.Series(index=preds.index, dtype=float)
        failures = 0
        solve_time = 0.0
        for _, day in preds.groupby("local_date", sort=True):
            ordered = day.sort_values("timestamp_utc")
            pv_scenarios, rt_scenarios = build_residual_scenarios(
                ordered,
                train_residuals,
                model_name=model_name,
                rated_capacity=rated_capacity,
                scenario_count=scenario_count,
                rng=rng,
                residual_scale=residual_scale,
            )
            start = time.perf_counter()
            try:
                q_day = solve_cvar_bids(
                    pv_scenarios=pv_scenarios,
                    rt_scenarios=rt_scenarios,
                    da_lmp=ordered["da_lmp"].to_numpy(dtype=float),
                    rated_capacity=rated_capacity,
                    gamma=cvar_gamma,
                    shortage_penalty=deviation_penalty,
                    surplus_penalty=deviation_penalty,
                )
            except RuntimeError:
                failures += 1
                q_day = ordered[f"pv_{model_name}"].to_numpy(dtype=float)
            solve_time += time.perf_counter() - start
            bid_series.loc[ordered.index] = q_day

        q = np.clip(np.nan_to_num(bid_series.to_numpy(dtype=float), nan=0.0), 0.0, rated_capacity)
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
        seed_rows.append(
            {
                "strategy": strategy_name,
                "model": model_name,
                "seed": int(seed),
                "scenario_count": int(scenario_count),
                "residual_scale": float(residual_scale),
                "cvar_gamma": float(cvar_gamma),
                "deviation_penalty": float(deviation_penalty),
                "hours": int(len(q)),
                "total_revenue": float(np.sum(revenue)),
                "avg_revenue_per_hour": float(np.mean(revenue)),
                "imbalance_mwh_proxy": float(np.sum(shortage + surplus)),
                "shortage_mwh_proxy": float(np.sum(shortage)),
                "surplus_mwh_proxy": float(np.sum(surplus)),
                "worst_5pct_avg_revenue": float(np.mean(np.sort(revenue)[: max(1, math.ceil(0.05 * len(revenue)))])),
                "cvar_95_loss": cvar_loss(-revenue, alpha=0.95),
                "solve_time_sec": float(solve_time),
                "optimization_failures": int(failures),
            }
        )

    seed_df = pd.DataFrame(seed_rows)
    if seed_df.empty:
        return seed_df, pd.DataFrame()

    summary: dict[str, float | int | str] = {
        "strategy": strategy_name,
        "model": model_name,
        "seed_count": int(len(seed_df)),
        "seeds": "|".join(str(seed) for seed in seeds),
        "scenario_count": int(scenario_count),
        "residual_scale": float(residual_scale),
        "cvar_gamma": float(cvar_gamma),
        "deviation_penalty": float(deviation_penalty),
        "hours": int(seed_df["hours"].iloc[0]),
        "solve_time_sec_total": float(seed_df["solve_time_sec"].sum()),
        "optimization_failures_total": int(seed_df["optimization_failures"].sum()),
    }
    metric_cols = [
        "total_revenue",
        "avg_revenue_per_hour",
        "imbalance_mwh_proxy",
        "shortage_mwh_proxy",
        "surplus_mwh_proxy",
        "worst_5pct_avg_revenue",
        "cvar_95_loss",
    ]
    for col in metric_cols:
        values = seed_df[col].to_numpy(dtype=float)
        summary[f"{col}_mean"] = float(np.mean(values))
        summary[f"{col}_std"] = float(np.std(values, ddof=0))
        summary[f"{col}_min"] = float(np.min(values))
        summary[f"{col}_max"] = float(np.max(values))

    return seed_df, pd.DataFrame([summary])


def evaluate_calibrated_s22_multiconfig_seed_robustness(
    preds: pd.DataFrame,
    train_residuals: pd.DataFrame,
    rated_capacity: float,
    seeds: tuple[int, ...] = (71_000, 71_001, 71_011, 71_021, 71_031),
    residual_scales: tuple[float, ...] = (1.0, 1.25, 1.5),
    cvar_gammas: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5),
    deviation_penalty: float = 50.0,
    scenario_count: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_frames = []
    summary_frames = []
    for scale in residual_scales:
        for gamma in cvar_gammas:
            seed_rows, summary = evaluate_calibrated_s22_seed_robustness(
                preds=preds,
                train_residuals=train_residuals,
                rated_capacity=rated_capacity,
                seeds=seeds,
                residual_scale=scale,
                cvar_gamma=gamma,
                deviation_penalty=deviation_penalty,
                scenario_count=scenario_count,
            )
            if not seed_rows.empty:
                seed_frames.append(seed_rows)
            if not summary.empty:
                summary_frames.append(summary)

    seed_df = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()
    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    return seed_df, summary_df


def evaluate_hybrid_scenario_seed_robustness(
    preds: pd.DataFrame,
    train_residuals: pd.DataFrame,
    rated_capacity: float,
    seeds: tuple[int, ...] = (71_000, 71_001, 71_011, 71_021, 71_031),
    pv_model_name: str = "mlp_text",
    rt_model_name: str = "transformer_text",
    residual_scale: float = 1.0,
    cvar_gamma: float = 0.0,
    deviation_penalty: float = 50.0,
    scenario_count: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy_name = (
        f"Hybrid_{pv_model_name}_{rt_model_name}_scale_{residual_scale:.2f}"
        f"_gamma_{cvar_gamma:.2f}_penalty_{deviation_penalty:g}"
    )
    required_pred = {
        "timestamp_utc",
        "local_date",
        "hour",
        "pv_mw",
        "da_lmp",
        "rt_lmp",
        f"pv_{pv_model_name}",
        f"rt_{rt_model_name}",
    }
    required_residual = {f"pv_{pv_model_name}_residual", f"rt_{rt_model_name}_residual", "hour"}
    if not required_pred.issubset(preds.columns) or not required_residual.issubset(train_residuals.columns):
        return pd.DataFrame(), pd.DataFrame()

    actual = preds["pv_mw"].to_numpy(dtype=float)
    da = preds["da_lmp"].to_numpy(dtype=float)
    rt = preds["rt_lmp"].to_numpy(dtype=float)
    seed_rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        bid_series = pd.Series(index=preds.index, dtype=float)
        failures = 0
        solve_time = 0.0
        for _, day in preds.groupby("local_date", sort=True):
            ordered = day.sort_values("timestamp_utc")
            pv_scenarios, rt_scenarios = build_hybrid_residual_scenarios(
                ordered,
                train_residuals,
                pv_model_name=pv_model_name,
                rt_model_name=rt_model_name,
                rated_capacity=rated_capacity,
                scenario_count=scenario_count,
                rng=rng,
                residual_scale=residual_scale,
            )
            start = time.perf_counter()
            try:
                q_day = solve_cvar_bids(
                    pv_scenarios=pv_scenarios,
                    rt_scenarios=rt_scenarios,
                    da_lmp=ordered["da_lmp"].to_numpy(dtype=float),
                    rated_capacity=rated_capacity,
                    gamma=cvar_gamma,
                    shortage_penalty=deviation_penalty,
                    surplus_penalty=deviation_penalty,
                )
            except RuntimeError:
                failures += 1
                q_day = ordered[f"pv_{pv_model_name}"].to_numpy(dtype=float)
            solve_time += time.perf_counter() - start
            bid_series.loc[ordered.index] = q_day

        q = np.clip(np.nan_to_num(bid_series.to_numpy(dtype=float), nan=0.0), 0.0, rated_capacity)
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
        seed_rows.append(
            {
                "strategy": strategy_name,
                "pv_model": pv_model_name,
                "rt_model": rt_model_name,
                "seed": int(seed),
                "scenario_count": int(scenario_count),
                "residual_scale": float(residual_scale),
                "cvar_gamma": float(cvar_gamma),
                "deviation_penalty": float(deviation_penalty),
                "hours": int(len(q)),
                "total_revenue": float(np.sum(revenue)),
                "avg_revenue_per_hour": float(np.mean(revenue)),
                "imbalance_mwh_proxy": float(np.sum(shortage + surplus)),
                "shortage_mwh_proxy": float(np.sum(shortage)),
                "surplus_mwh_proxy": float(np.sum(surplus)),
                "worst_5pct_avg_revenue": float(np.mean(np.sort(revenue)[: max(1, math.ceil(0.05 * len(revenue)))])),
                "cvar_95_loss": cvar_loss(-revenue, alpha=0.95),
                "solve_time_sec": float(solve_time),
                "optimization_failures": int(failures),
            }
        )

    seed_df = pd.DataFrame(seed_rows)
    if seed_df.empty:
        return seed_df, pd.DataFrame()

    summary: dict[str, float | int | str] = {
        "strategy": strategy_name,
        "pv_model": pv_model_name,
        "rt_model": rt_model_name,
        "seed_count": int(len(seed_df)),
        "seeds": "|".join(str(seed) for seed in seeds),
        "scenario_count": int(scenario_count),
        "residual_scale": float(residual_scale),
        "cvar_gamma": float(cvar_gamma),
        "deviation_penalty": float(deviation_penalty),
        "hours": int(seed_df["hours"].iloc[0]),
        "solve_time_sec_total": float(seed_df["solve_time_sec"].sum()),
        "optimization_failures_total": int(seed_df["optimization_failures"].sum()),
    }
    metric_cols = [
        "total_revenue",
        "avg_revenue_per_hour",
        "imbalance_mwh_proxy",
        "shortage_mwh_proxy",
        "surplus_mwh_proxy",
        "worst_5pct_avg_revenue",
        "cvar_95_loss",
    ]
    for col in metric_cols:
        values = seed_df[col].to_numpy(dtype=float)
        summary[f"{col}_mean"] = float(np.mean(values))
        summary[f"{col}_std"] = float(np.std(values, ddof=0))
        summary[f"{col}_min"] = float(np.min(values))
        summary[f"{col}_max"] = float(np.max(values))

    return seed_df, pd.DataFrame([summary])


def evaluate_hybrid_blended_scenario_seed_robustness(
    preds: pd.DataFrame,
    train_residuals: pd.DataFrame,
    rated_capacity: float,
    seeds: tuple[int, ...] = (71_000, 71_001, 71_011, 71_021, 71_031),
    pv_model_name: str = "mlp_text",
    rt_model_name: str = "transformer_text",
    anchor_model_name: str = "mlp_text",
    lp_weights: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    residual_scale: float = 1.0,
    cvar_gamma: float = 0.25,
    deviation_penalty: float = 50.0,
    scenario_count: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_pred = {
        "timestamp_utc",
        "local_date",
        "hour",
        "pv_mw",
        "da_lmp",
        "rt_lmp",
        f"pv_{pv_model_name}",
        f"rt_{rt_model_name}",
        f"pv_{anchor_model_name}",
    }
    required_residual = {f"pv_{pv_model_name}_residual", f"rt_{rt_model_name}_residual", "hour"}
    if not required_pred.issubset(preds.columns) or not required_residual.issubset(train_residuals.columns):
        return pd.DataFrame(), pd.DataFrame()

    actual = preds["pv_mw"].to_numpy(dtype=float)
    da = preds["da_lmp"].to_numpy(dtype=float)
    rt = preds["rt_lmp"].to_numpy(dtype=float)
    anchor_bid = np.clip(
        np.nan_to_num(preds[f"pv_{anchor_model_name}"].to_numpy(dtype=float), nan=0.0),
        0.0,
        rated_capacity,
    )
    seed_rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        lp_bid_series = pd.Series(index=preds.index, dtype=float)
        failures = 0
        solve_time = 0.0
        for _, day in preds.groupby("local_date", sort=True):
            ordered = day.sort_values("timestamp_utc")
            pv_scenarios, rt_scenarios = build_hybrid_residual_scenarios(
                ordered,
                train_residuals,
                pv_model_name=pv_model_name,
                rt_model_name=rt_model_name,
                rated_capacity=rated_capacity,
                scenario_count=scenario_count,
                rng=rng,
                residual_scale=residual_scale,
            )
            start = time.perf_counter()
            try:
                q_day = solve_cvar_bids(
                    pv_scenarios=pv_scenarios,
                    rt_scenarios=rt_scenarios,
                    da_lmp=ordered["da_lmp"].to_numpy(dtype=float),
                    rated_capacity=rated_capacity,
                    gamma=cvar_gamma,
                    shortage_penalty=deviation_penalty,
                    surplus_penalty=deviation_penalty,
                )
            except RuntimeError:
                failures += 1
                q_day = ordered[f"pv_{pv_model_name}"].to_numpy(dtype=float)
            solve_time += time.perf_counter() - start
            lp_bid_series.loc[ordered.index] = q_day

        lp_bid = np.clip(np.nan_to_num(lp_bid_series.to_numpy(dtype=float), nan=0.0), 0.0, rated_capacity)
        for lp_weight in lp_weights:
            q = np.clip(float(lp_weight) * lp_bid + (1.0 - float(lp_weight)) * anchor_bid, 0.0, rated_capacity)
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
            strategy_name = (
                f"HybridBlend_{pv_model_name}_{rt_model_name}_anchor_{anchor_model_name}"
                f"_lp{float(lp_weight):.2f}_scale_{residual_scale:.2f}"
                f"_gamma_{cvar_gamma:.2f}_penalty_{deviation_penalty:g}"
            )
            seed_rows.append(
                {
                    "strategy": strategy_name,
                    "pv_model": pv_model_name,
                    "rt_model": rt_model_name,
                    "anchor_model": anchor_model_name,
                    "lp_weight": float(lp_weight),
                    "seed": int(seed),
                    "scenario_count": int(scenario_count),
                    "residual_scale": float(residual_scale),
                    "cvar_gamma": float(cvar_gamma),
                    "deviation_penalty": float(deviation_penalty),
                    "hours": int(len(q)),
                    "total_revenue": float(np.sum(revenue)),
                    "avg_revenue_per_hour": float(np.mean(revenue)),
                    "imbalance_mwh_proxy": float(np.sum(shortage + surplus)),
                    "shortage_mwh_proxy": float(np.sum(shortage)),
                    "surplus_mwh_proxy": float(np.sum(surplus)),
                    "worst_5pct_avg_revenue": float(
                        np.mean(np.sort(revenue)[: max(1, math.ceil(0.05 * len(revenue)))])
                    ),
                    "cvar_95_loss": cvar_loss(-revenue, alpha=0.95),
                    "solve_time_sec": float(solve_time),
                    "optimization_failures": int(failures),
                }
            )

    seed_df = pd.DataFrame(seed_rows)
    if seed_df.empty:
        return seed_df, pd.DataFrame()

    summary_rows = []
    metric_cols = [
        "total_revenue",
        "avg_revenue_per_hour",
        "imbalance_mwh_proxy",
        "shortage_mwh_proxy",
        "surplus_mwh_proxy",
        "worst_5pct_avg_revenue",
        "cvar_95_loss",
    ]
    for (strategy_name, lp_weight), group in seed_df.groupby(["strategy", "lp_weight"], sort=True):
        summary: dict[str, float | int | str] = {
            "strategy": strategy_name,
            "pv_model": pv_model_name,
            "rt_model": rt_model_name,
            "anchor_model": anchor_model_name,
            "lp_weight": float(lp_weight),
            "seed_count": int(len(group)),
            "seeds": "|".join(str(seed) for seed in seeds),
            "scenario_count": int(scenario_count),
            "residual_scale": float(residual_scale),
            "cvar_gamma": float(cvar_gamma),
            "deviation_penalty": float(deviation_penalty),
            "hours": int(group["hours"].iloc[0]),
            "solve_time_sec_total": float(group["solve_time_sec"].sum()),
            "optimization_failures_total": int(group["optimization_failures"].sum()),
        }
        for col in metric_cols:
            values = group[col].to_numpy(dtype=float)
            summary[f"{col}_mean"] = float(np.mean(values))
            summary[f"{col}_std"] = float(np.std(values, ddof=0))
            summary[f"{col}_min"] = float(np.min(values))
            summary[f"{col}_max"] = float(np.max(values))
        summary_rows.append(summary)

    return seed_df, pd.DataFrame(summary_rows)


def _compute_risk_adaptive_lp_weights(
    risk: pd.Series | np.ndarray,
    base_lp_weight: float,
    risk_tilt: float,
    risk_center: float = 0.5,
) -> np.ndarray:
    risk_values = np.asarray(risk, dtype=float)
    risk_values = np.nan_to_num(risk_values, nan=float(risk_center), posinf=1.0, neginf=0.0)
    risk_values = np.clip(risk_values, 0.0, 1.0)
    weights = float(base_lp_weight) + float(risk_tilt) * (risk_values - float(risk_center))
    return np.clip(weights, 0.0, 1.0)


def evaluate_hybrid_risk_adaptive_blended_scenario_seed_robustness(
    preds: pd.DataFrame,
    train_residuals: pd.DataFrame,
    rated_capacity: float,
    seeds: tuple[int, ...] = (71_000, 71_001, 71_011, 71_021, 71_031),
    pv_model_name: str = "mlp_text",
    rt_model_name: str = "transformer_text",
    anchor_model_name: str = "mlp_text",
    base_lp_weights: tuple[float, ...] = (0.5,),
    risk_tilts: tuple[float, ...] = (-0.5, 0.0, 0.5),
    risk_col: str = "llm_prior_overall_risk_score",
    risk_center: float = 0.5,
    residual_scale: float = 1.0,
    cvar_gamma: float = 0.25,
    deviation_penalty: float = 50.0,
    scenario_count: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_pred = {
        "timestamp_utc",
        "local_date",
        "hour",
        "pv_mw",
        "da_lmp",
        "rt_lmp",
        f"pv_{pv_model_name}",
        f"rt_{rt_model_name}",
        f"pv_{anchor_model_name}",
        risk_col,
    }
    required_residual = {f"pv_{pv_model_name}_residual", f"rt_{rt_model_name}_residual", "hour"}
    if not required_pred.issubset(preds.columns) or not required_residual.issubset(train_residuals.columns):
        return pd.DataFrame(), pd.DataFrame()

    actual = preds["pv_mw"].to_numpy(dtype=float)
    da = preds["da_lmp"].to_numpy(dtype=float)
    rt = preds["rt_lmp"].to_numpy(dtype=float)
    anchor_bid = np.clip(
        np.nan_to_num(preds[f"pv_{anchor_model_name}"].to_numpy(dtype=float), nan=0.0),
        0.0,
        rated_capacity,
    )
    seed_rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        lp_bid_series = pd.Series(index=preds.index, dtype=float)
        failures = 0
        solve_time = 0.0
        for _, day in preds.groupby("local_date", sort=True):
            ordered = day.sort_values("timestamp_utc")
            pv_scenarios, rt_scenarios = build_hybrid_residual_scenarios(
                ordered,
                train_residuals,
                pv_model_name=pv_model_name,
                rt_model_name=rt_model_name,
                rated_capacity=rated_capacity,
                scenario_count=scenario_count,
                rng=rng,
                residual_scale=residual_scale,
            )
            start = time.perf_counter()
            try:
                q_day = solve_cvar_bids(
                    pv_scenarios=pv_scenarios,
                    rt_scenarios=rt_scenarios,
                    da_lmp=ordered["da_lmp"].to_numpy(dtype=float),
                    rated_capacity=rated_capacity,
                    gamma=cvar_gamma,
                    shortage_penalty=deviation_penalty,
                    surplus_penalty=deviation_penalty,
                )
            except RuntimeError:
                failures += 1
                q_day = ordered[f"pv_{pv_model_name}"].to_numpy(dtype=float)
            solve_time += time.perf_counter() - start
            lp_bid_series.loc[ordered.index] = q_day

        lp_bid = np.clip(np.nan_to_num(lp_bid_series.to_numpy(dtype=float), nan=0.0), 0.0, rated_capacity)
        for base_lp_weight in base_lp_weights:
            for risk_tilt in risk_tilts:
                lp_weights = _compute_risk_adaptive_lp_weights(
                    risk=preds[risk_col],
                    base_lp_weight=base_lp_weight,
                    risk_tilt=risk_tilt,
                    risk_center=risk_center,
                )
                q = np.clip(lp_weights * lp_bid + (1.0 - lp_weights) * anchor_bid, 0.0, rated_capacity)
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
                strategy_name = (
                    f"HybridRiskBlend_{pv_model_name}_{rt_model_name}_anchor_{anchor_model_name}"
                    f"_base{float(base_lp_weight):.2f}_tilt{float(risk_tilt):+.2f}_{risk_col}"
                    f"_scale_{residual_scale:.2f}_gamma_{cvar_gamma:.2f}_penalty_{deviation_penalty:g}"
                )
                seed_rows.append(
                    {
                        "strategy": strategy_name,
                        "pv_model": pv_model_name,
                        "rt_model": rt_model_name,
                        "anchor_model": anchor_model_name,
                        "lp_weight": float(base_lp_weight),
                        "base_lp_weight": float(base_lp_weight),
                        "risk_tilt": float(risk_tilt),
                        "risk_col": risk_col,
                        "risk_center": float(risk_center),
                        "realized_lp_weight_mean": float(np.mean(lp_weights)),
                        "realized_lp_weight_min": float(np.min(lp_weights)),
                        "realized_lp_weight_max": float(np.max(lp_weights)),
                        "seed": int(seed),
                        "scenario_count": int(scenario_count),
                        "residual_scale": float(residual_scale),
                        "cvar_gamma": float(cvar_gamma),
                        "deviation_penalty": float(deviation_penalty),
                        "hours": int(len(q)),
                        "total_revenue": float(np.sum(revenue)),
                        "avg_revenue_per_hour": float(np.mean(revenue)),
                        "imbalance_mwh_proxy": float(np.sum(shortage + surplus)),
                        "shortage_mwh_proxy": float(np.sum(shortage)),
                        "surplus_mwh_proxy": float(np.sum(surplus)),
                        "worst_5pct_avg_revenue": float(
                            np.mean(np.sort(revenue)[: max(1, math.ceil(0.05 * len(revenue)))])
                        ),
                        "cvar_95_loss": cvar_loss(-revenue, alpha=0.95),
                        "solve_time_sec": float(solve_time),
                        "optimization_failures": int(failures),
                    }
                )

    seed_df = pd.DataFrame(seed_rows)
    if seed_df.empty:
        return seed_df, pd.DataFrame()

    summary_rows = []
    metric_cols = [
        "total_revenue",
        "avg_revenue_per_hour",
        "imbalance_mwh_proxy",
        "shortage_mwh_proxy",
        "surplus_mwh_proxy",
        "worst_5pct_avg_revenue",
        "cvar_95_loss",
        "realized_lp_weight_mean",
        "realized_lp_weight_min",
        "realized_lp_weight_max",
    ]
    group_cols = ["strategy", "base_lp_weight", "risk_tilt"]
    for (strategy_name, base_lp_weight, risk_tilt), group in seed_df.groupby(group_cols, sort=True):
        summary: dict[str, float | int | str] = {
            "strategy": strategy_name,
            "pv_model": pv_model_name,
            "rt_model": rt_model_name,
            "anchor_model": anchor_model_name,
            "lp_weight": float(base_lp_weight),
            "base_lp_weight": float(base_lp_weight),
            "risk_tilt": float(risk_tilt),
            "risk_col": risk_col,
            "risk_center": float(risk_center),
            "seed_count": int(len(group)),
            "seeds": "|".join(str(seed) for seed in seeds),
            "scenario_count": int(scenario_count),
            "residual_scale": float(residual_scale),
            "cvar_gamma": float(cvar_gamma),
            "deviation_penalty": float(deviation_penalty),
            "hours": int(group["hours"].iloc[0]),
            "solve_time_sec_total": float(group["solve_time_sec"].sum()),
            "optimization_failures_total": int(group["optimization_failures"].sum()),
        }
        for col in metric_cols:
            values = group[col].to_numpy(dtype=float)
            summary[f"{col}_mean"] = float(np.mean(values))
            summary[f"{col}_std"] = float(np.std(values, ddof=0))
            summary[f"{col}_min"] = float(np.min(values))
            summary[f"{col}_max"] = float(np.max(values))
        summary_rows.append(summary)

    return seed_df, pd.DataFrame(summary_rows)


def _normalized_score(values: pd.Series, higher_is_better: bool) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    if not higher_is_better:
        arr = -arr
    scores = np.zeros(len(arr), dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return scores
    lo = float(np.min(arr[finite]))
    hi = float(np.max(arr[finite]))
    if abs(hi - lo) < 1e-12:
        scores[finite] = 0.5
    else:
        scores[finite] = (arr[finite] - lo) / (hi - lo)
    return scores


def _score_hybrid_blend_validation_summary(summary: pd.DataFrame, selection_objective: str) -> pd.DataFrame:
    out = summary.copy()
    if out.empty:
        return out

    if selection_objective == "total_revenue_mean":
        out["validation_selection_score"] = out["total_revenue_mean"].to_numpy(dtype=float)
    elif selection_objective == "cvar_95_loss_mean":
        out["validation_selection_score"] = -out["cvar_95_loss_mean"].to_numpy(dtype=float)
    elif selection_objective == "balanced_revenue_cvar":
        components = [
            _normalized_score(out["total_revenue_mean"], higher_is_better=True),
            _normalized_score(out["total_revenue_min"], higher_is_better=True),
            _normalized_score(out["cvar_95_loss_mean"], higher_is_better=False),
            _normalized_score(out["cvar_95_loss_max"], higher_is_better=False),
        ]
        out["validation_selection_score"] = np.mean(np.vstack(components), axis=0)
    else:
        raise ValueError(f"unknown selection_objective: {selection_objective}")

    ranked = out.sort_values(
        ["validation_selection_score", "total_revenue_mean", "total_revenue_min", "cvar_95_loss_mean", "lp_weight"],
        ascending=[False, False, False, True, True],
    )
    out["validation_selection_rank"] = np.nan
    out.loc[ranked.index, "validation_selection_rank"] = np.arange(1, len(ranked) + 1)
    out["selection_objective"] = selection_objective
    out["selected_by_validation"] = out["validation_selection_rank"] == 1
    return out


def evaluate_validation_selected_hybrid_blend(
    validation_preds: pd.DataFrame,
    validation_train_residuals: pd.DataFrame,
    test_preds: pd.DataFrame,
    test_train_residuals: pd.DataFrame,
    rated_capacity: float,
    seeds: tuple[int, ...] = (71_000, 71_001, 71_011, 71_021, 71_031),
    pv_model_name: str = "mlp_text",
    rt_model_name: str = "transformer_text",
    anchor_model_name: str = "mlp_text",
    lp_weights: tuple[float, ...] = (0.25, 0.5, 0.75),
    residual_scale: float = 1.0,
    cvar_gamma: float = 0.25,
    deviation_penalty: float = 50.0,
    scenario_count: int = 20,
    selection_objective: str = "balanced_revenue_cvar",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_seed_rows, validation_summary = evaluate_hybrid_blended_scenario_seed_robustness(
        preds=validation_preds,
        train_residuals=validation_train_residuals,
        rated_capacity=rated_capacity,
        seeds=seeds,
        pv_model_name=pv_model_name,
        rt_model_name=rt_model_name,
        anchor_model_name=anchor_model_name,
        lp_weights=lp_weights,
        residual_scale=residual_scale,
        cvar_gamma=cvar_gamma,
        deviation_penalty=deviation_penalty,
        scenario_count=scenario_count,
    )
    validation_summary = _score_hybrid_blend_validation_summary(validation_summary, selection_objective)
    if validation_summary.empty:
        return validation_seed_rows, validation_summary, pd.DataFrame(), pd.DataFrame()

    selected = validation_summary.sort_values("validation_selection_rank").iloc[0]
    selected_lp_weight = float(selected["lp_weight"])
    test_seed_rows, test_summary = evaluate_hybrid_blended_scenario_seed_robustness(
        preds=test_preds,
        train_residuals=test_train_residuals,
        rated_capacity=rated_capacity,
        seeds=seeds,
        pv_model_name=pv_model_name,
        rt_model_name=rt_model_name,
        anchor_model_name=anchor_model_name,
        lp_weights=(selected_lp_weight,),
        residual_scale=residual_scale,
        cvar_gamma=cvar_gamma,
        deviation_penalty=deviation_penalty,
        scenario_count=scenario_count,
    )

    selected_fields = [
        "validation_selection_score",
        "total_revenue_mean",
        "total_revenue_min",
        "cvar_95_loss_mean",
        "cvar_95_loss_max",
        "worst_5pct_avg_revenue_mean",
        "worst_5pct_avg_revenue_min",
    ]
    for frame in (test_seed_rows, test_summary):
        if frame.empty:
            continue
        frame["selected_lp_weight"] = selected_lp_weight
        frame["selection_objective"] = selection_objective
        for col in selected_fields:
            if col in selected:
                out_col = col if col.startswith("validation_") else f"validation_{col}"
                frame[out_col] = selected[col]
    return validation_seed_rows, validation_summary, test_seed_rows, test_summary


def evaluate_validation_selected_hybrid_risk_adaptive_blend(
    validation_preds: pd.DataFrame,
    validation_train_residuals: pd.DataFrame,
    test_preds: pd.DataFrame,
    test_train_residuals: pd.DataFrame,
    rated_capacity: float,
    seeds: tuple[int, ...] = (71_000, 71_001, 71_011, 71_021, 71_031),
    pv_model_name: str = "mlp_text",
    rt_model_name: str = "transformer_text",
    anchor_model_name: str = "mlp_text",
    base_lp_weights: tuple[float, ...] = (0.5,),
    risk_tilts: tuple[float, ...] = (-0.5, 0.0, 0.5),
    risk_col: str = "llm_prior_overall_risk_score",
    risk_center: float = 0.5,
    residual_scale: float = 1.0,
    cvar_gamma: float = 0.25,
    deviation_penalty: float = 50.0,
    scenario_count: int = 20,
    selection_objective: str = "balanced_revenue_cvar",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_seed_rows, validation_summary = evaluate_hybrid_risk_adaptive_blended_scenario_seed_robustness(
        preds=validation_preds,
        train_residuals=validation_train_residuals,
        rated_capacity=rated_capacity,
        seeds=seeds,
        pv_model_name=pv_model_name,
        rt_model_name=rt_model_name,
        anchor_model_name=anchor_model_name,
        base_lp_weights=base_lp_weights,
        risk_tilts=risk_tilts,
        risk_col=risk_col,
        risk_center=risk_center,
        residual_scale=residual_scale,
        cvar_gamma=cvar_gamma,
        deviation_penalty=deviation_penalty,
        scenario_count=scenario_count,
    )
    validation_summary = _score_hybrid_blend_validation_summary(validation_summary, selection_objective)
    if validation_summary.empty:
        return validation_seed_rows, validation_summary, pd.DataFrame(), pd.DataFrame()

    selected = validation_summary.sort_values("validation_selection_rank").iloc[0]
    selected_base_lp_weight = float(selected["base_lp_weight"])
    selected_risk_tilt = float(selected["risk_tilt"])
    test_seed_rows, test_summary = evaluate_hybrid_risk_adaptive_blended_scenario_seed_robustness(
        preds=test_preds,
        train_residuals=test_train_residuals,
        rated_capacity=rated_capacity,
        seeds=seeds,
        pv_model_name=pv_model_name,
        rt_model_name=rt_model_name,
        anchor_model_name=anchor_model_name,
        base_lp_weights=(selected_base_lp_weight,),
        risk_tilts=(selected_risk_tilt,),
        risk_col=risk_col,
        risk_center=risk_center,
        residual_scale=residual_scale,
        cvar_gamma=cvar_gamma,
        deviation_penalty=deviation_penalty,
        scenario_count=scenario_count,
    )

    selected_fields = [
        "validation_selection_score",
        "base_lp_weight",
        "risk_tilt",
        "risk_col",
        "total_revenue_mean",
        "total_revenue_min",
        "cvar_95_loss_mean",
        "cvar_95_loss_max",
        "worst_5pct_avg_revenue_mean",
        "worst_5pct_avg_revenue_min",
    ]
    for frame in (test_seed_rows, test_summary):
        if frame.empty:
            continue
        frame["selected_base_lp_weight"] = selected_base_lp_weight
        frame["selected_risk_tilt"] = selected_risk_tilt
        frame["selection_objective"] = selection_objective
        for col in selected_fields:
            if col in selected:
                out_col = col if col.startswith("validation_") else f"validation_{col}"
                frame[out_col] = selected[col]
    return validation_seed_rows, validation_summary, test_seed_rows, test_summary


def _metric_value(metrics: pd.DataFrame, target: str, subset: str, model: str, column: str) -> float:
    row = metrics[(metrics["target"] == target) & (metrics["subset"] == subset) & (metrics["model"] == model)]
    if row.empty:
        return math.nan
    return float(row.iloc[0][column])


def _bidding_value(bidding: pd.DataFrame, strategy: str, column: str) -> float:
    row = bidding[bidding["strategy"] == strategy]
    if row.empty:
        return math.nan
    return float(row.iloc[0][column])


def build_ablation_summary(metrics: pd.DataFrame, bidding: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_lower_better(name: str, target: str, subset: str, baseline: str, candidate: str, column: str = "rmse") -> None:
        baseline_value = _metric_value(metrics, target, subset, baseline, column)
        candidate_value = _metric_value(metrics, target, subset, candidate, column)
        rows.append(
            {
                "experiment": name,
                "metric": f"{target}_{subset}_{column}",
                "baseline": baseline,
                "candidate": candidate,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "relative_improvement_pct": percent_improvement(baseline_value, candidate_value),
                "higher_is_better": False,
            }
        )

    def add_higher_better(name: str, baseline: str, candidate: str, column: str = "total_revenue") -> None:
        baseline_value = _bidding_value(bidding, baseline, column)
        candidate_value = _bidding_value(bidding, candidate, column)
        improvement = (
            float((candidate_value - baseline_value) / abs(baseline_value) * 100.0)
            if np.isfinite(baseline_value) and abs(baseline_value) > 1e-12
            else math.nan
        )
        rows.append(
            {
                "experiment": name,
                "metric": column,
                "baseline": baseline,
                "candidate": candidate,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "relative_improvement_pct": improvement,
                "higher_is_better": True,
            }
        )

    add_lower_better("weather_text_ridge_pv_all", "pv", "all", "pv_ridge", "pv_ridge_text")
    add_lower_better("mlp_text_vs_caiso_public_solar_pv_all", "pv", "all", "pv_caiso_solar_dam", "pv_mlp_text")
    add_lower_better("weather_text_ridge_pv_extreme_solar", "pv", "extreme_solar_hours", "pv_ridge", "pv_ridge_text")
    add_lower_better("deep_mlp_text_pv_all", "pv", "all", "pv_ridge_text", "pv_mlp_text")
    add_lower_better("deep_mlp_text_pv_extreme_solar", "pv", "extreme_solar_hours", "pv_ridge_text", "pv_mlp_text")
    add_lower_better("transformer_text_rt_price_all", "rt_price", "all", "rt_ridge", "rt_transformer_text")
    add_lower_better("llm_vs_rule_ridge_pv_all", "pv", "all", "pv_ridge_text", "pv_ridge_llm")
    add_lower_better("llm_vs_rule_mlp_pv_all", "pv", "all", "pv_mlp_text", "pv_mlp_llm")
    add_lower_better(
        "llm_vs_rule_transformer_rt_price_all",
        "rt_price",
        "all",
        "rt_transformer_text",
        "rt_transformer_llm",
    )
    add_lower_better("combined_vs_rule_ridge_pv_all", "pv", "all", "pv_ridge_text", "pv_ridge_text_llm")
    add_lower_better("combined_vs_llm_ridge_pv_all", "pv", "all", "pv_ridge_llm", "pv_ridge_text_llm")
    add_lower_better("combined_vs_rule_mlp_pv_all", "pv", "all", "pv_mlp_text", "pv_mlp_text_llm")
    add_lower_better("combined_vs_llm_mlp_pv_all", "pv", "all", "pv_mlp_llm", "pv_mlp_text_llm")
    add_lower_better(
        "tuned_combined_vs_rule_mlp_pv_all",
        "pv",
        "all",
        "pv_mlp_text",
        "pv_mlp_text_llm_tuned",
    )
    add_lower_better(
        "tuned_combined_vs_llm_mlp_pv_all",
        "pv",
        "all",
        "pv_mlp_llm",
        "pv_mlp_text_llm_tuned",
    )
    add_lower_better(
        "combined_vs_rule_transformer_rt_price_all",
        "rt_price",
        "all",
        "rt_transformer_text",
        "rt_transformer_text_llm",
    )
    add_higher_better("bidding_mlp_text_vs_ridge_text", "S4_ridge_text_deterministic", "S11_mlp_text_deterministic")
    add_higher_better(
        "bidding_mlp_text_vs_caiso_public_solar",
        "S0_caiso_public_solar_forecast",
        "S11_mlp_text_deterministic",
    )
    add_higher_better(
        "bidding_residual_quantile_vs_mlp_text",
        "S11_mlp_text_deterministic",
        "S25_mlp_text_residual_quantile_10pct",
    )
    add_higher_better(
        "bidding_residual_quantile_vs_ridge_text_lp",
        "S7_ridge_text_stochastic_LP",
        "S25_mlp_text_residual_quantile_10pct",
    )
    add_higher_better(
        "bidding_ridge_text_llm_vs_ridge_text",
        "S4_ridge_text_deterministic",
        "S21_ridge_text_llm_deterministic",
    )
    add_higher_better("bidding_mlp_llm_vs_mlp_text", "S11_mlp_text_deterministic", "S16_mlp_llm_deterministic")
    add_higher_better(
        "bidding_mlp_text_llm_vs_mlp_text",
        "S11_mlp_text_deterministic",
        "S18_mlp_text_llm_deterministic",
    )
    add_higher_better(
        "bidding_mlp_text_llm_tuned_vs_mlp_text",
        "S11_mlp_text_deterministic",
        "S20_mlp_text_llm_tuned_deterministic",
    )
    add_higher_better("scenario_lp_vs_ridge_text_deterministic", "S4_ridge_text_deterministic", "S7_ridge_text_stochastic_LP")
    add_higher_better(
        "scenario_lp_ridge_text_llm_vs_ridge_text_lp",
        "S7_ridge_text_stochastic_LP",
        "S22_ridge_text_llm_stochastic_LP",
    )
    return pd.DataFrame(rows)


def select_extreme_case_days(preds: pd.DataFrame, max_cases: int = 5) -> pd.DataFrame:
    if "pv_mlp_text" not in preds.columns or "pv_ridge_text" not in preds.columns:
        return pd.DataFrame()
    candidate = preds[(preds["has_extreme_event"] == 1) & (preds["is_solar_hour"] == 1)].copy()
    rows = []
    for local_date, day in candidate.groupby("local_date", sort=True):
        ridge_rmse = regression_metrics(day["pv_mw"], day["pv_ridge_text"])["rmse"]
        mlp_rmse = regression_metrics(day["pv_mw"], day["pv_mlp_text"])["rmse"]
        rows.append(
            {
                "local_date": local_date,
                "event_types": ";".join(sorted(set(str(x) for x in day.get("event_types", pd.Series([""])).fillna("") if str(x)))),
                "solar_hours": int(len(day)),
                "ridge_text_pv_rmse": ridge_rmse,
                "mlp_text_pv_rmse": mlp_rmse,
                "mlp_vs_ridge_rmse_improvement_pct": percent_improvement(ridge_rmse, mlp_rmse),
                "actual_pv_mwh": float(day["pv_mw"].sum()),
                "max_rt_lmp": float(day["rt_lmp"].max()),
                "mean_rt_lmp": float(day["rt_lmp"].mean()),
                "prior_weather_risk": float(day["wx_prior_overall_risk_score"].mean())
                if "wx_prior_overall_risk_score" in day.columns
                else math.nan,
            }
        )
    cases = pd.DataFrame(rows)
    if cases.empty:
        return cases
    return cases.sort_values(["mlp_vs_ridge_rmse_improvement_pct", "ridge_text_pv_rmse"], ascending=[False, False]).head(max_cases)


def build_extreme_case_hourly(preds: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame()
    dates = set(cases["local_date"].astype(str))
    cols = [
        "timestamp_utc",
        "local_date",
        "hour",
        "event_types",
        "pv_mw",
        "pv_ridge_text",
        "pv_mlp_text",
        "rt_lmp",
        "rt_transformer_text",
        "da_lmp",
        "wx_prior_overall_risk_score",
        "wx_prior_cloud_score",
        "wx_prior_rain_score",
        "wx_prior_storm_score",
    ]
    cols = [col for col in cols if col in preds.columns]
    return preds[preds["local_date"].astype(str).isin(dates)][cols].sort_values(["local_date", "hour"])


def bidding_backtest(preds: pd.DataFrame, rated_capacity: float, train_residuals: pd.DataFrame | None = None) -> pd.DataFrame:
    actual = preds["pv_mw"].to_numpy()
    da = preds["da_lmp"].to_numpy()
    rt = preds["rt_lmp"].to_numpy()
    text_col = "pv_gbr_text" if "pv_gbr_text" in preds.columns else "pv_gbr"
    gbr_error = preds[text_col] - preds["pv_mw"]
    hour_residual_std = preds.assign(error=gbr_error).groupby("hour")["error"].std().fillna(gbr_error.std())
    risk_buffer = preds["hour"].map(hour_residual_std).to_numpy() * 0.5

    strategies = {
        "S1_persistence_deterministic": preds["pv_persistence"].to_numpy(),
        "S2_ridge_numerical_deterministic": preds["pv_ridge"].to_numpy(),
        "S3_gbr_numerical_deterministic": preds["pv_gbr"].to_numpy(),
        "S4_ridge_text_deterministic": preds.get("pv_ridge_text", preds["pv_ridge"]).to_numpy(),
        "S5_gbr_text_deterministic": preds.get("pv_gbr_text", preds["pv_gbr"]).to_numpy(),
        "S6_gbr_text_risk_averse_quantile_proxy": preds.get("pv_gbr_text", preds["pv_gbr"]).to_numpy() - risk_buffer,
        "Oracle_actual_pv": actual,
    }
    if "pv_caiso_solar_dam" in preds.columns:
        strategies["S0_caiso_public_solar_forecast"] = preds["pv_caiso_solar_dam"].to_numpy()
    optional_deep_strategies = [
        ("S21_ridge_text_llm_deterministic", "pv_ridge_text_llm"),
        ("S11_mlp_text_deterministic", "pv_mlp_text"),
        ("S12_lstm_text_deterministic", "pv_lstm_text"),
        ("S13_gru_text_deterministic", "pv_gru_text"),
        ("S14_transformer_text_deterministic", "pv_transformer_text"),
        ("S15_mlp_resid_text_deterministic", "pv_mlp_resid_text"),
        ("S16_mlp_llm_deterministic", "pv_mlp_llm"),
        ("S23_mlp_llm_rule_deterministic", "pv_mlp_llm_rule"),
        ("S17_transformer_llm_deterministic", "pv_transformer_llm"),
        ("S18_mlp_text_llm_deterministic", "pv_mlp_text_llm"),
        ("S19_transformer_text_llm_deterministic", "pv_transformer_text_llm"),
        ("S20_mlp_text_llm_tuned_deterministic", "pv_mlp_text_llm_tuned"),
    ]
    for strategy_name, pred_col in optional_deep_strategies:
        if pred_col in preds.columns:
            strategies[strategy_name] = preds[pred_col].to_numpy()
    if (
        train_residuals is not None
        and "pv_mlp_text" in preds.columns
        and "pv_mlp_text_residual" in train_residuals.columns
    ):
        strategies["S25_mlp_text_residual_quantile_10pct"] = quantile_residual_bid(
            preds,
            train_residuals,
            pred_col="pv_mlp_text",
            residual_col="pv_mlp_text_residual",
            quantile=0.10,
            rated_capacity=rated_capacity,
        )

    rows = []
    for name, bid in strategies.items():
        q = np.clip(np.nan_to_num(bid, nan=0.0), 0.0, rated_capacity)
        revenue = settlement_revenue(actual, q, da, rt)
        shortage = np.maximum(q - actual, 0.0)
        surplus = np.maximum(actual - q, 0.0)
        rows.append(
            {
                "strategy": name,
                "hours": int(len(q)),
                "total_revenue": float(np.sum(revenue)),
                "avg_revenue_per_hour": float(np.mean(revenue)),
                "imbalance_mwh_proxy": float(np.sum(shortage + surplus)),
                "shortage_mwh_proxy": float(np.sum(shortage)),
                "surplus_mwh_proxy": float(np.sum(surplus)),
                "worst_5pct_avg_revenue": float(np.mean(np.sort(revenue)[: max(1, math.ceil(0.05 * len(revenue)))])),
                "cvar_95_loss": cvar_loss(-revenue, alpha=0.95),
                "solve_time_sec": 0.0,
                "optimization_failures": 0,
            }
        )

    if train_residuals is not None and {"pv_ridge_text", "rt_ridge_text"}.issubset(preds.columns):
        rng = np.random.default_rng(42)
        formal_strategies = [
            ("S7_ridge_text_stochastic_LP", "ridge_text", 0.0),
            ("S8_ridge_text_CVaR_LP_gamma_0.25", "ridge_text", 0.25),
            ("S9_ridge_text_CVaR_LP_gamma_1.00", "ridge_text", 1.0),
            ("S10_ridge_text_CVaR_LP_gamma_2.00", "ridge_text", 2.0),
        ]
        if {
            "pv_ridge_text_llm",
            "rt_ridge_text_llm",
            "pv_ridge_text_llm_residual",
            "rt_ridge_text_llm_residual",
        }.issubset(set(preds.columns) | set(train_residuals.columns)):
            formal_strategies.extend(
                [
                    ("S22_ridge_text_llm_stochastic_LP", "ridge_text_llm", 0.0),
                    ("S23_ridge_text_llm_CVaR_LP_gamma_0.25", "ridge_text_llm", 0.25),
                    ("S24_ridge_text_llm_CVaR_LP_gamma_1.00", "ridge_text_llm", 1.0),
                ]
            )
        for strategy_name, model_name, gamma in formal_strategies:
            bid_series = pd.Series(index=preds.index, dtype=float)
            failures = 0
            solve_time = 0.0
            for _, day in preds.groupby("local_date", sort=True):
                ordered = day.sort_values("timestamp_utc")
                pv_scenarios, rt_scenarios = build_residual_scenarios(
                    ordered,
                    train_residuals,
                    model_name=model_name,
                    rated_capacity=rated_capacity,
                    scenario_count=20,
                    rng=rng,
                )
                start = time.perf_counter()
                try:
                    q = solve_cvar_bids(
                        pv_scenarios=pv_scenarios,
                        rt_scenarios=rt_scenarios,
                        da_lmp=ordered["da_lmp"].to_numpy(dtype=float),
                        rated_capacity=rated_capacity,
                        gamma=gamma,
                    )
                except RuntimeError:
                    failures += 1
                    q = ordered[f"pv_{model_name}"].to_numpy(dtype=float)
                solve_time += time.perf_counter() - start
                bid_series.loc[ordered.index] = q

            q = np.clip(np.nan_to_num(bid_series.to_numpy(dtype=float), nan=0.0), 0.0, rated_capacity)
            revenue = settlement_revenue(actual, q, da, rt)
            shortage = np.maximum(q - actual, 0.0)
            surplus = np.maximum(actual - q, 0.0)
            rows.append(
                {
                    "strategy": strategy_name,
                    "hours": int(len(q)),
                    "total_revenue": float(np.sum(revenue)),
                    "avg_revenue_per_hour": float(np.mean(revenue)),
                    "imbalance_mwh_proxy": float(np.sum(shortage + surplus)),
                    "shortage_mwh_proxy": float(np.sum(shortage)),
                    "surplus_mwh_proxy": float(np.sum(surplus)),
                    "worst_5pct_avg_revenue": float(np.mean(np.sort(revenue)[: max(1, math.ceil(0.05 * len(revenue)))])),
                    "cvar_95_loss": cvar_loss(-revenue, alpha=0.95),
                    "solve_time_sec": float(solve_time),
                    "optimization_failures": int(failures),
                }
            )
    return pd.DataFrame(rows)


def write_markdown_summary(
    path: Path,
    audit: dict[str, float | int | str],
    metrics: pd.DataFrame,
    bidding: pd.DataFrame,
    scenario_quality: pd.DataFrame,
    seed_robustness_summary: pd.DataFrame,
    ablation: pd.DataFrame,
    extreme_cases: pd.DataFrame,
) -> None:
    pv_metrics = metrics[(metrics["target"] == "pv") & (metrics["subset"] == "all")].sort_values("rmse")
    pv_extreme = metrics[(metrics["target"] == "pv") & (metrics["subset"] == "extreme")].sort_values("rmse")
    price_metrics = metrics[(metrics["target"] == "rt_price") & (metrics["subset"] == "all")].sort_values("rmse")
    spike_metrics = metrics[metrics["target"] == "rt_spike"].copy()
    lines = [
        "# Baseline Experiment Summary",
        "",
        "## Data Audit",
        "",
    ]
    for key, value in audit.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## PV Forecasting, All Test Hours",
            "",
            pv_metrics.to_markdown(index=False),
            "",
            "## PV Forecasting, Extreme-Weather Test Hours",
            "",
            pv_extreme.to_markdown(index=False),
            "",
            "## RT Price Forecasting",
            "",
            price_metrics.to_markdown(index=False),
            "",
            "## RT Price Spike Detection",
            "",
            "For `target=rt_spike`, columns map as: `mae=precision`, `rmse=recall`, `nrmse=F1`, `mape=AUC`.",
            "",
            spike_metrics.to_markdown(index=False),
            "",
            "## Bidding Backtest",
            "",
            bidding.to_markdown(index=False),
            "",
            "## Calibrated S22 Seed Robustness",
            "",
            seed_robustness_summary.to_markdown(index=False)
            if not seed_robustness_summary.empty
            else "No calibrated S22 seed robustness rows generated.",
            "",
            "## Scenario Quality",
            "",
            scenario_quality.to_markdown(index=False) if not scenario_quality.empty else "No scenario quality rows generated.",
            "",
            "## Ablation and Sensitivity Summary",
            "",
            ablation.to_markdown(index=False) if not ablation.empty else "No ablation rows generated.",
            "",
            "## Extreme Weather Case Days",
            "",
            extreme_cases.to_markdown(index=False) if not extreme_cases.empty else "No extreme case rows generated.",
            "",
            "## Interpretation Guardrails",
            "",
            "- `ridge` and `gbr` are numerical-only baselines; `*_text` variants add deterministic NWS keyword-risk proxy features shifted by one day.",
            "- `*_llm` variants add structured NWS narrative-risk features shifted by one day; check `nws_llm_feature_source` in the audit before calling them true LLM outputs.",
            "- `mlp`, `lstm_text`, `gru_text`, and `transformer_text` are neural/deep-learning baselines; sequence models use 24-hour trailing windows.",
            "- `mlp_resid_text` is a residual deep-learning model that learns corrections on top of the leakage-controlled `ridge_text` forecast.",
            "- If `nws_llm_feature_source=heuristic_structured_proxy`, the LLM columns are an offline structured proxy, not paid/API model outputs.",
            "- `S7` solves a daily risk-neutral scenario LP; `S8`-`S10` add linear CVaR terms with increasing risk aversion.",
            "- Settlement uses a symmetric 50 $/MWh proxy deviation penalty to discourage virtual-arbitrage behavior in physical PV bids.",
            "- HRRR has a few retained source-missing rows, but model features use `status=ok` rows only.",
            "- Price data availability starts in 2023-01, so bidding experiments use 2023 for training and 2024 for testing.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    output_dir: Path = Path("results"),
    processed_dir: Path = Path("data/processed"),
    data_suffix: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    master, audit = build_master_table(processed_dir, data_suffix=data_suffix)
    master_path = processed_dir / "master_hourly_caiso_noaa_2023-01-19_2024-12-31.csv"
    master.to_csv(master_path, index=False)
    preds, metrics, train_residuals, meta = fit_predict_models(master)
    audit.update(meta)
    bidding = bidding_backtest(preds, rated_capacity=float(meta["rated_capacity_mw"]), train_residuals=train_residuals)
    scenario_quality = evaluate_scenario_quality(
        preds,
        train_residuals,
        rated_capacity=float(meta["rated_capacity_mw"]),
    )
    ridge_text_llm_scenario_calibration, bidding_ridge_text_llm_calibrated = calibrate_ridge_text_llm_scenarios(
        preds,
        train_residuals,
        rated_capacity=float(meta["rated_capacity_mw"]),
    )
    (
        bidding_ridge_text_llm_seed_robustness,
        bidding_ridge_text_llm_seed_robustness_summary,
    ) = evaluate_calibrated_s22_seed_robustness(
        preds,
        train_residuals,
        rated_capacity=float(meta["rated_capacity_mw"]),
    )
    ablation = build_ablation_summary(metrics, bidding)
    extreme_cases = select_extreme_case_days(preds)
    extreme_case_hourly = build_extreme_case_hourly(preds, extreme_cases)

    pd.DataFrame([audit]).to_csv(output_dir / "data_audit.csv", index=False)
    preds.to_csv(output_dir / "test_predictions.csv", index=False)
    train_residuals.to_csv(output_dir / "train_residuals.csv", index=False)
    metrics.to_csv(output_dir / "forecast_metrics.csv", index=False)
    bidding.to_csv(output_dir / "bidding_metrics.csv", index=False)
    scenario_quality.to_csv(output_dir / "scenario_quality.csv", index=False)
    ridge_text_llm_scenario_calibration.to_csv(output_dir / "ridge_text_llm_scenario_calibration.csv", index=False)
    bidding_ridge_text_llm_calibrated.to_csv(output_dir / "bidding_ridge_text_llm_calibrated.csv", index=False)
    bidding_ridge_text_llm_seed_robustness.to_csv(
        output_dir / "bidding_ridge_text_llm_seed_robustness.csv", index=False
    )
    bidding_ridge_text_llm_seed_robustness_summary.to_csv(
        output_dir / "bidding_ridge_text_llm_seed_robustness_summary.csv", index=False
    )
    ablation.to_csv(output_dir / "ablation_summary.csv", index=False)
    extreme_cases.to_csv(output_dir / "extreme_case_days.csv", index=False)
    extreme_case_hourly.to_csv(output_dir / "extreme_case_hourly.csv", index=False)
    write_markdown_summary(
        output_dir / "baseline_experiment_summary.md",
        audit,
        metrics,
        bidding,
        scenario_quality,
        bidding_ridge_text_llm_seed_robustness_summary,
        ablation,
        extreme_cases,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline CAISO/NOAA forecasting and bidding experiments.")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--data-suffix", help="Processed data suffix such as 2023-01-01_2025-01-01.")
    args = parser.parse_args()
    run(args.output, args.processed_dir, data_suffix=args.data_suffix)


if __name__ == "__main__":
    main()
