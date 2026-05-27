from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RULE_SELECTED_STEM = "bidding_hybrid_blended_mlp_rule_core_transformer_text_anchor_mlp_rule_core_validation_selected"
LLM_SELECTED_STEM = "bidding_hybrid_blended_mlp_llm_rule_cloud_transformer_llm_rule_anchor_mlp_rule_core_validation_selected"
RULE_FIXED_STEM = "bidding_hybrid_blended_mlp_rule_core_transformer_text_anchor_mlp_rule_core_10seed_penalty50"
LLM_FIXED_STEM = "bidding_hybrid_blended_mlp_llm_rule_cloud_transformer_llm_rule_anchor_mlp_rule_core_10seed_penalty50"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required upstream result is missing: {path}")
    return pd.read_csv(path)


def _summary_row(frame: pd.DataFrame, lp_weight: float | None = None) -> pd.Series:
    if lp_weight is None:
        if frame.empty:
            raise ValueError("summary frame is empty")
        return frame.iloc[0]
    match = frame[np.isclose(pd.to_numeric(frame["lp_weight"], errors="coerce"), lp_weight)]
    if match.empty:
        raise ValueError(f"summary frame has no lp_weight={lp_weight:.2f}")
    return match.iloc[0]


def _display_row(
    *,
    comparison: str,
    strategy: str,
    scenario_source: str,
    row: pd.Series,
    reference_row: pd.Series,
) -> dict[str, object]:
    value_musd = float(row["total_revenue_mean"]) / 1_000_000.0
    cvar_kusd_h = float(row["cvar_95_loss_mean"]) / 1_000.0
    imbalance_gwh = float(row["imbalance_mwh_proxy_mean"]) / 1_000.0
    ref_value_musd = float(reference_row["total_revenue_mean"]) / 1_000_000.0
    ref_cvar_kusd_h = float(reference_row["cvar_95_loss_mean"]) / 1_000.0
    ref_imbalance_gwh = float(reference_row["imbalance_mwh_proxy_mean"]) / 1_000.0
    return {
        "comparison": comparison,
        "strategy": strategy,
        "scenario_source": scenario_source,
        "lp_weight": float(row["lp_weight"]),
        "value_musd": value_musd,
        "cvar95_loss_kusd_h": cvar_kusd_h,
        "imbalance_gwh": imbalance_gwh,
        "value_delta_musd_vs_rule": value_musd - ref_value_musd,
        "cvar_delta_kusd_h_vs_rule": ref_cvar_kusd_h - cvar_kusd_h,
        "imbalance_delta_gwh_vs_rule": ref_imbalance_gwh - imbalance_gwh,
        "seed_count": int(row["seed_count"]),
    }


def build_downstream_summary(results_dir: Path, date_tag: str) -> pd.DataFrame:
    rule_selected = _read_required(results_dir / f"{RULE_SELECTED_STEM}_test_summary.csv")
    llm_selected = _read_required(results_dir / f"{LLM_SELECTED_STEM}_test_summary.csv")
    rule_fixed = _read_required(results_dir / f"{RULE_FIXED_STEM}_summary.csv")
    llm_fixed = _read_required(results_dir / f"{LLM_FIXED_STEM}_summary.csv")

    rule_selected_row = _summary_row(rule_selected)
    llm_selected_row = _summary_row(llm_selected)
    pure_llm_row = _summary_row(llm_fixed, lp_weight=1.0)
    rule_fixed_row = _summary_row(rule_fixed, lp_weight=0.5)
    llm_fixed_row = _summary_row(llm_fixed, lp_weight=0.5)

    rows = [
        _display_row(
            comparison="validation_selected",
            strategy="Rule-core hybrid blend",
            scenario_source="PV MLP rule-core + RT Transformer rule-text",
            row=rule_selected_row,
            reference_row=rule_selected_row,
        ),
        _display_row(
            comparison="validation_selected",
            strategy="Pure LLM cloud-rule LP",
            scenario_source="PV MLP LLM rule-cloud + RT Transformer LLM-rule",
            row=pure_llm_row,
            reference_row=rule_selected_row,
        ),
        _display_row(
            comparison="validation_selected",
            strategy="LLM cloud-rule hybrid blend",
            scenario_source="PV MLP LLM rule-cloud + RT Transformer LLM-rule",
            row=llm_selected_row,
            reference_row=rule_selected_row,
        ),
        _display_row(
            comparison="fixed_w_0.50",
            strategy="Rule-core hybrid blend",
            scenario_source="PV MLP rule-core + RT Transformer rule-text",
            row=rule_fixed_row,
            reference_row=rule_fixed_row,
        ),
        _display_row(
            comparison="fixed_w_0.50",
            strategy="LLM cloud-rule hybrid blend",
            scenario_source="PV MLP LLM rule-cloud + RT Transformer LLM-rule",
            row=llm_fixed_row,
            reference_row=rule_fixed_row,
        ),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / f"selected_cloud_rule_downstream_summary_{date_tag}.csv", index=False)
    return summary


def build_weight_sensitivity(results_dir: Path, date_tag: str) -> pd.DataFrame:
    rule_fixed = _read_required(results_dir / f"{RULE_FIXED_STEM}_summary.csv")
    llm_fixed = _read_required(results_dir / f"{LLM_FIXED_STEM}_summary.csv")
    merged = rule_fixed.merge(llm_fixed, on="lp_weight", suffixes=("_rule", "_llm"))
    out = pd.DataFrame(
        {
            "lp_weight": merged["lp_weight"],
            "total_revenue_mean_rule": merged["total_revenue_mean_rule"],
            "total_revenue_mean_llm": merged["total_revenue_mean_llm"],
            "value_delta_musd": (merged["total_revenue_mean_llm"] - merged["total_revenue_mean_rule"]) / 1_000_000.0,
            "cvar_95_loss_mean_rule": merged["cvar_95_loss_mean_rule"],
            "cvar_95_loss_mean_llm": merged["cvar_95_loss_mean_llm"],
            "cvar95_loss_reduction_kusd_h": (merged["cvar_95_loss_mean_rule"] - merged["cvar_95_loss_mean_llm"]) / 1_000.0,
            "imbalance_mwh_proxy_mean_rule": merged["imbalance_mwh_proxy_mean_rule"],
            "imbalance_mwh_proxy_mean_llm": merged["imbalance_mwh_proxy_mean_llm"],
            "imbalance_reduction_gwh": (
                merged["imbalance_mwh_proxy_mean_rule"] - merged["imbalance_mwh_proxy_mean_llm"]
            )
            / 1_000.0,
        }
    ).sort_values("lp_weight")
    out.to_csv(results_dir / f"selected_cloud_rule_downstream_weight_sensitivity_{date_tag}.csv", index=False)
    return out


def _paired_summary_row(metric: str, values: pd.Series, unit: str) -> dict[str, object]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return {
            "metric": metric,
            "mean": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "unit": unit,
            "positive_seed_count": 0,
            "seed_count": 0,
            "paired_t_p_value": np.nan,
        }
    mean = float(np.mean(clean))
    if len(clean) > 1:
        sem = float(stats.sem(clean))
        half_width = float(stats.t.ppf(0.975, len(clean) - 1) * sem)
        p_value = float(stats.ttest_1samp(clean, popmean=0.0).pvalue)
    else:
        half_width = 0.0
        p_value = np.nan
    return {
        "metric": metric,
        "mean": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "unit": unit,
        "positive_seed_count": int(np.sum(clean > 0.0)),
        "seed_count": int(len(clean)),
        "paired_t_p_value": p_value,
    }


def build_paired_seed_tables(results_dir: Path, date_tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rule_seed = _read_required(results_dir / f"{RULE_SELECTED_STEM}_test_seed.csv")
    llm_seed = _read_required(results_dir / f"{LLM_SELECTED_STEM}_test_seed.csv")
    merged = rule_seed.merge(llm_seed, on="seed", suffixes=("_rule", "_llm"))
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
    summary = pd.DataFrame(
        [
            _paired_summary_row("Value gain", deltas["value_delta_musd"], "M USD"),
            _paired_summary_row("CVaR95 loss reduction", deltas["cvar95_loss_reduction_kusd_h"], "k USD/h"),
            _paired_summary_row("Imbalance reduction", deltas["imbalance_reduction_gwh"], "GWh"),
        ]
    )
    deltas.to_csv(results_dir / f"selected_cloud_rule_downstream_paired_seed_deltas_{date_tag}.csv", index=False)
    summary.to_csv(results_dir / f"selected_cloud_rule_downstream_paired_seed_summary_{date_tag}.csv", index=False)
    return deltas, summary


def write_markdown(results_dir: Path, date_tag: str, summary: pd.DataFrame, weights: pd.DataFrame, paired: pd.DataFrame) -> None:
    visible = [
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
        "These tables are derived from the validation-selected and fixed-weight residual-bootstrap decision runs.",
        "",
        "## Main Downstream Summary",
        "",
        summary[visible].to_markdown(index=False),
        "",
        "## LP-Weight Sensitivity",
        "",
        weights.to_markdown(index=False),
        "",
        "## Paired Seed Summary",
        "",
        paired.to_markdown(index=False),
        "",
    ]
    (results_dir / f"selected_cloud_rule_downstream_summary_{date_tag}.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run(results_dir: Path, date_tag: str) -> None:
    results_dir = _resolve(results_dir)
    summary = build_downstream_summary(results_dir, date_tag)
    weights = build_weight_sensitivity(results_dir, date_tag)
    _, paired = build_paired_seed_tables(results_dir, date_tag)
    write_markdown(results_dir, date_tag, summary, weights, paired)
    print(f"Wrote downstream tables to {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manuscript downstream summary tables from decision result files.")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/selected_cloud_rule_downstream"))
    parser.add_argument("--date-tag", default="20260527")
    args = parser.parse_args()
    run(args.results_dir, args.date_tag)


if __name__ == "__main__":
    main()
