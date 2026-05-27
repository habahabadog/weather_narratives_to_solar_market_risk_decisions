from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_baselines import evaluate_hybrid_blended_scenario_seed_robustness
from scripts.run_hybrid_scenario_robustness import add_bidding_comparisons


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def write_markdown(path: Path, ranked_summary: pd.DataFrame) -> None:
    if ranked_summary.empty:
        path.write_text("# Hybrid Blended Scenario Robustness\n\nNo rows generated.\n", encoding="utf-8")
        return

    columns = [
        "strategy",
        "seed_count",
        "lp_weight",
        "total_revenue_mean_vs_S7_pct",
        "total_revenue_min_vs_S7_pct",
        "cvar_95_loss_mean_vs_S7_pct",
        "cvar_95_loss_max_vs_S7_pct",
        "total_revenue_mean_vs_S11_pct",
        "cvar_95_loss_mean_vs_S11_pct",
        "total_revenue_mean_vs_S22_pct",
        "total_revenue_mean",
        "total_revenue_min",
        "cvar_95_loss_mean",
        "cvar_95_loss_max",
    ]
    visible_columns = [column for column in columns if column in ranked_summary.columns]
    lines = [
        "# Hybrid Blended Scenario Robustness",
        "",
        "Positive revenue deltas are better. Positive CVaR loss deltas mean lower loss than the baseline.",
        "",
        ranked_summary[visible_columns].to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blended hybrid PV/RT scenario robustness from cached results.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--pv-model", default="mlp_text")
    parser.add_argument("--rt-model", default="transformer_text")
    parser.add_argument("--anchor-model", default="mlp_text")
    parser.add_argument("--seeds", default="71000,71001,71011,71021,71031")
    parser.add_argument("--lp-weights", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--cvar-gamma", type=float, default=0.25)
    parser.add_argument("--deviation-penalty", type=float, default=50.0)
    parser.add_argument("--scenario-count", type=int, default=20)
    parser.add_argument("--output-stem", default="bidding_hybrid_blended_mlp_text_transformer_text_robustness")
    args = parser.parse_args()

    results_dir = args.results_dir
    preds = pd.read_csv(results_dir / "test_predictions.csv")
    train_residuals = pd.read_csv(results_dir / "train_residuals.csv")
    audit = pd.read_csv(results_dir / "data_audit.csv")
    bidding = pd.read_csv(results_dir / "bidding_metrics.csv")
    rated_capacity = float(audit.iloc[0]["rated_capacity_mw"])

    seed_rows, summary = evaluate_hybrid_blended_scenario_seed_robustness(
        preds=preds,
        train_residuals=train_residuals,
        rated_capacity=rated_capacity,
        seeds=_parse_int_tuple(args.seeds),
        pv_model_name=args.pv_model,
        rt_model_name=args.rt_model,
        anchor_model_name=args.anchor_model,
        lp_weights=_parse_float_tuple(args.lp_weights),
        residual_scale=args.residual_scale,
        cvar_gamma=args.cvar_gamma,
        deviation_penalty=args.deviation_penalty,
        scenario_count=args.scenario_count,
    )
    ranked_summary = add_bidding_comparisons(summary, bidding)
    if not ranked_summary.empty:
        ranked_summary = ranked_summary.sort_values(
            [
                "total_revenue_mean_vs_S7_pct",
                "total_revenue_min_vs_S7_pct",
                "cvar_95_loss_mean_vs_S7_pct",
            ],
            ascending=[False, False, False],
        )

    stem = args.output_stem
    seed_rows.to_csv(results_dir / f"{stem}.csv", index=False)
    summary.to_csv(results_dir / f"{stem}_summary.csv", index=False)
    ranked_summary.to_csv(results_dir / f"{stem}_summary_ranked.csv", index=False)
    write_markdown(results_dir / f"{stem}_summary.md", ranked_summary)


if __name__ == "__main__":
    main()
