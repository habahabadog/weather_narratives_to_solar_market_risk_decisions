from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_baselines import evaluate_hybrid_scenario_seed_robustness


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _bidding_value(bidding: pd.DataFrame, strategy: str, column: str) -> float:
    row = bidding[bidding["strategy"] == strategy]
    if row.empty:
        return np.nan
    return float(row.iloc[0][column])


def add_bidding_comparisons(summary: pd.DataFrame, bidding: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or bidding.empty:
        return summary

    baselines = {
        "S7": "S7_ridge_text_stochastic_LP",
        "S11": "S11_mlp_text_deterministic",
        "S22": "S22_ridge_text_llm_stochastic_LP",
    }
    out = summary.copy()
    for label, strategy in baselines.items():
        total_revenue = _bidding_value(bidding, strategy, "total_revenue")
        cvar_loss = _bidding_value(bidding, strategy, "cvar_95_loss")
        out[f"{label}_strategy"] = strategy
        out[f"{label}_total_revenue"] = total_revenue
        out[f"{label}_cvar_95_loss"] = cvar_loss
        if np.isfinite(total_revenue) and abs(total_revenue) > 1e-12:
            out[f"total_revenue_mean_vs_{label}_pct"] = (
                (out["total_revenue_mean"] - total_revenue) / abs(total_revenue) * 100.0
            )
            out[f"total_revenue_min_vs_{label}_pct"] = (
                (out["total_revenue_min"] - total_revenue) / abs(total_revenue) * 100.0
            )
        else:
            out[f"total_revenue_mean_vs_{label}_pct"] = np.nan
            out[f"total_revenue_min_vs_{label}_pct"] = np.nan
        if np.isfinite(cvar_loss) and abs(cvar_loss) > 1e-12:
            out[f"cvar_95_loss_mean_vs_{label}_pct"] = (
                (cvar_loss - out["cvar_95_loss_mean"]) / abs(cvar_loss) * 100.0
            )
            out[f"cvar_95_loss_max_vs_{label}_pct"] = (
                (cvar_loss - out["cvar_95_loss_max"]) / abs(cvar_loss) * 100.0
            )
        else:
            out[f"cvar_95_loss_mean_vs_{label}_pct"] = np.nan
            out[f"cvar_95_loss_max_vs_{label}_pct"] = np.nan
    return out


def write_markdown(path: Path, ranked_summary: pd.DataFrame) -> None:
    if ranked_summary.empty:
        path.write_text("# Hybrid Scenario Robustness\n\nNo rows generated.\n", encoding="utf-8")
        return

    columns = [
        "strategy",
        "seed_count",
        "total_revenue_mean_vs_S7_pct",
        "total_revenue_min_vs_S7_pct",
        "cvar_95_loss_mean_vs_S7_pct",
        "cvar_95_loss_max_vs_S7_pct",
        "total_revenue_mean_vs_S11_pct",
        "total_revenue_mean_vs_S22_pct",
        "total_revenue_mean",
        "total_revenue_min",
        "cvar_95_loss_mean",
        "cvar_95_loss_max",
    ]
    visible_columns = [column for column in columns if column in ranked_summary.columns]
    lines = [
        "# Hybrid Scenario Robustness",
        "",
        "Positive revenue deltas are better. Positive CVaR loss deltas mean lower loss than the baseline.",
        "",
        ranked_summary[visible_columns].to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid PV/RT scenario robustness from cached results.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--pv-model", default="mlp_text")
    parser.add_argument("--rt-model", default="transformer_text")
    parser.add_argument("--seeds", default="71000,71001,71011,71021,71031")
    parser.add_argument("--residual-scales", default="1.00")
    parser.add_argument("--cvar-gammas", default="0.00,0.25,0.50")
    parser.add_argument("--deviation-penalty", type=float, default=50.0)
    parser.add_argument("--scenario-count", type=int, default=20)
    parser.add_argument("--output-stem", default="bidding_hybrid_mlp_text_transformer_text_robustness")
    args = parser.parse_args()

    results_dir = args.results_dir
    preds = pd.read_csv(results_dir / "test_predictions.csv")
    train_residuals = pd.read_csv(results_dir / "train_residuals.csv")
    audit = pd.read_csv(results_dir / "data_audit.csv")
    bidding = pd.read_csv(results_dir / "bidding_metrics.csv")
    rated_capacity = float(audit.iloc[0]["rated_capacity_mw"])

    seed_frames = []
    summary_frames = []
    for scale in _parse_float_tuple(args.residual_scales):
        for gamma in _parse_float_tuple(args.cvar_gammas):
            seed_rows, summary = evaluate_hybrid_scenario_seed_robustness(
                preds=preds,
                train_residuals=train_residuals,
                rated_capacity=rated_capacity,
                seeds=_parse_int_tuple(args.seeds),
                pv_model_name=args.pv_model,
                rt_model_name=args.rt_model,
                residual_scale=scale,
                cvar_gamma=gamma,
                deviation_penalty=args.deviation_penalty,
                scenario_count=args.scenario_count,
            )
            if not seed_rows.empty:
                seed_frames.append(seed_rows)
            if not summary.empty:
                summary_frames.append(summary)

    seed_df = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()
    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    ranked_summary = add_bidding_comparisons(summary_df, bidding)
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
    seed_df.to_csv(results_dir / f"{stem}.csv", index=False)
    summary_df.to_csv(results_dir / f"{stem}_summary.csv", index=False)
    ranked_summary.to_csv(results_dir / f"{stem}_summary_ranked.csv", index=False)
    write_markdown(results_dir / f"{stem}_summary.md", ranked_summary)


if __name__ == "__main__":
    main()
