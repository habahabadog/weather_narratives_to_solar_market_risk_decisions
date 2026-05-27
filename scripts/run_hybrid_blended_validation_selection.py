from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_baselines import (
    evaluate_validation_selected_hybrid_blend,
    fit_hybrid_blend_split_predictions,
)
from scripts.run_hybrid_blended_robustness import _parse_float_tuple, _parse_int_tuple
from scripts.run_hybrid_scenario_robustness import add_bidding_comparisons


def write_markdown(
    path: Path,
    validation_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    compared_test_summary: pd.DataFrame,
) -> None:
    if validation_summary.empty or test_summary.empty:
        path.write_text("# Validation-Selected Hybrid Blend\n\nNo rows generated.\n", encoding="utf-8")
        return

    selected = validation_summary.sort_values("validation_selection_rank").iloc[0]
    validation_cols = [
        "lp_weight",
        "validation_selection_rank",
        "validation_selection_score",
        "total_revenue_mean",
        "total_revenue_min",
        "cvar_95_loss_mean",
        "cvar_95_loss_max",
    ]
    test_cols = [
        "strategy",
        "selected_lp_weight",
        "selection_objective",
        "seed_count",
        "total_revenue_mean_vs_S7_pct",
        "total_revenue_min_vs_S7_pct",
        "cvar_95_loss_mean_vs_S7_pct",
        "cvar_95_loss_max_vs_S7_pct",
        "total_revenue_mean",
        "total_revenue_min",
        "cvar_95_loss_mean",
        "cvar_95_loss_max",
    ]
    validation_visible = [col for col in validation_cols if col in validation_summary.columns]
    test_visible = [col for col in test_cols if col in compared_test_summary.columns]
    lines = [
        "# Validation-Selected Hybrid Blend",
        "",
        f"Selected `lp_weight={float(selected['lp_weight']):.2f}` using `{selected['selection_objective']}`.",
        "",
        "## Validation Weight Ranking",
        "",
        validation_summary.sort_values("validation_selection_rank")[validation_visible].to_markdown(index=False),
        "",
        "## Fixed-Weight Test Result",
        "",
        compared_test_summary[test_visible].to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select hybrid blend lp_weight on a 2023 validation split, then evaluate the fixed weight on 2024."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--master-path",
        type=Path,
        default=Path("data/processed/master_hourly_caiso_noaa_2023-01-19_2024-12-31.csv"),
    )
    parser.add_argument("--validation-train-end", default="2023-10-01")
    parser.add_argument("--validation-start", default="2023-10-01")
    parser.add_argument("--validation-end", default="2024-01-01")
    parser.add_argument("--pv-model", default="mlp_text")
    parser.add_argument("--rt-model", default="transformer_text")
    parser.add_argument("--anchor-model", default="mlp_text")
    parser.add_argument("--seeds", default="71000,71001,71011,71021,71031,71041,71051,71061,71071,71081")
    parser.add_argument("--lp-weights", default="0.25,0.50,0.75")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--cvar-gamma", type=float, default=0.25)
    parser.add_argument("--deviation-penalty", type=float, default=50.0)
    parser.add_argument("--scenario-count", type=int, default=20)
    parser.add_argument("--selection-objective", default="balanced_revenue_cvar")
    parser.add_argument("--output-stem", default="bidding_hybrid_blended_validation_selected")
    args = parser.parse_args()

    results_dir = args.results_dir
    master_path = args.master_path
    if not master_path.is_absolute():
        master_path = REPO_ROOT / master_path

    master = pd.read_csv(master_path)
    validation_preds, validation_train_residuals, validation_meta = fit_hybrid_blend_split_predictions(
        master=master,
        train_end=args.validation_train_end,
        eval_start=args.validation_start,
        eval_end=args.validation_end,
        pv_model_name=args.pv_model,
        rt_model_name=args.rt_model,
        anchor_model_name=args.anchor_model,
    )

    test_preds = pd.read_csv(results_dir / "test_predictions.csv")
    test_train_residuals = pd.read_csv(results_dir / "train_residuals.csv")
    audit = pd.read_csv(results_dir / "data_audit.csv")
    bidding = pd.read_csv(results_dir / "bidding_metrics.csv")
    rated_capacity = float(audit.iloc[0]["rated_capacity_mw"])

    validation_seed_rows, validation_summary, test_seed_rows, test_summary = evaluate_validation_selected_hybrid_blend(
        validation_preds=validation_preds,
        validation_train_residuals=validation_train_residuals,
        test_preds=test_preds,
        test_train_residuals=test_train_residuals,
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
        selection_objective=args.selection_objective,
    )
    compared_test_summary = add_bidding_comparisons(test_summary, bidding)

    stem = args.output_stem
    pd.DataFrame([validation_meta]).to_csv(results_dir / f"{stem}_validation_meta.csv", index=False)
    validation_preds.to_csv(results_dir / f"{stem}_validation_predictions.csv", index=False)
    validation_train_residuals.to_csv(results_dir / f"{stem}_validation_train_residuals.csv", index=False)
    validation_seed_rows.to_csv(results_dir / f"{stem}_validation_seed.csv", index=False)
    validation_summary.to_csv(results_dir / f"{stem}_validation_summary.csv", index=False)
    test_seed_rows.to_csv(results_dir / f"{stem}_test_seed.csv", index=False)
    test_summary.to_csv(results_dir / f"{stem}_test_summary.csv", index=False)
    compared_test_summary.to_csv(results_dir / f"{stem}_test_summary_compared.csv", index=False)
    write_markdown(results_dir / f"{stem}.md", validation_summary, test_summary, compared_test_summary)


if __name__ == "__main__":
    main()
