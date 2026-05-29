# Weather narratives to solar-market risk decisions

This repository contains the code and released data artifacts for:

**From public weather narratives to solar-market risk decisions using constrained language-model features**

The release has two reproducibility paths:

1. A fast cached path that regenerates the manuscript display tables and figures from the released CSV artifacts.
2. A public raw-data rebuild path that downloads CAISO/NOAA/NWS/HRRR inputs, rebuilds the reported neural forecast predictions, and reruns the decision experiments.

The repository is scoped to the reported manuscript workflow and excludes temporary intermediate outputs.

## Contents

### Cached manuscript artifacts

- `data/forecast_main.csv`: five-seed forecast summary for PV and real-time price.
- `data/forecast_seed_summary.csv`: paired five-seed forecast comparisons.
- `data/forecast_slice_metrics.csv`: forecast-slice metrics used in the supplementary forecast table.
- `data/decision_table.csv`: settlement-proxy rows for the common no-text-anchor decision comparison.
- `data/decision_frontier_points.csv`: no-text and LLM cloud-rule LP-anchor weight paths used in the decision frontier figure.
- `data/validation_weight_selection.csv`: validation split weight preselection table for the selected LP-anchor weight.
- `data/paired_seed_deltas.csv`: paired residual-bootstrap seed-level deltas for the selected common-anchor LLM comparison.
- `data/paired_seed_summary.csv`: paired residual-bootstrap summary intervals for the selected common-anchor LLM comparison.
- `data/event_day_examples.csv`: weather-event case-study example table.
- `data/case_2025_03_07.csv`: hourly inputs for the 7 March 2025 case study.
- `assets/fig_pipeline_method_overview.png`: manuscript method-overview figure asset.
- `scripts/reproduce_manuscript_outputs.py`: single script that rebuilds the released tables, forecast-signal figure, decision-frontier figure, and case-study figure.

### Raw-data rebuild code

- `config/dataset_config_multi_weather_2022_2025.json`: public data configuration for the 2022-2025 CAISO/NOAA/NWS/HRRR rebuild.
- `data/derived_inputs/nws_llm_weather_features_multi_nws_2022-01-01_2026-01-01.csv`: released structured LLM weather-feature cache used by the manuscript.
- `scripts/caiso_noaa_dataset.py`: downloader and processor for public CAISO/NOAA/NWS/HRRR data.
- `scripts/rebuild_multi_nws_llm_cache.py`: optional structured-weather-cache refresh from raw NWS text.
- `scripts/run_selected_cloud_rule_pipeline.py`: rebuilds the reported neural forecast predictions from processed public data.
- `scripts/run_selected_cloud_rule_downstream.py`: reruns the reported downstream decision evaluation.
- `RAW_DATA_REBUILD.md`: step-by-step raw-data rebuild instructions.

## Fast Manuscript Reproduction

Create an environment with Python 3.10 or newer, install dependencies, and run:

```bash
pip install -r requirements.txt
python scripts/reproduce_manuscript_outputs.py
```

The script writes:

- `outputs/tables/*.csv` and `outputs/tables/*.tex`
- `outputs/figures/*.pdf` and `outputs/figures/*.png`

## Raw-Data Rebuild

See `RAW_DATA_REBUILD.md` for the full command sequence. The short version is:

```bash
python scripts/caiso_noaa_dataset.py \
  --config config/dataset_config_multi_weather_2022_2025.json \
  --output data_multi_weather_2022_2025

cp data/derived_inputs/nws_llm_weather_features_multi_nws_2022-01-01_2026-01-01.csv \
  data_multi_weather_2022_2025/processed/nws_llm_weather_features_multi_nws_2022-01-01_2026-01-01.csv

python scripts/run_selected_cloud_rule_pipeline.py \
  --processed-dir data_multi_weather_2022_2025/processed \
  --data-suffix 2022-01-01_2026-01-01 \
  --output results_selected_cloud_rule_2022_2025_test_2024_2025

python scripts/run_selected_cloud_rule_downstream.py \
  --results-dir results_selected_cloud_rule_2022_2025_test_2024_2025 \
  --master-path data_multi_weather_2022_2025/processed/master_hourly_caiso_noaa_2022-01-01_2026-01-01.csv \
  --seeds 71000,71001,71011,71021,71031,71041,71051,71061,71071,71081
```

No LLM API key is required when using the released structured LLM feature cache. API credentials are only needed if you refresh that cache from raw NWS text.
