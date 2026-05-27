# Weather narratives to solar-market risk decisions

This repository contains the code and released data artifacts for:

**From public weather narratives to solar-market risk decisions using constrained language-model features**

The repository has two reproducibility paths:

1. A fast cached path that regenerates the manuscript display tables and figures from the released CSV artifacts.
2. A public raw-data rebuild path that downloads CAISO/NOAA/NWS/HRRR inputs, rebuilds the selected neural forecast predictions, and reruns the validation-selected decision experiments.

The repository is intentionally scoped to the retained manuscript workflow. It does not include the full exploratory experiment tree or temporary intermediate outputs.

## Contents

### Cached manuscript artifacts

- `data/forecast_matrix.csv`: retained neural forecast matrix for PV and real-time price.
- `data/forecast_slice_metrics.csv`: forecast-slice metrics used in the slice figure and supplementary table.
- `data/decision_summary.csv`: validation-selected settlement-proxy summary rows.
- `data/lp_weight_sensitivity.csv`: LP-anchor sensitivity table.
- `data/paired_seed_deltas.csv`: paired residual-bootstrap seed-level deltas.
- `data/paired_seed_summary.csv`: paired residual-bootstrap summary intervals.
- `data/case_2024_03_13.csv`: hourly inputs for the 13 March 2024 case study.
- `data/case_screening.csv`: extreme-weather case-screening cache.
- `scripts/reproduce_manuscript_outputs.py`: single script that rebuilds the released tables and figures.

### Raw-data rebuild code

- `config/dataset_config_multi_weather_2023_2024.json`: public data configuration for the 2023-2024 CAISO/NOAA/NWS/HRRR rebuild.
- `data/derived_inputs/nws_llm_weather_features_multi_nws_2023-01-01_2025-01-01.csv`: released structured LLM weather-feature cache used by the manuscript.
- `scripts/caiso_noaa_dataset.py`: downloader and processor for public CAISO/NOAA/NWS/HRRR data.
- `scripts/run_selected_model_predictions.py`: rebuilds the retained selected neural forecast predictions from processed public data.
- `scripts/run_hybrid_blended_validation_selection.py`: selects the hybrid decision weight on validation data and evaluates the locked test set.
- `scripts/run_hybrid_blended_robustness.py`: rebuilds fixed-weight decision sensitivity runs.
- `scripts/build_selected_downstream_tables.py`: converts decision result files into manuscript downstream tables.
- `scripts/build_selected_cloud_rule_enrichment.py`: rebuilds forecast-slice, decision-sensitivity, paired-seed, and case-study enrichment outputs.
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
  --config config/dataset_config_multi_weather_2023_2024.json \
  --output data_multi_weather_2023_2024

cp data/derived_inputs/nws_llm_weather_features_multi_nws_2023-01-01_2025-01-01.csv \
  data_multi_weather_2023_2024/processed/nws_llm_weather_features_multi_nws_2023-01-01_2025-01-01.csv

python scripts/run_selected_model_predictions.py \
  --processed-dir data_multi_weather_2023_2024/processed \
  --data-suffix 2023-01-01_2025-01-01 \
  --output-dir outputs/selected_cloud_rule_downstream
```

Then run the validation-selected and fixed-weight decision commands in `RAW_DATA_REBUILD.md`.

No LLM API key is required when using the released structured LLM feature cache. API credentials are only needed if you intentionally refresh that cache from raw NWS text.
