# Rebuild from public raw data

This document describes the raw-data path behind the reported manuscript results. It starts from public CAISO, NOAA/NWS, NOAA Storm Events, and HRRR Zarr inputs, then rebuilds the neural forecast predictions and the common-anchor downstream decision runs.

The cached manuscript path in `README.md` is the fastest way to regenerate the released figures and tables. Use this raw rebuild path when you need to audit or refresh the model outputs from source data.

## Data sources and reproducibility boundary

- CAISO OASIS day-ahead and real-time LMP data for node `TH_SP15_GEN-APND`.
- CAISO Today's Outlook actual solar, demand, and net-demand history.
- CAISO OASIS day-ahead solar forecast for the SP15 trading hub.
- NOAA/NWS Area Forecast Discussion text for `AFDLOX`, `AFDSGX`, `AFDHNX`, `AFDSTO`, and `AFDMTR`.
- NOAA Storm Events labels for California counties used in the study region.
- HRRR Zarr point features for the California proxy locations in `config/dataset_config_multi_weather_2022_2025.json`.

The released repository includes `data/derived_inputs/nws_llm_weather_features_multi_nws_2022-01-01_2026-01-01.csv`. This is the structured LLM weather-feature cache used by the manuscript. Keeping this cache fixed lets the raw rebuild avoid paid API calls while preserving the manuscript feature inputs. To refresh that cache from raw NWS text, set API credentials in a local `.env` file and run the optional cache command below.

Large raw and processed data directories are not committed.

## Environment

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

On Windows PowerShell, copy `.env.example` to `.env` if you want a local environment file. The default cache mode is:

```powershell
$env:NWS_LLM_FEATURE_MODE = "cache"
```

## 1. Download public raw data

The full 2022-2025 HRRR extraction is the slowest step and can take hours depending on network speed.

```bash
python scripts/caiso_noaa_dataset.py \
  --config config/dataset_config_multi_weather_2022_2025.json \
  --output data_multi_weather_2022_2025
```

For a quick downloader smoke test without HRRR:

```bash
python scripts/caiso_noaa_dataset.py \
  --config config/dataset_config_multi_weather_2022_2025.json \
  --output data_multi_weather_2022_2025 \
  --skip-hrrr
```

## 2. Place the released LLM feature cache

Copy the released structured LLM feature cache into the processed data directory produced by the downloader:

```bash
cp data/derived_inputs/nws_llm_weather_features_multi_nws_2022-01-01_2026-01-01.csv \
  data_multi_weather_2022_2025/processed/nws_llm_weather_features_multi_nws_2022-01-01_2026-01-01.csv
```

Windows PowerShell equivalent:

```powershell
Copy-Item data\derived_inputs\nws_llm_weather_features_multi_nws_2022-01-01_2026-01-01.csv `
  data_multi_weather_2022_2025\processed\nws_llm_weather_features_multi_nws_2022-01-01_2026-01-01.csv -Force
```

Optional API refresh:

```bash
python scripts/rebuild_multi_nws_llm_cache.py \
  --processed-dir data_multi_weather_2022_2025/processed \
  --data-suffix 2022-01-01_2026-01-01 \
  --provider deepseek \
  --model deepseek-v4-pro
```

## 3. Build neural forecast predictions

This command builds the master hourly table from processed public data, trains the reported neural forecast models, and writes `test_predictions.csv`, `train_residuals.csv`, `data_audit.csv`, and `forecast_metrics.csv`.

```bash
python scripts/run_selected_cloud_rule_pipeline.py \
  --processed-dir data_multi_weather_2022_2025/processed \
  --data-suffix 2022-01-01_2026-01-01 \
  --output results_selected_cloud_rule_2022_2025_test_2024_2025
```

The forecast path is:

- PV reference: MLP with rule-core narrative features.
- PV LLM: MLP with the LLM cloud-rule feature.
- PV no-text: MLP without narrative features.
- RT reference: Transformer with rule-text narrative features.
- RT LLM: Transformer with LLM rule-equivalent narrative features.
- RT no-text: Transformer without narrative features.

The downstream decision path uses the deterministic MLP no-text PV forecast as the common anchor. The two LP-anchor paths are:

- No-text path: PV MLP no-text + RT Transformer no-text, anchored to MLP no-text.
- LLM path: PV MLP LLM cloud-rule + RT Transformer LLM-rule, anchored to MLP no-text.

## 4. Run downstream decision experiments

```bash
python scripts/run_selected_cloud_rule_downstream.py \
  --results-dir results_selected_cloud_rule_2022_2025_test_2024_2025 \
  --master-path data_multi_weather_2022_2025/processed/master_hourly_caiso_noaa_2022-01-01_2026-01-01.csv \
  --output-dir results_selected_cloud_rule_2022_2025_downstream \
  --seeds 71000,71001,71011,71021,71031,71041,71051,71061,71071,71081 \
  --lp-weights 0.00,0.10,0.25,0.50,0.75,1.00 \
  --selected-weight 0.25 \
  --residual-scale 1.0 \
  --cvar-gamma 0.25 \
  --deviation-penalty 50 \
  --scenario-count 20
```

The downstream script writes the no-text and LLM common-anchor weight paths, the manuscript decision table, decision-frontier points, and paired residual-bootstrap seed summaries for the selected weight.

## 5. Regenerate cached manuscript outputs

```bash
python scripts/reproduce_manuscript_outputs.py
```

This final command rebuilds the released display tables and the three cached manuscript figures from the CSV artifacts in `data/`.
