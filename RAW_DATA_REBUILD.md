# Rebuild from public raw data

This document describes the raw-data path behind the retained manuscript results. It starts from public CAISO, NOAA/NWS, NOAA Storm Events, and HRRR Zarr inputs, then rebuilds the selected neural forecast predictions, validation-selected hybrid decision runs, and manuscript enrichment tables.

The cached manuscript path in `README.md` is the fastest way to regenerate the released figures and tables. Use this raw rebuild path when you need to audit or refresh the model outputs from source data.

## Data sources and reproducibility boundary

- CAISO OASIS day-ahead and real-time LMP data for node `TH_SP15_GEN-APND`.
- CAISO Today's Outlook actual solar, demand, and net-demand history.
- CAISO OASIS day-ahead solar forecast for the SP15 trading hub.
- NOAA/NWS Area Forecast Discussion text for `AFDLOX`, `AFDSGX`, `AFDHNX`, `AFDSTO`, and `AFDMTR`.
- NOAA Storm Events labels for California counties used in the study region.
- HRRR Zarr point features for the California proxy locations in `config/dataset_config_multi_weather_2023_2024.json`.

The released repository includes `data/derived_inputs/nws_llm_weather_features_multi_nws_2023-01-01_2025-01-01.csv`. This is the structured LLM weather-feature cache used by the manuscript. Keeping this cache fixed lets the raw rebuild avoid paid API calls while preserving the manuscript feature inputs. To refresh that cache from raw NWS text, set API credentials in a local `.env` file and run the optional cache command below.

Large raw and processed data directories are intentionally not committed.

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

The full 2023-2024 HRRR extraction is the slowest step and can take hours depending on network speed.

```bash
python scripts/caiso_noaa_dataset.py \
  --config config/dataset_config_multi_weather_2023_2024.json \
  --output data_multi_weather_2023_2024
```

For a quick downloader smoke test without HRRR:

```bash
python scripts/caiso_noaa_dataset.py \
  --config config/dataset_config_multi_weather_2023_2024.json \
  --output data_multi_weather_2023_2024 \
  --skip-hrrr
```

## 2. Place the released LLM feature cache

Copy the released structured LLM feature cache into the processed data directory produced by the downloader:

```bash
cp data/derived_inputs/nws_llm_weather_features_multi_nws_2023-01-01_2025-01-01.csv \
  data_multi_weather_2023_2024/processed/nws_llm_weather_features_multi_nws_2023-01-01_2025-01-01.csv
```

Windows PowerShell equivalent:

```powershell
Copy-Item data\derived_inputs\nws_llm_weather_features_multi_nws_2023-01-01_2025-01-01.csv `
  data_multi_weather_2023_2024\processed\nws_llm_weather_features_multi_nws_2023-01-01_2025-01-01.csv -Force
```

Optional API refresh, only if you intentionally want to regenerate the cache:

```bash
python scripts/rebuild_multi_nws_llm_cache.py \
  --processed-dir data_multi_weather_2023_2024/processed \
  --data-suffix 2023-01-01_2025-01-01 \
  --provider deepseek \
  --model deepseek-v4-pro
```

## 3. Build selected neural forecast predictions

This command builds the master hourly table from processed public data, trains the retained selected neural models, and writes `test_predictions.csv`, `train_residuals.csv`, `data_audit.csv`, and `forecast_metrics.csv`.

```bash
python scripts/run_selected_model_predictions.py \
  --processed-dir data_multi_weather_2023_2024/processed \
  --data-suffix 2023-01-01_2025-01-01 \
  --output-dir outputs/selected_cloud_rule_downstream
```

The retained forecast/decision path is:

- PV reference: MLP with rule-core narrative features.
- PV LLM: MLP with the selected LLM cloud-rule feature.
- RT reference: Transformer with rule-text narrative features.
- RT LLM: Transformer with LLM rule-equivalent narrative features.

## 4. Run validation-selected decision experiments

Rule-core matched neural reference:

```bash
python scripts/run_hybrid_blended_validation_selection.py \
  --results-dir outputs/selected_cloud_rule_downstream \
  --master-path data_multi_weather_2023_2024/processed/master_hourly_caiso_noaa_2023-01-19_2024-12-31.csv \
  --pv-model mlp_rule_core \
  --rt-model transformer_text \
  --anchor-model mlp_rule_core \
  --seeds 71000,71001,71011,71021,71031,71041,71051,71061,71071,71081 \
  --lp-weights 0.25,0.50,0.75 \
  --residual-scale 1.0 \
  --cvar-gamma 0.25 \
  --deviation-penalty 50 \
  --scenario-count 20 \
  --output-stem bidding_hybrid_blended_mlp_rule_core_transformer_text_anchor_mlp_rule_core_validation_selected
```

LLM cloud-rule hybrid:

```bash
python scripts/run_hybrid_blended_validation_selection.py \
  --results-dir outputs/selected_cloud_rule_downstream \
  --master-path data_multi_weather_2023_2024/processed/master_hourly_caiso_noaa_2023-01-19_2024-12-31.csv \
  --pv-model mlp_llm_rule_cloud \
  --rt-model transformer_llm_rule \
  --anchor-model mlp_rule_core \
  --seeds 71000,71001,71011,71021,71031,71041,71051,71061,71071,71081 \
  --lp-weights 0.25,0.50,0.75 \
  --residual-scale 1.0 \
  --cvar-gamma 0.25 \
  --deviation-penalty 50 \
  --scenario-count 20 \
  --output-stem bidding_hybrid_blended_mlp_llm_rule_cloud_transformer_llm_rule_anchor_mlp_rule_core_validation_selected
```

## 5. Run fixed-weight sensitivity

Rule-core matched neural reference:

```bash
python scripts/run_hybrid_blended_robustness.py \
  --results-dir outputs/selected_cloud_rule_downstream \
  --pv-model mlp_rule_core \
  --rt-model transformer_text \
  --anchor-model mlp_rule_core \
  --seeds 71000,71001,71011,71021,71031,71041,71051,71061,71071,71081 \
  --lp-weights 0.25,0.50,0.75,1.00 \
  --residual-scale 1.0 \
  --cvar-gamma 0.25 \
  --deviation-penalty 50 \
  --scenario-count 20 \
  --output-stem bidding_hybrid_blended_mlp_rule_core_transformer_text_anchor_mlp_rule_core_10seed_penalty50
```

LLM cloud-rule hybrid:

```bash
python scripts/run_hybrid_blended_robustness.py \
  --results-dir outputs/selected_cloud_rule_downstream \
  --pv-model mlp_llm_rule_cloud \
  --rt-model transformer_llm_rule \
  --anchor-model mlp_rule_core \
  --seeds 71000,71001,71011,71021,71031,71041,71051,71061,71071,71081 \
  --lp-weights 0.25,0.50,0.75,1.00 \
  --residual-scale 1.0 \
  --cvar-gamma 0.25 \
  --deviation-penalty 50 \
  --scenario-count 20 \
  --output-stem bidding_hybrid_blended_mlp_llm_rule_cloud_transformer_llm_rule_anchor_mlp_rule_core_10seed_penalty50
```

## 6. Build manuscript downstream tables and enrichment figures

```bash
python scripts/build_selected_downstream_tables.py \
  --results-dir outputs/selected_cloud_rule_downstream \
  --date-tag 20260527

python scripts/build_selected_cloud_rule_enrichment.py \
  --results-dir outputs/selected_cloud_rule_downstream \
  --case-date auto \
  --date-tag 20260527
```

The enrichment step writes forecast slice tables, decision sensitivity tables, paired-seed summaries, and the extreme-weather case-study inputs and figures.

## Optional neural feature search

The manuscript includes a retained neural forecast matrix. To refresh a neural validation-selected feature search from the rebuilt master table, run targeted searches such as:

```bash
python scripts/run_validation_text_feature_search.py \
  --master-path data_multi_weather_2023_2024/processed/master_hourly_caiso_noaa_2023-01-19_2024-12-31.csv \
  --output-dir outputs/neural_feature_search_rmse \
  --targets pv,rt_price \
  --feature-groups none,rule,rule_core,llm_rule,llm_rule_cloud \
  --base-variants full \
  --model-families mlp,cnn,gru,lstm,transformer \
  --metric rmse
```

Repeat with `--metric mae` for the MAE selection table.
