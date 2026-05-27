# Minimal reproducibility package

This repository contains the minimal cached data and code needed to reproduce the manuscript display tables and figures for:

**From public weather narratives to solar-market risk decisions using constrained language-model features**

The package is intentionally narrow. It does not include the full exploratory experiment tree, model-search scripts, or temporary diagnostic outputs. The included CSV files are the manuscript result caches used to regenerate the retained forecast tables, settlement-proxy decision tables, forecast-slice figure, value--CVaR figure, paired residual-bootstrap figure, and the 13 March 2024 extreme-weather case-study figure.

## Contents

- `data/forecast_matrix.csv`: retained neural forecast matrix for PV and real-time price.
- `data/forecast_slice_metrics.csv`: forecast-slice metrics used in the slice figure and supplementary table.
- `data/decision_summary.csv`: validation-selected settlement-proxy summary rows.
- `data/lp_weight_sensitivity.csv`: LP-anchor sensitivity table.
- `data/paired_seed_deltas.csv`: paired residual-bootstrap seed-level deltas.
- `data/paired_seed_summary.csv`: paired residual-bootstrap summary intervals.
- `data/case_2024_03_13.csv`: hourly inputs for the 13 March 2024 case study.
- `data/case_screening.csv`: extreme-weather case-screening cache.
- `scripts/reproduce_manuscript_outputs.py`: single script that rebuilds the released tables and figures.

## Reproduce

Create an environment with Python 3.10 or newer, install the small plotting stack, and run:

```bash
pip install -r requirements.txt
python scripts/reproduce_manuscript_outputs.py
```

The script writes:

- `outputs/tables/*.csv` and `outputs/tables/*.tex`
- `outputs/figures/*.pdf` and `outputs/figures/*.png`

## Scope

No LLM API key is required for this package because the structured weather-feature and experiment-result caches are already included. Rebuilding the full raw-data pipeline, refitting every exploratory model, or refreshing the LLM feature cache requires the larger working repository and external data/API credentials; those steps are outside this minimal manuscript reproducibility release.
