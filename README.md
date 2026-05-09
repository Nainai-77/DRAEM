# DRAEM: Conditional Semantic Residual Correction for Carbon-Price Direction Forecasting

This repository contains the code, data, and experimental scripts for the paper:

> **DRAEM: Conditional Semantic Residual Correction for Carbon-Price Direction Forecasting**
> (Anonymous Authors, 2026)

## Overview

DRAEM (Directional Residual Adjustment with Event Memory) is a conditional semantic residual correction framework for carbon-price direction forecasting. It combines a structured numeric base predictor with a semantic event branch that injects residual corrections through a learned gate, enabling selective semantic intervention under event-conflict regimes.

## Repository Structure

```
DRAEM_github/
├── models/                     # Core model implementations
│   ├── draem_v3_model.py       # DRAEM v3 model (main model used in paper)
│   └── draem_final_model.py    # DRAEM final model variant
├── config/                     # Configuration and utilities
│   ├── config.py               # Hyperparameters and paths
│   ├── dataset.py              # Data loading and preprocessing
│   └── utils.py                # Seed setting and helpers
├── scripts/
│   ├── full_main_exp.py        # Stage-I main experiment (H1/H5/H7)
│   ├── v3_5seed_formal.py      # 5-seed formal experiment
│   ├── v3_calibrated.py        # Calibration experiment
│   ├── ablation_exp.py         # Ablation study
│   ├── qef_draem_h7_ablation.py # H7 window ablation
│   ├── numerical_only_baselines.py # Numeric baselines (XGBoost, LSTM, etc.)
│   ├── stage1_lstm.py          # Stage-1 LSTM baseline
│   ├── cross_market_baselines.py # Cross-market experiments
│   ├── memory_beta_sensitivity.py # Memory decay sensitivity
│   ├── qwen_event_factor_extract.py # LEF extraction (requires Qwen2.5-7B)
│   ├── qwen_event_memory_build.py   # Event memory construction
│   └── plotting/
│       ├── generate_q1_figures.py   # Generate all paper figures
│       ├── generate_q1_figures_v2.py # V2 figure generation
│       ├── generate_all_figures.py   # Comprehensive figure generation
│       ├── redraw_fig08_ablation_trajectory.py # Figure 8 (ablation)
│       └── plot_fig*.py              # Individual figure scripts
├── data/                       # Datasets
│   ├── dataset_with_qwen_event_memory.csv  # Main dataset with QEF memory
│   └── dataset_with_qwen_cumulative.csv    # Cumulative QEF dataset
├── results/                    # Pre-computed results
│   ├── final_baseline_result_stage_1.csv   # Stage-I baseline results
│   ├── final_window_result_stage_2.csv     # Stage-II window results
│   ├── final_ablation_result.csv           # Ablation results
│   ├── qwen_event_daily.csv               # Daily QEF factors
│   └── residual_daily_predictions.csv      # Daily residual predictions
├── features/                   # Symlink to data/ (for backward compatibility)
├── figures/                    # Output directory for generated figures
└── requirements.txt
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Reproducing Paper Results

All pre-computed results are in `results/`. To regenerate figures:

```bash
cd scripts/plotting
python generate_q1_figures.py
```

### Running Experiments from Scratch

1. **Stage-I main comparison:**
   ```bash
   python scripts/full_main_exp.py
   ```

2. **5-seed formal experiment:**
   ```bash
   python scripts/v3_5seed_formal.py
   ```

3. **Ablation study:**
   ```bash
   python scripts/ablation_exp.py
   python scripts/qef_draem_h7_ablation.py
   ```

### LEF Extraction (requires Qwen2.5-7B-Instruct)

The LEF (Language Event Factor) extraction requires a local Qwen2.5-7B-Instruct model. Update the model path in `scripts/qwen_event_factor_extract.py` before running:

```bash
python scripts/qwen_event_factor_extract.py
python scripts/qwen_event_memory_build.py
```

## Data Description

- **dataset_with_qwen_event_memory.csv**: Main dataset with 146 structured features + 5 LEF channels (policy, market, finance, trade, compliance) + rolling event memory features. Time range: 2013-12-19 to 2025-04-03 (2448 trading days).
- **dataset_with_qwen_cumulative.csv**: Dataset with cumulative QEF features.
- **qwen_event_daily.csv**: Daily QEF extraction results per semantic channel.

## Key Results

| Metric | DRAEM (H7) | Best Baseline (H7) | Δ |
|--------|-----------|-------------------|---|
| MCC | 0.149 | 0.115 (XGBoost) | +0.034 |
| DA | 61.5% | 58.6% (XGBoost) | +2.9pp |

Under the H7 extreme-disagreement event-conflict window (`signal90_disagree90`):
- MCC = 0.358, gain over best baseline = +0.123

## Citation

```bibtex
@article{draem2026,
  title={DRAEM: Conditional Semantic Residual Correction for Carbon-Price Direction Forecasting},
  author={Anonymous},
  year={2026}
}
```

## License

This project is for academic research purposes. Please cite the paper if you use this code or data.
