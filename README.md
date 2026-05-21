# DRAEM: A Gate-Controlled Semantic Residual Correction Framework for Selective Multimodal Information Integration in Time-Series Forecasting

This repository contains the code, data, and experimental scripts for the paper:

> **DRAEM: A Gate-Controlled Semantic Residual Correction Framework for Selective Multimodal Information Integration in Time-Series Forecasting**


## Overview

DRAEM (Directional Residual Adjustment with Event Memory) is a conditional semantic residual correction framework for carbon-price direction forecasting. It combines a structured numeric base predictor with a semantic event branch that injects residual corrections through a learned gate, enabling selective semantic intervention under event-conflict regimes.

### Prerequisites

```bash
pip install -r requirements.txt
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



## License

This project is for academic research purposes. Please cite the paper if you use this code or data.
