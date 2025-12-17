# Wavelet Compression of Smart Grid Time Series

Rolling window statistical analytics for improved wavelet-based compression of smart grid energy data.

## Overview

This repository implements the Garofalakis-Kumar wavelet synopsis algorithm with two error minimization strategies (L₂ and L∞), enhanced by rolling window analysis to identify and handle problematic signal segments. The framework automatically detects high-entropy, high-variance, and high-inequality regions that resist compression.

## Method

1. **Rolling Window Analysis** — Scan signal with multiple window sizes (128–4096) to find segments with extreme statistical properties
2. **Segment Identification** — Extract windows with global min/max entropy, std, and Gini coefficient
3. **Adaptive Compression** — Apply wavelet synopsis with coefficient budgets from 0.5% to 30%
4. **Error Targeting** — Stop when L∞ error ≤ 15% of global max amplitude

## Repository Structure

```
├── results_method1_l2/           # L2-optimized compression outputs
├── results_method1_linf/         # L∞-optimized compression outputs
│
├── garo_l2_for_csv_quick.py      # Garofalakis-Kumar L2 algorithm (Numba JIT)
├── garo_inf_for_csv.py           # Garofalakis-Kumar L∞ algorithm
├── metrics_analysis.py           # Rolling window statistics scanner
├── problematic_segments.py       # Segment extraction and compression
│
├── test_data.csv                 # Input energy time series
└── window_stats_summary.csv      # Detected extreme windows
```

## Algorithms

| Algorithm | Error Metric | Optimization Target |
|-----------|--------------|---------------------|
| `garo_l2_for_csv_quick.py` | RMSE (L₂) | Minimize overall squared error |
| `garo_inf_for_csv.py` | Max error (L∞) | Bound worst-case reconstruction error |

Both use Haar wavelet decomposition with dynamic programming to select optimal coefficient subsets.

## Signal Statistics

The framework computes per-window:
- **Entropy** — Information content (high = hard to compress)
- **Std / Coefficient of Variation** — Variability measures
- **Gini Coefficient** — Inequality/sparsity indicator
- **Autocorrelation (lag-1)** — Temporal structure

## Quick Start

```bash
# 1. Scan for problematic windows
python metrics_analysis.py

# 2. Compress identified segments with L∞ targeting
python problematic_segments.py

# 3. Or run full L2 compression
python garo_l2_for_csv_quick.py
```

## Output

Each compression run produces:
- `*_metrics.csv` — Per-iteration error metrics (L∞, RMSE, SNR, R², etc.)
- `*_reconstruction.csv` — Original vs reconstructed values with timestamps

## Requirements

```
numpy
pandas
pywt
scipy
numba
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coeff_pct_start` | 0.5% | Initial coefficient budget |
| `coeff_pct_max` | 30% | Maximum coefficient budget |
| `target_linf` | 15% | L∞ error threshold (% of global max) |
| `window_sizes` | 128–4096 | Rolling window sizes for analysis |

## License

MIT License
