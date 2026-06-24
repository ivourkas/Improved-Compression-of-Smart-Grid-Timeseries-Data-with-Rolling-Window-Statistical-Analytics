# Wavelet Synopsis — BPL Energy Data

Haar wavelet synopsis compression of building energy time-series data, using the Garofalakis-Kumar algorithm. Two error modes are supported:

- **L2**: minimizes total squared reconstruction error (RMSE)
- **Linf**: minimizes maximum per-point reconstruction error

## Data

| File | Description |
|---|---|
| `data/test_data_v2.csv` | Working dataset — hourly energy readings, June 2025–June 2026 (~8,784 points) |
| `data/test_data_v1.csv` | Reference dataset — original test signal used during algorithm development |

Both files share the same column schema: `unit`, `interval_end`, `demand`, `energy`.

## Environment

- Python `3.9.6`, macOS arm64
- Pinned dependencies in `requirements.txt`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> First run is slower — `numba` JIT-compiles the inner loops on startup.

## Workflow

Scripts must be run from the repo root directory.

### Step 1 — Full-signal synopsis

Run L2 and Linf compression across a range of coefficient budgets on the full signal:

```bash
python garo_l2_for_csv.py
python garo_inf_for_csv.py
```

Outputs written to `results_method1_l2/` and `results_method1_linf/` respectively, including:
- `garo_fullsignal_metrics.csv` — per-budget error metrics
- `reconstructed_method1.csv` — original vs reconstructed signal

### Step 2 — Find interesting segments

Scan the full signal with a sliding window to identify segments that are hardest/easiest to compress (by entropy, std, Gini coefficient):

```bash
python metrics_analysis.py
```

Output: `window_stats_summary.csv` (one row per metric, with the window address of the global min and max).

### Step 3 — Segment-level synopsis

Run Linf synopsis on each interesting segment identified in Step 2:

```bash
python problematic_segments.py
```

Output written to `results_segments_linf/`, one subfolder per segment, plus `segments_overall_summary.csv`.

### Optional — Interactive demo

Explore compression interactively or with preset flags:

```bash
python compression_demo.py                                      # interactive
python compression_demo.py --method L2   --target 0.3 --no-show
python compression_demo.py --method Linf --target 15  --no-show
python compression_demo.py --csv data/test_data_v1.csv --method L2 --target 0.3 --no-show
```

Key flags: `--method`, `--target`, `--csv`, `--signal-col`, `--time-col`, `--outlier`, `--no-show`.

## Repository Layout

```
.
├── data/
│   ├── test_data_v2.csv       # working dataset
│   └── test_data_v1.csv       # reference dataset
├── garo_l2_for_csv.py         # L2 synopsis algorithm
├── garo_inf_for_csv.py        # Linf synopsis algorithm
├── compression_demo.py        # interactive demo
├── metrics_analysis.py        # Step 2: find interesting segments
├── problematic_segments.py    # Step 3: compress segments
└── requirements.txt
```

Generated output directories (`results_*/`, `window_stats_summary.csv`) are gitignored and recreated on each run.
