import numpy as np
import pandas as pd
from typing import Tuple, List

def _read_csv(filename: str) -> Tuple[int, List[float], int, pd.DataFrame, int, int]:
    """Read CSV file and prepare data"""
    # Read CSV
    df = pd.read_csv(filename)
    
    # Extract energy values
    energy_values = df['energy'].values
    original_N = len(energy_values)
    
    # TRUNCATE to the largest power of 2 ≤ original_N (instead of padding up)
    N = 2 ** int(np.floor(np.log2(original_N)))

    # Truncate data 
    if len(energy_values) > N:
        truncated_data = energy_values[:N]
        deferred_count = original_N - N
        pad_count = 0
    else:
        # Edge case: if original_N is already a power of 2
        truncated_data = energy_values
        deferred_count = 0
        pad_count = 0

    print(f"Read {original_N} values from CSV")
    if deferred_count > 0:
        print(f"Truncated to {N} (largest power of 2 <= {original_N})")
        print(f"Deferred {deferred_count} points ({deferred_count/original_N*100:.1f}%)")
    else:
        print(f"Using {N} points (already a power of 2)")
    print(f"Data range: [{truncated_data.min():.2f}, {truncated_data.max():.2f}]")

    return N, truncated_data.tolist(), original_N, df, deferred_count, pad_count


def compute_signal_stats(signal: np.ndarray, bins: int = 50) -> dict:
    # Shannon entropy
    hist, _ = np.histogram(
        signal,
        bins=min(bins, max(5, len(signal)//10)),
        density=True
    )
    hist = hist[hist > 0]
    probs = hist / (hist.sum() + 1e-10)
    H = -np.sum(probs * np.log2(probs + 1e-10))

    mean = float(signal.mean())
    std  = float(signal.std())
    data_min = float(signal.min())
    data_max = float(signal.max())
    data_range = data_max - data_min
    coeff_var = std / (mean + 1e-10) if abs(mean) > 1e-10 else 0.0
    med = float(np.median(signal))
    zero_frac = float(np.mean(np.isclose(signal, 0.0, atol=1e-6)))
    
    # Gini coefficient (inequality measure, high = sparse/peaked)
    sorted_signal = np.sort(np.abs(signal))
    n = len(sorted_signal)
    cumsum = np.cumsum(sorted_signal)
    gini = (
        (2 * np.sum((np.arange(1, n+1)) * sorted_signal)) / (n * cumsum[-1])
        - (n + 1) / n
        if cumsum[-1] > 0 else 0
    )
    
    # Sparsity index (ratio of L1 to L2 norm, high = sparse)
    l1_norm = np.sum(np.abs(signal))
    l2_norm = np.sqrt(np.sum(signal**2))
    sparsity_index = l1_norm / (np.sqrt(n) * l2_norm) if l2_norm > 0 else 0

    # Autocorrelation at lag-1 (temporal correlation)
    if len(signal) > 1:
        # Check if signal has zero or near-zero variance (constant/near-constant values)
        if std < 1e-10:
            acf_lag1 = np.nan  # Undefined for constant signals
        else:
            try:
                # Use warnings filter to suppress numpy's internal warnings
                with np.errstate(invalid='ignore', divide='ignore'):
                    corr_matrix = np.corrcoef(signal[:-1], signal[1:])
                    acf_lag1 = float(corr_matrix[0, 1])
                    # Check if result is NaN (happens with constant data)
                    if not np.isfinite(acf_lag1):
                        acf_lag1 = np.nan
            except:
                acf_lag1 = np.nan
    else:
        acf_lag1 = 0.0

    return {
        "entropy": H,
        "mean": mean,
        "std": std,
        "data_min": data_min,
        "data_max": data_max,
        "data_range": data_range,
        "coeff_var": coeff_var,
        "median": med,
        "zero_fraction": zero_frac,
        "gini_coefficient": float(gini),
        "sparsity_index": float(sparsity_index),
        "autocorr_lag1": acf_lag1
    }


def scan_windows_and_export_csv(filename: str,
                                window_sizes=(4048, 1024, 512, 256, 128),
                                output_csv: str = "window_stats_summary.csv") -> pd.DataFrame:
    # Read and truncate data
    N, truncated_list, *_ = _read_csv(filename)
    signal = np.array(truncated_list, dtype=float)

    # We only care about these four metrics
    target_metrics = ["entropy", "std", "gini_coefficient"]

    # Global min/max over ALL windows and ALL window sizes
    global_stats = {
        m: {
            "min_value": None,
            "min_window": None,        # (start_idx_1based, end_idx_1based)
            "min_window_size": None,
            "max_value": None,
            "max_window": None,
            "max_window_size": None
        }
        for m in target_metrics
    }

    for w in window_sizes:
        if w > N:
            continue  # just in case

        print(f"\nScanning window size {w} over {N} samples...")
        num_windows = N - w + 1

        for start in range(num_windows):
            end = start + w
            window = signal[start:end]

            stats = compute_signal_stats(window)

            # 1-based indices for address (as you specified: 1-4096, 2-4097, ...)
            addr = (start + 1, end)

            for m in target_metrics:
                val = stats[m if m != "gini_coefficient" else "gini_coefficient"]

                # Initialize if needed
                if global_stats[m]["min_value"] is None:
                    global_stats[m]["min_value"] = val
                    global_stats[m]["min_window"] = addr
                    global_stats[m]["min_window_size"] = w

                    global_stats[m]["max_value"] = val
                    global_stats[m]["max_window"] = addr
                    global_stats[m]["max_window_size"] = w
                else:
                    # Update min
                    if val < global_stats[m]["min_value"]:
                        global_stats[m]["min_value"] = val
                        global_stats[m]["min_window"] = addr
                        global_stats[m]["min_window_size"] = w
                    # Update max
                    if val > global_stats[m]["max_value"]:
                        global_stats[m]["max_value"] = val
                        global_stats[m]["max_window"] = addr
                        global_stats[m]["max_window_size"] = w

    # Build final CSV: only the min of mins & max of maxs for each metric
    rows = []
    for m in target_metrics:
        s = global_stats[m]
        min_start, min_end = s["min_window"]
        max_start, max_end = s["max_window"]

        rows.append({
            "metric": m,
            "global_min": s["min_value"],
            "min_window_address": f"{min_start}-{min_end}",
            "min_window_size": s["min_window_size"],
            "global_max": s["max_value"],
            "max_window_address": f"{max_start}-{max_end}",
            "max_window_size": s["max_window_size"],
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)
    print("\nFinal global min/max summary (also saved to CSV):")
    print(result_df)

    return result_df


scan_windows_and_export_csv("test_data.csv")