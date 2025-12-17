"""


Use the extreme-metric segments (one global min + four max windows)
and run the Garofalakis-Kumar L_infinity wavelet synopsis algorithm on each
segment separately.

For each segment:
  - Target: L_inf(error) <= 15% of the global max |x_i| over the full-year signal
  - Start at 5% of coefficients
  - Increase in 1% steps
  - Stop at 30% of N or when the L_inf target is met

Outputs:
  results_segments_linf/
    ├─ <segment_label>/
    │    ├─ <segment_label>_metrics.csv
    │    ├─ <segment_label>_reconstruction.csv
    └─ segments_overall_summary.csv
"""

import os
import shutil
import numpy as np
import pandas as pd

from garo_inf_for_csv import GaroOptimizedLinf, compute_signal_stats


WINDOW_STATS_FILE = "window_stats_summary.csv"
FULL_DATA_FILE = "test_data.csv"
OUTPUT_ROOT = "results_segments_linf"


def parse_address(address: str):
    """
    Parse 'start-end' into (start, end), both 1-based inclusive.
    Example: '5187-5442' -> (5187, 5442)
    """
    start_str, end_str = address.split("-")
    return int(start_str), int(end_str)


def build_segments(summary_df: pd.DataFrame):
    """
    Build the list of segments we will compress:
      - Min segments grouped by window address (metrics sharing same min window)
      - 1 max segment per metric

    Returns a list of dicts with:
      {
        "label": str,
        "start": int,   # 1-based
        "end": int,     # 1-based
        "kind": "min" or "max",
        "metrics": [metric_names]
      }
    """
    segments = []

    # --- Group metrics by their min_window_address ---
    min_groups = summary_df.groupby("min_window_address")

    for min_addr, group in min_groups:
        min_start, min_end = parse_address(min_addr)
        metrics_list = list(group["metric"].values)

        # Create label based on how many metrics share this min
        if len(metrics_list) > 1:
            label = f"shared_min_{min_start}_{min_end}"
        else:
            label = f"min_{metrics_list[0]}_{min_start}_{min_end}"

        segments.append({
            "label": label,
            "start": min_start,
            "end": min_end,
            "kind": "min",
            "metrics": metrics_list
        })

    # --- Max per metric ---
    for _, row in summary_df.iterrows():
        metric = row["metric"]
        max_addr = row["max_window_address"]
        max_start, max_end = parse_address(max_addr)

        segments.append({
            "label": f"max_{metric}_{max_start}_{max_end}",
            "start": max_start,
            "end": max_end,
            "kind": "max",
            "metrics": [metric]
        })

    return segments


def compress_segment(segment: dict,
                     full_df: pd.DataFrame,
                     global_max_abs: float) -> pd.DataFrame:
    """
    Run the Garo L_inf compression on a single segment, increasing
    coefficient percentage from 5% to 30% in 1% steps until
    L_inf(error) <= 15% of global_max_abs, or we hit 30%.

    Returns:
      results_df: DataFrame of per-iteration metrics for this segment.
    Also writes:
      - <segment_label>_metrics.csv
      - <segment_label>_reconstruction.csv
    inside the segment-specific directory.
    """
    label = segment["label"]
    start_1b = segment["start"]  # 1-based inclusive
    end_1b = segment["end"]      # 1-based inclusive

    # Convert to 0-based slice indices for pandas .iloc
    start_idx = start_1b - 1
    end_idx = end_1b  # Python slice end is exclusive

    # Slice the full data
    seg_df = full_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    energy = seg_df["energy"].values.astype(float)
    N_seg = len(energy)

    if N_seg <= 0:
        raise ValueError(f"Segment {label} has non-positive length!")

    print(f"\n=== SEGMENT: {label} ===")
    print(f"  Indices (1-based): {start_1b}-{end_1b}")
    print(f"  Length N = {N_seg}")

    # Prepare output directory for this segment
    segment_dir = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(segment_dir, exist_ok=True)

    # Write a segment-only CSV for Garo to consume
    segment_csv_path = os.path.join(segment_dir, f"{label}_data.csv")
    seg_df.to_csv(segment_csv_path, index=False)

    # --- L_inf target for this segment (relative to global max) ---
    seg_max_abs = float(np.max(np.abs(energy)))
    target_linf = 0.15 * global_max_abs

    print(f"  Segment max |x_i|: {seg_max_abs:.4f}")
    print(f"  Global max |x_i|:  {global_max_abs:.4f}")
    print(f"  Target L_inf error: {target_linf:.4f} (15% of global max |x_i|)")

    # --- Coefficient percentage configuration ---
    coeff_pct_start = 0.005   # 5%
    coeff_pct_step = 0.005   # 1%
    coeff_pct_max = 0.30     # 30%

    results = []
    coeff_pct = coeff_pct_start
    iteration = 1
    best_row_idx = None

    while coeff_pct <= coeff_pct_max and iteration <= 200:
        B = max(1, int(round(coeff_pct * N_seg)))
        print(f"\nSEGMENT {label} - ITERATION {iteration}: "
              f"B={B} ({coeff_pct*100:.1f}% of N={N_seg})")

        try:
            # Run Garo (L_inf) on this segment-only CSV
            gk = GaroOptimizedLinf(B=B, scale_factor=1.0, csv_file=segment_csv_path)
        except Exception as e:
            print(f"  ❌ ERROR in Garo for segment {label}, iteration {iteration}: {e}")
            print("  Stopping this segment.")
            break

        # Garo writes "compressed_energy_with_timestamps.csv" in the CWD
        out_file = "compressed_energy_with_timestamps.csv"
        if not os.path.exists(out_file):
            print("  ❌ ERROR: Garo did not produce compressed_energy_with_timestamps.csv")
            print("  Stopping this segment.")
            break

        df_out = pd.read_csv(out_file)

        original = df_out["original_energy"].values.astype(float)
        recon = df_out["reconstructed_energy"].values.astype(float)
        errors = original - recon

        # --- PRIMARY METRIC: L_inf (max absolute error) ---
        linf_error = float(np.max(np.abs(errors)))
        linf_pct_global = 100.0 * linf_error / (global_max_abs if global_max_abs != 0 else 1.0)
        seg_max_for_pct = seg_max_abs if seg_max_abs != 0 else 1.0
        linf_pct_segment = 100.0 * linf_error / seg_max_for_pct

        status_met = (linf_error <= target_linf)

        # --- Additional diagnostics (L2-style) ---
        sse = float(np.sum(errors**2))
        rmse = float(np.sqrt(np.mean(errors**2)))
        l2_error = float(np.linalg.norm(errors))
        l2_norm_seg = float(np.linalg.norm(energy)) if np.linalg.norm(energy) != 0 else 1.0
        l2_error_pct = 100.0 * l2_error / l2_norm_seg

        noise_power = np.mean(errors**2) + 1e-10
        signal_power = np.mean(original**2) + 1e-10
        snr_db = 10 * np.log10(signal_power / noise_power)

        max_sig = np.max(np.abs(original)) + 1e-10
        psnr_db = 20 * np.log10(max_sig / np.sqrt(noise_power))

        ss_tot = np.sum((original - original.mean())**2) + 1e-10
        r2 = 1 - sse / ss_tot

        mae = float(np.mean(np.abs(errors)))

        # --- PRINT WHAT MATTERS ---
        print(f"  L_inf error: {linf_error:.4f} "
              f"(global target: {target_linf:.4f}, "
              f"{linf_pct_global:.2f}% of global max)")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Status: {'✅ TARGET MET' if status_met else '❌ TARGET NOT MET'}")

        results.append({
            "iteration": iteration,
            "segment_label": label,
            "kind": segment["kind"],
            "metrics": ",".join(segment["metrics"]),
            "start_1based": start_1b,
            "end_1based": end_1b,
            "N_segment": N_seg,
            "coefficients": B,
            "coeff_pct": coeff_pct * 100.0,
            "compression_pct": B / N_seg * 100.0,
            "linf_error": linf_error,
            "linf_pct_global": linf_pct_global,
            "linf_pct_segment": linf_pct_segment,
            "sse": sse,
            "rmse": rmse,
            "l2_error": l2_error,
            "l2_error_pct": l2_error_pct,
            "mean_abs_error": mae,
            "snr_db": float(snr_db),
            "psnr_db": float(psnr_db),
            "r_squared": float(r2),
            "target_met": status_met,
        })

        if status_met and best_row_idx is None:
            best_row_idx = len(results) - 1
            print(f"  ✅ L_inf target first met at "
                  f"{coeff_pct*100:.1f}% coefficients (iteration {iteration})")
            break

        coeff_pct += coeff_pct_step
        iteration += 1

    # Attach signal statistics (same for all iterations of this segment)
    signal_stats = compute_signal_stats(energy)
    for r in results:
        r.update({
            "signal_entropy": signal_stats["entropy"],
            "signal_mean": signal_stats["mean"],
            "signal_std": signal_stats["std"],
            "signal_min": signal_stats["data_min"],
            "signal_max": signal_stats["data_max"],
            "signal_range": signal_stats["data_range"],
            "signal_coeff_var": signal_stats["coeff_var"],
            "signal_median": signal_stats["median"],
            "signal_zero_fraction": signal_stats["zero_fraction"],
            "signal_gini_coefficient": signal_stats["gini_coefficient"],
        })

    results_df = pd.DataFrame(results)

    # Save per-segment metrics
    metrics_path = os.path.join(segment_dir, f"{label}_metrics.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f"  💾 Segment metrics saved to: {metrics_path}")

    # Save final reconstruction (i.e., from last iteration run)
    if os.path.exists("compressed_energy_with_timestamps.csv"):
        recon_out_path = os.path.join(segment_dir, f"{label}_reconstruction.csv")
        shutil.copy("compressed_energy_with_timestamps.csv", recon_out_path)
        print(f"  💾 Final reconstruction saved to: {recon_out_path}")
    else:
        print("  ⚠️  WARNING: No reconstruction file found to copy.")

    return results_df


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("Loading window statistics summary...")
    summary_df = pd.read_csv(WINDOW_STATS_FILE)
    print(summary_df)

    print("\nLoading full data file...")
    full_df = pd.read_csv(FULL_DATA_FILE)

    # Global max |x| over the full signal (for the 15% L_inf bound)
    global_max_abs = float(np.max(np.abs(full_df["energy"].values)))
    print(f"\nGlobal max |x_i| over full signal: {global_max_abs:.4f}")

    segments = build_segments(summary_df)

    overall_summary_rows = []

    for segment in segments:
        results_df = compress_segment(segment, full_df, global_max_abs)

        if results_df.empty:
            # Something failed; still record minimal info
            overall_summary_rows.append({
                "segment_label": segment["label"],
                "kind": segment["kind"],
                "metrics": ",".join(segment["metrics"]),
                "start_1based": segment["start"],
                "end_1based": segment["end"],
                "N_segment": segment["end"] - segment["start"] + 1,
                "best_coeff_pct": np.nan,
                "best_linf_error": np.nan,
                "best_linf_pct_global": np.nan,
                "target_reached": False,
            })
            continue

        # First row where target_met is True (if any)
        met_rows = results_df[results_df["target_met"] == True]
        if not met_rows.empty:
            best_row = met_rows.iloc[0]
            target_reached = True
        else:
            best_row = results_df.iloc[-1]  # last attempt
            target_reached = False

        overall_summary_rows.append({
            "segment_label": segment["label"],
            "kind": segment["kind"],
            "metrics": ",".join(segment["metrics"]),
            "start_1based": segment["start"],
            "end_1based": segment["end"],
            "N_segment": best_row["N_segment"],
            "best_coeff_pct": best_row["coeff_pct"],
            "best_linf_error": best_row["linf_error"],
            "best_linf_pct_global": best_row["linf_pct_global"],
            "best_sse": best_row["sse"],
            "target_reached": target_reached,
        })

    overall_df = pd.DataFrame(overall_summary_rows)
    overall_path = os.path.join(OUTPUT_ROOT, "segments_overall_summary.csv")
    overall_df.to_csv(overall_path, index=False)

    print("\n================= OVERALL SUMMARY =================")
    print(overall_df)
    print(f"\n💾 Overall summary saved to: {overall_path}")


if __name__ == "__main__":
    main()
