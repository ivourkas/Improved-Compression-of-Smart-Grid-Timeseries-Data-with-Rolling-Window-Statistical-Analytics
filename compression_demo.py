"""
Wavelet Compression Demo
========================
Interactive demo of the Garofalakis-Kumar algorithm.
Supports L2 (minimum squared error) and L-inf (minimum maximum error).

Usage (interactive):
    python compression_demo.py

Usage (pre-configured, skips all prompts):
    python compression_demo.py --method L2   --target 0.3
    python compression_demo.py --method Linf --target 15 --outlier --denoise --baseline

Flags:
    --method      L2 or Linf               (required if skipping prompts)
    --target      error % as float         (required if skipping prompts)
    --csv         path to CSV file         (default: data/test_data_v2.csv)
    --signal-col  name of signal column    (default: auto-detected)
    --time-col    name of timestamp column (default: auto-detected)
    --outlier     enable outlier removal
    --denoise     enable wavelet denoising
    --baseline    enable diurnal baseline subtraction
    --noise       add removed noise back when evaluating error
    --no-show     save plot without opening an interactive window
"""

import argparse
import datetime
import warnings
import numpy as np
import pandas as pd
import os
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, message=".*dateutil.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*infer format.*")


# =============================================================================
# INPUT HELPERS
# =============================================================================

def ask_choice(prompt, options):
    """Ask user to pick from a numbered list. Returns the chosen key."""
    for i, (_, label) in enumerate(options.items(), 1):
        print(f"  [{i}] {label}")
    while True:
        raw = input(f"{prompt}: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return list(options.keys())[int(raw) - 1]
        print(f"      Enter a number between 1 and {len(options)}.")


def ask_yes_no(prompt, default=False):
    tag = "Y/n" if default else "y/N"
    while True:
        raw = input(f"  {prompt} [{tag}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("      Please enter y or n.")


def ask_float(prompt, default, min_val=None, max_val=None):
    while True:
        raw = input(f"  {prompt} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"      Must be >= {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"      Must be <= {max_val}.")
                continue
            return val
        except ValueError:
            print("      Please enter a number.")


# =============================================================================
# COLUMN DETECTION
# =============================================================================

def _detect_columns(df):
    """Return (numeric_cols, time_cols) found in df."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    time_cols = []
    for col in df.columns:
        if col in numeric_cols:
            continue
        try:
            pd.to_datetime(df[col].iloc[:5])
            time_cols.append(col)
        except Exception:
            pass
    return numeric_cols, time_cols


def _pick_column(label, candidates, arg_value, df=None, allow_prompt=True):
    """Return chosen column name, prompting only when interactive selection is allowed."""
    if arg_value and arg_value in candidates:
        return arg_value
    if arg_value and arg_value not in candidates:
        print(f"  WARNING: --{label} '{arg_value}' not found. Auto-detecting.")
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    if not allow_prompt:
        chosen = candidates[0]
        print(f"  Auto-selected {label} column: {chosen}")
        return chosen
    print(f"\n  Multiple {label} columns found — pick one:")
    for i, col in enumerate(candidates, 1):
        if df is not None:
            col_min = df[col].min()
            col_max = df[col].max()
            print(f"    [{i}] {col:<20}  range: [{col_min:.2f}, {col_max:.2f}]")
        else:
            print(f"    [{i}] {col}")
    while True:
        choice = input(f"  Choose {label} column [default: {candidates[0]}]: ").strip()
        if choice == "":
            return candidates[0]
        if choice in candidates:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        print(f"      Enter column name or number 1–{len(candidates)}.")


# =============================================================================
# MAIN DEMO
# =============================================================================

def main(args=None):
    # Determine whether we are running in preset mode (all args supplied)
    preset_mode = (
        args is not None
        and args.method is not None
        and args.target is not None
    )

    print()
    print("=" * 62)
    print("   WAVELET COMPRESSION DEMO")
    print("   Garofalakis-Kumar Algorithm")
    if preset_mode:
        print("   [preset mode — no prompts]")
    print("=" * 62)

    # ── 1. Compression method ─────────────────────────────────────
    if preset_mode:
        method = args.method
        print(f"\nMethod   : {method}")
    else:
        print()
        print("Step 1 — Compression method")
        method = ask_choice("Choose", {
            "L2":   "L2   minimises total squared error  (best overall accuracy)",
            "Linf": "Linf minimises maximum absolute error  (worst-case guarantee)",
        })

    # ── 2. Preprocessing options ──────────────────────────────────
    if preset_mode:
        use_outlier  = args.outlier
        use_denoise  = args.denoise
        use_baseline = args.baseline
        add_noise    = args.noise
        print(f"Outlier  : {use_outlier}")
        print(f"Denoise  : {use_denoise}")
        print(f"Baseline : {use_baseline}")
        print(f"Noise eval: {add_noise}")
    else:
        print()
        print("Step 2 — Preprocessing options")
        use_outlier  = ask_yes_no("Remove outliers before compression?",              default=False)
        use_denoise  = ask_yes_no("Apply wavelet denoising?",                         default=False)
        use_baseline = ask_yes_no("Subtract hourly-average baseline (diurnal)?",      default=False)
        add_noise    = ask_yes_no("Add removed noise back when evaluating error?",     default=False)

    # ── 3. Target error ───────────────────────────────────────────
    if preset_mode:
        target_pct = args.target
        print(f"Target   : {target_pct}%")
    else:
        print()
        print("Step 3 — Target error")
        if method == "L2":
            print("  L2 target: compression stops when")
            print("    sum(errors^2) <= X% of sum(signal^2)")
            print("  Typical range: 0.1 – 2.0")
            target_pct = ask_float("Target %", default=0.3, min_val=0.001, max_val=100.0)
        else:
            print("  L-inf target: compression stops when")
            print("    max(|error|) <= X% of signal peak")
            print("  Typical range: 5 – 20")
            target_pct = ask_float("Target %", default=15.0, min_val=0.001, max_val=100.0)

    # ── 4. Load data ──────────────────────────────────────────────
    print()
    csv_file = args.csv if (args and args.csv) else "data/test_data_v2.csv"
    if not os.path.exists(csv_file):
        print(f"ERROR: '{csv_file}' not found in the current directory.")
        return

    df = pd.read_csv(csv_file)
    numeric_cols, time_cols = _detect_columns(df)

    if not numeric_cols:
        print("ERROR: no numeric columns found in CSV.")
        return

    signal_col = _pick_column(
        "signal", numeric_cols,
        arg_value=args.signal_col if args else None,
        df=df,
        allow_prompt=not preset_mode,
    )
    time_col = _pick_column(
        "timestamp", time_cols,
        arg_value=args.time_col if args else None,
        allow_prompt=not preset_mode,
    )

    raw_values = df[signal_col].values
    original_N = len(raw_values)
    N = 2 ** int(np.floor(np.log2(original_N)))
    signal = raw_values[:N]

    unit = signal_col   # use column name as axis/unit label
    print(f"CSV        : {csv_file}")
    print(f"Signal col : {signal_col}   |   Timestamp col: {time_col or 'none'}")
    print(f"Points     : {original_N}  -->  using {N} (largest power of 2)")
    print(f"Range      : [{signal.min():.2f}, {signal.max():.2f}]")

    # ── 5. Preprocessing ──────────────────────────────────────────
    from garo_l2_for_csv import (
        detect_and_interpolate_outliers,
        quick_denoise_signal,
    )

    # Outlier removal
    if use_outlier:
        print("\nDetecting outliers...")
        data_clean, outlier_info = detect_and_interpolate_outliers(
            signal, low_threshold=10.0, jump_threshold=60.0
        )
        outlier_indices = np.array(outlier_info["indices"], dtype=int)
        clean_mask = np.ones(len(signal), dtype=bool)
        clean_mask[outlier_indices] = False
        original_outlier_values = (
            signal[outlier_indices] if outlier_indices.size > 0 else np.array([])
        )
        print(f"  {outlier_indices.size} outlier(s) replaced by interpolation.")
    else:
        data_clean = signal.copy()
        outlier_indices = np.array([], dtype=int)
        clean_mask = np.ones(len(signal), dtype=bool)
        original_outlier_values = np.array([])

    # Denoising
    if use_denoise:
        print("Denoising signal...")
        clean_signal, noise_component = quick_denoise_signal(data_clean)
        noise_rms = np.sqrt(np.mean(noise_component ** 2))
        print(f"  Noise RMS removed: {noise_rms:.4f} [{unit}]")
    else:
        clean_signal = data_clean.copy()
        noise_component = np.zeros_like(clean_signal)

    # Baseline
    if use_baseline and time_col:
        dfN = df.iloc[:N].copy()
        dfN[time_col] = pd.to_datetime(dfN[time_col])
        hours = dfN[time_col].dt.hour.values
        hour_means = np.array([
            clean_signal[hours == h].mean() if np.any(hours == h) else 0.0
            for h in range(24)
        ])
        baseline = hour_means[hours]
        print(f"Baseline subtracted  (mean={baseline.mean():.2f}, "
              f"range [{baseline.min():.2f}, {baseline.max():.2f}]).")
    else:
        if use_baseline and not time_col:
            print("  WARNING: no timestamp column found — baseline skipped.")
            use_baseline = False
        hours = np.zeros(N, dtype=int)
        baseline = np.zeros_like(clean_signal)

    residual_signal = clean_signal - baseline

    # Write temp CSV for the Garo class (always uses "energy"/"demand" internally)
    df_temp = df.copy()
    padded = np.pad(residual_signal, (0, original_N - N), constant_values=residual_signal[-1])
    df_temp["energy"] = padded
    df_temp["demand"] = padded
    df_temp.to_csv("_demo_temp.csv", index=False)

    # ── 6. Target threshold ───────────────────────────────────────
    clean_for_target = clean_signal[clean_mask]
    if method == "L2":
        total_energy = np.sum(clean_for_target ** 2)
        target_error = (target_pct / 100.0) * total_energy
    else:
        data_max = np.max(np.abs(clean_for_target))
        target_error = (target_pct / 100.0) * data_max

    # ── 7. Compression search loop ────────────────────────────────
    if method == "L2":
        from garo_l2_for_csv import GaroOptimized as GaroClass
    else:
        from garo_inf_for_csv import GaroOptimizedLinf as GaroClass

    coeff_pct_start = 0.05
    coeff_pct_step  = 0.01
    coeff_pct_max   = 0.50

    print()
    print("=" * 62)
    print(f"  Running {method} compression search")
    if method == "L2":
        print(f"  Target: sum(errors^2) <= {target_error:.2f}  ({target_pct:.3f}% of signal energy)")
    else:
        print(f"  Target: max|error| <= {target_error:.4f} [{unit}]  ({target_pct:.1f}% of peak {data_max:.2f})")
    print(f"  Searching coefficients from {coeff_pct_start*100:.0f}% to {coeff_pct_max*100:.0f}% of N={N}")
    print("=" * 62)
    print(f"  {'B':>6}  {'B/N':>6}  {'metric error':>18}  {'target':>18}  status")
    print(f"  {'-'*6}  {'-'*6}  {'-'*18}  {'-'*18}  ------")

    B_init = max(1, int(coeff_pct_start * N))
    gk = GaroClass(
        B=B_init, csv_file="_demo_temp.csv",
        scale_factor=1.0, verbose=False, write_files=False
    )

    coeff_pct  = coeff_pct_start
    iteration  = 1
    found      = False
    final_B    = B_init
    final_coeff_pct = coeff_pct_start
    metric_error = np.inf
    final_reconstructed = None

    while coeff_pct <= coeff_pct_max and iteration <= 200:
        B = max(1, int(coeff_pct * N))

        if iteration > 1:
            gk.recompress(B, write_files=False)

        residual_recon   = gk.df_output["reconstructed_energy"].values[:N]
        clean_recon      = baseline + residual_recon
        err_clean        = clean_signal - clean_recon
        err_non_outliers = err_clean[clean_mask]

        if method == "L2":
            metric_error = np.sum(err_non_outliers ** 2)
            metric_label = f"{metric_error:>18.2f}"
            target_label = f"{target_error:>18.2f}"
        else:
            metric_error = np.max(np.abs(err_non_outliers))
            metric_label = f"{metric_error:>14.4f} {unit}"
            target_label = f"{target_error:>14.4f} {unit}"

        if add_noise:
            recon_eval = clean_recon + noise_component
        else:
            recon_eval = clean_recon.copy()
        if outlier_indices.size > 0:
            recon_eval[outlier_indices] = original_outlier_values

        status = "OK" if metric_error <= target_error else "  "
        print(f"  {B:>6}  {coeff_pct*100:>5.1f}%  {metric_label}  {target_label}  {status}")

        if metric_error <= target_error:
            found = True
            final_B = B
            final_coeff_pct = coeff_pct
            final_reconstructed = recon_eval.copy()
            break

        final_B = B
        final_coeff_pct = coeff_pct
        final_reconstructed = recon_eval.copy()

        coeff_pct = round(coeff_pct + coeff_pct_step, 4)
        iteration += 1

    # ── 8. Final metrics ──────────────────────────────────────────
    original = signal
    errors   = original - final_reconstructed
    rmse     = float(np.sqrt(np.mean(errors ** 2)))
    max_err  = float(np.max(np.abs(errors)))
    ss_res   = float(np.sum(errors ** 2))
    ss_tot   = float(np.sum((original - original.mean()) ** 2)) + 1e-10
    r2       = float(1.0 - ss_res / ss_tot)
    snr_db   = float(10 * np.log10(np.mean(original**2) / (np.mean(errors**2) + 1e-10)))
    l2_pct   = (ss_res / (np.sum(original**2) + 1e-10)) * 100.0
    linf_pct = (max_err / (np.max(np.abs(original)) + 1e-10)) * 100.0

    print()
    print("=" * 62)
    print(f"  RESULTS  —  {method} compression")
    print("=" * 62)
    if found:
        print(f"  Target met at {final_coeff_pct*100:.1f}% coefficients")
    else:
        print(f"  Target NOT met  (reached {final_coeff_pct*100:.1f}% budget limit)")
    print()
    baseline_vars = 24 if use_baseline else 0
    total_vars = final_B + baseline_vars
    total_pct  = total_vars / N * 100
    if use_baseline:
        print(f"  Coefficients kept : {final_B} wavelet + 24 baseline = {total_vars} / {N}  ({total_pct:.1f}%)")
    else:
        print(f"  Coefficients kept : {final_B} / {N}  ({total_pct:.1f}%)")
    print(f"  RMSE              : {rmse:.4f} [{unit}]")
    print(f"  Max absolute error: {max_err:.4f} [{unit}]  ({linf_pct:.2f}% of peak)")
    print(f"  L2 error          : {l2_pct:.4f}% of signal energy")
    print(f"  R squared         : {r2:.6f}")
    print(f"  SNR               : {snr_db:.2f} dB")
    print("=" * 62)

    # ── 9. Save reconstruction ────────────────────────────────────
    output_dir = Path("results_demo")
    output_dir.mkdir(exist_ok=True)

    recon_cols = {}
    if time_col and time_col in df.columns:
        recon_cols[time_col] = df[time_col].iloc[:N].values
    recon_cols[f"original_{signal_col}"]      = original
    recon_cols[f"reconstructed_{signal_col}"] = final_reconstructed
    recon_cols["error"]                       = errors
    df_recon = pd.DataFrame(recon_cols)
    out_csv = output_dir / f"reconstruction_{method.lower()}.csv"
    df_recon.to_csv(out_csv, index=False)
    print(f"\n  Reconstruction saved    : {out_csv}")

    # ── 9b. Save compressed coefficients ─────────────────────────
    # The compressed representation is:
    #   • (optional) 24 hourly-mean baseline values  [if baseline was subtracted]
    #   • B sparse Haar wavelet coefficients          [tree index + value]
    # Together these B (+ 24) numbers fully reconstruct the N-point signal.
    if hasattr(gk, "source"):
        rows = []
        total_stored = final_B + (24 if (use_baseline and time_col) else 0)
        # --- metadata (type=meta) ---
        for key, val in [
            ("N",            str(N)),
            ("B",            str(final_B)),
            ("total_stored", str(total_stored)),
            ("storage_pct",  f"{total_stored/N*100:.2f}"),
            ("method",       method),
            ("signal_col",   signal_col),
            ("baseline",     str(use_baseline and bool(time_col))),
        ]:
            rows.append({"type": "meta", "index": key, "value": val})
        # --- baseline (type=baseline, one row per hour) ---
        if use_baseline and time_col:
            for h in range(24):
                rows.append({"type": "baseline", "index": str(h),
                             "value": f"{hour_means[h]:.8f}"})
        # --- sparse Haar wavelet coefficients (type=coeff) ---
        for idx in np.nonzero(gk.source)[0]:
            rows.append({"type": "coeff", "index": str(idx),
                         "value": f"{gk.source[idx]:.8f}"})

        coeffs_csv = output_dir / f"compressed_coeffs_{method.lower()}.csv"
        pd.DataFrame(rows).to_csv(coeffs_csv, index=False)
        baseline_note = f" + 24 baseline = {total_stored}" if (use_baseline and time_col) else ""
        print(f"  Compressed coefficients : {coeffs_csv}")
        print(f"  Storage: {final_B} wavelet coeffs{baseline_note} values  "
              f"vs {N} original  =>  {total_stored/N*100:.1f}% storage")

    # ── 9c. Append to metrics log ─────────────────────────────────
    log_csv = output_dir / "metrics_log.csv"
    log_row = {
        "timestamp":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method":          method,
        "target_pct":      target_pct,
        "outlier":         use_outlier,
        "denoise":         use_denoise,
        "baseline":        use_baseline,
        "noise_eval":      add_noise,
        "signal_col":      signal_col,
        "csv_file":        csv_file,
        "N":               N,
        "B":               final_B,
        "compression_pct": round(final_coeff_pct * 100, 2),
        "target_met":      found,
        "RMSE":            round(rmse, 4),
        "max_err":         round(max_err, 4),
        "R2":              round(r2, 6),
        "SNR_dB":          round(snr_db, 2),
        "L2_pct":          round(l2_pct, 4),
        "Linf_pct":        round(linf_pct, 4),
    }
    write_header = not log_csv.exists()
    pd.DataFrame([log_row]).to_csv(log_csv, mode="a", header=write_header, index=False)
    print(f"  Metrics logged          : {log_csv}")

    # ── 10. Plot ──────────────────────────────────────────────────
    _plot(df, original, final_reconstructed, errors,
          method, final_B, N, r2, rmse, max_err, found, output_dir,
          time_col=time_col, unit=unit, target_pct=target_pct,
          show_plot=not (args.no_show if args else False))

    # Cleanup temp file
    if os.path.exists("_demo_temp.csv"):
        os.remove("_demo_temp.csv")

    return df_recon


# =============================================================================
# PLOT
# =============================================================================

def _plot(df, original, reconstructed, errors,
          method, B, N, r2, rmse, max_err, found, output_dir,
          time_col=None, unit="signal", target_pct=None, show_plot=True):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  matplotlib not available — skipping plot.")
        return

    use_timestamps = time_col is not None and time_col in df.columns
    if use_timestamps:
        timestamps = pd.to_datetime(df[time_col].iloc[:N])
    else:
        timestamps = np.arange(N)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    status_str = "Target met" if found else "Target NOT met"
    target_str = f"Target: {target_pct}%   |   " if target_pct is not None else ""
    fig.suptitle(
        f"{method} Wavelet Compression  |  {B}/{N} coefficients ({B/N*100:.1f}%)  |  "
        f"{target_str}"
        f"R\u00b2={r2:.4f}   RMSE={rmse:.3f}   Max err={max_err:.3f}   [{status_str}]",
        fontsize=10
    )

    ax1.plot(timestamps, original,      color="steelblue", lw=0.8, alpha=0.9, label="Original")
    ax1.plot(timestamps, reconstructed, color="tomato",    lw=0.8, alpha=0.9, label="Reconstructed")
    ax1.set_ylabel(unit)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.25)

    ax2.fill_between(timestamps, errors, 0, color="gray", alpha=0.5)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_ylabel(f"Error [{unit}]")
    ax2.set_xlabel("Time" if use_timestamps else "Sample index")
    ax2.grid(True, alpha=0.25)

    if use_timestamps:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plot_path = output_dir / f"compression_{method.lower()}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {plot_path}")
    backend = plt.get_backend().lower()
    if show_plot and "agg" not in backend:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wavelet Compression Demo")
    parser.add_argument("--method",      choices=["L2", "Linf"], default=None)
    parser.add_argument("--target",      type=float,             default=None)
    parser.add_argument("--csv",         default=None,           help="Path to CSV file (default: data/test_data_v2.csv)")
    parser.add_argument("--signal-col",  default=None,           help="Name of signal column (default: auto-detect)")
    parser.add_argument("--time-col",    default=None,           help="Name of timestamp column (default: auto-detect)")
    parser.add_argument("--outlier",     action="store_true",    default=False)
    parser.add_argument("--denoise",     action="store_true",    default=False)
    parser.add_argument("--baseline",    action="store_true",    default=False)
    parser.add_argument("--noise",       action="store_true",    default=False)
    parser.add_argument("--no-show",     action="store_true",    default=False,
                        help="Save the plot without opening a GUI window")
    args = parser.parse_args()
    main(args)
