"""
plot_week_zoom.py
=================
Selects a fixed representative week from the latest compression results
and plots a side-by-side comparison of L2 vs L-inf reconstruction.

Run:
    py plot_week_zoom.py

Output:
    results_demo/week_zoom.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────
RESULTS_DIR   = Path("results_demo")
WEEK_SEED     = 42          # fixed seed → always the same week for both methods
WEEK_DAYS     = 7           # how many days to show
OUTPUT_FILE   = RESULTS_DIR / "week_zoom.png"

# ── Load results ──────────────────────────────────────────────────
def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}\nRun the demo first.")
    df = pd.read_csv(path)
    # find timestamp column (first non-numeric column)
    time_col = next(
        (c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])), None
    )
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col).sort_index()
    return df

df_l2   = load_csv(RESULTS_DIR / "reconstruction_l2.csv")
df_linf = load_csv(RESULTS_DIR / "reconstruction_linf.csv")

# ── Find a complete week that exists in both datasets ─────────────
# Use the intersection of their date ranges
common_start = max(df_l2.index[0],   df_linf.index[0])
common_end   = min(df_l2.index[-1],  df_linf.index[-1])

# Build list of all possible week start dates (Monday or any weekday)
week_td   = pd.Timedelta(days=WEEK_DAYS)
all_starts = pd.date_range(
    start=common_start,
    end=common_end - week_td,
    freq="D"
)

if len(all_starts) == 0:
    raise ValueError("Not enough overlapping data to find a full week.")

# Pick deterministically with seed
rng = np.random.default_rng(WEEK_SEED)
week_start = all_starts[rng.integers(0, len(all_starts))]
week_end   = week_start + week_td

print(f"Selected week: {week_start.date()}  →  {week_end.date()}")

# Slice both DataFrames to that week
sl2   = df_l2[  (df_l2.index   >= week_start) & (df_l2.index   < week_end)]
slinf = df_linf[(df_linf.index >= week_start) & (df_linf.index < week_end)]

# Detect column names (original / reconstructed / error)
def find_col(df, keyword):
    matches = [c for c in df.columns if keyword.lower() in c.lower()]
    return matches[0] if matches else df.columns[0]

orig_col  = find_col(df_l2, "original")
recon_col = find_col(df_l2, "reconstruct")
err_col   = find_col(df_l2, "error")

# ── Plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 2, figsize=(16, 8),
    gridspec_kw={"height_ratios": [3, 1]},
    sharex="col"
)
fig.suptitle(
    f"One-Week Zoom  ·  {week_start.strftime('%d %b')} – {week_end.strftime('%d %b %Y')}  "
    f"·  L2 (left)  vs  L∞ (right)",
    fontsize=13, fontweight="bold"
)

def _fmt_axis(ax_top, ax_bot, df_slice, method_label, color):
    ts = df_slice.index

    ax_top.plot(ts, df_slice[orig_col],  color="steelblue", lw=1.2, alpha=0.9, label="Original")
    ax_top.plot(ts, df_slice[recon_col], color=color,       lw=1.2, alpha=0.9, label=f"{method_label} reconstructed")
    ax_top.set_ylabel("Signal")
    ax_top.legend(loc="upper right", fontsize=9)
    ax_top.grid(True, alpha=0.25)
    ax_top.set_title(method_label, fontsize=11, pad=4)

    # RMSE and max error for this week
    errs = df_slice[err_col].values
    rmse = np.sqrt(np.mean(errs**2))
    maxe = np.max(np.abs(errs))
    ax_top.text(
        0.01, 0.03,
        f"RMSE={rmse:.2f}   Max|err|={maxe:.2f}",
        transform=ax_top.transAxes,
        fontsize=8, color="dimgray", va="bottom"
    )

    ax_bot.fill_between(ts, errs, 0, color=color, alpha=0.45)
    ax_bot.axhline(0, color="black", lw=0.6)
    ax_bot.set_ylabel("Error")
    ax_bot.grid(True, alpha=0.25)
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax_bot.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=30, ha="right")

_fmt_axis(axes[0, 0], axes[1, 0], sl2,   "L2",  "tomato")
_fmt_axis(axes[0, 1], axes[1, 1], slinf, "L∞",  "darkorange")

plt.tight_layout()
RESULTS_DIR.mkdir(exist_ok=True)
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_FILE}")
plt.show()
