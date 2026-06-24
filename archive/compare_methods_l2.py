"""
Method 1 Visualization Script with Range Zoom and Blackout Analysis
===================================================================
Creates three figures:
1. Full signal comparison for Method 1 (Garofalakis-Kumar)
2. Zoomed view of samples 5720-5975 to show compression detail
3. Blackout period analysis (samples 5188-5649) with 3h before and 3h after

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configure matplotlib for better quality
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_compression_ratio_method1(results_dir):
    """Load compression ratio for Method 1 from garo_fullsignal_metrics.csv"""
    results_path = Path(results_dir)
    
    # Read from garo_fullsignal_metrics.csv (third trial - successful one)
    metrics_file = results_path / "garo_fullsignal_metrics.csv"
    if metrics_file.exists():
        try:
            df = pd.read_csv(metrics_file)
            if 'compression_pct' in df.columns:
                # Get the third trial (index 2) or last row if there are fewer trials
                if len(df) >= 3:
                    return float(df['compression_pct'].iloc[2])
                else:
                    return float(df['compression_pct'].iloc[-1])
        except Exception as e:
            print(f"  Warning: Could not read compression ratio from {metrics_file}: {e}")
    
    print(f"  Warning: Could not find garo_fullsignal_metrics.csv for Method 1")
    return None


def load_method1_data(results_dir):
    """
    Load data for Method 1
    
    Returns:
        original, reconstructed, compression_ratio
    """
    results_path = Path(results_dir)
    
    # Load data from reconstructed_method1.csv
    data_file = results_path / 'reconstructed_method1.csv'
    
    if not data_file.exists():
        raise FileNotFoundError(f"Could not find {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Check for required columns
    if 'original_energy' not in df.columns or 'reconstructed_energy' not in df.columns:
        raise ValueError("Required columns not found in data file")
    
    original = df['original_energy'].values
    reconstructed = df['reconstructed_energy'].values
    
    print(f"    ✓ Loaded original signal: {len(original):,} samples")
    print(f"    ✓ Loaded reconstruction: {len(reconstructed):,} samples")
    
    # Get compression ratio
    compression_ratio = load_compression_ratio_method1(results_path)
    
    return original, reconstructed, compression_ratio


def calculate_metrics(original, reconstructed):
    """Calculate performance metrics"""
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    # Compute metrics from data
    mae = mean_absolute_error(original, reconstructed)
    rmse = np.sqrt(mean_squared_error(original, reconstructed))
    nrmse = rmse / np.mean(original)
    r2 = r2_score(original, reconstructed)
    
    return mae, rmse, nrmse, r2


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_full_signal_plot(original, reconstructed, mae, rmse, r2, compression_ratio, output_file):
    """Create full signal comparison plot"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    # Time axis in DAYS (1 sample = 1 hour, so 24 samples = 1 day)
    time_axis = np.arange(len(original)) / 24
    
    # Plot original signal
    ax.plot(time_axis, original, 
            label='Original', 
            color='red', 
            linewidth=1.0, 
            alpha=0.8)
    
    # Plot reconstruction
    ax.plot(time_axis, reconstructed, 
            label='Garo-Kumar', 
            color='blue',
            linewidth=1.0, 
            alpha=0.8)
    
    # Format axes
    ax.set_xlabel('Time (Days)', fontsize=11)
    ax.set_ylabel('Energy (kWh)', fontsize=11)
    
    # No title
    
    # Signal legend (upper left)
    ax.legend(loc='upper left', fontsize=9)
    
    # Metrics box (lower right)
    cr_text = f"{compression_ratio:.2f}%" if compression_ratio is not None else "N/A"
    nrmse = rmse / np.mean(original)
    
    legend_text = (f"RMSE: {rmse:.4f}\n"
                   f"NRMSE: {nrmse:.4f}\n"
                   f"R²: {r2:.4f}\n"
                   f"Compression: {cr_text}")
    
    ax.text(0.98, 0.02, legend_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path.name}")
    
    plt.close(fig)


def create_blackout_plot(original, reconstructed, mae, rmse, r2, compression_ratio, output_file):
    """Create plot showing blackout period with context"""
    
    blackout_start = 5188
    blackout_end = 5649
    hours_context = 24
    
    start_idx = blackout_start - hours_context
    end_idx = blackout_end + hours_context
    
    original_range = original[start_idx:end_idx]
    reconstructed_range = reconstructed[start_idx:end_idx]
    
    # Calculate metrics for this range
    rmse_range = np.sqrt(mean_squared_error(original_range, reconstructed_range))
    nrmse_range = rmse_range / np.mean(original_range)
    r2_range = r2_score(original_range, reconstructed_range)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    # Time axis in DAYS
    time_axis = np.arange(start_idx, end_idx) / 24
    
    # Plot signals
    ax.plot(time_axis, original_range, 
            label='Original', 
            color='red', 
            linewidth=1.0, 
            alpha=0.8)
    
    ax.plot(time_axis, reconstructed_range, 
            label='Garo-Kumar', 
            color='blue',
            linewidth=1.0, 
            alpha=0.8)
    
    # Highlight blackout region
    ax.axvspan(blackout_start/24, blackout_end/24, alpha=0.2, color='gray', 
               label='Blackout Period')
    
    # Format axes
    ax.set_xlabel('Time (Days)', fontsize=11)
    ax.set_ylabel('Energy (kWh)', fontsize=11)
    
    plt.tight_layout()
    
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path.name}")
    
    plt.close(fig)


def create_weekly_zoom_plot(original, reconstructed, mae, rmse, r2, compression_ratio, output_file):
    """Create zoomed plot showing samples 5720-5975"""
    start_idx = 6472
    end_idx = 6640
    original_week = original[start_idx:end_idx]
    reconstructed_week = reconstructed[start_idx:end_idx]
    
    # Calculate metrics for this range
    rmse_week = np.sqrt(mean_squared_error(original_week, reconstructed_week))
    nrmse_week = rmse_week / np.mean(original_week)
    r2_week = r2_score(original_week, reconstructed_week)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    # Time axis in DAYS
    time_axis = np.arange(start_idx, end_idx) / 24
    
    # Plot signals
    ax.plot(time_axis, original_week, 
            label='Original', 
            color='red', 
            linewidth=1.0, 
            alpha=0.8)
    
    ax.plot(time_axis, reconstructed_week, 
            label='Garo-Kumar', 
            color='blue',
            linewidth=1.0, 
            alpha=0.8)
    
    # Format axes
    ax.set_xlabel('Time (Days)', fontsize=11)
    ax.set_ylabel('Energy (kWh)', fontsize=11)
    
    plt.tight_layout()
    
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path.name}")
    
    plt.close(fig)
    
# ============================================================================
# MAIN FUNCTION
# ============================================================================

def create_visualizations():
    """Create both full signal and weekly zoom visualizations"""
    
    print("\n" + "="*70)
    print("CREATING METHOD 1 VISUALIZATIONS")
    print("="*70 + "\n")
    
    try:
        # Load data
        print("📊 Loading Method 1 data...")
        original, reconstructed, comp_ratio = load_method1_data('results_method1_l2')
        
        # Calculate metrics
        mae, rmse, nrmse, r2  = calculate_metrics(original, reconstructed)
        
        # Print summary
        print(f"\n  Total data points: {len(original):,}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  NRMSE: {nrmse:.6f}")
        print(f"  R²:   {r2:.6f}")
        if comp_ratio is not None:
            print(f"  Compression Ratio: {comp_ratio:.2f}%")
        else:
            print(f"  Compression Ratio: N/A")
        
        # Create full signal plot
        print("\n📈 Creating full signal plot...")
        create_full_signal_plot(
            original, reconstructed, mae, rmse, r2, comp_ratio,
            'method1_full_comparison.png'
        )
        
        # Create zoom plot for samples 5720-5975
        print("\n🔍 Creating zoomed plot (samples 5720-5975)...")
        create_weekly_zoom_plot(
            original, reconstructed, mae, rmse, r2, comp_ratio,
            'method1_range_zoom.png'
        )
        
        # Create blackout analysis plot
        print("\n⚡ Creating blackout analysis plot (samples 5183-5654 with 5h context)...")
        create_blackout_plot(
            original, reconstructed, mae, rmse, r2, comp_ratio,
            'method1_blackout_analysis.png'
        )
        
        print("\n" + "="*70)
        print("✅ COMPLETED: All visualizations created successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    create_visualizations()