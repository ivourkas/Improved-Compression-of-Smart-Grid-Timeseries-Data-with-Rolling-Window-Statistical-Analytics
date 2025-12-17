"""
Garofalakis-Kumar L2 (Squared Error) Wavelet Synopsis Algorithm
FULLY OPTIMIZED VERSION with Numba JIT

MATHEMATICAL OVERVIEW:
====================
This algorithm creates a compressed representation of a time series using wavelets
while minimizing the root mean squared error (L2 norm).

Key Concepts:
1. HAAR WAVELETS: Decomposes signal into averages and differences at multiple scales
   - Average coefficient: (left + right) / 2
   - Detail coefficient: (left - right) / 2

2. L2 ERROR: Minimizes the ROOT MEAN SQUARED ERROR across all points
   - L2 norm = sqrt(sum of squared errors / N)
   - Unlike L-inf (max error), L2 focuses on overall accuracy
   - More tolerant of occasional large errors than L-inf

3. DYNAMIC PROGRAMMING: Finds optimal subset of B coefficients to keep
   - Considers all possible ways to distribute budget B across tree
   - Tracks error for all possible ancestor paths

4. BINARY TREE STRUCTURE: 
   - N leaf nodes contain original data
   - Internal nodes contain wavelet coefficients
   - Root (index 0) contains overall average

OPTIMIZATIONS:
- Numba JIT compilation for DP loops
- Deferred coefficient backtracking
- Vectorized operations throughout
- Cached incoming values
- Single transform across iterations
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import os
import time 
from pathlib import Path
import pywt
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from numba import njit, prange
from numba.typed import Dict as NumbaDict
from numba import types

# Structured dtype for wavelet tree nodes
TREE_DTYPE = np.dtype([
    ('average', np.float64),
    ('detail', np.float64),
    ('weight', np.float64)
])

USE_OUTLIER_REMOVAL = False  
ADD_NOISE_BACK_FOR_EVAL = False
USE_DENOISING = False

os.makedirs("results_method1_l2", exist_ok=True)


# =============================================================================
# NUMBA JIT COMPILED FUNCTIONS
# =============================================================================

@njit(cache=True)
def compute_incoming_numba(index: int, paths: int, logN: int, 
                           tree_average_0: float, tree_detail: np.ndarray) -> np.ndarray:
    """Numba-optimized incoming value computation."""
    incoming = np.zeros(paths, dtype=np.float64)
    
    for j in range(paths):
        cur = j
        coeffindex = index
        val = 0.0
        
        for bit in range(logN):
            if coeffindex <= 0:
                break
            
            if coeffindex == 1:
                if (cur & 1) == 1:
                    val += tree_average_0
            else:
                if (cur & 1) == 1 and (coeffindex & 1) == 1:
                    val -= tree_detail[coeffindex // 2]
                if (cur & 1) == 1 and (coeffindex & 1) == 0:
                    val += tree_detail[coeffindex // 2]
            
            cur = cur >> 1
            coeffindex = coeffindex >> 1
        
        incoming[j] = val
    
    return incoming


@njit(cache=True)
def compute_leaf_errors(incoming: np.ndarray, detail: float, 
                        left_avg: float, right_avg: float,
                        left_weight: float, right_weight: float,
                        paths: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute leaf-level errors for keep/don't-keep decisions."""
    # KEEP errors
    klerror = (incoming + detail - left_avg) ** 2
    krerror = (incoming - detail - right_avg) ** 2
    kerror_all = klerror + krerror
    
    # DON'T KEEP errors  
    dlerror = left_weight * (incoming - left_avg) ** 2
    drerror = right_weight * (incoming - right_avg) ** 2
    derror_all = dlerror + drerror
    
    return kerror_all, derror_all


@njit(cache=True)
def dp_internal_loop(localB: int, paths: int, fullchildB: int,
                     L_error: np.ndarray, R_error: np.ndarray,
                     lenL: int, lenR: int, L_paths: int, R_paths: int):
    """
    Numba-optimized DP loop for internal nodes.
    Returns: (dp_error, dp_keep, dp_leftspace)
    """
    dp_error = np.full((localB + 1, paths), np.inf, dtype=np.float64)
    dp_keep = np.zeros((localB + 1, paths), dtype=np.bool_)
    dp_leftspace = np.zeros((localB + 1, paths), dtype=np.int32)
    
    for b in range(localB + 1):
        for j in range(paths):
            # OPTION 1: KEEP coefficient
            klocalchildB = min(fullchildB, b - 1)
            kchildpath = 2 * j + 1
            kbesterror = np.inf
            kbestleftspace = -1
            
            if klocalchildB >= 0 and kchildpath < L_paths and kchildpath < R_paths:
                start_left = max(0, b - 1 - klocalchildB)
                end_left = min(klocalchildB + 1, lenL)
                
                for leftspace in range(start_left, end_left):
                    rightspace = b - 1 - leftspace
                    if 0 <= rightspace < lenR:
                        error = L_error[leftspace, kchildpath] + R_error[rightspace, kchildpath]
                        if error < kbesterror:
                            kbesterror = error
                            kbestleftspace = leftspace
            
            # OPTION 2: DON'T KEEP coefficient
            dlocalchildB = min(fullchildB, b)
            dchildpath = 2 * j
            dbesterror = np.inf
            dbestleftspace = -1
            
            if dchildpath < L_paths and dchildpath < R_paths:
                start_left = max(0, b - dlocalchildB)
                end_left = min(dlocalchildB + 1, lenL)
                
                for leftspace in range(start_left, end_left):
                    rightspace = b - leftspace
                    if 0 <= rightspace < lenR:
                        error = L_error[leftspace, dchildpath] + R_error[rightspace, dchildpath]
                        if error < dbesterror:
                            dbesterror = error
                            dbestleftspace = leftspace
            
            # Choose better option
            if dbesterror <= kbesterror:
                dp_keep[b, j] = False
                dp_leftspace[b, j] = dbestleftspace
                dp_error[b, j] = dbesterror
            else:
                dp_keep[b, j] = True
                dp_leftspace[b, j] = kbestleftspace
                dp_error[b, j] = kbesterror
    
    return dp_error, dp_keep, dp_leftspace


@njit(cache=True)
def reconstruct_signal_numba(source: np.ndarray, N: int) -> np.ndarray:
    """Numba-optimized signal reconstruction from wavelet coefficients."""
    recon = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        pnt = i + N
        val = source[0]
        
        while pnt > 1:
            if (pnt % 2) == 0:
                pnt = pnt // 2
                val = val + source[pnt]
            else:
                pnt = (pnt - 1) // 2
                val = val - source[pnt]
        
        recon[i] = val
    
    return recon


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_and_interpolate_outliers(data, low_threshold=1.0, jump_threshold=60.0):
    data = np.asarray(data)
    
    low_mask = data < low_threshold
    diffs = np.diff(data, prepend=data[0])
    jump_mask = np.abs(diffs) > jump_threshold
    
    outlier_mask = low_mask | jump_mask
    outlier_indices = np.where(outlier_mask)[0]
    
    if len(outlier_indices) == 0:
        return data.copy(), {'indices': np.array([]), 'values': np.array([])}
    
    print(f"   ⚠️  Detected {len(outlier_indices)} outliers "
          f"({100*len(outlier_indices)/len(data):.1f}%)")
    
    outlier_info = {
        'indices': outlier_indices,
        'values': data[outlier_indices].copy()
    }
    
    clean_data = data.copy()
    valid_mask = ~outlier_mask
    valid_indices = np.where(valid_mask)[0]
    
    interpolator = interp1d(valid_indices, data[valid_indices],
                            kind='linear', fill_value='extrapolate')
    clean_data[outlier_indices] = interpolator(outlier_indices)
    
    return clean_data, outlier_info


def compute_signal_stats(signal: np.ndarray, bins: int = 50) -> dict:
    hist, _ = np.histogram(signal, bins=min(bins, max(5, len(signal)//10)), density=True)
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
    
    sorted_signal = np.sort(np.abs(signal))
    n = len(sorted_signal)
    cumsum = np.cumsum(sorted_signal)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_signal)) / (n * cumsum[-1]) - (n + 1) / n if cumsum[-1] > 0 else 0

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
        "gini_coefficient": float(gini)
    }


def quick_denoise_signal(signal):
    """Simple wavelet denoising."""
    median_filtered = median_filter(signal, size=5)
    coeffs = pywt.wavedec(median_filtered, 'db4', level=3)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = 1.2 * sigma * np.sqrt(2 * np.log(len(median_filtered)))
    
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    
    clean = pywt.waverec(new_coeffs, 'db4')[:len(signal)]
    noise = signal - clean
    
    return clean, noise


# =============================================================================
# OPTIMIZED GARO CLASS
# =============================================================================

class GaroOptimized:
    """Fully optimized Garofalakis-Kumar L2 Wavelet Synopsis Algorithm."""
    
    def __init__(self, B: int, csv_file: str = "test_data.csv",
                 scale_factor: float = 1.0, verbose: bool = False):
        
        self.B = B
        self.scale_factor = scale_factor
        self.csv_file = csv_file
        self.verbose = verbose
        self.ultimate_error = float('inf')
        self._incoming_cache = {}
        
        # Read and parse CSV data
        self.N, self.data, self.original_N, self.original_df, self.deferred_count, self.pad_count = self._read_csv(csv_file)
        
        # Allocate memory for wavelet tree as structured array
        self.tree = np.zeros(2 * self.N, dtype=TREE_DTYPE)
        self.tree['weight'] = 1.0
        
        # Store data in leaf nodes
        data_scaled = np.array(self.data) / self.scale_factor
        self.tree['average'][self.N:2*self.N] = data_scaled
        self.tree['detail'][self.N:2*self.N] = 0.0
        self.tree['average'][:self.N] = self.tree['average'][self.N:2*self.N]
        
        if self.verbose:
            print(f"\n\t**Garofalakis L_2 Method (OPTIMIZED) ***\n")
            print(f"Original data size: {self.original_N}")
            print(f"Truncated signal size N = {self.N}")
            print(f"Scaled data by factor: {self.scale_factor}")
            if self.deferred_count > 0:
                print(f"Note: {self.deferred_count} points deferred")
        
        # Perform Haar transformation ONCE
        self._transform()
        print("Transform complete")
        
        # Precompute all incoming values for leaf nodes (they don't change)
        self._precompute_incoming()
        print("Incoming values precomputed")
        
        # Run DP
        self._recurse_root(write_files=True)
        print("Recurse complete!")
    
    def _read_csv(self, filename: str) -> Tuple[int, List, int, pd.DataFrame, int, int]:
        """Read CSV file and prepare data."""
        df = pd.read_csv(filename)
        energy_values = df['energy'].values
        original_N = len(energy_values)
        
        N = 2 ** int(np.floor(np.log2(original_N)))
        
        if len(energy_values) > N:
            truncated_data = energy_values[:N]
            deferred_count = original_N - N
            pad_count = 0
        else:
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
    
    def _get_depth(self, index: int) -> int:
        """Get depth of node in tree."""
        if index == 0:
            return 0
        return index.bit_length() - 1
    
    def _transform(self):
        """Vectorized Haar Transformation."""
        start = self.N // 2
        while start >= 1:
            indices = np.arange(start, 2 * start)
            left_children = 2 * indices
            right_children = 2 * indices + 1
            
            left_avg = self.tree['average'][left_children]
            right_avg = self.tree['average'][right_children]
            
            self.tree['average'][indices] = (left_avg + right_avg) / 2.0
            self.tree['detail'][indices] = (left_avg - right_avg) / 2.0
            start //= 2
        
        self.tree['average'][0] = self.tree['average'][1]
        self.tree['detail'][0] = 0.0
    
    def _precompute_incoming(self):
        """Precompute incoming values for all leaf-level nodes."""
        logN = int(np.log2(self.N)) + 1
        tree_avg_0 = self.tree['average'][0]
        tree_detail = self.tree['detail']
        
        for index in range(self.N // 2, self.N):
            depth = self._get_depth(index)
            paths = 1 << (depth + 1)
            incoming = compute_incoming_numba(index, paths, logN, tree_avg_0, tree_detail)
            self._incoming_cache[index] = incoming
    
    def _recurse_root(self, write_files: bool = True):
        """Start recursion from root and reconstruct signal."""
        # Solve DP for entire tree
        dp_error, dp_keep, dp_leftspace, node_details = self._recurse(1)
        
        # Calculate local budget for root decision
        depth = self._get_depth(1)
        logN = int(np.ceil(np.log2(2 * self.N)))
        localB = (1 << (logN - depth - 1)) - 1
        localB = min(localB, self.B)
        
        # Initialize source array
        source = np.zeros(self.N, dtype=np.float64)
        
        # Decide keep/discard root
        kerror = dp_error[localB - 1, 1]
        derror = np.inf
        if localB < self.N:
            derror = dp_error[localB, 0]
        
        if kerror < derror:
            # KEEP ROOT - backtrack to get coefficients
            source[0] = self.tree['average'][0]
            self.ultimate_error = kerror
            self._backtrack_coefficients(source, 1, localB - 1, 1, node_details, keep_root=True)
        else:
            # DISCARD ROOT
            self.ultimate_error = derror
            self._backtrack_coefficients(source, 1, localB, 0, node_details, keep_root=False)
        
        # Reconstruct using Numba-optimized function
        recon = reconstruct_signal_numba(source, self.N)
        
        # Scale back
        recon_scaled = recon * self.scale_factor
        
        # Store results in memory
        actual_length = min(self.original_N, self.N)
        reconstructed_values = recon_scaled[:actual_length]
        
        self.df_output = pd.DataFrame({
            'interval_end': self.original_df['interval_end'].iloc[:actual_length],
            'original_energy': self.original_df['energy'].iloc[:actual_length],
            'reconstructed_energy': reconstructed_values,
            'error': np.abs(self.original_df['energy'].iloc[:actual_length].values - reconstructed_values)
        })
        self.recon_scaled = np.array(reconstructed_values, dtype=np.float64)
        
        # Only write files if requested
        if write_files:
            self._write_output_files(recon_scaled)
    
    def _backtrack_coefficients(self, source: np.ndarray, index: int, 
                                budget: int, path: int, 
                                node_details: Dict, keep_root: bool):
        """Backtrack through DP decisions to collect coefficients."""
        if index >= self.N:
            return
        
        if index not in node_details:
            return
            
        info = node_details[index]
        dp_keep = info['keep']
        dp_leftspace = info['leftspace']
        detail_val = info['detail']
        
        if budget < 0 or budget >= dp_keep.shape[0]:
            return
        if path < 0 or path >= dp_keep.shape[1]:
            return
        
        if index >= self.N // 2:
            # Leaf node
            if budget == 1 and dp_keep[1, path]:
                source[index] = detail_val
            return
        
        # Internal node
        keep = dp_keep[budget, path]
        leftspace = dp_leftspace[budget, path]
        
        if keep:
            source[index] = detail_val
            child_path = 2 * path + 1
            if leftspace >= 0:
                self._backtrack_coefficients(source, 2 * index, leftspace, child_path, node_details, False)
                self._backtrack_coefficients(source, 2 * index + 1, budget - 1 - leftspace, child_path, node_details, False)
        else:
            child_path = 2 * path
            if leftspace >= 0:
                self._backtrack_coefficients(source, 2 * index, leftspace, child_path, node_details, False)
                self._backtrack_coefficients(source, 2 * index + 1, budget - leftspace, child_path, node_details, False)
    
    def _recurse(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Optimized DP recursion using Numba for inner loops."""
        depth = self._get_depth(index)
        paths = 1 << (depth + 1)
        
        logN = int(np.ceil(np.log2(2 * self.N)))
        localB = (1 << (logN - depth - 1)) - 1
        localB = min(localB, self.B)
        
        if index >= self.N // 2:
            localB = max(localB, 1)
        
        # Store node info for backtracking
        node_details = {}
        
        if index < self.N // 2:  # Internal nodes
            fullchildB = (1 << (logN - depth - 2)) - 1
            
            # Recurse on children
            L_error, L_keep, L_leftspace, L_details = self._recurse(2 * index)
            R_error, R_keep, R_leftspace, R_details = self._recurse(2 * index + 1)
            
            # Merge child details
            node_details.update(L_details)
            node_details.update(R_details)
            
            lenL = L_error.shape[0]
            lenR = R_error.shape[0]
            
            # Use Numba-optimized DP loop
            dp_error, dp_keep, dp_leftspace = dp_internal_loop(
                localB, paths, fullchildB,
                L_error, R_error,
                lenL, lenR, L_error.shape[1], R_error.shape[1]
            )
            
            # Store this node's info
            node_details[index] = {
                'keep': dp_keep,
                'leftspace': dp_leftspace,
                'detail': self.tree['detail'][index]
            }
            
        else:  # Leaf level
            dp_error = np.full((localB + 1, paths), np.inf, dtype=np.float64)
            dp_keep = np.zeros((localB + 1, paths), dtype=np.bool_)
            dp_leftspace = np.full((localB + 1, paths), -1, dtype=np.int32)
            
            # Get cached incoming values
            incoming = self._incoming_cache[index]
            
            detail = self.tree['detail'][index]
            left_avg = self.tree['average'][2 * index]
            right_avg = self.tree['average'][2 * index + 1]
            left_weight = self.tree['weight'][2 * index]
            right_weight = self.tree['weight'][2 * index + 1]
            
            # Use Numba-optimized error computation
            kerror_all, derror_all = compute_leaf_errors(
                incoming, detail, left_avg, right_avg,
                left_weight, right_weight, paths
            )
            
            dp_keep[1, :] = True
            dp_leftspace[1, :] = -1
            dp_error[1, :paths] = kerror_all
            
            dp_keep[0, :] = False
            dp_leftspace[0, :] = -1
            dp_error[0, :paths] = derror_all
            
            node_details[index] = {
                'keep': dp_keep,
                'leftspace': dp_leftspace,
                'detail': detail
            }
        
        return dp_error, dp_keep, dp_leftspace, node_details
    
    def _write_output_files(self, recon_scaled: np.ndarray):
        """Write output CSV files."""
        import csv
        
        with open("compressed_energy_data.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'reconstructed_energy'])
            for i in range(self.N):
                writer.writerow([i+1, f"{recon_scaled[i]:.6f}"])
        
        print(f"✓ Output written to compressed_energy_data.csv")
        
        self.df_output.to_csv("compressed_energy_with_timestamps.csv", index=False)
        print(f"✓ Output with timestamps written to compressed_energy_with_timestamps.csv")
        
        print(f"\nError Statistics:")
        print(f"  Mean error: {self.df_output['error'].mean():.6f} kWh")
        print(f"  RMS error: {np.sqrt((self.df_output['error']**2).mean()):.6f} kWh")
        print(f"  Std error: {self.df_output['error'].std():.6f} kWh")
        
        print(f"\n🔬 DIAGNOSTIC - Worst Error Points:")
        errors_array = self.df_output['error'].values
        worst_indices = np.argsort(errors_array)[-10:][::-1]

        for rank, idx in enumerate(worst_indices, 1):
            orig = self.df_output.loc[idx, 'original_energy']
            recon_val = self.df_output.loc[idx, 'reconstructed_energy']
            err = self.df_output.loc[idx, 'error']
            timestamp = self.df_output.loc[idx, 'interval_end']
            
            print(f"  {rank}. Index {idx} ({timestamp})")
            print(f"     Original: {orig:.2f} kWh, Reconstructed: {recon_val:.2f} kWh, Error: {err:.2f} kWh")
            
            if idx > 0:
                prev_val = self.df_output.loc[idx-1, 'original_energy']
                jump = abs(orig - prev_val)
                if jump > 50:
                    print(f"     ⚠️  Large jump from previous: {prev_val:.2f} → {orig:.2f} (Δ={jump:.2f})")
        
        compression_ratio = self.B / self.N * 100
        print(f"\nCompression Statistics:")
        print(f"Original data points: {self.original_N}")
        print(f"Processed data points: {self.N}")
        print(f"Deferred data points: {self.deferred_count}")
        print(f"Coefficients stored: {self.B}")
        print(f"Compression ratio: {compression_ratio:.2f}% of processed size")
    
    def recompress(self, new_B: int, write_files: bool = True):
        """Re-run DP with different budget without re-transforming."""
        self.B = new_B
        self.ultimate_error = float('inf')
        # Note: Don't clear incoming cache - it's still valid!
        self._recurse_root(write_files=write_files)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """L2 Wavelet Compression with Simple Denoising - OPTIMIZED."""
    print("="*60)
    print("ADAPTIVE L2 WAVELET COMPRESSION (OPTIMIZED + NUMBA JIT)")
    print("="*60)
    
    # Configuration
    df = pd.read_csv("test_data.csv")
    energy = df["energy"].values
    original_N = len(energy)
    N =  2 ** int(np.floor(np.log2(original_N)))
    signal = energy[:N]  # Use a segment for testing
    
    # ------------------------------------------------------------------
    # 1. Outlier handling
    # ------------------------------------------------------------------
    if USE_OUTLIER_REMOVAL:
        print("\n🔍 Detecting outliers...")
        data_clean, outlier_info = detect_and_interpolate_outliers(
            signal, low_threshold=10.0, jump_threshold=60.0
        )
        outlier_indices = np.array(outlier_info["indices"], dtype=int)
        clean_mask = np.ones(len(signal), dtype=bool)
        clean_mask[outlier_indices] = False
        original_outlier_values = signal[outlier_indices] if outlier_indices.size > 0 else np.array([])
    else:
        print("\n🔍 Skipping outlier detection...")
        data_clean = signal.copy()
        outlier_info = {"indices": np.array([], dtype=int), "values": np.array([], dtype=float)}
        outlier_indices = np.array([], dtype=int)
        clean_mask = np.ones(len(signal), dtype=bool)
        original_outlier_values = np.array([], dtype=float)

    # ------------------------------------------------------------------
    # 2. Denoise
    # ------------------------------------------------------------------
    if USE_DENOISING:
        print("\n🔧 Denoising signal...")
        clean_signal, noise_component = quick_denoise_signal(data_clean)
    else:
        print("\n🔧 Skipping denoising...")
        clean_signal = data_clean.copy()
        noise_component = np.zeros_like(clean_signal)

    dfN = df.iloc[:N].copy()
    dfN['interval_end'] = pd.to_datetime(dfN['interval_end'])
    hours = dfN['interval_end'].dt.hour.values
    hour_means = np.array([clean_signal[hours == h].mean() for h in range(24)])
    baseline = np.zeros_like(clean_signal)  # Not using baseline

    noise_power = np.mean(noise_component**2)
    signal_power = np.mean(clean_signal**2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10)) if noise_power > 0 else np.inf
    
    print(f"   ✓ SNR (clean vs noise): {snr_db:.2f} dB")
    print(f"   ✓ Noise RMS: {np.sqrt(noise_power):.2f} kWh")
    smoothness = (1 - np.sum(np.diff(clean_signal, 2)**2) / 
                     (np.sum(np.diff(signal, 2)**2) + 1e-10)) * 100
    print(f"   ✓ Smoothness improvement: {smoothness:.1f}%\n")
    
    # ------------------------------------------------------------------
    # 3. Save clean signal to temporary CSV
    # ------------------------------------------------------------------
    residual_signal = clean_signal - baseline

    df_clean = df.copy()
    df_clean['energy'] = np.pad(residual_signal, (0, original_N - N), constant_values=residual_signal[-1])
    df_clean['demand'] = df_clean['energy']
    df_clean.to_csv("temp_clean_signal.csv", index=False)
    residual_stats = compute_signal_stats(residual_signal)

    # ------------------------------------------------------------------
    # 4. L2 target
    # ------------------------------------------------------------------
    clean_for_target = clean_signal[clean_mask]
    total_energy_clean = np.sum(clean_for_target**2)
    target_error_pct = 0.3
    target_error = (target_error_pct / 100.0) * total_energy_clean
     
    coeff_pct_start = 0.13
    coeff_pct_step  = 0.001
    coeff_pct_max   = 0.30
    
    results = []
    coeff_pct = coeff_pct_start
    iteration = 1
    
    print(f"📊 Configuration:")
    print(f"   Original size: {original_N}")
    print(f"   Truncated size (N): {N}")
    print(f"   Target L2 error: {target_error:.4f} ({target_error_pct}% of {total_energy_clean:.2f})")
    print(f"   Coefficient range: {coeff_pct_start*100:.0f}% to {coeff_pct_max*100:.0f}%")
    print(f"   Step: {coeff_pct_step*100:.0f}%\n")
    
    os.makedirs("results_method1_l2", exist_ok=True)
    
    # ------------------------------------------------------------------
    # 5. Compression loop - CREATE GARO ONCE
    # ------------------------------------------------------------------
    l2_error_clean = None
    best_final_reconstructed_with_outliers = None
    
    print("🚀 Initializing optimized Garo (with Numba JIT compilation)...")
    print("   (First run includes JIT compilation overhead)\n")
    
    # Create Garo ONCE - transform happens here
    gk = GaroOptimized(B=max(1, int(coeff_pct_start * N)), scale_factor=1.0, 
                       csv_file="temp_clean_signal.csv", verbose=False)
    
    while coeff_pct <= coeff_pct_max and iteration <= 100:
        B = max(1, int(coeff_pct * N))
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}: B={B} ({coeff_pct*100:.1f}% of N={N})")
        print(f"{'='*60}")
        
        try:
            # Reuse existing Garo - only run DP, skip file writes until final
            is_final = (coeff_pct + coeff_pct_step > coeff_pct_max)
            if iteration > 1:
                gk.recompress(B, write_files=is_final)
            
            if not os.path.exists("compressed_energy_with_timestamps.csv"):
                print(f"❌ ERROR: Compression failed (output file missing)")
                break
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            break
        
        df_out = gk.df_output
        
        # Reconstructed RESIDUAL
        residual_reconstructed = df_out["reconstructed_energy"].values[:N]
        clean_reconstructed = baseline + residual_reconstructed
        
        # Stopping error on CLEAN
        err_clean = clean_signal - clean_reconstructed
        err_clean_non_outliers = err_clean[clean_mask]
        l2_error_clean = np.sum(err_clean_non_outliers**2)
        error_pct_clean = (l2_error_clean / (np.sum(clean_for_target**2) + 1e-10)) * 100.0

        # Add noise + restore outliers for metrics
        if ADD_NOISE_BACK_FOR_EVAL:
            final_reconstructed = clean_reconstructed + noise_component
        else:
            final_reconstructed = clean_reconstructed.copy()

        final_reconstructed_with_outliers = final_reconstructed.copy()
        if outlier_indices.size > 0:
            final_reconstructed_with_outliers[outlier_indices] = original_outlier_values

        original = signal
        errors = original - final_reconstructed_with_outliers
        
        l2_error_original = np.sum(errors**2)
        error_pct_original = (l2_error_original / (np.sum(original**2) + 1e-10)) * 100.0
        
        signal_power_final = np.mean(original**2) + 1e-10
        noise_power_final = np.mean(errors**2) + 1e-10
        snr_db_final = 10 * np.log10(signal_power_final / noise_power_final)
        
        max_sig = np.max(np.abs(original)) + 1e-10
        psnr_db = 20 * np.log10(max_sig / np.sqrt(noise_power_final))
        
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((original - original.mean())**2) + 1e-10
        r2 = 1 - ss_res / ss_tot
        
        ss_res_residual_vs_orig = np.sum((original - residual_signal)**2)
        r2_residual_vs_original = 1 - ss_res_residual_vs_orig / ss_tot
        
        res_errors = residual_signal - residual_reconstructed
        ss_res_resid = np.sum(res_errors**2)
        ss_tot_resid = np.sum((residual_signal - residual_signal.mean())**2) + 1e-10
        r2_residual_recon = 1 - ss_res_resid / ss_tot_resid

        rmse = np.sqrt(noise_power_final)
        nrmse = rmse / original.mean()
        mae = np.mean(np.abs(errors))
        
        results.append({
            "iteration": iteration,
            "coefficients": B,
            "coeff_pct": coeff_pct * 100.0,
            "compression_pct": B / N * 100.0,
            "l2_error_clean": l2_error_clean,
            "l2_error": l2_error_original,
            "mean_abs_error": mae,
            "rmse": rmse,
            "nrmse": nrmse,
            "snr_db": float(snr_db_final),
            "psnr_db": float(psnr_db),
            "r_squared": float(r2),
            "r_squared_residual_vs_original": float(r2_residual_vs_original),
            "r_squared_residual_reconstructed": float(r2_residual_recon)
        })
        
        best_final_reconstructed_with_outliers = final_reconstructed_with_outliers.copy()
        
        print(f"   L2 (clean): {l2_error_clean:.4f} ({error_pct_clean:.2f}%)")
        print(f"   L2 (original): {l2_error_original:.4f} ({error_pct_original:.2f}%)")
        print(f"   RMSE: {rmse:.4f} kWh")
        print(f"   Target: {target_error:.4f} ({target_error_pct}%)")
        print(f"   Status: {'✅ TARGET MET' if l2_error_clean <= target_error else '❌ TARGET NOT MET'}")
        
        if l2_error_clean <= target_error:
            print(f"\n{'='*60}")
            print(f"✅ SUCCESS! Target reached at coeff_pct = {coeff_pct*100:.1f}%")
            print(f"{'='*60}")
            # Write final files
            if not is_final:
                gk.recompress(B, write_files=True)
            break
        
        coeff_pct += coeff_pct_step
        iteration += 1
    
    if (coeff_pct > coeff_pct_max or iteration > 100) and (l2_error_clean is not None) and (l2_error_clean > target_error):
        print(f"\n{'='*60}")
        print(f"⚠️  Reached coefficient limit.")
        print(f"    Final L2 error: {l2_error_clean:.4f} (target: {target_error:.4f})")
        print(f"{'='*60}")
    
    # ------------------------------------------------------------------
    # 6. Add signal stats
    # ------------------------------------------------------------------
    signal_stats = compute_signal_stats(signal)
    for r in results:
        r.update({
            'signal_entropy': signal_stats['entropy'],
            'signal_mean': signal_stats['mean'],
            'signal_std': signal_stats['std'],
            'signal_min': signal_stats['data_min'],
            'signal_max': signal_stats['data_max'],
            'signal_range': signal_stats['data_range'],
            'signal_coeff_var': signal_stats['coeff_var'],
            'signal_median': signal_stats['median'],
            'signal_zero_fraction': signal_stats['zero_fraction'],
            'signal_gini_coefficient': signal_stats['gini_coefficient'],
        })
        r.update({
            'residual_entropy': residual_stats['entropy'],
            'residual_mean': residual_stats['mean'],
            'residual_std': residual_stats['std'],
            'residual_min': residual_stats['data_min'],
            'residual_max': residual_stats['data_max'],
            'residual_range': residual_stats['data_range'],
            'residual_coeff_var': residual_stats['coeff_var'],
            'residual_median': residual_stats['median'],
            'residual_zero_fraction': residual_stats['zero_fraction'],
            'residual_gini_coefficient': residual_stats['gini_coefficient'],
        })
    
    # 7. Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results_method1_l2/garo_fullsignal_metrics.csv", index=False)
    
    # 8. Save final reconstruction
    if best_final_reconstructed_with_outliers is not None:
        df_recon = df.iloc[:N].copy()
        df_recon["original_energy"] = signal
        df_recon["reconstructed_energy"] = best_final_reconstructed_with_outliers
        df_recon["error"] = df_recon["original_energy"] - df_recon["reconstructed_energy"]
        
        output_dir = "results_method1_l2"
        os.makedirs(output_dir, exist_ok=True)
        df_recon.to_csv(os.path.join(output_dir, "reconstructed_method1.csv"), index=False)
    else:
        print("⚠️ No reconstruction stored.")
    
    print(f"\n{'='*60}")
    print(f"💾 Results saved to results_method1_l2/")
    print(f"{'='*60}")
    
    if os.path.exists("temp_clean_signal.csv"):
        os.remove("temp_clean_signal.csv")
    
    return results_df


if __name__ == "__main__":
    start_time = time.time()
    main_df = main()
    elapsed_time = time.time() - start_time

    output_dir = Path("results_method1_l2")
    output_dir.mkdir(exist_ok=True)

    timing_file = output_dir / "timing.txt"
    with open(timing_file, 'w') as f:
        f.write(f"{elapsed_time:.4f}")

    print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds")
    print(f"💾 Timing saved to: {timing_file}")