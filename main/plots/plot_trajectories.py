import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchcde

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12, 'axes.titlesize': 14, 
    'axes.labelsize': 13, 'lines.linewidth': 2, 'lines.markersize': 5,
    'legend.fontsize': 11, 'figure.figsize': (15, 9)
})

def get_gp_interpolation(t, x, t_dense, length_scale, noise_std):
    t_tensor = torch.tensor(t, dtype=torch.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    t_dense_tensor = torch.tensor(t_dense, dtype=torch.float32)
    diff_sq = (t_tensor.unsqueeze(1) - t_tensor.unsqueeze(0))**2
    K_tt = torch.exp(-0.5 * diff_sq / length_scale**2) + (noise_std**2) * torch.eye(len(t_tensor))
    alpha = torch.linalg.solve(K_tt, x_tensor)
    diff_star = t_dense_tensor.unsqueeze(1) - t_tensor.unsqueeze(0)
    K_star = torch.exp(-0.5 * diff_star**2 / length_scale**2)
    return torch.matmul(K_star, alpha).numpy()

def main():
    path = 'data/preprocessed_nz_filled.csv'
    if not os.path.exists(path): return
    df = pd.read_csv(path, low_memory=False)
    
    feature = 'GR'
    seq_len = 100
    target_well, start_idx = df['WELLNAME'].unique()[0], 1000
    
    for well in df['WELLNAME'].unique():
        well_data = df[df['WELLNAME'] == well]
        if len(well_data) > seq_len + 1000:
            for start in range(1000, len(well_data) - seq_len, 200):
                y_chunk = well_data[feature].iloc[start:start+seq_len].values
                if np.nanstd(y_chunk) > 0.5 and not np.isnan(y_chunk).all():
                    target_well, start_idx = well, start
                    break
        if target_well != df['WELLNAME'].unique()[0]: break

    y_filled_norm = df[df['WELLNAME'] == target_well].iloc[start_idx:start_idx+seq_len][feature].values
    depth = df[df['WELLNAME'] == target_well].iloc[start_idx:start_idx+seq_len]['DEPT'].values
    
    np.random.seed(42)
    y_sparse_norm = y_filled_norm.copy()
    y_sparse_norm[np.random.rand(seq_len) < 0.25] = np.nan

    os.makedirs("pictures", exist_ok=True)

    # 1. Preprocessing
    fig1, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
    fig1.suptitle(f"Well-Log Preprocessing: Handling Missing Observations ({feature})", fontweight='bold', fontsize=16)
    ax1.plot(depth, y_sparse_norm, 'ko-', alpha=0.5, label='Raw Observations', markersize=4)
    ax1.set_title("Before: Sparse Data (NaNs present)")
    ax1.set_xlabel("Depth (m)")
    ax1.set_ylabel(f"Normalized {feature}")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.plot(depth, y_filled_norm, 'bo-', alpha=0.7, label='Processed & Filled', markersize=4)
    ax2.set_title("After: Processed Data for Neural CDE")
    ax2.set_xlabel("Depth (m)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    plt.tight_layout()
    fig1.savefig('pictures/preprocessing_comparison.pdf', dpi=300, bbox_inches='tight')
    fig1.savefig('pictures/preprocessing_comparison.png', dpi=300, bbox_inches='tight')

    # 2. Interpolation
    t_discrete = np.linspace(0, seq_len - 1, seq_len)
    t_dense = np.linspace(0, seq_len - 1, 1000) 
    X_tensor = torch.tensor(y_filled_norm, dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
    
    X_linear = torchcde.LinearInterpolation(torchcde.linear_interpolation_coeffs(X_tensor)).evaluate(torch.tensor(t_dense, dtype=torch.float32)).squeeze().numpy()
    X_cubic = torchcde.CubicSpline(torchcde.natural_cubic_coeffs(X_tensor)).evaluate(torch.tensor(t_dense, dtype=torch.float32)).squeeze().numpy()
    X_gp_narrow = get_gp_interpolation(t_discrete, y_filled_norm, t_dense, length_scale=4.5, noise_std=0.01)
    X_gp_wide = get_gp_interpolation(t_discrete, y_filled_norm, t_dense, length_scale=10.0, noise_std=0.1)

    fig2, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    fig2.suptitle(f"Continuous Path Construction for ODE Solver (Feature: {feature})", fontweight='bold', fontsize=18)

    plots = [
        (axes[0, 0], X_linear, 'Linear Interpolation', 'blue', 'Discontinuous derivatives cause solver stops.'),
        (axes[0, 1], X_cubic, 'Natural Cubic Spline', 'red', 'Overshoots on noise causing extreme stiffness.'),
        (axes[1, 0], X_gp_narrow, 'GP Smoothing (Narrow Kernel)', 'green', 'Filters micro-noise, retains local shape.'),
        (axes[1, 1], X_gp_wide, 'GP Smoothing (Wide Kernel)', 'orange', 'Extracts macro-trends, highly efficient solver.')
    ]

    for ax, path, title, color, subtitle in plots:
        ax.plot(t_discrete, y_filled_norm, 'ko', alpha=0.3, markersize=5, label='Observations')
        ax.plot(t_dense, path, color=color, linewidth=2, label='Continuous Path $X(t)$')
        ax.set_title(title, fontweight='bold')
        ax.text(0.02, 0.95, subtitle, transform=ax.transAxes, fontsize=10.5, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.grid(True, linestyle='--', alpha=0.5)

    fig2.text(0.53, 0.07, "Time Step (Index within Window)", ha='center', va='center', fontsize=14, fontweight='bold')
    fig2.text(0.03, 0.5, f"Normalized {feature}", ha='center', va='center', rotation='vertical', fontsize=14, fontweight='bold')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower center', ncol=2, fontsize=12, frameon=True, bbox_to_anchor=(0.53, 0.01))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.15, hspace=0.18, wspace=0.08)

    fig2.savefig('pictures/interpolation_comparison.pdf', dpi=300, bbox_inches='tight')
    fig2.savefig('pictures/interpolation_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved pictures/preprocessing and interpolation comparisons!")

if __name__ == "__main__":
    main()