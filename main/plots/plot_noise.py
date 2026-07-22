import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 13, 'axes.labelsize': 14, 
    'axes.titlesize': 16, 'legend.fontsize': 13
})

def plot_noise_robustness(csv_path="pictures/results_summary_sparse.csv"):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Run results.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Ищем лучших представителей
    idx = df.groupby('Model')['ARI_Class'].idxmax()
    df_best = df.loc[idx]
    
    models_to_plot = {
        'Cubic CDE': ('Cubic Spline', 'red', 'X', '-'),
        'Vanilla GRU (No Mask)': ('Vanilla GRU', 'gray', 's', '--'),
        'GP CDE': ('GP CDE', 'orange', '^', '-'),
        'MV-CDE (Q-Former) (Learnable)': ('MV-CDE (Ours)', 'green', 'o', '-')
    }
    
    noise_levels = [0.0, 0.1, 0.3, 0.5]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Robustness to Additive Noise (Petroleum Sparse)', fontweight='bold', y=0.98)

    lines = []
    labels = []
    
    for model_name, (display_name, color, marker, ls) in models_to_plot.items():
        row = df_best[df_best['Model'] == model_name]
        if row.empty: continue
        row = row.iloc[0]
        
        aris = [row['ARI_Class'], row['Noise_0.1_ARI'], row['Noise_0.3_ARI'], row['Noise_0.5_ARI']]
        line, = ax1.plot(noise_levels, aris, color=color, marker=marker, markersize=9, linewidth=2.5, linestyle=ls)
        lines.append(line)
        labels.append(display_name)
        
        if display_name == 'Vanilla GRU': nfes = [0, 0, 0, 0]
        else: nfes = [row['Test_NFE'], row['Noise_0.1_NFE'], row['Noise_0.3_NFE'], row['Noise_0.5_NFE']]
        ax2.plot(noise_levels, nfes, color=color, marker=marker, markersize=9, linewidth=2.5, linestyle=ls)

    ax1.set_title('Classification Performance (ARI) ↓', pad=10)
    ax1.set_xlabel('Gaussian Noise Level (Std)')
    ax1.set_ylabel('ARI (Class)')
    ax1.set_xticks(noise_levels)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.set_title('Computational Cost (Average NFE) ↑', pad=10)
    ax2.set_xlabel('Gaussian Noise Level (Std)')
    ax2.set_ylabel('Test NFE (Log Scale)')
    ax2.set_xticks(noise_levels)
    ax2.set_yscale('symlog') 
    ax2.grid(True, linestyle='--', alpha=0.6)

    # ОБЩАЯ ЛЕГЕНДА
    fig.legend(lines, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), frameon=True, edgecolor='black')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.15)
    os.makedirs('pictures', exist_ok=True)
    plt.savefig('pictures/noise_robustness.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('pictures/noise_robustness.png', dpi=300, bbox_inches='tight')
    print("Saved pictures/noise_robustness.pdf/.png")

if __name__ == "__main__":
    plot_noise_robustness()