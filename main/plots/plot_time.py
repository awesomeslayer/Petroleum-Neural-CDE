import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12, 'axes.labelsize': 14, 
    'axes.titlesize': 15, 'legend.fontsize': 12
})

def plot_time_analysis(csv_path="pictures/results_summary_sparse.csv"):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    
    # Берем лучших представителей
    idx = df.groupby('Model')['ARI_Class'].idxmax()
    df = df.loc[idx]
    
    name_map = {
        'Vanilla GRU (No Mask)': 'Vanilla GRU',
        'Linear CDE': 'Linear CDE', 'Cubic CDE': 'Cubic CDE',
        'Kernel CDE': 'Kernel CDE', 'GP CDE': 'GP CDE',
        'MVC-CDE (Conv) (Learnable)': 'MVC-CDE (Ours)',
        'MV-CDE (Q-Former) (Learnable)': 'MV-CDE (Ours)'
    }
    df['Display_Name'] = df['Model'].map(name_map)
    df = df.dropna(subset=['Display_Name']).set_index('Display_Name')
    
    order = ['Vanilla GRU', 'Linear CDE', 'Cubic CDE', 'Kernel CDE', 'GP CDE', 'MVC-CDE (Ours)', 'MV-CDE (Ours)']
    df = df.reindex([o for o in order if o in df.index])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Computational Time Analysis (Petroleum_NZ_Sparse)", fontsize=18, y=0.98, fontweight='bold')

    def draw_bars(ax, data, color, title, ylabel):
        bars = ax.bar(df.index, data, color=color, edgecolor='black', width=0.6)
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        for rect in bars:
            h = rect.get_height()
            if h > 0.01: ax.text(rect.get_x() + rect.get_width()/2., h, f'{h:.2f}s', ha='center', va='bottom', fontsize=11)
        ax.tick_params(axis='x', rotation=30)
        for tick in ax.get_xticklabels(): tick.set_ha('right')

    draw_bars(axes[0], df['Train_Time'], '#1f77b4', "Pure Train Time (Model Update)", "Time (seconds)")
    draw_bars(axes[1], df['Test_Time'], '#2ca02c', "Pure Test Time (Inference)", "")
    draw_bars(axes[2], df['Test_Fit_Time'], '#ff7f0e', "Overhead: Path Fit Time (Test)", "")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.25, wspace=0.15) 
    os.makedirs('pictures', exist_ok=True)
    plt.savefig('pictures/time_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('pictures/time_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved pictures/time_analysis.pdf/.png")

if __name__ == "__main__":
    plot_time_analysis()