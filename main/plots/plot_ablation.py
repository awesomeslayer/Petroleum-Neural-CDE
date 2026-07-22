import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 13, 'axes.labelsize': 14, 
    'axes.titlesize': 16, 'legend.fontsize': 12
})

def plot_num_heads_ablation(csv_path="pictures/results_summary_sparse.csv"):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Оставляем только MV и MVC
    df_mv = df[df['Model'].str.contains('MV-CDE|MVC-CDE')].copy()
    
    # Исключаем гетерогенные масштабы (где есть 200, 400 и тд), оставляем только чистые 100 и 300
    df_mv = df_mv[~df_mv['Kernel_Scale'].str.contains('200.0')]
    
    # Разделяем данные на 4 линии
    qf_100 = df_mv[(df_mv['Model'].str.contains('Q-Former')) & (df_mv['Kernel_Scale'].str.contains('100.0'))].sort_values('Num_Heads')
    qf_300 = df_mv[(df_mv['Model'].str.contains('Q-Former')) & (df_mv['Kernel_Scale'].str.contains('300.0'))].sort_values('Num_Heads')
    
    cv_100 = df_mv[(df_mv['Model'].str.contains('Conv')) & (df_mv['Kernel_Scale'].str.contains('100.0'))].sort_values('Num_Heads')
    cv_300 = df_mv[(df_mv['Model'].str.contains('Conv')) & (df_mv['Kernel_Scale'].str.contains('300.0'))].sort_values('Num_Heads')

    # Отрисовка
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle('Ablation: Impact of Head Count on Accuracy', fontweight='bold', y=0.95)

    # ИСПОЛЬЗУЕМ СТАНДАРТНЫЕ ЦВЕТА MATPLOTLIB (tab: палитра)
    ax.plot(qf_100['Num_Heads'], qf_100['ARI_Class'], marker='o', linewidth=2.5, color='tab:blue', linestyle='-', label='MV-CDE (h=100)')
    ax.plot(qf_300['Num_Heads'], qf_300['ARI_Class'], marker='s', linewidth=2.5, color='tab:orange', linestyle='--', label='MV-CDE (h=300)')
    
    ax.plot(cv_100['Num_Heads'], cv_100['ARI_Class'], marker='^', linewidth=2.5, color='tab:green', linestyle='-', label='MVC-CDE (h=100)')
    ax.plot(cv_300['Num_Heads'], cv_300['ARI_Class'], marker='D', linewidth=2.5, color='tab:red', linestyle='--', label='MVC-CDE (h=300)')

    ax.set_xlabel('Number of Attention Heads')
    ax.set_ylabel('Classification Performance (ARI Class)')
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.legend(loc='lower left', frameon=True, edgecolor='black')

    plt.tight_layout()
    os.makedirs('pictures', exist_ok=True)
    plt.savefig('pictures/ablation_heads.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('pictures/ablation_heads.png', dpi=300, bbox_inches='tight')
    print("Saved pictures/ablation_heads.pdf")

if __name__ == "__main__":
    plot_num_heads_ablation()