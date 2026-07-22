import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import matplotlib.gridspec as gridspec
import os

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 13, 'axes.labelsize': 14,
    'axes.titlesize': 16, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
})

def find_latest_pkl(root_dir, target_substring):
    path = Path(root_dir) / "Petroleum_NZ_Sparse" / "attention_maps"
    if not path.exists(): return None
    # Ищем файлы, содержащие нужную подстроку
    candidates = [p for p in path.glob("*.pkl") if target_substring.lower() in p.name.lower()]
    if not candidates: return None
    # Берем самый свежий
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]

def plot_single_variant(X_seq, attn1, feature_name, sample_idx, out_dir, name1):
    # Уменьшили высоту до 6, так как теперь 2 графика вместо 3
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"Attention Map (Feature: {feature_name} | Sample: {sample_idx})", fontsize=18, fontweight='bold', y=0.96)
    
    # Сетка из 2 строк
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.15)
    time_steps = np.arange(len(X_seq))

    # --- 1. Raw Data ---
    ax_traj = plt.subplot(gs[0])
    ax_traj.plot(time_steps, X_seq, color='#2c3e50', linewidth=2, marker='o', markersize=4)
    ax_traj.set_xlim(-0.5, len(X_seq) - 0.5)
    ax_traj.grid(True, linestyle='--', alpha=0.5)
    plt.setp(ax_traj.get_xticklabels(), visible=False)
    ax_traj.set_ylabel(f"Normalized\n{feature_name}", fontweight='bold')

    # --- 2. MV-CDE (Q-Former) ---
    ax_m1 = plt.subplot(gs[1], sharex=ax_traj)
    if attn1 is not None:
        colors = sns.color_palette("Set1", n_colors=attn1.shape[0])
        for h in range(attn1.shape[0]):
            ax_m1.fill_between(time_steps, attn1[h, :], step='mid', alpha=0.3, color=colors[h])
            ax_m1.plot(time_steps, attn1[h, :], drawstyle='steps-mid', color=colors[h], linewidth=2, label=f"Head {h+1}")
        ax_m1.set_ylim(0, np.max(attn1) * 1.15)
    ax_m1.grid(True, linestyle='--', alpha=0.4)
    ax_m1.set_ylabel(f"{name1}\nAttention", fontweight='bold')
    # Перенесли подпись оси X сюда, так как это теперь нижний график
    ax_m1.set_xlabel("Depth Step (Index within Window)", fontweight='bold')
    ax_m1.legend(loc='upper right', fontsize=10)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.15)
    
    # === СОХРАНЕНИЕ В PDF И PNG ===
    out_path_pdf = out_dir / f"attention_{feature_name}_sample{sample_idx}.pdf"
    out_path_png = out_dir / f"attention_{feature_name}_sample{sample_idx}.png"
    plt.savefig(out_path_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    root = "experiment_petroleum_sparse"
    
    # Ищем наш лучший 3-головый Q-Former
    MODEL_1_TARGET = "qformer_gp_ls-[200.0, 300.0, 400.0]"
    
    p1 = find_latest_pkl(root, MODEL_1_TARGET)
    
    if not p1:
        # Если с точным именем не найдено, берем просто последний qformer
        p1 = find_latest_pkl(root, "qformer")
        if not p1:
            print("PKL files not found! Please check the names.")
            return

    print(f"Loading Model from: {p1.name}")

    with open(p1, 'rb') as f: d1 = pickle.load(f)

    X_all = d1['fixed_Xs'][:5].numpy()
    attns1 = d1['fixed_attns'][:5].numpy()
    
    features_names = ["GR", "DENS", "DTC", "DRHO", "RESS", "RESD", "RESM"]
    out_dir = Path('pictures/attention_variants')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating cherry-pick images (.pdf and .png) in {out_dir}...")
    for s_idx in range(X_all.shape[0]):
        for f_idx in range(X_all.shape[2]):
            f_name = features_names[f_idx] if f_idx < len(features_names) else f"Feat{f_idx}"
            plot_single_variant(X_all[s_idx, :, f_idx], attns1[s_idx], 
                                f_name, s_idx, out_dir, "MV-CDE")
            
    print("Done! All PDFs and PNGs are saved.")

if __name__ == "__main__":
    main()