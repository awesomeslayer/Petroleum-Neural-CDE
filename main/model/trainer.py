import torch
import torch.nn.functional as F
import torchcde
import time
import scipy.interpolate as spi
import numpy as np
import copy
import os
import pickle
import json
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

def compute_attention_diversity(attn_weights, return_per_sample=False):
    B, H, L = attn_weights.shape
    if H <= 1:
        if return_per_sample: return torch.zeros(B), torch.zeros(B)
        return 0.0, 0.0

    # 1. Cosine Distance
    attn_norm = F.normalize(attn_weights, p=2, dim=-1)
    sim_matrix = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
    
    triu_idx = torch.triu_indices(H, H, offset=1)
    
    # Считаем попарную косинусную схожесть для каждого семпла
    sample_cosine_sim = sim_matrix[:, triu_idx[0], triu_idx[1]]
    
    # Нам нужно МАКСИМАЛЬНОЕ расстояние. 
    # Distance = 1 - Similarity. Макс. дистанция = 1 - Мин. схожесть.
    sample_max_cosine_dist = 1.0 - sample_cosine_sim.min(dim=-1)[0]

    # 2. Jensen-Shannon Divergence (JSD)
    P = attn_weights.unsqueeze(2) 
    Q = attn_weights.unsqueeze(1) 
    M = 0.5 * (P + Q)

    P_safe = torch.clamp(P, 1e-9, 1.0)
    Q_safe = torch.clamp(Q, 1e-9, 1.0)
    M_safe = torch.clamp(M, 1e-9, 1.0)

    kl_pm = torch.sum(P_safe * (torch.log(P_safe) - torch.log(M_safe)), dim=-1)
    kl_qm = torch.sum(Q_safe * (torch.log(Q_safe) - torch.log(M_safe)), dim=-1)
    jsd_matrix = 0.5 * kl_pm + 0.5 * kl_qm 

    # Берем максимальную дивергенцию между любыми двумя головами для семпла
    sample_max_jsd = jsd_matrix[:, triu_idx[0], triu_idx[1]].max(dim=-1)[0]

    if return_per_sample:
        return sample_max_cosine_dist.cpu(), sample_max_jsd.cpu()
    
    return sample_max_cosine_dist.mean().item(), sample_max_jsd.mean().item()

def compute_diversity_loss(attn_weights):
    """
    Loss-функция для регуляризации. Штрафует сеть, если головы смотрят на одно и то же.
    Минимизирует среднюю косинусную схожесть между всеми парами голов.
    """
    B, H, L = attn_weights.shape
    if H <= 1:
        return torch.tensor(0.0, device=attn_weights.device)
        
    attn_norm = F.normalize(attn_weights, p=2, dim=-1)
    sim_matrix = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
    
    triu_idx = torch.triu_indices(H, H, offset=1)
    # Берем среднюю схожесть по всем парам в батче
    mean_sim = sim_matrix[:, triu_idx[0], triu_idx[1]].mean()
    
    return mean_sim # Возвращаем схожесть (оптимизатор будет ее минимизировать)

def get_nfe(model):
    if hasattr(model, 'func') and hasattr(model.func, 'nfe'):
        return model.func.nfe
    elif hasattr(model, 'cde_func') and hasattr(model.cde_func, 'nfe'):
        return model.cde_func.nfe
    return 0

def reset_nfe(model):
    if hasattr(model, 'func'):
        model.func.nfe = 0
    if hasattr(model, 'cde_func'):
        model.cde_func.nfe = 0

def reset_fit_timer(model):
    if hasattr(model, 'reset_fit_timer'):
        model.reset_fit_timer()

def get_fit_time(model):
    return getattr(model, 'fit_time_accum', 0.0)

def get_tensorboard_path(cfg, exp_name, interp_type, kernel_params, seed, dr):
    base_dir = os.path.join(cfg['runs_dir'], cfg['dataset_name'])
    tol = kernel_params.get('tol', 'unknown')
    tol_str = f"tol-{tol}_dr-{dr}"  
    
    if interp_type in ['linear', 'cubic']:
        sub_dir = "baseline"
    elif interp_type == "log_ncde":  
        sub_dir = "Log-NCDE"
    elif interp_type == "odernn":
        sub_dir = "ODE-RNN"
    elif interp_type == "grud":
        sub_dir = "GRU-D"
    else:
        k_name = kernel_params.get('kernel', interp_type)
        sub_dir = f"{interp_type}/{k_name}"

    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_name = f"{exp_name}_s{seed}_{timestamp}"
    return os.path.join(base_dir, sub_dir, tol_str, run_name)


def get_static_inputs(X, interpolation_type, kernel_params=None):
    if kernel_params is None: kernel_params = {}
    start_time = time.perf_counter()
    
    if interpolation_type == "smoothing_spline":
        s_factor = kernel_params.get('smoothing_factor', 0.5)
        
        X_smooth = torch.zeros_like(X)
        batch_size, seq_len, channels = X.shape
        has_time_channel = (X[0, :, 0].diff() > 0).all().item() 
        t_fallback = np.arange(seq_len)
        
        for b in range(batch_size):
            t_raw = X[b, :, 0].cpu().numpy() if has_time_channel else t_fallback
            for c in range(channels):
                if has_time_channel and c == 0:
                    X_smooth[b, :, c] = X[b, :, c]
                    continue
                
                valid_mask = ~torch.isnan(X[b, :, c]).cpu().numpy()
                t_valid = t_raw[valid_mask]
                x_valid = X[b, :, c].cpu().numpy()[valid_mask]
                
               
                t_valid, unique_idx = np.unique(t_valid, return_index=True)
                x_valid = x_valid[unique_idx]
                
                if len(x_valid) > 3:
                    
                    s_val = len(x_valid) * s_factor
                    spl = spi.UnivariateSpline(t_valid, x_valid, s=s_val) 
                    X_smooth[b, :, c] = torch.tensor(spl(t_raw), dtype=X.dtype, device=X.device)
                else:
                    X_smooth[b, :, c] = 0.0

     
        coeffs = torchcde.natural_cubic_coeffs(X_smooth)
        
    elif interpolation_type == "cubic":
        coeffs = torchcde.natural_cubic_coeffs(X)
    elif interpolation_type in ["linear", "log_ncde"]: 
        coeffs = torchcde.linear_interpolation_coeffs(X)
    else:
        coeffs = X
        
    fit_time = time.perf_counter() - start_time
    return coeffs, fit_time

def evaluate_loop(model, loader, device):
    model.eval()
    reset_fit_timer(model)
    reset_nfe(model)
    
    total_loss = 0
    correct = 0
    total = 0
    total_nfe = 0
    batch_count = 0
    
    start_wall = time.perf_counter()
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            reset_nfe(model)
            
            pred_y = model(batch_X)
            loss = F.cross_entropy(pred_y, batch_y)
            
            total_loss += loss.item()
            _, predicted = torch.max(pred_y, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            total_nfe += get_nfe(model)
            batch_count += 1
            
    end_wall = time.perf_counter()
    
    avg_loss = total_loss / len(loader)
    acc = correct / total
    avg_nfe = total_nfe / batch_count if batch_count > 0 else 0
    
    wall_clock = end_wall - start_wall
    dynamic_fit = get_fit_time(model)
    pure_time = wall_clock - dynamic_fit
    
    times = {
        'wall': wall_clock,
        'pure': pure_time,
        'dynamic_fit': dynamic_fit
    }
    
    return acc, avg_loss, avg_nfe, times

def log_banner(logger, text, char="=", length=65):
    logger.info("")
    logger.info(char * length)
    logger.info(f" {text}".center(length))
    logger.info(char * length)

def run_experiment(model, data, cfg, logger, exp_name, interp_type, kernel_params=None, dr=0.0):
    if kernel_params is None: kernel_params = {}
    seed = cfg.get('seed', 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

   
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
   
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    writer_path = get_tensorboard_path(cfg, exp_name, interp_type, kernel_params, seed, dr)
    writer = SummaryWriter(log_dir=writer_path)
    
    log_banner(logger, f"START EXPERIMENT: {exp_name}")
    logger.info(f"{'Type':<15} : {interp_type.upper()}")
    logger.info(f"{'Seed':<15} : {seed}")
    logger.info(f"{'Device':<15} : {device}")
    logger.info(f"{'Params':<15} : {num_params:,}") 
    logger.info(f"{'TensorBoard':<15} : {writer_path}")
    if kernel_params:
        logger.info(f"{'Kernel Params':<15} : {json.dumps(kernel_params, default=str)}")
    logger.info("-" * 65)

    train_X_raw, val_X_raw, test_X_raw, train_y, val_y, test_y, _ = data
    
    logger.info(">>> Phase 1: Static Fit (Interpolation Pre-calc)")
    train_coeffs, t_static_train = get_static_inputs(train_X_raw, interp_type, kernel_params)
    val_coeffs, t_static_val     = get_static_inputs(val_X_raw, interp_type, kernel_params)
    test_coeffs_clean, t_static_test = get_static_inputs(test_X_raw, interp_type, kernel_params)

    logger.info(f"    Train Fit Time : {t_static_train:.4f}s")
    logger.info(f"    Val Fit Time   : {t_static_val:.4f}s")
    logger.info(f"    Test Fit Time  : {t_static_test:.4f}s")

    batch_size = cfg['batch_size']
    train_loader = DataLoader(TensorDataset(train_coeffs, train_y), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_coeffs, val_y), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(test_coeffs_clean, test_y), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg["weight_decay"])

    if cfg.get('inspect_interpolation', False):
            log_banner(logger, "Interpolation Inspection", char="-")
            
            sample_coeffs = train_coeffs[:1].to(device)
            
            if hasattr(model, 'make_interpolation'):
                res = model.inspect_interpolation(sample_coeffs)
                if res:
                    t_points, derivs = res
                    
                    add_time_bool = (str(cfg.get('add_time', 'yes')).lower() == 'yes')
                    
                    time_idx = 0 if add_time_bool else -1
                    feat_idx = 1 if add_time_bool else 0 
                    
                    logger.info(f"{'Metric':<25} | {'Start (5%)':<12} | {'Mid (50%)':<12} | {'End (95%)':<12}")
                    logger.info("-" * 75)
                    
                    if add_time_bool:
                        d_t_start = derivs[0][time_idx].item()
                        d_t_mid   = derivs[1][time_idx].item()
                        d_t_end   = derivs[2][time_idx].item()
                        logger.info(f"{'Time deriv (dt/ds)':<25} | {d_t_start:<12.4f} | {d_t_mid:<12.4f} | {d_t_end:<12.4f}")
                    else:
                        logger.info(f"{'Time deriv (dt/ds)':<25} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12}")

                    d_x_start = derivs[0][feat_idx].item()
                    d_x_mid   = derivs[1][feat_idx].item()
                    d_x_end   = derivs[2][feat_idx].item()
                    
                    logger.info(f"{'Feat deriv (dX/ds)':<25} | {d_x_start:<12.4f} | {d_x_mid:<12.4f} | {d_x_end:<12.4f}")
                    
                    logger.info("-" * 75)
                    logger.info("NOTE: 'Feat deriv' should be roughly constant across different time_scalings,")
                    logger.info("      while 'Time deriv' should change proportionally to scaling factor.")
            else:
                logger.info(f"Skipping: Model '{type(model).__name__}' does not use continuous interpolation.")
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    
    train_stats = {
        'pure_time': 0.0,
        'dynamic_fit': 0.0,
        'avg_nfe_sum': 0.0
    }

    log_banner(logger, f"Phase 2: Training ({cfg['num_epochs']} Epochs)", char="-")
    
    header_fmt = "{:<6} | {:<10} | {:<10} | {:<10} | {:<6} | {:<8}"
    row_fmt    = "{:<6} | {:<10} | {:<10} | {:<10} | {:<6} | {:<8}"
    logger.info(header_fmt.format("Epoch", "Train Acc", "Val Acc", "Trn Loss", "NFE", "Fit(s)"))
    logger.info("-" * 65)
    
    for epoch in range(1, cfg['num_epochs'] + 1):
        epoch_start = time.perf_counter()
        
        model.train()
        reset_fit_timer(model)
        
        train_loss = 0.0
        correct = 0
        total = 0
        epoch_nfe_accum = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            reset_nfe(model) 
            
            optimizer.zero_grad()
            pred_y = model(batch_X) 
            loss = F.cross_entropy(pred_y, batch_y)
            # --- ВНЕДРЕНИЕ РЕГУЛЯРИЗАЦИИ ВНИМАНИЯ ---
            if cfg.get('regularization', False) and hasattr(model, '_last_attn') and model._last_attn is not None:
                reg_coef = cfg.get('reg_coef', 0.1) # Сила штрафа
                reg_loss = compute_diversity_loss(model._last_attn)
                loss = loss + reg_coef * reg_loss
            # ----------------------------------------

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(pred_y, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            epoch_nfe_accum += get_nfe(model)
            batch_count += 1
        
        epoch_wall = time.perf_counter() - epoch_start
        epoch_dyn_fit = get_fit_time(model)
        epoch_pure = epoch_wall - epoch_dyn_fit
        
        train_stats['pure_time'] += epoch_pure
        train_stats['dynamic_fit'] += epoch_dyn_fit
        
        epoch_acc = correct / total
        epoch_avg_loss = train_loss / len(train_loader)
        epoch_avg_nfe = epoch_nfe_accum / batch_count
        train_stats['avg_nfe_sum'] += epoch_avg_loss 

        val_acc, val_loss, _, _ = evaluate_loop(model, val_loader, device)
        
        writer.add_scalar('Loss/Train', epoch_avg_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_acc * 100, epoch)
        writer.add_scalar('Accuracy/Val', val_acc * 100, epoch)
        writer.add_scalar('NFE/Train_Avg', epoch_avg_nfe, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == cfg['num_epochs']:
            logger.info(row_fmt.format(
                epoch, 
                f"{epoch_acc*100:.2f}%", 
                f"{val_acc*100:.2f}%", 
                f"{epoch_avg_loss:.4f}", 
                f"{epoch_avg_nfe:.1f}", 
                f"{epoch_dyn_fit:.3f}"
            ))

    train_total_fit = t_static_train + train_stats['dynamic_fit']
    train_all_time = train_stats['pure_time'] + train_total_fit
    #final_avg_train_nfe = train_stats['avg_nfe_sum'] / cfg['num_epochs'] 
# Было:
    final_avg_train_nfe = train_stats['avg_nfe_sum'] / max(cfg.get('num_epochs', 0), 1)
    
    logger.info("-" * 65)
    logger.info(">>> Training Summary:")
    logger.info(f"    Best Val Acc    : {best_val_acc*100:.2f}% (Epoch {best_epoch})")
    logger.info(f"    Total Pure Time : {train_stats['pure_time']:.2f}s")
    logger.info(f"    Total Fit Time  : {train_total_fit:.2f}s")

    if best_model_state:
        model.load_state_dict(best_model_state)

    log_banner(logger, "Phase 3: Clean Evaluation", char="-")
    
    test_acc, test_loss, test_avg_nfe, test_times = evaluate_loop(model, test_loader, device)
    
    test_total_fit = t_static_test + test_times['dynamic_fit']
    test_all_time = test_times['pure'] + test_total_fit
    
    logger.info(f"{'Metric':<20} | {'Value':<15}")
    logger.info("-" * 40)
    logger.info(f"{'Test Accuracy':<20} | {test_acc*100:.2f}%")
    logger.info(f"{'Avg NFE':<20} | {test_avg_nfe:.2f}")
    logger.info(f"{'Pure Time (s)':<20} | {test_times['pure']:.3f}")
    logger.info(f"{'Fit Time (s)':<20} | {test_total_fit:.3f}")
    
    writer.add_scalar('Accuracy/Test_Clean', test_acc * 100, best_epoch)

    noise_results = {}
    if cfg.get('test_noise_robustness', False):
        log_banner(logger, "Phase 4: Noise Robustness Test", char="-")
        noise_levels = cfg.get('test_noise_levels', [])
        
        logger.info(f"{'Noise Std':<10} | {'Accuracy':<10} | {'NFE':<10}")
        logger.info("-" * 36)
        
        for noise_std in noise_levels:
            noise = torch.randn_like(test_X_raw) * noise_std
            if cfg.get('add_time', 'yes').lower() == 'yes':
                noise[:, :, 0] = 0.0 
            noisy_X_raw = test_X_raw + noise
            noisy_coeffs, t_static_noise = get_static_inputs(noisy_X_raw, interp_type, kernel_params)
            
            noise_loader = DataLoader(TensorDataset(noisy_coeffs, test_y), batch_size=batch_size, shuffle=False)
            
            acc_noise, _, nfe_noise, times_noise = evaluate_loop(model, noise_loader, device)
            
            noise_total_fit = t_static_noise + times_noise['dynamic_fit']
            noise_all_time = times_noise['pure'] + noise_total_fit
            
            logger.info(f"{noise_std:<10} | {acc_noise*100:6.2f}%    | {nfe_noise:6.2f}")
            
            noise_results[str(noise_std)] = {
                "accuracy": acc_noise * 100,
                "avg_nfe": nfe_noise,
                "pure_time": times_noise['pure'],
                "fit_time": noise_total_fit,
                "all_time": noise_all_time
            }
            writer.add_scalar('Accuracy/Test_Noise', acc_noise * 100, int(noise_std*100))
    
    if cfg.get('save_attention_weights', False) and hasattr(model, '_last_attn'):
        if model._last_attn is not None:
            attn_dir = os.path.join(cfg['results_dir'], cfg['dataset_name'], 'attention_maps')
            os.makedirs(attn_dir, exist_ok=True)
            
            attn_path = os.path.join(attn_dir, f"{exp_name}_seed-{seed}_attn.pkl")
            
            save_data = {
                'attention_weights': model._last_attn,
                'exp_name': exp_name,
                'seed': seed,
                't_grid': model.t_grid.cpu() if model.t_grid is not None else None
            }
            
            with open(attn_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info("")
            logger.info(f"[INFO] Attention weights saved to: {attn_path}")

    attn_cosine_dist, attn_jsd = 0.0, 0.0
    if cfg.get('save_attention_weights', False) and hasattr(model, '_last_attn'):
        log_banner(logger, "Phase X: Attention Diversity & Cherry-Picking", char="-")
        
        all_attns = []
        all_Xs = [] # <--- НОВОЕ: Собираем траектории
        model.eval()
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                _ = model(batch_X) # Триггерим вычисление _last_attn
                if model._last_attn is not None:
                    all_attns.append(model._last_attn.detach().cpu())
                    all_Xs.append(batch_X.detach().cpu()) # <--- НОВОЕ: Сохраняем X
        
        if len(all_attns) > 0:
            full_attns = torch.cat(all_attns, dim=0)
            full_Xs = torch.cat(all_Xs, dim=0) # <--- НОВОЕ
            
            # Считаем разнообразие для КАЖДОГО семпла в тесте
            cos_dists, jsds = compute_attention_diversity(full_attns, return_per_sample=True)
            
            attn_cosine_dist = cos_dists.mean().item()
            attn_jsd = jsds.mean().item()
            
            logger.info(f"--- Global Attention Diversity Metrics (Max per sample, averaged over Test) ---")
            logger.info(f"    Mean Max Cosine Distance : {attn_cosine_dist:.4f} (Higher is more diverse)")
            logger.info(f"    Mean Max JSD             : {attn_jsd:.4f} (Higher is more diverse)")
            
            # Cherry-picking: берем Топ-5 самых разнообразных и 5 наименее разнообразных
            k_samples = min(5, len(cos_dists))
            top5_idx = torch.topk(cos_dists, k=k_samples).indices
            bottom5_idx = torch.topk(cos_dists, k=k_samples, largest=False).indices
            
            # === НОВОЕ: Берем первые 5 сэмплов теста (они гарантированно одинаковые для всех моделей) ===
            fixed_idx = torch.arange(k_samples) 
            
            attn_dir = os.path.join(cfg['results_dir'], cfg['dataset_name'], 'attention_maps')
            os.makedirs(attn_dir, exist_ok=True)
            attn_path = os.path.join(attn_dir, f"{exp_name}_seed-{seed}_attn.pkl")
            
            save_data = {
                'exp_name': exp_name,
                'seed': seed,
                'mean_cosine_distance': attn_cosine_dist,
                'mean_jsd': attn_jsd,
                'top5_diverse_attns': full_attns[top5_idx],
                'bottom5_diverse_attns': full_attns[bottom5_idx],
                'top5_Xs': full_Xs[top5_idx],       
                'bottom5_Xs': full_Xs[bottom5_idx], 
                # СОХРАНЯЕМ ФИКСИРОВАННЫЕ:
                'fixed_attns': full_attns[fixed_idx],
                'fixed_Xs': full_Xs[fixed_idx],
                't_grid': model.t_grid.cpu() if model.t_grid is not None else None,
            }
            with open(attn_path, 'wb') as f:
                pickle.dump(save_data, f)
            logger.info(f"[INFO] Attention stats, trajectories & cherry-picked samples saved to: {attn_path}")

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        logger.info(f"[INFO] Peak VRAM usage: {peak_vram_mb:.2f} MB")

    writer.close()

    results = {
        "final_test_accuracy": test_acc * 100,
        "best_val_accuracy": best_val_acc * 100,
        "best_epoch": best_epoch,  
        "avg_nfe_train": final_avg_train_nfe, 
        "avg_test_test": test_avg_nfe,             
        
        "train_pure_time": train_stats['pure_time'],
        "train_fit_time": train_total_fit,
        "train_all_time": train_all_time,
        
        "test_pure_time": test_times['pure'],
        "test_fit_time": test_total_fit,
        "test_all_time": test_all_time,
        
        "noise_results": noise_results,
        
        # --- Новые метрики для статьи ---
        "num_params": num_params,
        "peak_vram_mb": peak_vram_mb,
        # === ВЫНОСИМ В ФИНАЛЬНЫЙ JSON ДЛЯ ПОСТРОЕНИЯ ТАБЛИЦ ===
        "attention_cosine_distance": attn_cosine_dist,
        "attention_jsd": attn_jsd
    }
    
    log_banner(logger, "EXPERIMENT COMPLETED", char="=")
    return results