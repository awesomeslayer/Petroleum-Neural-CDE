import torch
import torch.nn.functional as F
import time
import os
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from main.model.trainer import get_nfe, reset_nfe, reset_fit_timer, get_fit_time

def get_static_inputs_petroleum(X, interpolation_type, kernel_params=None):
    import torchcde
    if kernel_params is None: kernel_params = {}
    start_time = time.perf_counter()
    
    if interpolation_type in ["cubic", "linear"]:
        # Переносим на GPU для моментального расчета
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_gpu = X.to(device)
        
        # ВНИМАНИЕ: torchcde НАТИВНО умеет интерполировать NaNs!
        # Вместо заполнения нулями мы позволяем библиотеке построить 
        # плавный сплайн напрямую между физически существующими точками.
        try:
            if interpolation_type == "cubic": 
                coeffs = torchcde.natural_cubic_coeffs(X_gpu)
            else: 
                coeffs = torchcde.linear_interpolation_coeffs(X_gpu)
        except Exception as e:
            # Резервный случай: если в каком-то окне колонка полностью состоит из NaNs
            X_safe = torch.nan_to_num(X_gpu, nan=0.0)
            if interpolation_type == "cubic": 
                coeffs = torchcde.natural_cubic_coeffs(X_safe)
            else: 
                coeffs = torchcde.linear_interpolation_coeffs(X_safe)
                
        coeffs = coeffs.cpu()
    else:
        # Для GP, Kernel и GRU — просто пробрасываем тензор с NaNs дальше
        coeffs = X
        
    return coeffs, time.perf_counter() - start_time

class TripletDataset(Dataset):
    def __init__(self, coeffs, labels, wells):
        self.coeffs = coeffs
        self.labels = labels.numpy() 
        self.wells = wells.numpy()   
        self.unique_wells = np.unique(self.wells)
        self.well_indices = {w: np.where(self.wells == w)[0] for w in self.unique_wells}
        
    def __len__(self): return len(self.coeffs)
        
    def __getitem__(self, idx):
        anchor = self.coeffs[idx]
        well_id = self.wells[idx]
        pos_indices = self.well_indices[well_id]
        pos_idx = idx
        if len(pos_indices) > 1:
            while pos_idx == idx: pos_idx = np.random.choice(pos_indices)
        positive = self.coeffs[pos_idx]
        
        neg_wells = self.unique_wells[self.unique_wells != well_id]
        if len(neg_wells) == 0: neg_idx = np.random.choice(pos_indices)
        else:
            neg_well = np.random.choice(neg_wells)
            neg_idx = np.random.choice(self.well_indices[neg_well])
        negative = self.coeffs[neg_idx]
        
        return anchor, positive, negative

def evaluate_ari_clustering(model, coeffs, labels, wells, bs, device):
    model.eval()
    reset_fit_timer(model)
    reset_nfe(model)
    
    loader = DataLoader(TensorDataset(coeffs, labels, wells), batch_size=bs, shuffle=False)
    all_embs, all_labels, all_wells = [], [], []
    total_nfe = 0
    start_wall = time.perf_counter()
    
    with torch.no_grad():
        for batch_X, batch_y, batch_w in loader:
            batch_X = batch_X.to(device)
            reset_nfe(model)
            emb = model(batch_X)
            
            all_embs.append(emb.cpu().numpy())
            all_labels.append(batch_y.numpy())
            all_wells.append(batch_w.numpy())
            total_nfe += get_nfe(model)
            
    wall_clock = time.perf_counter() - start_wall
    dyn_fit = get_fit_time(model)
    
    embs = np.concatenate(all_embs, axis=0)
    lbls = np.concatenate(all_labels, axis=0)
    wlls = np.concatenate(all_wells, axis=0)
    
    ari_class, ari_well = 0.0, 0.0
    if len(np.unique(lbls)) > 1:
        ari_class = adjusted_rand_score(lbls, AgglomerativeClustering(n_clusters=len(np.unique(lbls))).fit_predict(embs))
    if len(np.unique(wlls)) > 1:
        ari_well = adjusted_rand_score(wlls, AgglomerativeClustering(n_clusters=len(np.unique(wlls))).fit_predict(embs))
        
    return {'ari_class': ari_class, 'ari_well': ari_well, 'nfe': total_nfe / len(loader), 'pure': wall_clock - dyn_fit, 'dynamic_fit': dyn_fit}

def run_experiment_petroleum(model, data, cfg, logger, exp_name, interp_type, kernel_params=None):
    if kernel_params is None: kernel_params = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_X, val_X, test_X, train_y, val_y, test_y, train_w, val_w, test_w, t_grid, num_classes = data
    
    logger.info(">>> Pre-calculating Interpolation...")
    train_coeffs, t_tr = get_static_inputs_petroleum(train_X, interp_type, kernel_params)
    val_coeffs, t_val  = get_static_inputs_petroleum(val_X, interp_type, kernel_params)
    test_coeffs, t_te  = get_static_inputs_petroleum(test_X, interp_type, kernel_params)

    bs = cfg.get('batch_size', 256)
    train_loader = DataLoader(TripletDataset(train_coeffs, train_y, train_w), batch_size=bs, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))
    triplet_loss_fn = torch.nn.TripletMarginLoss(margin=0.3, p=2)
    
    best_val_ari_class, best_model_state, best_epoch = -1.0, None, 0
    train_stats = {'pure_time': 0.0, 'dynamic_fit': 0.0, 'avg_nfe_sum': 0.0}
    
    for epoch in range(1, cfg.get('num_epochs', 40) + 1):
        epoch_start = time.perf_counter()
        model.train()
        reset_fit_timer(model)
        train_loss, batch_nfe, batch_count = 0.0, 0, 0
        
        for anchor_X, pos_X, neg_X in train_loader:
            optimizer.zero_grad(); reset_nfe(model) 
            loss = triplet_loss_fn(model(anchor_X.to(device)), model(pos_X.to(device)), model(neg_X.to(device)))
            loss.backward(); optimizer.step()
            train_loss += loss.item(); batch_nfe += get_nfe(model); batch_count += 1
            
        epoch_dyn_fit = get_fit_time(model)
        train_stats['pure_time'] += ((time.perf_counter() - epoch_start) - epoch_dyn_fit)
        train_stats['dynamic_fit'] += epoch_dyn_fit
        train_stats['avg_nfe_sum'] += batch_nfe / max(batch_count, 1)
        
        val_metrics = evaluate_ari_clustering(model, val_coeffs, val_y, val_w, bs, device)
        if val_metrics['ari_class'] >= best_val_ari_class:
            best_val_ari_class = val_metrics['ari_class']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            
        
        logger.info(f"Epoch {epoch:<3} | TrnLoss: {train_loss/max(batch_count,1):.4f} | Val ARI(Cls): {val_metrics['ari_class']:.4f} | Val ARI(Well): {val_metrics['ari_well']:.4f} | NFE: {batch_nfe/max(batch_count,1):.1f}")
        
    if best_model_state: model.load_state_dict(best_model_state)
    test_metrics = evaluate_ari_clustering(model, test_coeffs, test_y, test_w, bs, device)
    
    logger.info(f"FINAL TEST ARI (CLASS): {test_metrics['ari_class']:.4f} | TEST NFE: {test_metrics['nfe']:.1f}")
    
    # === ФАЗА 1: ТЕСТ НА ШУМ ===
    noise_results = {}
    if cfg.get('test_noise_robustness', False):
        logger.info(">>> Running Noise Robustness Test...")
        for noise_std in cfg.get('test_noise_levels', [0.1, 0.2, 0.3]):
            # Добавляем шум. NaNs останутся NaNs, что правильно имитирует зашумленный сенсор с пропусками!
            noise = torch.randn_like(test_X) * noise_std
            if cfg.get('add_time', 'no').lower() == 'yes': noise[:, :, 0] = 0.0 
            noisy_test_X = test_X + noise
            
            noisy_coeffs, t_noise = get_static_inputs_petroleum(noisy_test_X, interp_type, kernel_params)
            metrics_noise = evaluate_ari_clustering(model, noisy_coeffs, test_y, test_w, bs, device)
            logger.info(f"    Noise {noise_std}: ARI Class = {metrics_noise['ari_class']:.4f}, NFE = {metrics_noise['nfe']:.1f}")
            noise_results[str(noise_std)] = {
                "ari_class": metrics_noise['ari_class'], "ari_well": metrics_noise['ari_well'],
                "avg_nfe": metrics_noise['nfe'], "test_pure_time": metrics_noise['pure'], "test_fit_time": t_noise + metrics_noise['dynamic_fit']
            }

    # === ФАЗА 2: ВЫГРУЗКА ATTENTION MAPS ===
    if cfg.get('save_attention_weights', False):
        logger.info(">>> Extracting Attention Maps...")
        model.eval()
        with torch.no_grad():
            sample_X = test_X[:15].to(device)
            sample_coeffs, _ = get_static_inputs_petroleum(sample_X, interp_type, kernel_params)
            _ = model(sample_coeffs)
            if hasattr(model, '_last_attn') and model._last_attn is not None:
                attn_dir = Path(cfg.get('results_dir', 'experiment_petroleum_sparse')) / cfg['dataset_name'] / 'attention_maps'
                attn_dir.mkdir(parents=True, exist_ok=True)
                attn_path = attn_dir / f"{exp_name}_reg-on_seed-{cfg.get('seed', 42)}_attn.pkl"
                with open(attn_path, 'wb') as f:
                    pickle.dump({'exp_name': exp_name, 'seed': cfg.get('seed', 42), 'fixed_attns': model._last_attn.cpu(), 'fixed_Xs': sample_X.cpu()}, f)
                logger.info(f"[INFO] Attention weights saved to {attn_path}")
    
    return {
        "final_test_ari_class": test_metrics['ari_class'], "final_test_ari_well": test_metrics['ari_well'],
        "best_val_ari_class": best_val_ari_class, "best_epoch": best_epoch,
        "avg_nfe_train": train_stats['avg_nfe_sum'] / max(cfg.get('num_epochs', 40), 1), "avg_test_test": test_metrics['nfe'],
        "train_pure_time": train_stats['pure_time'], "train_fit_time": t_tr + train_stats['dynamic_fit'],
        "test_pure_time": test_metrics['pure'], "test_fit_time": t_te + test_metrics['dynamic_fit'],
        "noise_results": noise_results 
    }