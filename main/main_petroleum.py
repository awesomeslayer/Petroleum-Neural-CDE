import yaml
import argparse
import os
import json
from pathlib import Path
from main.loading.utils import set_seed, setup_logger
from main.loading.data_loader_petroleum import get_petroleum_data
from main.model.model_petroleum import EmbeddingNeuralCDE, EmbeddingODERNN, EmbeddingGRUD, EmbeddingQFormerCDE, EmbeddingConvCDE
from main.model.trainer_petroleum import run_experiment_petroleum

def load_config(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f: json.dump(data, f, indent=4, default=str)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./main/configs/petroleum_main.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    logger = setup_logger(cfg.get('log_dir', 'logs'))
    logger.info(f"=== STARTING METRIC LEARNING SIMILARITY ({args.config}) ===")
    
    for seed in cfg.get('seeds', [42]):
        set_seed(seed)
        cfg['seed'] = seed
        
        data_tuple = get_petroleum_data(cfg, logger)
        train_X = data_tuple[0]
        
        input_channels = train_X.shape[2]
        seq_len = train_X.shape[1]
        add_time_bool = str(cfg.get('add_time', 'no')).lower() == 'yes'
        # --- ФЛАГ ДЛЯ RNN (МАСКА) ---
        use_mask_bool = str(cfg.get('use_mask', 'yes')).lower() == 'yes'
        bw_multiplier = float(seq_len)
        t_grid = data_tuple[9]
        
        embed_dim = 64
        
        def run_and_save(model, exp_name, group_folder, interp_type, k_params=None):
            res = run_experiment_petroleum(model, data_tuple, cfg, logger, exp_name, interp_type, k_params)
            res['dataset'] = cfg['dataset_name']
            res['seed'] = seed
            out_dir = Path(cfg.get('results_dir', 'experiment_petroleum')) / cfg['dataset_name'] / group_folder
            out_path = out_dir / f"{exp_name}_seed-{seed}.json"
            save_json(str(out_path), res)
            logger.info(f"Saved results to: {out_path}")
        
        for tol in cfg.get('tolerances', [1e-3]):
            
            if cfg.get('experiments_to_run', {}).get('baseline', False):
                for interp in cfg['baseline_params']['interpolations']:
                    model = EmbeddingNeuralCDE(input_channels, cfg['hidden_dim'], embed_dim, seq_len, 
                                              interpolation=interp, tol=tol, add_time=add_time_bool, t_grid=t_grid)
                    run_and_save(model, f"baseline_{interp}_tol{tol}", "baseline", interp, {'tol': tol})
                    
            if cfg.get('experiments_to_run', {}).get('kernel', False):
                learn_flag = cfg['kernel_params'].get('learnable', False)
                folder = "kernel_learnable" if learn_flag else "kernel"
                for k_name in cfg['kernel_params']['kernels']:
                    for bw in cfg['kernel_params']['bandwidths']:
                        bw_scaled = bw * bw_multiplier
                        kp = {'kernel': k_name, 'bandwidth': bw_scaled, 'tol': tol, 'learnable': learn_flag}
                        model = EmbeddingNeuralCDE(input_channels, cfg['hidden_dim'], embed_dim, seq_len, 
                                                  interpolation="kernel", kernel_params=kp, tol=tol, add_time=add_time_bool, t_grid=t_grid)
                        run_and_save(model, f"kernel-{k_name}_bw-{bw_scaled}_tol{tol}", folder, "kernel", kp)
                        
            if cfg.get('experiments_to_run', {}).get('gp', False):
                learn_flag = cfg['gp_params'].get('learnable', False)
                folder = "GP_learnable" if learn_flag else "GP"
                for ls in cfg['gp_params']['length_scales']:
                    for noise in cfg['gp_params']['noise_stds']:
                        ls_scaled = ls * bw_multiplier
                        kp = {'length_scale': ls_scaled, 'noise_std': noise, 'tol': tol, 'learnable': learn_flag}
                        model = EmbeddingNeuralCDE(input_channels, cfg['hidden_dim'], embed_dim, seq_len, 
                                                  interpolation="gp", kernel_params=kp, tol=tol, add_time=add_time_bool, t_grid=t_grid)
                        run_and_save(model, f"gp_ls-{ls_scaled}_noise-{noise}_tol{tol}", folder, "gp", kp)

            if cfg.get('experiments_to_run', {}).get('qformer', False):
                q_cfg = cfg['qformer_params']
                learn_flag = q_cfg.get('learnable', False)
                folder = "qformer_learnable" if learn_flag else "qformer"
                bw_raw_lists = q_cfg.get('bandwidth_lists_different', [])
                for bws_raw in bw_raw_lists:
                    bws_scaled = [round(b * bw_multiplier, 4) for b in bws_raw]
                    for k_name in q_cfg.get('kernels', []):
                        aggr = q_cfg.get('aggregations', ['concat'])[0]
                        q_params = {'kernel': k_name, 'bandwidths': bws_scaled, 'tol': tol, 'aggregation': aggr, 'learnable': learn_flag}
                        if k_name == 'gp': q_params['noise_std'] = 0.01
                        model = EmbeddingQFormerCDE(input_channels, cfg['hidden_dim'], embed_dim, seq_len, 
                                                   qformer_params=q_params, add_time=add_time_bool, t_grid=t_grid)
                        run_and_save(model, f"qformer_{k_name}_ls-{bws_scaled}_tol{tol}", folder, "qformer", q_params)

            if cfg.get('experiments_to_run', {}).get('conv', False):
                c_cfg = cfg['conv_params']
                k_size = c_cfg['kernel_size']
                learn_flag = c_cfg.get('learnable', False)
                folder = "conv_learnable" if learn_flag else "conv"
                bw_raw_lists = c_cfg.get('bandwidth_lists_different', [])
                for bws_raw in bw_raw_lists:
                    bws_scaled = [round(b * bw_multiplier, 4) for b in bws_raw]
                    for k_name in c_cfg.get('kernels', []):
                        aggr = c_cfg.get('aggregations', ['concat'])[0]
                        c_params = {'kernel': k_name, 'bandwidths': bws_scaled, 'tol': tol, 'aggregation': aggr, 'conv_kernel_size': k_size, 'learnable': learn_flag}
                        if k_name == 'gp': c_params['noise_std'] = 0.01
                        model = EmbeddingConvCDE(input_channels, cfg['hidden_dim'], embed_dim, seq_len, 
                                                conv_params=c_params, add_time=add_time_bool, t_grid=t_grid)
                        run_and_save(model, f"conv_{k_name}_ls-{bws_scaled}_tol{tol}", folder, "conv", c_params)

            if cfg.get('experiments_to_run', {}).get('odernn', False):
                model = EmbeddingODERNN(input_channels, cfg['hidden_dim'], embed_dim, seq_len, tol=tol, add_time=add_time_bool, t_grid=t_grid, use_mask=use_mask_bool)
                run_and_save(model, f"odernn_mask-{use_mask_bool}_tol{tol}", "ODE-RNN", "odernn", {'tol': tol})
                
            if cfg.get('experiments_to_run', {}).get('grud', False):
                model = EmbeddingGRUD(input_channels, cfg['hidden_dim'], embed_dim, seq_len, add_time=add_time_bool, t_grid=t_grid, use_mask=use_mask_bool)
                run_and_save(model, f"grud_mask-{use_mask_bool}_tol{tol}", "GRU-D", "grud", {'tol': tol})

if __name__ == "__main__":
    main()