import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import LabelEncoder

def get_petroleum_data(cfg, logger):
    csv_filename = cfg.get('csv_name', "preprocessed_nz_filled.csv")
    csv_path = os.path.join(cfg.get('data_dir', './data/'), csv_filename)
    logger.info(f"Loading data for METRIC LEARNING from {csv_path}...")
    
    df = pd.read_csv(csv_path, low_memory=False)
    
    features = cfg.get('features', ["GR", "DENS", "DTC", "DRHO", "RESS", "RESD", "RESM"])
    base_target = cfg.get('target_col', "CLASS")
    seq_len = cfg['seq_len']
    add_time = str(cfg.get('add_time', 'no')).lower() == 'yes'
    
    target_col = base_target
    
    # Кодируем макро-классы
    valid_mask = df[target_col].notna()
    le_class = LabelEncoder()
    df.loc[valid_mask, target_col] = le_class.fit_transform(df.loc[valid_mask, target_col].astype(str))
    num_classes = len(le_class.classes_)
    
    # Кодируем ID скважин
    le_well = LabelEncoder()
    df['WELL_ID'] = le_well.fit_transform(df['WELLNAME'].astype(str))
    
    logger.info(f"Target Column: {target_col} | Found {num_classes} unique Macro-Classes.")
    logger.info(f"Total Wells: {len(le_well.classes_)}")

    wells = df['WELL_ID'].unique()
    np.random.seed(cfg.get('seed', 42))
    np.random.shuffle(wells)
    
    n_wells = len(wells)
    train_wells = set(wells[:int(0.6 * n_wells)])
    val_wells = set(wells[int(0.6 * n_wells):int(0.8 * n_wells)])
    test_wells = set(wells[int(0.8 * n_wells):])

    def extract_windows(well_set):
        X_list, y_list, well_ids = [], [], []
        sub_df = df[df['WELL_ID'].isin(well_set)]
        
        for well_id, group in sub_df.groupby('WELL_ID'):
            group = group.sort_values('DEPT')
            
            depth = group['DEPT'].values.astype(np.float32)
            feats = group[features].values.astype(np.float32)
            
            # --- Поскважинная нормализация с ИГНОРИРОВАНИЕМ NaN ---
            for f_col_idx in range(feats.shape[1]):
                valid_vals = feats[:, f_col_idx][~np.isnan(feats[:, f_col_idx])]
                if len(valid_vals) > 0:
                    w_mean = valid_vals.mean()
                    w_std = valid_vals.std() + 1e-8
                    feats[:, f_col_idx] = (feats[:, f_col_idx] - w_mean) / w_std
                else:
                    # Если признак вообще пустой во всей скважине, ставим нули
                    feats[:, f_col_idx] = 0.0
            
            window_data = np.concatenate([depth[:, None], feats], axis=1) if add_time else feats
                
            targets = group[target_col].values
            targets = np.nan_to_num(targets, nan=-100).astype(np.int64)
            
            n_windows = len(window_data) // seq_len
            for i in range(n_windows):
                start = i * seq_len
                end = start + seq_len
                
                win_targs = targets[start:end]
                valid_targs = win_targs[win_targs != -100]
                
                if len(valid_targs) > 0:
                    win_class = int(np.bincount(valid_targs).argmax())
                    
                    chunk = window_data[start:end, :].copy()
                    if add_time: chunk[:, 0] = chunk[:, 0] - chunk[0, 0]
                    
                    X_list.append(chunk)
                    y_list.append(win_class)
                    well_ids.append(well_id)
                
        return np.array(X_list), np.array(y_list), np.array(well_ids)

    logger.info("Extracting Class-labeled windows with Per-Well normalization...")
    X_train_np, y_train_np, w_train_np = extract_windows(train_wells)
    X_val_np, y_val_np, w_val_np = extract_windows(val_wells)
    X_test_np, y_test_np, w_test_np = extract_windows(test_wells)
    
    logger.info(f"Valid Windows -> Train: {len(X_train_np)}, Val: {len(X_val_np)}, Test: {len(X_test_np)}")

    # ВНИМАНИЕ: Мы УБРАЛИ np.nan_to_num().
    # Теперь NaN остаются в данных и доходят до моделей.
    # Библиотека torchcde (для cubic/linear сплайнов) упадет, если в нее дать NaN.
    # Но мы уже защитили ее в trainer_petroleum.py (функция get_static_inputs_petroleum), 
    # где X_safe = torch.nan_to_num(X, nan=0.0) применяется только для кубических/линейных сплайнов!
    # А наши GP, Gaussian и GRU-D будут обрабатывать NaN напрямую!

    t_grid = torch.linspace(0, seq_len - 1, seq_len) if not add_time else None
        
    return (torch.tensor(X_train_np), torch.tensor(X_val_np), torch.tensor(X_test_np),
            torch.tensor(y_train_np), torch.tensor(y_val_np), torch.tensor(y_test_np),
            torch.tensor(w_train_np), torch.tensor(w_val_np), torch.tensor(w_test_np),
            t_grid, num_classes)