import os
import json
import re
import pandas as pd

def parse_filename(filename, group):
    params = {'Model_Subtype': 'Unknown', 'Kernel_Scale': 'N/A', 'Num_Heads': 1, 'Tolerance': 'N/A', 'Seed': 'N/A'}
    name = os.path.splitext(filename)[0]
    g = group.lower()
    
    if 'baseline' in g:
        params['Model_Subtype'] = 'Cubic CDE' if 'cubic' in name.lower() else 'Linear CDE'
    elif 'gru' in g:
        params['Model_Subtype'] = 'Vanilla GRU (No Mask)' if ('mask-false' in name.lower() or 'mask-no' in name.lower()) else 'GRU-D (With Mask)'
    elif 'gp' in g and 'qformer' not in g and 'conv' not in g:
        params['Model_Subtype'] = 'GP CDE'
        ls_match = re.search(r'ls-([^_\n]+)', name)
        if ls_match: params['Kernel_Scale'] = f"ls={ls_match.group(1)}"
    elif 'kernel' in g:
        params['Model_Subtype'] = 'Kernel CDE'
        bw_match = re.search(r'bw-([^_\n]+)', name)
        if bw_match: params['Kernel_Scale'] = f"bw={bw_match.group(1)}"
    elif 'qformer' in g:
        params['Model_Subtype'] = 'MV-CDE (Q-Former)'
        ls_match = re.search(r'ls-([^_\n]+)', name)
        if ls_match: 
            val = ls_match.group(1)
            params['Kernel_Scale'] = f"scales={val}"
            params['Num_Heads'] = len(val.split(',')) if '[' in val else 1
    elif 'conv' in g:
        params['Model_Subtype'] = 'MVC-CDE (Conv)'
        ls_match = re.search(r'ls-([^_\n]+)', name)
        if ls_match: 
            val = ls_match.group(1)
            params['Kernel_Scale'] = f"scales={val}"
            params['Num_Heads'] = len(val.split(',')) if '[' in val else 1

    if 'learnable' in g: params['Model_Subtype'] += ' (Learnable)'
    
    tol_match = re.search(r'tol-?([\d\.]+)', name)
    if tol_match: params['Tolerance'] = tol_match.group(1)
    
    return params

def build_summary(base_dir="experiment_petroleum_sparse"):
    data = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                try:
                    with open(os.path.join(root, file), 'r') as f: res = json.load(f)
                except Exception: continue
                
                hparams = parse_filename(file, os.path.basename(root))
                row = {
                    'Model': hparams['Model_Subtype'],
                    'Kernel_Scale': hparams['Kernel_Scale'],
                    'Num_Heads': hparams['Num_Heads'],
                    'Tol': hparams['Tolerance'],
                    'ARI_Class': res.get('final_test_ari_class', 0),
                    'ARI_Well': res.get('final_test_ari_well', 0),
                    'Test_NFE': res.get('avg_test_test', 0),
                    'Train_NFE': res.get('avg_nfe_train', 0),
                    'Train_Time': res.get('train_pure_time', 0),
                    'Train_Fit_Time': res.get('train_fit_time', 0),
                    'Test_Time': res.get('test_pure_time', 0),
                    'Test_Fit_Time': res.get('test_fit_time', 0),
                }
                
                noise_data = res.get('noise_results', {})
                for lvl in ['0.1', '0.3', '0.5']:
                    row[f'Noise_{lvl}_ARI'] = noise_data.get(lvl, {}).get('ari_class', None)
                    row[f'Noise_{lvl}_NFE'] = noise_data.get(lvl, {}).get('avg_nfe', None)
                data.append(row)
                
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = build_summary()
    if not df.empty:
        os.makedirs("pictures", exist_ok=True)
        df = df.sort_values(by=['Model', 'Num_Heads', 'ARI_Class'], ascending=[True, True, False])
        float_cols = [c for c in df.columns if df[c].dtype in ['float64', 'float32']]
        df[float_cols] = df[float_cols].round(4)
        df.to_csv("pictures/results_summary_sparse.csv", index=False)
        print("Successfully saved pictures/results_summary_sparse.csv!")