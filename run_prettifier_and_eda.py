import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import csv

logging.basicConfig(level=logging.INFO)

try:
    import data_prettifier
    from data_prettifier import prepare_general_data_with_expert_labels
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("Ошибка: Не удалось импортировать data_prettifier.py.")
    exit(1)

# === ПАТЧ СОВМЕСТИМОСТИ PANDAS ===
def patched_preprocess_expert_labeling(path_to_labeling, dataframe, well_column, depth_column, 
                                       formation_column, class_column, layer_column, target_column, fm_postfix=False):
    well_dataset = dataframe.copy()
    well_dataset[well_column] = well_dataset[well_column].apply(lambda x: re.sub("[^0-9A-z]", "", str(x)))
    
    if isinstance(path_to_labeling, str):
        path_to_labeling = [path_to_labeling]

    for filename in path_to_labeling:
        formation = re.match("[a-zA-Z]+", filename.name)[0]
        target = pd.read_excel(filename, sheet_name="Sheet1")
        if "WellName" in target: target = target.rename(columns={"WellName": "Well"})
        if "Top" in target: target = target.rename(columns={"Top": "top"})
        if "Bottom" in target: target = target.rename(columns={"Bottom": "bottom"})
        target["Well"] = target["Well"].apply(lambda x: re.sub("[^0-9A-z]", "", str(x)))

        mask = (well_dataset[formation_column] == formation + " Fm.") if fm_postfix else (well_dataset[formation_column] == formation)
        data = well_dataset[mask].copy()
        other_formations = well_dataset[~mask]

        if class_column not in data: data[class_column] = pd.Series(np.nan, index=data.index, dtype=object)
        if layer_column not in data: data[layer_column] = pd.Series(np.nan, index=data.index, dtype=object)

        for i in range(len(target)):
            data.loc[
                (data[well_column] == target["Well"].iloc[i]) &
                (data[depth_column] >= target["top"].iloc[i]) &
                (data[depth_column] <= target["bottom"].iloc[i]),
                [class_column, layer_column]
            ] = [target["Class"].iloc[i], target["Layer"].iloc[i]]

        mask_valid = data[class_column].notna() & data[layer_column].notna()
        data[target_column] = pd.Series(np.nan, index=data.index, dtype=object)
        data.loc[mask_valid, target_column] = data.loc[mask_valid, class_column].astype(str) + data.loc[mask_valid, layer_column].astype(str)

        le = LabelEncoder()
        data.loc[mask_valid, target_column] = le.fit_transform(data.loc[mask_valid, target_column])
        well_dataset = pd.concat([data, other_formations]).sort_index()

    return well_dataset

data_prettifier.preprocess_expert_labeling = patched_preprocess_expert_labeling
print("[Система] Патч совместимости Pandas применен успешно.")

def process_dataset(raw_csv, output_csv, label_files, fm_postfix=False, fill_nans=True):
    print(f"\n{'='*50}\nОБРАБОТКА ДАТАСЕТА: {output_csv} (Fill NaNs: {fill_nans})\n{'='*50}")
    
    if not os.path.exists(raw_csv):
        print(f"Файл {raw_csv} не найден! Пропуск.")
        return

    with open(raw_csv, 'r') as f:
        first_line = f.readline()
        try:
            dialect = csv.Sniffer().sniff(first_line)
            file_delimiter = dialect.delimiter
        except Exception:
            file_delimiter = ';' if ';' in first_line else ','
            
    print(f"Определен разделитель файла: '{file_delimiter}'")
    
    raw_df = pd.read_csv(raw_csv, low_memory=False, delimiter=file_delimiter)
    raw_cols = raw_df.columns.tolist()
    
    well_col = "WELL" if "WELL" in raw_cols else ("WELLNAME" if "WELLNAME" in raw_cols else raw_cols[0])
    depth_col = "DEPTH_MD" if "DEPTH_MD" in raw_cols else ("DEPT" if "DEPT" in raw_cols else raw_cols[1])
    formation_col = "FORMATION" if "FORMATION" in raw_cols else ("LITHO_FORM" if "LITHO_FORM" in raw_cols else None)

    possible_cols = ["DRHO", "DENS", "GR", "DTC", "CALI", "BS", "RESS", "RESD", "RESM", "NEUT", "SP", "RHOB", "NPHI", "DTS"]
    required_cols = [well_col, depth_col]
    if formation_col: required_cols.append(formation_col)
    for col in possible_cols:
        if col in raw_cols: required_cols.append(col)

    log_cols = [c for c in ["RESS", "RESD", "RESM"] if c in raw_cols]
    norm_cols = [c for c in ["GR", "NEUT"] if c in raw_cols]
    
    # === ГЛАВНОЕ ИЗМЕНЕНИЕ ДЛЯ SPARSE ДАТАСЕТОВ ===
    if fill_nans:
        features_to_fill = [c for c in ["DRHO", "DENS", "GR", "DTC", "RHOB", "NPHI", "DTS"] if c in raw_cols]
    else:
        features_to_fill = [] # Заменили None на [], чтобы логгер не падал
    
    fix_outlier_values = [(c, 0.0, 100000.0) for c in log_cols]
    diff_cols = [("CALI", "BS", 19.685)] if "CALI" in raw_cols and "BS" in raw_cols else None

    print(f"Запуск data_prettifier (fm_postfix={fm_postfix}, fill_nans={fill_nans})...")
    
    processed_df, _ = prepare_general_data_with_expert_labels(
        path_to_data=raw_csv,
        path_to_labeling=label_files,
        required_cols=required_cols,
        well_column=well_col,
        depth_column=depth_col,
        formation_column=formation_col if formation_col else "FORMATION",
        class_column="CLASS",
        layer_column="LAYER",
        target_column="TARGET",
        delimiter=file_delimiter,
        encode_cols=[well_col, formation_col] if formation_col else [well_col],
        fix_outlier_values=fix_outlier_values,
        drop_cols=["SP"] if "SP" in raw_cols else None,
        log_cols=log_cols,
        diff_cols=diff_cols,
        norm_cols=norm_cols,
        group_key_cols=well_col,
        norm_group_key_cols=[well_col],
        features_to_fill=features_to_fill,
        fm_postfix=fm_postfix
    )

    processed_df = processed_df.rename(columns={well_col: "WELLNAME", depth_col: "DEPT"})
    processed_df.to_csv(output_csv, index=False)
    print(f"[Успешно] Сохранено строк: {len(processed_df)} -> {output_csv}")

if __name__ == "__main__":
    # 1. ОБРАБОТКА НОВОЙ ЗЕЛАНДИИ
    nz_labels = [Path(f"data/{f}") for f in [
        'Manganui.xlsx', 'Marinui.xlsx', 'Matemateaonga.xlsx', 
        'Mohakatino.xlsx', 'Moki.xlsx', 'Urenui.xlsx', 'Urenui-another.xlsx'
    ] if os.path.exists(f"data/{f}")]
    
    process_dataset("data/logs.csv", "data/preprocessed_nz_filled.csv", nz_labels, fm_postfix=False, fill_nans=True)
    process_dataset("data/logs.csv", "data/preprocessed_nz_sparse.csv", nz_labels, fm_postfix=False, fill_nans=False)

    # 2. ОБРАБОТКА НОРВЕГИИ
    norway_labels = [Path("data/Utsira.xlsx")] if os.path.exists("data/Utsira.xlsx") else []
    
    process_dataset("data/Norwey.csv", "data/preprocessed_norway_filled.csv", norway_labels, fm_postfix=True, fill_nans=True)
    process_dataset("data/Norwey.csv", "data/preprocessed_norway_sparse.csv", norway_labels, fm_postfix=True, fill_nans=False)