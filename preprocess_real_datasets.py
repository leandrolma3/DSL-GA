# preprocess_real_datasets.py (Versão Final e Corrigida)
#
# OBJETIVO:
# Ler os datasets reais brutos, tratar dados faltantes, padronizar
# e salvar como um único arquivo .csv limpo para cada um na pasta 'datasets/processed/'.

import os
import pandas as pd
import logging

# A CORREÇÃO ESTÁ NA LINHA ABAIXO
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S') # <<< REMOVIDO PARÊNTESES E '%' extras

logger = logging.getLogger("DataPreprocessor_v2.1")

# --- CONFIGURAÇÃO DOS DATASETS ---
# CORREÇÃO 1: Ajuste dos caminhos para remover a subpasta 'raw'.
DATASET_CONFIG = {
    'shuttle': {
        'raw_files': ['datasets/shuttle/shuttle.trn'],
        'output_file': 'datasets/processed/shuttle_processed.csv',
        'columns': [f'attr_{i}' for i in range(9)] + ['class'],
        'read_params': {'delimiter': ' ', 'header': None}
    },
    'covertype': {
        'raw_files': ['datasets/covertype/covtype.data'],
        'output_file': 'datasets/processed/covertype_processed.csv',
        'columns': [f'attr_{i}' for i in range(54)] + ['class'],
        'read_params': {'delimiter': ',', 'header': None}
    },
    'poker': {
        'raw_files': ['datasets/poker/poker-hand-training-true.data'],
        'output_file': 'datasets/processed/poker_processed.csv',
        'columns': ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'class'],
        'read_params': {'delimiter': ',', 'header': None}
    },
    'intellabsensors': {
        'raw_files': ['datasets/intellabsensors/data.txt'],
        'output_file': 'datasets/processed/intellabsensors_processed.csv',
        'columns': ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage'],
        'drop_columns': ['date', 'time'],
        'target_column': 'moteid',
        'impute_cols_interpolate': ['temperature', 'humidity', 'light', 'voltage'],
        'read_params': {'delimiter': r'\s+', 'header': None, 'on_bad_lines': 'skip'}
    }
}

def handle_missing_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Trata dados faltantes em um DataFrame usando as estratégias definidas no config."""
    cols_to_impute = config.get('impute_cols_interpolate', [])
    if not cols_to_impute:
        initial_rows = len(df)
        df.dropna(inplace=True)
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            logger.warning(f"Removed {rows_removed} rows with missing values.")
        return df

    logger.info(f"Starting imputation for columns: {cols_to_impute}")
    for col in cols_to_impute:
        if col in df.columns:
            initial_missing = df[col].isnull().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].interpolate(method='linear', limit_direction='both', axis=0)
            final_missing = df[col].isnull().sum()
            imputed_count = initial_missing - final_missing
            if imputed_count > 0:
                logger.info(f"Imputed {imputed_count} missing values in column '{col}'.")
    return df

def process_dataset(name: str, config: dict):
    """Função genérica para carregar, processar e salvar um dataset."""
    logger.info(f"--- Processing dataset: {name} ---")
    
    try:
        df_list = [pd.read_csv(f, **config['read_params']) for f in config['raw_files']]
        df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Loaded {len(df)} rows from {len(df_list)} file(s).")
        
        df.columns = config['columns']
        
        if 'target_column' in config and config['target_column'] in df.columns:
            target = config['target_column']
            other_cols = [col for col in df.columns if col != target]
            df = df[other_cols + [target]]
            df.rename(columns={target: 'class'}, inplace=True)
            logger.info(f"Set '{target}' as the target column, renaming to 'class'.")

        if 'drop_columns' in config:
            # <<< CORREÇÃO 2: Removido 'inplace=True' para evitar o erro AttributeError >>>
            df = df.drop(columns=config['drop_columns'])
            logger.info(f"Dropped columns: {config['drop_columns']}")
            
        df = handle_missing_data(df, config)

        output_path = config['output_file']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved processed file to: {output_path}")

    except FileNotFoundError:
        logger.error(f"RAW FILE NOT FOUND for '{name}'. Searched in: {config['raw_files']}. Please check the path.")
    except Exception:
        logger.exception(f"An error occurred while processing '{name}'.")

def main():
    logger.info("Starting preprocessing of real-world datasets.")
    for name, config in DATASET_CONFIG.items():
        process_dataset(name, config)
    logger.info("Preprocessing finished.")

if __name__ == '__main__':
    main()