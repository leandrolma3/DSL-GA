import logging
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Any

# --- Importações do seu projeto ---
try:
    import data_handling
    from constants import RANDOM_SEED
except ImportError as e:
    print(f"Erro de importação: {e}. Certifique-se que o script está na pasta raiz do projeto.")
    exit()

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("FeatureImportanceDebugger")

# --- Parâmetros do Teste ---
CONFIG_FILE_PATH = 'config.yaml'
TARGET_EXPERIMENT_ID = 'CovType'
RF_PARAMS = {
    'n_estimators': 20,
    'max_depth': 10,
    'min_samples_leaf': 10,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

if __name__ == "__main__":
    print("\n" + "="*80)
    logger.info("INICIANDO SCRIPT DE VALIDAÇÃO DA EXTRAÇÃO DE IMPORTÂNCIA DAS FEATURES")
    print("="*80)

    # --- FASE 1: SETUP E TREINAMENTO COM 'FOCUS TRAINING' ---
    # Esta parte é idêntica à do script de teste anterior e já está validada.
    try:
        logger.info("Carregando configuração e chunk de dados...")
        config = data_handling.load_full_config(CONFIG_FILE_PATH)
        chunks = data_handling.generate_dataset_chunks(
            stream_or_dataset_name=TARGET_EXPERIMENT_ID,
            chunk_size=config['data_params']['chunk_size'],
            num_chunks=2,
            max_instances=config['data_params']['chunk_size'] * 3,
            config_path=CONFIG_FILE_PATH
        )
        for i in range(len(chunks)):
            X, y = chunks[i]
            if not y: continue
            chunks[i] = (X, [int(label) for label in y])
        
        train_data, train_target = chunks[0]
    except Exception as e:
        logger.error(f"Falha ao carregar os dados: {e}", exc_info=True); exit()

    TARGET_CLASS_TO_EXTRACT = 4
    logger.info(f"Criando dataset focado para a Classe '{TARGET_CLASS_TO_EXTRACT}'...")

    positive_indices = [i for i, label in enumerate(train_target) if label == TARGET_CLASS_TO_EXTRACT]
    negative_indices = [i for i, label in enumerate(train_target) if label != TARGET_CLASS_TO_EXTRACT]

    if not positive_indices:
        logger.error(f"Nenhuma instância da Classe {TARGET_CLASS_TO_EXTRACT} encontrada. Abortando."); exit()
        
    sample_negative_indices = random.sample(negative_indices, min(len(positive_indices) * 3, len(negative_indices)))
    focused_indices = positive_indices + sample_negative_indices
    random.shuffle(focused_indices)

    X_focused = [train_data[i] for i in focused_indices]
    y_focused = [1 if train_target[i] == TARGET_CLASS_TO_EXTRACT else 0 for i in focused_indices]
    
    X_df_focused = pd.DataFrame(X_focused).apply(pd.to_numeric, errors='coerce')
    X_df_focused.fillna(X_df_focused.median(), inplace=True)

    logger.info(f"Treinando modelo RF especialista...")
    rf_guide = RandomForestClassifier(**RF_PARAMS)
    rf_guide.fit(X_df_focused, y_focused)
    logger.info("FASE 1 CONCLUÍDA. Modelo especialista 'rf_guide' treinado.")


    # --- FASE 2: EXTRAÇÃO E ANÁLISE DA IMPORTÂNCIA DAS FEATURES ---
    print("\n" + "="*80)
    logger.info("INICIANDO FASE 2: ANÁLISE DE IMPORTÂNCIA DAS FEATURES")
    print("="*80)

    try:
        # 1. Extrai as importâncias e os nomes das features do modelo treinado
        importances = rf_guide.feature_importances_
        feature_names = X_df_focused.columns

        # 2. Cria uma Série do Pandas para associar os nomes às suas importâncias
        feature_importance_series = pd.Series(importances, index=feature_names)

        # 3. Ordena a série em ordem decrescente para ver as mais importantes primeiro
        sorted_importances = feature_importance_series.sort_values(ascending=False)

        logger.info(f"Extração de importância concluída com sucesso!")
        print("\nTop 10 Features Mais Importantes para a Classe 4:\n")
        print(sorted_importances.head(10))

    except Exception as e:
        logger.error(f"Falha ao extrair a importância das features: {e}", exc_info=True)

    print("\n" + "="*80)
    logger.info("SCRIPT DE VALIDAÇÃO CONCLUÍDO!")
    print("="*80)