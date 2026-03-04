# =============================================================================
# CDCMS PARTE 7 COMPLETA - Execucao nos Experimentos chunk_500 e chunk_1000
# =============================================================================
# Este arquivo contem todas as celulas da Parte 7 para executar no Google Colab
# Copie cada secao (CELULA 7.X) para uma celula separada no notebook
#
# Data: 2026-01-26
# Versao: 2.2 - Com metricas prequential e holdout
#              FIX: Conversao de boolean/string para int na coluna de classe
#              FIX: Detecta e marca datasets multiclasse como N/A
#                   (CDCMS.CIL suporta apenas classificacao binaria)
# =============================================================================


# =============================================================================
# CELULA 7.1: Setup e Constantes
# =============================================================================
# Execute esta celula primeiro para configurar todos os paths e constantes

from pathlib import Path
import json
import os

print("="*70)
print("CELULA 7.1: SETUP E CONSTANTES")
print("="*70)

# -----------------------------------------------------------------------------
# 1. Montar Google Drive
# -----------------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# -----------------------------------------------------------------------------
# 2. Paths Principais
# -----------------------------------------------------------------------------
# Path base no Drive (AJUSTE SE NECESSARIO)
DRIVE_BASE = Path('/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid')

# Verificar se existe
if DRIVE_BASE.exists():
    print(f"[OK] DRIVE_BASE: {DRIVE_BASE}")
else:
    print(f"[ERRO] Path nao encontrado: {DRIVE_BASE}")
    # Tentar alternativas
    alternatives = [
        Path('/content/drive/MyDrive/DSL-AG-hybrid'),
        Path('/content/drive/Shareddrives/DSL-AG-hybrid'),
    ]
    for alt in alternatives:
        if alt.exists():
            DRIVE_BASE = alt
            print(f"[OK] Usando alternativa: {DRIVE_BASE}")
            break

# Diretorios de dados
UNIFIED_CHUNKS_DIR = DRIVE_BASE / 'unified_chunks'
EXPERIMENTS_DIR = DRIVE_BASE / 'experiments_unified'

# Diretorios de trabalho no Colab (temporarios)
WORK_DIR = Path('/content')
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
ROSE_JARS_DIR = WORK_DIR / 'rose_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'
TEMP_ARFF_DIR = WORK_DIR / 'cdcms_temp_arff'

# Criar diretorios de trabalho
for d in [CDCMS_JARS_DIR, ROSE_JARS_DIR, TEST_DIR, TEMP_ARFF_DIR]:
    d.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 3. JARs do CDCMS (devem existir das celulas anteriores)
# -----------------------------------------------------------------------------
CDCMS_JAR = CDCMS_JARS_DIR / 'cdcms_cil_final.jar'
MOA_DEPS_JAR = ROSE_JARS_DIR / 'MOA-dependencies.jar'

print(f"\n[INFO] Verificando JARs...")
if CDCMS_JAR.exists():
    print(f"[OK] CDCMS_JAR: {CDCMS_JAR.name} ({CDCMS_JAR.stat().st_size/(1024*1024):.1f} MB)")
else:
    print(f"[AVISO] CDCMS_JAR nao encontrado: {CDCMS_JAR}")
    print("        Execute as celulas 4.1 e 4.2 primeiro!")

if MOA_DEPS_JAR.exists():
    print(f"[OK] MOA_DEPS_JAR: {MOA_DEPS_JAR.name} ({MOA_DEPS_JAR.stat().st_size/(1024*1024):.1f} MB)")
else:
    print(f"[AVISO] MOA_DEPS_JAR nao encontrado: {MOA_DEPS_JAR}")
    print("        Execute a celula 3.3 primeiro!")

# -----------------------------------------------------------------------------
# 4. Configuracoes do Experimento
# -----------------------------------------------------------------------------
# Tamanhos de chunk disponiveis
CHUNK_SIZES = {
    'chunk_500': {'size': 500, 'num_chunks': 24},
    'chunk_1000': {'size': 1000, 'num_chunks': 12},
    # 'chunk_2000': {'size': 2000, 'num_chunks': 6},  # Ainda nao executado
}

# Batches por chunk_size
BATCHES = {
    'chunk_500': ['batch_1', 'batch_2', 'batch_3'],
    'chunk_1000': ['batch_1', 'batch_2', 'batch_3', 'batch_4'],
}

# Janela de holdout para comparacao justa com EGIS
HOLDOUT_WINDOW_SIZE = 100  # Primeiras 100 instancias de cada chunk

# Timeout para execucao do CDCMS (segundos)
CDCMS_TIMEOUT = 600  # 10 minutos

# Datasets a excluir (problematicos)
EXCLUDE_DATASETS = ['IntelLabSensors', 'PokerHand']  # NaN ou muito lentos

# -----------------------------------------------------------------------------
# IMPORTANTE: Datasets MULTICLASSE (CDCMS suporta apenas binario)
# -----------------------------------------------------------------------------
# Referencia: Paper "The Value of Diversity for Dealing with Concept Drift
#             in Class-Imbalanced Data Streams" (Chiu & Minku, IEEE DSAA 2025)
# Citacao: "Covtype and INSECTS were originally multi-class problems. They have
#          been adapted into several versions of binary classification problems"
# -----------------------------------------------------------------------------
MULTICLASS_DATASETS = {
    # LED - 10 classes (digitos 0-9)
    'LED_Abrupt_Simple': 10,
    'LED_Gradual_Simple': 10,
    'LED_Stationary': 10,
    # WAVEFORM - 3 classes
    'WAVEFORM_Abrupt_Simple': 3,
    'WAVEFORM_Gradual_Simple': 3,
    'WAVEFORM_Stationary': 3,
    # CovType - 7 classes (forest cover types)
    'CovType': 7,
    # Shuttle - 7 classes
    'Shuttle': 7,
    # RBF_Stationary - 4 classes
    'RBF_Stationary': 4,
}

def is_multiclass_dataset(dataset_name: str) -> bool:
    """Verifica se dataset e multiclasse (nao suportado pelo CDCMS)."""
    return dataset_name in MULTICLASS_DATASETS

print(f"\n[INFO] Configuracoes:")
print(f"  HOLDOUT_WINDOW_SIZE: {HOLDOUT_WINDOW_SIZE}")
print(f"  CDCMS_TIMEOUT: {CDCMS_TIMEOUT}s")
print(f"  EXCLUDE_DATASETS: {EXCLUDE_DATASETS}")
print(f"  MULTICLASS_DATASETS: {len(MULTICLASS_DATASETS)} (CDCMS suporta apenas binario)")

# -----------------------------------------------------------------------------
# 5. Verificar estrutura de dados
# -----------------------------------------------------------------------------
print(f"\n[INFO] Verificando estrutura de dados...")

for chunk_name, config in CHUNK_SIZES.items():
    chunk_dir = UNIFIED_CHUNKS_DIR / chunk_name
    if chunk_dir.exists():
        datasets = [d.name for d in chunk_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        print(f"[OK] {chunk_name}: {len(datasets)} datasets")
    else:
        print(f"[X] {chunk_name}: nao encontrado")

print(f"\n[INFO] Verificando experimentos existentes (EGIS)...")
for chunk_name in ['chunk_500', 'chunk_1000']:
    exp_dir = EXPERIMENTS_DIR / chunk_name
    if exp_dir.exists():
        batches = [d.name for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith('batch')]
        print(f"[OK] {chunk_name}: {len(batches)} batches")
    else:
        print(f"[X] {chunk_name}: nao encontrado")

print("\n" + "="*70)
print("[OK] Setup concluido!")
print("="*70)


# =============================================================================
# CELULA 7.2: Funcoes Auxiliares
# =============================================================================
# Funcoes para conversao de dados e calculo de metricas

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CELULA 7.2: FUNCOES AUXILIARES")
print("="*70)

def calculate_gmean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula G-Mean (media geometrica dos recalls por classe).

    G-Mean = sqrt(Recall_0 * Recall_1 * ... * Recall_n)
    """
    classes = np.unique(y_true)
    recalls = []

    for cls in classes:
        mask = (y_true == cls)
        if mask.sum() == 0:
            continue
        recall = (y_pred[mask] == cls).sum() / mask.sum()
        recalls.append(recall)

    if len(recalls) == 0:
        return 0.0

    # Media geometrica
    gmean = np.prod(recalls) ** (1.0 / len(recalls))
    return float(gmean)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula todas as metricas: G-Mean, F1, F1-weighted, Accuracy.
    """
    metrics = {}

    # G-Mean
    metrics['gmean'] = calculate_gmean(y_true, y_pred)

    # F1-Score (macro para binario, weighted para multiclasse)
    try:
        metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    except:
        metrics['f1'] = 0.0
        metrics['f1_weighted'] = 0.0

    # Accuracy
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
    except:
        metrics['accuracy'] = 0.0

    return metrics


def load_chunks_from_csv(dataset_path: Path, chunk_size_name: str) -> Tuple[pd.DataFrame, int]:
    """
    Carrega todos os chunks CSV de um dataset e concatena.
    CORRECAO: Converte coluna de classe para inteiros.

    Returns:
        Tuple[DataFrame, num_chunks]
    """
    chunks = sorted(dataset_path.glob("chunk_*.csv"),
                   key=lambda x: int(x.stem.split('_')[1]))

    if not chunks:
        return None, 0

    all_data = []
    for chunk_file in chunks:
        df = pd.read_csv(chunk_file)
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    # =========================================================================
    # CORRECAO: Converter coluna de classe para inteiros
    # =========================================================================
    class_col = combined.columns[-1]

    # Se for booleano, converter para int
    if combined[class_col].dtype == bool or str(combined[class_col].dtype) == 'bool':
        combined[class_col] = combined[class_col].astype(int)
        print(f"  [CONVERTIDO] Classe: bool -> int")

    # Se for objeto/string com True/False, converter
    elif combined[class_col].dtype == object:
        try:
            combined[class_col] = combined[class_col].map(
                lambda x: 1 if str(x).lower() in ['true', '1', '1.0'] else 0
            )
            print(f"  [CONVERTIDO] Classe: string -> int")
        except:
            pass

    # Se for float, converter para int
    elif 'float' in str(combined[class_col].dtype):
        combined[class_col] = combined[class_col].astype(int)
        print(f"  [CONVERTIDO] Classe: float -> int")

    return combined, len(chunks)


def create_arff_from_dataframe(df: pd.DataFrame, arff_path: Path, relation_name: str) -> bool:
    """
    Converte DataFrame para formato ARFF.
    CORRECAO: Garante que valores de classe sao inteiros.
    """
    try:
        # Fazer copia para nao modificar original
        df = df.copy()

        # =====================================================================
        # CORRECAO: Garantir que classe e inteiro
        # =====================================================================
        class_col = df.columns[-1]

        # Converter booleanos
        if df[class_col].dtype == bool or str(df[class_col].dtype) == 'bool':
            df[class_col] = df[class_col].astype(int)

        # Converter strings True/False
        elif df[class_col].dtype == object:
            df[class_col] = df[class_col].map(
                lambda x: 1 if str(x).lower() in ['true', '1', '1.0'] else 0
            )

        # Converter float para int
        elif 'float' in str(df[class_col].dtype):
            df[class_col] = df[class_col].astype(int)

        # Obter classes unicas (agora como inteiros)
        unique_classes = sorted(df[class_col].unique())

        with open(arff_path, 'w') as f:
            f.write(f"@relation {relation_name}\n\n")

            # Atributos (todas colunas exceto a ultima = classe)
            for col in df.columns[:-1]:
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    f.write(f"@attribute {col} numeric\n")
                elif df[col].dtype == bool:
                    f.write(f"@attribute {col} {{0,1}}\n")
                else:
                    unique_vals = sorted(df[col].dropna().unique())
                    vals_str = ",".join(str(v) for v in unique_vals)
                    f.write(f"@attribute {col} {{{vals_str}}}\n")

            # Classe - usar valores inteiros
            class_str = ",".join(str(int(c)) for c in unique_classes)
            f.write(f"@attribute class {{{class_str}}}\n\n")

            # Dados
            f.write("@data\n")
            for _, row in df.iterrows():
                # Converter cada valor, garantindo que classe e int
                values = []
                for i, v in enumerate(row):
                    if i == len(row) - 1:  # Ultima coluna (classe)
                        values.append(str(int(v)))
                    else:
                        values.append(str(v))
                f.write(",".join(values) + "\n")

        return True

    except Exception as e:
        print(f"[ERRO] Criando ARFF: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_cdcms_output(output_file: Path) -> Optional[pd.DataFrame]:
    """
    Parseia o arquivo de saida do CDCMSEvaluator.

    Formato esperado: instance,accuracy,prediction,actual
    """
    try:
        df = pd.read_csv(output_file)
        required_cols = ['instance', 'prediction', 'actual']

        # Verificar colunas
        for col in required_cols:
            if col not in df.columns:
                print(f"[ERRO] Coluna '{col}' nao encontrada no output")
                return None

        return df
    except Exception as e:
        print(f"[ERRO] Parseando output: {e}")
        return None


def calculate_metrics_per_chunk(
    predictions_df: pd.DataFrame,
    chunk_size: int,
    holdout_window: int = 100
) -> List[Dict]:
    """
    Calcula metricas por chunk a partir das predicoes do CDCMS.

    Para cada chunk, calcula:
    - Metricas prequential (chunk completo)
    - Metricas holdout (primeiras N instancias)

    Args:
        predictions_df: DataFrame com colunas [instance, prediction, actual]
        chunk_size: Tamanho do chunk (500 ou 1000)
        holdout_window: Numero de instancias para holdout (default: 100)

    Returns:
        Lista de dicts com metricas por chunk
    """
    total_instances = len(predictions_df)
    num_chunks = total_instances // chunk_size

    results = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size

        # Dados do chunk completo
        chunk_data = predictions_df.iloc[start_idx:end_idx]
        y_true = chunk_data['actual'].values.astype(int)
        y_pred = chunk_data['prediction'].values.astype(int)

        # Metricas prequential (chunk completo)
        prequential_metrics = calculate_all_metrics(y_true, y_pred)

        chunk_result = {
            'chunk': chunk_idx,
            'instances_in_chunk': len(chunk_data),
            # Metricas prequential
            'prequential_gmean': prequential_metrics['gmean'],
            'prequential_f1': prequential_metrics['f1'],
            'prequential_f1_weighted': prequential_metrics['f1_weighted'],
            'prequential_accuracy': prequential_metrics['accuracy'],
        }

        # Metricas holdout (apenas para chunk >= 1)
        # No chunk 0, o modelo esta aprendendo do zero
        if chunk_idx >= 1:
            # Usar primeiras N instancias do chunk (holdout window)
            holdout_end = min(start_idx + holdout_window, end_idx)
            holdout_data = predictions_df.iloc[start_idx:holdout_end]

            y_true_holdout = holdout_data['actual'].values.astype(int)
            y_pred_holdout = holdout_data['prediction'].values.astype(int)

            holdout_metrics = calculate_all_metrics(y_true_holdout, y_pred_holdout)

            chunk_result['holdout_gmean'] = holdout_metrics['gmean']
            chunk_result['holdout_f1'] = holdout_metrics['f1']
            chunk_result['holdout_f1_weighted'] = holdout_metrics['f1_weighted']
            chunk_result['holdout_accuracy'] = holdout_metrics['accuracy']
            chunk_result['holdout_window_size'] = holdout_end - start_idx
        else:
            # Chunk 0 - sem holdout (modelo iniciando)
            chunk_result['holdout_gmean'] = None
            chunk_result['holdout_f1'] = None
            chunk_result['holdout_f1_weighted'] = None
            chunk_result['holdout_accuracy'] = None
            chunk_result['holdout_window_size'] = 0
            chunk_result['note'] = 'Chunk 0 - modelo iniciando do zero'

        results.append(chunk_result)

    return results


print("[OK] Funcoes auxiliares carregadas (v2.1 - com fix boolean->int):")
print("  - calculate_gmean(y_true, y_pred)")
print("  - calculate_all_metrics(y_true, y_pred)")
print("  - load_chunks_from_csv(dataset_path, chunk_size_name) [CORRIGIDO]")
print("  - create_arff_from_dataframe(df, arff_path, relation_name) [CORRIGIDO]")
print("  - parse_cdcms_output(output_file)")
print("  - calculate_metrics_per_chunk(predictions_df, chunk_size, holdout_window)")


# =============================================================================
# CELULA 7.3: Funcao Principal run_cdcms_on_dataset
# =============================================================================
# Funcao principal que executa o CDCMS em um dataset

import subprocess
import time
from datetime import datetime

print("="*70)
print("CELULA 7.3: FUNCAO PRINCIPAL run_cdcms_on_dataset")
print("="*70)

def run_cdcms_on_dataset(
    dataset_name: str,
    chunk_size_name: str,  # 'chunk_500' ou 'chunk_1000'
    batch_name: str,       # 'batch_1', 'batch_2', etc.
    classifier: str = "CDCMS_CIL_GMean",
    timeout: int = None,
    save_results: bool = True,
    verbose: bool = True
) -> Optional[Dict]:
    """
    Executa CDCMS em um dataset e calcula metricas por chunk.

    Args:
        dataset_name: Nome do dataset (ex: 'SEA_Abrupt_Simple')
        chunk_size_name: 'chunk_500' ou 'chunk_1000'
        batch_name: 'batch_1', 'batch_2', etc.
        classifier: 'CDCMS_CIL_GMean' ou 'CDCMS_CIL'
        timeout: Timeout em segundos (default: CDCMS_TIMEOUT)
        save_results: Se deve salvar resultados no formato compativel
        verbose: Se deve imprimir mensagens de progresso

    Returns:
        Dict com resultados ou None se falhar
    """
    if timeout is None:
        timeout = CDCMS_TIMEOUT

    # -------------------------------------------------------------------------
    # 1. Verificar paths
    # -------------------------------------------------------------------------
    chunk_config = CHUNK_SIZES.get(chunk_size_name)
    if chunk_config is None:
        print(f"[ERRO] chunk_size_name invalido: {chunk_size_name}")
        return None

    chunk_size = chunk_config['size']

    # Path dos dados (chunks CSV)
    data_path = UNIFIED_CHUNKS_DIR / chunk_size_name / dataset_name
    if not data_path.exists():
        print(f"[ERRO] Dataset nao encontrado: {data_path}")
        return None

    # Path dos resultados EGIS (para salvar ao lado)
    egis_results_path = EXPERIMENTS_DIR / chunk_size_name / batch_name / dataset_name

    if verbose:
        print(f"\n--- {dataset_name} ({chunk_size_name}/{batch_name}) ---")

    # -------------------------------------------------------------------------
    # 2. Carregar e converter dados
    # -------------------------------------------------------------------------
    df, num_chunks = load_chunks_from_csv(data_path, chunk_size_name)
    if df is None:
        print(f"[ERRO] Falha ao carregar chunks")
        return None

    if verbose:
        print(f"  Instancias: {len(df)} | Chunks: {num_chunks} | Features: {len(df.columns)-1}")

    # Criar ARFF temporario
    arff_file = TEMP_ARFF_DIR / f"{dataset_name}_{chunk_size_name}.arff"
    if not create_arff_from_dataframe(df, arff_file, dataset_name):
        print(f"[ERRO] Falha ao criar ARFF")
        return None

    # -------------------------------------------------------------------------
    # 3. Executar CDCMS
    # -------------------------------------------------------------------------
    output_file = TEST_DIR / f"{dataset_name}_{classifier}_output.csv"

    # Limpar arquivo anterior se existir
    if output_file.exists():
        output_file.unlink()

    # Classpath
    classpath = f"{CDCMS_JAR}:{MOA_DEPS_JAR}"

    # Comando
    cmd = [
        "java", "-Xmx4g",
        "-cp", f"{classpath}:{TEST_DIR}",
        "CDCMSEvaluator",
        str(arff_file),
        str(output_file),
        classifier
    ]

    if verbose:
        print(f"  Executando {classifier}...")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        execution_time = time.time() - start_time

        if result.returncode != 0:
            if verbose:
                print(f"  [ERRO] Return code: {result.returncode}")
                if result.stderr:
                    errors = [l for l in result.stderr.split('\n') if 'Exception' in l or 'Error' in l][:3]
                    for e in errors:
                        print(f"    {e[:80]}")
            return None

    except subprocess.TimeoutExpired:
        if verbose:
            print(f"  [TIMEOUT] Excedeu {timeout}s")
        return None
    except Exception as e:
        if verbose:
            print(f"  [ERRO] {e}")
        return None

    # -------------------------------------------------------------------------
    # 4. Parsear output e calcular metricas
    # -------------------------------------------------------------------------
    if not output_file.exists() or output_file.stat().st_size == 0:
        if verbose:
            print(f"  [ERRO] Arquivo de saida vazio ou inexistente")
        return None

    predictions_df = parse_cdcms_output(output_file)
    if predictions_df is None:
        return None

    # Calcular metricas por chunk
    chunk_metrics = calculate_metrics_per_chunk(
        predictions_df,
        chunk_size,
        HOLDOUT_WINDOW_SIZE
    )

    # -------------------------------------------------------------------------
    # 5. Calcular medias gerais
    # -------------------------------------------------------------------------
    # Media prequential (todos os chunks)
    avg_prequential_gmean = np.mean([m['prequential_gmean'] for m in chunk_metrics])
    avg_prequential_f1 = np.mean([m['prequential_f1'] for m in chunk_metrics])

    # Media holdout (chunks >= 1 apenas)
    holdout_metrics = [m for m in chunk_metrics if m['holdout_gmean'] is not None]
    if holdout_metrics:
        avg_holdout_gmean = np.mean([m['holdout_gmean'] for m in holdout_metrics])
        avg_holdout_f1 = np.mean([m['holdout_f1'] for m in holdout_metrics])
    else:
        avg_holdout_gmean = None
        avg_holdout_f1 = None

    if verbose:
        print(f"  [OK] Tempo: {execution_time:.1f}s")
        print(f"       Prequential G-Mean: {avg_prequential_gmean:.4f}")
        if avg_holdout_gmean is not None:
            print(f"       Holdout G-Mean:     {avg_holdout_gmean:.4f}")

    # -------------------------------------------------------------------------
    # 6. Preparar resultado
    # -------------------------------------------------------------------------
    result_data = {
        'dataset': dataset_name,
        'chunk_size_name': chunk_size_name,
        'chunk_size': chunk_size,
        'batch': batch_name,
        'classifier': classifier,
        'total_instances': len(df),
        'num_chunks': len(chunk_metrics),
        'num_features': len(df.columns) - 1,
        'num_classes': len(df[df.columns[-1]].unique()),
        'execution_time_seconds': execution_time,
        'holdout_window_size': HOLDOUT_WINDOW_SIZE,
        # Medias
        'avg_prequential_gmean': avg_prequential_gmean,
        'avg_prequential_f1': avg_prequential_f1,
        'avg_holdout_gmean': avg_holdout_gmean,
        'avg_holdout_f1': avg_holdout_f1,
        # Metricas por chunk
        'chunk_metrics': chunk_metrics,
        # Timestamp
        'executed_at': datetime.now().isoformat()
    }

    # -------------------------------------------------------------------------
    # 7. Salvar resultados (se solicitado)
    # -------------------------------------------------------------------------
    if save_results and egis_results_path.exists():
        cdcms_results_dir = egis_results_path / 'cdcms_results'
        cdcms_results_dir.mkdir(exist_ok=True)

        # Salvar chunk_metrics.json
        metrics_file = cdcms_results_dir / 'chunk_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(chunk_metrics, f, indent=2)

        # Salvar run_config.json
        config_data = {
            'classifier': classifier,
            'chunk_size': chunk_size,
            'chunk_size_name': chunk_size_name,
            'batch': batch_name,
            'holdout_window_size': HOLDOUT_WINDOW_SIZE,
            'total_instances': len(df),
            'num_chunks': len(chunk_metrics),
            'execution_time_seconds': execution_time,
            'executed_at': datetime.now().isoformat(),
            'avg_prequential_gmean': avg_prequential_gmean,
            'avg_holdout_gmean': avg_holdout_gmean
        }
        config_file = cdcms_results_dir / 'run_config.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

        # Copiar output bruto
        import shutil
        raw_output_file = cdcms_results_dir / 'cdcms_raw_output.csv'
        shutil.copy(output_file, raw_output_file)

        if verbose:
            print(f"  [SALVO] {cdcms_results_dir}")

    return result_data


print("[OK] Funcao run_cdcms_on_dataset() definida!")
print("\nExemplo de uso:")
print('  result = run_cdcms_on_dataset("SEA_Abrupt_Simple", "chunk_500", "batch_1")')


# =============================================================================
# CELULA 7.4: Funcoes de Execucao em Lote
# =============================================================================
# Funcoes para executar CDCMS em multiplos datasets

print("="*70)
print("CELULA 7.4: FUNCOES DE EXECUCAO EM LOTE")
print("="*70)

def get_datasets_for_batch(chunk_size_name: str, batch_name: str) -> List[str]:
    """
    Retorna lista de datasets disponiveis para um batch.
    Baseado nos experimentos EGIS existentes.
    """
    batch_path = EXPERIMENTS_DIR / chunk_size_name / batch_name

    if not batch_path.exists():
        print(f"[AVISO] Batch nao encontrado: {batch_path}")
        return []

    datasets = []
    for d in batch_path.iterdir():
        if d.is_dir() and not d.name.startswith('.') and d.name not in ['desktop.ini']:
            # Verificar se tem resultados EGIS (run_1)
            run_dir = d / 'run_1'
            if run_dir.exists():
                datasets.append(d.name)

    # Filtrar datasets problematicos
    datasets = [d for d in datasets if d not in EXCLUDE_DATASETS]

    return sorted(datasets)


def run_cdcms_batch(
    chunk_size_name: str,
    batch_name: str,
    max_datasets: int = None,
    skip_existing: bool = True
) -> pd.DataFrame:
    """
    Executa CDCMS em todos os datasets de um batch.

    Args:
        chunk_size_name: 'chunk_500' ou 'chunk_1000'
        batch_name: 'batch_1', 'batch_2', etc.
        max_datasets: Limite de datasets (None = todos)
        skip_existing: Pular datasets que ja tem resultados CDCMS

    Returns:
        DataFrame com resultados (incluindo N/A para multiclasse)
    """
    print("="*70)
    print(f"EXECUTAR CDCMS: {chunk_size_name} / {batch_name}")
    print("="*70)

    datasets = get_datasets_for_batch(chunk_size_name, batch_name)

    if max_datasets:
        datasets = datasets[:max_datasets]

    print(f"Datasets a processar: {len(datasets)}")

    results = []
    skipped = 0
    failed = 0
    multiclass_skipped = 0

    for i, dataset in enumerate(datasets, 1):
        # =====================================================================
        # VERIFICAR SE E MULTICLASSE (CDCMS suporta apenas binario)
        # =====================================================================
        if is_multiclass_dataset(dataset):
            num_classes = MULTICLASS_DATASETS[dataset]
            print(f"[{i}/{len(datasets)}] {dataset} - N/A (multiclasse: {num_classes} classes)")

            # Registrar como N/A
            results.append({
                'dataset': dataset,
                'chunk_size': chunk_size_name,
                'batch': batch_name,
                'status': 'N/A',
                'prequential_gmean': None,
                'holdout_gmean': None,
                'prequential_f1': None,
                'holdout_f1': None,
                'time_seconds': 0,
                'num_chunks': 0,
                'note': f'Multiclass ({num_classes} classes) - CDCMS binary only'
            })
            multiclass_skipped += 1
            continue

        # Verificar se ja existe
        if skip_existing:
            cdcms_dir = EXPERIMENTS_DIR / chunk_size_name / batch_name / dataset / 'cdcms_results'
            if cdcms_dir.exists() and (cdcms_dir / 'chunk_metrics.json').exists():
                print(f"[{i}/{len(datasets)}] {dataset} - SKIP (ja existe)")
                skipped += 1
                continue

        print(f"[{i}/{len(datasets)}]", end="")
        result = run_cdcms_on_dataset(dataset, chunk_size_name, batch_name)

        if result:
            results.append({
                'dataset': dataset,
                'chunk_size': chunk_size_name,
                'batch': batch_name,
                'status': 'OK',
                'prequential_gmean': result['avg_prequential_gmean'],
                'holdout_gmean': result['avg_holdout_gmean'],
                'prequential_f1': result['avg_prequential_f1'],
                'holdout_f1': result['avg_holdout_f1'],
                'time_seconds': result['execution_time_seconds'],
                'num_chunks': result['num_chunks'],
                'note': ''
            })
        else:
            failed += 1

    print("\n" + "="*70)
    print(f"RESUMO: {chunk_size_name} / {batch_name}")
    print("="*70)
    print(f"  Processados com sucesso: {len([r for r in results if r.get('status') == 'OK'])}")
    print(f"  Pulados (existentes): {skipped}")
    print(f"  Multiclasse (N/A): {multiclass_skipped}")
    print(f"  Falhas: {failed}")

    if results:
        df = pd.DataFrame(results)

        # Filtrar apenas resultados validos (excluir N/A) para calcular medias
        valid_df = df[df['status'] == 'OK']

        if not valid_df.empty:
            print(f"\n  Media Prequential G-Mean: {valid_df['prequential_gmean'].mean():.4f}")
            if valid_df['holdout_gmean'].notna().any():
                print(f"  Media Holdout G-Mean:     {valid_df['holdout_gmean'].mean():.4f}")

        return df

    return pd.DataFrame()


def run_cdcms_all_batches(chunk_size_name: str, skip_existing: bool = True) -> pd.DataFrame:
    """
    Executa CDCMS em todos os batches de um chunk_size.
    """
    all_results = []

    batches = BATCHES.get(chunk_size_name, [])

    for batch in batches:
        df = run_cdcms_batch(chunk_size_name, batch, skip_existing=skip_existing)
        if not df.empty:
            all_results.append(df)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


print("[OK] Funcoes de execucao em lote carregadas:")
print("  - get_datasets_for_batch(chunk_size_name, batch_name)")
print("  - run_cdcms_batch(chunk_size_name, batch_name)")
print("  - run_cdcms_all_batches(chunk_size_name)")


# =============================================================================
# CELULA 7.5: Teste Rapido (3 datasets)
# =============================================================================
# Executar em alguns datasets para validar que tudo funciona

print("="*70)
print("CELULA 7.5: TESTE RAPIDO")
print("="*70)

# Datasets de teste (representativos)
TEST_DATASETS = [
    ('SEA_Abrupt_Simple', 'chunk_500', 'batch_1'),
    ('HYPERPLANE_Abrupt_Simple', 'chunk_500', 'batch_1'),
    ('AGRAWAL_Abrupt_Simple_Mild', 'chunk_500', 'batch_1'),
]

print(f"Executando teste em {len(TEST_DATASETS)} datasets...")
print()

test_results = []

for dataset, chunk_size, batch in TEST_DATASETS:
    result = run_cdcms_on_dataset(
        dataset,
        chunk_size,
        batch,
        save_results=True,  # Salvar para validacao
        verbose=True
    )

    if result:
        test_results.append({
            'dataset': dataset,
            'prequential_gmean': result['avg_prequential_gmean'],
            'holdout_gmean': result['avg_holdout_gmean'],
            'time': result['execution_time_seconds']
        })

if test_results:
    print("\n" + "="*70)
    print("RESULTADOS DO TESTE")
    print("="*70)
    test_df = pd.DataFrame(test_results)
    print(test_df.to_string(index=False))
    print("\n[OK] Teste concluido com sucesso!")
else:
    print("\n[ERRO] Nenhum resultado obtido. Verifique:")
    print("  1. JARs do CDCMS (celulas 3 e 4)")
    print("  2. CDCMSEvaluator compilado (celula 5)")
    print("  3. Paths dos dados")


# =============================================================================
# CELULA 7.6: Executar em chunk_500 (Todos os Batches)
# =============================================================================
# AVISO: Isso pode demorar bastante (~85 datasets)

print("="*70)
print("CELULA 7.6: EXECUTAR CDCMS EM chunk_500")
print("="*70)

# Descomente para executar:
# results_chunk500 = run_cdcms_all_batches('chunk_500', skip_existing=True)

# Ou execute batch por batch:
# results_b1 = run_cdcms_batch('chunk_500', 'batch_1')
# results_b2 = run_cdcms_batch('chunk_500', 'batch_2')
# results_b3 = run_cdcms_batch('chunk_500', 'batch_3')

print("Para executar, descomente uma das opcoes acima.")
print()
print("Opcao 1 - Todos os batches:")
print("  results_chunk500 = run_cdcms_all_batches('chunk_500', skip_existing=True)")
print()
print("Opcao 2 - Batch por batch:")
print("  results_b1 = run_cdcms_batch('chunk_500', 'batch_1')")


# =============================================================================
# CELULA 7.7: Executar em chunk_1000 (Todos os Batches)
# =============================================================================
# AVISO: Isso pode demorar bastante (~93 datasets)

print("="*70)
print("CELULA 7.7: EXECUTAR CDCMS EM chunk_1000")
print("="*70)

# Descomente para executar:
# results_chunk1000 = run_cdcms_all_batches('chunk_1000', skip_existing=True)

# Ou execute batch por batch:
# results_b1 = run_cdcms_batch('chunk_1000', 'batch_1')
# results_b2 = run_cdcms_batch('chunk_1000', 'batch_2')
# results_b3 = run_cdcms_batch('chunk_1000', 'batch_3')
# results_b4 = run_cdcms_batch('chunk_1000', 'batch_4')

print("Para executar, descomente uma das opcoes acima.")
print()
print("Opcao 1 - Todos os batches:")
print("  results_chunk1000 = run_cdcms_all_batches('chunk_1000', skip_existing=True)")
print()
print("Opcao 2 - Batch por batch:")
print("  results_b1 = run_cdcms_batch('chunk_1000', 'batch_1')")


# =============================================================================
# CELULA 7.8: Consolidar e Comparar com EGIS
# =============================================================================
# Funcoes para consolidar resultados e comparar com EGIS

print("="*70)
print("CELULA 7.8: CONSOLIDAR E COMPARAR COM EGIS")
print("="*70)

def load_egis_results(chunk_size_name: str, batch_name: str, dataset_name: str) -> Optional[Dict]:
    """
    Carrega resultados do EGIS para um dataset.
    """
    metrics_file = EXPERIMENTS_DIR / chunk_size_name / batch_name / dataset_name / 'run_1' / 'chunk_metrics.json'

    if not metrics_file.exists():
        return None

    with open(metrics_file) as f:
        chunk_metrics = json.load(f)

    # Calcular media do test_gmean (chunks >= 1)
    test_gmeans = [m['test_gmean'] for m in chunk_metrics if m.get('test_gmean') is not None]

    if not test_gmeans:
        return None

    return {
        'avg_test_gmean': np.mean(test_gmeans),
        'std_test_gmean': np.std(test_gmeans),
        'num_chunks': len(chunk_metrics)
    }


def load_cdcms_results(chunk_size_name: str, batch_name: str, dataset_name: str) -> Optional[Dict]:
    """
    Carrega resultados do CDCMS para um dataset.
    """
    metrics_file = EXPERIMENTS_DIR / chunk_size_name / batch_name / dataset_name / 'cdcms_results' / 'chunk_metrics.json'

    if not metrics_file.exists():
        return None

    with open(metrics_file) as f:
        chunk_metrics = json.load(f)

    # Metricas prequential
    prequential_gmeans = [m['prequential_gmean'] for m in chunk_metrics]

    # Metricas holdout (chunks >= 1)
    holdout_gmeans = [m['holdout_gmean'] for m in chunk_metrics if m.get('holdout_gmean') is not None]

    return {
        'avg_prequential_gmean': np.mean(prequential_gmeans),
        'avg_holdout_gmean': np.mean(holdout_gmeans) if holdout_gmeans else None,
        'num_chunks': len(chunk_metrics)
    }


def compare_egis_cdcms(chunk_size_name: str) -> pd.DataFrame:
    """
    Compara resultados EGIS vs CDCMS para um chunk_size.
    """
    print(f"\nComparando EGIS vs CDCMS para {chunk_size_name}...")

    results = []
    batches = BATCHES.get(chunk_size_name, [])

    for batch in batches:
        datasets = get_datasets_for_batch(chunk_size_name, batch)

        for dataset in datasets:
            egis = load_egis_results(chunk_size_name, batch, dataset)
            cdcms = load_cdcms_results(chunk_size_name, batch, dataset)

            if egis and cdcms:
                results.append({
                    'dataset': dataset,
                    'batch': batch,
                    'egis_gmean': egis['avg_test_gmean'],
                    'cdcms_prequential': cdcms['avg_prequential_gmean'],
                    'cdcms_holdout': cdcms['avg_holdout_gmean'],
                    'diff_prequential': cdcms['avg_prequential_gmean'] - egis['avg_test_gmean'],
                    'diff_holdout': (cdcms['avg_holdout_gmean'] - egis['avg_test_gmean']) if cdcms['avg_holdout_gmean'] else None
                })

    if results:
        df = pd.DataFrame(results)

        print(f"\nDatasets comparados: {len(df)}")
        print(f"\nMedias:")
        print(f"  EGIS G-Mean:           {df['egis_gmean'].mean():.4f}")
        print(f"  CDCMS Prequential:     {df['cdcms_prequential'].mean():.4f}")
        if df['cdcms_holdout'].notna().any():
            print(f"  CDCMS Holdout:         {df['cdcms_holdout'].mean():.4f}")

        print(f"\nDiferencas (CDCMS - EGIS):")
        print(f"  Prequential: {df['diff_prequential'].mean():+.4f}")
        if df['diff_holdout'].notna().any():
            print(f"  Holdout:     {df['diff_holdout'].mean():+.4f}")

        return df

    return pd.DataFrame()


def generate_comparison_table(chunk_size_name: str) -> None:
    """
    Gera tabela de comparacao formatada.
    """
    df = compare_egis_cdcms(chunk_size_name)

    if df.empty:
        print("Sem dados para comparacao.")
        return

    # Salvar CSV
    output_file = EXPERIMENTS_DIR / chunk_size_name / f'comparison_egis_cdcms_{chunk_size_name}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n[SALVO] {output_file}")

    # Top 5 onde CDCMS e melhor
    print("\nTop 5 - CDCMS melhor que EGIS (holdout):")
    if df['diff_holdout'].notna().any():
        top5_cdcms = df.nlargest(5, 'diff_holdout')[['dataset', 'egis_gmean', 'cdcms_holdout', 'diff_holdout']]
        print(top5_cdcms.to_string(index=False))

    # Top 5 onde EGIS e melhor
    print("\nTop 5 - EGIS melhor que CDCMS (holdout):")
    if df['diff_holdout'].notna().any():
        top5_egis = df.nsmallest(5, 'diff_holdout')[['dataset', 'egis_gmean', 'cdcms_holdout', 'diff_holdout']]
        print(top5_egis.to_string(index=False))


print("[OK] Funcoes de comparacao carregadas:")
print("  - load_egis_results(chunk_size_name, batch_name, dataset_name)")
print("  - load_cdcms_results(chunk_size_name, batch_name, dataset_name)")
print("  - compare_egis_cdcms(chunk_size_name)")
print("  - generate_comparison_table(chunk_size_name)")
print()
print("Para gerar comparacao:")
print("  generate_comparison_table('chunk_500')")
print("  generate_comparison_table('chunk_1000')")


# =============================================================================
# FIM DO ARQUIVO
# =============================================================================
print("\n" + "="*70)
print("CDCMS PARTE 7 COMPLETA - CARREGADA COM SUCESSO!")
print("="*70)
print()
print("Proximos passos:")
print("  1. Execute CELULA 7.5 para teste rapido")
print("  2. Execute CELULA 7.6 para chunk_500")
print("  3. Execute CELULA 7.7 para chunk_1000")
print("  4. Execute CELULA 7.8 para comparar com EGIS")
