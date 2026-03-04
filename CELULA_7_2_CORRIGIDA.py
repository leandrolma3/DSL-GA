# =============================================================================
# CELULA 7.2 CORRIGIDA: Funcoes Auxiliares
# =============================================================================
# CORRECAO: Converter valores booleanos (True/False) para inteiros (0/1)
#           antes de criar o arquivo ARFF
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CELULA 7.2 CORRIGIDA: FUNCOES AUXILIARES")
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

    gmean = np.prod(recalls) ** (1.0 / len(recalls))
    return float(gmean)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula todas as metricas: G-Mean, F1, F1-weighted, Accuracy.
    """
    metrics = {}

    metrics['gmean'] = calculate_gmean(y_true, y_pred)

    try:
        metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    except:
        metrics['f1'] = 0.0
        metrics['f1_weighted'] = 0.0

    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
    except:
        metrics['accuracy'] = 0.0

    return metrics


def load_chunks_from_csv(dataset_path, chunk_size_name: str) -> Tuple[pd.DataFrame, int]:
    """
    Carrega todos os chunks CSV de um dataset e concatena.
    CORRECAO: Converte coluna de classe para inteiros.
    """
    from pathlib import Path
    dataset_path = Path(dataset_path)

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
    original_dtype = combined[class_col].dtype

    # Se for booleano, converter para int
    if combined[class_col].dtype == bool or str(combined[class_col].dtype) == 'bool':
        combined[class_col] = combined[class_col].astype(int)
        print(f"  [CONVERTIDO] Classe: bool -> int")

    # Se for objeto/string com True/False, converter
    elif combined[class_col].dtype == object:
        # Tentar converter strings 'True'/'False'
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


def create_arff_from_dataframe(df: pd.DataFrame, arff_path, relation_name: str) -> bool:
    """
    Converte DataFrame para formato ARFF.
    CORRECAO: Garante que valores de classe sao inteiros.
    """
    from pathlib import Path
    arff_path = Path(arff_path)

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

            # Atributos (exceto classe)
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


def parse_cdcms_output(output_file) -> Optional[pd.DataFrame]:
    """
    Parseia o arquivo de saida do CDCMSEvaluator.
    Formato: instance,accuracy,prediction,actual
    """
    from pathlib import Path
    output_file = Path(output_file)

    try:
        df = pd.read_csv(output_file)
        required_cols = ['instance', 'prediction', 'actual']

        for col in required_cols:
            if col not in df.columns:
                print(f"[ERRO] Coluna '{col}' nao encontrada")
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
    """
    total_instances = len(predictions_df)
    num_chunks = total_instances // chunk_size

    results = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size

        chunk_data = predictions_df.iloc[start_idx:end_idx]
        y_true = chunk_data['actual'].values.astype(int)
        y_pred = chunk_data['prediction'].values.astype(int)

        prequential_metrics = calculate_all_metrics(y_true, y_pred)

        chunk_result = {
            'chunk': chunk_idx,
            'instances_in_chunk': len(chunk_data),
            'prequential_gmean': prequential_metrics['gmean'],
            'prequential_f1': prequential_metrics['f1'],
            'prequential_f1_weighted': prequential_metrics['f1_weighted'],
            'prequential_accuracy': prequential_metrics['accuracy'],
        }

        if chunk_idx >= 1:
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
            chunk_result['holdout_gmean'] = None
            chunk_result['holdout_f1'] = None
            chunk_result['holdout_f1_weighted'] = None
            chunk_result['holdout_accuracy'] = None
            chunk_result['holdout_window_size'] = 0
            chunk_result['note'] = 'Chunk 0 - modelo iniciando do zero'

        results.append(chunk_result)

    return results


print("[OK] Funcoes auxiliares CORRIGIDAS carregadas!")
print()
print("CORRECOES APLICADAS:")
print("  - load_chunks_from_csv: Converte booleanos para int")
print("  - create_arff_from_dataframe: Garante classe como int no ARFF")
print()
print("Funcoes disponiveis:")
print("  - calculate_gmean(y_true, y_pred)")
print("  - calculate_all_metrics(y_true, y_pred)")
print("  - load_chunks_from_csv(dataset_path, chunk_size_name)")
print("  - create_arff_from_dataframe(df, arff_path, relation_name)")
print("  - parse_cdcms_output(output_file)")
print("  - calculate_metrics_per_chunk(predictions_df, chunk_size, holdout_window)")
