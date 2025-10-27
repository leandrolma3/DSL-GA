# metrics.py
#
# OBJETIVO:
# Módulo centralizado para todos os cálculos de métricas de performance.
# Fornece funções eficientes para métricas individuais e uma função
# de conveniência para obter um relatório completo.
# Inclui uma implementação robusta do G-mean contextual para dados desbalanceados.

import numpy as np
import logging
from sklearn.metrics import accuracy_score, f1_score, recall_score
from imblearn.metrics import geometric_mean_score

logger = logging.getLogger(__name__)

def calculate_gmean_contextual(y_true: list, y_pred: list, classes: list) -> float:
    """
    Calcula o G-mean de forma contextual e robusta.

    Esta função calcula o G-mean apenas para as classes que estão efetivamente
    presentes no conjunto de rótulos verdadeiros (`y_true`), evitando o
    'UndefinedMetricWarning' para classes ausentes em um chunk.

    Args:
        y_true (list): A lista de rótulos verdadeiros.
        y_pred (list): A lista de rótulos previstos.
        classes (list): A lista de todas as classes possíveis no dataset.

    Returns:
        float: O valor do G-mean calculado.
    """
    # 1. Identifica as classes únicas que estão PRESENTES neste chunk.
    present_classes = np.unique(y_true)

    # 2. Se houver apenas uma classe presente no chunk, o G-mean não é informativo.
    #    Neste caso, a acurácia é uma métrica de fallback mais estável.
    if len(present_classes) < 2:
        return float(accuracy_score(y_true, y_pred)) # type: ignore
    
    # 3. Se houver múltiplas classes, calcula o G-mean apenas sobre as classes presentes.
    #    Isso evita a divisão por zero para classes ausentes e torna a métrica mais justa.
    try:
        # Usamos 'labels=present_classes' para focar o cálculo.
        g_mean = geometric_mean_score(y_true, y_pred, labels=present_classes, average='multiclass')
        return g_mean
    except Exception as e:
        logger.error(f"Error calculating contextual G-mean: {e}", exc_info=True)
        return 0.0

def calculate_accuracy(y_true: list, y_pred: list) -> float:
    """Calcula a acurácia geral."""
    try:
        return float(accuracy_score(y_true, y_pred)) # type: ignore
    except Exception as e:
        logger.error(f"Error calculating accuracy: {e}", exc_info=True)
        return 0.0

def calculate_f1_weighted(y_true: list, y_pred: list) -> float:
    """Calcula o F1-Score ponderado, ideal para dados desbalanceados."""
    try:
        return float(f1_score(y_true, y_pred, average='weighted', zero_division=0)) # type: ignore
    except Exception as e:
        logger.error(f"Error calculating F1-score: {e}", exc_info=True)
        return 0.0

# --- Função de Conveniência ---

def calculate_all_metrics(y_true: list, y_pred: list, classes: list) -> dict:
    """
    Calcula e retorna um dicionário com todas as principais métricas de performance.
    Ideal para relatórios no final do processamento de um chunk.

    Args:
        y_true (list): A lista de rótulos verdadeiros.
        y_pred (list): A lista de rótulos previstos.
        classes (list): A lista de todas as classes possíveis no dataset.

    Returns:
        dict: Um dicionário contendo as métricas calculadas.
    """
    if not y_true or not y_pred:
        return {
            'g_mean': 0.0,
            'accuracy': 0.0,
            'f1_weighted': 0.0
        }

    results = {
        'g_mean': calculate_gmean_contextual(y_true, y_pred, classes),
        'accuracy': calculate_accuracy(y_true, y_pred),
        'f1_weighted': calculate_f1_weighted(y_true, y_pred)
    }
    
    return results