"""
Módulo de Conversão entre Formatos de Dados

PROPÓSITO:
    Converter dados entre diferentes formatos usados por GBML, River e ACDWM:
    - River: Lista de dicts [{'x_0': 0.5, 'x_1': 0.3}, ...]
    - NumPy: Arrays (N, features)
    - Pandas: DataFrames

DESIGN:
    - Funções puras (sem efeitos colaterais)
    - Type hints para clareza
    - Validação de entrada
    - Preserva ordem de features

AUTOR: Claude Code
DATA: 2025-01-07
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union

logger = logging.getLogger("data_converters")


# ============================================================================
# CONVERSÃO: RIVER → NUMPY
# ============================================================================

def river_to_numpy(
    X_river: List[Dict[str, float]],
    feature_names: Optional[List[str]] = None,
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, List[str]]:
    """
    Converte lista de dicts (River format) para array NumPy.

    Args:
        X_river: Lista de dicts com features
            Exemplo: [{'x_0': 0.5, 'x_1': 0.3}, {'x_0': 0.2, 'x_1': 0.8}]
        feature_names: Lista ordenada de nomes das features
            Se None, extrai automaticamente do primeiro exemplo (sorted)
        dtype: Tipo de dado do array resultante

    Returns:
        (X_array, feature_names):
            - X_array: Array NumPy de shape (N, n_features)
            - feature_names: Lista de nomes das features usadas

    Raises:
        ValueError: Se X_river vazio ou features inconsistentes

    Example:
        >>> X = [{'x_0': 0.5, 'x_1': 0.3}, {'x_0': 0.2, 'x_1': 0.8}]
        >>> X_array, names = river_to_numpy(X)
        >>> X_array.shape
        (2, 2)
        >>> names
        ['x_0', 'x_1']
    """
    if not X_river:
        raise ValueError("X_river está vazio! Nenhum dado para converter.")

    # Extrai nomes das features se não fornecidos
    if feature_names is None:
        feature_names = sorted(X_river[0].keys())
        logger.debug(f"Features detectadas automaticamente: {feature_names}")

    n_features = len(feature_names)
    n_samples = len(X_river)

    logger.debug(f"Convertendo {n_samples} samples com {n_features} features")

    # Valida que todos os samples têm as mesmas features
    for i, x_dict in enumerate(X_river):
        if set(x_dict.keys()) != set(feature_names):
            missing = set(feature_names) - set(x_dict.keys())
            extra = set(x_dict.keys()) - set(feature_names)
            raise ValueError(
                f"Sample {i} tem features inconsistentes!\n"
                f"  Esperado: {feature_names}\n"
                f"  Recebido: {list(x_dict.keys())}\n"
                f"  Faltando: {missing}\n"
                f"  Extra: {extra}"
            )

    # Converte para array NumPy
    X_array = np.zeros((n_samples, n_features), dtype=dtype)

    for i, x_dict in enumerate(X_river):
        for j, fname in enumerate(feature_names):
            X_array[i, j] = x_dict[fname]

    logger.debug(f"Conversão concluída: shape={X_array.shape}, dtype={X_array.dtype}")

    return X_array, feature_names


def river_labels_to_numpy(
    y_river: List[Union[int, str]],
    dtype: np.dtype = np.int32
) -> np.ndarray:
    """
    Converte lista de labels para array NumPy.

    Args:
        y_river: Lista de labels
        dtype: Tipo de dado do array

    Returns:
        Array NumPy de shape (N,)

    Example:
        >>> y = [0, 1, 1, 0]
        >>> y_array = river_labels_to_numpy(y)
        >>> y_array.shape
        (4,)
    """
    if not y_river:
        raise ValueError("y_river está vazio!")

    y_array = np.array(y_river, dtype=dtype)

    logger.debug(f"Labels convertidos: shape={y_array.shape}, unique={np.unique(y_array)}")

    return y_array


# ============================================================================
# CONVERSÃO: NUMPY → RIVER
# ============================================================================

def numpy_to_river(
    X_array: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> List[Dict[str, float]]:
    """
    Converte array NumPy para lista de dicts (River format).

    Args:
        X_array: Array NumPy de shape (N, n_features)
        feature_names: Lista de nomes das features
            Se None, usa 'x_0', 'x_1', ..., 'x_{n_features-1}'

    Returns:
        Lista de dicts com features

    Example:
        >>> X = np.array([[0.5, 0.3], [0.2, 0.8]])
        >>> X_river = numpy_to_river(X)
        >>> X_river
        [{'x_0': 0.5, 'x_1': 0.3}, {'x_0': 0.2, 'x_1': 0.8}]
    """
    if X_array.ndim != 2:
        raise ValueError(f"X_array deve ter 2 dimensões, recebido: {X_array.ndim}")

    n_samples, n_features = X_array.shape

    # Gera nomes padrão se não fornecidos
    if feature_names is None:
        feature_names = [f'x_{i}' for i in range(n_features)]
        logger.debug(f"Nomes de features gerados: {feature_names}")

    if len(feature_names) != n_features:
        raise ValueError(
            f"Número de feature_names ({len(feature_names)}) "
            f"não corresponde a n_features ({n_features})"
        )

    # Converte para lista de dicts
    X_river = []
    for i in range(n_samples):
        x_dict = {fname: float(X_array[i, j]) for j, fname in enumerate(feature_names)}
        X_river.append(x_dict)

    logger.debug(f"Conversão concluída: {n_samples} samples convertidos")

    return X_river


# ============================================================================
# CONVERSÃO EM LOTE (CHUNKS)
# ============================================================================

def convert_chunks_river_to_numpy(
    chunks: List[Tuple[List[Dict], List]],
    feature_names: Optional[List[str]] = None
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[str]]:
    """
    Converte múltiplos chunks de River para NumPy.

    Args:
        chunks: Lista de (X_train, y_train) ou (X_train, y_train, X_test, y_test)
        feature_names: Lista de nomes das features (opcional)

    Returns:
        (converted_chunks, feature_names):
            - converted_chunks: Lista de tuplas com arrays NumPy
            - feature_names: Lista de nomes das features

    Example:
        >>> chunks = [
        ...     ([{'x_0': 0.5}], [0]),
        ...     ([{'x_0': 0.3}], [1])
        ... ]
        >>> converted, names = convert_chunks_river_to_numpy(chunks)
        >>> len(converted)
        2
    """
    if not chunks:
        raise ValueError("Lista de chunks está vazia!")

    logger.info(f"Convertendo {len(chunks)} chunks...")

    # Detecta feature_names do primeiro chunk se não fornecido
    if feature_names is None:
        first_X = chunks[0][0]  # Primeiro elemento da primeira tupla
        if first_X:
            feature_names = sorted(first_X[0].keys())
            logger.info(f"Features detectadas: {feature_names}")
        else:
            raise ValueError("Primeiro chunk está vazio, não é possível detectar features")

    converted_chunks = []

    for i, chunk_data in enumerate(chunks):
        if len(chunk_data) == 2:
            # (X, y)
            X_river, y_river = chunk_data
            X_array, _ = river_to_numpy(X_river, feature_names)
            y_array = river_labels_to_numpy(y_river)
            converted_chunks.append((X_array, y_array))

        elif len(chunk_data) == 4:
            # (X_train, y_train, X_test, y_test)
            X_train_river, y_train_river, X_test_river, y_test_river = chunk_data

            X_train_array, _ = river_to_numpy(X_train_river, feature_names)
            y_train_array = river_labels_to_numpy(y_train_river)

            X_test_array, _ = river_to_numpy(X_test_river, feature_names)
            y_test_array = river_labels_to_numpy(y_test_river)

            converted_chunks.append((X_train_array, y_train_array, X_test_array, y_test_array))

        else:
            raise ValueError(f"Chunk {i} tem formato inesperado: {len(chunk_data)} elementos")

        logger.debug(f"Chunk {i} convertido")

    logger.info(f"✓ {len(converted_chunks)} chunks convertidos com sucesso")

    return converted_chunks, feature_names


# ============================================================================
# UTILITÁRIOS
# ============================================================================

def validate_river_format(X_river: List[Dict]) -> bool:
    """
    Valida se dados estão no formato River correto.

    Args:
        X_river: Lista de dicts

    Returns:
        True se válido

    Raises:
        ValueError: Se formato inválido
    """
    if not isinstance(X_river, list):
        raise ValueError(f"X_river deve ser lista, recebido: {type(X_river)}")

    if not X_river:
        raise ValueError("X_river está vazio")

    if not isinstance(X_river[0], dict):
        raise ValueError(f"Elementos devem ser dicts, recebido: {type(X_river[0])}")

    # Valida que todos têm as mesmas keys
    keys = set(X_river[0].keys())
    for i, x in enumerate(X_river[1:], 1):
        if set(x.keys()) != keys:
            raise ValueError(f"Sample {i} tem keys diferentes do sample 0")

    logger.debug(f"Formato River válido: {len(X_river)} samples, {len(keys)} features")
    return True


def get_feature_names_from_river(X_river: List[Dict]) -> List[str]:
    """
    Extrai nomes das features de dados River.

    Args:
        X_river: Lista de dicts

    Returns:
        Lista ordenada de nomes das features
    """
    if not X_river:
        raise ValueError("X_river está vazio")

    return sorted(X_river[0].keys())


def summarize_conversion(
    X_river: List[Dict],
    X_array: np.ndarray,
    feature_names: List[str]
) -> Dict:
    """
    Gera resumo de uma conversão River → NumPy.

    Returns:
        Dicionário com estatísticas da conversão
    """
    return {
        'n_samples': len(X_river),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'array_shape': X_array.shape,
        'array_dtype': str(X_array.dtype),
        'array_memory_mb': X_array.nbytes / (1024 * 1024),
        'sample_values': {
            'min': float(X_array.min()),
            'max': float(X_array.max()),
            'mean': float(X_array.mean()),
            'std': float(X_array.std())
        }
    }
