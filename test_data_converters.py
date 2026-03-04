"""
Script de Teste: Módulo data_converters.py

PROPÓSITO:
    Validar conversões entre formatos River ↔ NumPy

TESTES:
    1. Conversão River → NumPy (básica)
    2. Conversão River → NumPy (com feature_names customizado)
    3. Conversão NumPy → River
    4. Conversão round-trip (River → NumPy → River)
    5. Conversão de chunks em lote
    6. Validação de erros

AUTOR: Claude Code
DATA: 2025-01-07
"""

import numpy as np
import logging
from data_converters import (
    river_to_numpy,
    river_labels_to_numpy,
    numpy_to_river,
    convert_chunks_river_to_numpy,
    validate_river_format,
    get_feature_names_from_river,
    summarize_conversion
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
)
logger = logging.getLogger("test")


def print_test_header(test_name):
    """Imprime cabeçalho de teste"""
    print("\n" + "="*70)
    print(f"TEST: {test_name}")
    print("="*70)


def assert_equal(actual, expected, message=""):
    """Helper para asserts"""
    if actual != expected:
        raise AssertionError(f"{message}\nEsperado: {expected}\nRecebido: {actual}")
    print(f"  [OK] {message}")


def assert_array_equal(actual, expected, message=""):
    """Helper para asserts de arrays"""
    if not np.array_equal(actual, expected):
        raise AssertionError(f"{message}\nEsperado:\n{expected}\nRecebido:\n{actual}")
    print(f"  [OK] {message}")


# ============================================================================
# TESTE 1: Conversão River → NumPy (Básica)
# ============================================================================

def test_river_to_numpy_basic():
    """Teste básico de conversão River → NumPy"""
    print_test_header("River to NumPy (Basica)")

    # Dados de exemplo
    X_river = [
        {'x_0': 0.5, 'x_1': 0.3},
        {'x_0': 0.2, 'x_1': 0.8},
        {'x_0': 0.7, 'x_1': 0.1}
    ]

    y_river = [0, 1, 0]

    # Converte
    X_array, feature_names = river_to_numpy(X_river)
    y_array = river_labels_to_numpy(y_river)

    # Validações
    assert_equal(X_array.shape, (3, 2), "Shape de X")
    assert_equal(y_array.shape, (3,), "Shape de y")
    assert_equal(feature_names, ['x_0', 'x_1'], "Feature names")

    # Valida valores
    expected_X = np.array([[0.5, 0.3], [0.2, 0.8], [0.7, 0.1]], dtype=np.float32)
    assert_array_equal(X_array, expected_X, "Valores de X")

    expected_y = np.array([0, 1, 0], dtype=np.int32)
    assert_array_equal(y_array, expected_y, "Valores de y")

    print("  [OK] Teste PASSOU!")


# ============================================================================
# TESTE 2: Conversão com Feature Names Customizado
# ============================================================================

def test_river_to_numpy_custom_features():
    """Teste com nomes de features customizados"""
    print_test_header("River to NumPy (Custom Feature Names)")

    X_river = [
        {'feature_a': 1.0, 'feature_b': 2.0, 'feature_c': 3.0},
        {'feature_a': 4.0, 'feature_b': 5.0, 'feature_c': 6.0}
    ]

    # Ordem customizada
    custom_names = ['feature_c', 'feature_a', 'feature_b']

    X_array, feature_names = river_to_numpy(X_river, feature_names=custom_names)

    # Valida ordem
    assert_equal(feature_names, custom_names, "Feature names preservados")

    # Valida que valores seguem a ordem especificada
    # feature_c=3.0, feature_a=1.0, feature_b=2.0
    expected_X = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], dtype=np.float32)
    assert_array_equal(X_array, expected_X, "Valores na ordem correta")

    print("  [OK] Teste PASSOU!")


# ============================================================================
# TESTE 3: Conversão NumPy → River
# ============================================================================

def test_numpy_to_river():
    """Teste de conversão NumPy → River"""
    print_test_header("NumPy to River")

    X_array = np.array([[0.5, 0.3], [0.2, 0.8]], dtype=np.float32)
    feature_names = ['feat_0', 'feat_1']

    X_river = numpy_to_river(X_array, feature_names)

    # Validações
    assert_equal(len(X_river), 2, "Número de samples")
    assert_equal(set(X_river[0].keys()), set(feature_names), "Feature names")

    # Valida valores (arredonda para evitar erro de precisão float)
    assert_equal(round(X_river[0]['feat_0'], 6), 0.5, "Valor [0, 0]")
    assert_equal(round(X_river[0]['feat_1'], 6), 0.3, "Valor [0, 1]")
    assert_equal(round(X_river[1]['feat_0'], 6), 0.2, "Valor [1, 0]")
    assert_equal(round(X_river[1]['feat_1'], 6), 0.8, "Valor [1, 1]")

    print("  [OK] Teste PASSOU!")


# ============================================================================
# TESTE 4: Round-Trip (River → NumPy → River)
# ============================================================================

def test_roundtrip():
    """Teste de conversão round-trip"""
    print_test_header("Round-Trip (River to NumPy to River)")

    # Dados originais
    X_original = [
        {'x_0': 0.1, 'x_1': 0.2, 'x_2': 0.3},
        {'x_0': 0.4, 'x_1': 0.5, 'x_2': 0.6}
    ]

    # River → NumPy
    X_array, feature_names = river_to_numpy(X_original)

    # NumPy → River
    X_reconstructed = numpy_to_river(X_array, feature_names)

    # Valida que é idêntico
    assert_equal(len(X_reconstructed), len(X_original), "Número de samples")

    for i, (orig, recon) in enumerate(zip(X_original, X_reconstructed)):
        for key in orig.keys():
            assert_equal(
                round(recon[key], 6),  # Arredonda para evitar erros de float
                round(orig[key], 6),
                f"Sample {i}, feature {key}"
            )

    print("  [OK] Teste PASSOU!")


# ============================================================================
# TESTE 5: Conversão de Chunks em Lote
# ============================================================================

def test_convert_chunks():
    """Teste de conversão de múltiplos chunks"""
    print_test_header("Conversao de Chunks em Lote")

    # Simula 3 chunks com formato (X_train, y_train, X_test, y_test)
    chunks = [
        (
            [{'x_0': 0.1, 'x_1': 0.2}],  # X_train
            [0],                          # y_train
            [{'x_0': 0.3, 'x_1': 0.4}],  # X_test
            [1]                           # y_test
        ),
        (
            [{'x_0': 0.5, 'x_1': 0.6}],
            [1],
            [{'x_0': 0.7, 'x_1': 0.8}],
            [0]
        ),
        (
            [{'x_0': 0.9, 'x_1': 1.0}],
            [0],
            [{'x_0': 1.1, 'x_1': 1.2}],
            [1]
        )
    ]

    # Converte
    converted_chunks, feature_names = convert_chunks_river_to_numpy(chunks)

    # Validações
    assert_equal(len(converted_chunks), 3, "Número de chunks")
    assert_equal(feature_names, ['x_0', 'x_1'], "Feature names")

    # Valida primeiro chunk
    X_train, y_train, X_test, y_test = converted_chunks[0]
    assert_equal(X_train.shape, (1, 2), "Shape X_train chunk 0")
    assert_equal(y_train.shape, (1,), "Shape y_train chunk 0")
    assert_equal(X_test.shape, (1, 2), "Shape X_test chunk 0")
    assert_equal(y_test.shape, (1,), "Shape y_test chunk 0")

    print("  [OK] Teste PASSOU!")


# ============================================================================
# TESTE 6: Validação de Erros
# ============================================================================

def test_error_handling():
    """Teste de tratamento de erros"""
    print_test_header("Validacao de Erros")

    # Teste 1: Lista vazia
    try:
        river_to_numpy([])
        raise AssertionError("Deveria ter lançado ValueError para lista vazia")
    except ValueError as e:
        print(f"  [OK] Erro esperado para lista vazia: {e}")

    # Teste 2: Features inconsistentes
    try:
        X_bad = [
            {'x_0': 0.1, 'x_1': 0.2},
            {'x_0': 0.3, 'x_2': 0.4}  # x_2 em vez de x_1!
        ]
        river_to_numpy(X_bad)
        raise AssertionError("Deveria ter lançado ValueError para features inconsistentes")
    except ValueError as e:
        print(f"  [OK] Erro esperado para features inconsistentes: {e}")

    # Teste 3: Array 1D em vez de 2D
    try:
        X_1d = np.array([1, 2, 3])
        numpy_to_river(X_1d)
        raise AssertionError("Deveria ter lançado ValueError para array 1D")
    except ValueError as e:
        print(f"  [OK] Erro esperado para array 1D: {e}")

    print("  [OK] Teste PASSOU!")


# ============================================================================
# TESTE 7: Utilitários
# ============================================================================

def test_utilities():
    """Teste de funções utilitárias"""
    print_test_header("Funcoes Utilitarias")

    X_river = [
        {'x_0': 0.5, 'x_1': 0.3},
        {'x_0': 0.2, 'x_1': 0.8}
    ]

    # Teste validate_river_format
    result = validate_river_format(X_river)
    assert_equal(result, True, "validate_river_format")

    # Teste get_feature_names_from_river
    names = get_feature_names_from_river(X_river)
    assert_equal(names, ['x_0', 'x_1'], "get_feature_names_from_river")

    # Teste summarize_conversion
    X_array, feature_names = river_to_numpy(X_river)
    summary = summarize_conversion(X_river, X_array, feature_names)

    assert_equal(summary['n_samples'], 2, "Summary: n_samples")
    assert_equal(summary['n_features'], 2, "Summary: n_features")
    assert_equal(summary['array_shape'], (2, 2), "Summary: array_shape")

    print("  [OK] Teste PASSOU!")


# ============================================================================
# EXECUÇÃO DE TODOS OS TESTES
# ============================================================================

def run_all_tests():
    """Executa todos os testes"""

    print("\n" + "="*70)
    print("INICIANDO SUITE DE TESTES: data_converters.py")
    print("="*70)

    tests = [
        ("Test 1", test_river_to_numpy_basic),
        ("Test 2", test_river_to_numpy_custom_features),
        ("Test 3", test_numpy_to_river),
        ("Test 4", test_roundtrip),
        ("Test 5", test_convert_chunks),
        ("Test 6", test_error_handling),
        ("Test 7", test_utilities)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            logger.error(f"[X] {test_name} FALHOU: {e}", exc_info=True)

    # Resumo
    print("\n" + "="*70)
    print("RESUMO DOS TESTES")
    print("="*70)
    print(f"Total: {len(tests)}")
    print(f"[OK] Passou: {passed}")
    print(f"[X] Falhou: {failed}")

    if failed == 0:
        print("\n[SUCCESS] TODOS OS TESTES PASSARAM!")
        return 0
    else:
        print(f"\n[WARNING] {failed} teste(s) falharam")
        return 1


if __name__ == '__main__':
    import sys
    exit_code = run_all_tests()
    sys.exit(exit_code)
