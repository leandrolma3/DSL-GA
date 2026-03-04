"""
Script de Teste: Integracao ACDWM

PROPOSITO:
    Testar a integracao completa do ACDWM com o framework

TESTES:
    1. Importacao dos modulos ACDWM
    2. Inicializacao do ACDWMEvaluator
    3. Conversao de labels (0/1 <-> -1/+1)
    4. Treinamento e teste com dados sinteticos
    5. Avaliacao test-then-train (prequential)
    6. Avaliacao train-then-test (compativel GBML)

AUTOR: Claude Code
DATA: 2025-01-07
"""

import sys
import numpy as np
import logging
from pathlib import Path

# Configuracao de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
)
logger = logging.getLogger("test")


def print_test_header(test_name):
    """Imprime cabecalho de teste"""
    print("\n" + "="*70)
    print(f"TEST: {test_name}")
    print("="*70)


def generate_synthetic_chunk(n_samples=100, n_features=4, seed=42):
    """
    Gera um chunk sintetico de dados em formato River.

    Returns:
        (X_river, y_river) onde X_river eh lista de dicts
    """
    np.random.seed(seed)

    X_river = []
    y_river = []

    for _ in range(n_samples):
        # Gera features
        x_dict = {f'x_{i}': np.random.randn() for i in range(n_features)}
        X_river.append(x_dict)

        # Gera label (0 ou 1)
        y = 1 if np.random.rand() > 0.5 else 0
        y_river.append(y)

    return X_river, y_river


# ============================================================================
# TESTE 1: Importacao de Modulos
# ============================================================================

def test_import_modules():
    """Testa importacao dos modulos ACDWM"""
    print_test_header("Importacao de Modulos ACDWM")

    try:
        from baseline_acdwm import import_acdwm_modules

        acdwm_path = Path(__file__).parent / 'ACDWM'

        if not acdwm_path.exists():
            print(f"[X] Diretorio ACDWM nao encontrado: {acdwm_path}")
            print("Execute antes: git clone https://github.com/jasonyanglu/ACDWM.git")
            return False

        modules = import_acdwm_modules(str(acdwm_path))

        print(f"  [OK] dwmil: {modules['dwmil']}")
        print(f"  [OK] chunk_size_select: {modules['chunk_size_select']}")
        print(f"  [OK] underbagging: {modules['underbagging']}")
        print(f"  [OK] subunderbagging: {modules['subunderbagging']}")

        print("\n  [OK] Teste PASSOU!")
        return True

    except Exception as e:
        print(f"  [X] Teste FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TESTE 2: Conversao de Labels
# ============================================================================

def test_label_conversion():
    """Testa conversao de labels 0/1 <-> -1/+1"""
    print_test_header("Conversao de Labels")

    try:
        from baseline_acdwm import convert_labels_to_acdwm, convert_labels_from_acdwm

        # Teste: 0/1 -> -1/+1
        y_original = np.array([0, 1, 0, 1, 1, 0])
        y_acdwm = convert_labels_to_acdwm(y_original)

        expected = np.array([-1, 1, -1, 1, 1, -1])
        assert np.array_equal(y_acdwm, expected), f"Esperado {expected}, recebido {y_acdwm}"
        print(f"  [OK] 0/1 -> -1/+1: {y_original} -> {y_acdwm}")

        # Teste: -1/+1 -> 0/1 (round-trip)
        y_converted_back = convert_labels_from_acdwm(y_acdwm)
        assert np.array_equal(y_converted_back, y_original), "Round-trip falhou"
        print(f"  [OK] -1/+1 -> 0/1: {y_acdwm} -> {y_converted_back}")

        print("\n  [OK] Teste PASSOU!")
        return True

    except Exception as e:
        print(f"  [X] Teste FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TESTE 3: Inicializacao do Evaluator
# ============================================================================

def test_evaluator_initialization():
    """Testa inicializacao do ACDWMEvaluator"""
    print_test_header("Inicializacao do ACDWMEvaluator")

    try:
        from baseline_acdwm import ACDWMEvaluator

        acdwm_path = Path(__file__).parent / 'ACDWM'

        if not acdwm_path.exists():
            print(f"[X] Diretorio ACDWM nao encontrado")
            return False

        # Cria evaluator
        evaluator = ACDWMEvaluator(
            acdwm_path=str(acdwm_path),
            classes=[0, 1],
            evaluation_mode='train-then-test',
            theta=0.001
        )

        print(f"  [OK] Evaluator criado: {evaluator.model_name}")
        print(f"  [OK] Modo de avaliacao: {evaluator.evaluation_mode}")
        print(f"  [OK] Theta: {evaluator.theta}")
        print(f"  [OK] Modelo DWMIL: {evaluator.model}")

        # Valida atributos
        assert evaluator.model is not None, "Modelo nao foi inicializado"
        assert hasattr(evaluator.model, 'ensemble'), "Modelo nao tem atributo ensemble"
        assert hasattr(evaluator.model, 'update_chunk'), "Modelo nao tem metodo update_chunk"

        print("\n  [OK] Teste PASSOU!")
        return True

    except Exception as e:
        print(f"  [X] Teste FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TESTE 4: Treino e Teste Basico
# ============================================================================

def test_train_and_test():
    """Testa treino e teste basico"""
    print_test_header("Treino e Teste Basico")

    try:
        from baseline_acdwm import ACDWMEvaluator

        acdwm_path = Path(__file__).parent / 'ACDWM'

        if not acdwm_path.exists():
            print(f"[X] Diretorio ACDWM nao encontrado")
            return False

        # Cria evaluator
        evaluator = ACDWMEvaluator(
            acdwm_path=str(acdwm_path),
            classes=[0, 1],
            evaluation_mode='train-then-test',
            theta=0.01
        )

        # Gera chunk sintetico
        X_train, y_train = generate_synthetic_chunk(n_samples=200, seed=42)
        X_test, y_test = generate_synthetic_chunk(n_samples=100, seed=43)

        print(f"\n  Dados de treino: {len(X_train)} samples")
        print(f"  Dados de teste: {len(X_test)} samples")

        # Treina
        print("\n  Treinando...")
        train_info = evaluator.train_on_chunk(X_train, y_train)
        print(f"    [OK] Treino concluido: {train_info}")
        print(f"    [OK] Ensemble size: {train_info['ensemble_size']}")

        # Testa
        print("\n  Testando...")
        test_metrics = evaluator.test_on_chunk(X_test, y_test)
        print(f"    [OK] Teste concluido")
        print(f"    [OK] Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        print(f"    [OK] G-mean: {test_metrics.get('gmean', 0):.4f}")
        print(f"    [OK] F1: {test_metrics.get('f1_weighted', 0):.4f}")

        # Valida que metricas foram calculadas
        assert 'accuracy' in test_metrics, "Accuracy nao calculado"
        assert 'gmean' in test_metrics, "G-mean nao calculado"
        assert test_metrics['accuracy'] >= 0, "Accuracy invalido"

        print("\n  [OK] Teste PASSOU!")
        return True

    except Exception as e:
        print(f"  [X] Teste FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TESTE 5: Avaliacao Test-Then-Train (Prequential)
# ============================================================================

def test_prequential_evaluation():
    """Testa avaliacao test-then-train (prequential)"""
    print_test_header("Avaliacao Test-Then-Train (Prequential)")

    try:
        from baseline_acdwm import ACDWMEvaluator

        acdwm_path = Path(__file__).parent / 'ACDWM'

        if not acdwm_path.exists():
            print(f"[X] Diretorio ACDWM nao encontrado")
            return False

        # Cria evaluator em modo prequential
        evaluator = ACDWMEvaluator(
            acdwm_path=str(acdwm_path),
            classes=[0, 1],
            evaluation_mode='test-then-train',
            theta=0.01
        )

        # Gera 3 chunks
        chunks = []
        for i in range(3):
            X_train, y_train = generate_synthetic_chunk(n_samples=150, seed=100+i)
            X_test, y_test = generate_synthetic_chunk(n_samples=50, seed=200+i)
            chunks.append((X_train, y_train, X_test, y_test))

        print(f"\n  Avaliando {len(chunks)} chunks em modo prequential...")

        # Executa avaliacao
        results = evaluator.evaluate_test_then_train(chunks)

        print(f"\n  [OK] Avaliacao concluida: {len(results)} chunks")

        # Imprime resultados
        for i, result in enumerate(results):
            print(f"\n  Chunk {i+1}:")
            print(f"    G-mean: {result.get('gmean', 0):.4f}")
            print(f"    Accuracy: {result.get('accuracy', 0):.4f}")
            print(f"    Ensemble size: {result['train_info']['ensemble_size']}")

        # Valida
        assert len(results) == 3, f"Esperado 3 resultados, recebido {len(results)}"
        for r in results:
            assert 'gmean' in r, "G-mean nao calculado"

        print("\n  [OK] Teste PASSOU!")
        return True

    except Exception as e:
        print(f"  [X] Teste FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TESTE 6: Avaliacao Train-Then-Test
# ============================================================================

def test_train_then_test_evaluation():
    """Testa avaliacao train-then-test (compativel GBML)"""
    print_test_header("Avaliacao Train-Then-Test")

    try:
        from baseline_acdwm import ACDWMEvaluator

        acdwm_path = Path(__file__).parent / 'ACDWM'

        if not acdwm_path.exists():
            print(f"[X] Diretorio ACDWM nao encontrado")
            return False

        # Cria evaluator em modo train-then-test
        evaluator = ACDWMEvaluator(
            acdwm_path=str(acdwm_path),
            classes=[0, 1],
            evaluation_mode='train-then-test',
            theta=0.01
        )

        # Gera 3 chunks
        chunks = []
        for i in range(3):
            X_train, y_train = generate_synthetic_chunk(n_samples=150, seed=300+i)
            X_test, y_test = generate_synthetic_chunk(n_samples=50, seed=400+i)
            chunks.append((X_train, y_train, X_test, y_test))

        print(f"\n  Avaliando {len(chunks)} chunks em modo train-then-test...")

        # Executa avaliacao
        results = evaluator.evaluate_train_then_test(chunks)

        print(f"\n  [OK] Avaliacao concluida: {len(results)} chunks")

        # Imprime resultados
        for i, result in enumerate(results):
            print(f"\n  Chunk {i+1}:")
            print(f"    G-mean: {result.get('gmean', 0):.4f}")
            print(f"    Accuracy: {result.get('accuracy', 0):.4f}")
            print(f"    Ensemble size: {result['train_info']['ensemble_size']}")

        # Valida
        assert len(results) == 3, f"Esperado 3 resultados, recebido {len(results)}"

        print("\n  [OK] Teste PASSOU!")
        return True

    except Exception as e:
        print(f"  [X] Teste FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# EXECUCAO DE TODOS OS TESTES
# ============================================================================

def run_all_tests():
    """Executa todos os testes"""

    print("\n" + "="*70)
    print("SUITE DE TESTES: Integracao ACDWM")
    print("="*70)

    tests = [
        ("Importacao de Modulos", test_import_modules),
        ("Conversao de Labels", test_label_conversion),
        ("Inicializacao do Evaluator", test_evaluator_initialization),
        ("Treino e Teste Basico", test_train_and_test),
        ("Avaliacao Prequential", test_prequential_evaluation),
        ("Avaliacao Train-Then-Test", test_train_then_test_evaluation)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
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
        print("\nProximo passo: Integrar ACDWM ao compare_gbml_vs_river.py")
        return 0
    else:
        print(f"\n[WARNING] {failed} teste(s) falharam")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
