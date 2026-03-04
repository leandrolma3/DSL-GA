#!/usr/bin/env python3
"""
Teste de Integracao ERulesD2S

Testa integracao completa: conversao ARFF + execucao ERulesD2S + parsing resultados.
"""

import logging
import sys
from pathlib import Path
import numpy as np

# Importar modulos criados
from arff_converter import ARFFConverter, validate_arff
from erulesd2s_wrapper import ERulesD2SWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(message):
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80 + "\n")


def print_section(message):
    print("\n" + "-" * 80)
    print(message)
    print("-" * 80 + "\n")


def generate_test_data(n_samples=1000, n_features=5):
    """Gera dados de teste simples"""
    logger.info(f"Gerando dados de teste: {n_samples} samples, {n_features} features")

    # Dados sinteticos simples (2 classes linearmente separaveis)
    X = np.random.rand(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)  # Regra simples

    logger.info(f"Classes: {np.bincount(y)}")

    return X, y


def main():
    print_header("TESTE DE INTEGRACAO ERULESD2S")

    # Criar diretorio de teste
    test_dir = Path("test_erulesd2s_output")
    test_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # 1. GERAR DADOS DE TESTE
    # ========================================
    print_section("1. GERANDO DADOS DE TESTE")

    X, y = generate_test_data(n_samples=1000, n_features=5)

    logger.info(f"Dados gerados: X.shape={X.shape}, y.shape={y.shape}")

    # ========================================
    # 2. CONVERTER PARA ARFF
    # ========================================
    print_section("2. CONVERTENDO PARA ARFF")

    converter = ARFFConverter(relation_name="test_integration")

    arff_file = test_dir / "test_data.arff"

    try:
        arff_path = converter.convert_chunk(
            X, y,
            feature_names=[f"att{i+1}" for i in range(X.shape[1])],
            class_names=["0", "1"],
            output_file=arff_file
        )

        logger.info(f"Arquivo ARFF criado: {arff_path}")

        # Validar ARFF
        if not validate_arff(arff_path):
            logger.error("Validacao ARFF falhou!")
            return 1

        logger.info("ARFF validado com sucesso")

    except Exception as e:
        logger.error(f"Erro ao converter ARFF: {e}")
        return 1

    # ========================================
    # 3. VERIFICAR ERULESD2S JAR
    # ========================================
    print_section("3. VERIFICANDO ERULESD2S JAR")

    jar_candidates = [
        Path("erulesd2s.jar"),
        Path("ERulesD2S/target/erulesd2s-1.0-SNAPSHOT.jar"),
        Path("moa.jar")
    ]

    erulesd2s_jar = None
    for jar in jar_candidates:
        if jar.exists():
            erulesd2s_jar = jar
            logger.info(f"JAR encontrado: {jar}")
            break

    if not erulesd2s_jar:
        logger.error("ERulesD2S JAR nao encontrado!")
        logger.error("Execute primeiro: python setup_erulesd2s.py")
        return 1

    # ========================================
    # 4. CRIAR WRAPPER ERULESD2S
    # ========================================
    print_section("4. CONFIGURANDO WRAPPER ERULESD2S")

    try:
        wrapper = ERulesD2SWrapper(
            moa_jar_path=str(erulesd2s_jar),
            java_memory="2g",  # Memoria reduzida para teste
            gpu_enabled=False  # Desabilitar GPU no teste local
        )

        logger.info("Wrapper ERulesD2S criado")

    except Exception as e:
        logger.error(f"Erro ao criar wrapper: {e}")
        return 1

    # ========================================
    # 5. EXECUTAR ERULESD2S
    # ========================================
    print_section("5. EXECUTANDO ERULESD2S (TESTE RAPIDO)")

    logger.info("Parametros:")
    logger.info("  Population size: 10 (reduzido para teste)")
    logger.info("  Generations: 20 (reduzido para teste)")
    logger.info("  Rules per class: 3")
    logger.info("  Timeout: 2 minutos")

    try:
        success, results = wrapper.run(
            arff_file=arff_path,
            output_dir=test_dir / "erulesd2s_run",
            population_size=10,  # Reduzido para teste rapido
            num_generations=20,  # Reduzido para teste rapido
            rules_per_class=3,
            chunk_size=1000,
            max_instances=1000,
            timeout=120  # 2 min timeout
        )

        if not success:
            logger.error("ERulesD2S falhou!")
            logger.error("Verifique os logs em: test_erulesd2s_output/erulesd2s_run/")
            return 1

        logger.info("ERulesD2S executado com sucesso!")

    except Exception as e:
        logger.error(f"Erro ao executar ERulesD2S: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ========================================
    # 6. VERIFICAR RESULTADOS
    # ========================================
    print_section("6. VERIFICANDO RESULTADOS")

    logger.info("Metricas obtidas:")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    # Verificar se metricas sao razoaveis
    if 'accuracy' in results:
        acc = results['accuracy']
        if 0.3 < acc < 1.0:
            logger.info(f"[OK] Accuracy razoavel: {acc:.4f}")
        else:
            logger.warning(f"[AVISO] Accuracy suspeita: {acc:.4f}")

    # Verificar arquivos gerados
    output_run_dir = test_dir / "erulesd2s_run"
    generated_files = list(output_run_dir.glob("*"))

    logger.info(f"Arquivos gerados: {len(generated_files)}")
    for f in generated_files:
        size_kb = f.stat().st_size / 1024
        logger.info(f"  - {f.name} ({size_kb:.2f} KB)")

    # ========================================
    # CONCLUSAO
    # ========================================
    print_section("TESTE CONCLUIDO")

    print("Status: SUCESSO")
    print()
    print("Resumo:")
    print(f"  - Dados de teste: {X.shape[0]} instancias")
    print(f"  - Arquivo ARFF: {arff_file}")
    print(f"  - ERulesD2S executado: Sim")
    print(f"  - Accuracy: {results.get('accuracy', 'N/A'):.4f}" if 'accuracy' in results else "  - Accuracy: N/A")
    print(f"  - Tempo de execucao: {results.get('execution_time', 'N/A'):.1f}s" if 'execution_time' in results else "  - Tempo: N/A")
    print()
    print("Proximo passo:")
    print("  python test_erulesd2s_local_validation.py")
    print()

    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
