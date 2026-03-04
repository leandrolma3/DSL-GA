#!/usr/bin/env python3
"""
Teste Local de Validacao com ERulesD2S

Executa teste completo com RBF_Abrupt_Severe incluindo ERulesD2S.
Compara com resultados anteriores (5 modelos Python).
"""

import subprocess
import sys
import time
from datetime import datetime, timedelta
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
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


def format_duration(seconds):
    """Formata duracao em horas, minutos, segundos"""
    td = timedelta(seconds=int(seconds))
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}min {secs}s"
    elif minutes > 0:
        return f"{minutes}min {secs}s"
    else:
        return f"{secs}s"


def main():
    print_header("VALIDACAO LOCAL COM ERULESD2S")

    print(f"Data/hora inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuracao:")
    print("  Dataset:    RBF_Abrupt_Severe")
    print("  Chunks:     3 (2 avaliacoes)")
    print("  Chunk size: 3000 instancias")
    print("  Modelos:    GBML, ACDWM, HAT, ARF, SRP, ERulesD2S (6 modelos)")
    print()
    print("Tempo estimado:")
    print("  - Modelos Python (5): ~45 min (ja validado)")
    print("  - ERulesD2S: ~12 min (3 chunks × 4 min)")
    print("  - Total: ~57 minutos")
    print()

    # Verificar se ERulesD2S esta configurado
    print_section("VERIFICANDO SETUP ERULESD2S")

    if not os.path.exists("erulesd2s.jar") and not os.path.exists("ERulesD2S"):
        logger.error("ERulesD2S nao encontrado!")
        logger.error("Execute primeiro:")
        logger.error("  1. python setup_erulesd2s.py")
        logger.error("  2. python test_erulesd2s_integration.py")
        return 1

    logger.info("ERulesD2S encontrado")

    # Confirmar execucao
    print()
    response = input("Iniciar teste local com ERulesD2S? (y/n): ")
    if response.lower() != 'y':
        logger.info("Teste cancelado")
        return 0

    # ========================================
    # FASE 1: MODELOS PYTHON (se nao executado)
    # ========================================
    print_section("FASE 1: MODELOS PYTHON")

    python_results_dir = "validation_local_results"

    if os.path.exists(python_results_dir):
        logger.info(f"Resultados Python ja existem em: {python_results_dir}")
        response = input("Reutilizar resultados existentes? (y/n): ")
        if response.lower() == 'y':
            logger.info("Reutilizando resultados Python existentes")
            python_done = True
        else:
            python_done = False
    else:
        python_done = False

    if not python_done:
        logger.info("Executando modelos Python...")

        cmd_python = [
            sys.executable,
            'compare_gbml_vs_river.py',
            '--stream', 'RBF_Abrupt_Severe',
            '--config', 'config_validation_local.yaml',
            '--models', 'HAT', 'ARF', 'SRP',
            '--chunks', '3',
            '--chunk-size', '3000',
            '--acdwm',
            '--seed', '42',
            '--output', python_results_dir
        ]

        logger.info("Comando: " + " ".join(cmd_python))
        print()

        start_python = time.time()

        try:
            result = subprocess.run(
                cmd_python,
                capture_output=False,
                text=True,
                check=True
            )

            duration_python = time.time() - start_python

            logger.info(f"Modelos Python concluidos em {format_duration(duration_python)}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Modelos Python falharam (code={e.returncode})")
            return 1

    # ========================================
    # FASE 2: ERULESD2S
    # ========================================
    print_section("FASE 2: ERULESD2S")

    logger.info("Preparando execucao do ERulesD2S...")

    # Criar script Python que usa o wrapper
    erulesd2s_script = """
import sys
import logging
from pathlib import Path
import pandas as pd

# Adicionar diretorio atual ao path
sys.path.insert(0, '.')

from arff_converter import ARFFConverter
from erulesd2s_wrapper import ERulesD2SWrapper, ERulesD2SEvaluator
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar configuracao
with open('config_validation_local.yaml') as f:
    config = yaml.safe_load(f)

# Importar data_handling_final para gerar chunks
import data_handling_final

# Gerar chunks
logger.info("Gerando chunks...")
chunks = data_handling_final.generate_stream(
    stream_name='RBF_Abrupt_Severe',
    config=config,
    seed=42
)

logger.info(f"Gerados {len(chunks)} chunks")

# Converter para ARFF
logger.info("Convertendo chunks para ARFF...")
converter = ARFFConverter(relation_name="RBF_Abrupt_Severe")

arff_dir = Path("arff_chunks")
arff_files = converter.convert_stream(
    chunks,
    output_dir=arff_dir,
    base_name="rbf_chunk"
)

logger.info(f"Arquivos ARFF: {[str(f) for f in arff_files]}")

# Criar wrapper ERulesD2S
logger.info("Criando wrapper ERulesD2S...")
wrapper = ERulesD2SWrapper(
    moa_jar_path="erulesd2s.jar",
    java_memory="4g",
    gpu_enabled=False
)

# Criar evaluator
evaluator = ERulesD2SEvaluator(
    wrapper=wrapper,
    output_dir=Path("validation_local_results_erulesd2s"),
    population_size=25,
    num_generations=50,
    rules_per_class=5
)

# Avaliar chunks
logger.info("Avaliando com ERulesD2S...")
results_df = evaluator.evaluate_chunks(chunks, arff_files)

# Salvar resultados
output_file = Path("validation_local_results_erulesd2s/erulesd2s_results.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_file, index=False)

logger.info(f"Resultados salvos em: {output_file}")
logger.info(f"\\nResultados:\\n{results_df}")
"""

    # Salvar script temporario
    temp_script = Path("_temp_erulesd2s_run.py")
    with open(temp_script, 'w') as f:
        f.write(erulesd2s_script)

    logger.info("Script ERulesD2S criado")
    logger.info("Executando ERulesD2S...")
    print()

    start_erulesd2s = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=False,
            text=True,
            check=True,
            timeout=1800  # 30 min timeout
        )

        duration_erulesd2s = time.time() - start_erulesd2s

        logger.info(f"ERulesD2S concluido em {format_duration(duration_erulesd2s)}")

    except subprocess.CalledProcessError as e:
        logger.error(f"ERulesD2S falhou (code={e.returncode})")
        return 1
    except subprocess.TimeoutExpired:
        logger.error("ERulesD2S timeout (30 min)")
        return 1
    finally:
        # Limpar script temporario
        if temp_script.exists():
            temp_script.unlink()

    # ========================================
    # FASE 3: CONSOLIDAR RESULTADOS
    # ========================================
    print_section("FASE 3: CONSOLIDANDO RESULTADOS")

    logger.info("Carregando resultados...")

    try:
        import pandas as pd
        import glob

        # Resultados Python
        python_csv = glob.glob(f"{python_results_dir}/**/comparison_table.csv", recursive=True)
        if python_csv:
            df_python = pd.read_csv(python_csv[0])
            logger.info(f"Resultados Python: {len(df_python)} linhas")
        else:
            logger.error("Resultados Python nao encontrados!")
            df_python = pd.DataFrame()

        # Resultados ERulesD2S
        erulesd2s_csv = "validation_local_results_erulesd2s/erulesd2s_results.csv"
        if os.path.exists(erulesd2s_csv):
            df_erulesd2s = pd.read_csv(erulesd2s_csv)
            logger.info(f"Resultados ERulesD2S: {len(df_erulesd2s)} linhas")
        else:
            logger.warning("Resultados ERulesD2S nao encontrados!")
            df_erulesd2s = pd.DataFrame()

        # Consolidar
        if not df_python.empty and not df_erulesd2s.empty:
            # Renomear coluna de chunk se necessario
            if 'chunk_idx' in df_erulesd2s.columns:
                df_erulesd2s.rename(columns={'chunk_idx': 'chunk'}, inplace=True)

            df_all = pd.concat([df_python, df_erulesd2s], ignore_index=True)

            # Salvar consolidado
            consolidated_file = "validation_local_results_with_erulesd2s.csv"
            df_all.to_csv(consolidated_file, index=False)

            logger.info(f"Resultados consolidados salvos em: {consolidated_file}")

            # Mostrar resumo
            print_section("RESUMO DOS RESULTADOS")

            print("G-mean medio por modelo:")
            print("-" * 60)
            summary = df_all.groupby('model')['gmean'].agg(['mean', 'std', 'min', 'max'])
            print(summary.round(4))
            print()

            print("Ranking por G-mean medio:")
            ranking = summary['mean'].sort_values(ascending=False)
            for idx, (model, gmean) in enumerate(ranking.items(), 1):
                print(f"  {idx}. {model:<12} : {gmean:.4f}")

        else:
            logger.error("Nao foi possivel consolidar resultados")

    except Exception as e:
        logger.error(f"Erro ao consolidar resultados: {e}")
        import traceback
        traceback.print_exc()

    # ========================================
    # CONCLUSAO
    # ========================================
    print_section("TESTE CONCLUIDO")

    total_time = (time.time() - start_python) if not python_done else duration_erulesd2s
    print(f"Tempo total: {format_duration(total_time)}")
    print()
    print("Arquivos gerados:")
    print(f"  - Resultados Python: {python_results_dir}/")
    print(f"  - Resultados ERulesD2S: validation_local_results_erulesd2s/")
    print(f"  - Consolidado: validation_local_results_with_erulesd2s.csv")
    print()
    print("Proximo passo:")
    print("  Analisar resultados e decidir sobre experimento completo")
    print()

    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
