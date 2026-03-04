#!/usr/bin/env python3
"""
Script para executar experimento completo com 11 datasets e 6 modelos.

Executa:
- FASE 1: GBML, ACDWM, HAT, ARF, SRP (Python)
- FASE 2: ERulesD2S (Java/MOA)

Tempo total estimado: 14 horas
"""

import subprocess
import sys
import yaml
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def format_duration(seconds):
    """Formata duração em horas, minutos, segundos"""
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
    print("=" * 80)
    print("EXPERIMENTO COMPLETO - 11 DATASETS + 6 MODELOS")
    print("=" * 80)
    print(f"\nInício: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Carregar configuração
    config_file = 'config_experiment_expanded.yaml'
    with open(config_file) as f:
        config = yaml.safe_load(f)

    datasets = config['experiment_settings']['drift_simulation_experiments']
    chunk_size = config['data_params']['chunk_size']
    num_chunks = config['data_params']['num_chunks']

    print("Configuração:")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Chunks por dataset: {num_chunks}")
    print(f"  Total de chunks: {len(datasets) * num_chunks}")
    print()
    print("Modelos:")
    print("  - GBML (Python/GA)")
    print("  - ACDWM (Python/Ensemble)")
    print("  - HAT, ARF, SRP (Python/River)")
    print("  - ERulesD2S (Java/MOA)")
    print()
    print("Tempo estimado:")
    print("  - Modelos Python: ~11 horas")
    print("  - ERulesD2S: ~3 horas")
    print("  - Total: ~14 horas")
    print()
    print("=" * 80)

    output_base = Path('experiment_expanded_results')
    output_base.mkdir(parents=True, exist_ok=True)

    start_total = time.time()

    # ========================================
    # FASE 1: MODELOS PYTHON
    # ========================================
    print()
    print("=" * 80)
    print("FASE 1/2: MODELOS PYTHON (GBML, ACDWM, HAT, ARF, SRP)")
    print("=" * 80)
    print()

    results_python = []

    for idx, dataset in enumerate(datasets, 1):
        print()
        print("-" * 80)
        print(f"[{idx}/{len(datasets)}] Processando: {dataset}")
        print("-" * 80)
        print()

        start_dataset = time.time()

        # Comando para executar modelos Python
        cmd_python = [
            sys.executable,
            'compare_gbml_vs_river.py',
            '--stream', dataset,
            '--config', config_file,
            '--models', 'HAT', 'ARF', 'SRP',
            '--chunks', str(num_chunks),
            '--chunk-size', str(chunk_size),
            '--acdwm',
            '--seed', '42',
            '--output', str(output_base)
        ]

        logger.info(f"Executando: {' '.join(cmd_python)}")

        try:
            result = subprocess.run(
                cmd_python,
                capture_output=True,
                text=True,
                check=True
            )

            duration = time.time() - start_dataset

            logger.info(f"✓ Dataset {dataset} concluído em {format_duration(duration)}")

            results_python.append({
                'dataset': dataset,
                'status': 'success',
                'duration': duration
            })

        except subprocess.CalledProcessError as e:
            duration = time.time() - start_dataset
            logger.error(f"✗ Dataset {dataset} falhou após {format_duration(duration)}")
            logger.error(f"Código de erro: {e.returncode}")
            logger.error(f"Stderr: {e.stderr[:500]}")

            results_python.append({
                'dataset': dataset,
                'status': 'failed',
                'duration': duration,
                'error': str(e)
            })

            # Decidir se continua ou para
            response = input("\nContinuar com próximo dataset? (y/n): ")
            if response.lower() != 'y':
                logger.warning("Experimento interrompido pelo usuário")
                return 1

    duration_python = time.time() - start_total

    print()
    print("=" * 80)
    print("RESUMO FASE 1 - MODELOS PYTHON")
    print("=" * 80)
    print(f"Duração total: {format_duration(duration_python)}")
    print(f"Datasets bem-sucedidos: {sum(1 for r in results_python if r['status'] == 'success')}/{len(datasets)}")
    print()

    # ========================================
    # FASE 2: ERULESD2S
    # ========================================
    print()
    print("=" * 80)
    print("FASE 2/2: ERULESD2S")
    print("=" * 80)
    print()

    # Verificar se ERulesD2S está disponível
    if not Path('erulesd2s.jar').exists():
        logger.warning("ERulesD2S JAR não encontrado!")
        logger.warning("Pulando fase ERulesD2S")
        logger.warning("Execute 'python setup_erulesd2s.py' para instalar")
    else:
        logger.info("ERulesD2S detectado - executando para todos os datasets...")

        # Criar script para executar ERulesD2S
        erulesd2s_script = """
import sys
sys.path.insert(0, '.')

from pathlib import Path
import yaml
import logging
from arff_converter import ARFFConverter
from erulesd2s_wrapper import ERulesD2SWrapper, ERulesD2SEvaluator
import data_handling_final

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar config
with open('config_experiment_expanded.yaml') as f:
    config = yaml.safe_load(f)

datasets = config['experiment_settings']['drift_simulation_experiments']

for dataset in datasets:
    logger.info(f"Processando {dataset} com ERulesD2S...")

    # Gerar chunks
    chunks = data_handling_final.generate_stream(
        stream_name=dataset,
        config=config,
        seed=42
    )

    # Converter para ARFF
    converter = ARFFConverter(relation_name=dataset)
    arff_dir = Path(f"arff_chunks/{dataset}")
    arff_files = converter.convert_stream(chunks, arff_dir, f"{dataset}_chunk")

    # Criar wrapper
    wrapper = ERulesD2SWrapper(moa_jar_path="erulesd2s.jar", java_memory="4g", gpu_enabled=False)

    # Avaliar
    evaluator = ERulesD2SEvaluator(
        wrapper=wrapper,
        output_dir=Path(f"experiment_expanded_results_erulesd2s/{dataset}"),
        population_size=25,
        num_generations=50,
        rules_per_class=5
    )

    results_df = evaluator.evaluate_chunks(chunks, arff_files)

    # Salvar
    output_file = Path(f"experiment_expanded_results_erulesd2s/{dataset}_erulesd2s.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    logger.info(f"✓ {dataset} ERulesD2S concluído")

logger.info("ERulesD2S concluído para todos os datasets!")
"""

        script_file = Path('_temp_run_erulesd2s.py')
        with open(script_file, 'w') as f:
            f.write(erulesd2s_script)

        start_erulesd2s = time.time()

        try:
            result = subprocess.run(
                [sys.executable, str(script_file)],
                capture_output=True,
                text=True,
                check=True,
                timeout=14400  # 4 horas timeout
            )

            duration_erulesd2s = time.time() - start_erulesd2s
            logger.info(f"✓ ERulesD2S concluído em {format_duration(duration_erulesd2s)}")

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ ERulesD2S falhou: {e}")
        except subprocess.TimeoutExpired:
            logger.error("✗ ERulesD2S timeout (4 horas)")
        finally:
            if script_file.exists():
                script_file.unlink()

    # ========================================
    # RESUMO FINAL
    # ========================================
    duration_total = time.time() - start_total

    print()
    print("=" * 80)
    print("EXPERIMENTO COMPLETO FINALIZADO")
    print("=" * 80)
    print(f"\nTérmino: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duração total: {format_duration(duration_total)}")
    print()
    print("Resultados salvos em:")
    print(f"  - {output_base}/ (modelos Python)")
    print(f"  - experiment_expanded_results_erulesd2s/ (ERulesD2S)")
    print()
    print("Próximo passo:")
    print("  Consolidar resultados com script de análise")
    print()
    print("=" * 80)

    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nExperimento interrompido pelo usuário")
        sys.exit(1)
