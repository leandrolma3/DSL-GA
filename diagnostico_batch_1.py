#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Diagnostico - Batch 1
Verifica se todos os arquivos necessarios existem antes do pos-processamento
"""

import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Detectar ambiente
IS_COLAB = 'COLAB_GPU' in os.environ or '/content/' in os.getcwd()

if IS_COLAB:
    BASE_DIR = Path("/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid")
    RESULTS_DIR = BASE_DIR / "experiments_6chunks_phase2_gbml" / "batch_1"
    CONFIG_FILE = BASE_DIR / "configs" / "config_batch_1.yaml"
else:
    BASE_DIR = Path(__file__).parent
    RESULTS_DIR = BASE_DIR / "experiments_6chunks_phase2_gbml" / "batch_1"
    CONFIG_FILE = BASE_DIR / "configs" / "config_batch_1.yaml"

BATCH_1_EXPERIMENTS = [
    "SEA_Abrupt_Simple",
    "SEA_Abrupt_Chain",
    "SEA_Abrupt_Recurring",
    "AGRAWAL_Abrupt_Simple_Mild",
    "AGRAWAL_Abrupt_Simple_Severe",
    "AGRAWAL_Abrupt_Chain_Long",
    "RBF_Abrupt_Severe",
    "RBF_Abrupt_Blip",
    "STAGGER_Abrupt_Chain",
    "STAGGER_Abrupt_Recurring",
    "HYPERPLANE_Abrupt_Simple",
    "RANDOMTREE_Abrupt_Simple"
]

def check_file(file_path, description):
    """Verifica se um arquivo existe"""
    if file_path.exists():
        size = file_path.stat().st_size
        size_str = f"{size:,} bytes" if size < 1024*1024 else f"{size/(1024*1024):.2f} MB"
        logger.info(f"  [OK] {description}: {size_str}")
        return True
    else:
        logger.warning(f"  [FALTA] {description}")
        return False

def diagnostico_experimento(exp_name):
    """Verifica arquivos de um experimento"""
    logger.info(f"\n{'='*80}")
    logger.info(f"DIAGNOSTICO: {exp_name}")
    logger.info(f"{'='*80}")

    exp_dir = RESULTS_DIR / exp_name
    run_dir = exp_dir / "run_1"

    if not run_dir.exists():
        logger.error(f"[ERRO] Diretorio run_1 nao existe: {run_dir}")
        return {
            'exists': False,
            'results_dict': False,
            'rules_history': False,
            'plots_dir': False,
            'plot_count': 0
        }

    logger.info(f"Diretorio: {run_dir}")
    logger.info("")

    # Verificar ResultsDictionary (necessario para generate_plots.py)
    results_dict_files = list(run_dir.glob("ResultsDictionary_*.pkl"))
    has_results_dict = len(results_dict_files) > 0

    if has_results_dict:
        check_file(results_dict_files[0], "ResultsDictionary.pkl")
    else:
        logger.warning(f"  [FALTA] ResultsDictionary_*.pkl")

    # Verificar RulesHistory (necessario para rule_diff_analyzer.py)
    rules_history_files = list(run_dir.glob("RulesHistory_*.txt"))
    has_rules_history = len(rules_history_files) > 0

    if has_rules_history:
        check_file(rules_history_files[0], "RulesHistory.txt")
    else:
        logger.warning(f"  [FALTA] RulesHistory_*.txt")

    # Verificar se plots ja foram gerados
    plots_dir = run_dir / "plots"
    has_plots_dir = plots_dir.exists()
    plot_count = 0

    if has_plots_dir:
        plot_files = list(plots_dir.glob("*.png"))
        plot_count = len(plot_files)
        logger.info(f"  [INFO] Pasta plots/ ja existe com {plot_count} arquivos PNG")

        # Listar plots
        for plot_file in plot_files:
            logger.info(f"    - {plot_file.name}")
    else:
        logger.info(f"  [INFO] Pasta plots/ ainda nao foi criada (sera criada pelo pos-processamento)")

    # Verificar outros arquivos
    test_pred_files = list(run_dir.glob("test_predictions_chunk*.csv"))
    train_pred_files = list(run_dir.glob("train_predictions_chunk*.csv"))

    logger.info(f"\n  Arquivos adicionais:")
    logger.info(f"    - test_predictions: {len(test_pred_files)} chunks")
    logger.info(f"    - train_predictions: {len(train_pred_files)} chunks")

    return {
        'exists': True,
        'results_dict': has_results_dict,
        'rules_history': has_rules_history,
        'plots_dir': has_plots_dir,
        'plot_count': plot_count
    }

def main():
    """Funcao principal"""
    logger.info("="*80)
    logger.info("DIAGNOSTICO BATCH 1 - VERIFICACAO DE ARQUIVOS")
    logger.info("="*80)
    logger.info(f"\nAmbiente: {'Google Colab' if IS_COLAB else 'Local'}")
    logger.info(f"Base DIR: {BASE_DIR}")
    logger.info(f"Results DIR: {RESULTS_DIR}")
    logger.info("")

    # Verificar arquivos globais
    logger.info("Verificando arquivos globais...")
    logger.info("-"*80)

    config_ok = check_file(CONFIG_FILE, "config_batch_1.yaml")

    analyze_script = BASE_DIR / "analyze_concept_difference.py"
    analyze_ok = check_file(analyze_script, "analyze_concept_difference.py")

    generate_plots_script = BASE_DIR / "generate_plots.py"
    plots_ok = check_file(generate_plots_script, "generate_plots.py")

    rule_diff_script = BASE_DIR / "rule_diff_analyzer.py"
    rule_ok = check_file(rule_diff_script, "rule_diff_analyzer.py")

    post_process_script = BASE_DIR / "post_process_batch_1.py"
    post_ok = check_file(post_process_script, "post_process_batch_1.py")

    logger.info("")

    if not all([config_ok, analyze_ok, plots_ok, rule_ok, post_ok]):
        logger.error("\n[ERRO] Alguns scripts essenciais estao faltando!")
        logger.error("Verifique se todos os arquivos foram copiados corretamente.")
        return 1

    # Verificar cada experimento
    resultados = {}

    for exp_name in BATCH_1_EXPERIMENTS:
        resultados[exp_name] = diagnostico_experimento(exp_name)

    # Resumo final
    logger.info("\n" + "="*80)
    logger.info("RESUMO DO DIAGNOSTICO")
    logger.info("="*80)
    logger.info("")

    logger.info(f"{'Experimento':<35} {'Run1':<8} {'Results':<8} {'Rules':<8} {'Plots':<8}")
    logger.info("-"*80)

    total = len(BATCH_1_EXPERIMENTS)
    count_exists = 0
    count_results = 0
    count_rules = 0
    count_plots = 0

    for exp_name, info in resultados.items():
        exists_str = "OK" if info['exists'] else "FALTA"
        results_str = "OK" if info['results_dict'] else "FALTA"
        rules_str = "OK" if info['rules_history'] else "FALTA"
        plots_str = f"{info['plot_count']}png" if info['plots_dir'] else "FALTA"

        logger.info(f"{exp_name:<35} {exists_str:<8} {results_str:<8} {rules_str:<8} {plots_str:<8}")

        if info['exists']:
            count_exists += 1
        if info['results_dict']:
            count_results += 1
        if info['rules_history']:
            count_rules += 1
        if info['plots_dir']:
            count_plots += 1

    logger.info("-"*80)
    logger.info(f"Total: {total} experimentos")
    logger.info(f"  - Com run_1/: {count_exists}/{total}")
    logger.info(f"  - Com ResultsDictionary: {count_results}/{total}")
    logger.info(f"  - Com RulesHistory: {count_rules}/{total}")
    logger.info(f"  - Com plots/: {count_plots}/{total}")

    logger.info("")
    logger.info("="*80)

    if count_exists < total:
        logger.error(f"\n[ATENCAO] {total - count_exists} experimentos nao foram executados ainda!")
        logger.error("Execute o experimento GBML primeiro antes do pos-processamento.")
        return 1

    if count_results < total or count_rules < total:
        logger.warning(f"\n[ATENCAO] Alguns experimentos estao incompletos:")
        if count_results < total:
            logger.warning(f"  - {total - count_results} sem ResultsDictionary (plots nao serao gerados)")
        if count_rules < total:
            logger.warning(f"  - {total - count_rules} sem RulesHistory (rule_diff nao sera executado)")
        logger.warning("\nO pos-processamento ira pular os experimentos incompletos.")
        return 0

    if count_plots == total:
        logger.info("\n[INFO] Todos os experimentos ja tem plots gerados!")
        logger.info("Execute post_process_batch_1.py para re-gerar se necessario.")
    else:
        logger.info(f"\n[OK] {count_exists} experimentos prontos para pos-processamento!")
        logger.info(f"Execute: python post_process_batch_1.py")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
