#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Pos-Processamento - Batch 3 (8 bases Noise & Mixed)
Executa analyze_concept_difference.py, generate_plots.py e rule_diff_analyzer.py
para todos os experimentos do Batch 3
"""

import os
import sys
import subprocess
import glob
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuracoes - AJUSTAR CONFORME SEU AMBIENTE
# Se rodando no Colab, usar paths do Google Drive
# Se rodando localmente, usar paths locais

# Detectar ambiente
IS_COLAB = 'COLAB_GPU' in os.environ or '/content/' in os.getcwd()

if IS_COLAB:
    # Paths para Colab
    BASE_DIR = Path("/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid")
    RESULTS_DIR = BASE_DIR / "experiments_6chunks_phase2_gbml" / "batch_3"
    CONFIG_FILE = BASE_DIR / "configs" / "config_batch_3.yaml"
    HEATMAP_DIR = BASE_DIR / "experiments_6chunks_phase2_gbml" / "test_real_results_heatmaps" / "concept_heatmaps"
else:
    # Paths para local
    BASE_DIR = Path(__file__).parent
    RESULTS_DIR = BASE_DIR / "experiments_6chunks_phase2_gbml" / "batch_3"
    CONFIG_FILE = BASE_DIR / "configs" / "config_batch_3.yaml"
    HEATMAP_DIR = BASE_DIR / "experiments_6chunks_phase2_gbml" / "test_real_results_heatmaps" / "concept_heatmaps"

# Scripts de analise
ANALYZE_CONCEPT_DIFF_SCRIPT = BASE_DIR / "analyze_concept_difference.py"
GENERATE_PLOTS_SCRIPT = BASE_DIR / "generate_plots.py"
RULE_DIFF_ANALYZER_SCRIPT = BASE_DIR / "rule_diff_analyzer.py"

# Lista dos 8 experimentos do Batch 3 (Noise & Mixed)
BATCH_3_EXPERIMENTS = [
    "SEA_Abrupt_Chain_Noise",
    "STAGGER_Abrupt_Chain_Noise",
    "AGRAWAL_Abrupt_Simple_Severe_Noise",
    "SINE_Abrupt_Recurring_Noise",
    "RBF_Abrupt_Blip_Noise",
    "RBF_Gradual_Severe_Noise",
    "HYPERPLANE_Gradual_Noise",
    "RANDOMTREE_Gradual_Noise"
]

def run_command(cmd, description):
    """Executa um comando e retorna True se sucesso"""
    logger.info(f"[EXECUTANDO] {description}")
    logger.info(f"  Comando: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"[OK] {description} - Concluido com sucesso")
        if result.stdout:
            # Mostrar output completo para debug
            for line in result.stdout.split('\n')[:50]:  # Primeiras 50 linhas
                if line.strip():
                    logger.info(f"  | {line}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERRO] {description} - Falhou")
        logger.error(f"  Codigo de saida: {e.returncode}")
        if e.stdout:
            logger.error(f"  Stdout:")
            for line in e.stdout.split('\n')[:50]:
                if line.strip():
                    logger.error(f"    | {line}")
        if e.stderr:
            logger.error(f"  Stderr:")
            for line in e.stderr.split('\n')[:50]:
                if line.strip():
                    logger.error(f"    | {line}")
        return False
    except Exception as e:
        logger.error(f"[ERRO] {description} - Excecao: {e}")
        return False

def step1_analyze_concept_difference():
    """STEP 1: Executar analyze_concept_difference.py"""
    logger.info("="*80)
    logger.info("STEP 1: Analise de Diferenca entre Conceitos")
    logger.info("="*80)

    if not CONFIG_FILE.exists():
        logger.error(f"Config file nao encontrado: {CONFIG_FILE}")
        return False

    if not ANALYZE_CONCEPT_DIFF_SCRIPT.exists():
        logger.error(f"Script nao encontrado: {ANALYZE_CONCEPT_DIFF_SCRIPT}")
        return False

    # Importar e executar diretamente (mais confiavel que subprocess)
    logger.info(f"Importando analyze_concept_difference...")

    # Adicionar o BASE_DIR ao path para poder importar
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    try:
        import analyze_concept_difference

        logger.info(f"Executando analyze_concept_difference.main('{CONFIG_FILE}')")
        analyze_concept_difference.main(str(CONFIG_FILE))

        logger.info("[OK] Analise de Diferenca entre Conceitos - Concluido com sucesso")

        # Verificar se o JSON foi criado
        expected_json = HEATMAP_DIR / "concept_differences.json"
        if expected_json.exists():
            logger.info(f"[OK] concept_differences.json criado em: {expected_json}")
        else:
            logger.warning(f"[ATENCAO] concept_differences.json NAO encontrado em: {expected_json}")

        return True

    except Exception as e:
        logger.error(f"[ERRO] Analise de Diferenca entre Conceitos - Falhou: {e}", exc_info=True)
        return False

def step2_generate_plots():
    """STEP 2: Executar generate_plots.py para cada experimento"""
    logger.info("="*80)
    logger.info("STEP 2: Geracao de Plots para Cada Experimento")
    logger.info("="*80)

    if not GENERATE_PLOTS_SCRIPT.exists():
        logger.error(f"Script nao encontrado: {GENERATE_PLOTS_SCRIPT}")
        return False

    # Path para concept_differences.json
    concept_diff_json = HEATMAP_DIR / "concept_differences.json"

    if not concept_diff_json.exists():
        logger.warning(f"concept_differences.json nao encontrado em: {concept_diff_json}")
        logger.warning("Plots serao gerados sem informacao de severity")

    success_count = 0
    fail_count = 0

    for exp_name in BATCH_3_EXPERIMENTS:
        logger.info(f"\n--- Processando: {exp_name} ---")

        # Encontrar diretorio run_1 para este experimento
        exp_dir = RESULTS_DIR / exp_name
        run_dir = exp_dir / "run_1"

        if not run_dir.exists():
            logger.warning(f"Diretorio run_1 nao encontrado para {exp_name}: {run_dir}")
            fail_count += 1
            continue

        # Comando: python generate_plots.py <run_dir> -d <concept_diff_json>
        cmd = [
            sys.executable,
            str(GENERATE_PLOTS_SCRIPT),
            str(run_dir)
        ]

        if concept_diff_json.exists():
            cmd.extend(["-d", str(concept_diff_json)])

        if run_command(cmd, f"Generate plots para {exp_name}"):
            # Verificar se a pasta plots foi criada
            plots_dir = run_dir / "plots"
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))
                logger.info(f"  [OK] Pasta plots/ criada com {len(plot_files)} arquivos PNG")
            else:
                logger.warning(f"  [ATENCAO] Pasta plots/ NAO foi criada em {run_dir}")
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"\nResumo generate_plots: {success_count} sucessos, {fail_count} falhas")
    return fail_count == 0

def step3_rule_diff_analyzer():
    """STEP 3: Executar rule_diff_analyzer.py para cada experimento"""
    logger.info("="*80)
    logger.info("STEP 3: Analise de Diferencas de Regras para Cada Experimento")
    logger.info("="*80)

    if not RULE_DIFF_ANALYZER_SCRIPT.exists():
        logger.error(f"Script nao encontrado: {RULE_DIFF_ANALYZER_SCRIPT}")
        return False

    success_count = 0
    fail_count = 0

    for exp_name in BATCH_3_EXPERIMENTS:
        logger.info(f"\n--- Processando: {exp_name} ---")

        # Encontrar diretorio run_1 para este experimento
        exp_dir = RESULTS_DIR / exp_name
        run_dir = exp_dir / "run_1"

        if not run_dir.exists():
            logger.warning(f"Diretorio run_1 nao encontrado para {exp_name}: {run_dir}")
            fail_count += 1
            continue

        # Procurar arquivo RulesHistory_*.txt
        rules_history_files = list(run_dir.glob("RulesHistory_*.txt"))

        if not rules_history_files:
            logger.warning(f"RulesHistory nao encontrado em {run_dir}")
            fail_count += 1
            continue

        rules_history_file = rules_history_files[0]
        logger.info(f"  Encontrado: {rules_history_file.name}")

        # Output base path
        output_base = run_dir / f"rule_diff_analysis_{exp_name}"

        # Comando: python rule_diff_analyzer.py <rules_history> -o <output_base>
        cmd = [
            sys.executable,
            str(RULE_DIFF_ANALYZER_SCRIPT),
            str(rules_history_file),
            "-o", str(output_base),
            "-t", "0.35"  # threshold padrao
        ]

        if run_command(cmd, f"Rule diff analyzer para {exp_name}"):
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"\nResumo rule_diff_analyzer: {success_count} sucessos, {fail_count} falhas")
    return fail_count == 0

def main():
    """Funcao principal"""
    logger.info("="*80)
    logger.info("POS-PROCESSAMENTO BATCH 3 - 8 BASES NOISE & MIXED")
    logger.info("="*80)
    logger.info(f"\nAmbiente: {'Google Colab' if IS_COLAB else 'Local'}")
    logger.info(f"Base DIR: {BASE_DIR}")
    logger.info(f"Results DIR: {RESULTS_DIR}")
    logger.info(f"Config File: {CONFIG_FILE}")
    logger.info("")

    # Verificar diretorios
    if not RESULTS_DIR.exists():
        logger.error(f"Diretorio de resultados nao encontrado: {RESULTS_DIR}")
        logger.error("Por favor, ajuste as variaveis BASE_DIR e RESULTS_DIR no script")
        return 1

    results = {
        "step1_concept_diff": False,
        "step2_plots": False,
        "step3_rule_diff": False
    }

    # Executar steps
    try:
        results["step1_concept_diff"] = step1_analyze_concept_difference()
        results["step2_plots"] = step2_generate_plots()
        results["step3_rule_diff"] = step3_rule_diff_analyzer()
    except KeyboardInterrupt:
        logger.warning("\nInterrompido pelo usuario")
        return 130
    except Exception as e:
        logger.error(f"\nErro inesperado: {e}", exc_info=True)
        return 1

    # Resumo final
    logger.info("\n" + "="*80)
    logger.info("RESUMO FINAL")
    logger.info("="*80)
    for step, success in results.items():
        status = "[OK]" if success else "[FALHA]"
        logger.info(f"  {step:<25} {status}")

    all_success = all(results.values())

    if all_success:
        logger.info("\n[SUCESSO] Todos os steps de pos-processamento foram executados com sucesso!")
        return 0
    else:
        logger.warning("\n[ATENCAO] Alguns steps falharam. Verifique os logs acima.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
