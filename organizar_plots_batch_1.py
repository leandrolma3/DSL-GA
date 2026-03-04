#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para Organizar Plots - Batch 1
Move os plots gerados durante o experimento para a subpasta plots/
"""

import os
import shutil
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
else:
    BASE_DIR = Path(__file__).parent
    RESULTS_DIR = BASE_DIR / "experiments_6chunks_phase2_gbml" / "batch_1"

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

# Padroes de arquivos que sao plots
PLOT_PATTERNS = [
    "*.png",
    "*Plot*.png",
    "*Evolution*.png",
    "*Heatmap*.png",
    "*Radar*.png"
]

# Arquivos que NAO devem ser movidos para plots/
EXCLUDE_PATTERNS = [
    "rule_diff_analysis_*",  # Ja sao analises, nao plots do experimento
]

def should_move_file(filename):
    """Verifica se um arquivo deve ser movido para plots/"""
    # Verificar exclusoes
    for pattern in EXCLUDE_PATTERNS:
        if pattern.replace('*', '') in filename:
            return False

    # Verificar se e um PNG
    return filename.endswith('.png')

def organizar_plots_experimento(exp_name):
    """Organiza plots de um experimento"""
    logger.info(f"\n{'='*80}")
    logger.info(f"ORGANIZANDO: {exp_name}")
    logger.info(f"{'='*80}")

    exp_dir = RESULTS_DIR / exp_name
    run_dir = exp_dir / "run_1"

    if not run_dir.exists():
        logger.error(f"[ERRO] Diretorio run_1 nao existe: {run_dir}")
        return {'moved': 0, 'skipped': 0, 'error': True}

    # Criar subpasta plots se nao existir
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    logger.info(f"Pasta plots/: {plots_dir}")

    # Encontrar todos os PNG no run_dir (nao em subdiretorios)
    png_files = [f for f in run_dir.iterdir() if f.is_file() and f.suffix == '.png']

    if not png_files:
        logger.warning(f"[ATENCAO] Nenhum arquivo PNG encontrado em {run_dir}")
        return {'moved': 0, 'skipped': 0, 'error': False}

    logger.info(f"Encontrados {len(png_files)} arquivos PNG")

    moved_count = 0
    skipped_count = 0

    for png_file in png_files:
        filename = png_file.name

        if should_move_file(filename):
            dest_file = plots_dir / filename

            # Verificar se arquivo ja existe no destino
            if dest_file.exists():
                logger.info(f"  [SKIP] {filename} (ja existe)")
                skipped_count += 1
            else:
                try:
                    # Copiar (nao mover) para manter original
                    shutil.copy2(png_file, dest_file)
                    logger.info(f"  [OK] {filename} -> plots/")
                    moved_count += 1
                except Exception as e:
                    logger.error(f"  [ERRO] Falha ao copiar {filename}: {e}")
        else:
            logger.info(f"  [SKIP] {filename} (exclusao)")
            skipped_count += 1

    logger.info(f"\nResumo: {moved_count} copiados, {skipped_count} pulados")
    return {'moved': moved_count, 'skipped': skipped_count, 'error': False}

def main():
    """Funcao principal"""
    logger.info("="*80)
    logger.info("ORGANIZACAO DE PLOTS - BATCH 1")
    logger.info("="*80)
    logger.info(f"\nAmbiente: {'Google Colab' if IS_COLAB else 'Local'}")
    logger.info(f"Base DIR: {BASE_DIR}")
    logger.info(f"Results DIR: {RESULTS_DIR}")
    logger.info("")

    if not RESULTS_DIR.exists():
        logger.error(f"Diretorio de resultados nao encontrado: {RESULTS_DIR}")
        return 1

    # Processar cada experimento
    resultados = {}

    for exp_name in BATCH_1_EXPERIMENTS:
        resultados[exp_name] = organizar_plots_experimento(exp_name)

    # Resumo final
    logger.info("\n" + "="*80)
    logger.info("RESUMO DA ORGANIZACAO")
    logger.info("="*80)
    logger.info("")

    total_moved = sum(r['moved'] for r in resultados.values())
    total_skipped = sum(r['skipped'] for r in resultados.values())
    total_errors = sum(1 for r in resultados.values() if r['error'])

    logger.info(f"{'Experimento':<35} {'Copiados':<12} {'Pulados':<10}")
    logger.info("-"*80)

    for exp_name, info in resultados.items():
        status = "ERRO" if info['error'] else "OK"
        logger.info(f"{exp_name:<35} {info['moved']:<12} {info['skipped']:<10}")

    logger.info("-"*80)
    logger.info(f"Total: {len(BATCH_1_EXPERIMENTS)} experimentos")
    logger.info(f"  - Plots copiados: {total_moved}")
    logger.info(f"  - Plots pulados: {total_skipped}")
    logger.info(f"  - Erros: {total_errors}")

    logger.info("")
    logger.info("="*80)

    if total_moved > 0:
        logger.info(f"\n[SUCESSO] {total_moved} plots organizados em subpastas plots/")
        logger.info("\nAgora voce pode:")
        logger.info("  1. Visualizar os plots em cada run_1/plots/")
        logger.info("  2. Executar rule_diff_analyzer se necessario")
        logger.info("  3. Prosseguir com analises comparativas")
    elif total_skipped > 0:
        logger.info(f"\n[INFO] {total_skipped} plots ja estavam organizados")
    else:
        logger.warning("\n[ATENCAO] Nenhum plot foi encontrado para organizar")
        logger.warning("Verifique se o experimento GBML foi executado corretamente")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
