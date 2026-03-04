#!/usr/bin/env python3
"""
Reprocess 4 repaired datasets through the per-dataset analysis pipeline.

Instead of running batch_per_dataset_analysis.py (which takes ~3h for all 336 items),
this script only processes the 4 datasets that were repaired after EXP-500 re-execution.

For each dataset, runs:
1. analyze_standard_drift.py -- drift detection + annotated accuracy plot
2. generate_plots.py -- 5 types of plots
3. rule_diff_analyzer.py -- rule diff report, evolution matrix CSV/PNG

Output: paper_data/per_dataset_analysis/{config_name}/{dataset_name}/
"""

import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR / "experiments_unified"
OUTPUT_BASE = BASE_DIR / "paper_data" / "per_dataset_analysis"

# The 4 repaired datasets
DATASETS_TO_REPROCESS = [
    {
        'config': 'chunk_500',
        'batch': 'batch_1',
        'dataset': 'SEA_Gradual_Simple_Fast',
    },
    {
        'config': 'chunk_500',
        'batch': 'batch_2',
        'dataset': 'RBF_Gradual_Severe_Noise',
    },
    {
        'config': 'chunk_500_penalty',
        'batch': 'batch_2',
        'dataset': 'WAVEFORM_Abrupt_Simple',
    },
    {
        'config': 'chunk_500_penalty',
        'batch': 'batch_3',
        'dataset': 'AGRAWAL_Stationary',
    },
]


def run_analyze_standard_drift(run_dir, output_dir):
    cmd = [
        sys.executable, "analyze_standard_drift.py",
        str(run_dir), "-t", "0.03", "-o", str(output_dir),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(BASE_DIR))
        if result.returncode != 0:
            logger.warning(f"  analyze_standard_drift failed: {result.stderr[:300]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"  analyze_standard_drift timed out for {run_dir}")
        return False
    except Exception as e:
        logger.warning(f"  analyze_standard_drift error: {e}")
        return False


def run_generate_plots(run_dir, output_dir):
    cmd = [
        sys.executable, "generate_plots.py",
        str(run_dir), "-o", str(output_dir),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=str(BASE_DIR))
        if result.returncode != 0:
            logger.warning(f"  generate_plots failed: {result.stderr[:300]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"  generate_plots timed out for {run_dir}")
        return False
    except Exception as e:
        logger.warning(f"  generate_plots error: {e}")
        return False


def run_rule_diff_analyzer(history_file, output_base):
    cmd = [
        sys.executable, "rule_diff_analyzer.py",
        str(history_file), "-t", "0.35", "-o", str(output_base),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(BASE_DIR))
        if result.returncode != 0:
            logger.warning(f"  rule_diff_analyzer failed: {result.stderr[:300]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"  rule_diff_analyzer timed out for {history_file}")
        return False
    except Exception as e:
        logger.warning(f"  rule_diff_analyzer error: {e}")
        return False


def main():
    print("=" * 70)
    print("REPROCESSING 4 REPAIRED DATASETS")
    print("=" * 70)

    errors = 0

    for item in DATASETS_TO_REPROCESS:
        config = item['config']
        batch = item['batch']
        dataset = item['dataset']

        run_dir = EXPERIMENTS_DIR / config / batch / dataset / "run_1"
        out_dir = OUTPUT_BASE / config / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing: {config}/{dataset}")
        print(f"  Run dir: {run_dir}")
        print(f"  Output:  {out_dir}")

        if not run_dir.exists():
            logger.error(f"  Run dir does not exist: {run_dir}")
            errors += 1
            continue

        # 1. analyze_standard_drift
        ok1 = run_analyze_standard_drift(run_dir, out_dir)
        print(f"  1. analyze_standard_drift: {'OK' if ok1 else 'FAIL'}")

        # 2. generate_plots
        ok2 = run_generate_plots(run_dir, out_dir)
        print(f"  2. generate_plots: {'OK' if ok2 else 'FAIL'}")

        # 3. rule_diff_analyzer
        rules_files = list(run_dir.glob("RulesHistory_*.txt"))
        ok3 = False
        if rules_files:
            output_base_path = str(out_dir / "rule_evolution")
            ok3 = run_rule_diff_analyzer(rules_files[0], output_base_path)
            print(f"  3. rule_diff_analyzer: {'OK' if ok3 else 'FAIL'}")
        else:
            logger.warning(f"  No RulesHistory file found in {run_dir}")
            print(f"  3. rule_diff_analyzer: SKIP (no RulesHistory)")

        if not (ok1 and ok2 and ok3):
            errors += 1

    print(f"\n{'=' * 70}")
    print(f"DONE: {len(DATASETS_TO_REPROCESS)} datasets, {errors} with errors")
    print(f"{'=' * 70}")

    return errors == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
