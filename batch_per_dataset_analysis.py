#!/usr/bin/env python3
"""
Batch Per-Dataset Analysis Pipeline (Item 9).

For each EGIS config (7) x dataset (48), runs:
1. analyze_standard_drift.py -- drift detection + annotated accuracy plot
2. generate_plots.py -- 5 types of plots (periodic gmean, GA evolution, etc.)
3. rule_diff_analyzer.py -- rule diff report, evolution matrix CSV/PNG

Output: paper_data/per_dataset_analysis/{config_name}/{dataset_name}/

Author: Automated Analysis
Date: 2026-02-23
"""

import os
import sys
import signal
import logging
import json
import subprocess
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

EXPERIMENTS_BASE = Path("experiments_unified")
OUTPUT_BASE = Path("paper_data") / "per_dataset_analysis"
EXCLUDED_DATASETS = ['CovType', 'IntelLabSensors', 'PokerHand', 'Shuttle']

EXPERIMENT_CONFIGS = {
    'chunk_500':            {'chunk_size': 500,  'penalty': 0.0, 'label': 'EXP-500-NP'},
    'chunk_500_penalty':    {'chunk_size': 500,  'penalty': 0.1, 'label': 'EXP-500-P'},
    'chunk_500_penalty_03': {'chunk_size': 500,  'penalty': 0.3, 'label': 'EXP-500-P03'},
    'chunk_1000':           {'chunk_size': 1000, 'penalty': 0.0, 'label': 'EXP-1000-NP'},
    'chunk_1000_penalty':   {'chunk_size': 1000, 'penalty': 0.1, 'label': 'EXP-1000-P'},
    'chunk_2000':           {'chunk_size': 2000, 'penalty': 0.0, 'label': 'EXP-2000-NP'},
    'chunk_2000_penalty':   {'chunk_size': 2000, 'penalty': 0.1, 'label': 'EXP-2000-P'},
}

# Concept differences file (if available)
CONCEPT_DIFF_FILE = "results/concept_heatmaps/concept_differences.json"


def run_analyze_standard_drift(run_dir, output_dir):
    """Run analyze_standard_drift.py via subprocess."""
    cmd = [
        sys.executable, "analyze_standard_drift.py",
        str(run_dir),
        "-t", "0.03",
        "-o", str(output_dir),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).parent)
        )
        if result.returncode != 0:
            logger.warning(f"  analyze_standard_drift failed: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"  analyze_standard_drift timed out for {run_dir}")
        return False
    except Exception as e:
        logger.warning(f"  analyze_standard_drift error: {e}")
        return False


def run_generate_plots(run_dir, output_dir):
    """Run generate_plots.py via subprocess."""
    cmd = [
        sys.executable, "generate_plots.py",
        str(run_dir),
        "-o", str(output_dir),
    ]
    # Add concept diff file if available
    diff_file = Path(CONCEPT_DIFF_FILE)
    if diff_file.exists():
        cmd.extend(["-d", str(diff_file)])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).parent)
        )
        if result.returncode != 0:
            logger.warning(f"  generate_plots failed: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"  generate_plots timed out for {run_dir}")
        return False
    except Exception as e:
        logger.warning(f"  generate_plots error: {e}")
        return False


def run_rule_diff_analyzer(history_file, output_base):
    """Run rule_diff_analyzer.py via subprocess."""
    cmd = [
        sys.executable, "rule_diff_analyzer.py",
        str(history_file),
        "-t", "0.35",
        "-o", str(output_base),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(Path(__file__).parent)
        )
        if result.returncode != 0:
            logger.warning(f"  rule_diff_analyzer failed: {result.stderr[:200]}")
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
    print("BATCH PER-DATASET ANALYSIS PIPELINE")
    print(f"Output: {OUTPUT_BASE}")
    print("=" * 70)

    # Count total work
    total_items = 0
    work_items = []
    for config_name, config in EXPERIMENT_CONFIGS.items():
        config_dir = EXPERIMENTS_BASE / config_name
        if not config_dir.exists():
            continue
        for batch_dir in sorted(config_dir.iterdir()):
            if not batch_dir.is_dir() or not batch_dir.name.startswith('batch_'):
                continue
            for dataset_dir in sorted(batch_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                dataset_name = dataset_dir.name
                if dataset_name in EXCLUDED_DATASETS:
                    continue
                run_dir = dataset_dir / "run_1"
                if not run_dir.exists():
                    continue
                work_items.append((config_name, config, dataset_name, run_dir))
                total_items += 1

    print(f"Total items to process: {total_items}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    processed = 0
    errors = 0
    results_log = []

    # SIGINT handler
    def sigint_handler(sig, frame):
        print(f"\nInterrupted after {processed}/{total_items} items. Saving log...", flush=True)
        _save_log(results_log)
        sys.exit(1)

    signal.signal(signal.SIGINT, sigint_handler)

    for config_name, config, dataset_name, run_dir in work_items:
        processed += 1
        config_label = config['label']
        out_dir = OUTPUT_BASE / config_name / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        pct = (processed / total_items * 100) if total_items > 0 else 0
        print(f"[{processed}/{total_items} ({pct:.0f}%)] {config_label} / {dataset_name}",
              flush=True)

        item_result = {
            'config': config_name,
            'config_label': config_label,
            'dataset': dataset_name,
            'drift_analysis': False,
            'plots': False,
            'rule_diff': False,
        }

        # 1. analyze_standard_drift
        item_result['drift_analysis'] = run_analyze_standard_drift(run_dir, out_dir)

        # 2. generate_plots
        item_result['plots'] = run_generate_plots(run_dir, out_dir)

        # 3. rule_diff_analyzer
        rules_files = list(run_dir.glob("RulesHistory_*.txt"))
        if rules_files:
            output_base_path = str(out_dir / "rule_evolution")
            item_result['rule_diff'] = run_rule_diff_analyzer(rules_files[0], output_base_path)
        else:
            logger.warning(f"  No RulesHistory file found in {run_dir}")

        results_log.append(item_result)

        if not (item_result['drift_analysis'] and item_result['plots'] and item_result['rule_diff']):
            errors += 1

        # Progress report every 20 items
        if processed % 20 == 0:
            print(f"  --- Progress: {processed}/{total_items}, errors so far: {errors} ---", flush=True)

    # Save summary
    _save_log(results_log)

    print(f"\n{'=' * 70}")
    print(f"COMPLETED: {processed}/{total_items} items, {errors} with errors")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    return 0


def _save_log(results_log):
    """Save processing log to CSV."""
    if not results_log:
        return
    import pandas as pd
    df_log = pd.DataFrame(results_log)
    log_path = OUTPUT_BASE / "processing_log.csv"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    df_log.to_csv(log_path, index=False)
    print(f"Saved processing log: {log_path} ({len(df_log)} records)")


if __name__ == "__main__":
    sys.exit(main())
