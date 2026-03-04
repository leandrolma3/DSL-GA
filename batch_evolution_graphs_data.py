#!/usr/bin/env python3
"""
Batch Evolution Graphs Data Collection.

For each EGIS config x dataset, uses chunk_transition_analyzer to compute
detailed AST-based transition metrics (RIR/MI, AMS/SMM, TCS/STT) and
AST quantitative data. Saves paper_data/evolution_analysis_summary.csv.

Author: Automated Analysis
Date: 2026-02-23
"""

import os
import sys
import signal
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# Import AST-based analysis functions
from chunk_transition_analyzer import (
    parse_rules_history_to_asts,
    analyze_chunk_transition,
    collect_ast_quantitatives,
)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPERIMENTS_BASE = Path("experiments_unified")
OUTPUT_DIR = Path("paper_data")
EXCLUDED_DATASETS = ['CovType', 'IntelLabSensors', 'PokerHand', 'Shuttle']

# Thresholds matching generate_evolution_graphs.py
LEVENSHTEIN_SIMILARITY_PRE_FILTER = 0.5
SM_THRESHOLD_FOR_MODIFIED = 0.8

# NOTE: AST-based analysis is computationally expensive.
# Process only no-penalty configs by default to keep runtime reasonable.
# Set RUN_ALL_CONFIGS=True to process penalty configs too (will be much slower).
RUN_ALL_CONFIGS = True

_ALL_CONFIGS = {
    'chunk_500':            {'chunk_size': 500,  'penalty': 0.0, 'label': 'EXP-500-NP'},
    'chunk_500_penalty':    {'chunk_size': 500,  'penalty': 0.1, 'label': 'EXP-500-P'},
    'chunk_500_penalty_03': {'chunk_size': 500,  'penalty': 0.3, 'label': 'EXP-500-P03'},
    'chunk_1000':           {'chunk_size': 1000, 'penalty': 0.0, 'label': 'EXP-1000-NP'},
    'chunk_1000_penalty':   {'chunk_size': 1000, 'penalty': 0.1, 'label': 'EXP-1000-P'},
    'chunk_2000':           {'chunk_size': 2000, 'penalty': 0.0, 'label': 'EXP-2000-NP'},
    'chunk_2000_penalty':   {'chunk_size': 2000, 'penalty': 0.1, 'label': 'EXP-2000-P'},
}

_NP_CONFIGS = {
    'chunk_500':  {'chunk_size': 500,  'penalty': 0.0, 'label': 'EXP-500-NP'},
    'chunk_1000': {'chunk_size': 1000, 'penalty': 0.0, 'label': 'EXP-1000-NP'},
    'chunk_2000': {'chunk_size': 2000, 'penalty': 0.0, 'label': 'EXP-2000-NP'},
}

EXPERIMENT_CONFIGS = _ALL_CONFIGS if RUN_ALL_CONFIGS else _NP_CONFIGS

DATASET_METADATA = {
    'SEA_Abrupt_Simple': 'abrupt', 'SEA_Abrupt_Chain': 'abrupt', 'SEA_Abrupt_Recurring': 'abrupt',
    'AGRAWAL_Abrupt_Simple_Mild': 'abrupt', 'AGRAWAL_Abrupt_Simple_Severe': 'abrupt',
    'AGRAWAL_Abrupt_Chain_Long': 'abrupt', 'RBF_Abrupt_Severe': 'abrupt', 'RBF_Abrupt_Blip': 'abrupt',
    'STAGGER_Abrupt_Chain': 'abrupt', 'STAGGER_Abrupt_Recurring': 'abrupt',
    'HYPERPLANE_Abrupt_Simple': 'abrupt', 'RANDOMTREE_Abrupt_Simple': 'abrupt',
    'SINE_Abrupt_Simple': 'abrupt', 'LED_Abrupt_Simple': 'abrupt', 'WAVEFORM_Abrupt_Simple': 'abrupt',
    'RANDOMTREE_Abrupt_Recurring': 'abrupt',
    'SEA_Gradual_Simple_Fast': 'gradual', 'SEA_Gradual_Simple_Slow': 'gradual',
    'SEA_Gradual_Recurring': 'gradual', 'STAGGER_Gradual_Chain': 'gradual',
    'RBF_Gradual_Moderate': 'gradual', 'RBF_Gradual_Severe': 'gradual',
    'HYPERPLANE_Gradual_Simple': 'gradual', 'RANDOMTREE_Gradual_Simple': 'gradual',
    'LED_Gradual_Simple': 'gradual', 'SINE_Gradual_Recurring': 'gradual',
    'WAVEFORM_Gradual_Simple': 'gradual',
    'SEA_Abrupt_Chain_Noise': 'noisy', 'STAGGER_Abrupt_Chain_Noise': 'noisy',
    'AGRAWAL_Abrupt_Simple_Severe_Noise': 'noisy', 'SINE_Abrupt_Recurring_Noise': 'noisy',
    'RBF_Abrupt_Blip_Noise': 'noisy', 'RBF_Gradual_Severe_Noise': 'noisy',
    'HYPERPLANE_Gradual_Noise': 'noisy', 'RANDOMTREE_Gradual_Noise': 'noisy',
    'Electricity': 'real', 'AssetNegotiation_F2': 'real',
    'AssetNegotiation_F3': 'real', 'AssetNegotiation_F4': 'real',
    'SEA_Stationary': 'stationary', 'AGRAWAL_Stationary': 'stationary',
    'RBF_Stationary': 'stationary', 'LED_Stationary': 'stationary',
    'HYPERPLANE_Stationary': 'stationary', 'RANDOMTREE_Stationary': 'stationary',
    'STAGGER_Stationary': 'stationary', 'WAVEFORM_Stationary': 'stationary',
    'SINE_Stationary': 'stationary',
}


def collect_chunk_quantitatives(chunk_data):
    """Collect AST quantitative summary for a single chunk."""
    total_and = 0
    total_or = 0
    total_atomic = 0
    all_thresholds = []
    total_rules = 0

    for class_label, asts in chunk_data.get('rules_asts', {}).items():
        for ast in asts:
            total_rules += 1
            quants = collect_ast_quantitatives(ast)
            total_and += quants['and_count']
            total_or += quants['or_count']
            total_atomic += quants['atomic_condition_count']
            all_thresholds.extend(quants['threshold_values'])

    return {
        'total_rules': total_rules,
        'total_and_ops': total_and,
        'total_or_ops': total_or,
        'total_atomic_conditions': total_atomic,
        'avg_conditions_per_rule': total_atomic / total_rules if total_rules > 0 else 0,
        'n_thresholds': len(all_thresholds),
    }


def save_partial_results(transition_records, chunk_records, label="partial"):
    """Save current results incrementally."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if transition_records:
        df_trans = pd.DataFrame(transition_records)
        trans_path = OUTPUT_DIR / "evolution_analysis_summary.csv"
        df_trans.to_csv(trans_path, index=False)
        print(f"  [{label}] Saved: {trans_path} ({len(df_trans)} transition records)", flush=True)

    if chunk_records:
        df_chunks = pd.DataFrame(chunk_records)
        chunks_path = OUTPUT_DIR / "ast_chunk_quantitatives.csv"
        df_chunks.to_csv(chunks_path, index=False)
        print(f"  [{label}] Saved: {chunks_path} ({len(df_chunks)} chunk records)", flush=True)


def main():
    print("=" * 60)
    print("BATCH EVOLUTION GRAPHS DATA COLLECTION (AST-based)")
    print(f"Processing ALL {len(EXPERIMENT_CONFIGS)} configs")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transition_records = []
    chunk_records = []

    # Count total datasets for progress (same filters as main loop)
    total_datasets = 0
    for config_name in EXPERIMENT_CONFIGS:
        config_dir = EXPERIMENTS_BASE / config_name
        if not config_dir.exists():
            continue
        for batch_dir in sorted(config_dir.iterdir()):
            if not batch_dir.is_dir() or not batch_dir.name.startswith('batch_'):
                continue
            for dataset_dir in sorted(batch_dir.iterdir()):
                if not dataset_dir.is_dir() or dataset_dir.name in EXCLUDED_DATASETS:
                    continue
                run_dir = dataset_dir / "run_1"
                if run_dir.exists() and list(run_dir.glob("RulesHistory_*.txt")):
                    total_datasets += 1

    print(f"Total datasets across all configs: {total_datasets}")
    global_processed = 0

    # SIGINT handler: save partial results on Ctrl+C
    def sigint_handler(sig, frame):
        print(f"\n\nInterrupted! Saving {len(transition_records)} transitions and "
              f"{len(chunk_records)} chunk records...", flush=True)
        save_partial_results(transition_records, chunk_records, label="interrupted")
        sys.exit(1)

    signal.signal(signal.SIGINT, sigint_handler)

    for config_name, config in EXPERIMENT_CONFIGS.items():
        config_dir = EXPERIMENTS_BASE / config_name
        if not config_dir.exists():
            print(f"  SKIP {config_name}: directory not found")
            continue

        n_processed = 0
        n_errors = 0

        print(f"\n--- Config: {config['label']} ({config_name}) ---", flush=True)

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
                rules_files = list(run_dir.glob("RulesHistory_*.txt"))

                if not rules_files:
                    continue

                drift_type = DATASET_METADATA.get(dataset_name, 'unknown')

                try:
                    # Parse rules to ASTs (no file size limit)
                    chunks_data = parse_rules_history_to_asts(str(rules_files[0]))
                    if not chunks_data or len(chunks_data) < 2:
                        continue

                    chunk_indices = sorted(chunks_data.keys())

                    # Collect per-chunk quantitatives
                    for cidx in chunk_indices:
                        try:
                            quants = collect_chunk_quantitatives(chunks_data[cidx])
                            chunk_records.append({
                                'config': config_name,
                                'config_label': config['label'],
                                'dataset': dataset_name,
                                'drift_type': drift_type,
                                'chunk': cidx,
                                **quants,
                            })
                        except Exception as e:
                            logger.warning(f"Error in chunk quants {cidx} for {dataset_name}: {e}")

                    # Compute transitions
                    for i in range(len(chunk_indices) - 1):
                        ci = chunk_indices[i]
                        cj = chunk_indices[i + 1]

                        try:
                            summary, _details = analyze_chunk_transition(
                                chunks_data[ci], chunks_data[cj],
                                levenshtein_similarity_threshold=LEVENSHTEIN_SIMILARITY_PRE_FILTER,
                                sm_threshold_for_modified=SM_THRESHOLD_FOR_MODIFIED,
                            )

                            transition_records.append({
                                'config': config_name,
                                'config_label': config['label'],
                                'dataset': dataset_name,
                                'drift_type': drift_type,
                                'chunk_from': ci,
                                'chunk_to': cj,
                                'RIR': summary.get('MI', 0.0),
                                'AMS': summary.get('SMM', 0.0),
                                'TCS': summary.get('STT', 0.0),
                                'unchanged': summary.get('unchanged_count', 0),
                                'modified': summary.get('modified_count', 0),
                                'new': summary.get('new_count', 0),
                                'deleted': summary.get('deleted_count', 0),
                                'rules_from': summary.get('total_rules_i', 0),
                                'rules_to': summary.get('total_rules_j', 0),
                            })
                        except Exception as e:
                            logger.warning(f"Error in transition {ci}->{cj} for {dataset_name}: {e}")

                    n_processed += 1
                    global_processed += 1
                    if n_processed % 10 == 0:
                        pct = (global_processed / total_datasets * 100) if total_datasets > 0 else 0
                        print(f"    {config['label']}: {n_processed} datasets done "
                              f"(global: {global_processed}/{total_datasets} = {pct:.0f}%)", flush=True)

                except Exception as e:
                    logger.warning(f"Error processing {dataset_name} ({config_name}): {e}")
                    n_errors += 1
                    global_processed += 1

                sys.stdout.flush()

        print(f"  {config['label']}: {n_processed} processed, {n_errors} errors", flush=True)

        # Incremental save after each config
        save_partial_results(transition_records, chunk_records, label=config['label'])

    # Final save
    save_partial_results(transition_records, chunk_records, label="FINAL")

    # Summary
    df_trans = pd.DataFrame(transition_records) if transition_records else pd.DataFrame()
    if not df_trans.empty:
        print(f"\nConfigs: {df_trans['config_label'].nunique()}")
        print(f"Datasets: {df_trans['dataset'].nunique()}")
        print("\nAST-based Transition Metrics Summary per Config:")
        for cfg in sorted(df_trans['config_label'].unique()):
            cfg_df = df_trans[df_trans['config_label'] == cfg]
            print(f"  {cfg}: TCS={cfg_df['TCS'].mean():.4f}, "
                  f"RIR={cfg_df['RIR'].mean():.4f}, "
                  f"AMS={cfg_df['AMS'].mean():.4f}, "
                  f"N={len(cfg_df)}")

    print(f"\nTotal processed: {global_processed}/{total_datasets}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
