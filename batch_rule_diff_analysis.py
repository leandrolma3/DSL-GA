#!/usr/bin/env python3
"""
Batch Rule Diff Analysis.

For each EGIS config x dataset, loads RulesHistory files and computes
rule evolution statistics (unchanged/modified/new/deleted counts,
similarity stats). Saves paper_data/rule_evolution_summary.csv.

Author: Automated Analysis
Date: 2026-02-23
"""

import os
import sys
import re
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    print("WARNING: Levenshtein package not found. Similarity calculations will be limited.")

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPERIMENTS_BASE = Path("experiments_unified")
OUTPUT_DIR = Path("paper_data")
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

SIMILARITY_THRESHOLD = 0.35  # Normalized Levenshtein distance threshold


def parse_rules_from_history(file_path):
    """Parse RulesHistory file into list of per-chunk rule lists."""
    if not file_path.exists():
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        chunk_sections = re.split(r'--- Chunk \d+ \(Trained\)', content)[1:]
        chunks = []

        for section in chunk_sections:
            rules = re.findall(r'IF (.+?) THEN Class (\S+)', section)
            chunk_rules = [cond.strip() for cond, _ in rules]
            chunks.append(chunk_rules)

        return chunks
    except Exception as e:
        logger.warning(f"Error parsing {file_path}: {e}")
        return None


def compare_consecutive_chunks(rules_i, rules_j):
    """Compare two rule sets and count unchanged/modified/new/deleted."""
    if not rules_i or not rules_j:
        return {
            'unchanged': 0, 'modified': 0, 'new': len(rules_j or []),
            'deleted': len(rules_i or []), 'avg_similarity': 0.0
        }

    # Find exact matches (unchanged)
    i_matched = set()
    j_matched = set()

    for idx_i, ri in enumerate(rules_i):
        for idx_j, rj in enumerate(rules_j):
            if idx_j in j_matched:
                continue
            if ri == rj:
                i_matched.add(idx_i)
                j_matched.add(idx_j)
                break

    unchanged = len(i_matched)

    # Find modified rules (similar but not identical)
    modified = 0
    similarities = []

    if HAS_LEVENSHTEIN:
        for idx_i, ri in enumerate(rules_i):
            if idx_i in i_matched:
                continue
            best_sim = 0.0
            best_j = -1
            for idx_j, rj in enumerate(rules_j):
                if idx_j in j_matched:
                    continue
                max_len = max(len(ri), len(rj), 1)
                dist = Levenshtein.distance(ri, rj)
                norm_dist = dist / max_len
                sim = 1.0 - norm_dist
                if norm_dist < SIMILARITY_THRESHOLD and sim > best_sim:
                    best_sim = sim
                    best_j = idx_j

            if best_j >= 0:
                modified += 1
                similarities.append(best_sim)
                i_matched.add(idx_i)
                j_matched.add(best_j)

    new_rules = len(rules_j) - len(j_matched)
    deleted_rules = len(rules_i) - len(i_matched)
    avg_sim = float(np.mean(similarities)) if similarities else 0.0

    return {
        'unchanged': unchanged,
        'modified': modified,
        'new': new_rules,
        'deleted': deleted_rules,
        'avg_similarity': avg_sim,
    }


def main():
    print("=" * 60)
    print("BATCH RULE DIFF ANALYSIS")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for config_name, config in EXPERIMENT_CONFIGS.items():
        config_dir = EXPERIMENTS_BASE / config_name
        if not config_dir.exists():
            print(f"  SKIP {config_name}: directory not found")
            continue

        n_processed = 0
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

                chunks = parse_rules_from_history(rules_files[0])
                if not chunks or len(chunks) < 2:
                    continue

                drift_type = DATASET_METADATA.get(dataset_name, 'unknown')

                # Compute diffs for all consecutive chunk pairs
                total_unchanged = 0
                total_modified = 0
                total_new = 0
                total_deleted = 0
                all_similarities = []

                for i in range(len(chunks) - 1):
                    diff = compare_consecutive_chunks(chunks[i], chunks[i + 1])
                    total_unchanged += diff['unchanged']
                    total_modified += diff['modified']
                    total_new += diff['new']
                    total_deleted += diff['deleted']
                    if diff['avg_similarity'] > 0:
                        all_similarities.append(diff['avg_similarity'])

                n_transitions = len(chunks) - 1

                records.append({
                    'config': config_name,
                    'config_label': config['label'],
                    'dataset': dataset_name,
                    'drift_type': drift_type,
                    'n_chunks': len(chunks),
                    'n_transitions': n_transitions,
                    'total_unchanged': total_unchanged,
                    'total_modified': total_modified,
                    'total_new': total_new,
                    'total_deleted': total_deleted,
                    'avg_unchanged_per_transition': total_unchanged / n_transitions,
                    'avg_modified_per_transition': total_modified / n_transitions,
                    'avg_new_per_transition': total_new / n_transitions,
                    'avg_deleted_per_transition': total_deleted / n_transitions,
                    'avg_similarity': float(np.mean(all_similarities)) if all_similarities else 0.0,
                    'avg_rules_per_chunk': np.mean([len(c) for c in chunks]),
                })
                n_processed += 1

        print(f"  {config['label']}: {n_processed} datasets processed")

    df = pd.DataFrame(records)
    output_path = OUTPUT_DIR / "rule_evolution_summary.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path} ({len(df)} records)")
    print(f"Configs: {df['config_label'].nunique()}")
    print(f"Datasets: {df['dataset'].nunique()}")

    # Summary
    print("\nRule evolution summary per config:")
    for cfg in sorted(df['config_label'].unique()):
        cfg_df = df[df['config_label'] == cfg]
        print(f"  {cfg}: avg_unchanged={cfg_df['avg_unchanged_per_transition'].mean():.1f}, "
              f"avg_modified={cfg_df['avg_modified_per_transition'].mean():.1f}, "
              f"avg_new={cfg_df['avg_new_per_transition'].mean():.1f}, "
              f"avg_deleted={cfg_df['avg_deleted_per_transition'].mean():.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
