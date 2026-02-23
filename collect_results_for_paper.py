#!/usr/bin/env python3
"""
Data Collection Script for IEEE TKDE Paper - Section VI Enhancement

This script consolidates all experimental results from experiments_unified/ directory:
- EGIS results from chunk_metrics.json
- Comparative models from CSV files (ARF, SRP, HAT, ROSE, ACDWM, ERulesD2S)
- Rule complexity from rule_details_per_chunk.json
- Transition metrics calculated from RulesHistory files

Output: paper_data/consolidated_results.csv and related analysis files

Author: Automated Analysis
Date: 2026-01-27
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directory for experiments
EXPERIMENTS_BASE = Path("experiments_unified")

# Experiment configurations to process
EXPERIMENT_CONFIGS = {
    'chunk_500': {
        'chunk_size': 500,
        'penalty': 0.0,
        'label': 'EXP-500-NP'
    },
    'chunk_500_penalty': {
        'chunk_size': 500,
        'penalty': 0.1,
        'label': 'EXP-500-P'
    },
    'chunk_1000': {
        'chunk_size': 1000,
        'penalty': 0.0,
        'label': 'EXP-1000-NP'
    },
    'chunk_1000_penalty': {
        'chunk_size': 1000,
        'penalty': 0.1,
        'label': 'EXP-1000-P'
    },
    'chunk_2000': {
        'chunk_size': 2000,
        'penalty': 0.0,
        'label': 'EXP-2000-NP'
    },
    'chunk_2000_penalty': {
        'chunk_size': 2000,
        'penalty': 0.1,
        'label': 'EXP-2000-P'
    }
}

# Model result file mapping
MODEL_FILES = {
    'EGIS': 'chunk_metrics.json',
    'ARF': 'river_ARF_results.csv',
    'SRP': 'river_SRP_results.csv',
    'HAT': 'river_HAT_results.csv',
    'ROSE_Original': 'rose_original_results.csv',
    'ROSE_ChunkEval': 'rose_chunk_eval_results.csv',
    'ACDWM': 'acdwm_results.csv',
    'ERulesD2S': 'erulesd2s_results.csv',
    # CDCMS results are in a separate directory (cdcms_results/chunk_metrics.json)
    # Handled by extract_cdcms_results() below, not via extract_model_results()
}

# Dataset metadata
DATASET_METADATA = {
    # Batch 1 - Abrupt Drift (12 datasets)
    'SEA_Abrupt_Simple': {'drift_type': 'abrupt', 'batch': 'batch_1'},
    'SEA_Abrupt_Chain': {'drift_type': 'abrupt', 'batch': 'batch_1'},
    'SEA_Abrupt_Recurring': {'drift_type': 'abrupt', 'batch': 'batch_1'},
    'AGRAWAL_Abrupt_Simple_Mild': {'drift_type': 'abrupt', 'batch': 'batch_1'},
    'AGRAWAL_Abrupt_Simple_Severe': {'drift_type': 'abrupt', 'batch': 'batch_1'},
    'AGRAWAL_Abrupt_Chain_Long': {'drift_type': 'abrupt', 'batch': 'batch_1'},
    'RBF_Abrupt_Severe': {'drift_type': 'abrupt', 'batch': 'batch_1'},
    'RBF_Abrupt_Blip': {'drift_type': 'abrupt', 'batch': 'batch_1'},
    'STAGGER_Abrupt_Chain': {'drift_type': 'abrupt', 'batch': 'batch_1'},
    'STAGGER_Abrupt_Recurring': {'drift_type': 'abrupt', 'batch': 'batch_1'},
    'HYPERPLANE_Abrupt_Simple': {'drift_type': 'abrupt', 'batch': 'batch_1'},
    'RANDOMTREE_Abrupt_Simple': {'drift_type': 'abrupt', 'batch': 'batch_1'},

    # Batch 2 - Gradual Drift (9 datasets)
    'SEA_Gradual_Simple_Fast': {'drift_type': 'gradual', 'batch': 'batch_2'},
    'SEA_Gradual_Simple_Slow': {'drift_type': 'gradual', 'batch': 'batch_2'},
    'SEA_Gradual_Recurring': {'drift_type': 'gradual', 'batch': 'batch_2'},
    'STAGGER_Gradual_Chain': {'drift_type': 'gradual', 'batch': 'batch_2'},
    'RBF_Gradual_Moderate': {'drift_type': 'gradual', 'batch': 'batch_2'},
    'RBF_Gradual_Severe': {'drift_type': 'gradual', 'batch': 'batch_2'},
    'HYPERPLANE_Gradual_Simple': {'drift_type': 'gradual', 'batch': 'batch_2'},
    'RANDOMTREE_Gradual_Simple': {'drift_type': 'gradual', 'batch': 'batch_2'},
    'LED_Gradual_Simple': {'drift_type': 'gradual', 'batch': 'batch_2'},

    # Batch 2 Extended - Noisy (8 datasets)
    'SEA_Abrupt_Chain_Noise': {'drift_type': 'noisy', 'batch': 'batch_2'},
    'STAGGER_Abrupt_Chain_Noise': {'drift_type': 'noisy', 'batch': 'batch_2'},
    'AGRAWAL_Abrupt_Simple_Severe_Noise': {'drift_type': 'noisy', 'batch': 'batch_2'},
    'SINE_Abrupt_Recurring_Noise': {'drift_type': 'noisy', 'batch': 'batch_2'},
    'RBF_Abrupt_Blip_Noise': {'drift_type': 'noisy', 'batch': 'batch_2'},
    'RBF_Gradual_Severe_Noise': {'drift_type': 'noisy', 'batch': 'batch_2'},
    'HYPERPLANE_Gradual_Noise': {'drift_type': 'noisy', 'batch': 'batch_2'},
    'RANDOMTREE_Gradual_Noise': {'drift_type': 'noisy', 'batch': 'batch_2'},

    # Batch 2 Extended - Additional (6 datasets)
    'SINE_Abrupt_Simple': {'drift_type': 'abrupt', 'batch': 'batch_2'},
    'SINE_Gradual_Recurring': {'drift_type': 'gradual', 'batch': 'batch_2'},
    'LED_Abrupt_Simple': {'drift_type': 'abrupt', 'batch': 'batch_2'},
    'WAVEFORM_Abrupt_Simple': {'drift_type': 'abrupt', 'batch': 'batch_2'},
    'WAVEFORM_Gradual_Simple': {'drift_type': 'gradual', 'batch': 'batch_2'},
    'RANDOMTREE_Abrupt_Recurring': {'drift_type': 'abrupt', 'batch': 'batch_2'},

    # Batch 3 - Real-world and Stationary
    'Electricity': {'drift_type': 'real', 'batch': 'batch_3'},
    'Shuttle': {'drift_type': 'real', 'batch': 'batch_3'},
    'CovType': {'drift_type': 'real', 'batch': 'batch_3'},
    'PokerHand': {'drift_type': 'real', 'batch': 'batch_3'},
    'IntelLabSensors': {'drift_type': 'real', 'batch': 'batch_4'},

    # Stationary datasets
    'SEA_Stationary': {'drift_type': 'stationary', 'batch': 'batch_3'},
    'AGRAWAL_Stationary': {'drift_type': 'stationary', 'batch': 'batch_3'},
    'RBF_Stationary': {'drift_type': 'stationary', 'batch': 'batch_3'},
    'LED_Stationary': {'drift_type': 'stationary', 'batch': 'batch_3'},
    'HYPERPLANE_Stationary': {'drift_type': 'stationary', 'batch': 'batch_3'},
    'RANDOMTREE_Stationary': {'drift_type': 'stationary', 'batch': 'batch_3'},
    'STAGGER_Stationary': {'drift_type': 'stationary', 'batch': 'batch_3'},
    'WAVEFORM_Stationary': {'drift_type': 'stationary', 'batch': 'batch_3'},
    'SINE_Stationary': {'drift_type': 'stationary', 'batch': 'batch_3'},

    # Asset Negotiation datasets
    'AssetNegotiation_F2': {'drift_type': 'real', 'batch': 'batch_3'},
    'AssetNegotiation_F3': {'drift_type': 'real', 'batch': 'batch_3'},
    'AssetNegotiation_F4': {'drift_type': 'real', 'batch': 'batch_3'},
}

# Datasets excluded from all analyses (removed from paper)
EXCLUDED_DATASETS = ['CovType', 'IntelLabSensors', 'PokerHand', 'Shuttle']

# Transition metrics weights
W_INSTABILITY = 0.6
W_MODIFICATION_IMPACT = 0.4

# Output directory
OUTPUT_DIR = Path("paper_data")

# =============================================================================
# DATA EXTRACTION FUNCTIONS
# =============================================================================

def extract_egis_results(run_dir: Path) -> Optional[Dict]:
    """Extract EGIS results from chunk_metrics.json."""
    metrics_file = run_dir / "chunk_metrics.json"

    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        # Handle list format (typical)
        if isinstance(data, list):
            gmeans = [chunk.get('test_gmean', 0) for chunk in data]
            f1s = [chunk.get('test_f1', 0) for chunk in data]
            accs = [chunk.get('test_accuracy', chunk.get('accuracy', 0)) for chunk in data]
        elif isinstance(data, dict):
            gmeans = data.get('test_gmean', [])
            f1s = data.get('test_f1', [])
            accs = data.get('test_accuracy', [])
            if not isinstance(gmeans, list):
                gmeans = [gmeans]
                f1s = [f1s]
                accs = [accs]
        else:
            return None

        # Filter NaN values
        gmeans = [g for g in gmeans if g is not None and not np.isnan(g)]

        if not gmeans:
            return None

        return {
            'gmean_mean': np.mean(gmeans),
            'gmean_std': np.std(gmeans),
            'gmean_per_chunk': gmeans,
            'f1_mean': np.mean(f1s) if f1s else 0,
            'n_chunks': len(gmeans)
        }

    except Exception as e:
        logger.warning(f"Error reading {metrics_file}: {e}")
        return None


def extract_model_results(run_dir: Path, model: str, filename: str) -> Optional[Dict]:
    """Extract results from comparative model CSV files."""
    result_file = run_dir / filename

    if not result_file.exists():
        return None

    try:
        df = pd.read_csv(result_file)

        # Find G-Mean column
        gmean_col = None
        for col in ['test_gmean', 'gmean', 'g_mean', 'G-mean', 'G-Mean']:
            if col in df.columns:
                gmean_col = col
                break

        if gmean_col is None:
            # Try to find any column containing 'gmean'
            for col in df.columns:
                if 'gmean' in col.lower():
                    gmean_col = col
                    break

        if gmean_col is None:
            return None

        # Convert to numeric, coercing errors to NaN
        gmeans = pd.to_numeric(df[gmean_col], errors='coerce').dropna().values

        if len(gmeans) == 0:
            return None

        return {
            'gmean_mean': float(np.mean(gmeans)),
            'gmean_std': float(np.std(gmeans)),
            'gmean_per_chunk': [float(g) for g in gmeans],
            'n_chunks': len(gmeans)
        }

    except Exception as e:
        logger.warning(f"Error reading {result_file}: {e}")
        return None


def extract_cdcms_results(dataset_dir: Path) -> Optional[Dict]:
    """Extract CDCMS results from cdcms_results/chunk_metrics.json.

    CDCMS results are stored outside run_1/, in dataset_dir/cdcms_results/.
    The JSON format is a LIST of per-chunk objects:
      [{"chunk": 0, "holdout_gmean": null, ...}, {"chunk": 1, "holdout_gmean": 0.89, ...}, ...]

    Uses holdout_gmean for fair comparison with EGIS evaluation methodology.
    Falls back to prequential_gmean if all holdout values are null.
    """
    metrics_file = dataset_dir / "cdcms_results" / "chunk_metrics.json"

    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            # Real format: list of per-chunk dicts
            holdout_gmeans = []
            prequential_gmeans = []
            for chunk_obj in data:
                h = chunk_obj.get('holdout_gmean')
                if h is not None and not np.isnan(h):
                    holdout_gmeans.append(float(h))
                p = chunk_obj.get('prequential_gmean')
                if p is not None and not np.isnan(p):
                    prequential_gmeans.append(float(p))

            # Prefer holdout (comparable to EGIS), fallback to prequential
            per_chunk_gmeans = holdout_gmeans if holdout_gmeans else prequential_gmeans
        elif isinstance(data, dict):
            # Legacy/hypothetical dict format
            per_chunk_gmeans = []
            for cm in data.get('chunk_metrics', []):
                g = cm.get('holdout_gmean', cm.get('prequential_gmean'))
                if g is not None and not np.isnan(g):
                    per_chunk_gmeans.append(float(g))
        else:
            return None

        if not per_chunk_gmeans:
            return None

        return {
            'gmean_mean': float(np.mean(per_chunk_gmeans)),
            'gmean_std': float(np.std(per_chunk_gmeans)),
            'gmean_per_chunk': per_chunk_gmeans,
            'n_chunks': len(per_chunk_gmeans)
        }

    except Exception as e:
        logger.warning(f"Error reading CDCMS results {metrics_file}: {e}")
        return None


def parse_rules_history(run_dir: Path) -> List[Dict]:
    """Parse RulesHistory file to extract rules per chunk."""
    rules_files = list(run_dir.glob("RulesHistory_*.txt"))

    if not rules_files:
        return []

    rules_file = rules_files[0]

    try:
        with open(rules_file, 'r', encoding='utf-8') as f:
            content = f.read()

        chunks_data = []
        chunk_sections = re.split(r'--- Chunk \d+ \(Trained\) ---', content)[1:]

        for idx, section in enumerate(chunk_sections):
            chunk_data = {
                'chunk': idx,
                'rules': [],
                'n_rules': 0,
                'total_conditions': 0,
                'total_and_ops': 0,
                'total_or_ops': 0,
                'avg_conditions_per_rule': 0
            }

            # Extract rules
            rule_pattern = r'IF (.+?) THEN Class (\d+)'
            rules = re.findall(rule_pattern, section)

            chunk_data['n_rules'] = len(rules)

            total_conditions = 0
            total_and = 0
            total_or = 0

            for rule_text, class_label in rules:
                conditions = len(re.findall(r'[<>=]+', rule_text))
                total_conditions += conditions
                total_and += rule_text.count(' AND ')
                total_or += rule_text.count(' OR ')

                chunk_data['rules'].append({
                    'condition': rule_text,
                    'class': class_label,
                    'n_conditions': conditions
                })

            chunk_data['total_conditions'] = total_conditions
            chunk_data['total_and_ops'] = total_and
            chunk_data['total_or_ops'] = total_or
            chunk_data['avg_conditions_per_rule'] = total_conditions / len(rules) if rules else 0

            chunks_data.append(chunk_data)

        return chunks_data

    except Exception as e:
        logger.warning(f"Error parsing {rules_file}: {e}")
        return []


def calculate_rule_similarity(rule1: str, rule2: str) -> float:
    """Calculate similarity using normalized Levenshtein distance."""
    if rule1 == rule2:
        return 1.0

    len1, len2 = len(rule1), len(rule2)
    if len1 == 0 or len2 == 0:
        return 0.0

    try:
        from rapidfuzz.distance import Levenshtein as rf_lev
        return rf_lev.normalized_similarity(rule1, rule2)
    except ImportError:
        pass

    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if rule1[i-1] == rule2[j-1] else 1
            matrix[i][j] = min(
                matrix[i-1][j] + 1,
                matrix[i][j-1] + 1,
                matrix[i-1][j-1] + cost
            )

    distance = matrix[len1][len2]
    max_len = max(len1, len2)
    similarity = 1.0 - (distance / max_len)

    return similarity


def calculate_transition_metrics(chunks_data: List[Dict]) -> List[Dict]:
    """Calculate transition metrics (TCS, RIR, AMS) between consecutive chunks."""
    transitions = []
    MAX_RULES = 50
    SIMILARITY_THRESHOLD = 0.5
    SEVERITY_THRESHOLD = 0.8

    for i in range(len(chunks_data) - 1):
        chunk_i = chunks_data[i]
        chunk_j = chunks_data[i + 1]

        rules_i = [r['condition'] for r in chunk_i.get('rules', [])]
        rules_j = [r['condition'] for r in chunk_j.get('rules', [])]

        if not rules_i or not rules_j:
            transitions.append({
                'transition': f"{i}->{i+1}",
                'chunk_from': i,
                'chunk_to': i + 1,
                'RIR': 1.0,
                'AMS': 1.0,
                'TCS': 1.0,
                'unchanged_count': 0,
                'modified_count': 0,
                'new_count': len(rules_j),
                'deleted_count': len(rules_i)
            })
            continue

        # Identify unchanged rules
        unchanged = []
        rules_i_matched = set()
        rules_j_matched = set()

        for idx_i, rule_i in enumerate(rules_i):
            for idx_j, rule_j in enumerate(rules_j):
                if idx_j in rules_j_matched:
                    continue
                if rule_i == rule_j:
                    unchanged.append((idx_i, idx_j))
                    rules_i_matched.add(idx_i)
                    rules_j_matched.add(idx_j)
                    break

        # Identify modified rules
        modified_pairs = []
        severities = []

        n_unmatched_i = len(rules_i) - len(rules_i_matched)
        n_unmatched_j = len(rules_j) - len(rules_j_matched)
        skip_similarity = n_unmatched_i * n_unmatched_j > MAX_RULES ** 2

        if not skip_similarity:
            candidates = []
            for idx_i, rule_i in enumerate(rules_i):
                if idx_i in rules_i_matched:
                    continue
                for idx_j, rule_j in enumerate(rules_j):
                    if idx_j in rules_j_matched:
                        continue

                    similarity = calculate_rule_similarity(rule_i, rule_j)
                    if similarity >= SIMILARITY_THRESHOLD:
                        severity = 1.0 - similarity
                        if severity < SEVERITY_THRESHOLD:
                            candidates.append((idx_i, idx_j, similarity, severity))

            candidates.sort(key=lambda x: x[3])

            for idx_i, idx_j, sim, sev in candidates:
                if idx_i not in rules_i_matched and idx_j not in rules_j_matched:
                    modified_pairs.append((idx_i, idx_j, sev))
                    severities.append(sev)
                    rules_i_matched.add(idx_i)
                    rules_j_matched.add(idx_j)

        # Count categories
        new_count = len(rules_j) - len(rules_j_matched)
        deleted_count = len(rules_i) - len(rules_i_matched)
        unchanged_count = len(unchanged)
        modified_count = len(modified_pairs)

        # Calculate metrics
        total_rules = len(rules_i) + len(rules_j)

        RIR = (new_count + deleted_count) / total_rules if total_rules > 0 else 0.0
        AMS = np.mean(severities) if severities else 0.0

        prop_modified = modified_count / len(rules_j) if len(rules_j) > 0 else 0.0
        TCS = W_INSTABILITY * RIR + W_MODIFICATION_IMPACT * prop_modified * AMS
        TCS = min(max(TCS, 0.0), 1.0)

        transitions.append({
            'transition': f"{i}->{i+1}",
            'chunk_from': i,
            'chunk_to': i + 1,
            'RIR': round(RIR, 4),
            'AMS': round(AMS, 4),
            'TCS': round(TCS, 4),
            'unchanged_count': unchanged_count,
            'modified_count': modified_count,
            'new_count': new_count,
            'deleted_count': deleted_count,
            'total_rules_from': len(rules_i),
            'total_rules_to': len(rules_j)
        })

    return transitions


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_experiment_config(config_name: str, config: Dict, base_dir: Path) -> Dict:
    """Process a single experiment configuration."""
    logger.info(f"Processing configuration: {config_name} ({config['label']})")

    config_dir = base_dir / config_name

    if not config_dir.exists():
        logger.warning(f"Configuration directory not found: {config_dir}")
        return {'results': [], 'rules': [], 'transitions': []}

    all_results = []
    all_rules = []
    all_transitions = []

    # Process each batch
    for batch_dir in sorted(config_dir.iterdir()):
        if not batch_dir.is_dir() or not batch_dir.name.startswith('batch_'):
            continue

        batch_name = batch_dir.name

        # Process each dataset
        for dataset_dir in sorted(batch_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name

            # Skip excluded datasets
            if dataset_name in EXCLUDED_DATASETS:
                continue

            run_dir = dataset_dir / "run_1"

            if not run_dir.exists():
                continue

            # Get drift type from metadata
            meta = DATASET_METADATA.get(dataset_name, {'drift_type': 'unknown', 'batch': batch_name})
            drift_type = meta.get('drift_type', 'unknown')

            # Extract EGIS results
            egis_results = extract_egis_results(run_dir)
            if egis_results:
                all_results.append({
                    'config': config_name,
                    'config_label': config['label'],
                    'chunk_size': config['chunk_size'],
                    'penalty': config['penalty'],
                    'batch': batch_name,
                    'dataset': dataset_name,
                    'drift_type': drift_type,
                    'model': 'EGIS',
                    'gmean_mean': egis_results['gmean_mean'],
                    'gmean_std': egis_results['gmean_std'],
                    'n_chunks': egis_results['n_chunks']
                })

            # Extract comparative model results
            for model, filename in MODEL_FILES.items():
                if model == 'EGIS':
                    continue

                model_results = extract_model_results(run_dir, model, filename)
                if model_results:
                    all_results.append({
                        'config': config_name,
                        'config_label': config['label'],
                        'chunk_size': config['chunk_size'],
                        'penalty': config['penalty'],
                        'batch': batch_name,
                        'dataset': dataset_name,
                        'drift_type': drift_type,
                        'model': model,
                        'gmean_mean': model_results['gmean_mean'],
                        'gmean_std': model_results['gmean_std'],
                        'n_chunks': model_results['n_chunks']
                    })

            # Extract CDCMS results (stored in dataset_dir/cdcms_results/, not run_1/)
            cdcms_results = extract_cdcms_results(dataset_dir)
            if cdcms_results:
                all_results.append({
                    'config': config_name,
                    'config_label': config['label'],
                    'chunk_size': config['chunk_size'],
                    'penalty': config['penalty'],
                    'batch': batch_name,
                    'dataset': dataset_name,
                    'drift_type': drift_type,
                    'model': 'CDCMS',
                    'gmean_mean': cdcms_results['gmean_mean'],
                    'gmean_std': cdcms_results['gmean_std'],
                    'n_chunks': cdcms_results['n_chunks']
                })

            # Parse rules (skip transition metrics calculation for speed)
            # Transition metrics can be calculated separately if needed
            rules_data = parse_rules_history(run_dir)
            if rules_data:
                for chunk_data in rules_data:
                    all_rules.append({
                        'config': config_name,
                        'config_label': config['label'],
                        'batch': batch_name,
                        'dataset': dataset_name,
                        'drift_type': drift_type,
                        'chunk': chunk_data['chunk'],
                        'n_rules': chunk_data['n_rules'],
                        'total_conditions': chunk_data['total_conditions'],
                        'total_and_ops': chunk_data['total_and_ops'],
                        'total_or_ops': chunk_data['total_or_ops'],
                        'avg_conditions_per_rule': chunk_data['avg_conditions_per_rule']
                    })

                # Calculate real transition metrics (TCS, RIR, AMS)
                transitions = calculate_transition_metrics(rules_data)
                for t in transitions:
                    all_transitions.append({
                        'config': config_name,
                        'config_label': config['label'],
                        'batch': batch_name,
                        'dataset': dataset_name,
                        'drift_type': drift_type,
                        'transition': t['transition'],
                        'chunk_from': t['chunk_from'],
                        'chunk_to': t['chunk_to'],
                        'RIR': t['RIR'],
                        'AMS': t['AMS'],
                        'TCS': t['TCS'],
                        'rules_from': t.get('total_rules_from', 0),
                        'rules_to': t.get('total_rules_to', 0)
                    })

    return {
        'results': all_results,
        'rules': all_rules,
        'transitions': all_transitions
    }


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("DATA COLLECTION FOR IEEE TKDE PAPER - SECTION VI")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_rules = []
    all_transitions = []

    # Process each configuration
    for config_name, config in EXPERIMENT_CONFIGS.items():
        data = process_experiment_config(config_name, config, EXPERIMENTS_BASE)
        all_results.extend(data['results'])
        all_rules.extend(data['rules'])
        all_transitions.extend(data['transitions'])

        logger.info(f"  {config_name}: {len(data['results'])} results, "
                   f"{len(data['rules'])} rule records, {len(data['transitions'])} transitions")

    # Create DataFrames
    df_results = pd.DataFrame(all_results)
    df_rules = pd.DataFrame(all_rules)
    df_transitions = pd.DataFrame(all_transitions)

    # Save consolidated results
    if not df_results.empty:
        df_results.to_csv(OUTPUT_DIR / "consolidated_results.csv", index=False)
        logger.info(f"\nSaved: consolidated_results.csv ({len(df_results)} records)")

        # Create pivot table by model
        pivot = df_results.pivot_table(
            index=['config_label', 'dataset'],
            columns='model',
            values='gmean_mean',
            aggfunc='mean'
        )
        pivot.to_csv(OUTPUT_DIR / "pivot_gmean_by_model.csv")
        logger.info(f"Saved: pivot_gmean_by_model.csv")

        # Summary by config and model
        summary = df_results.groupby(['config_label', 'model']).agg({
            'gmean_mean': ['mean', 'std', 'count']
        }).round(4)
        summary.columns = ['gmean_avg', 'gmean_std', 'n_datasets']
        summary.to_csv(OUTPUT_DIR / "summary_by_config_model.csv")
        logger.info(f"Saved: summary_by_config_model.csv")

        # Summary by drift type and model
        drift_summary = df_results.groupby(['drift_type', 'model']).agg({
            'gmean_mean': ['mean', 'std', 'count']
        }).round(4)
        drift_summary.columns = ['gmean_avg', 'gmean_std', 'n_datasets']
        drift_summary.to_csv(OUTPUT_DIR / "summary_by_drift_model.csv")
        logger.info(f"Saved: summary_by_drift_model.csv")

    if not df_rules.empty:
        df_rules.to_csv(OUTPUT_DIR / "egis_rules_per_chunk.csv", index=False)
        logger.info(f"Saved: egis_rules_per_chunk.csv ({len(df_rules)} records)")

        # Rule complexity summary
        complexity_summary = df_rules.groupby(['config_label', 'dataset']).agg({
            'n_rules': 'mean',
            'total_conditions': 'mean',
            'avg_conditions_per_rule': 'mean',
            'total_and_ops': 'mean',
            'total_or_ops': 'mean'
        }).round(2)
        complexity_summary.to_csv(OUTPUT_DIR / "egis_complexity_summary.csv")
        logger.info(f"Saved: egis_complexity_summary.csv")

    if not df_transitions.empty:
        df_transitions.to_csv(OUTPUT_DIR / "egis_transition_metrics.csv", index=False)
        logger.info(f"Saved: egis_transition_metrics.csv ({len(df_transitions)} records)")

        # Transition metrics by drift type
        trans_summary = df_transitions.groupby(['config_label', 'drift_type']).agg({
            'TCS': ['mean', 'std'],
            'RIR': ['mean', 'std'],
            'AMS': ['mean', 'std']
        }).round(4)
        trans_summary.columns = ['_'.join(col) for col in trans_summary.columns]
        trans_summary.to_csv(OUTPUT_DIR / "transition_metrics_by_drift.csv")
        logger.info(f"Saved: transition_metrics_by_drift.csv")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("DATA COLLECTION SUMMARY")
    logger.info("=" * 80)

    if not df_results.empty:
        logger.info(f"\nTotal results: {len(df_results)}")
        logger.info(f"Unique datasets: {df_results['dataset'].nunique()}")
        logger.info(f"Unique models: {df_results['model'].nunique()}")
        logger.info(f"Configurations: {df_results['config_label'].nunique()}")

        logger.info("\n--- Performance Summary by Model (All Configs) ---")
        model_summary = df_results.groupby('model')['gmean_mean'].agg(['mean', 'std', 'count'])
        model_summary = model_summary.sort_values('mean', ascending=False)
        for model, row in model_summary.iterrows():
            logger.info(f"  {model:20s}: {row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])})")

        logger.info("\n--- EGIS Performance by Configuration ---")
        egis_results = df_results[df_results['model'] == 'EGIS']
        if not egis_results.empty:
            config_summary = egis_results.groupby('config_label')['gmean_mean'].agg(['mean', 'std', 'count'])
            for config, row in config_summary.iterrows():
                logger.info(f"  {config:15s}: {row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])})")

    if not df_transitions.empty:
        logger.info("\n--- Transition Metrics Summary ---")
        for metric in ['TCS', 'RIR', 'AMS']:
            mean_val = df_transitions[metric].mean()
            std_val = df_transitions[metric].std()
            logger.info(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

    logger.info(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
