#!/usr/bin/env python3
"""
Unified Analysis Script for IEEE Paper

This script consolidates all experimental results from three configurations:
- EXP-A: chunk_size=1000, PENALTY_WEIGHT=0.0 (baseline)
- EXP-B: chunk_size=2000, PENALTY_WEIGHT=0.0 (larger window)
- EXP-C: chunk_size=2000, PENALTY_WEIGHT=0.1 (balanced)

Extracts:
- Performance metrics (G-Mean, F1, Accuracy) from all models
- Interpretability metrics (rules, conditions) from GBML and ERulesD2S
- Transition metrics (TCS, RIR, AMS) from GBML only

Outputs:
- Consolidated CSV files for each experiment
- LaTeX tables ready for paper
- Data files for figure generation
- Statistical analysis results

Author: Automated Analysis
Date: 2025-12-16
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

# Optional imports for statistical analysis
try:
    from scipy import stats
    from scipy.stats import friedmanchisquare, wilcoxon, rankdata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available. Statistical tests will be skipped.")

# Optional imports for figure generation
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server/script use
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    # Set style for IEEE paper
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'serif'
except ImportError:
    PLOTTING_AVAILABLE = False
    print("WARNING: matplotlib/seaborn not available. Figure generation will be skipped.")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Experiment configurations
EXPERIMENT_CONFIGS = {
    'exp_a_chunk1000': {
        'dirs': {
            'batch_1': 'experiments_6chunks_phase2_gbml/batch_1',
            'batch_2': 'experiments_6chunks_phase2_gbml/batch_2',
            'batch_3': 'experiments_6chunks_phase2_gbml/batch_3',
            'batch_4': 'experiments_6chunks_phase2_gbml/batch_4',
            'batch_5': 'experiments_6chunks_phase3_real/batch_5',
            'batch_6': 'experiments_6chunks_phase3_real/batch_6',
            'batch_7': 'experiments_6chunks_phase3_real/batch_7',
        },
        'chunk_size': 1000,
        'penalty_weight': 0.0,
        'description': 'Baseline configuration (chunk_size=1000)'
    },
    'exp_b_chunk2000': {
        'dirs': {
            'batch_1': 'experiments_chunk2000_phase1/batch_1',
            'batch_2': 'experiments_chunk2000_phase1/batch_2',
            'batch_3': 'experiments_chunk2000_phase1/batch_3',
            'batch_4': 'experiments_chunk2000_phase1/batch_4',
            'batch_5': 'experiments_chunk2000_phase2/batch_5',
            'batch_6': 'experiments_chunk2000_phase2/batch_6',
            'batch_7': 'experiments_chunk2000_phase2/batch_7',
        },
        'chunk_size': 2000,
        'penalty_weight': 0.0,
        'description': 'Larger evaluation window (chunk_size=2000)'
    },
    'exp_c_balanced': {
        'dirs': {
            'batch_1': 'experiments_balanced_phase1/batch_1',
            'batch_2': 'experiments_balanced_phase1/batch_2',
            'batch_3': 'experiments_balanced_phase1/batch_3',
            'batch_4': 'experiments_balanced_phase1/batch_4',
            'batch_5': 'experiments_balanced_phase2/batch_5',
            'batch_6': 'experiments_balanced_phase2/batch_6',
            'batch_7': 'experiments_balanced_phase2/batch_7',
        },
        'chunk_size': 2000,
        'penalty_weight': 0.1,
        'description': 'Performance-complexity balance (PENALTY_WEIGHT=0.1)'
    }
}

# Models configuration
MODELS = [
    'GBML',           # Proposed method (results from GBML experiments)
    'ROSE_Original',  # ROSE prequential evaluation
    'ROSE_ChunkEval', # ROSE chunk-based evaluation
    'HAT',            # Hoeffding Adaptive Tree
    'ARF',            # Adaptive Random Forest
    'SRP',            # Streaming Random Patches
    'ACDWM',          # Adaptive Chunk-based DWM (binary only)
    'ERulesD2S'       # Evolutionary Rules for Data Streams
]

# Output directory
OUTPUT_BASE_DIR = Path('analysis_output')

# Dataset configuration (from Execute_All_Comparative_Models.ipynb)
BATCH_DATASETS = {
    'batch_1': [
        'SEA_Abrupt_Simple', 'SEA_Abrupt_Chain', 'SEA_Abrupt_Recurring',
        'AGRAWAL_Abrupt_Simple_Mild', 'AGRAWAL_Abrupt_Simple_Severe', 'AGRAWAL_Abrupt_Chain_Long',
        'RBF_Abrupt_Severe', 'RBF_Abrupt_Blip',
        'STAGGER_Abrupt_Chain', 'STAGGER_Abrupt_Recurring',
        'HYPERPLANE_Abrupt_Simple', 'RANDOMTREE_Abrupt_Simple'
    ],  # 12 datasets - Abrupt Drift
    'batch_2': [
        'SEA_Gradual_Simple_Fast', 'SEA_Gradual_Simple_Slow', 'SEA_Gradual_Recurring',
        'STAGGER_Gradual_Chain',
        'RBF_Gradual_Moderate', 'RBF_Gradual_Severe',
        'HYPERPLANE_Gradual_Simple', 'RANDOMTREE_Gradual_Simple', 'LED_Gradual_Simple'
    ],  # 9 datasets - Gradual Drift
    'batch_3': [
        'SEA_Abrupt_Chain_Noise', 'STAGGER_Abrupt_Chain_Noise',
        'AGRAWAL_Abrupt_Simple_Severe_Noise', 'SINE_Abrupt_Recurring_Noise',
        'RBF_Abrupt_Blip_Noise', 'RBF_Gradual_Severe_Noise',
        'HYPERPLANE_Gradual_Noise', 'RANDOMTREE_Gradual_Noise'
    ],  # 8 datasets - Drift with Noise
    'batch_4': [
        'SINE_Abrupt_Simple', 'SINE_Gradual_Recurring',
        'LED_Abrupt_Simple',
        'WAVEFORM_Abrupt_Simple', 'WAVEFORM_Gradual_Simple',
        'RANDOMTREE_Abrupt_Recurring'
    ],  # 6 datasets - SINE/LED/WAVEFORM
    'batch_5': [
        'Electricity', 'Shuttle', 'CovType', 'PokerHand', 'IntelLabSensors'
    ],  # 5 datasets - Real-world
    'batch_6': [
        'SEA_Stationary', 'AGRAWAL_Stationary', 'RBF_Stationary',
        'LED_Stationary', 'HYPERPLANE_Stationary', 'RANDOMTREE_Stationary'
    ],  # 6 datasets - Synthetic Stationary
    'batch_7': [
        'STAGGER_Stationary', 'WAVEFORM_Stationary', 'SINE_Stationary',
        'AssetNegotiation_F2', 'AssetNegotiation_F3', 'AssetNegotiation_F4'
    ]   # 6 datasets - Synthetic Stationary
}
# Total: 52 datasets

# Dataset metadata for paper
DATASET_METADATA = {
    # Batch 1 - Abrupt Drift
    'SEA_Abrupt_Simple': {'generator': 'SEA', 'drift_type': 'abrupt', 'n_features': 3, 'n_classes': 2},
    'SEA_Abrupt_Chain': {'generator': 'SEA', 'drift_type': 'abrupt', 'n_features': 3, 'n_classes': 2},
    'SEA_Abrupt_Recurring': {'generator': 'SEA', 'drift_type': 'abrupt', 'n_features': 3, 'n_classes': 2},
    'AGRAWAL_Abrupt_Simple_Mild': {'generator': 'AGRAWAL', 'drift_type': 'abrupt', 'n_features': 9, 'n_classes': 2},
    'AGRAWAL_Abrupt_Simple_Severe': {'generator': 'AGRAWAL', 'drift_type': 'abrupt', 'n_features': 9, 'n_classes': 2},
    'AGRAWAL_Abrupt_Chain_Long': {'generator': 'AGRAWAL', 'drift_type': 'abrupt', 'n_features': 9, 'n_classes': 2},
    'RBF_Abrupt_Severe': {'generator': 'RBF', 'drift_type': 'abrupt', 'n_features': 10, 'n_classes': 5},
    'RBF_Abrupt_Blip': {'generator': 'RBF', 'drift_type': 'abrupt', 'n_features': 10, 'n_classes': 5},
    'STAGGER_Abrupt_Chain': {'generator': 'STAGGER', 'drift_type': 'abrupt', 'n_features': 3, 'n_classes': 2},
    'STAGGER_Abrupt_Recurring': {'generator': 'STAGGER', 'drift_type': 'abrupt', 'n_features': 3, 'n_classes': 2},
    'HYPERPLANE_Abrupt_Simple': {'generator': 'HYPERPLANE', 'drift_type': 'abrupt', 'n_features': 10, 'n_classes': 2},
    'RANDOMTREE_Abrupt_Simple': {'generator': 'RANDOMTREE', 'drift_type': 'abrupt', 'n_features': 10, 'n_classes': 2},
    # Batch 2 - Gradual Drift
    'SEA_Gradual_Simple_Fast': {'generator': 'SEA', 'drift_type': 'gradual', 'n_features': 3, 'n_classes': 2},
    'SEA_Gradual_Simple_Slow': {'generator': 'SEA', 'drift_type': 'gradual', 'n_features': 3, 'n_classes': 2},
    'SEA_Gradual_Recurring': {'generator': 'SEA', 'drift_type': 'gradual', 'n_features': 3, 'n_classes': 2},
    'STAGGER_Gradual_Chain': {'generator': 'STAGGER', 'drift_type': 'gradual', 'n_features': 3, 'n_classes': 2},
    'RBF_Gradual_Moderate': {'generator': 'RBF', 'drift_type': 'gradual', 'n_features': 10, 'n_classes': 5},
    'RBF_Gradual_Severe': {'generator': 'RBF', 'drift_type': 'gradual', 'n_features': 10, 'n_classes': 5},
    'HYPERPLANE_Gradual_Simple': {'generator': 'HYPERPLANE', 'drift_type': 'gradual', 'n_features': 10, 'n_classes': 2},
    'RANDOMTREE_Gradual_Simple': {'generator': 'RANDOMTREE', 'drift_type': 'gradual', 'n_features': 10, 'n_classes': 2},
    'LED_Gradual_Simple': {'generator': 'LED', 'drift_type': 'gradual', 'n_features': 24, 'n_classes': 10},
    # Batch 3 - Drift with Noise
    'SEA_Abrupt_Chain_Noise': {'generator': 'SEA', 'drift_type': 'abrupt_noise', 'n_features': 3, 'n_classes': 2},
    'STAGGER_Abrupt_Chain_Noise': {'generator': 'STAGGER', 'drift_type': 'abrupt_noise', 'n_features': 3, 'n_classes': 2},
    'AGRAWAL_Abrupt_Simple_Severe_Noise': {'generator': 'AGRAWAL', 'drift_type': 'abrupt_noise', 'n_features': 9, 'n_classes': 2},
    'SINE_Abrupt_Recurring_Noise': {'generator': 'SINE', 'drift_type': 'abrupt_noise', 'n_features': 2, 'n_classes': 2},
    'RBF_Abrupt_Blip_Noise': {'generator': 'RBF', 'drift_type': 'abrupt_noise', 'n_features': 10, 'n_classes': 5},
    'RBF_Gradual_Severe_Noise': {'generator': 'RBF', 'drift_type': 'gradual_noise', 'n_features': 10, 'n_classes': 5},
    'HYPERPLANE_Gradual_Noise': {'generator': 'HYPERPLANE', 'drift_type': 'gradual_noise', 'n_features': 10, 'n_classes': 2},
    'RANDOMTREE_Gradual_Noise': {'generator': 'RANDOMTREE', 'drift_type': 'gradual_noise', 'n_features': 10, 'n_classes': 2},
    # Batch 4 - SINE/LED/WAVEFORM
    'SINE_Abrupt_Simple': {'generator': 'SINE', 'drift_type': 'abrupt', 'n_features': 2, 'n_classes': 2},
    'SINE_Gradual_Recurring': {'generator': 'SINE', 'drift_type': 'gradual', 'n_features': 2, 'n_classes': 2},
    'LED_Abrupt_Simple': {'generator': 'LED', 'drift_type': 'abrupt', 'n_features': 24, 'n_classes': 10},
    'WAVEFORM_Abrupt_Simple': {'generator': 'WAVEFORM', 'drift_type': 'abrupt', 'n_features': 40, 'n_classes': 3},
    'WAVEFORM_Gradual_Simple': {'generator': 'WAVEFORM', 'drift_type': 'gradual', 'n_features': 40, 'n_classes': 3},
    'RANDOMTREE_Abrupt_Recurring': {'generator': 'RANDOMTREE', 'drift_type': 'abrupt', 'n_features': 10, 'n_classes': 2},
    # Batch 5 - Real-world
    'Electricity': {'generator': 'Real', 'drift_type': 'real', 'n_features': 8, 'n_classes': 2},
    'Shuttle': {'generator': 'Real', 'drift_type': 'real', 'n_features': 9, 'n_classes': 7},
    'CovType': {'generator': 'Real', 'drift_type': 'real', 'n_features': 54, 'n_classes': 7},
    'PokerHand': {'generator': 'Real', 'drift_type': 'real', 'n_features': 10, 'n_classes': 10},
    'IntelLabSensors': {'generator': 'Real', 'drift_type': 'real', 'n_features': 4, 'n_classes': 2},
    # Batch 6 - Synthetic Stationary
    'SEA_Stationary': {'generator': 'SEA', 'drift_type': 'stationary', 'n_features': 3, 'n_classes': 2},
    'AGRAWAL_Stationary': {'generator': 'AGRAWAL', 'drift_type': 'stationary', 'n_features': 9, 'n_classes': 2},
    'RBF_Stationary': {'generator': 'RBF', 'drift_type': 'stationary', 'n_features': 10, 'n_classes': 5},
    'LED_Stationary': {'generator': 'LED', 'drift_type': 'stationary', 'n_features': 24, 'n_classes': 10},
    'HYPERPLANE_Stationary': {'generator': 'HYPERPLANE', 'drift_type': 'stationary', 'n_features': 10, 'n_classes': 2},
    'RANDOMTREE_Stationary': {'generator': 'RANDOMTREE', 'drift_type': 'stationary', 'n_features': 10, 'n_classes': 2},
    # Batch 7 - Synthetic Stationary
    'STAGGER_Stationary': {'generator': 'STAGGER', 'drift_type': 'stationary', 'n_features': 3, 'n_classes': 2},
    'WAVEFORM_Stationary': {'generator': 'WAVEFORM', 'drift_type': 'stationary', 'n_features': 40, 'n_classes': 3},
    'SINE_Stationary': {'generator': 'SINE', 'drift_type': 'stationary', 'n_features': 2, 'n_classes': 2},
    'AssetNegotiation_F2': {'generator': 'AssetNegotiation', 'drift_type': 'stationary', 'n_features': 5, 'n_classes': 2},
    'AssetNegotiation_F3': {'generator': 'AssetNegotiation', 'drift_type': 'stationary', 'n_features': 5, 'n_classes': 2},
    'AssetNegotiation_F4': {'generator': 'AssetNegotiation', 'drift_type': 'stationary', 'n_features': 5, 'n_classes': 2},
}

# Transition metrics weights (as per original implementation)
W_INSTABILITY = 0.6
W_MODIFICATION_IMPACT = 0.4

# Batch categories for analysis by drift type
BATCH_CATEGORIES = {
    'batch_1': {'category': 'Abrupt Drift', 'description': 'Abrupt concept drift scenarios'},
    'batch_2': {'category': 'Gradual Drift', 'description': 'Gradual concept drift scenarios'},
    'batch_3': {'category': 'Drift with Noise', 'description': 'Concept drift with added noise'},
    'batch_4': {'category': 'Mixed Generators', 'description': 'SINE, LED, WAVEFORM generators'},
    'batch_5': {'category': 'Real-world', 'description': 'Real-world benchmark datasets'},
    'batch_6': {'category': 'Stationary', 'description': 'Stationary synthetic datasets'},
    'batch_7': {'category': 'Stationary', 'description': 'Stationary synthetic datasets'}
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA EXTRACTION FUNCTIONS
# =============================================================================

def extract_gbml_performance(dataset_dir: Path) -> Optional[Dict]:
    """
    Extract GBML performance metrics from chunk_metrics.json

    Args:
        dataset_dir: Path to dataset directory containing run_1/

    Returns:
        Dictionary with performance metrics or None if not found
    """
    metrics_file = dataset_dir / "run_1" / "chunk_metrics.json"

    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        # Handle different formats
        if isinstance(data, list):
            # List of chunk metrics
            gmeans = [chunk.get('test_gmean', chunk.get('gmean', 0)) for chunk in data]
            f1s = [chunk.get('test_f1', chunk.get('f1', 0)) for chunk in data]
            accs = [chunk.get('test_accuracy', chunk.get('accuracy', 0)) for chunk in data]
        elif isinstance(data, dict):
            # Dictionary format
            gmeans = data.get('test_gmean', data.get('gmean', []))
            f1s = data.get('test_f1', data.get('f1', []))
            accs = data.get('test_accuracy', data.get('accuracy', []))

            if not isinstance(gmeans, list):
                gmeans = [gmeans]
                f1s = [f1s]
                accs = [accs]
        else:
            return None

        # Calculate mean metrics
        return {
            'gmean': np.mean(gmeans) if gmeans else 0.0,
            'f1': np.mean(f1s) if f1s else 0.0,
            'accuracy': np.mean(accs) if accs else 0.0,
            'gmean_per_chunk': gmeans,
            'n_chunks': len(gmeans)
        }

    except Exception as e:
        logger.warning(f"Error reading {metrics_file}: {e}")
        return None


def parse_rules_history(dataset_dir: Path) -> List[Dict]:
    """
    Parse RulesHistory file to extract rules per chunk.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        List of dictionaries with rules per chunk
    """
    run_dir = dataset_dir / "run_1"
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

            # Extract rules (lines starting with IF)
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
            chunk_data['avg_conditions_per_rule'] = (
                total_conditions / len(rules) if rules else 0
            )

            chunks_data.append(chunk_data)

        return chunks_data

    except Exception as e:
        logger.warning(f"Error parsing {rules_file}: {e}")
        return []


def calculate_rule_similarity(rule1: str, rule2: str) -> float:
    """
    Calculate similarity between two rules using normalized Levenshtein distance.
    """
    if rule1 == rule2:
        return 1.0

    len1, len2 = len(rule1), len(rule2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Distance matrix
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
    """
    Calculate transition metrics (TCS, RIR, AMS) between consecutive chunks.

    Args:
        chunks_data: List of chunk data from parse_rules_history

    Returns:
        List of transition metrics
    """
    transitions = []
    MAX_RULES_FOR_SIMILARITY = 50
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
        skip_similarity = (
            n_unmatched_i * n_unmatched_j > MAX_RULES_FOR_SIMILARITY ** 2
        )

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

        # RIR (Rule Instability Rate)
        RIR = (new_count + deleted_count) / total_rules if total_rules > 0 else 0.0

        # AMS (Average Modification Severity)
        AMS = np.mean(severities) if severities else 0.0

        # TCS (Transition Change Score)
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


def extract_comparative_results(dataset_dir: Path) -> Dict[str, Dict]:
    """
    Extract results from comparative models (ROSE, HAT, ARF, SRP, ACDWM, ERulesD2S).

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        Dictionary with results per model
    """
    results = {}
    run_dir = dataset_dir / "run_1"

    # Model result files mapping
    model_files = {
        'ROSE_Original': 'rose_original_results.csv',
        'ROSE_ChunkEval': 'rose_chunk_eval_results.csv',
        'HAT': 'river_HAT_results.csv',
        'ARF': 'river_ARF_results.csv',
        'SRP': 'river_SRP_results.csv',
        'ACDWM': 'acdwm_results.csv',
        'ERulesD2S': 'erulesd2s_results.csv'
    }

    for model, filename in model_files.items():
        result_file = run_dir / filename

        if not result_file.exists():
            # Try alternative locations
            alt_file = dataset_dir / filename
            if alt_file.exists():
                result_file = alt_file
            else:
                continue

        try:
            df = pd.read_csv(result_file)

            # Extract G-Mean (handle different column names)
            # River models use 'test_gmean', others use 'gmean' variants
            gmean_cols = ['test_gmean', 'gmean', 'g_mean', 'G-mean', 'geometric_mean', 'G-Mean']
            gmean = None
            for col in gmean_cols:
                if col in df.columns:
                    gmean = df[col].mean()
                    break

            if gmean is None:
                # Fallback: search for any column containing 'gmean' (case-insensitive)
                for col in df.columns:
                    if 'gmean' in col.lower():
                        gmean = df[col].mean()
                        break

            # Extract accuracy (River models use 'test_accuracy')
            acc_cols = ['test_accuracy', 'accuracy', 'acc', 'Accuracy']
            accuracy = None
            for col in acc_cols:
                if col in df.columns:
                    accuracy = df[col].mean()
                    break

            if accuracy is None:
                # Fallback: search for any column containing 'accuracy' (case-insensitive)
                for col in df.columns:
                    if 'accuracy' in col.lower():
                        accuracy = df[col].mean()
                        break

            results[model] = {
                'gmean': gmean if gmean is not None else 0.0,
                'accuracy': accuracy if accuracy is not None else 0.0,
                'status': 'OK'
            }

        except Exception as e:
            logger.warning(f"Error reading {result_file}: {e}")
            results[model] = {
                'gmean': 0.0,
                'accuracy': 0.0,
                'status': 'ERROR'
            }

    return results


def extract_erulesd2s_complexity(dataset_dir: Path) -> Optional[Dict]:
    """
    Extract complexity metrics from ERulesD2S results.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        Dictionary with complexity metrics or None
    """
    run_dir = dataset_dir / "run_1"

    # Look for ERulesD2S results
    erulesd2s_file = run_dir / "erulesd2s_results.csv"

    if not erulesd2s_file.exists():
        return None

    try:
        df = pd.read_csv(erulesd2s_file)

        return {
            'n_rules': df['NumberRules'].mean() if 'NumberRules' in df.columns else 0,
            'n_conditions': df['NumberConditions'].mean() if 'NumberConditions' in df.columns else 0,
            'n_nodes': df['NumberNodes'].mean() if 'NumberNodes' in df.columns else 0
        }

    except Exception as e:
        logger.warning(f"Error reading ERulesD2S complexity: {e}")
        return None


# =============================================================================
# DATASET CHARACTERISTICS TABLE
# =============================================================================

def generate_dataset_characteristics_table(output_path: Path) -> str:
    """
    Generate LaTeX table with dataset characteristics for the paper (Table 1).

    Args:
        output_path: Path to save the LaTeX file

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Dataset Characteristics}")
    lines.append("\\label{tab:datasets}")
    lines.append("\\footnotesize")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Batch & Dataset & Generator & Features & Classes & Drift Type \\\\")
    lines.append("\\midrule")

    current_batch = None

    for batch_name, datasets in BATCH_DATASETS.items():
        batch_info = BATCH_CATEGORIES.get(batch_name, {})
        batch_category = batch_info.get('category', batch_name)

        for i, dataset in enumerate(datasets):
            meta = DATASET_METADATA.get(dataset, {})

            # Only show batch name for first dataset in batch
            batch_display = batch_category if i == 0 else ""

            # Format drift type
            drift_type = meta.get('drift_type', 'unknown').replace('_', ' ').title()

            lines.append(f"{batch_display} & {dataset} & {meta.get('generator', '-')} & "
                        f"{meta.get('n_features', '-')} & {meta.get('n_classes', '-')} & "
                        f"{drift_type} \\\\")

        lines.append("\\midrule")

    # Remove last midrule and add bottomrule
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    latex_content = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(latex_content)

    return latex_content


def generate_per_dataset_performance_table(
    pivot_data: pd.DataFrame,
    models: List[str],
    exp_name: str,
    output_path: Path,
    max_rows_per_table: int = 25
) -> List[str]:
    """
    Generate LaTeX tables with per-dataset G-Mean values for all models.

    Args:
        pivot_data: DataFrame with datasets as index, models as columns
        models: List of model names
        exp_name: Experiment name
        output_path: Base path for output files
        max_rows_per_table: Maximum rows per table (for pagination)

    Returns:
        List of LaTeX table strings
    """
    available_models = [m for m in models if m in pivot_data.columns]

    if not available_models:
        return []

    # Sort by GBML performance if available
    if 'GBML' in pivot_data.columns:
        pivot_data = pivot_data.sort_values('GBML', ascending=False)

    tables = []
    n_tables = (len(pivot_data) + max_rows_per_table - 1) // max_rows_per_table

    for table_idx in range(n_tables):
        start_idx = table_idx * max_rows_per_table
        end_idx = min((table_idx + 1) * max_rows_per_table, len(pivot_data))
        subset = pivot_data.iloc[start_idx:end_idx]

        lines = []
        lines.append("\\begin{table*}[htbp]")
        lines.append("\\centering")

        if n_tables > 1:
            lines.append(f"\\caption{{G-Mean Performance by Dataset - {exp_name} (Part {table_idx + 1}/{n_tables})}}")
            lines.append(f"\\label{{tab:gmean_{exp_name.lower().replace(' ', '_')}_{table_idx + 1}}}")
        else:
            lines.append(f"\\caption{{G-Mean Performance by Dataset - {exp_name}}}")
            lines.append(f"\\label{{tab:gmean_{exp_name.lower().replace(' ', '_')}}}")

        lines.append("\\footnotesize")

        # Create column specification
        col_spec = "l" + "c" * len(available_models)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Header row
        header = "Dataset & " + " & ".join(available_models) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        # Find best model for each row
        for dataset in subset.index:
            row_values = subset.loc[dataset, available_models]
            best_model = row_values.idxmax() if not row_values.isna().all() else None

            # Format row
            cells = [dataset[:25]]  # Truncate long names
            for model in available_models:
                val = subset.loc[dataset, model]
                if pd.isna(val):
                    cells.append("-")
                elif model == best_model:
                    cells.append(f"\\textbf{{{val:.4f}}}")
                else:
                    cells.append(f"{val:.4f}")

            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\midrule")

        # Add average row
        avg_cells = ["\\textbf{Average}"]
        for model in available_models:
            avg = subset[model].mean()
            avg_cells.append(f"{avg:.4f}" if not pd.isna(avg) else "-")
        lines.append(" & ".join(avg_cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")

        table_content = "\n".join(lines)
        tables.append(table_content)

        # Save individual table
        suffix = f"_{table_idx + 1}" if n_tables > 1 else ""
        table_path = output_path.parent / f"{output_path.stem}{suffix}{output_path.suffix}"
        with open(table_path, 'w') as f:
            f.write(table_content)

    return tables


def generate_performance_by_drift_type_table(
    results_df: pd.DataFrame,
    models: List[str],
    exp_name: str,
    output_path: Path
) -> str:
    """
    Generate LaTeX table with performance grouped by drift type.

    Args:
        results_df: DataFrame with results
        models: List of model names
        exp_name: Experiment name
        output_path: Path to save the table

    Returns:
        LaTeX table string
    """
    if results_df.empty:
        return ""

    # Add batch category
    results_df = results_df.copy()
    results_df['category'] = results_df['batch'].map(
        lambda x: BATCH_CATEGORIES.get(x, {}).get('category', 'Unknown')
    )

    # Aggregate by category and model
    agg = results_df.groupby(['category', 'model'])['gmean'].agg(['mean', 'std']).reset_index()
    pivot_mean = agg.pivot(index='category', columns='model', values='mean')
    pivot_std = agg.pivot(index='category', columns='model', values='std')

    available_models = [m for m in models if m in pivot_mean.columns]

    # Order categories
    category_order = ['Abrupt Drift', 'Gradual Drift', 'Drift with Noise',
                      'Mixed Generators', 'Real-world', 'Stationary']
    categories = [c for c in category_order if c in pivot_mean.index]

    lines = []
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{G-Mean Performance by Drift Type - {exp_name}}}")
    lines.append(f"\\label{{tab:gmean_by_drift_{exp_name.lower().replace(' ', '_')}}}")
    lines.append("\\footnotesize")

    col_spec = "l" + "c" * len(available_models)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header = "Drift Type & " + " & ".join(available_models) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for category in categories:
        if category in pivot_mean.index:
            row_values = pivot_mean.loc[category, available_models]
            best_model = row_values.idxmax() if not row_values.isna().all() else None

            cells = [category]
            for model in available_models:
                mean_val = pivot_mean.loc[category, model]
                std_val = pivot_std.loc[category, model] if category in pivot_std.index else 0

                if pd.isna(mean_val):
                    cells.append("-")
                elif model == best_model:
                    cells.append(f"\\textbf{{{mean_val:.3f}}} ({std_val:.3f})")
                else:
                    cells.append(f"{mean_val:.3f} ({std_val:.3f})")

            lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    latex_content = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(latex_content)

    return latex_content


def load_existing_consolidated_results(dataset_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load existing consolidated results from previous comparative model runs.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        DataFrame with consolidated results or None
    """
    possible_files = [
        'all_models_consolidated_results.csv',
        'consolidated_results.csv',
        'comparison_results.csv'
    ]

    for filename in possible_files:
        # Check in dataset directory
        filepath = dataset_dir / filename
        if filepath.exists():
            try:
                return pd.read_csv(filepath)
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")

        # Check in run_1 subdirectory
        filepath = dataset_dir / 'run_1' / filename
        if filepath.exists():
            try:
                return pd.read_csv(filepath)
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")

    return None


# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def friedman_test(data: pd.DataFrame, models: List[str]) -> Dict:
    """
    Perform Friedman test for comparing multiple classifiers.

    Args:
        data: DataFrame with columns for each model's G-Mean per dataset
        models: List of model names to compare

    Returns:
        Dictionary with test results
    """
    if not SCIPY_AVAILABLE:
        return {'error': 'scipy not available'}

    # Prepare data for Friedman test
    model_data = []
    for model in models:
        if model in data.columns:
            model_data.append(data[model].values)

    if len(model_data) < 3:
        return {'error': 'Need at least 3 models for Friedman test'}

    try:
        statistic, p_value = friedmanchisquare(*model_data)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_models': len(model_data),
            'n_datasets': len(model_data[0])
        }

    except Exception as e:
        return {'error': str(e)}


def calculate_average_ranks(data: pd.DataFrame, models: List[str]) -> Dict[str, float]:
    """
    Calculate average ranks for each model across datasets.

    Args:
        data: DataFrame with G-Mean values
        models: List of model names

    Returns:
        Dictionary with average rank per model
    """
    available_models = [m for m in models if m in data.columns]

    if not available_models:
        return {}

    # Calculate ranks for each dataset (higher G-Mean = rank 1)
    ranks_df = data[available_models].rank(axis=1, ascending=False)

    # Calculate average ranks
    avg_ranks = ranks_df.mean().to_dict()

    return avg_ranks


def wilcoxon_test(data: pd.DataFrame, model1: str, model2: str) -> Dict:
    """
    Perform Wilcoxon signed-rank test for paired comparison.

    Args:
        data: DataFrame with G-Mean values
        model1, model2: Models to compare

    Returns:
        Dictionary with test results
    """
    if not SCIPY_AVAILABLE:
        return {'error': 'scipy not available'}

    if model1 not in data.columns or model2 not in data.columns:
        return {'error': f'Model not found in data'}

    try:
        x = data[model1].values
        y = data[model2].values

        # Remove pairs with NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        if len(x) < 5:
            return {'error': 'Not enough data points'}

        statistic, p_value = wilcoxon(x, y)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_diff': np.mean(x - y),
            'n_pairs': len(x)
        }

    except Exception as e:
        return {'error': str(e)}


def nemenyi_critical_difference(n_models: int, n_datasets: int, alpha: float = 0.05) -> float:
    """
    Calculate Nemenyi critical difference.

    Args:
        n_models: Number of models being compared
        n_datasets: Number of datasets
        alpha: Significance level

    Returns:
        Critical difference value
    """
    # q-alpha values for Nemenyi test (for alpha=0.05)
    q_alpha = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }

    q = q_alpha.get(n_models, 3.0)

    cd = q * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))

    return cd


# =============================================================================
# OUTPUT GENERATION FUNCTIONS
# =============================================================================

def generate_latex_performance_table(
    data: pd.DataFrame,
    models: List[str],
    exp_name: str,
    output_path: Path
) -> str:
    """
    Generate LaTeX table for performance comparison.

    Args:
        data: DataFrame with G-Mean per model per dataset
        models: List of model names
        exp_name: Experiment name for caption
        output_path: Path to save the table

    Returns:
        LaTeX table string
    """
    available_models = [m for m in models if m in data.columns]

    # Calculate statistics
    stats = {}
    for model in available_models:
        values = data[model].dropna()
        stats[model] = {
            'mean': values.mean(),
            'std': values.std(),
            'n': len(values)
        }

    # Find best model
    best_model = max(stats.keys(), key=lambda m: stats[m]['mean'])

    # Generate LaTeX
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Performance Comparison - {exp_name}}}")
    lines.append(f"\\label{{tab:performance_{exp_name.lower().replace(' ', '_')}}}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Model & G-Mean & Std. Dev. \\\\")
    lines.append("\\midrule")

    # Sort by mean G-Mean descending
    sorted_models = sorted(available_models, key=lambda m: stats[m]['mean'], reverse=True)

    for model in sorted_models:
        mean = stats[model]['mean']
        std = stats[model]['std']

        # Bold for best model
        if model == best_model:
            lines.append(f"\\textbf{{{model}}} & \\textbf{{{mean:.4f}}} & {std:.4f} \\\\")
        else:
            lines.append(f"{model} & {mean:.4f} & {std:.4f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_content = "\n".join(lines)

    # Save to file
    with open(output_path, 'w') as f:
        f.write(latex_content)

    return latex_content


def generate_latex_statistical_table(
    friedman_result: Dict,
    wilcoxon_results: Dict,
    avg_ranks: Dict,
    output_path: Path
) -> str:
    """
    Generate LaTeX table for statistical tests results.
    """
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Statistical Significance Tests}")
    lines.append("\\label{tab:statistical_tests}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Comparison & Statistic & p-value & Significant & Mean Diff. \\\\")
    lines.append("\\midrule")

    # Friedman test
    if 'statistic' in friedman_result:
        sig = "Yes" if friedman_result['significant'] else "No"
        lines.append(f"Friedman (overall) & {friedman_result['statistic']:.2f} & "
                    f"{friedman_result['p_value']:.4f} & {sig} & - \\\\")
        lines.append("\\midrule")

    # Wilcoxon tests (vs proposed method)
    for comparison, result in wilcoxon_results.items():
        if 'statistic' in result:
            sig = "Yes" if result['significant'] else "No"
            lines.append(f"{comparison} & {result['statistic']:.2f} & "
                        f"{result['p_value']:.4f} & {sig} & {result['mean_diff']:.4f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_content = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(latex_content)

    return latex_content


def generate_latex_complexity_table(
    gbml_complexity: pd.DataFrame,
    erulesd2s_complexity: pd.DataFrame,
    output_path: Path
) -> str:
    """
    Generate LaTeX table for interpretability metrics comparison.
    """
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Interpretability Metrics Comparison}")
    lines.append("\\label{tab:complexity}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Model & Avg. Rules & Avg. Conditions & Conditions/Rule \\\\")
    lines.append("\\midrule")

    # GBML statistics
    if not gbml_complexity.empty:
        avg_rules = gbml_complexity['n_rules'].mean()
        avg_cond = gbml_complexity['total_conditions'].mean()
        cond_per_rule = gbml_complexity['avg_conditions_per_rule'].mean()
        lines.append(f"Proposed Method & {avg_rules:.2f} & {avg_cond:.2f} & {cond_per_rule:.2f} \\\\")

    # ERulesD2S statistics
    if not erulesd2s_complexity.empty:
        avg_rules = erulesd2s_complexity['n_rules'].mean()
        avg_cond = erulesd2s_complexity['n_conditions'].mean()
        cond_per_rule = avg_cond / avg_rules if avg_rules > 0 else 0
        lines.append(f"ERulesD2S & {avg_rules:.2f} & {avg_cond:.2f} & {cond_per_rule:.2f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_content = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(latex_content)

    return latex_content


def generate_latex_transition_table(
    transitions: pd.DataFrame,
    output_path: Path
) -> str:
    """
    Generate LaTeX table for transition metrics by drift type.
    """
    if transitions.empty:
        return ""

    # Classify by drift type
    transitions['drift_type'] = transitions['dataset'].apply(
        lambda x: 'Abrupt' if 'Abrupt' in x else ('Gradual' if 'Gradual' in x else 'Stationary')
    )

    # Aggregate by drift type
    agg = transitions.groupby('drift_type').agg({
        'TCS': ['mean', 'std'],
        'RIR': ['mean', 'std'],
        'AMS': ['mean', 'std']
    }).round(4)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Transition Metrics by Drift Type}")
    lines.append("\\label{tab:transition_metrics}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Drift Type & TCS & RIR & AMS \\\\")
    lines.append("\\midrule")

    for drift_type in ['Abrupt', 'Gradual', 'Stationary']:
        if drift_type in agg.index:
            tcs = f"{agg.loc[drift_type, ('TCS', 'mean')]:.4f}"
            rir = f"{agg.loc[drift_type, ('RIR', 'mean')]:.4f}"
            ams = f"{agg.loc[drift_type, ('AMS', 'mean')]:.4f}"
            lines.append(f"{drift_type} & {tcs} & {rir} & {ams} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_content = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(latex_content)

    return latex_content


# =============================================================================
# FIGURE GENERATION FUNCTIONS
# =============================================================================

def generate_performance_boxplot(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Performance Comparison"
) -> None:
    """
    Generate boxplot comparing G-Mean across all models.

    Args:
        df: DataFrame with columns ['model', 'gmean', 'dataset']
        output_path: Path to save the figure
        title: Plot title
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available. Skipping boxplot generation.")
        return

    if df.empty or 'gmean' not in df.columns:
        logger.warning("No data for boxplot generation.")
        return

    # Filter out models with all zeros (not executed)
    model_means = df.groupby('model')['gmean'].mean()
    valid_models = model_means[model_means > 0].index.tolist()

    if not valid_models:
        logger.warning("No models with valid G-Mean values for boxplot.")
        return

    df_filtered = df[df['model'].isin(valid_models)]

    # Order models by median G-Mean (descending)
    model_order = df_filtered.groupby('model')['gmean'].median().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create boxplot
    sns.boxplot(
        data=df_filtered,
        x='model',
        y='gmean',
        order=model_order,
        palette='Set2',
        ax=ax
    )

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('G-Mean', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"  Saved: {output_path.name}")


def generate_rules_evolution_plot(
    df_rules: pd.DataFrame,
    output_path: Path,
    title: str = "Rules Evolution Over Stream"
) -> None:
    """
    Generate line plot showing rules per chunk evolution.

    Args:
        df_rules: DataFrame with columns ['dataset', 'chunk', 'n_rules', 'avg_conditions_per_rule']
        output_path: Path to save the figure
        title: Plot title
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available. Skipping rules evolution plot.")
        return

    if df_rules.empty or 'n_rules' not in df_rules.columns:
        logger.warning("No rules data for evolution plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Rules per chunk (averaged across datasets)
    avg_by_chunk = df_rules.groupby('chunk').agg({
        'n_rules': ['mean', 'std'],
        'avg_conditions_per_rule': ['mean', 'std']
    }).reset_index()

    ax1 = axes[0]
    chunks = avg_by_chunk['chunk']
    rules_mean = avg_by_chunk[('n_rules', 'mean')]
    rules_std = avg_by_chunk[('n_rules', 'std')]

    ax1.plot(chunks, rules_mean, marker='o', linewidth=2, label='Average Rules')
    ax1.fill_between(chunks, rules_mean - rules_std, rules_mean + rules_std, alpha=0.3)
    ax1.set_xlabel('Chunk', fontsize=12)
    ax1.set_ylabel('Number of Rules', fontsize=12)
    ax1.set_title('Rules per Chunk', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Conditions per rule
    ax2 = axes[1]
    cond_mean = avg_by_chunk[('avg_conditions_per_rule', 'mean')]
    cond_std = avg_by_chunk[('avg_conditions_per_rule', 'std')]

    ax2.plot(chunks, cond_mean, marker='s', linewidth=2, color='green', label='Avg Conditions/Rule')
    ax2.fill_between(chunks, cond_mean - cond_std, cond_mean + cond_std, alpha=0.3, color='green')
    ax2.set_xlabel('Chunk', fontsize=12)
    ax2.set_ylabel('Conditions per Rule', fontsize=12)
    ax2.set_title('Rule Complexity per Chunk', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"  Saved: {output_path.name}")


def generate_transition_metrics_plot(
    df_transitions: pd.DataFrame,
    output_path: Path,
    title: str = "Transition Metrics Over Stream"
) -> None:
    """
    Generate multi-line plot showing TCS, RIR, AMS evolution.

    Args:
        df_transitions: DataFrame with columns ['dataset', 'transition', 'TCS', 'RIR', 'AMS']
        output_path: Path to save the figure
        title: Plot title
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available. Skipping transition metrics plot.")
        return

    if df_transitions.empty:
        logger.warning("No transition data for metrics plot.")
        return

    required_cols = ['TCS', 'RIR', 'AMS']
    if not all(col in df_transitions.columns for col in required_cols):
        logger.warning(f"Missing columns for transition plot. Required: {required_cols}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Average by transition index
    if 'chunk_from' in df_transitions.columns:
        avg_by_trans = df_transitions.groupby('chunk_from')[required_cols].mean().reset_index()
        x_col = 'chunk_from'
        x_label = 'Transition (from chunk)'
    else:
        avg_by_trans = df_transitions.groupby('transition')[required_cols].mean().reset_index()
        x_col = 'transition'
        x_label = 'Transition'

    colors = {'TCS': 'blue', 'RIR': 'red', 'AMS': 'green'}
    markers = {'TCS': 'o', 'RIR': 's', 'AMS': '^'}

    for metric in required_cols:
        ax.plot(
            avg_by_trans[x_col],
            avg_by_trans[metric],
            marker=markers[metric],
            linewidth=2,
            label=metric,
            color=colors[metric]
        )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"  Saved: {output_path.name}")


def generate_performance_vs_complexity_scatter(
    df_results: pd.DataFrame,
    df_rules: pd.DataFrame,
    output_path: Path,
    title: str = "Performance vs Complexity Trade-off"
) -> None:
    """
    Generate scatter plot of G-Mean vs number of rules.

    Args:
        df_results: DataFrame with performance metrics
        df_rules: DataFrame with rules per chunk
        output_path: Path to save the figure
        title: Plot title
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available. Skipping scatter plot.")
        return

    if df_results.empty or df_rules.empty:
        logger.warning("No data for performance vs complexity scatter.")
        return

    # Get GBML results
    gbml_results = df_results[df_results['model'] == 'GBML'].copy()
    if gbml_results.empty:
        logger.warning("No GBML results for scatter plot.")
        return

    # Average rules per dataset
    avg_rules = df_rules.groupby('dataset')['n_rules'].mean().reset_index()
    avg_rules.columns = ['dataset', 'avg_rules']

    # Merge with performance
    merged = gbml_results.merge(avg_rules, on='dataset', how='inner')

    if merged.empty:
        logger.warning("No merged data for scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        merged['avg_rules'],
        merged['gmean'],
        c=merged['gmean'],
        cmap='viridis',
        s=80,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('G-Mean', fontsize=10)

    ax.set_xlabel('Average Number of Rules', fontsize=12)
    ax.set_ylabel('G-Mean', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add trend line
    if len(merged) > 2:
        z = np.polyfit(merged['avg_rules'], merged['gmean'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged['avg_rules'].min(), merged['avg_rules'].max(), 100)
        ax.plot(x_line, p(x_line), '--', color='red', alpha=0.8, label='Trend')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"  Saved: {output_path.name}")


def generate_critical_difference_diagram(
    ranks: Dict[str, float],
    cd_value: float,
    output_path: Path,
    title: str = "Critical Difference Diagram"
) -> None:
    """
    Generate a simple critical difference diagram.

    Args:
        ranks: Dictionary of model names to average ranks
        cd_value: Critical difference value
        output_path: Path to save the figure
        title: Plot title
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available. Skipping CD diagram.")
        return

    if not ranks:
        logger.warning("No ranks data for CD diagram.")
        return

    # Sort models by rank
    sorted_models = sorted(ranks.items(), key=lambda x: x[1])
    n_models = len(sorted_models)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Draw axis
    min_rank = 1
    max_rank = n_models
    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    ax.set_ylim(0, 1)

    # Draw horizontal line
    ax.axhline(y=0.5, color='black', linewidth=2)

    # Draw tick marks and labels
    for i, (model, rank) in enumerate(sorted_models):
        ax.plot([rank, rank], [0.45, 0.55], 'k-', linewidth=2)

        # Alternate label positions
        if i % 2 == 0:
            ax.text(rank, 0.65, model, ha='center', va='bottom', fontsize=10, rotation=45)
        else:
            ax.text(rank, 0.35, model, ha='center', va='top', fontsize=10, rotation=45)

    # Draw CD bar
    if cd_value > 0:
        cd_y = 0.85
        ax.plot([1, 1 + cd_value], [cd_y, cd_y], 'r-', linewidth=3)
        ax.text(1 + cd_value / 2, cd_y + 0.05, f'CD = {cd_value:.3f}', ha='center', fontsize=10, color='red')

    ax.set_xlabel('Average Rank', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"  Saved: {output_path.name}")


def generate_chunk_size_impact_plot(
    all_experiments_df: pd.DataFrame,
    output_path: Path,
    title: str = "Impact of Chunk Size on Performance"
) -> None:
    """
    Generate bar chart comparing experiments with different chunk sizes.

    Args:
        all_experiments_df: Combined DataFrame from all experiments
        output_path: Path to save the figure
        title: Plot title
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available. Skipping chunk size impact plot.")
        return

    if all_experiments_df.empty:
        logger.warning("No data for chunk size impact plot.")
        return

    # Get GBML results only for fair comparison
    gbml_df = all_experiments_df[all_experiments_df['model'] == 'GBML'].copy()

    if gbml_df.empty:
        logger.warning("No GBML data for chunk size impact plot.")
        return

    # Calculate mean and std per experiment
    summary = gbml_df.groupby('experiment')['gmean'].agg(['mean', 'std']).reset_index()
    summary.columns = ['experiment', 'mean', 'std']

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create bar chart
    x = range(len(summary))
    bars = ax.bar(x, summary['mean'], yerr=summary['std'], capsize=5,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(summary)],
                   edgecolor='black', linewidth=1)

    ax.set_xlabel('Experiment Configuration', fontsize=12)
    ax.set_ylabel('Average G-Mean', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(summary['experiment'], rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add value labels on bars
    for bar, mean, std in zip(bars, summary['mean'], summary['std']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"  Saved: {output_path.name}")


def generate_heatmap_by_drift_type(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Performance by Drift Type and Model"
) -> None:
    """
    Generate heatmap showing G-Mean by drift type and model.

    Args:
        df: DataFrame with columns ['model', 'batch', 'gmean']
        output_path: Path to save the figure
        title: Plot title
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available. Skipping heatmap.")
        return

    if df.empty:
        logger.warning("No data for heatmap generation.")
        return

    # Filter out models with all zeros
    model_means = df.groupby('model')['gmean'].mean()
    valid_models = model_means[model_means > 0].index.tolist()

    if not valid_models:
        logger.warning("No models with valid G-Mean for heatmap.")
        return

    df_filtered = df[df['model'].isin(valid_models)]

    # Pivot table
    pivot = df_filtered.pivot_table(
        values='gmean',
        index='batch',
        columns='model',
        aggfunc='mean'
    )

    if pivot.empty:
        logger.warning("Pivot table is empty for heatmap.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5
    )

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Batch (Drift Type)', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"  Saved: {output_path.name}")


def generate_all_figures(exp_name: str, data: Dict, output_dir: Path) -> None:
    """
    Generate all figures for an experiment.

    Args:
        exp_name: Experiment identifier
        data: Dictionary with DataFrames
        output_dir: Output directory
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available. Skipping all figure generation.")
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_results = data.get('results', pd.DataFrame())
    df_rules = data.get('rules', pd.DataFrame())
    df_transitions = data.get('transitions', pd.DataFrame())

    # 1. Performance Boxplot
    generate_performance_boxplot(
        df_results,
        figures_dir / "performance_boxplot.pdf",
        f"Performance Comparison - {exp_name}"
    )

    # 2. Rules Evolution Plot
    if not df_rules.empty:
        generate_rules_evolution_plot(
            df_rules,
            figures_dir / "rules_evolution.pdf",
            f"Rules Evolution - {exp_name}"
        )

    # 3. Transition Metrics Plot
    if not df_transitions.empty:
        generate_transition_metrics_plot(
            df_transitions,
            figures_dir / "transition_metrics.pdf",
            f"Transition Metrics - {exp_name}"
        )

    # 4. Performance vs Complexity Scatter
    if not df_rules.empty:
        generate_performance_vs_complexity_scatter(
            df_results,
            df_rules,
            figures_dir / "performance_vs_complexity.pdf",
            f"Performance vs Complexity - {exp_name}"
        )

    # 5. Heatmap by Drift Type
    generate_heatmap_by_drift_type(
        df_results,
        figures_dir / "heatmap_drift_type.pdf",
        f"Performance by Drift Type - {exp_name}"
    )

    logger.info(f"  Generated figures in {figures_dir}")


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_experiment(exp_name: str, config: Dict, base_dir: Path) -> Dict:
    """
    Process a single experiment configuration.

    Args:
        exp_name: Experiment identifier
        config: Experiment configuration
        base_dir: Base directory for experiments

    Returns:
        Dictionary with all extracted data
    """
    logger.info(f"Processing experiment: {exp_name}")
    logger.info(f"Description: {config['description']}")

    all_results = []
    all_rules = []
    all_transitions = []
    all_erulesd2s_complexity = []

    for batch_name, batch_dir in config['dirs'].items():
        batch_path = base_dir / batch_dir

        if not batch_path.exists():
            logger.warning(f"Batch directory not found: {batch_path}")
            continue

        # Get all dataset directories
        datasets = [d for d in batch_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

        logger.info(f"  {batch_name}: {len(datasets)} datasets")

        for dataset_dir in datasets:
            dataset_name = dataset_dir.name

            # Extract GBML performance
            gbml_perf = extract_gbml_performance(dataset_dir)

            if gbml_perf:
                result = {
                    'experiment': exp_name,
                    'batch': batch_name,
                    'dataset': dataset_name,
                    'model': 'GBML',
                    'gmean': gbml_perf['gmean'],
                    'accuracy': gbml_perf.get('accuracy', 0),
                    'f1': gbml_perf.get('f1', 0),
                    'status': 'OK'
                }
                all_results.append(result)

            # Extract GBML rules
            rules_data = parse_rules_history(dataset_dir)

            if rules_data:
                for chunk_data in rules_data:
                    chunk_data['experiment'] = exp_name
                    chunk_data['batch'] = batch_name
                    chunk_data['dataset'] = dataset_name
                    all_rules.append(chunk_data)

                # Calculate transitions
                transitions = calculate_transition_metrics(rules_data)
                for trans in transitions:
                    trans['experiment'] = exp_name
                    trans['batch'] = batch_name
                    trans['dataset'] = dataset_name
                    all_transitions.append(trans)

            # Extract comparative model results
            comparative = extract_comparative_results(dataset_dir)

            for model, metrics in comparative.items():
                result = {
                    'experiment': exp_name,
                    'batch': batch_name,
                    'dataset': dataset_name,
                    'model': model,
                    'gmean': metrics['gmean'],
                    'accuracy': metrics.get('accuracy', 0),
                    'status': metrics.get('status', 'OK')
                }
                all_results.append(result)

            # Extract ERulesD2S complexity
            erulesd2s_comp = extract_erulesd2s_complexity(dataset_dir)
            if erulesd2s_comp:
                erulesd2s_comp['experiment'] = exp_name
                erulesd2s_comp['batch'] = batch_name
                erulesd2s_comp['dataset'] = dataset_name
                all_erulesd2s_complexity.append(erulesd2s_comp)

    return {
        'results': pd.DataFrame(all_results),
        'rules': pd.DataFrame(all_rules) if all_rules else pd.DataFrame(),
        'transitions': pd.DataFrame(all_transitions) if all_transitions else pd.DataFrame(),
        'erulesd2s_complexity': pd.DataFrame(all_erulesd2s_complexity) if all_erulesd2s_complexity else pd.DataFrame()
    }


def save_experiment_outputs(exp_name: str, data: Dict, output_dir: Path):
    """
    Save all outputs for an experiment.

    Args:
        exp_name: Experiment identifier
        data: Dictionary with DataFrames
        output_dir: Output directory
    """
    exp_output = output_dir / exp_name

    # Create subdirectories
    (exp_output / 'data').mkdir(parents=True, exist_ok=True)
    (exp_output / 'tables').mkdir(parents=True, exist_ok=True)
    (exp_output / 'figures').mkdir(parents=True, exist_ok=True)

    # Save CSV files
    if not data['results'].empty:
        data['results'].to_csv(exp_output / 'data' / 'consolidated_results.csv', index=False)
        logger.info(f"  Saved: consolidated_results.csv ({len(data['results'])} records)")

    if not data['rules'].empty:
        # Save summary (without full rules text)
        rules_summary = data['rules'].drop(columns=['rules'], errors='ignore')
        rules_summary.to_csv(exp_output / 'data' / 'gbml_rules_per_chunk.csv', index=False)
        logger.info(f"  Saved: gbml_rules_per_chunk.csv ({len(rules_summary)} records)")

    if not data['transitions'].empty:
        data['transitions'].to_csv(exp_output / 'data' / 'gbml_transition_metrics.csv', index=False)
        logger.info(f"  Saved: gbml_transition_metrics.csv ({len(data['transitions'])} records)")

    if not data['erulesd2s_complexity'].empty:
        data['erulesd2s_complexity'].to_csv(exp_output / 'data' / 'erulesd2s_complexity.csv', index=False)
        logger.info(f"  Saved: erulesd2s_complexity.csv")

    # Generate pivot table for G-Mean
    if not data['results'].empty:
        pivot = data['results'].pivot_table(
            index='dataset',
            columns='model',
            values='gmean',
            aggfunc='mean'
        )
        pivot.to_csv(exp_output / 'data' / 'pivot_gmean_by_model.csv')
        logger.info(f"  Saved: pivot_gmean_by_model.csv")

        # Generate summary LaTeX performance table
        generate_latex_performance_table(
            pivot,
            MODELS,
            exp_name,
            exp_output / 'tables' / 'performance_summary.tex'
        )
        logger.info(f"  Saved: performance_summary.tex")

        # Generate per-dataset performance tables
        generate_per_dataset_performance_table(
            pivot,
            MODELS,
            exp_name,
            exp_output / 'tables' / 'performance_by_dataset.tex'
        )
        logger.info(f"  Saved: performance_by_dataset.tex")

        # Generate performance by drift type table
        generate_performance_by_drift_type_table(
            data['results'],
            MODELS,
            exp_name,
            exp_output / 'tables' / 'performance_by_drift_type.tex'
        )
        logger.info(f"  Saved: performance_by_drift_type.tex")

        # Statistical analysis
        if SCIPY_AVAILABLE and len(pivot.columns) >= 3:
            # Friedman test
            friedman_result = friedman_test(pivot, MODELS)

            # Average ranks
            avg_ranks = calculate_average_ranks(pivot, MODELS)

            # Wilcoxon tests vs GBML
            wilcoxon_results = {}
            if 'GBML' in pivot.columns:
                for model in MODELS:
                    if model != 'GBML' and model in pivot.columns:
                        result = wilcoxon_test(pivot, 'GBML', model)
                        wilcoxon_results[f'GBML vs {model}'] = result

            # Save statistical results
            stats_data = {
                'friedman': friedman_result,
                'average_ranks': avg_ranks,
                'wilcoxon': wilcoxon_results
            }

            with open(exp_output / 'data' / 'statistical_analysis.json', 'w') as f:
                json.dump(stats_data, f, indent=2, default=str)
            logger.info(f"  Saved: statistical_analysis.json")

            # Generate LaTeX table
            generate_latex_statistical_table(
                friedman_result,
                wilcoxon_results,
                avg_ranks,
                exp_output / 'tables' / 'statistical_tests.tex'
            )
            logger.info(f"  Saved: statistical_tests.tex")

    # Generate complexity comparison table
    if not data['rules'].empty:
        generate_latex_complexity_table(
            data['rules'],
            data['erulesd2s_complexity'],
            exp_output / 'tables' / 'complexity_comparison.tex'
        )
        logger.info(f"  Saved: complexity_comparison.tex")

    # Generate transition metrics table
    if not data['transitions'].empty:
        generate_latex_transition_table(
            data['transitions'],
            exp_output / 'tables' / 'transition_metrics.tex'
        )
        logger.info(f"  Saved: transition_metrics.tex")

    # Generate figures
    logger.info(f"Generating figures for {exp_name}...")
    generate_all_figures(exp_name, data, exp_output)

    # Generate Critical Difference Diagram if we have statistical analysis
    if SCIPY_AVAILABLE and PLOTTING_AVAILABLE and not data['results'].empty:
        try:
            pivot = data['results'].pivot_table(
                index='dataset',
                columns='model',
                values='gmean',
                aggfunc='mean'
            )
            # Filter to models with valid data
            valid_models = [m for m in MODELS if m in pivot.columns and pivot[m].mean() > 0]
            if len(valid_models) >= 2:
                avg_ranks = calculate_average_ranks(pivot, valid_models)
                n_datasets = len(pivot)
                cd = nemenyi_critical_difference(len(valid_models), n_datasets)
                generate_critical_difference_diagram(
                    avg_ranks,
                    cd,
                    exp_output / 'figures' / 'critical_difference.pdf',
                    f"Critical Difference Diagram - {exp_name}"
                )
        except Exception as e:
            logger.warning(f"Could not generate CD diagram: {e}")

    # Count unique datasets and models
    n_datasets = data['results']['dataset'].nunique() if not data['results'].empty else 0
    n_models = data['results']['model'].nunique() if not data['results'].empty else 0
    logger.info(f"  Summary: {n_datasets} datasets, {n_models} models")


def generate_cross_experiment_analysis(all_data: Dict, output_dir: Path):
    """
    Generate cross-experiment comparison analysis.

    Args:
        all_data: Dictionary with data from all experiments
        output_dir: Output directory
    """
    cross_output = output_dir / 'cross_experiment'
    cross_output.mkdir(parents=True, exist_ok=True)

    # Combine all results
    all_results = []
    for exp_name, data in all_data.items():
        if not data['results'].empty:
            all_results.append(data['results'])

    if not all_results:
        logger.warning("No data for cross-experiment analysis")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(cross_output / 'all_experiments_combined.csv', index=False)
    logger.info(f"Saved: all_experiments_combined.csv ({len(combined)} records)")

    # Summary by experiment and model
    summary = combined.groupby(['experiment', 'model']).agg({
        'gmean': ['mean', 'std', 'count']
    }).round(4)
    summary.columns = ['gmean_mean', 'gmean_std', 'n_datasets']
    summary.to_csv(cross_output / 'summary_by_experiment_model.csv')
    logger.info(f"Saved: summary_by_experiment_model.csv")

    # Generate cross-experiment figures
    if PLOTTING_AVAILABLE:
        figures_dir = cross_output / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Chunk size impact plot
        generate_chunk_size_impact_plot(
            combined,
            figures_dir / 'chunk_size_impact.pdf',
            "Impact of Experimental Configuration on Performance"
        )

        # Cross-experiment heatmap
        generate_heatmap_by_drift_type(
            combined[combined['model'] == 'GBML'],
            figures_dir / 'gbml_heatmap_all_experiments.pdf',
            "GBML Performance by Batch (All Experiments)"
        )

        logger.info(f"Saved cross-experiment figures to {figures_dir}")

    # Generate comparison summary
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    for exp_name in EXPERIMENT_CONFIGS.keys():
        exp_data = combined[combined['experiment'] == exp_name]
        if not exp_data.empty:
            logger.info(f"\n{exp_name}:")
            exp_summary = exp_data.groupby('model')['gmean'].mean().sort_values(ascending=False)
            for model, gmean in exp_summary.items():
                logger.info(f"  {model:20s}: {gmean:.4f}")


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("UNIFIED ANALYSIS SCRIPT FOR IEEE PAPER")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Determine base directory
    base_dir = Path(".")

    # Create output directory
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_BASE_DIR.absolute()}")
    logger.info("")

    # Generate dataset characteristics table (shared across experiments)
    common_tables = OUTPUT_BASE_DIR / 'common_tables'
    common_tables.mkdir(parents=True, exist_ok=True)

    generate_dataset_characteristics_table(common_tables / 'dataset_characteristics.tex')
    logger.info("Generated: dataset_characteristics.tex (Table 1 for paper)")
    logger.info("")

    # Process each experiment
    all_data = {}

    for exp_name, config in EXPERIMENT_CONFIGS.items():
        logger.info("-" * 80)
        data = process_experiment(exp_name, config, base_dir)
        all_data[exp_name] = data

        # Save outputs
        logger.info(f"\nSaving outputs for {exp_name}...")
        save_experiment_outputs(exp_name, data, OUTPUT_BASE_DIR)

        # Print summary
        if not data['results'].empty:
            n_datasets = data['results']['dataset'].nunique()
            n_models = data['results']['model'].nunique()
            logger.info(f"  Summary: {n_datasets} datasets, {n_models} models")

        logger.info("")

    # Cross-experiment analysis
    logger.info("-" * 80)
    logger.info("Generating cross-experiment analysis...")
    generate_cross_experiment_analysis(all_data, OUTPUT_BASE_DIR)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput saved to: {OUTPUT_BASE_DIR.absolute()}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # List generated files
    logger.info("\nGenerated files:")
    for exp_dir in OUTPUT_BASE_DIR.iterdir():
        if exp_dir.is_dir():
            logger.info(f"\n  {exp_dir.name}/")
            for subdir in exp_dir.iterdir():
                if subdir.is_dir():
                    files = list(subdir.glob("*"))
                    logger.info(f"    {subdir.name}/ ({len(files)} files)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
