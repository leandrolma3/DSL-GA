#!/usr/bin/env python3
"""
calculate_transition_metrics_report.py

Standalone script that calculates transition metrics (RIR, AMS, TCS) from
RulesHistory files across all unified experiment directories and generates
a comprehensive Markdown report.

Uses the same formulas and parsing logic as collect_results_for_paper.py.
"""

import re
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

# ============================================================================
# Constants
# ============================================================================

BASE_DIR = Path(__file__).parent
EXPERIMENT_DIRS = [
    "experiments_unified/chunk_500",
    "experiments_unified/chunk_500_penalty",
    "experiments_unified/chunk_1000",
    "experiments_unified/chunk_1000_penalty",
    "experiments_unified/chunk_2000",
    "experiments_unified/chunk_2000_penalty",
]

EXCLUDED_DATASETS = {"CovType", "IntelLabSensors", "PokerHand", "Shuttle"}

W_INSTABILITY = 0.6
W_MODIFICATION_IMPACT = 0.4
MAX_RULES = 50
SIMILARITY_THRESHOLD = 0.5
SEVERITY_THRESHOLD = 0.8

DATASET_METADATA = {
    'SEA_Abrupt_Simple': 'abrupt', 'SEA_Abrupt_Chain': 'abrupt', 'SEA_Abrupt_Recurring': 'abrupt',
    'AGRAWAL_Abrupt_Simple_Mild': 'abrupt', 'AGRAWAL_Abrupt_Simple_Severe': 'abrupt',
    'AGRAWAL_Abrupt_Chain_Long': 'abrupt', 'RBF_Abrupt_Severe': 'abrupt', 'RBF_Abrupt_Blip': 'abrupt',
    'STAGGER_Abrupt_Chain': 'abrupt', 'STAGGER_Abrupt_Recurring': 'abrupt',
    'HYPERPLANE_Abrupt_Simple': 'abrupt', 'RANDOMTREE_Abrupt_Simple': 'abrupt',
    'SINE_Abrupt_Simple': 'abrupt', 'LED_Abrupt_Simple': 'abrupt',
    'WAVEFORM_Abrupt_Simple': 'abrupt', 'RANDOMTREE_Abrupt_Recurring': 'abrupt',
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
    'Electricity': 'real', 'AssetNegotiation_F2': 'real', 'AssetNegotiation_F3': 'real',
    'AssetNegotiation_F4': 'real',
    'SEA_Stationary': 'stationary', 'AGRAWAL_Stationary': 'stationary', 'RBF_Stationary': 'stationary',
    'LED_Stationary': 'stationary', 'HYPERPLANE_Stationary': 'stationary',
    'RANDOMTREE_Stationary': 'stationary', 'STAGGER_Stationary': 'stationary',
    'WAVEFORM_Stationary': 'stationary', 'SINE_Stationary': 'stationary',
}

# Map directory suffix to experiment label
EXPERIMENT_LABELS = {
    'chunk_500': 'EXP-500',
    'chunk_500_penalty': 'EXP-500-P',
    'chunk_1000': 'EXP-1000',
    'chunk_1000_penalty': 'EXP-1000-P',
    'chunk_2000': 'EXP-2000',
    'chunk_2000_penalty': 'EXP-2000-P',
}


# ============================================================================
# Similarity / Levenshtein
# ============================================================================

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
            cost = 0 if rule1[i - 1] == rule2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )
    distance = matrix[len1][len2]
    return 1.0 - (distance / max(len1, len2))


# ============================================================================
# Parsing
# ============================================================================

def parse_rules_history(run_dir: Path) -> List[Dict]:
    """Parse RulesHistory file to extract rules per chunk."""
    rules_files = list(run_dir.glob("RulesHistory_*.txt"))
    if not rules_files:
        return []
    rules_file = rules_files[0]
    try:
        with open(rules_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return []

    chunks_data = []
    chunk_sections = re.split(r'--- Chunk \d+ \(Trained\) ---', content)[1:]
    for idx, section in enumerate(chunk_sections):
        rules = re.findall(r'IF (.+?) THEN Class (\d+)', section)
        chunk_data = {
            'chunk': idx,
            'rules': [{'condition': r[0], 'class': r[1]} for r in rules],
            'n_rules': len(rules),
        }
        chunks_data.append(chunk_data)
    return chunks_data


# ============================================================================
# Transition metric calculation (same as collect_results_for_paper.py)
# ============================================================================

def calculate_transition_metrics(chunks_data: List[Dict]) -> List[Dict]:
    """Calculate transition metrics (RIR, AMS, TCS) between consecutive chunks."""
    transitions = []

    for i in range(len(chunks_data) - 1):
        rules_i = [r['condition'] for r in chunks_data[i].get('rules', [])]
        rules_j = [r['condition'] for r in chunks_data[i + 1].get('rules', [])]

        if not rules_i or not rules_j:
            transitions.append({
                'RIR': 1.0, 'AMS': 1.0, 'TCS': 1.0,
                'unchanged': 0, 'modified': 0,
                'new': len(rules_j), 'deleted': len(rules_i),
            })
            continue

        # Exact matches (unchanged)
        rules_i_matched = set()
        rules_j_matched = set()
        for idx_i, ri in enumerate(rules_i):
            for idx_j, rj in enumerate(rules_j):
                if idx_j in rules_j_matched:
                    continue
                if ri == rj:
                    rules_i_matched.add(idx_i)
                    rules_j_matched.add(idx_j)
                    break
        unchanged_count = len(rules_i_matched)

        # Similarity-based matching for modified
        severities = []
        n_ui = len(rules_i) - len(rules_i_matched)
        n_uj = len(rules_j) - len(rules_j_matched)
        skip_similarity = n_ui * n_uj > MAX_RULES ** 2

        if not skip_similarity:
            candidates = []
            for idx_i, ri in enumerate(rules_i):
                if idx_i in rules_i_matched:
                    continue
                for idx_j, rj in enumerate(rules_j):
                    if idx_j in rules_j_matched:
                        continue
                    sim = calculate_rule_similarity(ri, rj)
                    if sim >= SIMILARITY_THRESHOLD:
                        sev = 1.0 - sim
                        if sev < SEVERITY_THRESHOLD:
                            candidates.append((idx_i, idx_j, sim, sev))
            candidates.sort(key=lambda x: x[3])
            for idx_i, idx_j, sim, sev in candidates:
                if idx_i not in rules_i_matched and idx_j not in rules_j_matched:
                    severities.append(sev)
                    rules_i_matched.add(idx_i)
                    rules_j_matched.add(idx_j)

        modified_count = len(severities)
        new_count = len(rules_j) - len(rules_j_matched)
        deleted_count = len(rules_i) - len(rules_i_matched)

        total_rules = len(rules_i) + len(rules_j)
        RIR = (new_count + deleted_count) / total_rules if total_rules > 0 else 0.0
        AMS = float(np.mean(severities)) if severities else 0.0
        prop_modified = modified_count / len(rules_j) if len(rules_j) > 0 else 0.0
        TCS = W_INSTABILITY * RIR + W_MODIFICATION_IMPACT * prop_modified * AMS
        TCS = min(max(TCS, 0.0), 1.0)

        transitions.append({
            'RIR': round(RIR, 4), 'AMS': round(AMS, 4), 'TCS': round(TCS, 4),
            'unchanged': unchanged_count, 'modified': modified_count,
            'new': new_count, 'deleted': deleted_count,
        })

    return transitions


# ============================================================================
# Data collection
# ============================================================================

def collect_all_results() -> List[Dict]:
    """Iterate over all experiment directories and collect per-dataset mean metrics."""
    all_results = []

    for exp_rel in EXPERIMENT_DIRS:
        exp_dir = BASE_DIR / exp_rel
        if not exp_dir.is_dir():
            print(f"[SKIP] Directory not found: {exp_dir}")
            continue

        exp_key = exp_rel.split("/")[-1]  # e.g. chunk_500
        exp_label = EXPERIMENT_LABELS.get(exp_key, exp_key)

        # Each subdirectory is a dataset
        for dataset_dir in sorted(exp_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name
            if dataset_name in EXCLUDED_DATASETS:
                continue

            drift_type = DATASET_METADATA.get(dataset_name, 'unknown')

            # Collect across batches and runs
            dataset_rir = []
            dataset_ams = []
            dataset_tcs = []

            for batch_dir in sorted(dataset_dir.iterdir()):
                if not batch_dir.is_dir():
                    continue
                for run_dir in sorted(batch_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    chunks = parse_rules_history(run_dir)
                    if len(chunks) < 2:
                        continue
                    transitions = calculate_transition_metrics(chunks)
                    if not transitions:
                        continue
                    # Mean across transitions for this run
                    dataset_rir.append(np.mean([t['RIR'] for t in transitions]))
                    dataset_ams.append(np.mean([t['AMS'] for t in transitions]))
                    dataset_tcs.append(np.mean([t['TCS'] for t in transitions]))

            if not dataset_rir:
                continue

            all_results.append({
                'experiment': exp_label,
                'exp_key': exp_key,
                'dataset': dataset_name,
                'drift_type': drift_type,
                'n_runs': len(dataset_rir),
                'RIR_mean': np.mean(dataset_rir),
                'RIR_std': np.std(dataset_rir),
                'AMS_mean': np.mean(dataset_ams),
                'AMS_std': np.std(dataset_ams),
                'TCS_mean': np.mean(dataset_tcs),
                'TCS_std': np.std(dataset_tcs),
            })

    return all_results


# ============================================================================
# Report generation
# ============================================================================

def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def generate_report(results: List[Dict]) -> str:
    lines = []
    lines.append("# Transition Metrics Report (RIR, AMS, TCS)")
    lines.append("")
    lines.append("Generated automatically from `experiments_unified/` directories.")
    lines.append("")
    lines.append("**Formulas:**")
    lines.append("- **RIR** (Rule Instability Rate) = (new + deleted) / (rules_from + rules_to)")
    lines.append("- **AMS** (Average Modification Severity) = mean(1 - similarity) for modified rules")
    lines.append("- **TCS** (Transition Change Score) = 0.6 * RIR + 0.4 * prop_modified * AMS")
    lines.append("")
    lines.append(f"**Excluded datasets:** {', '.join(sorted(EXCLUDED_DATASETS))}")
    lines.append("")

    if not results:
        lines.append("**No results found.**")
        return "\n".join(lines)

    # ---- Section 1: Summary by drift type ----
    lines.append("## 1. Summary by Drift Type")
    lines.append("")

    # Group by (experiment, drift_type)
    by_exp_drift = defaultdict(list)
    for r in results:
        by_exp_drift[(r['experiment'], r['drift_type'])].append(r)

    experiments = sorted(set(r['experiment'] for r in results))
    drift_types = sorted(set(r['drift_type'] for r in results))

    for exp in experiments:
        lines.append(f"### {exp}")
        lines.append("")
        lines.append("| Drift Type | N datasets | RIR | AMS | TCS |")
        lines.append("|---|---|---|---|---|")
        for dt in drift_types:
            items = by_exp_drift.get((exp, dt), [])
            if not items:
                continue
            rir_vals = [i['RIR_mean'] for i in items]
            ams_vals = [i['AMS_mean'] for i in items]
            tcs_vals = [i['TCS_mean'] for i in items]
            lines.append(
                f"| {dt} | {len(items)} "
                f"| {np.mean(rir_vals):.4f} +/- {np.std(rir_vals):.4f} "
                f"| {np.mean(ams_vals):.4f} +/- {np.std(ams_vals):.4f} "
                f"| {np.mean(tcs_vals):.4f} +/- {np.std(tcs_vals):.4f} |"
            )
        lines.append("")

    # ---- Section 2: By individual dataset ----
    lines.append("## 2. Results by Individual Dataset")
    lines.append("")

    for exp in experiments:
        lines.append(f"### {exp}")
        lines.append("")
        lines.append("| Dataset | Drift Type | Runs | RIR | AMS | TCS |")
        lines.append("|---|---|---|---|---|---|")
        exp_results = sorted(
            [r for r in results if r['experiment'] == exp],
            key=lambda r: (r['drift_type'], r['dataset']),
        )
        for r in exp_results:
            lines.append(
                f"| {r['dataset']} | {r['drift_type']} | {r['n_runs']} "
                f"| {fmt(r['RIR_mean'], r['RIR_std'])} "
                f"| {fmt(r['AMS_mean'], r['AMS_std'])} "
                f"| {fmt(r['TCS_mean'], r['TCS_std'])} |"
            )
        lines.append("")

    # ---- Section 3: Comparison across chunk sizes ----
    lines.append("## 3. Comparison EXP-500 vs EXP-1000 vs EXP-2000")
    lines.append("")
    lines.append("Aggregated across all datasets (without penalty variants).")
    lines.append("")

    main_exps = ['EXP-500', 'EXP-1000', 'EXP-2000']
    lines.append("| Experiment | N | RIR | AMS | TCS |")
    lines.append("|---|---|---|---|---|")
    for exp in main_exps:
        items = [r for r in results if r['experiment'] == exp]
        if not items:
            continue
        rir_vals = [i['RIR_mean'] for i in items]
        ams_vals = [i['AMS_mean'] for i in items]
        tcs_vals = [i['TCS_mean'] for i in items]
        lines.append(
            f"| {exp} | {len(items)} "
            f"| {np.mean(rir_vals):.4f} +/- {np.std(rir_vals):.4f} "
            f"| {np.mean(ams_vals):.4f} +/- {np.std(ams_vals):.4f} "
            f"| {np.mean(tcs_vals):.4f} +/- {np.std(tcs_vals):.4f} |"
        )
    lines.append("")

    # Penalty comparison
    lines.append("### With vs Without Penalty")
    lines.append("")
    lines.append("| Experiment | N | RIR | AMS | TCS |")
    lines.append("|---|---|---|---|---|")
    for base in ['500', '1000', '2000']:
        for suffix, label in [('', ''), ('_penalty', ' (penalty)')]:
            exp = EXPERIMENT_LABELS.get(f'chunk_{base}{suffix}', '')
            items = [r for r in results if r['experiment'] == exp]
            if not items:
                continue
            rir_vals = [i['RIR_mean'] for i in items]
            ams_vals = [i['AMS_mean'] for i in items]
            tcs_vals = [i['TCS_mean'] for i in items]
            lines.append(
                f"| {exp} | {len(items)} "
                f"| {np.mean(rir_vals):.4f} +/- {np.std(rir_vals):.4f} "
                f"| {np.mean(ams_vals):.4f} +/- {np.std(ams_vals):.4f} "
                f"| {np.mean(tcs_vals):.4f} +/- {np.std(tcs_vals):.4f} |"
            )
    lines.append("")

    # ---- Section 4: Comparison by drift type across chunk sizes ----
    lines.append("## 4. Drift Type x Chunk Size (main experiments only)")
    lines.append("")
    lines.append("| Drift Type | EXP-500 TCS | EXP-1000 TCS | EXP-2000 TCS |")
    lines.append("|---|---|---|---|")
    for dt in drift_types:
        row = [dt]
        for exp in main_exps:
            items = [r for r in results if r['experiment'] == exp and r['drift_type'] == dt]
            if items:
                tcs_vals = [i['TCS_mean'] for i in items]
                row.append(f"{np.mean(tcs_vals):.4f} +/- {np.std(tcs_vals):.4f}")
            else:
                row.append("-")
        lines.append(f"| {' | '.join(row)} |")
    lines.append("")

    # ---- Section 5: Pattern analysis ----
    lines.append("## 5. Pattern Analysis")
    lines.append("")

    # Most/least stable datasets
    for exp in main_exps:
        items = [r for r in results if r['experiment'] == exp]
        if not items:
            continue
        items_sorted = sorted(items, key=lambda r: r['TCS_mean'])
        lines.append(f"### {exp}")
        lines.append("")
        lines.append("**Top 5 most stable (lowest TCS):**")
        lines.append("")
        for r in items_sorted[:5]:
            lines.append(f"- {r['dataset']} ({r['drift_type']}): TCS={r['TCS_mean']:.4f}")
        lines.append("")
        lines.append("**Top 5 most unstable (highest TCS):**")
        lines.append("")
        for r in items_sorted[-5:]:
            lines.append(f"- {r['dataset']} ({r['drift_type']}): TCS={r['TCS_mean']:.4f}")
        lines.append("")

    # Stationary vs drift comparison
    lines.append("### Stationary vs Drift Datasets")
    lines.append("")
    for exp in main_exps:
        stat = [r['TCS_mean'] for r in results if r['experiment'] == exp and r['drift_type'] == 'stationary']
        drift = [r['TCS_mean'] for r in results if r['experiment'] == exp and r['drift_type'] != 'stationary']
        if stat and drift:
            lines.append(
                f"- **{exp}**: Stationary TCS={np.mean(stat):.4f} vs Drift TCS={np.mean(drift):.4f} "
                f"(delta={np.mean(drift) - np.mean(stat):.4f})"
            )
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    print("Collecting transition metrics from experiments_unified/...")
    results = collect_all_results()
    print(f"Collected {len(results)} dataset-experiment combinations.")

    report = generate_report(results)
    output_path = BASE_DIR / "RELATORIO_METRICAS_TRANSICAO.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
