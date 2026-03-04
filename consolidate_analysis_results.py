# consolidate_analysis_results.py
# Consolidates all analysis results from rule_diff_analyzer into a single CSV
# and calculates transition metrics (TCS, RIR, AMS) for the IEEE TKDE paper.

import os
import json
import pandas as pd
import numpy as np
import glob
import logging
import argparse
from typing import Dict, List, Tuple, Optional

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ConsolidateResults")


def parse_matrix_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Parse a matrix CSV file from rule_diff_analyzer.

    Expected format:
    - Index column: transition labels (e.g., "0->1", "1->2")
    - Columns: Unchanged, Modified, New, Deleted, Remain Rules (Chunk i+1)
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)
        # Transpose so rows are transitions and columns are categories
        df = df.T
        return df
    except Exception as e:
        logger.error(f"Failed to parse {csv_path}: {e}")
        return None


def calculate_transition_metrics(unchanged: int, modified: int, new_rules: int,
                                  deleted: int, total_prev: int = None,
                                  total_next: int = None) -> Dict[str, float]:
    """
    Calculate TCS, RIR, and AMS transition metrics.

    TCS (Total Change Score): Proportion of rules that changed
    RIR (Rule Instability Rate): Rate of rule turnover
    AMS (Average Modification Score): Average structural change per rule

    Args:
        unchanged: Number of unchanged rules
        modified: Number of modified rules
        new_rules: Number of new rules
        deleted: Number of deleted rules
        total_prev: Total rules in previous chunk (calculated if None)
        total_next: Total rules in next chunk (calculated if None)

    Returns:
        Dictionary with TCS, RIR, AMS metrics
    """
    # Calculate totals if not provided
    if total_prev is None:
        total_prev = unchanged + modified + deleted
    if total_next is None:
        total_next = unchanged + modified + new_rules

    # Total unique rule events (ensures metrics are in [0, 1] range)
    # Each rule can be: unchanged, modified, new, or deleted
    # Total denominator = unchanged + modified + new + deleted (union of all rule events)
    total_rule_events = unchanged + modified + new_rules + deleted
    total_changes = modified + new_rules + deleted

    # TCS: Total Change Score (proportion of ruleset that changed)
    # Range: 0 to 1, where 0 = no change, 1 = complete change
    tcs = total_changes / total_rule_events if total_rule_events > 0 else 0.0

    # RIR: Rule Instability Rate (proportion of rules added or removed)
    # Range: 0 to 1
    rir = (new_rules + deleted) / total_rule_events if total_rule_events > 0 else 0.0

    # AMS: Average Modification Score (proportion of rules that were modified)
    # Range: 0 to 1
    ams = modified / total_rule_events if total_rule_events > 0 else 0.0

    return {
        'tcs': round(tcs, 4),
        'rir': round(rir, 4),
        'ams': round(ams, 4),
        'total_prev': total_prev,
        'total_next': total_next,
        'total_changes': total_changes,
        'total_rule_events': total_rule_events
    }


def get_drift_type(dataset_name: str) -> str:
    """
    Determine drift type from dataset name.
    """
    name_lower = dataset_name.lower()

    if "_stationary" in name_lower or name_lower.endswith("_stationary"):
        return "stationary"
    elif "_abrupt_" in name_lower:
        return "abrupt"
    elif "_gradual_" in name_lower:
        return "gradual"
    elif "_noise" in name_lower:
        return "noisy"
    else:
        # Real-world datasets
        real_world = ["assetnegotiation", "covtype", "electricity",
                      "intellabsensors", "pokerhand", "shuttle"]
        for rw in real_world:
            if rw in name_lower:
                return "real"
        return "unknown"


def process_matrix_file(csv_path: str) -> List[Dict]:
    """
    Process a single matrix CSV file and extract transition data.

    Returns list of dictionaries, one per transition.
    """
    df = parse_matrix_csv(csv_path)
    if df is None:
        return []

    transitions = []

    # Extract file metadata from path
    # Expected path: .../rule_diff_reports/chunk_500/batch_1/SEA_Abrupt_Simple_run_1_matrix.csv
    path_parts = csv_path.replace("\\", "/").split("/")

    # Find config and batch from path
    config = None
    batch = None
    for i, part in enumerate(path_parts):
        if part.startswith("chunk_"):
            config = part
        elif part.startswith("batch_"):
            batch = part

    # Extract dataset and run from filename
    filename = os.path.basename(csv_path)
    # Remove _matrix.csv suffix
    base_name = filename.replace("_matrix.csv", "")

    # Split on _run_ to get dataset and run number
    if "_run_" in base_name:
        parts = base_name.rsplit("_run_", 1)
        dataset = parts[0]
        run = f"run_{parts[1]}"
    else:
        dataset = base_name
        run = "run_1"

    # Get drift type
    drift_type = get_drift_type(dataset)

    # Process each transition
    for transition in df.index:
        try:
            # Parse transition (e.g., "0->1")
            if "->" not in str(transition):
                continue

            chunk_from, chunk_to = str(transition).split("->")
            chunk_from = int(chunk_from)
            chunk_to = int(chunk_to)

            row = df.loc[transition]

            unchanged = int(row.get('Unchanged', 0))
            modified = int(row.get('Modified', 0))
            new_rules = int(row.get('New', 0))
            deleted = int(row.get('Deleted', 0))
            remain = int(row.get('Remain Rules (Chunk i+1)', unchanged + modified + new_rules))

            # Calculate metrics
            metrics = calculate_transition_metrics(unchanged, modified, new_rules, deleted)

            transitions.append({
                'dataset': dataset,
                'config': config,
                'batch': batch,
                'run': run,
                'drift_type': drift_type,
                'chunk_from': chunk_from,
                'chunk_to': chunk_to,
                'transition': transition,
                'unchanged': unchanged,
                'modified': modified,
                'new': new_rules,
                'deleted': deleted,
                'remain': remain,
                **metrics
            })

        except Exception as e:
            logger.warning(f"Error processing transition {transition} in {csv_path}: {e}")

    return transitions


def aggregate_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transition metrics by dataset (averaging across all transitions).
    """
    agg_df = df.groupby(['dataset', 'config', 'drift_type']).agg({
        'unchanged': 'mean',
        'modified': 'mean',
        'new': 'mean',
        'deleted': 'mean',
        'tcs': 'mean',
        'rir': 'mean',
        'ams': 'mean',
        'chunk_from': 'count'  # Count transitions
    }).rename(columns={'chunk_from': 'n_transitions'}).reset_index()

    # Calculate standard deviations
    std_df = df.groupby(['dataset', 'config', 'drift_type']).agg({
        'tcs': 'std',
        'rir': 'std',
        'ams': 'std'
    }).rename(columns={
        'tcs': 'tcs_std',
        'rir': 'rir_std',
        'ams': 'ams_std'
    }).reset_index()

    # Merge
    agg_df = agg_df.merge(std_df, on=['dataset', 'config', 'drift_type'])

    return agg_df


def aggregate_by_drift_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transition metrics by drift type (for Table XIV in paper).
    """
    agg_df = df.groupby(['drift_type', 'config']).agg({
        'tcs': ['mean', 'std'],
        'rir': ['mean', 'std'],
        'ams': ['mean', 'std'],
        'dataset': 'nunique'
    }).reset_index()

    # Flatten column names
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]

    return agg_df


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate rule_diff_analyzer results into CSV for paper."
    )
    parser.add_argument(
        "--input_dir",
        default="analysis_batch_output/rule_diff_reports",
        help="Directory containing rule_diff_analyzer outputs"
    )
    parser.add_argument(
        "--output_dir",
        default="analysis_batch_output/consolidated",
        help="Output directory for consolidated results"
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, args.input_dir) if not os.path.isabs(args.input_dir) else args.input_dir
    output_dir = os.path.join(script_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Find all matrix CSV files
    csv_files = glob.glob(os.path.join(input_dir, "**", "*_matrix.csv"), recursive=True)
    logger.info(f"Found {len(csv_files)} matrix CSV files")

    if not csv_files:
        logger.warning("No matrix CSV files found. Run run_all_analysis_scripts.py first.")
        return

    # Process all files
    all_transitions = []
    for csv_path in csv_files:
        transitions = process_matrix_file(csv_path)
        all_transitions.extend(transitions)
        if transitions:
            logger.debug(f"Processed {csv_path}: {len(transitions)} transitions")

    logger.info(f"Total transitions extracted: {len(all_transitions)}")

    if not all_transitions:
        logger.warning("No transitions extracted. Check input data.")
        return

    # Create main DataFrame
    df = pd.DataFrame(all_transitions)

    # Save full transition data
    full_csv_path = os.path.join(output_dir, "transition_metrics_all.csv")
    df.to_csv(full_csv_path, index=False)
    logger.info(f"Full transition data saved to: {full_csv_path}")

    # Aggregate by dataset
    df_by_dataset = aggregate_by_dataset(df)
    dataset_csv_path = os.path.join(output_dir, "transition_metrics_by_dataset.csv")
    df_by_dataset.to_csv(dataset_csv_path, index=False)
    logger.info(f"Dataset aggregates saved to: {dataset_csv_path}")

    # Aggregate by drift type (for Table XIV)
    df_by_drift = aggregate_by_drift_type(df)
    drift_csv_path = os.path.join(output_dir, "transition_metrics_by_drift_type.csv")
    df_by_drift.to_csv(drift_csv_path, index=False)
    logger.info(f"Drift type aggregates saved to: {drift_csv_path}")

    # Generate summary statistics
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS (for Table XIV)")
    logger.info("="*60)

    # Group by drift type and config, show mean +/- std
    for config in df['config'].unique():
        logger.info(f"\nConfiguration: {config}")
        logger.info("-" * 40)

        config_df = df[df['config'] == config]

        for drift_type in sorted(config_df['drift_type'].unique()):
            type_df = config_df[config_df['drift_type'] == drift_type]

            tcs_mean = type_df['tcs'].mean()
            tcs_std = type_df['tcs'].std()
            rir_mean = type_df['rir'].mean()
            rir_std = type_df['rir'].std()
            ams_mean = type_df['ams'].mean()
            ams_std = type_df['ams'].std()

            n_datasets = type_df['dataset'].nunique()
            n_transitions = len(type_df)

            logger.info(f"  {drift_type:15s} | TCS: {tcs_mean:.3f}+/-{tcs_std:.3f} | "
                       f"RIR: {rir_mean:.3f}+/-{rir_std:.3f} | "
                       f"AMS: {ams_mean:.3f}+/-{ams_std:.3f} | "
                       f"({n_datasets} datasets, {n_transitions} transitions)")

    # Save summary for paper
    summary = {
        'total_transitions': len(all_transitions),
        'total_datasets': df['dataset'].nunique(),
        'configs': df['config'].unique().tolist(),
        'drift_types': df['drift_type'].unique().tolist(),
        'by_drift_type': df_by_drift.to_dict(orient='records')
    }

    summary_path = os.path.join(output_dir, "consolidation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
