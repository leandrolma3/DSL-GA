# run_all_analysis_scripts.py
# Batch execution script to run analysis scripts on all EGIS experiments
# for IEEE TKDE paper data generation.

import os
import sys
import subprocess
import logging
import json
import glob
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("BatchAnalysis")

# --- Dataset Classification ---

DRIFT_SIMULATION_PREFIXES = [
    "AGRAWAL", "SEA", "RBF", "HYPERPLANE", "RANDOMTREE",
    "STAGGER", "SINE", "LED", "WAVEFORM"
]

REAL_WORLD_DATASETS = [
    "AssetNegotiation_F2", "AssetNegotiation_F3", "AssetNegotiation_F4",
    "CovType", "Electricity", "IntelLabSensors", "PokerHand", "Shuttle"
]

STATIONARY_SUFFIX = "_Stationary"


def get_dataset_type(dataset_name: str) -> str:
    """
    Classify a dataset as 'stationary', 'real', 'drift_simulation', or 'unknown'.
    """
    # Check for stationary datasets first
    if dataset_name.endswith(STATIONARY_SUFFIX):
        return "stationary"

    # Check for real-world datasets
    for real in REAL_WORLD_DATASETS:
        if dataset_name == real or dataset_name.startswith(real):
            return "real"

    # Check for drift simulation datasets
    for sim_prefix in DRIFT_SIMULATION_PREFIXES:
        if dataset_name.startswith(sim_prefix):
            return "drift_simulation"

    return "unknown"


def get_applicable_scripts(dataset_type: str) -> List[str]:
    """
    Return list of scripts applicable to a dataset type.
    """
    scripts = ["rule_diff_analyzer"]  # Applies to ALL datasets

    if dataset_type == "drift_simulation":
        scripts.extend(["generate_plots", "analyze_concept_difference"])
    elif dataset_type == "real":
        scripts.append("analyze_standard_drift")
    # For stationary datasets, only rule_diff_analyzer applies

    return scripts


def find_all_experiments(base_dir: str, configs: List[str] = None) -> List[Dict]:
    """
    Find all experiment runs in the experiments_unified directory.

    Returns list of dicts with keys:
    - config: e.g., 'chunk_500'
    - batch: e.g., 'batch_1'
    - dataset: e.g., 'SEA_Abrupt_Simple'
    - run_path: full path to run directory
    - rules_history_path: path to RulesHistory file (if exists)
    """
    if configs is None:
        configs = ["chunk_500", "chunk_500_penalty", "chunk_1000", "chunk_1000_penalty"]

    experiments = []

    for config in configs:
        config_path = os.path.join(base_dir, config)
        if not os.path.isdir(config_path):
            logger.warning(f"Config directory not found: {config_path}")
            continue

        # Find all batch directories
        for batch_name in os.listdir(config_path):
            batch_path = os.path.join(config_path, batch_name)
            if not os.path.isdir(batch_path) or not batch_name.startswith("batch_"):
                continue

            # Find all dataset directories
            for dataset_name in os.listdir(batch_path):
                dataset_path = os.path.join(batch_path, dataset_name)
                if not os.path.isdir(dataset_path):
                    continue

                # Skip non-experiment directories
                if dataset_name in ['desktop.ini', '.DS_Store'] or dataset_name.endswith('.csv') or dataset_name.endswith('.png'):
                    continue

                # Find all run directories
                for run_name in os.listdir(dataset_path):
                    run_path = os.path.join(dataset_path, run_name)
                    if not os.path.isdir(run_path) or not run_name.startswith("run_"):
                        continue

                    # Look for RulesHistory file
                    rules_history_files = glob.glob(os.path.join(run_path, "RulesHistory_*.txt"))
                    rules_history_path = rules_history_files[0] if rules_history_files else None

                    experiments.append({
                        'config': config,
                        'batch': batch_name,
                        'dataset': dataset_name,
                        'run': run_name,
                        'run_path': run_path,
                        'rules_history_path': rules_history_path,
                        'dataset_type': get_dataset_type(dataset_name)
                    })

    return experiments


def run_rule_diff_analyzer(exp: Dict, output_base: str, threshold: float = 0.35) -> Dict:
    """
    Run rule_diff_analyzer.py on an experiment.

    Returns dict with status and output paths.
    """
    result = {
        'experiment': f"{exp['config']}/{exp['batch']}/{exp['dataset']}/{exp['run']}",
        'script': 'rule_diff_analyzer',
        'success': False,
        'output_files': []
    }

    if not exp.get('rules_history_path'):
        result['error'] = "RulesHistory file not found"
        return result

    # Create output directory
    output_dir = os.path.join(output_base, "rule_diff_reports", exp['config'], exp['batch'])
    os.makedirs(output_dir, exist_ok=True)

    # Output base path
    output_base_path = os.path.join(output_dir, f"{exp['dataset']}_{exp['run']}")

    # Build command
    cmd = [
        sys.executable,
        "rule_diff_analyzer.py",
        exp['rules_history_path'],
        "-t", str(threshold),
        "-o", output_base_path
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        if proc.returncode == 0:
            result['success'] = True
            result['output_files'] = [
                output_base_path + "_report.txt",
                output_base_path + "_matrix.csv",
                output_base_path + "_matrix.png"
            ]
        else:
            result['error'] = proc.stderr[:500] if proc.stderr else "Unknown error"

    except subprocess.TimeoutExpired:
        result['error'] = "Timeout (300s)"
    except Exception as e:
        result['error'] = str(e)

    return result


def run_generate_plots(exp: Dict, output_base: str, diff_file: str = None) -> Dict:
    """
    Run generate_plots.py on an experiment (for drift simulation datasets).
    """
    result = {
        'experiment': f"{exp['config']}/{exp['batch']}/{exp['dataset']}/{exp['run']}",
        'script': 'generate_plots',
        'success': False,
        'output_files': []
    }

    # Create output directory
    output_dir = os.path.join(output_base, "plots", "drift_simulation", exp['config'], exp['batch'], exp['dataset'])
    os.makedirs(output_dir, exist_ok=True)

    # Build command
    cmd = [
        sys.executable,
        "generate_plots.py",
        exp['run_path'],
        "-o", output_dir
    ]

    if diff_file and os.path.exists(diff_file):
        cmd.extend(["-d", diff_file])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        if proc.returncode == 0:
            result['success'] = True
            # List generated plot files
            plots_dir = os.path.join(output_dir, "plots")
            if os.path.isdir(plots_dir):
                result['output_files'] = glob.glob(os.path.join(plots_dir, "*.png"))
        else:
            result['error'] = proc.stderr[:500] if proc.stderr else "Unknown error"

    except subprocess.TimeoutExpired:
        result['error'] = "Timeout (300s)"
    except Exception as e:
        result['error'] = str(e)

    return result


def run_analyze_standard_drift(exp: Dict, output_base: str, threshold: float = 0.10) -> Dict:
    """
    Run analyze_standard_drift.py on an experiment (for real-world datasets).
    """
    result = {
        'experiment': f"{exp['config']}/{exp['batch']}/{exp['dataset']}/{exp['run']}",
        'script': 'analyze_standard_drift',
        'success': False,
        'output_files': []
    }

    # Create output directory
    output_dir = os.path.join(output_base, "plots", "real_world", exp['config'], exp['batch'], exp['dataset'])
    os.makedirs(output_dir, exist_ok=True)

    # Build command
    cmd = [
        sys.executable,
        "analyze_standard_drift.py",
        exp['run_path'],
        "-t", str(threshold),
        "-o", output_dir
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        if proc.returncode == 0:
            result['success'] = True
            # List generated plot files
            plots_dir = os.path.join(output_dir, "plots")
            if os.path.isdir(plots_dir):
                result['output_files'] = glob.glob(os.path.join(plots_dir, "*.png"))
        else:
            result['error'] = proc.stderr[:500] if proc.stderr else "Unknown error"

    except subprocess.TimeoutExpired:
        result['error'] = "Timeout (300s)"
    except Exception as e:
        result['error'] = str(e)

    return result


def process_experiment(exp: Dict, output_base: str, diff_file: str = None,
                       skip_plots: bool = False) -> List[Dict]:
    """
    Process a single experiment with all applicable scripts.
    """
    results = []

    applicable_scripts = get_applicable_scripts(exp['dataset_type'])

    # Always run rule_diff_analyzer
    if "rule_diff_analyzer" in applicable_scripts:
        results.append(run_rule_diff_analyzer(exp, output_base))

    # Run generate_plots for drift simulation datasets
    if not skip_plots and "generate_plots" in applicable_scripts:
        results.append(run_generate_plots(exp, output_base, diff_file))

    # Run analyze_standard_drift for real-world datasets
    if not skip_plots and "analyze_standard_drift" in applicable_scripts:
        results.append(run_analyze_standard_drift(exp, output_base))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch execution of analysis scripts on all EGIS experiments."
    )
    parser.add_argument(
        "--base_dir",
        default="experiments_unified",
        help="Base directory containing experiment results (default: experiments_unified)"
    )
    parser.add_argument(
        "--output",
        default="analysis_batch_output",
        help="Output directory for analysis results (default: analysis_batch_output)"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["chunk_500", "chunk_500_penalty", "chunk_1000", "chunk_1000_penalty"],
        help="Configurations to process (default: chunk_500, chunk_500_penalty, chunk_1000, chunk_1000_penalty)"
    )
    parser.add_argument(
        "--diff_file",
        default="results/concept_heatmaps/concept_differences.json",
        help="Path to concept differences JSON file"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip plot generation (only run rule_diff_analyzer)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List experiments without executing scripts"
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, args.base_dir) if not os.path.isabs(args.base_dir) else args.base_dir
    output_base = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output
    diff_file = os.path.join(script_dir, args.diff_file) if not os.path.isabs(args.diff_file) else args.diff_file

    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Output directory: {output_base}")
    logger.info(f"Configurations: {args.configs}")

    # Find all experiments
    logger.info("Discovering experiments...")
    experiments = find_all_experiments(base_dir, args.configs)
    logger.info(f"Found {len(experiments)} experiment runs")

    # Count by type
    type_counts = {}
    for exp in experiments:
        dt = exp['dataset_type']
        type_counts[dt] = type_counts.get(dt, 0) + 1

    logger.info(f"Dataset types: {type_counts}")

    # Count by config
    config_counts = {}
    for exp in experiments:
        c = exp['config']
        config_counts[c] = config_counts.get(c, 0) + 1

    logger.info(f"By configuration: {config_counts}")

    if args.dry_run:
        logger.info("Dry run mode - listing experiments:")
        for exp in experiments[:20]:
            logger.info(f"  {exp['config']}/{exp['batch']}/{exp['dataset']}/{exp['run']} [{exp['dataset_type']}]")
        if len(experiments) > 20:
            logger.info(f"  ... and {len(experiments) - 20} more")
        return

    # Create output directory
    os.makedirs(output_base, exist_ok=True)

    # Process experiments
    all_results = []
    success_count = 0
    fail_count = 0

    logger.info(f"Processing {len(experiments)} experiments with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_experiment,
                exp,
                output_base,
                diff_file,
                args.skip_plots
            ): exp
            for exp in experiments
        }

        for i, future in enumerate(as_completed(futures)):
            exp = futures[future]
            try:
                results = future.result()
                all_results.extend(results)

                for r in results:
                    if r['success']:
                        success_count += 1
                    else:
                        fail_count += 1

                # Progress logging
                if (i + 1) % 10 == 0 or (i + 1) == len(experiments):
                    logger.info(f"Progress: {i + 1}/{len(experiments)} experiments processed")

            except Exception as e:
                logger.error(f"Error processing {exp['dataset']}: {e}")
                fail_count += 1

    # Save results summary
    summary_path = os.path.join(output_base, "analysis_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'total_experiments': len(experiments),
            'total_script_runs': len(all_results),
            'success_count': success_count,
            'fail_count': fail_count,
            'results': all_results
        }, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH ANALYSIS COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total experiments: {len(experiments)}")
    logger.info(f"Script executions: {len(all_results)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
