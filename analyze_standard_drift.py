# analyze_standard_drift.py
# Script to analyze results from 'standard' mode runs
# and identify/visualize potential natural drifts based on performance drops.

import json
import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Optional, for style
from typing import Dict, List, Tuple, Any, Union

# Attempt to import plotting helpers, define placeholder if missing
try:
    # Assuming _save_plot might be in plotting.py
    from plotting import _save_plot as save_plot_helper
except ImportError:
    logging.info("plotting.py or _save_plot not found, using fallback for saving plots.")
    # Fallback save function if helper is not available
    def save_plot_helper(fig, save_path):
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, bbox_inches='tight')
                logging.info(f"Plot saved to: {save_path}") # Log success inside try
            except Exception as e:
                logging.error(f"Failed to save plot to {save_path}: {e}")
        # Always close the figure
        plt.close(fig)


# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("analyze_standard_drift")

# --- Helper Functions ---

def load_json_file(file_path: str) -> Union[Dict, List, None]:
    """Loads data from a JSON file."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.debug(f"Data loaded from {file_path}")
        return data
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON in: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading {file_path}: {e}")
        return None

def detect_performance_drifts(chunk_metrics: List[Dict], threshold: float = 0.10) -> List[int]:
    """
    Identifies chunk transitions where test accuracy dropped
    more than a relative threshold.

    Args:
        chunk_metrics (List[Dict]): List of dictionaries loaded from chunk_metrics.json.
                                     Expects keys 'chunk' and 'test_accuracy'.
        threshold (float): Relative drop threshold (e.g., 0.10 for 10%) to
                           consider a drift.

    Returns:
        List[int]: List of chunk indices WHERE THE DRIFT WAS OBSERVED
                   (i.e., the test chunk index i+1 where performance dropped).
                   Returns the index i+1.
    """
    drift_points_chunk_idx = []
    if not chunk_metrics or len(chunk_metrics) < 2:
        logger.warning("Insufficient metrics data to detect performance drift.")
        return drift_points_chunk_idx

    # Sort by chunk index to ensure correct sequence
    try:
        # Ensure 'chunk' key exists and handle potential errors during sorting
        sorted_metrics = sorted(chunk_metrics, key=lambda x: x.get('chunk', -1))
        if any(m.get('chunk', -1) == -1 for m in sorted_metrics):
             logger.warning("Some chunk metrics missing 'chunk' index, order might be incorrect.")
    except Exception as e:
        logger.error(f"Error sorting chunk metrics: {e}. Cannot detect drift.")
        return []


    # Use the test accuracy from the first transition (chunk 0 -> chunk 1) as the initial reference
    previous_accuracy = sorted_metrics[0].get('test_accuracy')
    if previous_accuracy is None and len(sorted_metrics) > 1:
         previous_accuracy = sorted_metrics[1].get('test_accuracy')
         logger.warning("First chunk test accuracy missing, using second as baseline.")

    if previous_accuracy is None:
         logger.warning("Could not establish a baseline test accuracy.")
         return drift_points_chunk_idx


    # Start checking from the second metric entry (representing test on chunk 2)
    for i in range(1, len(sorted_metrics)):
        current_accuracy = sorted_metrics[i].get('test_accuracy')
        current_train_chunk_index = sorted_metrics[i].get('chunk') # Train chunk index is i

        # Check if data is valid for comparison
        if previous_accuracy is not None and current_accuracy is not None and current_train_chunk_index is not None:
            try:
                # Convert to float just in case they are strings
                prev_acc_float = float(previous_accuracy)
                curr_acc_float = float(current_accuracy)

                # Calculate the relative drop
                if prev_acc_float > 1e-9: # Avoid division by zero or near-zero
                    relative_drop = (prev_acc_float - curr_acc_float) / prev_acc_float
                    if relative_drop > threshold:
                        # Drift is observed when testing on chunk (current_train_chunk_index + 1)
                        drift_observed_in_chunk = current_train_chunk_index + 1
                        logger.info(f"Performance drift detected transitioning to test chunk {drift_observed_in_chunk}. "
                                    f"Accuracy dropped from {prev_acc_float:.4f} to {curr_acc_float:.4f} (Drop: {relative_drop:.2%})")
                        drift_points_chunk_idx.append(drift_observed_in_chunk)
                elif curr_acc_float < prev_acc_float: # Handle case where previous accuracy was ~0
                     # If previous was 0, any drop is significant if threshold is > 0
                     if threshold >= 0: # Only report if threshold allows detecting drops from 0
                        drift_observed_in_chunk = current_train_chunk_index + 1
                        logger.info(f"Performance drift detected transitioning to test chunk {drift_observed_in_chunk}. "
                                    f"Accuracy dropped from near zero ({prev_acc_float:.4f}) to {curr_acc_float:.4f}")
                        drift_points_chunk_idx.append(drift_observed_in_chunk)


                # Update previous_accuracy for the next comparison
                previous_accuracy = curr_acc_float

            except (ValueError, TypeError) as e:
                 logger.warning(f"Could not compare accuracies between train chunk {current_train_chunk_index-1} and {current_train_chunk_index} due to invalid data: {e}")
                 # Keep the last valid previous_accuracy
                 continue
        else:
             # If current accuracy or index is missing, we cannot compare
             logger.warning(f"Missing test accuracy or chunk index for train chunk {current_train_chunk_index}, cannot compare drift.")
             # Keep the last valid previous_accuracy


    return drift_points_chunk_idx

# --- Plotting Function with Chunk Boundaries and Debug Logs ---
# (Copied from the Canvas artifact 'plot_annotated_chunks')
def plot_periodic_accuracy_with_detected_drifts(
    periodic_accuracies: List[Tuple[int, float]],
    chunk_metrics: List[Dict],
    detected_drift_chunks: List[int],
    chunk_size: int,
    experiment_id: str,
    run_number: int,
    save_path: str | None = None
    ):
    """
    Plots periodic test accuracy, final train accuracy, marks detected
    performance drifts (red lines), and chunk boundaries (grey lines).
    Includes enhanced logging. All labels/titles in English.
    """
    logger.debug(f"Attempting to generate plot for {experiment_id}-Run{run_number}...")
    logger.debug(f"  Received periodic_accuracies (len={len(periodic_accuracies)}): {str(periodic_accuracies)[:200]}...")
    logger.debug(f"  Received chunk_metrics (len={len(chunk_metrics)}): {str(chunk_metrics)[:200]}...")
    logger.debug(f"  Received detected_drift_chunks: {detected_drift_chunks}")
    logger.debug(f"  Received chunk_size: {chunk_size}")
    logger.debug(f"  Save path: {save_path}")

    if not periodic_accuracies and not chunk_metrics:
        logger.warning(f"[{experiment_id}-Run{run_number}] No accuracy data provided to plot. Skipping plot generation.")
        return

    fig, ax = plt.subplots(figsize=(15, 7))
    plot_generated = False

    try:
        # Plot Periodic Test Accuracy
        if periodic_accuracies:
            try:
                global_counts_test, accuracies_test = zip(*periodic_accuracies)
                ax.plot(global_counts_test, [acc * 100 for acc in accuracies_test], marker='.', linestyle='-', markersize=4, label=f'Periodic Test Accuracy (Run {run_number})', zorder=5)
                plot_generated = True
            except Exception as e: logger.error(f"Error plotting periodic test accuracy: {e}")
        else: logger.warning(f"[{experiment_id}-Run{run_number}] No periodic test accuracy data.")

        # Plot Final Train Accuracy
        if chunk_metrics:
            try:
                train_acc_x = [(m.get('chunk', idx) + 1) * chunk_size for idx, m in enumerate(chunk_metrics)]
                train_acc_y = [m.get('train_accuracy', np.nan) * 100 for m in chunk_metrics]
                valid_indices = [i for i, y in enumerate(train_acc_y) if not np.isnan(y)]
                if valid_indices:
                     ax.plot(np.array(train_acc_x)[valid_indices], np.array(train_acc_y)[valid_indices], marker='o', linestyle=':', markersize=5, label='Train Accuracy (End of Chunk)', color='orange', zorder=4)
                     plot_generated = True
                else: logger.warning(f"[{experiment_id}-Run{run_number}] No valid train accuracy data.")
            except Exception as e: logger.error(f"Error plotting train accuracy: {e}")
        else: logger.warning(f"[{experiment_id}-Run{run_number}] No chunk metrics data for train accuracy.")

        if not plot_generated:
             logger.warning("No data was plotted, skipping boundary and drift markers.")
             plt.close(fig); return

        # Add Chunk Boundary Markers
        num_chunk_transitions = len(chunk_metrics)
        ymin, ymax = ax.get_ylim(); y_pos_chunk_text = ymin + (ymax - ymin) * 0.02
        boundary_label_added = False
        for i in range(num_chunk_transitions):
            boundary_instance = (i + 1) * chunk_size
            label = "Chunk Boundary" if not boundary_label_added else None
            ax.axvline(x=boundary_instance, color='grey', linestyle=':', linewidth=0.8, label=label, zorder=2)
            ax.text(boundary_instance + chunk_size*0.05, y_pos_chunk_text, f'Test Chunk {i+1}', color='grey', ha='left', va='bottom', fontsize=8, rotation=90)
            boundary_label_added = True

        # Add Detected Drift Markers
        y_pos_drift_text = ymax - (ymax - ymin) * 0.02
        drift_label_added = False
        for chunk_idx in detected_drift_chunks:
            drift_occurs_at_instance = chunk_idx * chunk_size
            label = "Detected Drift (Performance Drop)" if not drift_label_added else None
            ax.axvline(x=drift_occurs_at_instance, color='red', linestyle='--', linewidth=1.5, label=label, zorder=3)
            ax.text(drift_occurs_at_instance, y_pos_drift_text, f'Drift\n@Chunk {chunk_idx}', color='red', ha='center', va='top', fontsize=9, backgroundcolor='white', alpha=0.8)
            drift_label_added = True

        ax.set_xlabel("Total Instances Processed (Test Phases)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Periodic Test & Final Train Accuracy: {experiment_id} (Run {run_number})")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_ylim(0, 105)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='best', fontsize='small')

        # Save the plot
        if save_path:
            save_plot_helper(fig, save_path) # Use helper function
        else:
            plt.show(); plt.close(fig)

        logger.debug(f"Plot generation function finished for {experiment_id}-Run{run_number}.")

    except Exception as plot_err:
        logger.error(f"Unexpected error during plot generation for {experiment_id}-Run{run_number}: {plot_err}", exc_info=True)
        plt.close(fig)


# --- Main Script Function ---
def main():
    parser = argparse.ArgumentParser(description="Analyzes saved results from standard runs to detect and visualize performance drifts.")
    parser.add_argument("run_directory", help="Path to the results directory of a single run (e.g., .../SEA/run_1)")
    parser.add_argument("-t", "--threshold", type=float, default=0.10, help="Relative accuracy drop threshold to detect drift (default: 0.10)")
    parser.add_argument("-o", "--output_dir", default=None, help="Directory to save generated plots (default: same as run directory)")
    args = parser.parse_args()

    run_dir = args.run_directory
    output_dir = args.output_dir if args.output_dir else run_dir
    drift_threshold = args.threshold

    logger.info(f"Analyzing results in: {run_dir}")
    logger.info(f"Performance drift detection threshold: {drift_threshold:.2%}")
    logger.info(f"Plots will be saved in: {output_dir}")

    # Load Necessary Data
    run_config = load_json_file(os.path.join(run_dir, "run_config.json"))
    chunk_metrics = load_json_file(os.path.join(run_dir, "chunk_metrics.json"))
    periodic_accuracy = load_json_file(os.path.join(run_dir, "periodic_accuracy.json"))

    if not run_config or not chunk_metrics or not periodic_accuracy:
        logger.error("Essential result files (run_config, chunk_metrics, periodic_accuracy) not found or invalid. Exiting.")
        return

    # Extract info from config
    experiment_id = run_config.get('experiment_id', 'Unknown_Dataset') # type: ignore
    run_number = run_config.get('run_number', 0) # type: ignore
    chunk_size = run_config.get('chunk_size', 500) # type: ignore

    # Detect Performance Drifts
    detected_drift_chunks = detect_performance_drifts(chunk_metrics, threshold=drift_threshold) # type: ignore

    # Generate Annotated Plot
    plot_output_dir = os.path.join(output_dir, "plots") # Save in 'plots' subfolder
    try: os.makedirs(plot_output_dir, exist_ok=True)
    except OSError as e: logger.error(f"Could not create plot dir {plot_output_dir}: {e}"); plot_output_dir = output_dir

    plot_filename = f"Plot_AccuracyPeriodic_DetectedDrifts_{experiment_id}_Run{run_number}.png"
    plot_save_path = os.path.join(plot_output_dir, plot_filename)

    logger.info("Generating annotated accuracy plot...")
    plot_periodic_accuracy_with_detected_drifts(
        periodic_accuracies=periodic_accuracy, # type: ignore
        chunk_metrics=chunk_metrics, # type: ignore
        detected_drift_chunks=detected_drift_chunks,
        chunk_size=chunk_size,
        experiment_id=experiment_id,
        run_number=run_number,
        save_path=plot_save_path
    )

    # --- (Optional) Generate Other Plots ---
    # Load ga_history, rule_details, attribute_usage etc. and call plotting functions
    # Example:
    # ga_history = load_json_file(os.path.join(run_dir, "ga_history_per_chunk.json"))
    # if ga_history and plotting:
    #     logger.info("Generating GA evolution plots...")
    #     for idx, history_data in enumerate(ga_history):
    #          plot_path_ga = os.path.join(plot_output_dir, f"Plot_GA_Evolution_Chunk{idx}_{experiment_id}_Run{run_number}.png")
    #          try: plotting.plot_ga_evolution(history_data, idx, experiment_id, run_number, save_path=plot_path_ga)
    #          except Exception as e: logger.error(f"Failed GA plot chunk {idx}: {e}", exc_info=False)

    logger.info(f"Analysis finished for: {run_dir}")

if __name__ == "__main__":
    # Check for plotting libraries
    try:
        import matplotlib
        import seaborn
        import pandas
    except ImportError as import_err:
        logging.critical(f"Required library missing: {import_err}. Install with 'pip install matplotlib seaborn pandas'. Exiting.")
        exit(1)
    main()
