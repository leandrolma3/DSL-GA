# generate_plots.py
# Script to load saved experiment results and generate plots offline into a 'plots' subfolder.

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import argparse
import logging
from typing import Dict, List, Tuple, Any, Union

# Import existing plotting functions
# Ensure plotting.py is accessible
try:
    import plotting
except ImportError:
    logging.error("Error: plotting.py not found. Ensure it's in the correct directory.")
    # Define placeholder functions to avoid crashing if plotting.py is missing
    class PlottingPlaceholder:
        def __getattr__(self, name):
            def _missing_plot(*args, **kwargs):
                logging.warning(f"Plotting function '{name}' not found (plotting.py missing?). Skipping plot.")
            return _missing_plot
    plotting = PlottingPlaceholder()


# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("generate_plots")

# --- Plotting Function with Drift Info (English) ---
# (plot_periodic_accuracy_with_drift_info function remains the same as before)
def plot_periodic_accuracy_with_drift_info(
    periodic_test_accuracies: List[Tuple[int, float]],
    chunk_train_metrics: List[Dict], 
    stream_definition: Dict,
    chunk_size: int,
    concept_differences: Dict, 
    dataset_type: str, 
    stream_name: str,
    run_number: int,
    save_path: Union[str, None] = None
    ):
    if (not periodic_test_accuracies or not periodic_test_accuracies[0]) and \
       (not chunk_train_metrics or not chunk_train_metrics[0]):
        logging.warning(f"[{stream_name}-Run{run_number}] No periodic test or chunk train accuracy data to plot.")
        return

    fig, ax = plt.subplots(figsize=(18, 9)) # Aumentar um pouco mais a largura

    global_counts_test = []
    if periodic_test_accuracies and periodic_test_accuracies[0]:
        try:
            global_counts_test, accuracies_test = zip(*periodic_test_accuracies)
            ax.plot(global_counts_test, [acc * 100 for acc in accuracies_test], marker='.', linestyle='-', markersize=4, label=f'Periodic Test Accuracy (Run {run_number})', zorder=5)
        except ValueError:
            logging.warning(f"[{stream_name}-Run{run_number}] Could not unpack periodic_test_accuracies.")
            global_counts_test = []
    else:
        logging.warning(f"[{stream_name}-Run{run_number}] No periodic test accuracy data.")

    current_ymin_plot, current_ymax_plot = ax.get_ylim()
    final_ymin = -5.0 # Garantir que 0% seja visível
    final_ymax = 105.0 # Dar espaço para textos no topo
    if current_ymax_plot < 80 and current_ymax_plot > 0: # Se a acurácia máxima for baixa, não precisa ir até 105
        final_ymax = current_ymax_plot + 10
    if current_ymin_plot > 10: # Se a acurácia mínima for alta, não precisa ir até -5
        final_ymin = current_ymin_plot -10
    ax.set_ylim(final_ymin, final_ymax)
    
    # Recalcular as posições y do texto com base nos novos limites
    y_pos_drift_text = final_ymax - (final_ymax - final_ymin) * 0.08 
    y_pos_chunk_label_top = final_ymax - (final_ymax - final_ymin) * 0.13 
    y_pos_train_chunk_text = final_ymin + (final_ymax-final_ymin) * 0.01 # Para texto de treino, um pouco acima da base

    # Plot Final Train Accuracy e adicionar texto de treino
    if chunk_train_metrics and chunk_train_metrics[0]:
        train_acc_x_positions = []
        train_acc_y_values = []
        
        for idx, m in enumerate(chunk_train_metrics):
            train_chunk_index_from_metrics = m.get('chunk', idx)
            x_pos = (train_chunk_index_from_metrics + 1) * chunk_size
            train_acc_x_positions.append(x_pos)
            train_acc_y_values.append(m.get('train_accuracy', np.nan) * 100)

        valid_indices = [i for i, y_val in enumerate(train_acc_y_values) if not np.isnan(y_val)]
        if valid_indices:
            plotted_train_x = np.array(train_acc_x_positions)[valid_indices]
            plotted_train_y = np.array(train_acc_y_values)[valid_indices]
            ax.plot(plotted_train_x, plotted_train_y,
                    marker='o', linestyle=':', markersize=7, label='Train Accuracy (End of Chunk)', color='darkorange', zorder=4)

            for i_val_idx in valid_indices:
                train_chunk_idx_for_label = chunk_train_metrics[i_val_idx].get('chunk', i_val_idx)
                # Posicionar texto do treino um pouco abaixo do ponto laranja
                text_y_pos = plotted_train_y[i_val_idx] - (final_ymax - final_ymin) * 0.001 
                if plotted_train_y[i_val_idx] < (final_ymin + (final_ymax-final_ymin)*0.1): # Se o ponto estiver muito baixo
                    text_y_pos = plotted_train_y[i_val_idx] + (final_ymax - final_ymin) * 0.02 # Coloca acima
                
                ax.text(train_acc_x_positions[i_val_idx], text_y_pos,
                        f"Train Chunk {train_chunk_idx_for_label}\nEnd", # Quebra de linha para melhor ajuste
                        color='chocolate', fontsize=7, ha='center', 
                        va='top' if plotted_train_y[i_val_idx] < (final_ymin + (final_ymax-final_ymin)*0.1) else 'bottom',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
        else:
            logging.warning(f"[{stream_name}-Run{run_number}] No valid train accuracy data.")

    # Add Drift Markers (lógica mantida, mas usa y_pos_drift_text recalculado)
    concept_sequence = stream_definition.get('concept_sequence', [])
    drift_type = stream_definition.get('drift_type', 'abrupt')
    gradual_width_chunks = stream_definition.get('gradual_drift_width_chunks', 0)
    current_train_chunk_idx_for_concept_start = 0 
    previous_concept_id = None
    drift_labels_added_to_legend = set()

    for i_stage, stage in enumerate(concept_sequence):
        concept_id = str(stage['concept_id'])
        duration_chunks = stage['duration_chunks']
        
        if i_stage > 0 and previous_concept_id is not None:
            drift_line_location = current_train_chunk_idx_for_concept_start * chunk_size
            old_id = previous_concept_id
            new_id = concept_id
            severity = 0.0
            pair_key_str = f"{min(str(old_id), str(new_id))}_vs_{max(str(old_id), str(new_id))}"
            try:
                severity = concept_differences.get(dataset_type.upper(), {}).get(pair_key_str, 0.0)
            except Exception as e:
                logging.warning(f"Could not get severity for {old_id}->{new_id} for dataset {dataset_type.upper()}: {e}")
            
            drift_label = f'Drift: {old_id} \u2192 {new_id} ({severity:.1f}%)'
            label_for_legend_entry = drift_label if drift_label not in drift_labels_added_to_legend else None
            
            text_drift_x_offset = chunk_size * 0.05 # Pequeno deslocamento para o texto não ficar sobre a linha
            
            if drift_type == 'abrupt':
                ax.axvline(x=drift_line_location, color='red', linestyle='--', linewidth=1.5, label=label_for_legend_entry, zorder=3)
                ax.text(drift_line_location + text_drift_x_offset, y_pos_drift_text, f'{severity:.1f}%', 
                        color='red', ha='left', va='center', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', pad=1.5, boxstyle='round,pad=0.3'))
            elif drift_type == 'gradual' and gradual_width_chunks > 0:
                transition_end_instance = drift_line_location + (gradual_width_chunks * chunk_size)
                ax.axvspan(drift_line_location, transition_end_instance, color='salmon', alpha=0.2, label=label_for_legend_entry, zorder=1)
                text_x_pos = drift_line_location + (gradual_width_chunks * chunk_size) / 2
                ax.text(text_x_pos, y_pos_drift_text, f'{severity:.1f}% (Gradual)', 
                        color='darkred', ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='darkred', pad=1.5, boxstyle='round,pad=0.3'))
            
            # # --- Adicione este debug ---
            # print("--- DEBUG DRIFT MARKER ---")
            # print(f"Transition to Chunk: {current_train_chunk_idx_for_concept_start}")
            # print(f"Drift Type: {drift_type}")
            # print(f"Gradual Width: {gradual_width_chunks}")
            # print(f"Severity: {severity}")
            # print(f"Drawing from instance {drift_line_location} to {drift_line_location + (gradual_width_chunks * chunk_size)}")
            # # ---------------------------


            if label_for_legend_entry: drift_labels_added_to_legend.add(drift_label)

        previous_concept_id = concept_id
        current_train_chunk_idx_for_concept_start += duration_chunks

    # Marcadores e Texto para Chunks de Teste
    if global_counts_test and chunk_size > 0:
        max_instance_val_on_plot = global_counts_test[-1]
        num_train_chunks_completed = len(chunk_train_metrics) 
        
        boundary_legend_label_added = False
        for k_test_segment_idx in range(1, num_train_chunks_completed + 1): 
            # k_test_segment_idx: 1, 2, ... (para Teste do Chunk 1, Teste do Chunk 2, etc.)
            # O modelo testado aqui foi treinado no chunk de TREINO k_test_segment_idx - 1
            train_chunk_source_idx = k_test_segment_idx - 1
            
            # Linha vertical ao FINAL do segmento de teste
            boundary_instance_loc = k_test_segment_idx * chunk_size 

            if boundary_instance_loc <= max_instance_val_on_plot or np.isclose(boundary_instance_loc, max_instance_val_on_plot):
                label_for_legend = None
                if not boundary_legend_label_added:
                    label_for_legend = "Test Chunk End" # Label para a legenda
                    boundary_legend_label_added = True
                
                ax.axvline(x=boundary_instance_loc, color='gray', linestyle=(0, (3, 5)), linewidth=0.9, label=label_for_legend, zorder=1.5) # Linha pontilhada diferente
                
                # Texto para o segmento de teste (centralizado no segmento)
                text_x_pos_segment = boundary_instance_loc - (chunk_size / 2)
                ax.text(text_x_pos_segment, y_pos_chunk_label_top, 
                        f'Test Phase {k_test_segment_idx}\n(Model from Chunk {train_chunk_source_idx})', 
                        color='dimgray', ha='center', va='top', fontsize=7,
                        bbox=dict(facecolor='white', alpha=0.0, edgecolor='none', pad=1)) # Transparente

    ax.set_xlabel("Total Instances Processed (Cumulative in Test Phases)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Periodic Test & Final Train Accuracy w/ Drifts: {stream_name} (Run {run_number})", fontsize=14)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5) # Apenas grades principais
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.7) # Grades menores mais suaves
    ax.minorticks_on() # Ligar minor ticks para que a grade menor apareça

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Mover a legenda para FORA do plot, no canto superior direito
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', 
              bbox_to_anchor=(1.01, 1), # (x, y) onde (0,0) é canto inf esq do eixo, (1,1) é canto sup dir
              borderaxespad=0., # preenchimento entre a caixa da legenda e o bbox_to_anchor
              fontsize='small')

    fig.tight_layout(rect=[0, 0, 0.85, 0.95]) # type: ignore # Ajustar rect para dar espaço para a legenda à DIREITA e título

    if save_path:
        try:
            base_dir = os.path.dirname(save_path)
            if base_dir and not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight')
            logging.info(f"Accuracy plot with drift info saved to: {save_path}")
        except Exception as e_save:
            logging.error(f"Failed to save accuracy plot to {save_path}: {e_save}")
        finally:
            plt.close(fig)
    else:
        try:
            plt.show()
        except Exception as e_show:
            logging.warning(f"Could not show plot interactively: {e_show}")
        finally:
            plt.close(fig)

# --- Main Script Function ---
def generate_plots_for_run(run_dir: str, output_dir: Union[str, None] = None, diff_data: Union[Dict, None] = None):
    """Loads data for a specific run and generates all plots into a 'plots' subfolder."""

    logger.info(f"--- Processing results directory: {run_dir} ---")

    if not os.path.isdir(run_dir): logger.error(f"Directory not found: {run_dir}"); return

    # --- <<< MODIFIED: Define plot output subfolder >>> ---
    if output_dir is None: output_dir = run_dir # Default to run dir if not specified
    plot_output_dir = os.path.join(output_dir, "plots") # Define subfolder name
    try:
        os.makedirs(plot_output_dir, exist_ok=True) # Ensure the subfolder exists
        logger.info(f"Plots will be saved in: {plot_output_dir}")
    except OSError as e:
         logger.error(f"Could not create plot output directory {plot_output_dir}: {e}")
         return # Cannot proceed without output directory
    # --- <<< END MODIFICATION >>> ---

    # --- Load Data Files ---
    run_config = None; periodic_accuracy = None; ga_history = None
    rule_details = None; attribute_usage = None; chunk_metrics = None

    try:
        # Load all necessary JSON files
        with open(os.path.join(run_dir, "run_config.json"), 'r') as f: run_config = json.load(f)
        with open(os.path.join(run_dir, "periodic_accuracy.json"), 'r') as f: periodic_accuracy = json.load(f)
        with open(os.path.join(run_dir, "ga_history_per_chunk.json"), 'r') as f: ga_history = json.load(f)
        with open(os.path.join(run_dir, "rule_details_per_chunk.json"), 'r') as f: rule_details = json.load(f)
        with open(os.path.join(run_dir, "attribute_usage_per_chunk.json"), 'r') as f: attribute_usage = json.load(f)
        with open(os.path.join(run_dir, "chunk_metrics.json"), 'r') as f: chunk_metrics = json.load(f)
        logger.info("Result files loaded successfully.")
    except FileNotFoundError as fnf_e: logger.error(f"Error loading result file: {fnf_e}. Skipping plots."); return
    except Exception as e: logger.error(f"Error loading result data: {e}", exc_info=True); return

    # Extract info needed for plots
    stream_name = run_config.get('experiment_id', 'UnknownExperiment'); run_number = run_config.get('run_number', 0)
    chunk_size = run_config.get('chunk_size', 500); stream_definition = run_config.get('stream_definition', {})
    dataset_type = run_config.get('dataset_type', 'UnknownType'); all_attributes = run_config.get('attributes')

    # --- Generate Plots (Saving to plot_output_dir) ---

    # 1. Periodic Accuracy Plot
    # <<< MODIFIED: Use plot_output_dir >>>
    plot_path_periodic = os.path.join(plot_output_dir, f"Plot_AccuracyPeriodic_{stream_name}_Run{run_number}.png")
    # <<< END MODIFICATION >>>
    if periodic_accuracy and chunk_metrics and stream_definition and chunk_size and diff_data:
        plot_periodic_accuracy_with_drift_info(periodic_accuracy, chunk_metrics, stream_definition, chunk_size, diff_data, dataset_type, stream_name, run_number, plot_path_periodic)
    else: logger.warning("Insufficient data for periodic accuracy plot with drift info.")

    # 2. GA Evolution Plots
    if ga_history:
        logger.info("Generating GA evolution plots per chunk...")
        for idx, history_data in enumerate(ga_history):
            # <<< MODIFIED: Use plot_output_dir >>>
            plot_path_ga = os.path.join(plot_output_dir, f"Plot_GA_Evolution_Chunk{idx}_{stream_name}_Run{run_number}.png")
            # <<< END MODIFICATION >>>
            try: plotting.plot_ga_evolution(history_data, idx, stream_name, run_number, save_path=plot_path_ga)
            except Exception as e: logger.error(f"Failed GA plot chunk {idx}: {e}", exc_info=False)
    else: logger.warning("No GA history data.")

    # 3. Rule Analysis Plots
    if rule_details:
        # <<< MODIFIED: Use plot_output_dir >>>
        plot_path_heatmap = os.path.join(plot_output_dir, f"Plot_RuleComponents_Heatmap_{stream_name}_Run{run_number}.png")
        plot_path_radar = os.path.join(plot_output_dir, f"Plot_RuleComponents_Radar_{stream_name}_Run{run_number}.png")
        # <<< END MODIFICATION >>>
        try: plotting.plot_rule_changes(rule_details, stream_name, run_number, save_path=plot_path_heatmap)
        except Exception as e: logger.error(f"Failed rule components heatmap: {e}", exc_info=False)
        try: plotting.plot_rule_info_radar(rule_details, stream_name, run_number, save_path=plot_path_radar)
        except Exception as e: logger.error(f"Failed rule components radar plot: {e}", exc_info=False)
    else: logger.warning("No detailed rule data.")

    # 4. Attribute Usage Plot
    if attribute_usage and all_attributes:
        # <<< MODIFIED: Use plot_output_dir >>>
        plot_path_attrs = os.path.join(plot_output_dir, f"Plot_AttributeUsage_{stream_name}_Run{run_number}.png")
        # <<< END MODIFICATION >>>
        try: plotting.plot_attribute_usage_over_time(attribute_usage, all_attributes, stream_name, run_number, save_path=plot_path_attrs)
        except Exception as e: logger.error(f"Failed attribute usage plot: {e}", exc_info=False)
    elif not all_attributes: logger.warning("Attribute list missing, skipping attribute usage plot.")
    else: logger.warning("No attribute usage data.")

    logger.info(f"--- Plot generation finished for: {run_dir} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from saved experiment run results.")
    parser.add_argument("run_directory", help="Path to the results directory of a single experiment run (e.g., .../SEA_Abrupt/run_1)")
    parser.add_argument("-o", "--output_dir", default=None, help="Base directory to save plots subfolder (default: same as run directory)")
    parser.add_argument("-d", "--diff_file", default="results/concept_heatmaps/concept_differences.json", help="Path to the JSON file with pre-calculated concept differences.")
    args = parser.parse_args()

    try: import matplotlib; import seaborn; import pandas; import json
    except ImportError as import_err: logger.critical(f"Missing required library: {import_err}. Install with 'pip install matplotlib seaborn pandas PyYAML numpy'. Exiting."); exit(1)

    concept_diff_data = None
    if os.path.exists(args.diff_file):
        try:
            with open(args.diff_file, 'r') as f_diff: concept_diff_data = json.load(f_diff)
            logger.info(f"Concept difference data loaded from {args.diff_file}")
        except Exception as e: logger.error(f"Failed to load concept difference file '{args.diff_file}': {e}.")
    else: logger.warning(f"Concept difference file not found at '{args.diff_file}'. Plots will lack severity info.")

    generate_plots_for_run(args.run_directory, args.output_dir, concept_diff_data)

    logger.info("Plot generation script finished.")
