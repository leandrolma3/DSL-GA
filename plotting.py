# plotting.py (Com função renomeada)

"""
Contains functions for generating various plots to visualize the results
of the Genetic Algorithm rule learning experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Used for heatmaps primarily
import os
import math
import logging
from typing import Dict, List, Tuple, Any # <<< ADICIONADO typing >>>

# Optional: Set a consistent style
# plt.style.use('seaborn-v0_8-darkgrid')
# sns.set_theme(style="whitegrid")


def _save_plot(fig, save_path):
    """Helper function to save a plot if a path is provided."""
    if save_path:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight')
            logging.debug(f"Plot saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save plot to {save_path}: {e}")
    plt.close(fig) # Close the figure to free memory


def plot_attribute_usage_over_time(used_attributes_over_time, all_attributes, dataset_name,
                                    run_number=1, ax=None, save_path=None):
    # ... (código como antes) ...
    standalone_plot = ax is None
    if standalone_plot:
        fig, ax = plt.subplots(figsize=(max(8, len(used_attributes_over_time) * 0.6),
                                         max(6, len(all_attributes) * 0.3))) # Dynamic sizing
    else:
        fig = ax.figure # Get figure from axes

    if not all_attributes:
         logging.warning("No attributes provided for attribute usage plot.")
         ax.text(0.5, 0.5, "No attributes to display", ha='center', va='center')
         ax.set_title(f'Attribute Usage Over Chunks\n{dataset_name} - Run {run_number}')
         if standalone_plot: _save_plot(fig, save_path)
         return

    # Create a matrix indicating if an attribute was used (1) or not (0)
    attribute_usage_matrix = []
    for used_attrs_in_chunk in used_attributes_over_time:
        # Handle case where used_attrs_in_chunk might be None or not a set/list
        if not isinstance(used_attrs_in_chunk, (set, list)):
             usage_vector = [0] * len(all_attributes) # Default to not used if data is bad
             logging.warning(f"Invalid type for used attributes in chunk: {type(used_attrs_in_chunk)}. Expected set or list.")
        else:
             usage_vector = [1 if attr in used_attrs_in_chunk else 0 for attr in all_attributes]
        attribute_usage_matrix.append(usage_vector)

    # Convert to numpy array and transpose for heatmap (features on y-axis)
    if not attribute_usage_matrix: # Handle empty history
         logging.warning("No attribute usage history to plot.")
         ax.text(0.5, 0.5, "No usage history", ha='center', va='center')
         ax.set_title(f'Attribute Usage Over Chunks\n{dataset_name} - Run {run_number}')
         if standalone_plot: _save_plot(fig, save_path)
         return

    try:
        attribute_usage_matrix = np.array(attribute_usage_matrix).T
    except ValueError as e:
         logging.error(f"Could not create numpy array for usage matrix: {e}")
         return


    # Generate the heatmap
    cax = sns.heatmap(attribute_usage_matrix, cmap='Blues', ax=ax, cbar=standalone_plot, # Only add cbar to standalone
                      linewidths=0.1, linecolor='lightgrey',
                      cbar_kws={'label': 'Attribute Usage (1=Used)'} if standalone_plot else {})

    # Configure axes
    ax.set_yticks(np.arange(len(all_attributes)) + 0.5) # Center ticks
    ax.set_yticklabels(all_attributes, rotation=0)
    ax.set_xticks(np.arange(len(used_attributes_over_time)) + 0.5) # type: ignore
    ax.set_xticklabels([f'C{i}' for i in range(len(used_attributes_over_time))], rotation=45, ha='right')

    ax.set_title(f'Attribute Usage Over Chunks\n{dataset_name} - Run {run_number}')
    if standalone_plot: ax.set_xlabel('Chunks') # Only add axis labels to standalone
    if standalone_plot: ax.set_ylabel('Attributes')

    if standalone_plot:
        fig.tight_layout()
        _save_plot(fig, save_path)


def plot_performance_metrics(performance_metrics, dataset_name, run_number=1, ax=None, save_path=None):
    # ... (código como antes) ...
    standalone_plot = ax is None
    if standalone_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    if not performance_metrics:
        logging.warning("No performance metrics to plot.")
        ax.text(0.5, 0.5, "No performance data", ha='center', va='center')
        ax.set_title(f'Performance Over Chunks\n{dataset_name} - Run {run_number}')
        if standalone_plot: _save_plot(fig, save_path)
        return

    chunks = [m.get('chunk', idx) for idx, m in enumerate(performance_metrics)] # Use index if 'chunk' key missing
    train_accuracies = [m.get('train_accuracy', np.nan) for m in performance_metrics]
    test_accuracies = [m.get('test_accuracy', np.nan) for m in performance_metrics]

    ax.plot(chunks, train_accuracies, marker='o', linestyle='-', label='Train Accuracy (Chunk i)')
    ax.plot(chunks, test_accuracies, marker='s', linestyle='--', label='Test Accuracy (Chunk i+1)')
    ax.set_title(f'Performance Over Chunks\n{dataset_name} - Run {run_number}')
    ax.set_xlabel("Chunk Index (i)")
    ax.set_ylabel("Accuracy")
    if chunks: # Avoid error if chunks is empty
         ax.set_xticks(chunks) # Ensure ticks align with chunk indices
    ax.legend()
    ax.grid(True, linestyle=':')
    ax.set_ylim(0, 1.05) # Accuracy is between 0 and 1

    if standalone_plot:
        fig.tight_layout()
        _save_plot(fig, save_path)


def plot_rules_conditionals(rules_conditionals, dataset_name, run_number=1, ax=None, save_path=None):
    # ... (código como antes) ...
    standalone_plot = ax is None
    if standalone_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    if not rules_conditionals:
        logging.warning("No rule/conditional counts to plot.")
        ax.text(0.5, 0.5, "No complexity data", ha='center', va='center')
        ax.set_title(f'Rule Complexity Over Chunks\n{dataset_name} - Run {run_number}')
        if standalone_plot: _save_plot(fig, save_path)
        return

    chunks = [rc.get('chunk', idx) for idx, rc in enumerate(rules_conditionals)]
    total_rules = [rc.get('total_rules', np.nan) for rc in rules_conditionals]
    total_nodes = [rc.get('total_conditionals', np.nan) for rc in rules_conditionals] # Assumes 'total_conditionals' is total nodes

    ax.plot(chunks, total_rules, marker='o', linestyle='-', label='Total Rules')
    ax.plot(chunks, total_nodes, marker='s', linestyle='--', label='Total Nodes (Conditions)') # Clarified label
    ax.set_title(f'Rule Complexity Over Chunks\n{dataset_name} - Run {run_number}')
    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Count")
    if chunks:
         ax.set_xticks(chunks)
    ax.legend()
    ax.grid(True, linestyle=':')
    ax.set_ylim(bottom=0) # Ensure y-axis starts at 0

    if standalone_plot:
        fig.tight_layout()
        _save_plot(fig, save_path)


# <<< RENOMEADO: plot_fitness_evolution para plot_ga_evolution >>>
def plot_ga_evolution(history, chunk_index, dataset_name, run_number=1, ax=None, save_path=None):
    """
    Plots the evolution of best/average fitness and accuracy over generations
    for a single chunk's GA run.
    """
    standalone_plot = ax is None

    # Check if history dictionary and required keys exist and are not empty
    if not history or \
       not history.get('best_fitness') or \
       not isinstance(history['best_fitness'], list) or \
       len(history['best_fitness']) == 0:
        logging.warning(f"No valid 'best_fitness' data in history for chunk {chunk_index} to plot.")
        # Optionally draw placeholder text if standalone_plot
        if standalone_plot:
             fig, ax_placeholder = plt.subplots(figsize=(7, 5))
             ax_placeholder.text(0.5, 0.5, f"No GA History for Chunk {chunk_index}", ha='center', va='center')
             ax_placeholder.set_title(f'GA Evolution - Chunk {chunk_index}\n{dataset_name} - Run {run_number}')
             _save_plot(fig, save_path)
        # If ax is provided, do nothing on the provided axes
        return

    num_generations = len(history['best_fitness'])
    generations = range(num_generations)

    # Ensure other keys exist and have the correct length, providing defaults if missing
    avg_fitness = history.get('avg_fitness', [np.nan] * num_generations)
    std_fitness = history.get('std_fitness', [0] * num_generations)
    best_accuracy = history.get('best_accuracy', [np.nan] * num_generations)
    avg_accuracy = history.get('avg_accuracy', [np.nan] * num_generations)
    std_accuracy = history.get('std_accuracy', [0] * num_generations)

    # Pad shorter lists if necessary (though ideally they should match)
    if len(avg_fitness) != num_generations: avg_fitness.extend([np.nan]*(num_generations-len(avg_fitness)))
    if len(std_fitness) != num_generations: std_fitness.extend([0]*(num_generations-len(std_fitness)))
    if len(best_accuracy) != num_generations: best_accuracy.extend([np.nan]*(num_generations-len(best_accuracy)))
    if len(avg_accuracy) != num_generations: avg_accuracy.extend([np.nan]*(num_generations-len(avg_accuracy)))
    if len(std_accuracy) != num_generations: std_accuracy.extend([0]*(num_generations-len(std_accuracy)))


    # If standalone, create a figure with two subplots
    if standalone_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_fitness = axes[0]
        ax_accuracy = axes[1]
        fig.suptitle(f'GA Evolution - Chunk {chunk_index}\n{dataset_name} - Run {run_number}', fontsize=14)
    else:
        # Assume ax is provided, plot fitness on it
        fig = ax.figure
        ax_fitness = ax # Plot fitness on the provided axis
        # Accuracy plot needs separate handling or a different function if ax is provided
        ax_accuracy = None # Indicate accuracy plot is not available on single axis

    # --- Plot Fitness ---
    ax_fitness.plot(generations, history['best_fitness'], label='Best Fitness', color='blue')
    ax_fitness.plot(generations, avg_fitness, label='Avg Fitness', color='cyan')
    lower_bound = np.array(avg_fitness) - np.array(std_fitness)
    upper_bound = np.array(avg_fitness) + np.array(std_fitness)
    ax_fitness.fill_between(generations, lower_bound, upper_bound, color='cyan', alpha=0.2, label='Std Dev Fitness')
    ax_fitness.set_title('Fitness Evolution')
    ax_fitness.set_xlabel('Generations')
    ax_fitness.set_ylabel('Fitness')
    ax_fitness.legend()
    ax_fitness.grid(True, linestyle=':')

    # --- Plot Accuracy (only if standalone plot was created) ---
    if ax_accuracy is not None: # Check if ax_accuracy exists
        ax_accuracy.plot(generations, best_accuracy, label='Best Accuracy', color='red')
        ax_accuracy.plot(generations, avg_accuracy, label='Avg Accuracy', color='orange')
        lower_bound_acc = np.array(avg_accuracy) - np.array(std_accuracy)
        upper_bound_acc = np.array(avg_accuracy) + np.array(std_accuracy)
        ax_accuracy.fill_between(generations, np.maximum(0, lower_bound_acc), np.minimum(1, upper_bound_acc),
                                   color='orange', alpha=0.2, label='Std Dev Accuracy')
        ax_accuracy.set_title('Accuracy Evolution (on Train Data)')
        ax_accuracy.set_xlabel('Generations')
        ax_accuracy.set_ylabel('Accuracy')
        ax_accuracy.legend()
        ax_accuracy.grid(True, linestyle=':')
        ax_accuracy.set_ylim(0, 1.05)

    if standalone_plot:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore # Adjust layout for suptitle
        _save_plot(fig, save_path)
    # Note: If ax was provided, only fitness is plotted on it.


def create_mosaic_plots(dataset_name, runs_results, attributes, save_path=None):
    # ... (código como antes, mas chame a função renomeada plot_ga_evolution) ...
    num_runs = len(runs_results)
    if num_runs == 0:
        logging.warning("No run results provided for mosaic plot.")
        return

    fig, axes = plt.subplots(nrows=3, ncols=num_runs, figsize=(num_runs * 5.5, 15), squeeze=False)

    for run_idx in range(num_runs):
        run_data = runs_results[run_idx]
        used_attributes_over_time = run_data.get('used_attributes_over_time', [])
        performance_metrics = run_data.get('performance_metrics', [])
        rules_conditionals = run_data.get('rules_conditionals', [])
        # <<< ADICIONADO: Obter histórico do GA para plotar evolução >>>
        # Assumindo que 'fitness_accuracy_history_per_chunk' está em run_data
        # E que plot_ga_evolution agora lida com a lista de históricos por chunk
        # Se plot_ga_evolution SÓ plota UM chunk, esta chamada está errada aqui.
        # Precisa de uma função que plote a evolução MÉDIA ou algo assim.
        # Vamos remover a chamada ao plot_ga_evolution daqui por enquanto.
        # O plot mosaico focará em Atributos, Performance, Complexidade.
        # A evolução do GA por chunk deve ser plotada separadamente.

        ax_attr = axes[0, run_idx]
        plot_attribute_usage_over_time(
            used_attributes_over_time, attributes, dataset_name,
            run_number=run_idx + 1, ax=ax_attr
        )
        ax_attr.set_title(f'Run {run_idx + 1}')
        if run_idx > 0: ax_attr.set_ylabel('')

        ax_perf = axes[1, run_idx]
        plot_performance_metrics(
            performance_metrics, dataset_name,
            run_number=run_idx + 1, ax=ax_perf
        )
        ax_perf.set_title('')
        if run_idx > 0: ax_perf.set_ylabel('')

        ax_rules = axes[2, run_idx]
        plot_rules_conditionals(
            rules_conditionals, dataset_name,
            run_number=run_idx + 1, ax=ax_rules
        )
        ax_rules.set_title('')
        if run_idx > 0: ax_rules.set_ylabel('')

    fig.suptitle(f"Per-Run Metrics Summary: {dataset_name}", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # type: ignore
    _save_plot(fig, save_path)


def plot_overall_performance_with_std(all_datasets_summary, save_path=None, group_by='dataset_type'):
    # ... (código como antes, talvez adicionar group_by se necessário) ...
    if not all_datasets_summary:
        logging.warning("No dataset summaries provided for overall performance plot.")
        return

    # Example: Simple plot without grouping if group_by isn't implemented downstream
    dataset_names = [ds.get('stream_name', ds.get('dataset_name', f'Stream {i}')) for i, ds in enumerate(all_datasets_summary)]
    avg_train_acc = [ds.get('average_train_accuracy', np.nan) for ds in all_datasets_summary]
    std_train_acc = [ds.get('train_accuracy_std', 0) for ds in all_datasets_summary] # Use 0 if std missing
    avg_test_acc = [ds.get('average_test_accuracy', np.nan) for ds in all_datasets_summary]
    std_test_acc = [ds.get('test_accuracy_std', 0) for ds in all_datasets_summary] # Use 0 if std missing

    x = np.arange(len(dataset_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(dataset_names) * 1.2), 6))

    rects1 = ax.bar(x - width/2, avg_train_acc, width, yerr=std_train_acc,
                      label='Avg Train Accuracy', capsize=5, color='lightblue', ecolor='gray')
    rects2 = ax.bar(x + width/2, avg_test_acc, width, yerr=std_test_acc,
                      label='Avg Test Accuracy', capsize=5, color='lightcoral', ecolor='gray')

    ax.set_ylabel('Accuracy')
    ax.set_title('Average Performance Across Streams (with Std Dev over runs)')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', linestyle=':')
    ax.set_ylim(0, 1.05)

    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')

    fig.tight_layout()
    _save_plot(fig, save_path)


# --- Plots for Rule Evolution Analysis ---
# (plot_rule_changes, plot_rule_feature_changes_arrow_polar, plot_rule_info_radar)
# ... (código como antes) ...
# --- Funções plot_rule_changes, plot_rule_feature_changes_arrow_polar, plot_rule_info_radar (sem alterações) ---

def plot_rule_changes(rule_change_history, dataset_name, run_number=1, save_path=None):
    if not rule_change_history: return
    num_chunks = len(rule_change_history)
    logical_ops = [len(chunk.get("logical_ops", [])) for chunk in rule_change_history]
    comp_ops = [len(chunk.get("comparison_ops", [])) for chunk in rule_change_history]
    # <<< CORRIGIDO: Usar chave correta >>>
    thresholds = [len(chunk.get("numeric_thresholds", [])) for chunk in rule_change_history]
    features = [len(chunk.get("features", [])) for chunk in rule_change_history]
    # <<< ADICIONADO: Valores Categóricos Usados >>>
    cat_values = [len(chunk.get("categorical_values_used", [])) for chunk in rule_change_history]

    data_matrix = np.array([logical_ops, comp_ops, thresholds, features, cat_values])
    y_labels = ["Logical Ops", "Comparison Ops", "Numeric Thresh.", "Features Used", "Categorical Vals"]

    fig, ax = plt.subplots(figsize=(max(8, num_chunks * 0.8), 6)) # Ajustado tamanho
    sns.heatmap(data_matrix, annot=True, fmt="d", cmap="viridis", ax=ax,
                xticklabels=[f'C{i}' for i in range(num_chunks)],
                yticklabels=y_labels, cbar=True, cbar_kws={'label': 'Count'}) # Adicionado label cbar

    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Rule Component")
    ax.set_title(f"Rule Component Evolution - {dataset_name} (Run {run_number})")
    plt.yticks(rotation=0)
    fig.tight_layout()
    _save_plot(fig, save_path)

def plot_rule_feature_changes_arrow_polar(rule_info_history, dataset_name, run_number=1, save_path=None):
    if len(rule_info_history) < 2: return
    # <<< CORRIGIDO: Usa as mesmas chaves do plot_rule_changes >>>
    categories = ['Logical Ops', 'Comparison Ops', 'Numeric Thresh.', 'Features Used', 'Categorical Vals']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1] # type: ignore # Fecha o círculo

    num_transitions = len(rule_info_history) - 1
    fig, axes = plt.subplots(1, num_transitions, subplot_kw=dict(polar=True),
                             figsize=(5 * num_transitions, 5.5), squeeze=False)

    max_val_overall = 0
    all_counts = []
    for i in range(len(rule_info_history)):
        info = rule_info_history[i]
        counts = np.array([
            len(info.get('logical_ops', [])),
            len(info.get('comparison_ops', [])),
            len(info.get('numeric_thresholds', [])),
            len(info.get('features', [])),
            len(info.get('categorical_values_used', []))
        ])
        all_counts.append(counts)
        if len(counts) > 0: # Evita erro se counts for vazio
             max_val_overall = max(max_val_overall, np.max(counts))

    radial_limit = max(1, max_val_overall + 2) # Garante limite > 0

    for i in range(num_transitions):
        prev_counts = all_counts[i]
        curr_counts = all_counts[i+1]
        ax = axes[0, i]
        ax.set_title(f'Change: C{i} -> C{i+1}', fontsize=10, pad=20)

        # Garante que os arrays tenham o tamanho N antes de concatenar
        plot_prev_counts = np.concatenate((prev_counts, prev_counts[:1])) if len(prev_counts)==N else np.zeros(N+1)
        plot_curr_counts = np.concatenate((curr_counts, curr_counts[:1])) if len(curr_counts)==N else np.zeros(N+1)

        # Plota linha base (pode ajudar a visualização)
        ax.plot(angles, plot_prev_counts, color='grey', linestyle='--', linewidth=0.8, label=f'C{i}' if i==0 else "")
        # Plota linha atual
        line, = ax.plot(angles, plot_curr_counts, color='blue', linewidth=1.5, linestyle='solid', label=f'C{i+1}' if i==0 else "")
        #ax.fill(angles, plot_curr_counts, color='blue', alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        # Define ticks radiais de forma mais robusta
        max_tick = math.ceil(radial_limit / 4) * 4 if radial_limit > 4 else radial_limit
        ax.set_yticks(np.linspace(0, max_tick, 5)) # 5 ticks incluindo 0
        ax.set_ylim(0, radial_limit)
        ax.grid(True, linestyle=':')

    # Adiciona legenda fora dos plots
    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=num_transitions+1)
    fig.suptitle(f'Rule Component Changes (Radar) - {dataset_name} (Run {run_number})', fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # type: ignore # Ajusta para legenda/título
    _save_plot(fig, save_path)


def plot_rule_info_radar(rule_info_history, dataset_name, run_number=1, save_path=None):
     # ESTA FUNÇÃO É MUITO SIMILAR à plot_rule_feature_changes_arrow_polar
     # A diferença principal seria não mostrar setas, mas sim linhas separadas para cada chunk.
     # Vou adaptar a função anterior para fazer isso, já que o nome 'radar' se encaixa melhor aqui.
    if not rule_info_history: return
    num_chunks = len(rule_info_history)
    # <<< CORRIGIDO: Usa as mesmas chaves >>>
    categories = ['Logical Ops', 'Comparison Ops', 'Numeric Thresh.', 'Features Used', 'Categorical Vals']
    num_vars = len(categories)

    data = []
    for info in rule_info_history:
        counts = [
            len(info.get('logical_ops', [])), len(info.get('comparison_ops', [])),
            len(info.get('numeric_thresholds', [])), len(info.get('features', [])),
            len(info.get('categorical_values_used', []))
        ]
        data.append(counts)
    if not data: return
    data = np.array(data)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # type: ignore

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.viridis(np.linspace(0, 1, num_chunks)) # type: ignore

    max_val_overall = data.max() if data.size > 0 else 0
    radial_limit = max(1, max_val_overall + 2)

    for i in range(num_chunks):
        chunk_data = np.concatenate((data[i], data[i,:1])) # Fecha o loop
        ax.plot(angles, chunk_data, color=colors[i], linewidth=1.5, linestyle='solid', label=f'Chunk {i}')
        # ax.fill(angles, chunk_data, color=colors[i], alpha=0.1) # Opcional

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    max_tick = math.ceil(radial_limit / 4) * 4 if radial_limit > 4 else radial_limit
    ax.set_yticks(np.linspace(0, max_tick, 5))
    ax.set_ylim(0, radial_limit)
    ax.set_title(f'Rule Component Evolution (Radar) - {dataset_name} (Run {run_number})', size=15, y=1.1)
    # Ajusta posição da legenda
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=min(5, num_chunks))

    fig.tight_layout()
    _save_plot(fig, save_path)