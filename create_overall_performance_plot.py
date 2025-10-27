import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: Try to import seaborn for better plot aesthetics
try:
    import seaborn as sns
    sns_available = True
    sns.set_style("whitegrid") # Apply a default style if seaborn is available
except ImportError:
    sns_available = False
    # print("Seaborn library not found. Plots will be generated using basic Matplotlib.")

# ----- FUNÇÃO DE PARSING (permanece a mesma) -----
def parse_rules_history_file(file_path):
    test_acc_values = []
    test_f1_values = []
    train_acc_values = []
    perf_line1_regex = re.compile(
        r"--- Chunk\s+\d+\s+\((.*?)\)\s*---\s*"
        r"Test Perf\s*\(Chunk\s+\d+\s*\)\s*:\s*"
        r"TestAcc\s*=\s*([\d.]+)\s*,"
        r"\s*TestF1\s*=\s*([\d.]+)"
    )
    perf_line2_regex = re.compile(
        r"^\s*Train Perf\s*\(Chunk\s+\d+\s*\)\s*:\s*"
        r"TrainAcc\s*=\s*([\d.]+)"
    )
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            match1 = perf_line1_regex.search(lines[i].strip())
            if match1:
                if i + 1 < len(lines):
                    match2 = perf_line2_regex.search(lines[i+1].strip())
                    if match2:
                        try:
                            test_acc_str = match1.group(2)
                            test_f1_str = match1.group(3)
                            train_acc_str = match2.group(1)
                            test_acc = float(test_acc_str)
                            test_f1 = float(test_f1_str)
                            train_acc = float(train_acc_str)
                            test_acc_values.append(test_acc)
                            test_f1_values.append(test_f1)
                            train_acc_values.append(train_acc)
                            i += 2 
                            continue
                        except ValueError:
                            print(f"Warning: Could not parse numbers from matched performance block: "
                                  f"L1='{lines[i].strip()}', L2='{lines[i+1].strip()}' in {file_path}")
                        except IndexError:
                            print(f"Warning: Regex matched but groups missing for performance block: "
                                  f"L1='{lines[i].strip()}', L2='{lines[i+1].strip()}' in {file_path}")
                        i += 2
                        continue
                else:
                    pass 
            i += 1
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return None
    except Exception as e:
        print(f"Error reading or parsing file {file_path}: {e}")
        return None
    if not test_acc_values and not test_f1_values and not train_acc_values:
        pass
    return {
        "TestAcc": test_acc_values,
        "TestF1": test_f1_values,
        "TrainAcc": train_acc_values
    }

# ----- FUNÇÃO DE PROCESSAMENTO DE DADOS (permanece a mesma) -----
def process_experiment_data(base_dir):
    all_results = []
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return None
    # Ajustado para aceitar nomes de pasta como "AGRAWAL" ou "AGRAWAL_suffix"
    folder_name_pattern = re.compile(r"^[A-Z]+_") # para drift simulation mode
    #folder_name_pattern = re.compile(r"^[A-Z]+(?:_.*)?$") 
    for item_name in os.listdir(base_dir):
        if not folder_name_pattern.match(item_name):
            continue
        experiment_folder_path = os.path.join(base_dir, item_name)
        if not os.path.isdir(experiment_folder_path):
            continue
        run_folder_path = os.path.join(experiment_folder_path, "run_1")
        if not os.path.isdir(run_folder_path):
            continue
        rules_files = glob.glob(os.path.join(run_folder_path, "RulesHistory_*.txt"))
        if not rules_files:
            continue
        for rules_file_path in rules_files:
            filename_base = os.path.basename(rules_file_path)
            dataset_name_match = re.match(r"RulesHistory_(.*?)_Run\d+\.txt", filename_base)
            if not dataset_name_match:
                dataset_name_match = re.match(r"RulesHistory_(.*?)\.txt", filename_base)
            if dataset_name_match:
                dataset_name = dataset_name_match.group(1)
            else:
                dataset_name = f"{item_name}_{filename_base.replace('.txt','')}" 
            performance_data = parse_rules_history_file(rules_file_path)
            if performance_data and (performance_data['TestAcc'] or performance_data['TestF1'] or performance_data['TrainAcc']):
                result_entry = {"Dataset": dataset_name}
                has_data = False
                for metric, values in performance_data.items():
                    if values:
                        result_entry[f"Mean_{metric}"] = np.mean(values)
                        result_entry[f"Std_{metric}"] = np.std(values)
                        has_data = True
                    else:
                        result_entry[f"Mean_{metric}"] = np.nan
                        result_entry[f"Std_{metric}"] = np.nan
                if has_data:
                    all_results.append(result_entry)
    if not all_results:
        print("No data successfully processed from any experiment folder matching the pattern.")
        return None
    return pd.DataFrame(all_results)

# ----- FUNÇÃO DE PLOTAGEM DETALHADA POR DATASET (permanece a mesma) -----
def plot_dataset_train_test_performance(dataset_stats_row, dataset_name, output_filename):
    metrics_to_display = ['Train Accuracy', 'Test Accuracy']
    mean_values = [
        dataset_stats_row.get('Mean_TrainAcc', np.nan), 
        dataset_stats_row.get('Mean_TestAcc', np.nan)
    ]
    std_dev_values = [
        dataset_stats_row.get('Std_TrainAcc', np.nan),
        dataset_stats_row.get('Std_TestAcc', np.nan)
    ]
    if pd.isna(mean_values).any() or pd.isna(std_dev_values).any():
        return
    x_pos = np.arange(len(metrics_to_display))
    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.bar(x_pos, mean_values, yerr=std_dev_values, align='center', alpha=0.75, ecolor='black', capsize=10, color=['dodgerblue', 'salmon'])
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_to_display, fontsize=11)
    title = f'Train vs. Test Mean Accuracy for:\n{dataset_name}'
    ax.set_title(title, fontsize=14, pad=20)
    upper_y_limit = 0
    if not pd.isna(mean_values).all() and not pd.isna(std_dev_values).all():
         non_nan_means = np.array(mean_values)[~pd.isna(mean_values)]
         non_nan_stds = np.array(std_dev_values)[~pd.isna(std_dev_values)]
         if len(non_nan_means) > 0 and len(non_nan_stds) == len(non_nan_means) :
            upper_y_limit = np.nanmax(non_nan_means + non_nan_stds) * 1.15
    ax.set_ylim(0, max(1.05, upper_y_limit if upper_y_limit > 0 else 1.05) )
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    for i, bar in enumerate(bars):
        if pd.notna(mean_values[i]) and pd.notna(std_dev_values[i]):
            yval = bar.get_height()
            text_y_position = yval + std_dev_values[i] + (ax.get_ylim()[1] * 0.01)
            plt.text(bar.get_x() + bar.get_width()/2.0, text_y_position, f"{yval:.4f}\n(±{std_dev_values[i]:.4f})", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    try:
        plt.savefig(output_filename)
        print(f"Plot successfully saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}")
    plt.close(fig)

# ----- FUNÇÃO plot_overall_comparison_metric ATUALIZADA COM RÓTULOS DE TEXTO -----
def plot_overall_comparison_metric(df_results, output_filename):
    """
    Gera e salva um gráfico de barras agrupadas comparando o desempenho médio de 
    treino e teste (Acurácia) para todos os datasets, com rótulos de valor e desvio.
    """
    if df_results is None or df_results.empty:
        print("No data to plot for overall train/test comparison.")
        return

    required_cols = ['Dataset', 'Mean_TrainAcc', 'Std_TrainAcc', 'Mean_TestAcc', 'Std_TestAcc']
    if not all(col in df_results.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_results.columns]
        print(f"Error: Missing required columns for overall train/test plot: {missing}. Skipping plot.")
        return

    df_plot = df_results.dropna(subset=required_cols).copy()
    if df_plot.empty:
        print("No datasets with complete Train/Test Accuracy data to plot for overall comparison.")
        return
    df_plot = df_plot.sort_values(by='Dataset')

    datasets = df_plot['Dataset'].tolist() # Usar .tolist() para garantir uma lista Python
    # Converte para numpy arrays para indexação numérica consistente e operações
    mean_train_acc = df_plot['Mean_TrainAcc'].to_numpy()
    std_train_acc = df_plot['Std_TrainAcc'].to_numpy()
    mean_test_acc = df_plot['Mean_TestAcc'].to_numpy()
    std_test_acc = df_plot['Std_TestAcc'].to_numpy()

    x = np.arange(len(datasets))
    width = 0.35
    fig_width = max(18, len(datasets) * 1.1) # Ajusta a largura da figura
    fig, ax = plt.subplots(figsize=(fig_width, 8.5)) # Aumenta um pouco a altura para os textos

    rects1 = ax.bar(x - width/2, mean_train_acc, width, 
                    yerr=std_train_acc, label='Train Accuracy', 
                    alpha=0.8, capsize=4, color='dodgerblue', ecolor='black')
    rects2 = ax.bar(x + width/2, mean_test_acc, width, 
                    yerr=std_test_acc, label='Test Accuracy', 
                    alpha=0.8, capsize=4, color='salmon', ecolor='black')

    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_title('Overall Comparison: Mean Train vs. Test Accuracy by Dataset', fontsize=15, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1,1)) # Move a legenda para fora

    # Ajusta limite do eixo Y para dar espaço aos textos
    max_val_with_std_train = mean_train_acc + std_train_acc
    max_val_with_std_test = mean_test_acc + std_test_acc
    # Considera um teto para o caso de std ser muito grande ou valores próximos de 1
    # para que o texto não saia do gráfico.
    # Calcula o máximo valor que o texto pode alcançar.
    max_text_y = 0
    for i in range(len(datasets)):
        max_text_y = max(max_text_y, mean_train_acc[i] + std_train_acc[i])
        max_text_y = max(max_text_y, mean_test_acc[i] + std_test_acc[i])
    
    # Dá um espaço extra (e.g., 15-20% do valor máximo ou uma fração do range do eixo y)
    # para o texto acima das barras/barras de erro.
    y_axis_top_limit = max(1.05, max_text_y * 1.20 if max_text_y > 0 else 1.05)
    ax.set_ylim(0, y_axis_top_limit)
    
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')

    # Função para adicionar rótulos de texto às barras
    def add_value_labels(bars, std_dev_values, ax_handle):
        for i, bar in enumerate(bars):
            mean_val = bar.get_height()
            std_val = std_dev_values[i]

            if pd.isna(mean_val) or pd.isna(std_val):
                continue

            label_text = f"{mean_val:.3f}\n(±{std_val:.3f})"
            
            # Posição Y para o texto: um pouco acima da barra de erro
            y_pos = mean_val + std_val + (ax_handle.get_ylim()[1] * 0.01) # Ajusta offset baseado no limite do eixo y

            # Se o texto for sair do gráfico, ajusta a posição para baixo
            if y_pos > ax_handle.get_ylim()[1] * 0.95: # Se estiver perto do topo
                 y_pos = mean_val - (ax_handle.get_ylim()[1] * 0.02) # Coloca abaixo do topo da barra
                 # Poderia também mudar a cor do texto ou alinhamento vertical para 'top' se abaixo da barra

            ax_handle.text(bar.get_x() + bar.get_width() / 2., y_pos, label_text,
                           ha='center', va='bottom', fontsize=6.5, linespacing=0.95, color='black',
                           bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.5)) # Pequeno fundo para legibilidade

    add_value_labels(rects1, std_train_acc, ax)
    add_value_labels(rects2, std_test_acc, ax)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajusta o layout para caber a legenda fora
    try:
        plt.savefig(output_filename, bbox_inches='tight') # bbox_inches='tight' pode ajudar com a legenda
        print(f"Overall Train vs. Test Accuracy comparison plot successfully saved to {output_filename}")
    except Exception as e:
        print(f"Error saving overall Train vs. Test Accuracy comparison plot: {e}")
    plt.close(fig)


# ----- FUNÇÃO PRINCIPAL (MAIN) MODIFICADA -----
def main():
    base_directory = "drift_experiment_results_big" 
    main_output_folder = "overall_performances"
    
    if not os.path.exists(main_output_folder):
        os.makedirs(main_output_folder)

    detailed_plot_dir_name = "detailed_performance_per_dataset"
    overall_plot_dir_name = "overall_comparisons"

    if not os.path.isdir(base_directory) or not os.listdir(base_directory):
        print(f"Warning: Base directory '{base_directory}' not found or is empty.")
        dummy_data = {
            'Dataset': ['SIM_Alpha', 'EXP_Beta', 'DATA_GammaVeryLongNameForTestingWrappingInPlot'],
            'Mean_TestAcc': [0.88, 0.75, 0.92], 'Std_TestAcc': [0.04, 0.07, 0.03],
            'Mean_TestF1': [0.86, 0.72, 0.90], 'Std_TestF1': [0.05, 0.08, 0.02],
            'Mean_TrainAcc': [0.96, 0.90, 0.98], 'Std_TrainAcc': [0.01, 0.03, 0.015]
        }
        results_df = pd.DataFrame(dummy_data)
        detailed_plot_dir = os.path.join(main_output_folder, f"{detailed_plot_dir_name}_dummy")
        overall_plot_dir = os.path.join(main_output_folder, f"{overall_plot_dir_name}_dummy")
    else:
        results_df = process_experiment_data(base_directory)
        detailed_plot_dir = os.path.join(main_output_folder, detailed_plot_dir_name)
        overall_plot_dir = os.path.join(main_output_folder, overall_plot_dir_name)

    if not os.path.exists(detailed_plot_dir):
        os.makedirs(detailed_plot_dir)
        # print(f"Created directory for detailed plots: {detailed_plot_dir}") # Reduz verbosidade
    if not os.path.exists(overall_plot_dir):
        os.makedirs(overall_plot_dir)
        # print(f"Created directory for overall comparison plots: {overall_plot_dir}")

    if results_df is not None and not results_df.empty:
        print("\n--- Summary of Processed Results ---")
        try:
            print(results_df.to_string())
        except Exception:
            print(results_df)
        print("------------------------------------\n")

        print("\n--- Generating detailed Train vs. Test Accuracy plots per dataset ---")
        for index, row in results_df.iterrows():
            dataset_name = str(row["Dataset"])
            safe_dataset_name = "".join([c if c.isalnum() or c in ['_', '-'] else "_" for c in dataset_name])
            plot_filename = os.path.join(detailed_plot_dir, f"ACC_TrainTest_{safe_dataset_name}.png")
            required_cols_detailed = ['Mean_TrainAcc', 'Std_TrainAcc', 'Mean_TestAcc', 'Std_TestAcc']
            if all(col in row and pd.notna(row[col]) for col in required_cols_detailed):
                plot_dataset_train_test_performance(row, dataset_name, plot_filename)
        
        print("\n--- Generating single overall Train vs. Test Accuracy comparison plot ---")
        overall_train_test_acc_filename = os.path.join(overall_plot_dir, "overall_Train_vs_Test_Accuracy_comparison.png")
        plot_overall_comparison_metric(results_df, overall_train_test_acc_filename)

    else:
        print("No results were generated. Exiting.")

if __name__ == "__main__":
    main()