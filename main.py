# main.py (Corrigido - Estrutura de previous_operator_info)
import json
import pickle
import os
import logging
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import random
import time
import itertools
import copy
import yaml
from sklearn.metrics import f1_score
from metrics import calculate_gmean_contextual
from typing import Dict, List, Tuple, Any, Union # Adicione Union
import matplotlib.pyplot as plt
import sys
from data_handling import load_full_config, get_stream_definition # Make sure get_stream_definition is imported
from utils import update_historical_reference_dataset

# Import components from other modules
from constants import RANDOM_SEED, CATEGORICAL_PREFIX
import data_handling
import ga
import fitness
import plotting
import utils
from individual import Individual
from collections import Counter

# --- Constante Padrão ---
BASE_RESULTS_DIR_DEFAULT = "experiment_results"


def make_json_serializable(obj):
    """Recursively converts sets and numpy arrays within nested structures to lists."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        # Convert set elements recursively if they might be complex
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.ndarray, np.generic)):
        # Convert numpy array/scalar to list/native Python type
        return obj.tolist()
    # Add other non-serializable types here if needed (e.g., custom objects without specific handling)
    # else:
    #    # Handle other types or raise error if needed
    #    pass
    return obj # Return basic types (int, float, str, bool, None) as is


# --- Função para Carregar Configuração ---
def load_config(config_path="config.yaml"):
    """Loads configuration parameters from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        # Use um logger específico aqui se quiser, ou deixe para o basicConfig formatar
        logging.info(f"Configuration loaded successfully from {config_path}") # Esta linha pode ser condicional ou removida se for redundante
        if config_data is None:
            logging.error(f"Config file {config_path} empty/invalid.")
            return None
        # ... (validações de seções como antes) ...
        return config_data
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config {config_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading config: {e}")
        return None

# # --- Configuração do Logging ---
# config_for_log = load_config()
# log_level_str = "INFO";
# if config_for_log and 'experiment_settings' in config_for_log: log_level_str = config_for_log['experiment_settings'].get('logging_level', 'INFO').upper()
# log_level = getattr(logging, log_level_str, logging.INFO);
# for handler in logging.root.handlers[:]: logging.root.removeHandler(handler);
# logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)-8s] %(module)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Funções Auxiliares ---
def print_individual_rules_summary(individual):
    if individual is None:
        return "Individual is None.\n"
    lines = []
    try:
        lines.append(f"  Fitness: {individual.fitness:.6f}")
        lines.append(f"  Default Class: {individual.default_class}")
        lines.append(f"  Rules ({individual.count_total_rules()} total):")

        if not hasattr(individual, 'rules') or not individual.rules:
            logging.warning("Individual object passed has no 'rules' attribute or it is empty.")
            lines.append("    (No rules attribute or empty rules dictionary)")
            return "\n".join(lines) + "\n" # Retorna o que tem até agora

        if not any(individual.rules.values()):
            logging.warning("Individual object has 'rules' attribute, but all class rule lists are empty.")
            # Continua para potentially imprimir "(No rules)" para cada classe

        # <<< DEBUG: Log das chaves de classe encontradas >>>
        logging.debug(f"Formatting rules for classes: {list(individual.rules.keys())}")

        for class_label in sorted(individual.rules.keys()):
            rule_list = individual.rules.get(class_label, [])
            lines.append(f"    Class {class_label}:")
            # <<< DEBUG: Log do número de regras para esta classe >>>
            logging.debug(f"  Class {class_label} has {len(rule_list)} rules.")

            if not rule_list:
                lines.append("      (No rules)")
            else:
                for i, rule in enumerate(rule_list):
                    rule_index_str = f"Rule {i+1} (Class {class_label})"
                    try:
                        # <<< DEBUG: Log do tipo e representação do objeto rule >>>
                        logging.debug(f"    Processing {rule_index_str}. Type: {type(rule)}, Repr: {repr(rule)}")

                        if hasattr(rule, 'to_string') and callable(rule.to_string):
                             # <<< DEBUG: Chama to_string() e loga o resultado >>>
                             rule_str = rule.to_string()
                             logging.debug(f"      {rule_index_str} -> to_string() returned: '{rule_str}' (Type: {type(rule_str)}, Len: {len(rule_str.strip()) if rule_str else 0})") # type: ignore

                             # Verifica se a string não está vazia após remover espaços
                             if rule_str and rule_str.strip(): # type: ignore
                                 lines.append(f"      {i+1}: IF {rule_str} THEN Class {class_label}")
                             else:
                                 lines.append(f"      {i+1}: (Empty or whitespace rule string returned)")
                                 logging.warning(f"      {rule_index_str} resulted in an empty/whitespace string.")
                        else:
                             lines.append(f"      {i+1}: (Error: Rule object type {type(rule)} has no to_string method)")
                             logging.warning(f"Rule object {i+1} for class {class_label} is not a valid RuleTree object.")
                    except Exception as e:
                        logging.error(f"Error converting rule {i+1} class {class_label} to string: {e}", exc_info=True) # Adiciona exc_info
                        lines.append(f"      {i+1}: (Error converting rule to string: {e})")

    except AttributeError as ae:
         logging.info(f"Attribute error formatting individual summary: {ae}. Object: {individual}")
         lines.append(f"  (Error accessing attributes: {ae})")
    except Exception as e:
         logging.info(f"Unexpected error formatting individual summary: {e}", exc_info=True)
         lines.append(f"  (Unexpected error: {e})")

    # <<< DEBUG: Log final antes de juntar >>>
    logging.debug(f"Final lines list before join (length {len(lines)}): {lines}")
    return "\n".join(lines) + "\n"

def adjust_parameter_based_on_performance(current_value, previous_best_fitness, current_best_fitness, min_val, max_val, adjustment_factor=0.1):
     if previous_best_fitness is None or current_best_fitness is None: return current_value
     if current_best_fitness > previous_best_fitness: new_value = current_value * (1 - adjustment_factor)
     elif current_best_fitness < previous_best_fitness: new_value = current_value * (1 + adjustment_factor)
     else: new_value = current_value
     return max(min_val, min(max_val, new_value))

# # Assume constants are available
try:
    from constants import RANDOM_SEED, CATEGORICAL_PREFIX
except ImportError:
    RANDOM_SEED = 42
    CATEGORICAL_PREFIX = 'cat_' # Example default
    logging.warning("Could not import constants, using defaults.")

# <<< ADDED >>> Import the periodic evaluation function (adjust path if needed)
try:
    from utils import evaluate_chunk_periodically
except ImportError:
    logging.error("Could not import 'evaluate_chunk_periodically' from utils. Please define it.")
    # Define a placeholder if needed, but it won't work correctly
    def evaluate_chunk_periodically(*args, **kwargs): return []

# <<< ADDED >>> Import plotting library
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    logging.warning("Matplotlib not found. G-mean plot generation will be skipped.")

# # --- Function to print rules (assuming it exists) ---
# def print_individual_rules_summary(individual):
#     # Placeholder: Replace with your actual function to get rule summaries
#     if individual and hasattr(individual, 'rules'):
#         summary = []
#         for class_label, rule_list in individual.rules.items():
#             summary.append(f"  Class {class_label}: {len(rule_list)} rules")
#             # for i, rule_tree in enumerate(rule_list):
#             #     summary.append(f"    Rule {i}: {rule_tree.to_string()}") # Example detail
#         return "\n".join(summary)
#     return "  No rules or invalid individual."
# Em main.py, adicione esta função perto do topo, após os imports.
# Se pandas não estiver importado, adicione: import pandas as pd
def _calculate_class_weights(target_chunk: list, all_classes: list) -> dict:
    """
    Calcula pesos para cada classe com base no inverso de sua frequência,
    normalizados para que a média seja 1.0.
    Classes raras recebem pesos > 1, classes comuns < 1.
    """
    if not target_chunk:
        return {c: 1.0 for c in all_classes} # Fallback: pesos iguais

    n_samples = len(target_chunk)
    counts = Counter(target_chunk)
    
    raw_weights = {}
    for c in all_classes:
        count = counts.get(c, 0)
        # Inverso da frequência com suavização (+1) para evitar divisão por zero
        # Se uma classe não está presente (count=0), ela recebe o peso máximo.
        raw_weights[c] = n_samples / (count + 1)
    
    sum_of_raw_weights = sum(raw_weights.values())
    num_classes = len(all_classes)
    
    normalized_weights = {}
    if sum_of_raw_weights > 0:
        # Normaliza para que a soma total dos pesos seja igual ao número de classes
        for c, rw in raw_weights.items():
            normalized_weights[c] = (rw / sum_of_raw_weights) * num_classes
    else:
        return {c: 1.0 for c in all_classes}

    logging.debug(f"Pesos de classe calculados (normalizados): { {k: round(v, 2) for k, v in normalized_weights.items()} }")
    return normalized_weights

def _save_chunk_to_csv(data: List[Dict], target: List, chunk_index: int, chunk_type: str, save_dir: str):
    """
    Salva um chunk de dados (features e target) em um arquivo CSV.

    Args:
        data (List[Dict]): Lista de dicionários com os atributos (X).
        target (List): Lista com os rótulos da classe (y).
        chunk_index (int): O índice do chunk.
        chunk_type (str): O tipo do chunk ('train' ou 'test').
        save_dir (str): O diretório onde o arquivo será salvo.
    """
    if not data:
        logging.warning(f"Tentativa de salvar chunk {chunk_index} ({chunk_type}), mas ele está vazio. Pulando.")
        return

    try:
        # Cria um diretório específico para os dados do chunk se não existir
        chunk_data_dir = os.path.join(save_dir, "chunk_data")
        os.makedirs(chunk_data_dir, exist_ok=True)

        # Constrói o nome do arquivo
        filename = f"chunk_{chunk_index}_{chunk_type}.csv"
        filepath = os.path.join(chunk_data_dir, filename)

        # Converte a lista de dicionários em um DataFrame do Pandas
        df = pd.DataFrame(data)
        # Adiciona a coluna da classe (target)
        df['class'] = target

        # Salva o DataFrame em um arquivo CSV, sem o índice
        df.to_csv(filepath, index=False)
        logging.info(f"Chunk {chunk_index} ({chunk_type}) salvo com sucesso em: {filepath}")

    except Exception as e:
        logging.error(f"Falha ao salvar o chunk {chunk_index} ({chunk_type}) em CSV: {e}", exc_info=True)

# --- Função Principal do Experimento (Modified for Periodic Evaluation) ---
def run_experiment(
    experiment_id: str, # <<< Nome genérico >>>
    run_number: int,
    config: Dict,
    base_results_dir: str,
    is_drift_simulation: bool # <<< Novo parâmetro flag >>>
    ) -> Union[Dict, None]: # Ou Dict | None para Python 3.10+
    """
    Runs a single full experiment for a given stream definition,
    tracking performance periodically and saving results.
    """
    # <<< 1. INICIALIZAR VARIÁVEIS ANTES DO LOOP DE CHUNKS >>>
    historical_reference_dataset: List[Tuple[Dict, Any]] = []
    

    # <<< MODIFIED: Use stream_name for logging and directories >>>
    run_start_time = time.time()
    # Usa experiment_id para nomear o diretório
    logger.info(f"--- Starting Experiment: ID='{experiment_id}', Run={run_number}, Mode={'Drift Simulation' if is_drift_simulation else 'Standard'} ---")
    run_seed = RANDOM_SEED + run_number; random.seed(run_seed); np.random.seed(run_seed)
    logger.info(f"Using Random Seed: {run_seed}")
    run_results_dir = os.path.join(base_results_dir, experiment_id, f"run_{run_number}")
    try:
        os.makedirs(run_results_dir, exist_ok=True)
        logging.info(f"Results will be saved in: {run_results_dir}")
    except OSError as e:
        logging.error(f"Could not create results dir {run_results_dir}: {e}")
        return None

    # --- Extract Parameters from Config ---
    exp_s = config.get('experiment_settings', {})
    data_p = config.get('data_params', {})
    ga_p = config.get('ga_params', {})
    fit_p = config.get('fitness_params', {})
    mem_p = config.get('memory_params', {})
    par_p = config.get('parallelism', {})
    # <<< ADDED: Get evaluation period >>>
    evaluation_period = exp_s.get('evaluation_period', 500)


    current_max_depth = int(ga_p.get('initial_max_depth', 4))
    current_reg_coeff = float(fit_p.get('initial_regularization_coefficient', 0.001))
    gmean_bonus_config = fit_p.get('gmean_bonus_coefficient', 0.1)
    coverage_coeff_config = fit_p.get('class_coverage_coefficient', 0.5)

    chunk_size = data_p.get('chunk_size', 500)
    num_chunks = data_p.get('num_chunks', 10)
    max_instances = data_p.get('max_instances', chunk_size * (num_chunks + 1))
    max_memory_size = mem_p.get('max_memory_size', 20)
    # Adicionar um novo parâmetro ao config.yaml, por exemplo, em 'memory_params'
    historical_reference_size = mem_p.get('historical_reference_size', 500) # Ex: 500 instâncias

    intelligent_mutation_rate = ga_p.get('intelligent_mutation_rate', 0.2)
    early_stopping_patience = int(ga_p.get('early_stopping_patience', 10)) # Add try-except if needed
    parallel_enabled = par_p.get('enabled', True)
    num_workers_config = par_p.get('num_workers', None)
    prev_best_mutant_ratio_config = float(ga_p.get('prev_best_mutant_ratio', 0.15))
    population_carry_over_rate_config = float(ga_p.get('population_carry_over_rate', 0.5))
    # <<< NOVOS PARÂMETROS DE CONFIGURAÇÃO PARA DT SEEDING E ADAPTAÇÃO (lidos de ga_p) >>>
    enable_dt_seeding_init = ga_p.get('enable_dt_seeding_on_init', False) # [config.yaml (planejado)]
    dt_seeding_ratio_init = ga_p.get('dt_seeding_ratio_on_init', 0.0) # [config.yaml (planejado)]
    dt_seeding_depths_init = ga_p.get('dt_seeding_depths_on_init', [3]) # Lista, default [3] [config.yaml (planejado)]
    dt_seeding_sample_size_init = ga_p.get('dt_seeding_sample_size_on_init', 200) # [config.yaml (planejado)]
    dt_seeding_rules_to_replace_init = ga_p.get('dt_seeding_rules_to_replace_per_class', 1) # [config.yaml (planejado)]

    # <<< CROSSOVER BALANCEADO INTELIGENTE >>>
    use_balanced_crossover = ga_p.get('use_balanced_crossover', False) # [config.yaml]

    # Parâmetro para mutantes agressivos na estratégia "diversify" de initialize_population
    recovery_aggressive_mutant_ratio = ga_p.get('recovery_aggressive_mutant_ratio', 0.3) # [config.yaml (planejado)]

    # --- Load Data using Drift Simulation ---
    stream_definition = None
    dataset_type = None

    concept_diff_data = None
    diff_file_path = config.get('drift_analysis', {}).get('heatmap_save_directory', 'results/concept_heatmaps')
    diff_file_path = os.path.join(diff_file_path, "concept_differences.json") # Assuming this is where it's saved by analyze_concept_difference.py
    if os.path.exists(diff_file_path):
        try:
            with open(diff_file_path, 'r') as f_diff:
                concept_diff_data = json.load(f_diff)
            if concept_diff_data: # Check if data was actually loaded
                logger.info(f"Concept difference data loaded successfully from {diff_file_path}")
        except Exception as e_diff:
            logger.error(f"Failed to load concept difference data from {diff_file_path}: {e_diff}")
    else:
        logger.warning(f"Concept difference data file not found at {diff_file_path}. Penalty reduction based on drift severity might be affected.")


    if is_drift_simulation:
        stream_definition = data_handling.get_stream_definition(experiment_id, config)
        if not stream_definition:
            logger.error(f"Could not retrieve stream definition for '{experiment_id}'. Stopping run.")
            return None
        dataset_type = stream_definition.get('dataset_type')
        if not dataset_type:
             logger.error(f"Stream definition for '{experiment_id}' missing 'dataset_type'. Stopping run.")
             return None
        logger.info(f"Drift Simulation Mode - Base Type: {dataset_type}")
    else:
        # No modo padrão, o experiment_id É o tipo de dataset
        dataset_type = experiment_id
        logger.info(f"Standard Mode - Dataset Type: {dataset_type}")

    try:
        # <<< MODIFIED: Pass stream_name to the updated data handling function >>>
        chunks = data_handling.generate_dataset_chunks(
            stream_or_dataset_name=experiment_id, # Passa o ID
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            max_instances=max_instances,
            run_number=run_number,
            results_dir=run_results_dir,
            config_path=config.get('config_path', 'config.yaml') # Pass config path if needed
        )
    except Exception as e:
        logger.error(f"Failed data generation for '{experiment_id}': {e}", exc_info=True)
        return None
    if not chunks or len(chunks) < 2:
        logger.error(f"Insufficient chunks for '{experiment_id}'.")
        return None

    # --- Extract Dataset Properties & Feature Types ---
    # (Using the corrected logic from previous step)
    attributes, classes = None, None
    categorical_features, numeric_features = set(), set()
    try:
        first_valid_chunk = next(((X, y) for X, y in chunks if X), None)
        if first_valid_chunk is None or not first_valid_chunk[0]:
            raise ValueError("No valid instances found in any chunk.")
        
        instance_keys = first_valid_chunk[0][0].keys()
        attributes = sorted(list(instance_keys))
        all_classes_set = set(itertools.chain.from_iterable([chunk[1] for chunk in chunks if chunk[1]]))
        try:
            classes = sorted([int(c) for c in all_classes_set])
        except (ValueError, TypeError):
            classes = sorted(list(all_classes_set))

        if not attributes or not classes:
            raise ValueError("Failed to extract attributes or classes.")

        # <<< MUDANÇA 2: Conversão de tipo em TODOS os chunks >>>
        # Este é o passo crucial. Os dados do PokerHand vêm como strings.
        logging.info("Converting data types for all chunks...")
        for i in range(len(chunks)):
            X_chunk, y_chunk = chunks[i]
            if not X_chunk: continue
            
            # Converte os rótulos y
            try:
                chunks[i] = (X_chunk, [int(label) for label in y_chunk])
            except (ValueError, TypeError):
                chunks[i] = (X_chunk, y_chunk) # Mantém como string se não for conversível

            # Converte os valores de X para float
            for instance in X_chunk:
                for key, value in instance.items():
                    try:
                        instance[key] = float(value)
                    except (ValueError, TypeError):
                        # Se não puder ser convertido para float, será tratado como categórico
                        pass
        logging.info("Data type conversion finished.")


        # <<< MUDANÇA 3: Detecção de tipo APRIMORADA após a conversão >>>
        # Analisa o primeiro chunk JÁ CONVERTIDO para inferir os tipos
        first_instance_processed = chunks[0][0][0]
        for attr in attributes:
            # Se o valor, após a tentativa de conversão, ainda for uma string, é categórico.
            if isinstance(first_instance_processed.get(attr), str):
                categorical_features.add(attr)
            else:
                numeric_features.add(attr)
        
        logging.info(f"Stream Properties: {len(attributes)} attrs ({len(numeric_features)} num, {len(categorical_features)} cat), {len(classes)} classes: {classes}")
        logging.info(f"Categorical features: {sorted(list(categorical_features))}")
        logging.info(f"Numeric features: {sorted(list(numeric_features))}")

    except Exception as e:
        logging.error(f"Failed to extract dataset properties: {e}", exc_info=True)
        return None
        # # <<< LÓGICA DE DETECÇÃO DE TIPO APRIMORADA >>>
        # # Analisa o primeiro chunk de treino para inferir os tipos dos atributos
        # train_data_chunk_df = pd.DataFrame(chunks[0][0])
        # for attr in attributes:
        #     # Regra 1: Se o nome corresponde ao prefixo, é categórico.
        #     if isinstance(attr, str) and attr.startswith(CATEGORICAL_PREFIX):
        #         categorical_features.add(attr)
        #     # Regra 2 (NOVA): Se a coluna tem 2 ou menos valores únicos, trate como categórico (binário).
        #     elif attr in train_data_chunk_df and train_data_chunk_df[attr].nunique() <= 2:
        #         categorical_features.add(attr)
        #         logger.debug(f"Attribute '{attr}' inferred as categorical (binary).")
        #     # Regra 3: Caso contrário, é numérico.
        #     else:
        #         numeric_features.add(attr)
        # # <<< FIM DA LÓGICA APRIMORADA >>>

        # logging.info(f"Stream Properties: {len(attributes)} attrs ({len(numeric_features)} num, {len(categorical_features)} cat), {len(classes)} classes: {classes}")
    #     logging.debug(f"Categorical features: {sorted(list(categorical_features))}")
    #     logging.debug(f"Numeric features: {sorted(list(numeric_features))}")

    # except Exception as e:
    #     logging.error(f"Failed to extract dataset properties: {e}", exc_info=True)
    #     return None

    # --- Initialize Tracking and State ---
    all_performance_metrics = [] # Stores OVERALL chunk metrics (for compatibility)
    # <<< ADDED: Store periodic results >>>
    all_periodic_gmean = []
    current_global_instance_count = 0 # Tracks instances processed in TEST phases
    previous_chunk_concept_id = None
    previous_full_concept_info_tuple = None
    all_used_attributes_over_time, all_rules_conditionals = [], []
    best_individuals_history, fitness_gmean_history_per_chunk, rule_info_history = [], [], []
    historical_gmean = [] # Store OVERALL test gmean for classify_performance
    previous_best_individuals_pop, best_ind_prev_chunk = None, None
    previous_best_fitness_overall = -float('inf')
    best_ever_memory = []; previous_used_features, previous_operator_info = None, None
    current_max_depth = int(ga_p.get('initial_max_depth', 4)) # Add try-except if needed
    current_reg_coeff = float(fit_p.get('initial_regularization_coefficient', 0.001)) # Add try-except if needed

    # --- Process Chunks Sequentially ---
    num_chunks_to_process = len(chunks) - 1
    for i in range(num_chunks_to_process):
        chunk_start_time = time.time()
        logging.info(f"===== Processing Chunk {i} (Train) / Chunk {i+1} (Test) =====")
        train_data_chunk, train_target_chunk = chunks[i]
        test_data_chunk, test_target_chunk = chunks[i + 1]

        #_save_chunk_to_csv(train_data_chunk, train_target_chunk, i, "train", base_results_dir)
        #_save_chunk_to_csv(test_data_chunk, test_target_chunk, i + 1, "test", base_results_dir)
        chunk_data_dir = os.path.join(run_results_dir, "chunk_data")
        os.makedirs(chunk_data_dir, exist_ok=True)

        if i == 0:
            # Na primeira iteração, salva o chunk de treino inicial (chunk 0)
            logging.info(f"Salvando chunk de treino inicial (chunk 0) para depuração.")
            _save_chunk_to_csv(train_data_chunk, train_target_chunk, i, "train", run_results_dir)
        
        # Salva o chunk de teste para a iteração atual (ex: chunk 1, depois 2, etc.)
        logging.info(f"Salvando chunk de teste (chunk {i+1}) para depuração.")
        _save_chunk_to_csv(test_data_chunk, test_target_chunk, i + 1, "test", run_results_dir)
        

        if not train_data_chunk or not train_target_chunk or len(train_data_chunk) != len(train_target_chunk):
            logging.warning(f"Skipping chunk {i}: Invalid training data.")
            continue
        if not test_data_chunk or not test_target_chunk or len(test_data_chunk) != len(test_target_chunk):
            logging.warning(f"Skipping transition {i}->{i+1}: Invalid test data (chunk {i+1}).")
            # If test data is invalid, we can't evaluate, so skip to next training chunk
            continue

        # --- Calculate Value Ranges & Category Values for Training Chunk ---
        value_ranges = {}
        category_values = {}
        # (Logic for calculating ranges/values remains the same as user provided)
        for attr in attributes:
            values_in_chunk = {inst.get(attr) for inst in train_data_chunk if inst.get(attr) is not None}
            
            if attr in numeric_features:
                numeric_vals = [v for v in values_in_chunk if isinstance(v, (int, float))]
                if numeric_vals:
                    value_ranges[attr] = (min(numeric_vals), max(numeric_vals))
                else:
                    value_ranges[attr] = (0, 0)
            
            elif attr in categorical_features:
                category_values[attr] = values_in_chunk

        logging.debug(f"Value Ranges (Chunk {i}): {value_ranges}")
        logging.debug(f"Category Values (Chunk {i}): {category_values}")
        # for attr in numeric_features:
        #     min_val, max_val = 0, 0
        #     try:
        #         values = [instance.get(attr) for instance in train_data_chunk if instance.get(attr) is not None]
        #         numeric_vals = [float(v) for v in values if isinstance(v, (int, float))]
        #         if numeric_vals: min_val, max_val = min(numeric_vals), max(numeric_vals)
        #     except (ValueError, TypeError) as e: logging.warning(f"Range calc error for '{attr}': {e}. Using (0,0).")
        #     value_ranges[attr] = (min_val, max_val)
        # for attr in categorical_features:
        #     try: category_values[attr] = {instance.get(attr) for instance in train_data_chunk if instance.get(attr) is not None}
        #     except Exception as e: logging.warning(f"Category value error for '{attr}': {e}"); category_values[attr] = set()
        # logging.debug(f"Value Ranges (Chunk {i}): {value_ranges}")
        # logging.debug(f"Category Values (Chunk {i}): {category_values}")


        # --- Adaptação Inter-Chunk ---
        # Em main.py
        fitness_params_config = config.get('fitness_params', {})
        absolute_bad_thr = fitness_params_config.get('absolute_bad_threshold_for_label', 0.3) # Pega do config, com default 0.3
        last_test_acc_for_label = historical_gmean[-1] if historical_gmean else 0.5
        logger.info(f"Classifying performance: current_acc_input={last_test_acc_for_label:.4f}, history_len={len(historical_gmean)}, history_tail={historical_gmean[-5:]}, abs_bad_thr={absolute_bad_thr:.2f}") # [main.py]
        
        performance_label = utils.classify_performance(
            last_test_acc_for_label,
            historical_gmean,
            absolute_bad_threshold=absolute_bad_thr # <<< PASSAR O NOVO ARGUMENTO
        )
        logger.info(f"Chunk {i}: Performance label='{performance_label}'.") # [main.py]

        
        # --- Determinar se as penalidades de mudança devem ser reduzidas (Lógica Aprimorada) ---
        reduce_change_penalties_flag = False
        
        # Obter informações completas do conceito para o chunk de treino ATUAL (i)
        # previous_full_concept_info_tuple é inicializado como None antes do loop de chunks
        # e atualizado no final de cada iteração do loop.
        current_full_concept_info_tuple = (None, None, 0.0) # Default
        if is_drift_simulation and stream_definition:
            current_full_concept_info_tuple = data_handling.get_concept_for_chunk(stream_definition, i) # [main.py, data_handling.py]

        current_id_a = str(current_full_concept_info_tuple[0]) if current_full_concept_info_tuple[0] is not None else None
        current_id_b = str(current_full_concept_info_tuple[1]) if current_full_concept_info_tuple[1] is not None else None
        # current_p_mix_b = current_full_concept_info_tuple[2] # Não usado diretamente na detecção abaixo, mas informativo

        # Informações do conceito do chunk de treino ANTERIOR (i-1)
        prev_id_a = None
        prev_id_b = None
        if previous_full_concept_info_tuple: # Assegura que não é a primeira iteração (onde é None)
            prev_id_a = str(previous_full_concept_info_tuple[0]) if previous_full_concept_info_tuple[0] is not None else None
            prev_id_b = str(previous_full_concept_info_tuple[1]) if previous_full_concept_info_tuple[1] is not None else None

        # Lógica de Detecção de Mudança de Transição de Conceito
        drift_transition_detected = False
        concept_pair_for_severity = None # Tupla (id1, id2) para buscar a severidade

        if is_drift_simulation and previous_full_concept_info_tuple is not None: # Só checa se não for o primeiro chunk
            # Condição 1: Mudança no conceito primário (id_a)
            if current_id_a != prev_id_a:
                drift_transition_detected = True
                concept_pair_for_severity = tuple(sorted([prev_id_a, current_id_a])) # type: ignore
                logger.info(f"Primary concept change (id_a) detected for training chunk {i}: From '{prev_id_a}' to '{current_id_a}'.") # [main.py]
            
            # Condição 2: Mudança no conceito secundário (id_b) - indica início, fim ou alteração de drift gradual
            elif current_id_b != prev_id_b: # id_a é o mesmo, mas id_b mudou
                drift_transition_detected = True
                if current_id_b is not None: # Gradual A->B está ativo ou mudando para um novo B
                    concept_pair_for_severity = tuple(sorted([current_id_a, current_id_b])) # type: ignore
                    logger.info(f"Secondary concept change (id_b) detected for training chunk {i}: From '{prev_id_b if prev_id_b else 'None'}' to '{current_id_b}' (with primary '{current_id_a}'). Gradual phase active/changing.") # [main.py]
                elif prev_id_b is not None: # Gradual A->B acabou, e id_b se tornou None (estabilizou em id_a)
                    concept_pair_for_severity = tuple(sorted([current_id_a, prev_id_b])) # type: ignore # Severidade do drift que acabou
                    logger.info(f"Secondary concept change (id_b) detected for training chunk {i}: Gradual drift involving '{prev_id_b}' ended. Stabilized on '{current_id_a}'.") # [main.py]
        
        if drift_transition_detected and concept_pair_for_severity and concept_pair_for_severity[0] and concept_pair_for_severity[1]:
            drift_severity_numeric = 0.0  # CORREÇÃO: Mantém valor numérico separado
            if concept_diff_data and dataset_type:
                # dataset_type é o tipo base (ex: "RBF", "SEA")
                dataset_specific_diffs = concept_diff_data.get(dataset_type.upper())
                if dataset_specific_diffs:
                    # pair_key_str usa os nomes dos conceitos ordenados
                    pair_key_str = f"{concept_pair_for_severity[0]}_vs_{concept_pair_for_severity[1]}"
                    drift_severity_percentage = dataset_specific_diffs.get(pair_key_str, 0.0)
                    drift_severity_numeric = drift_severity_percentage / 100.0
                    logger.debug(f"Looking up severity for {dataset_type.upper()}: {pair_key_str} (pair: {concept_pair_for_severity}), found: {drift_severity_percentage}%") # [main.py]

            fitness_params_config = config.get('fitness_params', {}) # [main.py]
            drift_penalty_reduction_threshold = fitness_params_config.get('drift_penalty_reduction_threshold', 0.10) # Ajustado para 0.10 [main.py]

            if drift_severity_numeric >= drift_penalty_reduction_threshold:
                logger.info(f"Significant concept transition (severity: {drift_severity_numeric:.2%}, threshold: {drift_penalty_reduction_threshold:.0%}) affecting train chunk {i}. Reducing change penalties.") # [main.py]
                reduce_change_penalties_flag = True
            else:
                logger.info(f"Concept transition (severity: {drift_severity_numeric:.2%}) below threshold. Penalties normal for train chunk {i}.") # [main.py]

        # Classificar severidade do drift para ajustes adicionais
        is_severe_drift = False
        if drift_transition_detected and drift_severity_numeric >= 0.25:
            is_severe_drift = True
            logger.warning(f"SEVERE DRIFT detected (severity: {drift_severity_numeric:.2%}) for chunk {i}. Activating severe drift countermeasures.")
        
        # Combinar com performance_label para ambos os modos (como antes)
        if performance_label == 'bad': # [main.py]
            if not reduce_change_penalties_flag: # Evitar log duplo
                logger.info(f"Performance label is 'bad' for chunk {i}. Reducing change penalties.") # [main.py]
            reduce_change_penalties_flag = True
        
        # <<< FIM DO BLOCO DE DETECÇÃO DE MUDANÇA E REDUÇÃO DE PENALIDADE >>>

        # <<< NEW: Determine effective max_generations for the current chunk >>>
        # Extrai os parâmetros do config (ga_p já deve estar definido no início de run_experiment)
        default_max_generations = ga_p.get('max_generations') # [config.yaml]
        recovery_max_generations = ga_p.get('max_generations_recovery') # Pode ser None [config.yaml]
        recovery_multiplier = ga_p.get('recovery_generation_multiplier') # [config.yaml]

        current_max_generations_for_ga = default_max_generations

        # Usaremos a flag reduce_change_penalties_flag como um proxy para "necessidade de adaptação intensificada"
        # Esta flag já considera:
        # 1. Mudança de conceito predefinida com severidade acima do limiar (apenas drift_simulation)
        # 2. Performance label == 'bad' (ambos os modos)

        if reduce_change_penalties_flag: # Se esta flag estiver True, indica necessidade de mais esforço
            # PRIORIDADE 1: Drift severo → dobrar gerações
            if is_severe_drift:
                current_max_generations_for_ga = int(default_max_generations * 2.0)
                logger.info(f"Recovery mode (SEVERE DRIFT): DOUBLING max_generations to {current_max_generations_for_ga} for chunk {i}")
            # PRIORIDADE 2: Configuração manual de recovery
            elif recovery_max_generations is not None:
                current_max_generations_for_ga = recovery_max_generations
                logger.info(f"Recovery mode: Using max_generations_recovery: {current_max_generations_for_ga} for chunk {i}")
            # PRIORIDADE 3: Multiplicador padrão
            else:
                current_max_generations_for_ga = int(default_max_generations * recovery_multiplier)
                logger.info(f"Recovery mode: Using recovery_multiplier. Max_generations increased to: {current_max_generations_for_ga} for chunk {i}")
        else:
            logger.info(f"Standard mode: Using default max_generations: {current_max_generations_for_ga} for chunk {i}")
        # <<< END NEW >>>

        # <<< NEW: Determine parameters for Explicit Drift Adaptation >>>
        # 1. Lógica para "Abandonar Memória" (define should_abandon_memory_now)
        mem_p_config = config.get('memory_params', {}) # [config.yaml]
        should_abandon_memory_now = False 
        
        # (consecutive_bad_performance_chunks e current_test_gmean devem ser calculados/obtidos aqui)
        # Exemplo de como obter current_test_gmean:
        current_test_gmean = historical_gmean[-1] if historical_gmean else 0.5
        consecutive_bad_performance_chunks=0
        # Exemplo de como atualizar consecutive_bad_performance_chunks:
        if 'consecutive_bad_performance_chunks' not in locals() and 'consecutive_bad_performance_chunks' not in globals():
           consecutive_bad_performance_chunks = 0
        if performance_label == 'bad':
           consecutive_bad_performance_chunks += 1
        else:
           consecutive_bad_performance_chunks = 0

        if mem_p_config.get('abandon_memory_on_severe_performance_drop', False) and reduce_change_penalties_flag: # [config.yaml]
            abandon_threshold = mem_p_config.get('performance_drop_threshold_for_memory_abandon', 0.1) # [config.yaml]
            consecutive_chunks_needed = mem_p_config.get('consecutive_bad_chunks_for_memory_abandon', 2) # [config.yaml]

            condition1 = (performance_label == 'bad' and consecutive_bad_performance_chunks >= consecutive_chunks_needed)
            condition2 = (current_test_gmean < abandon_threshold) # Supondo que current_test_gmean está disponível

            # NOTA: Drift detection foi MOVIDO para DEPOIS da linha 975 (após historical_gmean.append)
            # para garantir que compara o chunk atual com o anterior corretamente.
            # O bloco antigo aqui foi removido para evitar duplicação.

            if condition1 or condition2:
                should_abandon_memory_now = True

        if should_abandon_memory_now and best_ever_memory: # [main.py]
            # Limpa memória por performance crítica (condições 1 ou 2)
            # Nota: Drift detection (condition3) agora é tratado após linha 975
            logger.warning(f"PERFORMANCE CRITICAL (Acc: {current_test_gmean:.3f}, Bad Chunks: {consecutive_bad_performance_chunks}). Abandoning best_ever_memory.") # [main.py]
            best_ever_memory.clear()
            # previous_used_features = None # Opcional, pois reduce_change_penalties_flag deve zerar penalidades
            # previous_operator_info = None # Opcional


        # 2. Determinar Parâmetros para Adaptação Explícita ao Drift (considerando o passo anterior)
        ga_p_config = config.get('ga_params', {}) # [config.yaml]
        enable_explicit_adaptation = ga_p_config.get('enable_explicit_drift_adaptation', False) # [config.yaml]

        # Valores padrão que serão passados ao AG
        effective_mutation_config = {"override_rate": None, "override_generations": 0}
        effective_initialization_strategy = "full_random" # Valor inicial padrão
        effective_max_depth_for_ga = current_max_depth # Usa current_max_depth como base [main.py]
        effective_max_rules_for_ga = ga_p_config.get('max_rules_per_class', 5) # [config.yaml]

        if enable_explicit_adaptation and reduce_change_penalties_flag: # [main.py]
            logger.info(f"Explicit drift adaptation activated for chunk {i}.") # [main.py]

            # 2.1. Mutation Override (como você já tinha)
            override_rate = ga_p_config.get('recovery_mutation_override_rate') # [config.yaml]
            if override_rate is not None:
                effective_mutation_config["override_rate"] = override_rate
                effective_mutation_config["override_generations"] = ga_p_config.get('recovery_mutation_override_generations', 0) # [config.yaml]
                logger.info(f"  Mutation rate will be overridden to {override_rate} for {effective_mutation_config['override_generations'] or 'all'} generations.") # [main.py]

            # 2.2. Initialization Strategy (com hierarquia)
            # Primeiro, aplica a estratégia de recuperação da config, se houver
            strategy_from_config = ga_p_config.get('recovery_initialization_strategy') # [config.yaml]
            if strategy_from_config:
                effective_initialization_strategy = strategy_from_config
                # Não logue ainda, pois pode ser sobrescrito
                
            # Então, se a memória foi abandonada, esta estratégia tem prioridade MÁXIMA
            if should_abandon_memory_now:
                effective_initialization_strategy = "full_random" # Ou "total_reset"
                logger.info(f"  Memory abandoned. Overriding/setting initialization strategy to: '{effective_initialization_strategy}'.") # [main.py]
            elif strategy_from_config: # Loga a strategy_from_config apenas se não foi sobrescrita por abandono de memória
                logger.info(f"  Population initialization strategy (from explicit adapt config) set to: '{effective_initialization_strategy}'.") # [main.py]

            # 2.3. Complexity Adjustment (como você já tinha)
            depth_multiplier = ga_p_config.get('recovery_max_depth_multiplier', 1.0) # [config.yaml]
            if depth_multiplier > 1.0:
                effective_max_depth_for_ga = int(effective_max_depth_for_ga * depth_multiplier)
                logger.info(f"  Max depth temporarily increased to: {effective_max_depth_for_ga} (multiplier: {depth_multiplier}).") # [main.py]

            rules_multiplier = ga_p_config.get('recovery_max_rules_multiplier', 1.0) # [config.yaml]
            if rules_multiplier > 1.0:
                effective_max_rules_for_ga = int(effective_max_rules_for_ga * rules_multiplier)
                logger.info(f"  Max rules per class temporarily increased to: {effective_max_rules_for_ga} (multiplier: {rules_multiplier}).") # [main.py]

        # <<< END NEW >>>

        # --- Run Genetic Algorithm (Chunk N) ---
        # --- Run Genetic Algorithm (Chunk i) ---
        class_weights = _calculate_class_weights(train_target_chunk, classes)
        logger.info(f"Starting GA training on chunk {i} with MaxGen: {current_max_generations_for_ga}, MaxDepth: {effective_max_depth_for_ga}, MaxRules: {effective_max_rules_for_ga}, InitStrategy: {effective_initialization_strategy}") # [main.py]

        # CORREÇÃO: Detecta drift PREVENTIVAMENTE para adaptar treinamento do próximo chunk
        # Se performance do chunk anterior foi muito baixa, assume SEVERE drift
        drift_severity_to_pass = 'NONE'

        if 'drift_severity' in locals():
            # Usa drift detectado no chunk anterior
            drift_severity_to_pass = drift_severity
            if drift_severity_to_pass != 'NONE':
                logger.info(f"Chunk {i}: Using drift_severity='{drift_severity_to_pass}' from previous chunk for GA adaptation")

        # HEURÍSTICA PREDITIVA: Se não há histórico ainda, mas performance foi muito baixa, assume SEVERE
        if len(historical_gmean) >= 1 and historical_gmean[-1] < 0.50:
            logger.warning(f"Chunk {i}: Previous chunk had very low G-mean ({historical_gmean[-1]:.3f}) - assuming SEVERE drift preventively")
            drift_severity_to_pass = 'SEVERE'

        best_individual, final_population, history_ga_run = None, None, None
        try:
            best_individual, final_population, history_ga_run = ga.run_genetic_algorithm( # [main.py, ga.py]
                # Parâmetros do AG e do Dataset (existentes)
                attributes=attributes, value_ranges=value_ranges, classes=classes,
                class_weights=class_weights, # [main.py, ga.py]
                train_data=train_data_chunk, train_target=train_target_chunk, # [main.py, ga.py]
                categorical_features=categorical_features, category_values=category_values, # [main.py, ga.py]
                class_coverage_coefficient_ga=coverage_coeff_config,
                max_rules_per_class=effective_max_rules_for_ga,
                max_depth=effective_max_depth_for_ga,
                population_size=ga_p.get('population_size'), # [main.py, ga.py, config.yaml]
                max_generations=current_max_generations_for_ga, # Valor adaptado [main.py, ga.py]
                elitism_rate=ga_p.get('elitism_rate', 0.1), # [main.py, ga.py, config.yaml]
                gmean_bonus_coefficient_ga=gmean_bonus_config,
                initial_tournament_size=ga_p.get('initial_tournament_size', 2), # [main.py, ga.py, config.yaml]
                final_tournament_size=ga_p.get('final_tournament_size', 5), # [main.py, ga.py, config.yaml]
                regularization_coefficient=current_reg_coeff,
                feature_penalty_coefficient=fit_p.get('feature_penalty_coefficient', 0.0),
                operator_penalty_coefficient=fit_p.get('operator_penalty_coefficient', 0.0),
                threshold_penalty_coefficient=fit_p.get('threshold_penalty_coefficient', 0.0), # [main.py, ga.py, config.yaml]
                intelligent_mutation_rate = ga_p.get('intelligent_mutation_rate', 0.2),
                operator_change_coefficient=fit_p.get('operator_change_coefficient', 0.0), # [main.py, ga.py, config.yaml]
                gamma=fit_p.get('gamma', 0.0), # [main.py, ga.py, config.yaml]
                previous_rules_pop=previous_best_individuals_pop, 
                best_ever_memory=best_ever_memory, # Pode ter sido limpa [main.py, ga.py]
                performance_label=performance_label, # [main.py, ga.py]
                previous_used_features=previous_used_features, # [main.py, ga.py]
                previous_operator_info=previous_operator_info, # [main.py, ga.py]
                best_individual_from_previous_chunk=best_ind_prev_chunk,
                early_stopping_patience=early_stopping_patience,
                prev_best_mutant_ratio=ga_p.get('prev_best_mutant_ratio', 0.15), # Para estratégia 'default' [main.py, ga.py, config.yaml]
                parallel_enabled=parallel_enabled,        
                num_workers=num_workers_config,
                reduce_change_penalties_flag=reduce_change_penalties_flag,   
                mutation_override_config_ga=effective_mutation_config, # [main.py, ga.py]
                initialization_strategy=effective_initialization_strategy,
                enable_dt_seeding_on_init_config_ga=enable_dt_seeding_init,                  # [main.py, ga.py]                             
                dt_seeding_ratio_on_init_config_ga=dt_seeding_ratio_init, # [main.py, ga.py]
                dt_seeding_depths_on_init_config_ga=dt_seeding_depths_init, # [main.py, ga.py]
                dt_seeding_sample_size_on_init_config_ga=dt_seeding_sample_size_init, # [main.py, ga.py]
                dt_seeding_rules_to_replace_config_ga=dt_seeding_rules_to_replace_init, # [main.py, ga.py]
                recovery_aggressive_mutant_ratio_config_ga=recovery_aggressive_mutant_ratio,
                historical_reference_dataset=historical_reference_dataset,
                dt_rule_injection_ratio_config_ga=ga_p.get('dt_rule_injection_ratio', 1.0),  # Seeding Probabilístico
                enable_adaptive_seeding_config_ga=ga_p.get('enable_adaptive_seeding', False),  # Seeding Adaptativo
                hc_enable_adaptive=ga_p.get('hc_enable_adaptive', False),  # Hill Climbing adaptativo (legado)
                hc_gmean_threshold=ga_p.get('hc_gmean_threshold', 0.90),  # Threshold de G-mean para desabilitar HC (legado)
                hc_hierarchical_enabled=ga_p.get('hc_hierarchical_enabled', True),  # Hill Climbing Hierárquico v2.0
                stagnation_threshold=ga_p.get('stagnation_threshold', 15),  # Threshold de estagnação para ativar Hill Climbing
                use_balanced_crossover=use_balanced_crossover,  # Crossover Balanceado Inteligente
                drift_severity=drift_severity_to_pass  # Severidade do drift detectado (SEVERE, MODERATE, MILD, STABLE, NONE)
            )
        except Exception as ga_e:
            logger.error(f"GA execution error on chunk {i}: {ga_e}", exc_info=True) # [main.py]
            # (Seu tratamento de erro existente)
            return None 

        if best_individual:
            best_individual.creation_chunk_index = i # 'i' é o índice do chunk de treino atual

            # <<< NEW: Active Memory Pruning (Optional, based on config) >>>
            mem_p_config = config.get('memory_params', {})
            if mem_p_config.get('enable_active_memory_pruning', False) and \
               (is_drift_simulation and reduce_change_penalties_flag): # reduce_change_penalties_flag indica drift/baixa perf.

                if best_ever_memory: # Só faz sentido se a memória não estiver vazia
                    age_threshold = mem_p_config.get('memory_max_age_chunks', 10)
                    fitness_percentile_for_removal = mem_p_config.get('memory_fitness_threshold_percentile_for_old_removal', 0.25)
                    min_retain_count = mem_p_config.get('memory_min_retain_count_during_pruning', max(1, int(max_memory_size * 0.25))) # Ex: manter pelo menos 25%

                    fitness_values_in_memory = [ind.fitness for ind in best_ever_memory]
                    # Calcula o limiar de fitness: indivíduos velhos com fitness abaixo disso são candidatos
                    # Cuidado com percentil em listas pequenas; pode ser melhor usar média/desvio padrão ou um valor absoluto
                    if len(fitness_values_in_memory) > 1: # Evitar erro com np.percentile em lista < 2
                        removal_fitness_cutoff = np.percentile(fitness_values_in_memory, fitness_percentile_for_removal * 100)
                    else: # Se só tem 1 item, não há o que comparar em termos de percentil
                        removal_fitness_cutoff = -float('inf') # Não remove baseado em fitness se só tem 1 item

                    survivors = []
                    num_actively_removed = 0
                    # Ordenar por idade (mais velho primeiro) para dar preferência de remoção aos mais velhos
                    # entre aqueles que satisfazem os critérios.
                    sorted_for_pruning = sorted(best_ever_memory, key=lambda ind: ind.creation_chunk_index)

                    for ind in sorted_for_pruning:
                        is_old = (i - ind.creation_chunk_index) > age_threshold
                        
                        # Condição para remoção: velho E fitness abaixo do corte E não reduzir demais a memória
                        if is_old and ind.fitness < removal_fitness_cutoff and (len(best_ever_memory) - num_actively_removed > min_retain_count):
                            logger.info(f"Active Memory Pruning: Removing old individual (Chunk {ind.creation_chunk_index}, Fit: {ind.fitness:.4f}) due to age and fitness ({ind.fitness:.4f} < {removal_fitness_cutoff:.4f}).")
                            num_actively_removed += 1
                        else:
                            survivors.append(ind)
                    
                    if num_actively_removed > 0:
                        logger.info(f"Actively pruned {num_actively_removed} individuals from memory.")
                        best_ever_memory = survivors
            # <<< END NEW: Active Memory Pruning >>>

            # Adicionar o melhor indivíduo atual à memória (fazendo uma cópia)
            best_ever_memory.append(copy.deepcopy(best_individual))

            # Ordenar a memória:
            # 1. Primariamente por fitness (descendente - melhor fitness primeiro)
            # 2. Secundariamente por creation_chunk_index (descendente - mais novo primeiro em caso de empate de fitness)
            best_ever_memory.sort(key=lambda ind: (ind.fitness, ind.creation_chunk_index), reverse=True)

            # Truncar a memória para o tamanho máximo
            if len(best_ever_memory) > max_memory_size: # max_memory_size vem do config.memory_params
                best_ever_memory = best_ever_memory[:max_memory_size]                 
        else:
            logging.error(f"GA did not return a best individual for chunk {i}. Stopping run.")
            return None # Stop if GA provides no result

        # Salva o melhor indivíduo encontrado após treinar no chunk `i`
        if best_individual:
            individual_filename = f"best_individual_trained_on_chunk_{i}.pkl"
            individual_save_path = os.path.join(chunk_data_dir, individual_filename)
            try:
                with open(individual_save_path, 'wb') as f_pickle:
                    pickle.dump(best_individual, f_pickle)
                logging.info(f"Melhor indivíduo do chunk {i} salvo em: {individual_save_path}")
            except Exception as e_pickle:
                logging.error(f"Falha ao salvar o indivíduo do chunk {i} no arquivo pickle: {e_pickle}")

        # --- Fim do Bloco de Salvamento para Depuração ---

        logging.info(f"Finished GA training on chunk {i}.")
        fitness_gmean_history_per_chunk.append(history_ga_run or {}) # Store history even if empty
        current_best_fitness_overall = history_ga_run.get('best_fitness', [-float('inf')])[-1] if history_ga_run and history_ga_run.get('best_fitness') else -float('inf')
        train_gmean = history_ga_run.get('best_gmean', [0.0])[-1] if history_ga_run and history_ga_run.get('best_gmean') else 0.0 # Get final train accuracy

        # --- Evaluate on Test Chunk (N+1) ---
        logging.info(f"Starting evaluation on test chunk {i+1}...")

        # <<< MODIFIED: Periodic Evaluation >>>
        chunk_periodic_results = evaluate_chunk_periodically(
            ruleset=best_individual,
            test_X=test_data_chunk,
            test_y=test_target_chunk,
            evaluation_period=evaluation_period,
            global_instance_offset=current_global_instance_count
        )
        all_periodic_gmean.extend(chunk_periodic_results)
        # <<< END MODIFIED >>>

        # <<< ADDED: Calculate OVERALL test metrics for compatibility >>>
        test_gmean, test_f1 = 0.0, 0.0
        if test_data_chunk: # Avoid error on empty test chunk
            try:
                # --- LINHA INCORRETA ABAIXO ---
                # test_predictions = [utils.predict_instance(best_individual, inst) for inst in test_data_chunk] # Assumes predict_instance exists in utils
                # --- CORREÇÃO ---
                test_predictions = [best_individual._predict(inst) for inst in test_data_chunk] # Usa o método do objeto Individual
                # --- FIM CORREÇÃO ---
                #test_gmean = gmean_score(test_target_chunk, test_predictions)

                test_gmean = calculate_gmean_contextual(test_target_chunk, test_predictions, classes)

                test_f1 = f1_score(test_target_chunk, test_predictions, average='weighted', zero_division=0)
                #test_gmean=test_f1
            except AttributeError:
                 logging.error(f"Object 'best_individual' does not have a '_predict' method. Cannot calculate overall metrics.")
            except Exception as eval_e:
                logging.error(f"Error calculating overall metrics on test chunk {i+1}: {eval_e}", exc_info=True)
        # <<< END ADDED >>>

        # Store overall metrics (as before)
        perf_metrics = {'chunk': i, 'train_gmean': train_gmean, 'test_gmean': test_gmean, 'test_f1': test_f1}
        all_performance_metrics.append(perf_metrics)
        historical_gmean.append(test_gmean) # Use overall test accuracy here

        logging.info(f"Chunk {i} Results: TrainGmean{train_gmean:.4f}, TestGmean={test_gmean:.4f}, TestF1={test_f1:.4f}")
        logging.info(f"Chunk {i}: Best Individual Fitness={best_individual.fitness:.4f}")

        # DRIFT DETECTION: Classifica severidade do drift APÓS adicionar test_gmean atual
        drift_severity = 'NONE'  # Inicializa variável de severidade
        if len(historical_gmean) >= 2:
            current_gmean = historical_gmean[-1]
            previous_gmean = historical_gmean[-2]
            performance_drop = previous_gmean - current_gmean

            # Classificação de severidade do drift (4 níveis)
            if performance_drop > 0.20:  # Queda > 20%
                drift_severity = 'SEVERE'
                logger.warning(f"🔴 SEVERE DRIFT detected: {previous_gmean:.3f} → {current_gmean:.3f} (drop: {performance_drop:.1%})")
                # CORREÇÃO: Mantém top 10% da memory ao invés de limpar tudo
                if best_ever_memory:
                    original_size = len(best_ever_memory)
                    keep_size = max(1, original_size // 10)  # Mantém pelo menos 1, no máximo 10%
                    best_ever_memory = sorted(best_ever_memory, key=lambda ind: ind.fitness, reverse=True)[:keep_size]
                    logger.info(f"   → Memory REDUCED to top {keep_size} individuals (was {original_size}) - kept top 10%")

            elif performance_drop > 0.10:  # Queda > 10%
                drift_severity = 'MODERATE'
                logger.warning(f"🟡 MODERATE DRIFT detected: {previous_gmean:.3f} → {current_gmean:.3f} (drop: {performance_drop:.1%})")
                # Reduz best_ever_memory pela metade (mantém apenas os melhores)
                if best_ever_memory:
                    best_ever_memory = sorted(best_ever_memory, key=lambda ind: ind.fitness, reverse=True)[:len(best_ever_memory)//2]
                    logger.info(f"   → Memory reduced to top {len(best_ever_memory)} individuals")

            elif performance_drop > 0.05:  # Queda > 5%
                drift_severity = 'MILD'
                logger.info(f"🟢 MILD DRIFT detected: {previous_gmean:.3f} → {current_gmean:.3f} (drop: {performance_drop:.1%})")

            else:  # Estável ou melhorou
                drift_severity = 'STABLE'
                if performance_drop < 0:  # Melhorou
                    logger.info(f"✓ PERFORMANCE IMPROVED: {previous_gmean:.3f} → {current_gmean:.3f} (gain: {abs(performance_drop):.1%})")
                else:
                    logger.info(f"✓ STABLE: {previous_gmean:.3f} → {current_gmean:.3f} (drop: {performance_drop:.1%})")
        logging.debug(f"Best Individual Rules for Chunk {i}:\n{print_individual_rules_summary(best_individual)}")

        # --- Store Metrics / Rule Info / Update State (mostly unchanged) ---
        rules_cond_metrics = {'chunk': i, 'total_rules': best_individual.count_total_rules(), 'total_conditionals': best_individual.count_total_nodes()}
        all_rules_conditionals.append(rules_cond_metrics)
        current_used_features = best_individual.get_used_attributes() # Assumes method exists
        all_used_attributes_over_time.append(current_used_features)

        rule_info = {}
        try:
            # Assumes method exists and returns dict with specified keys
            l_ops, c_ops, num_thresh, cat_vals = best_individual._collect_ops_and_thresholds()
            rule_info = {"logical_ops": l_ops, "comparison_ops": c_ops, "numeric_thresholds": num_thresh, "categorical_values_used": cat_vals, "features": list(current_used_features)}
        except Exception as rule_info_e: logging.error(f"Error collecting detailed rule info: {rule_info_e}", exc_info=True)
        rule_info_history.append(rule_info)

        best_individuals_history.append(copy.deepcopy(best_individual))
        best_ever_memory.append(copy.deepcopy(best_individual))
        best_ever_memory.sort(key=lambda ind: ind.fitness, reverse=True)
        best_ever_memory = best_ever_memory[:max_memory_size]

        if best_individuals_history:
            last_saved_ind = best_individuals_history[-1]
            logging.debug(f"Checking last saved individual (Chunk {i}): Has rules? {'rules' in last_saved_ind.__dict__ if last_saved_ind else 'N/A'}")
            if hasattr(last_saved_ind, 'rules') and last_saved_ind.rules:
                first_class = list(last_saved_ind.rules.keys())[0]
                if last_saved_ind.rules[first_class]:
                      first_rule = last_saved_ind.rules[first_class][0]
                      try:
                          logging.debug(f"  First rule string from saved copy: '{first_rule.to_string()}'")
                      except Exception as e:
                          logging.debug(f"  Error calling to_string on first rule of saved copy: {e}")
                else:
                      logging.debug(f"  Rule list for first class '{first_class}' is empty in saved copy.")
            else:
                logging.debug(f"  No rules found in saved copy.")

        # Update state for next chunk
        previous_best_fitness_overall = current_best_fitness_overall
        best_ind_prev_chunk = copy.deepcopy(best_individual)

        # Ajuste dinâmico da taxa de herança baseado em severidade do drift
        base_carry_over = population_carry_over_rate_config
        if 'drift_severity' in locals():
            if drift_severity == 'SEVERE':
                # CORREÇÃO: Drift severo: Herança mínima 20% ao invés de 0% (reset menos agressivo)
                adjusted_carry_over = 0.20
                logger.warning(f"   → Inheritance REDUCED to {adjusted_carry_over:.0%} due to SEVERE drift (was {base_carry_over:.0%})")
            elif drift_severity == 'MODERATE':
                # Drift moderado: REDUZ herança para 25%
                adjusted_carry_over = 0.25
                logger.info(f"   → Inheritance REDUCED to {adjusted_carry_over:.0%} due to MODERATE drift (was {base_carry_over:.0%})")
            elif drift_severity == 'MILD':
                # Drift leve: MANTÉM herança padrão
                adjusted_carry_over = base_carry_over
                logger.info(f"   → Inheritance MAINTAINED at {adjusted_carry_over:.0%} (MILD drift)")
            else:  # STABLE ou NONE
                # Estável: AUMENTA herança para 65%
                adjusted_carry_over = 0.65
                logger.info(f"   → Inheritance INCREASED to {adjusted_carry_over:.0%} (stable performance)")
        else:
            adjusted_carry_over = base_carry_over

        num_to_keep = int(adjusted_carry_over * len(final_population)) if final_population else 0
        if num_to_keep > 0:
            sorted_final_pop = sorted(final_population, key=lambda ind: ind.fitness, reverse=True)
            previous_best_individuals_pop = [copy.deepcopy(ind) for ind in sorted_final_pop[:num_to_keep]]
        else: previous_best_individuals_pop = None
        previous_used_features = current_used_features
        if rule_info: # Use collected info
            previous_operator_info = {
                 "logical_ops": set(rule_info.get("logical_ops", [])),
                 "comparison_ops": set(rule_info.get("comparison_ops", [])),
                 "numeric_thresholds": rule_info.get("numeric_thresholds", []),
                 "categorical_values": rule_info.get("categorical_values_used", [])
             }
        else: previous_operator_info = None

        # <<< MODIFIED: Update global instance count AFTER evaluation >>>
        current_global_instance_count += len(test_data_chunk)

        chunk_end_time = time.time()
        logging.info(f"Chunk {i} processed in {chunk_end_time - chunk_start_time:.2f} seconds. Total instances tested: {current_global_instance_count}")

            # <<< ADD AT THE END OF THE CHUNK PROCESSING LOOP (after all uses of current_chunk_concept_id for this iteration 'i') >>>
        # <<< ATUALIZAR previous_full_concept_info_tuple PARA A PRÓXIMA ITERAÇÃO >>>
        if is_drift_simulation and stream_definition: # Só é relevante para modo drift
            previous_full_concept_info_tuple = current_full_concept_info_tuple # Armazena o tuple completo (id_a, id_b, p_mix_b)
        # <<< FIM DA ATUALIZAÇÃO >>>

        # --- End of Chunk Loop ---

    # --- Post-Experiment ---
    logging.info(f"Finished processing {num_chunks_to_process} chunk transitions for run {run_number}.")

# --- Plotting (Existing Plots + New Periodic Accuracy Plot) ---

    # Plot GA Evolution (Usa a função renomeada em plotting.py)
    if fitness_gmean_history_per_chunk: # Verifica se há dados
        try:
            ga_evol_save_path = os.path.join(run_results_dir, f"GA_Evolution_{stream_name}_Run{run_number}.png")
            # Esta função agora plota a evolução para CADA chunk em subplots separados
            num_histories = len(fitness_gmean_history_per_chunk)
            if num_histories > 0:
                 # Cria a figura e os eixos ANTES de chamar o plot para cada chunk
                 fig_evol, axes_evol = plt.subplots(nrows=num_histories, ncols=2, # type: ignore
                                                    figsize=(14, num_histories * 4), squeeze=False)
                 fig_evol.suptitle(f'GA Evolution per Chunk - {stream_name} (Run {run_number})', fontsize=16)

                 for idx, history_data in enumerate(fitness_gmean_history_per_chunk):
                      # Passa os eixos corretos para a função plotar (fitness no ax 0, accuracy no 1)
                      # A função plot_ga_evolution precisa ser ajustada para aceitar uma tupla de eixos talvez?
                      # OU: mais simples, fazemos o plot aqui mesmo como no exemplo original.
                      # Vamos manter a chamada à função, assumindo que ela gera a figura completa por chunk.
                      # NOTA: A versão anterior de plot_ga_evolution plotava uma figura por chamada.
                      # Se você quer uma figura com subplots, a lógica de plotagem precisa ser ajustada
                      # ou a chamada precisa ser feita de forma diferente.
                      # Vamos manter a chamada simples por enquanto, gerando um plot por chunk:
                      chunk_ga_evol_path = os.path.join(run_results_dir, f"GA_Evolution_Chunk{idx}_{stream_name}_Run{run_number}.png")
                      plotting.plot_ga_evolution(
                          history=history_data,
                          chunk_index=idx,
                          dataset_name=stream_name, # Passa stream_name como identificador
                          run_number=run_number,
                          save_path=chunk_ga_evol_path # Salva um arquivo por chunk
                      )
                 # plt.close(fig_evol) # Fecha a figura se foi criada aqui
                 logging.info(f"GA evolution plots saved individually per chunk.")

        except Exception as e:
            logger.error(f"Failed to generate GA evolution plots: {e}", exc_info=True) # Usa logger
    else:
         logger.warning("No GA history data to plot.") # Usa logger


    # Plot Rule Evolution (Substitui chamada genérica por chamadas específicas)
    if rule_info_history: # Verifica se há dados
        try:
            # 1. Plot Heatmap de Componentes
            rule_heatmap_path = os.path.join(run_results_dir, f"RuleComponents_Heatmap_{stream_name}_Run{run_number}.png")
            logger.info(f"Generating rule components heatmap for {stream_name}...") # Usa logger
            plotting.plot_rule_changes(
                rule_change_history=rule_info_history, # Passa os dados corretos
                dataset_name=stream_name,
                run_number=run_number,
                save_path=rule_heatmap_path
            )
        except Exception as e:
            logger.error(f"Failed to generate rule components heatmap plot: {e}", exc_info=True) # Usa logger

        try:
            # 2. Plot Radar de Componentes
            rule_radar_path = os.path.join(run_results_dir, f"RuleComponents_Radar_{stream_name}_Run{run_number}.png")
            logger.info(f"Generating rule components radar plot for {stream_name}...") # Usa logger
            plotting.plot_rule_info_radar(
                rule_info_history=rule_info_history,
                dataset_name=stream_name,
                run_number=run_number,
                save_path=rule_radar_path
            )
        except Exception as e:
            logger.error(f"Failed to generate rule components radar plot: {e}", exc_info=True) # Usa logger

        # 3. Plot Polar de Mudanças (Opcional, pode ser muito detalhado)
        # try:
        #    rule_polar_path = os.path.join(run_results_dir, f"RuleChanges_Polar_{stream_name}_Run{run_number}.png")
        #    logger.info(f"Generating rule changes polar plot for {stream_name}...")
        #    plotting.plot_rule_feature_changes_arrow_polar(
        #        rule_info_history=rule_info_history,
        #        dataset_name=stream_name,
        #        run_number=run_number,
        #        save_path=rule_polar_path
        #    )
        # except Exception as e:
        #    logger.error(f"Failed to generate rule changes polar plot: {e}", exc_info=True)

    else:
        logger.warning("No rule info history data to plot.") # Usa logger


    # Plot Periodic G-mean (Como adicionado anteriormente)
    if plt and all_periodic_gmean:
        try:
            global_counts, gmean = zip(*all_periodic_gmean)
            plt.figure(figsize=(12, 6))
            plt.plot(global_counts, [acc * 100 for acc in gmean], marker='.', linestyle='-', markersize=3, label=f'Run {run_number}')
            plt.xlabel("Total Instances Processed (Test Phases)")
            plt.ylabel("Periodic G-mean (%)")
            plt.title(f"Periodic G-mean Over Time: {stream_name} (Run {run_number})")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.ylim(0, 105)
            plt.legend()
            # <<< CORRIGIDO: Usa run_results_dir e stream_name >>>
            plot_filename = os.path.join(run_results_dir, f"G-meanPlot_Periodic_{stream_name}_Run{run_number}.png")
            # os.makedirs(run_results_dir, exist_ok=True) # _save_plot já faz isso
            plotting._save_plot(plt.gcf(), plot_filename) # Usa helper _save_plot
            # plt.savefig(plot_filename, bbox_inches='tight')
            logger.info(f"Periodic g-mean plot saved to {plot_filename}") # Usa logger
            # plt.close() # _save_plot já fecha
        except ImportError: pass
        except Exception as e:
            logger.error(f"Error generating periodic g-mean plot: {e}", exc_info=True) # Usa logger
    elif not all_periodic_gmean:
         logger.warning("No periodic g-mean data collected, skipping plot.") # Usa logger
    # <<< END ADDED >>>


    
    # --- Save Rule History (Corrected) ---
    rules_file_path = os.path.join(run_results_dir, f"RulesHistory_{stream_name}_Run{run_number}.txt")
    try:
        with open(rules_file_path, "w", encoding='utf-8') as f:
            # Write header information
            f.write(f"Rule History for Stream: {stream_name}, Run: {run_number}\n")
            # Include chunk size or other relevant config if available
            # f.write(f"Chunk Size: {chunk_size}\n") # Example if chunk_size is available here
            f.write("="*40 + "\n")

            # Iterate through the results collected for each chunk transition
            for chunk_idx in range(len(all_performance_metrics)):
                # Get the best individual saved for this chunk's training phase
                # Check bounds to prevent IndexError if lists have different lengths
                best_ind = best_individuals_history[chunk_idx] if chunk_idx < len(best_individuals_history) else None
                # Get the performance metrics calculated after this training
                perf = all_performance_metrics[chunk_idx]

                # Write chunk header and performance summary
                f.write(f"\n--- Chunk {chunk_idx} (Trained) ---")
                # Performance metrics are from testing on chunk_idx + 1
                f.write(f" Test Perf (Chunk {chunk_idx+1}): TestGmean={perf.get('test_gmean', float('nan')):.4f}, TestF1={perf.get('test_f1', float('nan')):.4f}\n")
                # Train performance is from the end of the GA run on chunk_idx
                f.write(f" Train Perf (Chunk {chunk_idx}): TrainGmean={perf.get('train_gmean', float('nan')):.4f}\n") # train_f1 might not be stored in perf, check this
                f.write("---\n")

                # --- Call the function to format and get the rule details ---
                # This function now includes logging for debugging if needed
                rules_summary_str = print_individual_rules_summary(best_ind)
                # --- Write the formatted rules to the file ---
                f.write(rules_summary_str) # This contains Fitness, Default Class, and the rules

                f.write("="*40 + "\n") # Separator between chunks

        logger.info(f"Rule history saved to {rules_file_path}") # Use logger

    except NameError as ne:
         # Catch specific errors if variables used in the loop aren't defined
         logger.error(f"Failed to save rule history due to missing variable: {ne}", exc_info=True)
    except Exception as e:
         logger.error(f"Failed to save rule history: {e}", exc_info=True) # Use logger

    # --- Run Summary ---
    avg_train_acc = np.nanmean([m.get('train_gmean', np.nan) for m in all_performance_metrics]) if all_performance_metrics else 0
    avg_test_acc = np.nanmean([m.get('test_gmean', np.nan) for m in all_performance_metrics]) if all_performance_metrics else 0
    avg_test_f1 = np.nanmean([m.get('test_f1', np.nan) for m in all_performance_metrics]) if all_performance_metrics else 0
    run_end_time = time.time()
    logging.info(f"--- Run {run_number} Summary (Stream: {stream_name}) ---")
    logging.info(f"Avg Train Gmean (Final Gen): {avg_train_acc:.4f}")
    logging.info(f"Avg Test Gmean (Overall Chunk): {avg_test_acc:.4f}")
    logging.info(f"Avg Test F1 (Overall Chunk): {avg_test_f1:.4f}")
    logging.info(f"Run {run_number} duration: {run_end_time - run_start_time:.2f} seconds")
    logging.info(f"--- End Experiment Run: Stream='{stream_name}', Run={run_number} ---")


    # --- <<< ADDED: Save Run Results to Files >>> ---
    logger.info(f"Saving results for Stream '{stream_name}', Run {run_number} to {run_results_dir}")
    results_saved_successfully = True # Flag to track overall success

    try:
        # 1. Save Run Configuration/Parameters (Example - customize as needed)
        run_config_info = {
            'experiment_id': experiment_id, # Nome do stream ou dataset base
            'run_mode': 'Drift Simulation' if is_drift_simulation else 'Standard',
            'run_number': run_number,
            'seed': run_seed,
            'dataset_type': dataset_type, # Tipo base do dataset
            'chunk_size': chunk_size,
            'num_chunks_processed': num_chunks_to_process,
            'ga_params': ga_p,
            'fitness_params': fit_p,
            'memory_params': mem_p,
            # <<< MODIFICADO: Salva stream_definition SÓ se for modo drift >>>
            'stream_definition': stream_definition if is_drift_simulation else None,
            'attributes': attributes
        }
        config_save_path = os.path.join(run_results_dir, "run_config.json")
        with open(config_save_path, 'w') as f_cfg:
            json.dump(run_config_info, f_cfg, indent=2)
        logger.debug(f"Run config saved to {config_save_path}")

        # 2. Save Overall Chunk Metrics (JSON)
        metrics_path = os.path.join(run_results_dir, f"chunk_metrics.json")
        with open(metrics_path, 'w') as f_metrics:
            json.dump(make_json_serializable(all_performance_metrics), f_metrics, indent=2) # Use helper
        logger.debug(f"Overall chunk metrics saved to {metrics_path}")

        # 3. Save Periodic Gmean (JSON)
        periodic_path = os.path.join(run_results_dir, f"periodic_gmean.json")
        with open(periodic_path, 'w') as f_periodic:
            json.dump(make_json_serializable(all_periodic_gmean), f_periodic, indent=2) # Use helper
        logger.debug(f"Periodic gmean data saved to {periodic_path}")

        # 4. Save GA History per Chunk (JSON)
        ga_history_path = os.path.join(run_results_dir, f"ga_history_per_chunk.json")
        with open(ga_history_path, 'w') as f_ga:
            serializable_ga_history = make_json_serializable(fitness_gmean_history_per_chunk)
            json.dump(serializable_ga_history, f_ga, indent=2)
        logger.debug(f"GA history saved to {ga_history_path}")

        # 5. Save Detailed Rule Info (JSON)
        rule_info_path = os.path.join(run_results_dir, f"rule_details_per_chunk.json")
        with open(rule_info_path, 'w') as f_rule_info:
             serializable_rule_info = make_json_serializable(rule_info_history)
             json.dump(serializable_rule_info, f_rule_info, indent=2)
        logger.debug(f"Detailed rule info saved to {rule_info_path}")

        # 6. Save Attribute Usage (JSON)
        attr_usage_path = os.path.join(run_results_dir, f"attribute_usage_per_chunk.json")
        serializable_attr_usage = make_json_serializable(all_used_attributes_over_time)
        with open(attr_usage_path, 'w') as f_attr:
            json.dump(serializable_attr_usage, f_attr, indent=2)
        logger.debug(f"Attribute usage saved to {attr_usage_path}")

        # 7. Save Best Individuals History (Pickle)
        # Check if the list actually contains individuals before saving
        if best_individuals_history and best_individuals_history[0] is not None:
            best_ind_path = os.path.join(run_results_dir, f"best_individuals.pkl")
            try:
                with open(best_ind_path, 'wb') as f_pickle: # Use 'wb' for binary write
                    pickle.dump(best_individuals_history, f_pickle)
                logger.debug(f"Best individuals history saved (pickle) to {best_ind_path}")
            except (pickle.PicklingError, TypeError, AttributeError) as e_pickle:
                # Catch specific errors related to pickling complex objects
                logger.error(f"Failed to save best individuals using pickle: {e_pickle}. Object structure might not be pickleable.")
                results_saved_successfully = False # Mark saving as partially failed
            except Exception as e_pickle_other: # Catch other potential file errors
                 logger.error(f"Unexpected error saving best individuals pickle: {e_pickle_other}")
                 results_saved_successfully = False
        else:
             logger.warning("Best individuals history is empty or invalid; skipping pickle save.")


        # 8. Rule History Text File (Logic is already present earlier in the function)
        # Just confirm it was saved successfully if possible, or rely on its own try/except
        rules_txt_path = os.path.join(run_results_dir, f"RulesHistory_{stream_name}_Run{run_number}.txt")
        if os.path.exists(rules_txt_path):
             logger.debug(f"Rule history text file already saved to {rules_txt_path}")
        else:
             logger.warning(f"Rule history text file {rules_txt_path} was not found. Saving might have failed earlier.")
             results_saved_successfully = False


    except Exception as save_e:
        logger.error(f"CRITICAL ERROR during results saving for Stream '{stream_name}', Run {run_number}: {save_e}", exc_info=True)
        results_saved_successfully = False # Mark as failed

    if results_saved_successfully:
         logger.info(f"All available results saved successfully for Stream '{stream_name}', Run {run_number}.")
    else:
         logger.warning(f"Some results might have failed to save for Stream '{stream_name}', Run {run_number}. Check logs.")

    # --- <<< END ADDED BLOCK >>> ---

    # Return a summary dictionary (can exclude large raw data lists now)
    run_summary_dict = {
        'experiment_id': experiment_id,
        'run_number': run_number,
        'dataset_type': dataset_type, # Include type for aggregation
        'attributes': attributes,
        # Include aggregated metrics
        'average_train_gmean': avg_train_acc,
        'average_test_gmean': avg_test_acc,
        'average_test_f1': avg_test_f1,
        # Include std devs if calculated by compute_average_metrics_over_runs
        'train_gmean_std': np.nanstd([m.get('train_gmean', np.nan) for m in all_performance_metrics]) if all_performance_metrics else 0,
        'test_gmean_std': np.nanstd([m.get('test_mean', np.nan) for m in all_performance_metrics]) if all_performance_metrics else 0,
        'test_f1_std': np.nanstd([m.get('test_f1', np.nan) for m in all_performance_metrics]) if all_performance_metrics else 0,
        'performance_metrics': all_performance_metrics, # List of dicts per chunk
        'periodic_gmean': all_periodic_gmean, # List of tuples
        'used_attributes_over_time': all_used_attributes_over_time, # List of sets/lists
        'rules_conditionals': all_rules_conditionals, # List of dicts per chunk
        'ga_history_per_chunk': fitness_gmean_history_per_chunk, # List of dicts (GA history)
        'rule_details_per_chunk': rule_info_history, # List of dicts
        # Include other summary info if needed
        'total_features': len(attributes) if attributes else 0,
        'average_used_features': np.nanmean([len(f_set) for f_set in all_used_attributes_over_time if isinstance(f_set, (set, list))]) if all_used_attributes_over_time else np.nan,
        'final_memory_size': len(best_ever_memory),
        'run_duration': run_end_time - run_start_time
    }
    return run_summary_dict
    # --- End of run_experiment function ---


# --- Funções de Agregação e Relatório Final ---
# ... (compute_average_metrics_over_runs, generate_performance_table, display_feature_usage_summary - sem alterações) ...
def compute_average_metrics_over_runs(runs_results):
    if not runs_results or not isinstance(runs_results, list): logging.warning("compute_average_metrics_over_runs received empty or invalid input."); return {'average_test_gmean': 0, 'test_gmean_std': 0,'average_train_gmean': 0, 'train_gmean_std': 0,'average_test_f1': 0, 'test_f1_std': 0}
    all_runs_test_acc, all_runs_train_acc, all_runs_test_f1 = [], [], []
    for run_data in runs_results:
        if run_data and 'performance_metrics' in run_data and run_data['performance_metrics']:
             perf_metrics = run_data['performance_metrics']; all_runs_test_acc.extend([m.get('test_gmean', float('nan')) for m in perf_metrics]); all_runs_train_acc.extend([m.get('train_gmean', float('nan')) for m in perf_metrics]); all_runs_test_f1.extend([m.get('test_f1', float('nan')) for m in perf_metrics])
    avg_test_acc = np.nanmean(all_runs_test_acc) if all_runs_test_acc else 0; std_test_acc = np.nanstd(all_runs_test_acc) if all_runs_test_acc else 0; avg_train_acc = np.nanmean(all_runs_train_acc) if all_runs_train_acc else 0; std_train_acc = np.nanstd(all_runs_train_acc) if all_runs_train_acc else 0; avg_test_f1 = np.nanmean(all_runs_test_f1) if all_runs_test_f1 else 0; std_test_f1 = np.nanstd(all_runs_test_f1) if all_runs_test_f1 else 0
    return {'average_test_gmean': avg_test_acc, 'test_gmean_std': std_test_acc,'average_train_gmean': avg_train_acc, 'train_gmean_std': std_train_acc,'average_test_f1': avg_test_f1, 'test_f1_std': std_test_f1}

def generate_performance_table(all_datasets_summaries, base_results_dir):
    if not all_datasets_summaries:
        logging.warning("No dataset summaries to generate performance table.")
        return None

    # <<< AJUSTE: Usa 'stream_name' em vez de 'dataset_name' >>>
    # <<< Nota: Verifica se as chaves de std dev existem no dict retornado por compute_average_metrics_over_runs >>>
    data_for_df = {
        'Stream Name':       [ds.get('stream_name', 'N/A') for ds in all_datasets_summaries], # <<< CORRIGIDO
        'Dataset Type':      [ds.get('dataset_type', 'N/A') for ds in all_datasets_summaries], # <<< Adicionado para informação
        'Avg Train Acc':     [ds.get('average_train_gmean', float('nan')) for ds in all_datasets_summaries],
        'Std Train Acc':     [ds.get('train_gmean_std', float('nan')) for ds in all_datasets_summaries], # <<< Verificar se essa chave existe no sumário
        'Avg Test Acc':      [ds.get('average_test_gmean', float('nan')) for ds in all_datasets_summaries],
        'Std Test Acc':      [ds.get('test_gmean_std', float('nan')) for ds in all_datasets_summaries], # <<< Verificar se essa chave existe no sumário
        'Avg Test F1':       [ds.get('average_test_f1', float('nan')) for ds in all_datasets_summaries],
        'Std Test F1':       [ds.get('test_f1_std', float('nan')) for ds in all_datasets_summaries], # <<< Verificar nome/existência desta chave
        'Total Feats':       [ds.get('total_features', 'N/A') for ds in all_datasets_summaries],
        'Avg Used Feats':    [ds.get('average_used_features', float('nan')) for ds in all_datasets_summaries]
    }
    # <<< FIM AJUSTE >>>

    try: # Adiciona try/except para criação do DataFrame
        df = pd.DataFrame(data_for_df)
        num_cols = ['Avg Train Acc', 'Std Train Acc', 'Avg Test Acc', 'Std Test Acc', 'Avg Test F1', 'Std Test F1', 'Avg Used Feats']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(4)

        # Imprime a tabela formatada no console
        df_string = df.to_string(index=False, na_rep='N/A')
        print("\n" + "="*80 + "\nOverall Performance Summary Table\n" + "="*80 + "\n" + df_string + "\n" + "="*80 + "\n")

        # Salva a tabela em CSV
        table_path = os.path.join(base_results_dir, "performance_summary_table.csv")
        df.to_csv(table_path, index=False, na_rep='N/A')
        logging.info(f"Performance summary table saved to {table_path}")

    except ImportError:
         logging.error("Pandas library is required to generate the performance table. Please install pandas.")
         return None
    except Exception as e:
         logging.error(f"Failed to create or save performance table DataFrame: {e}")
         return None # Retorna None se falhar

    return df # Retorna o DataFrame se sucesso

def display_feature_usage_summary(all_datasets_summaries):
    if not all_datasets_summaries:
        return # Não imprime nada se a lista estiver vazia

    print("\n" + "="*80 + "\nFeature Usage Summary\n" + "="*80)
    for ds_summary in all_datasets_summaries:
        # <<< AJUSTE: Usa 'stream_name' em vez de 'dataset_name' >>>
        stream_name = ds_summary.get('stream_name', 'N/A')
        # <<< FIM AJUSTE >>>
        total_feat = ds_summary.get('total_features', 'N/A')
        avg_feat = ds_summary.get('average_used_features', 'N/A')
        # Formata avg_feat apenas se for numérico
        avg_feat_str = f"{avg_feat:.2f}" if isinstance(avg_feat, (int, float)) else 'N/A'

        # <<< AJUSTE: Usa stream_name na impressão >>>
        print(f"Stream: {stream_name}")
        # <<< FIM AJUSTE >>>
        print(f"  Total Features Available: {total_feat}")
        print(f"  Average Features Used (across chunks/runs): {avg_feat_str}")
        print("-" * 30) # Separador entre streams
    print("="*80 + "\n")

# --- Bloco Principal de Execução (Modificado) ---
if __name__ == "__main__":
    # --- Setup Inicial ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(script_dir, "config.yaml")

    # --- Configuração do Logging ---
    # Carrega a config SOMENTE AQUI para determinar o nível de log
    temp_config_for_log_level = load_config(config_file_path)
    log_level_str = "INFO" # Default
    if temp_config_for_log_level and 'experiment_settings' in temp_config_for_log_level:
        log_level_str = temp_config_for_log_level['experiment_settings'].get('logging_level', 'INFO').upper()
    
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Remove handlers existentes para evitar duplicação se o script for reexecutado em algum contexto interativo
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(level=log_level,
                        format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', # Mudado de %(module)s para %(name)s
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Cria o logger principal para o módulo main
    logger = logging.getLogger("main") # Usar getLogger("main") ao invés de "root" para logs específicos do main

    # Agora, carregue a configuração principal que será usada pelo resto do script
    config = load_config(config_file_path)
    if config is None:
        logger.critical(f"Exiting: Could not load or parse config: {config_file_path}.")
        exit(1)

    logger.info("===== Starting Experiment Execution with Loaded Configuration =====")
    logger.info(f"Using configuration file: {config_file_path}")

    # Extrai parâmetros globais (como antes)
    exp_settings = config.get('experiment_settings', {}) # type: ignore
    num_runs = exp_settings.get('num_runs', 1)
    base_results_dir_main = exp_settings.get('base_results_dir', BASE_RESULTS_DIR_DEFAULT)

    # Resolve e cria diretório base (como antes)
    if not os.path.isabs(base_results_dir_main):
        base_results_dir_main = os.path.join(script_dir, base_results_dir_main)
    logger.info(f"Relative base_results_dir resolved: {base_results_dir_main}")
    try:
        os.makedirs(base_results_dir_main, exist_ok=True)
    except OSError as e:
        logger.critical(f"Could not create base results directory {base_results_dir_main}: {e}")
        exit(1)

    # --- <<< MODIFICADO: Seleciona Experimentos Baseado no Modo >>> ---
    run_mode = exp_settings.get('run_mode', 'standard').lower()
    logger.info(f"Execution mode set to: '{run_mode}'")
    
    stream_names_to_run = []
    is_drift_mode = False

    if run_mode == 'standard':
        stream_names_to_run = exp_settings.get('standard_experiments', [])
        is_drift_mode = False
        if not stream_names_to_run:
            logger.error("Run mode is 'standard' but the 'standard_experiments' list is missing or empty in config.yaml. Exiting.")
            exit(1)
        logger.info(f"Running in STANDARD mode. Found {len(stream_names_to_run)} experiments to run.")
        
    elif run_mode == 'drift_simulation':
        stream_names_to_run = exp_settings.get('drift_simulation_experiments', [])
        is_drift_mode = True
        if not stream_names_to_run:
            logger.error("Run mode is 'drift_simulation' but the 'drift_simulation_experiments' list is missing or empty in config.yaml. Exiting.")
            exit(1)
        logger.info(f"Running in DRIFT SIMULATION mode. Found {len(stream_names_to_run)} experiments to run.")
        
    else:
        logger.error(f"Invalid 'run_mode' in config: '{run_mode}'. Must be 'standard' or 'drift_simulation'. Exiting.")
        exit(1)
    # --- <<< FIM DA MODIFICAÇÃO >>> ---

    # --- Executa Experimentos ---
    all_datasets_summaries = [] # Armazena resumos por TIPO de dataset base
    overall_start_time = time.time()

    # Itera sobre os NOMES DOS FLUXOS definidos no config
    for stream_name in stream_names_to_run:
        stream_start_time = time.time()

        # <<< LINHA CORRIGIDA >>>
        logger.info(f"====== Processing Experiment ID: {stream_name} (Runs: {num_runs}) ======")

        # # <<< ADICIONADO: Obter o tipo base do dataset para agregação/logging >>>
        # try:
        #     dataset_type = experiment_streams_config[stream_name].get('dataset_type', 'UnknownType')
        # except KeyError:
        #      logger.error(f"Could not find definition for stream '{stream_name}' while trying to get its type. Skipping.")
        #      continue
        # except AttributeError:
        #      logger.error(f"Structure error in config for stream '{stream_name}'. Skipping.")
        #      continue

        # logger.info(f"====== Processing Stream: {stream_name} (Type: {dataset_type}, Runs: {num_runs}) ======")
        # <<< FIM ADICIONADO >>>

        runs_results = []
        dataset_attributes = None # Armazena atributos para o tipo de dataset (pega do primeiro run)

        for i in range(num_runs):
            # <<< MODIFICADO: Passa stream_name para run_experiment >>>
            run_result = run_experiment(
                experiment_id=stream_name, # Passa o ID atual
                run_number=i + 1,
                config=config, # type: ignore
                base_results_dir=base_results_dir_main,
                is_drift_simulation=is_drift_mode # Passa o flag do modo
            )
            # <<< FIM DA MODIFICAÇÃO >>>

            if run_result:
                runs_results.append(run_result)
                # Pega os atributos da primeira execução bem-sucedida deste TIPO de dataset
                if dataset_attributes is None and 'attributes' in run_result:
                    dataset_attributes = run_result['attributes'] # Usado para plot mosaico depois
            else:
                logger.error(f"Run {i+1} for experiment '{stream_name}' failed.")

        # --- Agrega e Plota Resultados por STREAM ---
        if runs_results:
            logger.info(f"Aggregating results for experiment: {stream_name}")
            # Calcula métricas médias para esta STREAM específica
            # (Pode adaptar compute_average_metrics_over_runs se necessário)
            logger.info(f"Aggregating results for experiment: {stream_name}")
            # Supondo que compute_average_metrics_over_runs funciona com a lista de dicts retornados
            avg_metrics = compute_average_metrics_over_runs(runs_results)
            dataset_type_agg = runs_results[0].get('dataset_type', 'Unknown') # Pega tipo do primeiro run
            total_features = len(dataset_attributes) if dataset_attributes else 0
            avg_used_features = np.nanmean([run.get('average_used_features', np.nan) for run in runs_results])

            # Armazena resumo usando dataset_type para agregação posterior
            stream_summary = {
                'stream_name': stream_name,
                'dataset_type': dataset_type_agg, # Armazena o tipo base
                'total_features': total_features,
                'average_used_features': avg_used_features,
                **avg_metrics # Adiciona métricas médias (avg_test_gmean, etc.)
            }
            all_datasets_summaries.append(stream_summary)
            # <<< FIM DA MODIFICAÇÃO >>>

            # --- Plot Mosaico (Ajustado para usar dataset_type no nome do arquivo) ---
            try:
                if dataset_attributes:
                    # Cria diretório específico para o tipo de dataset base, se não existir
                    dataset_type_dir = os.path.join(base_results_dir_main, dataset_type_agg)
                    os.makedirs(dataset_type_dir, exist_ok=True)
                    # Salva plot mosaico dentro do diretório do tipo de dataset
                    mosaic_save_path = os.path.join(dataset_type_dir, f"MosaicPlot_{stream_name}.png") # Nome inclui stream_name
                    logger.info(f"Generating mosaic plot for {stream_name}...")
                    # Passa dataset_type ou stream_name para a função de plot, dependendo do que ela espera
                    plotting.create_mosaic_plots(dataset_type_agg, runs_results, dataset_attributes, save_path=mosaic_save_path) # Ajuste se necessário
                else:
                    logger.warning(f"Skipping mosaic plot for {stream_name}: Attributes missing.")
            except Exception as plot_e:
                logger.error(f"Failed to generate mosaic plot for {stream_name}: {plot_e}", exc_info=True)
            # --- Fim Plot Mosaico ---
        else:
            logger.warning(f"No successful runs for {stream_name}. Skipping aggregation.")

        stream_end_time = time.time()
        logger.info(f"====== Finished Stream: {stream_name} in {stream_end_time - stream_start_time:.2f} seconds ======")
        # --- Fim do Loop por Stream ---

    # --- Relatório Final Agregado por TIPO de Dataset (Opcional) ---
    logger.info("====== Generating Final Summary Reports ======")
    if all_datasets_summaries:
        # Agrupa sumários por dataset_type para plots/tabelas gerais
        summary_by_type = {}
        for summary in all_datasets_summaries:
            dtype = summary['dataset_type']
            if dtype not in summary_by_type:
                 summary_by_type[dtype] = []
            summary_by_type[dtype].append(summary)

        # Gere plots/tabelas gerais aqui, talvez agregando por dataset_type
        # Exemplo: Plotar performance média por TIPO de dataset
        try:
            overall_perf_path = os.path.join(base_results_dir_main, "Overall_Performance_by_Type.png")
            plotting.plot_overall_performance_with_std(all_datasets_summaries, save_path=overall_perf_path, group_by='dataset_type') # Adaptar função de plot
            logger.info(f"Overall performance plot by type saved to {overall_perf_path}")
        except Exception as e:
            logger.error(f"Failed to generate overall performance plot by type: {e}", exc_info=True)

        # Gerar tabela de performance (pode precisar adaptar a função)
        try:
            generate_performance_table(all_datasets_summaries, base_results_dir_main)
        except Exception as e:
            logger.error(f"Failed to generate performance table: {e}", exc_info=True)

        # Mostrar resumo de uso de features (pode precisar adaptar a função)
        try:
            display_feature_usage_summary(all_datasets_summaries)
        except Exception as e:
            logger.error(f"Failed to display feature usage summary: {e}", exc_info=True)

    else:
        logger.warning("No dataset summaries available for final reports.")

    overall_end_time = time.time()
    logger.info(f"Total experiment duration: {overall_end_time - overall_start_time:.2f} seconds.")
    logger.info("===== Experiment Execution Finished =====")
