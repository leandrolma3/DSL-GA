# data_handling.py (Dual Mode: Standard River Datasets and Drift Simulation)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import logging
import os
import copy
import yaml
import math
from typing import Callable, Dict, List, Tuple, Any, Union, Iterable # Ensure all needed types are imported

# Import River components
from river import datasets, stream
# Use the centralized seed if available, otherwise define one
try:
    from constants import RANDOM_SEED
except ImportError:
    RANDOM_SEED = 42 # Default seed if constants.py is not available
    logging.warning("RANDOM_SEED not found in constants.py, using default 42.")

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("data_handling") # Use specific logger
CONFIG: Dict | None = None

# --- Concept Labeling Functions (Needed for Drift Simulation Mode) ---
# (Include get_sea_label, get_rbf_label, get_agrawal_label, get_stagger_label, get_sine_label, get_friedman_label)
# ... (Previous implementations of get_label functions go here) ...
def get_feature_vector(instance_dict: Dict, feature_names: List) -> np.ndarray:
    """Converts a River instance dictionary to a NumPy array in fixed order."""
    return np.array([instance_dict.get(feat, 0.0) for feat in feature_names], dtype=float)

def get_sea_label(instance_vec: np.ndarray, concept_params: Dict[str, Any]) -> int:
    threshold = concept_params.get('threshold', 7.0)
    try:
        if len(instance_vec) < 2: raise IndexError("SEA requires 2 features.")
        return 0 if float(instance_vec[0]) + float(instance_vec[1]) <= float(threshold) else 1
    except Exception as e: logger.error(f"Error SEA Label: {e}"); return 0

def get_rbf_label(instance_vec: np.ndarray, concept_params: Dict[str, Any]) -> int:
    centroids = concept_params.get('centroids')
    if not centroids: return 0
    min_dist_sq = float('inf'); nearest_class = 0
    try:
        for centroid in centroids:
            coords = np.array(centroid['coords'], dtype=float)
            if coords.shape != instance_vec.shape: continue
            dist_sq = np.sum((instance_vec - coords) ** 2)
            if dist_sq < min_dist_sq: min_dist_sq = dist_sq; nearest_class = int(centroid['class'])
        return nearest_class
    except Exception as e: logger.error(f"Error RBF Label: {e}"); return 0

def get_agrawal_label(instance_vec: np.ndarray, concept_params: Dict[str, Any]) -> int:
    func_id = concept_params.get('function_id')
    if func_id is None: return 0
    try:
        instance = instance_vec # Use vector directly
        sal = float(instance[0]) * 130000 + 20000; comm = float(instance[1]) * 100000 if float(instance[1]) > 0 else 0
        age = float(instance[2]) * 60 + 20; elevel = int(float(instance[3]) * 5)
        hvalue = float(instance[6]) * 900000 + 50000; hyears = float(instance[7]) * 30 + 1
        loan = float(instance[8]) * 500000
        label = 0
        if func_id == 1: label = 1 if (age < 40 or age >= 60) else 0
        elif func_id == 2: label = 1 if (((age < 40) and (50000 <= sal <= 100000)) or ((40 <= age < 60) and (75000 <= sal <= 125000)) or ((age >= 60) and (25000 <= sal <= 75000))) else 0
        elif func_id == 3: label = 1 if (((age < 40) and (elevel in [0, 1])) or ((40 <= age < 60) and (elevel in [1, 2, 3])) or ((age >= 60) and (elevel in [2, 3, 4]))) else 0
        elif func_id >= 4 and func_id <= 10: label = 1 if float(instance[0]) > 0.5 else 0 # Placeholder
        else: logger.warning(f"Unknown Agrawal function_id: {func_id}")
        return label
    except Exception as e: logger.error(f"Error AGRAWAL Label: {e}"); return 0

def get_stagger_label(instance_vec: np.ndarray, concept_params: Dict[str, Any]) -> int:
    concept_id = concept_params.get('concept_id')
    if concept_id is None: return 0
    try:
        if len(instance_vec) < 3: raise IndexError("STAGGER requires 3 features.")
        size = max(0, min(2, math.floor(float(instance_vec[0]))))
        color = max(0, min(2, math.floor(float(instance_vec[1]))))
        shape = max(0, min(2, math.floor(float(instance_vec[2]))))
        label = 0
        if concept_id == 1: label = 1 if (size == 0 and color == 0) else 0
        elif concept_id == 2: label = 1 if (color == 1 or shape == 0) else 0
        elif concept_id == 3: label = 1 if (size == 1 or size == 2) else 0
        else: logger.warning(f"Unknown STAGGER concept_id: {concept_id}")
        return label
    except Exception as e: logger.error(f"Error STAGGER Label: {e}"); return 0

def get_sine_label(instance_vec: np.ndarray, concept_params: Dict[str, Any]) -> int:
    concept_type = concept_params.get('type', 'sum'); threshold = concept_params.get('threshold', 1.0)
    try:
        if len(instance_vec) < 2: raise IndexError("Sine requires 2 features.")
        x1 = float(instance_vec[0]); x2 = float(instance_vec[1])
        value = 0.0
        if concept_type == 'sum': value = math.sin(x1) + math.sin(x2)
        elif concept_type == 'prod': value = math.sin(x1) * math.sin(x2)
        else: logger.warning(f"Unknown Sine type: {concept_type}"); return 0
        return 1 if value > float(threshold) else 0
    except Exception as e: logger.error(f"Error Sine Label: {e}"); return 0

def get_friedman_label(instance_vec: np.ndarray, concept_params: Dict[str, Any]) -> int: return 0

CONCEPT_FUNCTIONS: Dict[str, Callable] = {
    "SEA": get_sea_label, "RBF": get_rbf_label, "AGRAWAL": get_agrawal_label,
    "STAGGER": get_stagger_label, "SINE": get_sine_label, "FRIEDMAN": get_friedman_label,
    "HYPERPLANE": lambda inst, params: 0, "RANDOMTREE": lambda inst, params: 0,
    "LED": lambda inst, params: 0, "WAVEFORM": lambda inst, params: 0,
    "AIRLINE": lambda inst, params: 0, "ELEC2": lambda inst, params: 0,
    "PHISHING": lambda inst, params: 0, "CREDITCARD": lambda inst, params: 0,
    "COVERTYPE": lambda inst, params: 0,
}

# --- Configuration and Helper Functions ---
# (load_full_config, get_stream_definition, get_concept_params,
#  get_feature_bounds, get_concept_for_chunk - as before)
def load_full_config(config_path: str = 'config.yaml'):
    global CONFIG
    if CONFIG is None:
        if not os.path.exists(config_path): raise FileNotFoundError(f"Config file not found: {config_path}")
        try:
            with open(config_path, 'r') as f: CONFIG = yaml.safe_load(f)
            logger.info(f"Full configuration loaded from {config_path}")
            if 'drift_analysis' not in CONFIG: logger.warning(f"Config file {config_path} missing 'drift_analysis'.") # type: ignore
            if 'experiment_streams' not in CONFIG: logger.warning(f"Config file {config_path} missing 'experiment_streams'.") # type: ignore
        except Exception as e: logger.error(f"Error loading/parsing {config_path}: {e}"); CONFIG = {}; raise
    return CONFIG

def get_stream_definition(stream_name: str, config: Dict) -> Union[Dict, None]:
    try: return config['experiment_streams'][stream_name]
    except KeyError: logger.error(f"Stream definition '{stream_name}' not found."); return None
    except Exception as e: logger.error(f"Error accessing stream definition '{stream_name}': {e}"); return None

def get_concept_params(dataset_type: str, concept_id: str, config: Dict) -> Union[Dict, None]:
     try: dt_upper = dataset_type.upper(); return config['drift_analysis']['datasets'][dt_upper]['concepts'][str(concept_id)]
     except KeyError: logger.error(f"Concept params for ID '{concept_id}' type '{dataset_type}' not found."); return None
     except Exception as e: logger.error(f"Error accessing concept params for {dataset_type}/{concept_id}: {e}"); return None

def get_feature_bounds(dataset_type: str, config: Dict) -> Union[List[List[float]], None]:
    try: dt_upper = dataset_type.upper(); return config['drift_analysis']['datasets'][dt_upper]['feature_bounds']
    except KeyError: logger.error(f"Feature bounds for type '{dataset_type}' not found."); return None
    except Exception as e: logger.error(f"Error accessing feature bounds for {dataset_type}: {e}"); return None

def get_concept_for_chunk(stream_definition: Dict, chunk_index: int) -> Tuple[Union[str, None], Union[str, None], float]:
    concept_sequence = stream_definition.get('concept_sequence', []); drift_type = stream_definition.get('drift_type', 'abrupt')
    gradual_width = stream_definition.get('gradual_drift_width_chunks', 0)
    if not concept_sequence: return None, None, 0.0
    current_chunk_start = 0; previous_concept_id = None
    for i, stage in enumerate(concept_sequence):
        concept_id = str(stage['concept_id']); duration = stage['duration_chunks']
        current_chunk_end = current_chunk_start + duration
        if current_chunk_start <= chunk_index < current_chunk_end:
            mixture_prob_b = 0.0; concept_id_a = concept_id; concept_id_b = None
            if drift_type == 'gradual' and i > 0 and gradual_width > 0 and previous_concept_id is not None:
                transition_start_chunk = current_chunk_start; transition_end_chunk = current_chunk_start + gradual_width
                if transition_start_chunk <= chunk_index < transition_end_chunk:
                    concept_id_a = str(previous_concept_id); concept_id_b = concept_id
                    chunk_pos_in_window = chunk_index - transition_start_chunk
                    mid_proportion = (chunk_pos_in_window + 0.5) / gradual_width
                    mixture_prob_b = max(0.0, min(1.0, mid_proportion))
                    logger.debug(f"Chunk {chunk_index}: Gradual drift {concept_id_a}->{concept_id_b}. Mix_Prob(B): {mixture_prob_b:.3f}")
                    return concept_id_a, concept_id_b, mixture_prob_b
            logger.debug(f"Chunk {chunk_index}: Stable/Abrupt concept {concept_id_a}.")
            return concept_id_a, None, 0.0
        previous_concept_id = concept_id; current_chunk_start = current_chunk_end
    logger.warning(f"Chunk index {chunk_index} beyond sequence length.")
    last_concept_id = str(concept_sequence[-1]['concept_id']) if concept_sequence else None
    return last_concept_id, None, 0.0

# --- Internal Chunk Generation Functions ---

def _generate_chunks_from_iterator(
    stream_iterator: Iterable[Tuple[Dict, Any]],
    chunk_size: int,
    max_chunks: Union[int, None]
    ) -> List[Tuple[List[Dict], List]]:
    """Internal function: Generates chunks directly from a stream iterator."""
    # (Code as provided previously)
    if chunk_size <= 0: logger.error("Chunk size must be positive."); return []
    chunks = []; X_chunk = []; y_chunk = []; instance_count = 0
    try:
        for x, y in stream_iterator:
            X_chunk.append(x); y_chunk.append(y); instance_count += 1
            if len(X_chunk) == chunk_size:
                chunks.append((copy.deepcopy(X_chunk), copy.deepcopy(y_chunk)))
                X_chunk.clear(); y_chunk.clear()
                if max_chunks is not None and len(chunks) == max_chunks: logger.info(f"Reached max chunks ({max_chunks}) in standard mode."); break
        if X_chunk: chunks.append((X_chunk, y_chunk))
    except Exception as e: logger.error(f"Error reading standard stream iterator: {e}", exc_info=True)
    logger.info(f"Standard mode generated {len(chunks)} chunks ({instance_count} instances).")
    return chunks


def _generate_chunks_with_drift_simulation(
    base_stream_generator: Iterable[Tuple[Dict, Any]],
    chunk_size: int,
    max_chunks: Union[int, None],
    stream_name: str,
    config: Dict,
    feature_names: List[str]
    ) -> List[Tuple[List[Dict], List]]:
    """Internal function: Generates chunks with label recalculation."""
    # (Code as provided previously)
    if chunk_size <= 0: logger.error("Chunk size must be positive."); return []
    if not feature_names: logger.error("Feature names required."); return []
    stream_definition = get_stream_definition(stream_name, config);
    if not stream_definition: return []
    dataset_type = stream_definition.get('dataset_type')
    concept_func = CONCEPT_FUNCTIONS.get(dataset_type.upper()) if dataset_type else None
    if not concept_func: logger.error(f"Concept function for '{dataset_type}' not found."); return []

    chunks = []; X_chunk_dicts = []; y_chunk_recalculated = []
    instance_count = 0; chunk_index = 0
    concept_id_a, concept_id_b, p_mix_b = get_concept_for_chunk(stream_definition, chunk_index)
    params_a = get_concept_params(dataset_type, concept_id_a, config) if concept_id_a else None # type: ignore
    params_b = get_concept_params(dataset_type, concept_id_b, config) if concept_id_b else None # type: ignore
    logger.info(f"[Chunk {chunk_index}] DriftSim: Starting generation. Concept(s): {concept_id_a}{f' -> {concept_id_b} (Mix:{p_mix_b:.2f})' if concept_id_b else ''}")
    if (concept_id_a and not params_a) or (concept_id_b and not params_b): logger.error(f"Missing params chunk {chunk_index}."); return []

    try:
        for x_dict, _ in base_stream_generator:
            instance_vec = get_feature_vector(x_dict, feature_names); calculated_y = 0
            try:
                if concept_id_b and np.random.rand() < p_mix_b:
                    if params_b: calculated_y = concept_func(instance_vec, params_b)
                elif concept_id_a:
                    if params_a: calculated_y = concept_func(instance_vec, params_a)
            except Exception as e: logger.error(f"Label calc error inst {instance_count}: {e}", exc_info=False); calculated_y = 0
            X_chunk_dicts.append(x_dict); y_chunk_recalculated.append(calculated_y); instance_count += 1
            if len(X_chunk_dicts) == chunk_size:
                chunks.append((copy.deepcopy(X_chunk_dicts), copy.deepcopy(y_chunk_recalculated)))
                X_chunk_dicts.clear(); y_chunk_recalculated.clear(); chunk_index += 1
                if max_chunks is not None and len(chunks) == max_chunks: logger.info(f"Reached max chunks ({max_chunks}) drift sim."); break
                concept_id_a, concept_id_b, p_mix_b = get_concept_for_chunk(stream_definition, chunk_index)
                params_a = get_concept_params(dataset_type, concept_id_a, config) if concept_id_a else None # type: ignore
                params_b = get_concept_params(dataset_type, concept_id_b, config) if concept_id_b else None # type: ignore
                logger.info(f"[Chunk {chunk_index}] DriftSim: Starting generation. Concept(s): {concept_id_a}{f' -> {concept_id_b} (Mix:{p_mix_b:.2f})' if concept_id_b else ''}")
                if (concept_id_a and not params_a) or (concept_id_b and not params_b): logger.error(f"Missing params chunk {chunk_index}. Stopping."); break
        if X_chunk_dicts: chunks.append((X_chunk_dicts, y_chunk_recalculated))
    except Exception as e: logger.error(f"Error reading base River stream drift sim: {e}", exc_info=True)
    logger.info(f"Drift sim generated {len(chunks)} chunks ({instance_count} instances).")
    return chunks


# --- Concept Drift Detection (Feature-based) ---
def detect_concept_drift(chunks, name_for_plot: str, run_number=1, results_dir="results"):
    """Analyzes and visualizes feature drift between chunks."""
    num_chunks = len(chunks)
    # <<< CORRECTION: Split line >>>
    if num_chunks < 2:
        logger.warning("Need >= 2 chunks for drift detection. Skipping.")
        return
    # <<< END CORRECTION >>>

    # (Rest of the function remains the same as provided previously)
    feature_names = []
    for i, (X, y) in enumerate(chunks):
      if X: # Check if chunk is not empty and has at least one instance
                  try:
                      # Get keys from first instance dict
                      feature_names = sorted(list(X[0].keys()))
                      # If successful, break the loop to use these feature names
                      if feature_names: # Ensure keys were actually found
                          first_valid_chunk_idx = i
                          break
                  except (IndexError, KeyError, TypeError, Exception) as e:
                      # Log the error and continue to the next chunk if extraction fails
                      logger.warning(f"Could not extract features from first instance of chunk {i}: {e}")
                      continue # Try next chunk    if not feature_names: logger.warning("No features found. Skipping drift detection."); return
    num_features = len(feature_names); logger.info(f"Starting feature drift analysis: {num_features} features")
    feature_drift_magnitudes = []
    for i in range(num_chunks - 1):
        X_prev_dicts, _ = chunks[i]; X_next_dicts, _ = chunks[i + 1]
        if not X_prev_dicts or not X_next_dicts: feature_drift_magnitudes.append(np.full(num_features, np.nan)); continue
        try:
            X_prev = np.array([[x.get(feat, 0) for feat in feature_names] for x in X_prev_dicts], dtype=float)
            X_next = np.array([[x.get(feat, 0) for feat in feature_names] for x in X_next_dicts], dtype=float)
        except Exception as e: logger.error(f"Dict->Array error chunk {i}->{i+1}: {e}"); feature_drift_magnitudes.append(np.full(num_features, np.nan)); continue
        mean_prev, var_prev = np.nanmean(X_prev, axis=0), np.nanvar(X_prev, axis=0)
        mean_next, var_next = np.nanmean(X_next, axis=0), np.nanvar(X_next, axis=0)
        mean_drift = np.abs(mean_next - mean_prev); var_drift = np.abs(var_next - var_prev)
        drift_magnitude = np.nan_to_num(np.nansum([mean_drift, var_drift], axis=0), nan=0.0)
        feature_drift_magnitudes.append(drift_magnitude)
    drift_matrix = np.array(feature_drift_magnitudes).T
    if drift_matrix.size == 0: return
    try:
        plt.figure(figsize=(max(8, num_chunks), max(6, num_features / 2)))
        sns.heatmap(drift_matrix, cmap="coolwarm", xticklabels=[f'{i}->{i+1}' for i in range(num_chunks - 1)], yticklabels=feature_names, annot=False, fmt=".2f", cbar=True, cbar_kws={'label': 'Feature Drift Magnitude'})
        plt.xlabel("Chunk Transition"); plt.ylabel("Feature"); plt.title(f"Feature Drift Heatmap: {name_for_plot} (Run {run_number})")
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
        os.makedirs(results_dir, exist_ok=True)
        plot_filename = os.path.join(results_dir, f"FeatureDrift_{name_for_plot}_Run{run_number}.png") # Renamed plot
        plt.savefig(plot_filename, bbox_inches='tight'); logger.info(f"Feature drift heatmap saved: {plot_filename}")
    except Exception as e: logger.error(f"Failed to generate/save feature drift heatmap: {e}")
    finally: plt.close()


# --- Main Entry Point Function ---
def generate_dataset_chunks(
    stream_or_dataset_name: str,
    chunk_size: int,
    num_chunks: int = 10,
    max_instances: int = 100000,
    run_number: int = 1,
    results_dir: str = "results",
    config_path: str = 'config.yaml'
    ) -> List[Tuple[List[Dict], List]]:
    """
    Generates data chunks either using a standard River dataset or simulating
    a drift stream defined in the configuration file.
    """
    # (Code as provided previously, calling the internal functions)
    config = load_full_config(config_path)
    stream_definition = get_stream_definition(stream_or_dataset_name, config) # type: ignore
    base_stream_generator = None; instance_limit = max_instances
    dataset_type_for_init = None; mode = "Unknown"

    if stream_definition:
        mode = "Drift Simulation"; logger.info(f"--- Mode: {mode} for Stream: '{stream_or_dataset_name}' ---")
        dataset_type_for_init = stream_definition.get('dataset_type')
        if not dataset_type_for_init: logger.error(f"Stream '{stream_or_dataset_name}' missing 'dataset_type'."); return []
    else:
        mode = "Standard River Dataset"; logger.info(f"--- Mode: {mode} for Dataset Type: '{stream_or_dataset_name}' ---")
        dataset_type_for_init = stream_or_dataset_name

    logger.info(f"Initializing base generator for type: {dataset_type_for_init}")
    try:
        dt_upper = dataset_type_for_init.upper()
        if dt_upper == 'SEA': base_stream_generator = datasets.synth.SEA(seed=RANDOM_SEED).take(instance_limit)
        elif dt_upper == 'RBF': base_stream_generator = datasets.synth.RandomRBF(seed_model=RANDOM_SEED, seed_sample=RANDOM_SEED).take(instance_limit)
        elif dt_upper == 'AGRAWAL': base_stream_generator = datasets.synth.Agrawal(seed=RANDOM_SEED).take(instance_limit)
        elif dt_upper == 'STAGGER': base_stream_generator = datasets.synth.STAGGER(seed=RANDOM_SEED).take(instance_limit)
        elif dt_upper == 'SINE': base_stream_generator = datasets.synth.Sine(seed=RANDOM_SEED).take(instance_limit)
        elif dt_upper == 'HYPERPLANE': base_stream_generator = datasets.synth.Hyperplane(seed=RANDOM_SEED).take(instance_limit)
        elif dt_upper == 'RANDOMTREE': base_stream_generator = datasets.synth.RandomTree(seed_tree=RANDOM_SEED, seed_sample=RANDOM_SEED).take(instance_limit)
        elif dt_upper == 'LED': base_stream_generator = datasets.synth.LED(seed=RANDOM_SEED).take(instance_limit)
        elif dt_upper == 'WAVEFORM': base_stream_generator = datasets.synth.Waveform(seed=RANDOM_SEED).take(instance_limit)
        elif dt_upper == 'FRIEDMAN':
            data_stream = datasets.synth.Friedman(seed=RANDOM_SEED).take(instance_limit)
            data_list = list(data_stream); X_all = [x for x, y in data_list]; y_all = [y for x, y in data_list]
            if not data_list: logger.error("Friedman stream empty."); return []
            try: y_binned = pd.qcut(y_all, q=3, labels=False, duplicates='drop'); base_stream_generator = iter(zip(X_all, y_binned)); logger.info("Discretized Friedman target.")
            except ImportError: logger.error("Pandas required for Friedman."); return []
            except ValueError as e: logger.error(f"Friedman discretization failed: {e}"); raise ValueError("Friedman discretization failed.") from e
        elif dt_upper == 'AIRLINE': base_stream_generator = datasets.Airline().take(instance_limit) # type: ignore
        elif dt_upper == 'ELEC2': base_stream_generator = datasets.Elec2().take(instance_limit)
        elif dt_upper == 'PHISHING': base_stream_generator = datasets.Phishing().take(instance_limit)
        elif dt_upper == 'CREDITCARD': base_stream_generator = datasets.CreditCard().take(instance_limit)
        elif dt_upper == 'COVERTYPE':
             covtype_url = 'https://raw.githubusercontent.com/online-ml/river/master/river/datasets/covtype.csv.zip'
             try: base_stream_generator = stream.iter_csv(covtype_url, target='Cover_Type', compression='zip', converters={'Cover_Type': int}).take(instance_limit); logger.info(f"Using Covertype stream.") # type: ignore
             except Exception as e: logger.error(f"Failed to load Covertype: {e}"); return []
        else: logger.error(f"Dataset type '{dataset_type_for_init}' not recognized."); return []
    except Exception as e: logger.error(f"Error initializing base stream for {dataset_type_for_init}: {e}"); return []

    if base_stream_generator is None: logger.error(f"Base stream generator failed for {dataset_type_for_init}."); return []

    feature_names = []
    # Simplified feature name extraction for now - assumes dict and peeking works or fails gracefully
    try:
        # Attempt to get feature names from the first instance
        gen_peek, base_stream_generator = itertools.tee(base_stream_generator) # Use tee to avoid consuming
        first_x, _ = next(gen_peek)
        if isinstance(first_x, dict): feature_names = sorted(list(first_x.keys()))
        else: logger.error("First instance not dict."); return []
        logger.debug(f"Determined feature names: {feature_names}")
    except StopIteration: logger.error("Base stream empty."); return []
    except Exception as e: logger.error(f"Error peeking features: {e}."); return [] # Fallback might be needed here

    if not feature_names and mode == "Drift Simulation":
         logger.error("Could not determine feature names, required for drift simulation.")
         return []

    chunks = []
    if mode == "Drift Simulation":
        chunks = _generate_chunks_with_drift_simulation(base_stream_generator, chunk_size, num_chunks, stream_or_dataset_name, config, feature_names) # type: ignore
    elif mode == "Standard River Dataset":
        chunks = _generate_chunks_from_iterator(base_stream_generator, chunk_size, num_chunks)
    else: logger.error("Unknown operating mode."); return []

    if chunks: detect_concept_drift(chunks, stream_or_dataset_name, run_number, results_dir)
    else: logger.warning(f"No chunks generated for '{stream_or_dataset_name}', skipping drift detection.")

    return chunks

# --- Example Usage ---
if __name__ == '__main__':
    print("Testing Integrated Data Handling Generation...")
    CONFIG_FILE = 'config.yaml'
    if not os.path.exists(CONFIG_FILE): print(f"\nERROR: {CONFIG_FILE} not found.")
    else:
        try:
            # Test 1: Drift Simulation Mode
            stream_name_drift = "SEA_Abrupt_Mild_Recurrent"
            print(f"\n--- Generating Stream (Drift Sim Mode): {stream_name_drift} ---")
            stream1_chunks = generate_dataset_chunks(stream_name_drift, chunk_size=500, num_chunks=5, config_path=CONFIG_FILE)
            print(f"Generated {len(stream1_chunks)} chunks for {stream_name_drift}.")
            if stream1_chunks: print(f" Labels in first chunk: {np.bincount(stream1_chunks[0][1]).tolist()}")

            # Test 2: Standard Mode
            stream_name_standard = "RBF"
            print(f"\n--- Generating Stream (Standard Mode): {stream_name_standard} ---")
            stream2_chunks = generate_dataset_chunks(stream_name_standard, chunk_size=500, num_chunks=5, config_path=CONFIG_FILE)
            print(f"Generated {len(stream2_chunks)} chunks for {stream_name_standard}.")
            if stream2_chunks: print(f" Labels in first chunk: {np.bincount(stream2_chunks[0][1]).tolist()}")

        except Exception as e: print(f"\nError during test: {e}"); logger.exception("Test block error.")

