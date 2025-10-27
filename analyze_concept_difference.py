# analyze_concept_difference.py (Versão 3, Final e Robusta)

import yaml
import numpy as np
import itertools
import os
import logging
import math
from typing import Callable, Dict, List, Any, Tuple

# Imports para plotagem e salvamento
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("ConceptDifferenceAnalyzer")

# --- Funções de Rótulo SINCRONIZADAS com o config.yaml ---

def get_sea_label(instance: np.ndarray, concept_params: Dict[str, Any]) -> int:
    """Simula a lógica do gerador SEA do River, usando o parâmetro 'variant'."""
    variant_map = {0: 8.0, 1: 9.0, 2: 7.0, 3: 9.5}
    variant = concept_params.get('variant', 0)
    threshold = variant_map.get(variant, 8.0)
    return 0 if float(instance[0]) + float(instance[1]) <= threshold else 1

def get_agrawal_label(instance: np.ndarray, concept_params: Dict[str, Any]) -> int:
    """Simula a lógica do gerador Agrawal, usando 'classification_function' e com a sintaxe corrigida."""
    # O parâmetro no config.yaml é 'classification_function' (0 a 9)
    func_id = concept_params.get('classification_function', 0) + 1 # Converte para 1 a 10

    # Extrai os atributos com base na escala do gerador Agrawal
    salary = 20000 + instance[0] * 130000
    commission = 0 if instance[1] <= 0 else instance[1] * 100000
    age = 20 + instance[2] * 60
    elevel = int(instance[3] * 4) # 0 a 4
    car = int(instance[4] * 19) # 1 a 20
    zipcode = int(instance[5] * 8) # 0 a 8
    hvalue = 50000 + instance[6] * 900000
    hyears = 1 + instance[7] * 30
    loan = instance[8] * 500000

    # <<< CORREÇÃO: Lógica reescrita com if/elif/else padrão >>>
    if func_id == 1:
        if age < 40 or age >= 60:
            return 1
        return 0

    elif func_id == 2:
        if age < 40:
            return 1 if 50000 <= salary <= 100000 else 0
        elif age < 60:
            return 1 if 75000 <= salary <= 125000 else 0
        else: # age >= 60
            return 1 if 25000 <= salary <= 75000 else 0

    elif func_id == 3:
        if age < 40:
            return 1 if elevel in [0, 1] else 0
        elif age < 60:
            return 1 if elevel in [1, 2, 3] else 0
        else: # age >= 60
            return 1 if elevel in [2, 3, 4] else 0
            
    elif func_id == 4:
        if elevel in [0, 1]:
            return 1 if 100000 <= loan <= 299999 else 0
        elif elevel in [2, 3]:
            return 1 if 200000 <= loan <= 399999 else 0
        else: # elevel == 4
            return 1 if 300000 <= loan <= 499999 else 0

    elif func_id == 5:
        if age < 40:
            return 1 if (50000 <= salary <= 100000) and (100000 <= loan <= 300000) else 0
        elif age < 60:
            return 1 if (75000 <= salary <= 125000) and (200000 <= loan <= 400000) else 0
        else: # age >= 60
            return 1 if (25000 <= salary <= 75000) and (300000 <= loan <= 500000) else 0

    elif func_id == 6:
        total_salary = salary + commission
        return 1 if (total_salary > (2/3 * (loan - 50000) + 20000)) and (total_salary < (4/3 * (loan - 50000) + 25000)) else 0
        
    elif func_id == 7:
        total_salary = salary + commission
        return 1 if (total_salary > (2/3 * (5000 * elevel) + 20000)) and (total_salary < (4/3 * (5000 * elevel) + 25000)) else 0

    elif func_id == 8:
        total_salary = salary + commission
        return 1 if (total_salary > (2/3 * (loan - 50000 + 5000 * elevel) + 20000)) and (total_salary < (4/3 * (loan - 50000 + 5000 * elevel) + 25000)) else 0

    elif func_id == 9:
        total_salary = salary + commission
        return 1 if (total_salary > (2/3 * (loan - 50000) + (50000 * (hyears - 20) if hyears > 20 else 0) + 20000)) and (total_salary < (4/3 * (loan - 50000) + (50000 * (hyears - 20) if hyears > 20 else 0) + 25000)) else 0
        
    elif func_id == 10:
        equity = 0
        if hyears > 20:
            equity = hvalue * (hyears - 20) / 10
        else:
            equity = hvalue * (1 - 0.05 * hyears)
        total_salary = salary + commission
        return 1 if (total_salary > (2/3 * (loan - 50000) - equity/5 + 20000)) and (total_salary < (4/3 * (loan - 50000) - equity/5 + 25000)) else 0

    return 0



def get_stagger_label(instance: np.ndarray, concept_params: Dict[str, Any]) -> int:
    """Simula a lógica do gerador STAGGER, usando 'classification_function'."""
    concept_id = concept_params.get('classification_function', 0) + 1
    
    size = max(0, min(2, math.floor(float(instance[0]))))
    color = max(0, min(2, math.floor(float(instance[1]))))
    shape = max(0, min(2, math.floor(float(instance[2]))))

    if concept_id == 1: return 1 if (size == 0 and color == 0) else 0
    if concept_id == 2: return 1 if (color == 1 or shape == 0) else 0
    if concept_id == 3: return 1 if (size == 1 or size == 2) else 0
    return 0

def _get_seeded_model(seed, n_coeffs, bound=1):
    """Helper para criar um 'modelo' determinístico a partir de uma seed."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-bound, bound, n_coeffs)

def get_sine_label(instance: np.ndarray, concept_params: Dict[str, Any]) -> int:
    """Simula a lógica do gerador Sine, usando 'type' e 'threshold' do config."""
    x1, x2 = instance[0], instance[1]
    
    # CORREÇÃO: Usa os parâmetros do dicionário, com valores padrão se não forem encontrados
    concept_type = concept_params.get('type', 'sum')
    threshold = concept_params.get('threshold', 1.0) # Agora usa o threshold do config!
    
    if concept_type == 'prod':
        return 1 if math.sin(x1) * math.sin(x2) > threshold else 0
    else: # 'sum'
        return 1 if math.sin(x1) + math.sin(x2) > threshold else 0
    
def get_hyperplane_label(instance: np.ndarray, concept_params: Dict[str, Any]) -> int:
    seed = concept_params.get('seed', 42)
    mag_change = concept_params.get('mag_change', 0.0)
    n_features = len(instance)
    weights = _get_seeded_model(seed, n_features)
    weights[0] += mag_change # Simula a mudança de magnitude
    
    return 1 if np.dot(weights, instance) > 0 else 0

def get_randomtree_label(instance: np.ndarray, concept_params: Dict[str, Any]) -> int:
    seed = concept_params.get('seed_tree', 42)
    # Simula uma regra de árvore simples e determinística baseada na seed
    coeffs = _get_seeded_model(seed, 2)
    return 1 if instance[int(coeffs[0] * 5) % len(instance)] > coeffs[1] else 0

def get_led_label(instance: np.ndarray, concept_params: Dict[str, Any]) -> int:
    noise = concept_params.get('noise_percentage', 0.0)
    # Simulação simplificada da lógica do LED
    base_val = sum(int(d > 0.5) for d in instance[:7])
    if np.random.rand() < noise:
        return np.random.randint(0, 10)
    return base_val % 10

def get_waveform_label(instance: np.ndarray, concept_params: Dict[str, Any]) -> int:
    has_noise = concept_params.get('has_noise', False)
    # Simulação simplificada
    h1 = _get_seeded_model(1, len(instance))
    h2 = _get_seeded_model(2, len(instance))
    val = np.dot(h1, instance) + np.dot(h2, instance**2)
    if has_noise:
        val += np.random.normal(0, 0.1)
    return int(abs(val * 2) % 3)

def get_rbf_label(instance: np.ndarray, concept_params: Dict[str, Any]) -> int:
    """
    CORRIGIDO: Simula diferentes conceitos RBF de forma determinística com base na seed.
    """
    # 1. Pega a seed do conceito específico (ex: 42 para 'c1', 84 para 'c2_severe')
    seed = concept_params.get('seed_model', 42)
    n_features = len(instance)
    
    # 2. Usa a seed para criar um gerador de números aleatórios determinístico.
    #    Isso garante que para a mesma seed, os centróides serão sempre os mesmos.
    rng = np.random.RandomState(seed)
    
    # 3. Gera um conjunto único de centróides e classes com base na seed.
    #    Para a simulação, vamos usar 4 centróides e 2 classes.
    n_centroids = 4
    n_classes = 2
    centroids = rng.rand(n_centroids, n_features)
    centroid_classes = rng.randint(0, n_classes, n_centroids)
    
    # 4. Encontra o centróide mais próximo e retorna sua classe.
    #    Como os centróides são diferentes para cada seed, a fronteira de decisão muda.
    dists = [np.sum((instance - c)**2) for c in centroids]
    return centroid_classes[np.argmin(dists)]

# O dicionário CONCEPT_FUNCTIONS já deve ter a entrada para RBF, então não precisa mudar.
CONCEPT_FUNCTIONS: Dict[str, Callable] = {
    "SEA": get_sea_label,
    "AGRAWAL": get_agrawal_label,
    "STAGGER": get_stagger_label,
    "SINE": get_sine_label,
    "RBF": get_rbf_label, # Esta chave agora aponta para a função corrigida
    "HYPERPLANE": get_hyperplane_label,
    "RANDOMTREE": get_randomtree_label,
    "LED": get_led_label,
    "WAVEFORM": get_waveform_label,
}

def load_config(config_path: str = 'config.yaml') -> Dict:
    if not os.path.exists(config_path): raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    if 'drift_analysis' not in config: raise KeyError("'drift_analysis' section not found.")
    logger.info(f"Configuration loaded from {config_path}")
    return config['drift_analysis']

def uniform_sample_space(n_samples: int, feature_bounds: List[List[float]], n_features: int) -> np.ndarray:
    """Gera amostras aleatórias respeitando o n_features, mesmo que a lista de bounds seja menor."""
    samples = np.zeros((n_samples, n_features))
    for i in range(n_features):
        bound_idx = min(i, len(feature_bounds) - 1)
        min_val, max_val = feature_bounds[bound_idx]
        samples[:, i] = np.random.uniform(min_val, max_val, n_samples)
    return samples

def calculate_difference(concept_func: Callable, params_a: Dict, params_b: Dict, n_samples: int, feature_bounds: List, n_features: int) -> float:
    """Calcula a diferença percentual entre dois conceitos."""
    samples = uniform_sample_space(n_samples, feature_bounds, n_features)
    total_diff = 0
    for i in range(n_samples):
        instance = samples[i, :]
        try:
            label_a = concept_func(instance, params_a)
            label_b = concept_func(instance, params_b)
            total_diff += abs(label_a - label_b)
        except Exception:
            # Ignora erros de cálculo em uma única instância para não parar a análise
            continue
    return (total_diff / n_samples) * 100.0

def format_results_table(results: Dict[str, Dict[Tuple[str, str], float]]) -> str:
    # ... (código existente, sem alterações)
    output_str = "\n--- Concept Difference Percentage Table (Text) ---"
    for dataset_name, dataset_results in results.items():
        output_str += f"\n\nDataset: {dataset_name}\n"
        if not dataset_results: continue
        involved_concepts = sorted(list(set(itertools.chain.from_iterable(dataset_results.keys()))))
        if not involved_concepts: continue
        max_len = max(len(c) for c in involved_concepts) if involved_concepts else 4
        col_width = max(max_len + 2, 8)
        header = "".join(f"{c:<{col_width}}" for c in involved_concepts)
        output_str += f"{'':<{col_width}}{header}\n"
        for i, concept_a in enumerate(involved_concepts):
            row = f"{concept_a:<{col_width}}"
            for j, concept_b in enumerate(involved_concepts):
                if i == j: row += f"{'-':<{col_width}}"
                else:
                    pair_key = tuple(sorted((concept_a, concept_b)))
                    diff_val = dataset_results.get(pair_key)
                    if diff_val is not None: row += f"{diff_val:.1f}%".ljust(col_width)
                    else: row += f"{'':<{col_width}}"
            output_str += f"{row}\n"
    return output_str

def plot_difference_heatmap(dataset_name: str, dataset_results: Dict[Tuple[str, str], float], save_dir: str):
    # ... (código existente, sem alterações)
    logger.info(f"Generating concept difference heatmap for {dataset_name}...")
    involved_concepts = sorted(list(set(itertools.chain.from_iterable(dataset_results.keys()))))
    df_heatmap = pd.DataFrame(index=involved_concepts, columns=involved_concepts, dtype=float)
    for concept_a in involved_concepts:
        for concept_b in involved_concepts:
            if concept_a == concept_b: df_heatmap.loc[concept_a, concept_b] = 0.0
            else:
                pair_key = tuple(sorted((concept_a, concept_b)))
                df_heatmap.loc[concept_a, concept_b] = dataset_results.get(pair_key)
    plt.figure(figsize=(max(8, len(involved_concepts)*0.8), max(6, len(involved_concepts)*0.7)))
    sns.heatmap(df_heatmap, annot=True, fmt=".1f", cmap="viridis", cbar_kws={'label': 'Percentage Difference (%)'})
    plt.title(f"Concept Difference Heatmap: {dataset_name}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ConceptDifference_Heatmap_{dataset_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Heatmap saved to {save_path}")

def convert_results_for_json(results_dict: Dict) -> Dict:
    # ... (código existente, sem alterações)
    serializable_results = {}
    for dataset_name, data in results_dict.items():
        if isinstance(data, dict):
            serializable_data = {f"{k[0]}_vs_{k[1]}": v for k, v in data.items()}
            serializable_results[dataset_name] = serializable_data
    return serializable_results

def main(config_path: str = 'config.yaml'):
    """Função principal refatorada para analisar apenas datasets configurados para isso."""
    config = load_config(config_path)
    n_samples = config.get('severity_samples', 10000)
    datasets_config = config.get('datasets', {})
    results = {}

    for dataset_name, ds_config in datasets_config.items():
        required_keys = ['concepts', 'pairs_to_compare', 'feature_bounds', 'n_features', 'class']
        if not all(key in ds_config for key in required_keys):
            logger.info(f"Skipping '{dataset_name}': Not configured for concept difference analysis.")
            continue

        logger.info(f"--- Analyzing Dataset: {dataset_name} ---")
        
        concept_func = CONCEPT_FUNCTIONS.get(dataset_name.upper())
        if not concept_func:
            logger.warning(f"No label function implemented for '{dataset_name}' in CONCEPT_FUNCTIONS. Skipping.")
            continue
            
        results[dataset_name] = {}
        for pair in ds_config['pairs_to_compare']:
            concept_id_a, concept_id_b = str(pair[0]), str(pair[1])
            params_a = ds_config['concepts'][concept_id_a]
            params_b = ds_config['concepts'][concept_id_b]
            
            diff = calculate_difference(
                concept_func=concept_func,
                params_a=params_a,
                params_b=params_b,
                n_samples=n_samples,
                feature_bounds=ds_config['feature_bounds'],
                n_features=ds_config['n_features']
            )
            results[dataset_name][tuple(sorted(pair))] = diff
            logger.info(f"-> Difference {dataset_name} [{concept_id_a} vs {concept_id_b}]: {diff:.2f}%")
    
    # Salvar resultados
    table_string = format_results_table(results)
    print(table_string)
    heatmap_save_dir = config.get("heatmap_save_directory", "results")
    for dataset_name, data in results.items():
        plot_difference_heatmap(dataset_name, data, heatmap_save_dir)
    
    json_path = os.path.join(heatmap_save_dir, "concept_differences.json")
    with open(json_path, 'w') as f:
        json.dump(convert_results_for_json(results), f, indent=2)
    logger.info(f"Numerical results saved to {json_path}")


if __name__ == "__main__":
    main()