# utils.py (Corrigido - UnboundLocalError em classify_performance)

import random
import copy
import numpy as np
import logging
from rule_node import Node # Required for tree operations
# Can be added to main.py or a utils file
import math
from typing import List, Tuple, Dict, Any
import pandas as pd
import Levenshtein
from sklearn.model_selection import train_test_split

# Importa operadores específicos para o helper _generate_valid_leaf_for_pruning
from constants import (
    NUMERIC_COMPARISON_OPERATORS,
    CATEGORICAL_COMPARISON_OPERATORS,
    ALL_COMPARISON_OPERATORS # Pode ser usado para validação geral
)


def update_historical_reference_dataset(
    historical_data: List[Tuple[Dict, Any]],
    new_chunk_data: List[Dict],
    new_chunk_target: List[Any],
    max_size: int = 500,
    random_state: int = 42
) -> List[Tuple[Dict, Any]]:
    """
    Atualiza um dataset de referência histórico com uma amostra estratificada de novos dados.

    Args:
        historical_data: A lista atual de amostras históricas (tuplas de (X, y)).
        new_chunk_data: Os dados (X) do novo chunk.
        new_chunk_target: Os rótulos (y) do novo chunk.
        max_size: O tamanho máximo desejado para o dataset de referência.
        random_state: Seed para reprodutibilidade da amostragem.

    Returns:
        Uma nova lista de tuplas (X, y) representando o dataset de referência atualizado.
    """
    if not new_chunk_data:
        return historical_data

    # Converte os novos dados para um DataFrame para facilitar a amostragem estratificada
    df_new = pd.DataFrame(new_chunk_data)
    df_new['class'] = new_chunk_target

    # Combina com os dados históricos, se existirem
    if historical_data:
        X_hist, y_hist = zip(*historical_data)
        df_hist = pd.DataFrame(list(X_hist))
        df_hist['class'] = list(y_hist)
        df_combined = pd.concat([df_hist, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # Realiza a amostragem estratificada para manter o dataset com o tamanho máximo
    # A estratificação garante que classes raras sejam mantidas, se possível.
    if len(df_combined) > max_size:
        # Tenta usar a amostragem estratificada. Se falhar (ex: classes com 1 membro), usa a amostragem aleatória simples.
        try:
            _, df_sample = train_test_split(df_combined, test_size=max_size, stratify=df_combined['class'], random_state=random_state)
        except ValueError:
            df_sample = df_combined.sample(n=max_size, random_state=random_state)
    else:
        df_sample = df_combined

    # Converte o DataFrame de amostra de volta para o formato de lista de tuplas
    y_sample = df_sample.pop('class').tolist()
    X_sample = df_sample.to_dict('records')

    updated_reference_dataset = list(zip(X_sample, y_sample))
    logging.info(f"Dataset de referência histórico atualizado. Tamanho atual: {len(updated_reference_dataset)} instâncias.")
    return updated_reference_dataset


def calculate_population_diversity(population: List[Any], sample_size: int = 20) -> float:
    """
    Calcula a diversidade da população com base na distância de Levenshtein
    normalizada entre as representações em string de uma amostra de indivíduos.

    Retorna um score de 0.0 (todos idênticos) a 1.0 (muito diferentes).
    """
    if not population or len(population) < 2:
        return 0.0

    # Pega uma amostra da população para evitar comparações N^2
    sample_indices = np.random.choice(len(population), min(len(population), sample_size * 2), replace=False)
    
    dissimilarity_scores = []
    
    # Compara pares da amostra
    for i in range(0, len(sample_indices) - 1, 2):
        ind1 = population[sample_indices[i]]
        ind2 = population[sample_indices[i+1]]

        # Gera uma representação em string para cada indivíduo
        str1 = ind1.get_rules_as_string()
        str2 = ind2.get_rules_as_string()
        
        if not str1 and not str2:
            continue

        dist = Levenshtein.distance(str1, str2)
        max_len = max(len(str1), len(str2), 1)
        normalized_dissimilarity = dist / max_len
        dissimilarity_scores.append(normalized_dissimilarity)

    if not dissimilarity_scores:
        return 0.0

    return np.mean(dissimilarity_scores)

# --- Funções de Manipulação de Árvore ---

def node_to_string(node):
    # ... (código como antes) ...
    if node is None: return "None";
    if node.is_leaf(): value_str = f"{node.value:.4f}" if isinstance(node.value, (int, float)) and node.feature_type=='numeric' else str(node.value); return f"({node.attribute} {node.operator} {value_str})"
    elif node.is_internal(): left_expr = node_to_string(node.left); right_expr = node_to_string(node.right); return f"({left_expr} {node.operator} {right_expr})"
    else: return "(Unknown Node Type)"

def collect_all_nodes(node, nodes_list):
    # ... (código como antes) ...
    if node is not None: nodes_list.append(node); collect_all_nodes(node.left, nodes_list); collect_all_nodes(node.right, nodes_list)

def select_random_node(root_node):
    # ... (código como antes) ...
    nodes = []; collect_all_nodes(root_node, nodes);
    if not nodes: return root_node
    return random.choice(nodes)

def swap_subtrees(node1, node2):
    # ... (código como antes, incluindo feature_type) ...
    node1.attribute, node2.attribute = node2.attribute, node1.attribute; node1.operator, node2.operator = node2.operator, node1.operator; node1.value, node2.value = node2.value, node1.value; node1.left, node2.left = node2.left, node1.left; node1.right, node2.right = node2.right, node1.right; node1.feature_type, node2.feature_type = node2.feature_type, node1.feature_type

def get_tree_depth(node):
    # ... (código como antes) ...
    if node is None: return 0;
    if node.is_leaf(): return 1
    return 1 + max(get_tree_depth(node.left), get_tree_depth(node.right))

# Helper para prune_tree_to_depth
def _generate_valid_leaf_for_pruning(attributes, value_ranges, category_values, categorical_features):
    # ... (código como antes, usando tipos de feature) ...
    if not attributes: logging.warning("Attrs empty for pruning leaf."); return Node(attribute="dummy_pruned_attr", operator="<=", value=0, feature_type='numeric')
    attribute = random.choice(attributes); operator, value, feature_type = None, None, None
    if attribute in categorical_features:
        feature_type = 'categorical'; operator = random.choice(CATEGORICAL_COMPARISON_OPERATORS); possible_values = category_values.get(attribute)
        valid_cat_values = [v for v in possible_values if v is not None] if possible_values else []
        if valid_cat_values: value = random.choice(valid_cat_values)
        else: logging.warning(f"No values for cat '{attribute}' during pruning."); value = "MISSING"; operator = "!="
    else: # Assume numérico
        feature_type = 'numeric'; operator = random.choice(NUMERIC_COMPARISON_OPERATORS); min_val, max_val = value_ranges.get(attribute, (0, 0))
        if min_val >= max_val: value = min_val
        else: value = random.uniform(min_val, max_val)
    leaf_node = Node(attribute=attribute, operator=operator, value=value, feature_type=feature_type)
    is_valid = (leaf_node.attribute is not None and leaf_node.operator in ALL_COMPARISON_OPERATORS and leaf_node.value is not None and leaf_node.feature_type in ['numeric', 'categorical'])
    if feature_type == 'numeric' and operator not in NUMERIC_COMPARISON_OPERATORS: logging.error(f"Inconsistency: Num feat '{attribute}' got op '{operator}'. Fixing."); operator = random.choice(NUMERIC_COMPARISON_OPERATORS); leaf_node.operator = operator; is_valid=False
    elif feature_type == 'categorical' and operator not in CATEGORICAL_COMPARISON_OPERATORS: logging.error(f"Inconsistency: Cat feat '{attribute}' got op '{operator}'. Fixing."); operator = random.choice(CATEGORICAL_COMPARISON_OPERATORS); leaf_node.operator = operator; is_valid=False
    if not is_valid: logging.critical(f"Invalid leaf node after prune gen: {node_to_string(leaf_node)}"); return Node(attribute=attribute, operator="<=", value=0, feature_type='numeric')
    return leaf_node

# Função de Poda Refatorada
def prune_tree_to_depth(node, max_depth, attributes, value_ranges, category_values, categorical_features, current_depth=1):
    """ Recursively prunes a tree, handling feature types correctly when creating leaves. """
    # ... (código como antes) ...
    if node is None: return None
    if current_depth >= max_depth:
        if node.is_leaf(): return node
        else: logging.debug(f"Pruning internal node {node_to_string(node)} at depth {current_depth}."); return _generate_valid_leaf_for_pruning(attributes, value_ranges, category_values, categorical_features)
    if node.is_internal():
        node.left = prune_tree_to_depth(node.left, max_depth, attributes, value_ranges, category_values, categorical_features, current_depth + 1)
        node.right = prune_tree_to_depth(node.right, max_depth, attributes, value_ranges, category_values, categorical_features, current_depth + 1)
        if node.left is None and node.right is None: logging.debug(f"Both children None after prune. Converting parent {node_to_string(node)} to leaf."); return _generate_valid_leaf_for_pruning(attributes, value_ranges, category_values, categorical_features)
        elif node.left is None: logging.debug(f"Left child None after prune. Promoting right: {node_to_string(node.right)}"); return node.right
        elif node.right is None: logging.debug(f"Right child None after prune. Promoting left: {node_to_string(node.left)}"); return node.left
    return node

# --- Outras Funções Utilitárias ---

# --- Função Corrigida ---
def classify_performance(current_accuracy: float, 
                         historical_accuracies: list, 
                         absolute_bad_threshold: float = 0.3, # <<< NOVO PARÂMETRO com default
                         window_size: int = 5): # <<< ADICIONADO window_size como parâmetro
    """
    Classifies current performance ('good', 'medium', 'bad') based on historical data
    and an absolute threshold for 'bad'. Uses a sliding window for historical stats.
    """
    # Log de entrada para depuração
    # logging.debug(f"classify_performance called with: current_accuracy={current_accuracy:.4f}, historical_len={len(historical_accuracies)}, abs_bad_threshold={absolute_bad_threshold:.4f}, window_size={window_size}")

    # 1. Checagem do Limiar Absoluto para 'bad'
    if current_accuracy < absolute_bad_threshold: # [utils.py]
        logging.debug(f"Performance classified as 'bad' due to absolute threshold: {current_accuracy:.4f} < {absolute_bad_threshold:.4f}") # [utils.py]
        return 'bad' # [utils.py]

    # 2. Checagem de Histórico Insuficiente para análise estatística
    # Pelo menos 2 pontos na janela são necessários para std dev.
    # A janela em si precisa de pelo menos `window_size` pontos para ser efetiva,
    # mas para std, bastam 2.
    effective_history_for_stats = historical_accuracies[-window_size:] if len(historical_accuracies) > window_size else historical_accuracies # [utils.py]
    
    if len(effective_history_for_stats) < 2: # [utils.py]
        # Se não é "absolutamente ruim" e o histórico é muito curto para estatística,
        # considera 'medium' para evitar pessimismo prematuro.
        logging.debug(f"Performance classified as 'medium' due to insufficient recent history (len: {len(effective_history_for_stats)}) for statistical analysis (and not meeting absolute_bad_threshold).") # [utils.py]
        return 'medium' # [utils.py]

    # 3. Cálculo Estatístico com Janela Deslizante
    mean_acc = np.mean(effective_history_for_stats) # [utils.py]
    std_acc = np.std(effective_history_for_stats) # [utils.py]
    std_acc = max(std_acc, 1e-9)  # type: ignore # Evitar desvio padrão zero ou muito pequeno [utils.py]

    lower_bound = mean_acc - std_acc # [utils.py]
    upper_bound = mean_acc + std_acc # [utils.py]
    
    logging.debug(f"Performance classification stats: current_acc={current_accuracy:.4f}, mean_recent_hist={mean_acc:.4f}, std_recent_hist={std_acc:.4f}, lower_b={lower_bound:.4f}, upper_b={upper_bound:.4f}") # [utils.py]

    # 4. Classificação Final com Base Estatística
    if current_accuracy < lower_bound: # [utils.py]
        return 'bad' # [utils.py]
    elif current_accuracy > upper_bound: # [utils.py]
        return 'good' # [utils.py]
    else: # [utils.py]
        return 'medium' # [utils.py]
# --- Fim da Correção ---

def compute_features_distance(used_features, reference_features):
    # ... (código como antes) ...
    if not reference_features: return 0.0; used_features = set(used_features); reference_features = set(reference_features);
    if not used_features and not reference_features: return 0.0; union_set = used_features.union(reference_features); union_size = len(union_set)
    if union_size == 0: return 0.0; symmetric_diff_size = len(used_features.symmetric_difference(reference_features)) # type: ignore
    return symmetric_diff_size / (union_size + 1e-9) # type: ignore

def evaluate_chunk_periodically(
    ruleset: Any, # Your trained model/ruleset for this chunk transition
    test_X: List[Dict[Any, Any]], # List of instance dictionaries for the test chunk
    test_y: List[Any],           # List of true labels for the test chunk
    evaluation_period: int,      # How often to record accuracy (k)
    global_instance_offset: int  # Total instances processed in previous test phases
    ) -> List[Tuple[int, float]]:
    """
    Evaluates a model on a test chunk instance by instance and records
    accuracy periodically.

    Args:
        ruleset: The trained model/ruleset to evaluate.
        test_X: Feature dictionaries for the test chunk.
        test_y: True labels for the test chunk.
        evaluation_period (k): Record accuracy every k instances.
        global_instance_offset: The starting global instance count for this chunk.

    Returns:
        List of (global_instance_count, accuracy_over_period) tuples.
    """
    periodic_results = []
    correct_in_period = 0
    instances_in_period = 0
    total_instances_in_chunk = len(test_X)

    if evaluation_period <= 0:
        logging.error("Evaluation period must be positive.")
        return []
    if total_instances_in_chunk == 0:
         logging.warning("Test chunk is empty, skipping periodic evaluation.")
         return []

    # # --- Assume you have a function like this ---
    # # Needs to be adapted based on how your ruleset works
    # def predict_instance(rules: Any, instance: Dict) -> Any:
    #      # This is a placeholder - replace with your actual prediction logic
    #      # It might involve calling individual.predict(instance) or rule_tree.evaluate(instance)
    #      # and combining results based on your Individual/RuleSet structure.
    #      try:
    #          # Example: If ruleset is an 'Individual' object with a predict method
    #          if hasattr(rules, 'predict') and callable(rules.predict):
    #               return rules.predict(instance)
    #          else:
    #               # Add other ways your model might predict
    #               logging.warning("Prediction logic not fully implemented in placeholder.")
    #               return 0 # Default prediction
    #      except Exception as e:
    #          logging.error(f"Error during prediction for instance {instance}: {e}")
    #          return 0 # Default prediction on error
    # # --- End of assumed function ---
    def predict_instance(rules: Any, instance: Dict) -> Any:
         """Placeholder - REPLACE with logic calling your specific prediction method."""
         try:
             # <<< CORRIGIDO: Prioriza o método _predict >>>
             if hasattr(rules, '_predict') and callable(rules._predict):
                  return rules._predict(instance) # Chama o método _predict do Individual/Ruleset
             elif hasattr(rules, 'predict') and callable(rules.predict):
                  # Fallback se existir um método 'predict' público
                  return rules.predict(instance)
             # <<< FIM CORRIGIDO >>>
             else:
                  # Se nenhum método for encontrado, loga e retorna default
                  # Isso deve ser evitado garantindo que 'rules' sempre tenha _predict
                  logging.error(f"Ruleset object of type {type(rules)} lacks a usable prediction method (_predict or predict).")
                  return 0 # Default prediction
         except Exception as e:
             logging.error(f"Error during prediction for instance {instance}: {e}")
             return 0 # Default prediction on error


    logging.info(f"Starting periodic evaluation. Period size: {evaluation_period}")
    for i in range(total_instances_in_chunk):
        instance_dict = test_X[i]
        true_label = test_y[i]

        # 1. Predict
        predicted_label = predict_instance(ruleset, instance_dict)

        # 2. Check correctness
        if predicted_label == true_label:
            correct_in_period += 1

        instances_in_period += 1

        # 3. Record periodically or at the end of the chunk
        if instances_in_period == evaluation_period or (i == total_instances_in_chunk - 1):
            if instances_in_period > 0: # Avoid division by zero if period is short/empty
                accuracy = correct_in_period / instances_in_period
                global_instance_count = global_instance_offset + (i + 1)
                periodic_results.append((global_instance_count, accuracy))
                logging.debug(f"Instance {global_instance_count}: Accuracy over last {instances_in_period} instances = {accuracy:.4f}")

                # Reset counters for the next period
                correct_in_period = 0
                instances_in_period = 0
            else:
                 logging.warning("Period ended with zero instances processed.")


    return periodic_results