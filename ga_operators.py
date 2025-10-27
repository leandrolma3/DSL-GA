# ga_operators.py (Corrigido - Mutação de Operador/Valor Categórico)
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text # Para treinar e potencialmente visualizar
from rule_tree import RuleTree # Importe sua classe RuleTree [ga_operators.py]
from rule_node import Node # Importe sua classe Node [ga_operators.py]
import random
import copy
import logging
from typing import Dict, Any, List, Set, Optional
# Import necessary classes and functions
from individual import Individual
from rule_tree import RuleTree
from river import tree
from river.tree.nodes import branch
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Importa operadores específicos para mutação correta
from constants import (
    LOGICAL_OPERATORS,
    NUMERIC_COMPARISON_OPERATORS,
    CATEGORICAL_COMPARISON_OPERATORS,
    ALL_COMPARISON_OPERATORS # Usado apenas se tipo for desconhecido
)
from utils import (
    select_random_node,
    swap_subtrees,
    get_tree_depth,
    prune_tree_to_depth,
    node_to_string
)
# fitness não é importado aqui diretamente

def _extract_all_rules_from_dt(dt_model: DecisionTreeClassifier, feature_names: List[str], base_attributes_info: Dict) -> Dict[int, List[Dict[str, Any]]]:
    """
    Percorre uma Decision Tree e extrai TODOS os caminhos (regras),
    calculando um score para cada um baseado em pureza e suporte.
    Retorna um dicionário de regras com seus scores.
    """
    tree = dt_model.tree_
    gene_pool = {c: [] for c in dt_model.classes_}
    
    def recurse(node_id, conditions):
        if tree.children_left[node_id] == -1: # É uma folha
            class_distribution = tree.value[node_id][0]
            total_samples = tree.n_node_samples[node_id]
            if total_samples == 0 or not conditions: return

            purity = np.max(class_distribution) / total_samples
            predicted_class_idx = np.argmax(class_distribution)
            predicted_class = dt_model.classes_[predicted_class_idx]
            
            if len(conditions) == 1: rule_root = conditions[0]
            else:
                rule_root = Node(operator="AND", left=conditions[0], right=conditions[1])
                for i in range(2, len(conditions)): rule_root = Node(operator="AND", left=rule_root, right=conditions[i])

            # Remove max_depth de base_attributes_info para evitar conflito
            attrs_without_max_depth = {k: v for k, v in base_attributes_info.items() if k != 'max_depth'}
            rule_tree = RuleTree(max_depth=len(conditions), **attrs_without_max_depth, root_node=rule_root)
            if rule_tree.is_valid_rule():
                score = purity * np.log1p(total_samples)
                gene_pool[predicted_class].append({'rule': rule_tree, 'score': score})
            return

        feature_idx = tree.feature[node_id]
        feature = feature_names[feature_idx]
        threshold = tree.threshold[node_id]
        
        left_cond = Node(attribute=feature, operator="<=", value=threshold, feature_type='numeric')
        recurse(tree.children_left[node_id], conditions + [left_cond])
        
        right_cond = Node(attribute=feature, operator=">", value=threshold, feature_type='numeric')
        recurse(tree.children_right[node_id], conditions + [right_cond])

    recurse(0, [])
    return gene_pool


def _create_specialist_rule(class_label: Any, data: List[Dict], target: List, base_attributes_info: Dict) -> RuleTree | None:
    """
    "Terapia Gênica": Cria uma nova regra de alta qualidade para uma classe-alvo.
    Treina uma DT especialista binária e extrai sua melhor regra.
    """
    try:
        # 1. Cria o dataset binário para "Focus Training"
        y_binary = [1 if y == class_label else 0 for y in target]
        if sum(y_binary) < 5: return None

        df_train = pd.DataFrame(data)
        
        # 2. Treina uma DT especialista rasa
        specialist_dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=random.randint(0,10000), class_weight='balanced')
        specialist_dt.fit(df_train, y_binary)
        
        # 3. Extrai todas as regras que preveem a classe positiva (1)
        gene_pool_binary = _extract_all_rules_from_dt(specialist_dt, list(df_train.columns), base_attributes_info)
        positive_rules_with_scores = gene_pool_binary.get(1, [])
        
        if not positive_rules_with_scores: return None
        
        # 4. Encontra a melhor regra com base no score (pureza * log(suporte))
        best_rule_info = max(positive_rules_with_scores, key=lambda x: x['score'])
        
        logging.debug(f"  Terapia Gênica: Nova regra para a classe {class_label} extraída com score {best_rule_info['score']:.2f}.")
        return best_rule_info['rule']

    except Exception as e:
        logging.error(f"  Erro durante a 'Terapia Gênica': {e}", exc_info=True)
        return None

# --- Função Auxiliar para a "Marcha 1" (Exploração) ---
def _crossover_modo_expansao(parent1: Individual, parent2: Individual, max_depth: int,
                             attributes: list, value_ranges: dict, category_values: dict,
                             categorical_features: set, classes: list, max_rules_per_class: int) -> dict:
    """
    Usa o crossover de nó para fazer uma exploração agressiva, criando
    novas estruturas de regras a partir dos pais.
    Retorna um dicionário de regras para o filho.
    """
    child_rules_dict = {c: [] for c in classes}
    for class_label in classes:
        rules1 = parent1.rules.get(class_label, [])
        rules2 = parent2.rules.get(class_label, [])
        child_rules = []

        if rules1 and rules2:
            rule1, rule2 = random.choice(rules1), random.choice(rules2)
            r1_copy, r2_copy = copy.deepcopy(rule1), copy.deepcopy(rule2)
            node1, node2 = select_random_node(r1_copy.root), select_random_node(r2_copy.root)
            
            if node1 and node2:
                swap_subtrees(node1, node2)
            
            prune_tree_to_depth(r1_copy.root, max_depth, attributes, value_ranges, category_values, categorical_features)
            prune_tree_to_depth(r2_copy.root, max_depth, attributes, value_ranges, category_values, categorical_features)
            child_rules.extend([r1_copy, r2_copy])

        potential_rules = rules1 + rules2
        random.shuffle(potential_rules)
        while len(child_rules) < max_rules_per_class and potential_rules:
            child_rules.append(copy.deepcopy(potential_rules.pop(0)))

        child_rules_dict[class_label] = child_rules[:max_rules_per_class]
    return child_rules_dict

# --- Função Auxiliar para a "Marcha 2" (Refinamento) ---
def _crossover_modo_refinamento(parent1: Individual, parent2: Individual, classes: list) -> dict:
    """
    Usa o crossover de regra direcionado pela qualidade. Cria um filho a partir do Pai 1,
    substituindo suas regras de pior qualidade pelas de melhor qualidade do Pai 2.
    Retorna um dicionário de regras para o filho.
    """
    child_rules_dict = copy.deepcopy(parent1.rules)

    for class_label in classes:
        rules_child = child_rules_dict.get(class_label, [])
        rules_parent2 = parent2.rules.get(class_label, [])

        if not rules_child or not rules_parent2:
            continue

        worst_rule_child = min(rules_child, key=lambda r: r.quality_score)
        best_rule_parent2 = max(rules_parent2, key=lambda r: r.quality_score)

        if best_rule_parent2.quality_score > worst_rule_child.quality_score:
            idx_to_replace = rules_child.index(worst_rule_child)
            child_rules_dict[class_label][idx_to_replace] = copy.deepcopy(best_rule_parent2)
            logging.debug(f"  Crossover de Refinamento: Regra para classe {class_label} substituída.")
            
    return child_rules_dict

# --- ORQUESTRADOR PRINCIPAL (Substitua sua função crossover por esta) ---
def crossover(
    parent1: Individual, parent2: Individual, max_depth: int,
    attributes: list, value_ranges: dict, category_values: dict,
    categorical_features: set, classes: list, max_rules_per_class: int,
    **kwargs # Captura argumentos não utilizados como crossover_type
) -> Individual:
    """
    Orquestrador do Crossover Adaptativo.
    Verifica a "maturidade" dos pais e escolhe o modo de operação apropriado.
    """
    child = Individual(max_rules_per_class, max_depth, attributes, value_ranges,
                       category_values, categorical_features, classes)
    
    total_classes = len(classes)
    parent1_is_complete = len(parent1.covered_classes) == total_classes
    parent2_is_complete = len(parent2.covered_classes) == total_classes

    # Se ambos os pais forem "completos", entra em MODO REFINAMENTO (Marcha 2)
    if parent1_is_complete and parent2_is_complete:
        logging.debug("Crossover: Ambos os pais completos. Ativando MODO REFINAMENTO.")
        child.rules = _crossover_modo_refinamento(parent1, parent2, classes)
    # Caso contrário, entra em MODO EXPANSÃO (Marcha 1)
    else:
        logging.debug("Crossover: Pelo menos um pai incompleto. Ativando MODO EXPANSÃO (crossover de nó).")
        child.rules = _crossover_modo_expansao(parent1, parent2, max_depth, attributes, value_ranges,
                                              category_values, categorical_features, classes, max_rules_per_class)

    # Finaliza o filho
    child.default_class = parent1.default_class if random.random() < 0.5 else parent2.default_class
    child.remove_duplicate_rules()
    return child

# Em ga_operators.py (pode adicionar ao final do arquivo)

def mutate_hill_climbing(individual: Individual, value_ranges: dict, perturbation_factor=0.05) -> Individual:
    """
    Aplica uma mutação de baixo impacto (Hill Climbing) a um indivíduo.
    Ela perturba levemente os valores numéricos nas condições das regras.

    Args:
        individual: O indivíduo a ser mutado.
        value_ranges: Dicionário com os limites min/max de cada atributo.
        perturbation_factor: A porcentagem máxima de alteração do valor (ex: 0.05 para 5%).

    Returns:
        Um novo indivíduo mutante.
    """
    mutant = copy.deepcopy(individual)
    
    # Reúne todas as regras do indivíduo em uma única lista
    all_rules = [rule for rule_list in mutant.rules.values() for rule in rule_list]
    if not all_rules:
        return mutant # Retorna o original se não houver regras

    # Seleciona uma regra e um nó aleatórios para mutar
    rule_to_tweak = random.choice(all_rules)
    node_to_tweak = select_random_node(rule_to_tweak.root)

    # Aplica a mutação apenas se for um nó folha com atributo numérico
    if node_to_tweak and node_to_tweak.is_leaf() and node_to_tweak.feature_type == 'numeric':
        attribute = node_to_tweak.attribute
        current_value = node_to_tweak.value
        
        # Calcula um pequeno delta de perturbação
        delta = (value_ranges[attribute][1] - value_ranges[attribute][0]) * perturbation_factor
        perturbation = random.uniform(-delta, delta)
        new_value = current_value + perturbation

        # Garante que o novo valor permaneça dentro dos limites do atributo
        min_val, max_val = value_ranges[attribute]
        new_value = max(min_val, min(new_value, max_val))
        
        node_to_tweak.value = new_value
        logging.debug(f"  Hill Climbing: Atributo '{attribute}' valor alterado de {current_value:.4f} para {new_value:.4f}")

    return mutant

def _find_pure_leaf_path(
    start_node, 
    target_class, 
    purity_threshold=0.85
) -> Optional[List[int]]:
    """
    Realiza uma busca em profundidade (DFS) para encontrar um caminho da raiz
    até uma folha que seja 'pura' o suficiente para a classe alvo.
    
    Retorna:
        Uma lista de índices de filhos [0, 1, 0, ...] que representa o caminho, ou None.
    """
    # A pilha armazena (nó_atual, caminho_até_aqui)
    stack = [(start_node, [])]

    while stack:
        current_node, path = stack.pop()

        # Se for uma folha, verifica a pureza
        if not (hasattr(current_node, 'children') and current_node.children):
            if not current_node.stats:
                continue
            
            total_weight = sum(current_node.stats.values())
            target_weight = current_node.stats.get(target_class, 0)
            
            if total_weight > 0:
                purity = target_weight / total_weight
                if purity >= purity_threshold:
                    logging.debug(f"  Encontrado caminho para folha pura (Pureza: {purity:.2f}): {path}")
                    return path # Sucesso! Retorna o caminho.
            continue

        # Se for um galho, adiciona os filhos à pilha para continuar a busca
        if hasattr(current_node, 'children') and len(current_node.children) > 1:
            stack.append((current_node.children[1], path + [1])) # Adiciona a direita primeiro
            stack.append((current_node.children[0], path + [0])) # Adiciona a esquerda (será processada primeiro)
            
    logging.debug(f"  Nenhum caminho para uma folha com pureza >= {purity_threshold} foi encontrado para a classe {target_class}.")
    return None

def _mutate_from_classifier_dt(
    rule_to_replace: RuleTree, 
    class_label: Any, 
    **kwargs
) -> RuleTree:
    """
    Cria uma nova regra de alta qualidade para substituir uma regra ruim.
    Treina uma DT especialista binária e extrai sua melhor regra.
    """
    try:
        # Extrai o contexto necessário dos kwargs
        data, target = kwargs['data'], kwargs['target']
        attributes, base_attributes_info = kwargs['attributes'], kwargs['base_attributes_info']
        
        # 1. Cria o dataset binário para "Focus Training"
        y_binary = [1 if y == class_label else 0 for y in target]
        if sum(y_binary) < 5: return rule_to_replace # Retorna a original se não houver dados
        
        df_train = pd.DataFrame(data)
        
        # 2. Treina uma DT especialista rasa
        specialist_dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=random.randint(0,10000), class_weight='balanced')
        specialist_dt.fit(df_train, y_binary)
        
        # 3. Extrai TODAS as regras que preveem a classe positiva (1)
        gene_pool_binary = _extract_all_rules_from_dt(specialist_dt, attributes, base_attributes_info)
        positive_rules = gene_pool_binary.get(1, [])
        
        if not positive_rules: return rule_to_replace # Retorna a original se nenhuma regra foi extraída
        
        # 4. AVALIA as regras extraídas nos dados binários para encontrar a melhor
        best_rule = None
        best_gmean = -1.0
        
        for rule in positive_rules:
            # Para avaliar a regra, criamos um indivíduo temporário só com essa regra
            temp_ind = Individual(max_rules_per_class=1, max_depth=4, **base_attributes_info, classes=[0, 1], train_target=y_binary, initialize_random_rules=False)
            temp_ind.rules[1] = [rule]
            
            predictions = [temp_ind._predict(inst) for inst in data]
            gmean = metrics.calculate_gmean_contextual(y_binary, predictions, [0, 1])
            
            if gmean > best_gmean:
                best_gmean = gmean
                best_rule = rule
                
        logging.info(f"  Terapia Gênica: Nova regra encontrada para a classe {class_label} com G-mean (binário) de {best_gmean:.4f}")
        return best_rule if best_rule else rule_to_replace

    except Exception as e:
        logging.error(f"  Erro durante a mutação 'from_classifier': {e}", exc_info=True)
        return rule_to_replace 

def _extract_single_rule_from_river_ht(
    ht_model: tree.HoeffdingTreeClassifier,
    target_class: Any,
    base_attributes_info: Dict
) -> Optional[RuleTree]:
    """
    Navega por um caminho em uma Hoeffding Tree e o traduz para uma RuleTree.
    Prioriza caminhos que levam a folhas 'puras' para gerar regras mais úteis.
    """
    if not hasattr(ht_model, '_root'):
        return None

    try:
        # --- LÓGICA APRIMORADA: Busca por um caminho para uma folha pura ---
        path_indices = _find_pure_leaf_path(ht_model._root, target_class)

        # Se nenhum caminho puro for encontrado, não extrai a regra para evitar regras genéricas
        if path_indices is None:
            logging.debug(f"Não foi possível encontrar um caminho promissor para a classe {target_class}. Extração cancelada.")
            return None

        current_node = ht_model._root
        conditions: List[Node] = []

        for child_index in path_indices:
            if not (hasattr(current_node, 'children') and current_node.children):
                break # Chegou a uma folha antes do fim do caminho

            feature = current_node.feature
            
            if isinstance(current_node, branch.NumericBinaryBranch):
                operator = '<=' if child_index == 0 else '>'
                value = current_node.threshold
                ftype = 'numeric'
            
            elif isinstance(current_node, branch.NominalMultiwayBranch):
                operator = '=='
                # A forma correta de obter o valor é pelo índice do galho no splitter do nó.
                # No entanto, a forma mais robusta é parsear a representação do galho.
                # Ex: 'feature = value'
                branch_repr = current_node.repr_branch(child_index)
                parts = branch_repr.split(' = ')
                value = parts[1].strip() if len(parts) > 1 else None
                ftype = 'categorical'

                # O valor pode ser numérico mesmo sendo categórico (ex: PokerHand)
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass # Mantém como string se não for conversível
            else:
                logging.warning(f"Tipo de split desconhecido ({type(current_node)}), extração interrompida.")
                break
            
            if value is None: # Segurança
                logging.warning("Não foi possível extrair o valor para a condição do split nominal.")
                break

            conditions.append(Node(attribute=feature, operator=operator, value=value, feature_type=ftype))
            current_node = current_node.children[child_index]

        if not conditions:
            return None

        # Montagem da RuleTree (lógica como antes)
        if len(conditions) == 1:
            rule_root = conditions[0]
        else:
            rule_root = Node(operator="AND", left=conditions[0], right=conditions[1])
            for i in range(2, len(conditions)):
                new_root = Node(operator="AND", left=rule_root, right=conditions[i])
                rule_root = new_root
        
        final_rule_tree = RuleTree(max_depth=len(conditions), **base_attributes_info)
        final_rule_tree.root = rule_root

        return final_rule_tree if final_rule_tree.is_valid_rule() else None
    
    except Exception as e:
        logging.error(f"Erro inesperado durante extração da regra: {e}", exc_info=True)
        return None
    
def _preprocess_data_for_dt(data_dicts: list, target_list: list, all_attributes: list, categorical_features: set, sample_size: int = 200):
    """
    Prepares data for scikit-learn Decision Tree training.
    Focuses on numeric features for simplicity in rule extraction.
    Returns X_np, y_np, and the ordered list of numeric feature names used.
    """
    if not data_dicts:
        return np.array([]), np.array([]), []

    numeric_attributes_ordered = sorted(list(set(all_attributes) - categorical_features))
    if not numeric_attributes_ordered:
        logging.warning("Preprocess for DT: No numeric attributes found.")
        return np.array([]), np.array([]), []

    actual_sample_size = min(len(data_dicts), sample_size)
    if actual_sample_size == 0:
        return np.array([]), np.array([]), numeric_attributes_ordered
        
    sample_indices = random.sample(range(len(data_dicts)), actual_sample_size)
    
    X_sample_list_of_lists = []
    y_sample_list = []

    for k_idx in sample_indices:
        instance_dict = data_dicts[k_idx]
        feature_vector = []
        valid_instance = True
        for attr_name in numeric_attributes_ordered:
            value = instance_dict.get(attr_name)
            try:
                # Trata o caso de o valor já ser None
                if value is None:
                    numeric_value = 0.0
                else:
                    numeric_value = float(value)
                feature_vector.append(numeric_value)
            except (ValueError, TypeError):
                # Se a conversão para float falhar, é realmente não-numérico
                logging.warning(f"Could not convert value '{value}' to float for numeric attr '{attr_name}'. Using 0.0.")
                feature_vector.append(0.0)
        X_sample_list_of_lists.append(feature_vector)
        y_sample_list.append(target_list[k_idx])
    
    if not X_sample_list_of_lists:
        return np.array([]), np.array([]), numeric_attributes_ordered

    return np.array(X_sample_list_of_lists), np.array(y_sample_list), numeric_attributes_ordered

def _extract_single_rule_from_dt_path(dt_model, path_indices, feature_names_for_dt, target_class_of_rule,
                                     max_rule_depth_extracted, base_attributes_info):
    """
    Extracts a single RuleTree from a specific path in a trained scikit-learn Decision Tree.
    Focuses on numeric features.
    path_indices: list of node indices from root to a leaf.
    feature_names_for_dt: ordered list of feature names used to train dt_model.
    base_attributes_info: dict containing 'all_attributes', 'value_ranges', 'category_values', 'categorical_features'
                          for RuleTree and Node creation.
    """
    conditions = []
    tree = dt_model.tree_

    for i in range(len(path_indices) - 1): # Iterate up to the parent of the leaf
        node_id = path_indices[i]
        child_id = path_indices[i+1]

        if tree.feature[node_id] < 0: # Should not happen if it's not a leaf already
            continue

        attr_idx = tree.feature[node_id]
        attr_name = feature_names_for_dt[attr_idx]
        threshold = tree.threshold[node_id]
        
        # Check if it's a numeric feature (this function is simplified for numeric)
        if attr_name in base_attributes_info["categorical_features"]:
            logging.warning(f"Rule extraction encountered categorical feature '{attr_name}' in DT path, skipping this condition (extraction focused on numeric).")
            continue

        operator = "<=" if child_id == tree.children_left[node_id] else ">"
        
        # Create a leaf Node for this condition
        # Feature type is known to be numeric here for simplicity
        condition_node = Node(attribute=attr_name, operator=operator, value=threshold, feature_type='numeric')
        conditions.append(condition_node)

        if len(conditions) >= max_rule_depth_extracted : # Limit the rule complexity
            break
            
    if not conditions:
        logging.debug("No valid (numeric) conditions extracted from DT path.")
        return None

    # Build the RuleTree by ANDing all conditions
    # For a single path, all conditions are ANDed.
    # If only one condition, that's the root.
    if len(conditions) == 1:
        rule_root = conditions[0]
    else:
        current_root = Node(operator="AND")
        current_root.left = conditions[0]
        current_root.right = conditions[1]
        for j in range(2, len(conditions)):
            new_root = Node(operator="AND")
            new_root.left = current_root
            new_root.right = conditions[j]
            current_root = new_root
        rule_root = current_root
        
    # Create a RuleTree object and set its root
    # Need all_attributes for the RuleTree constructor
    extracted_rule_tree = RuleTree(max_depth=max_rule_depth_extracted, # Max depth for the rule itself
                                   attributes=base_attributes_info["attributes"], 
                                   value_ranges=base_attributes_info["value_ranges"],
                                   category_values=base_attributes_info["category_values"],
                                   categorical_features=base_attributes_info["categorical_features"])
    extracted_rule_tree.root = rule_root
    
    if extracted_rule_tree.is_valid_rule():
        logging.debug(f"Extracted valid rule from DT: {extracted_rule_tree.to_string()}")
        return extracted_rule_tree
    else:
        logging.warning(f"Extracted rule from DT was invalid: {extracted_rule_tree.to_string()}")
        return None

def _select_rule_by_tournament(rules: List[RuleTree], k: int = 3) -> Optional[RuleTree]:
    """
    Seleciona a melhor regra de uma lista usando um torneio de tamanho k.
    Retorna a regra com o maior quality_score entre as sorteadas.
    """
    if not rules:
        return None
    
    # Garante que o tamanho do torneio não seja maior que a lista de regras
    tournament_size = min(len(rules), k)
    
    # Sorteia k competidores
    tournament_contestants = random.sample(rules, tournament_size)
    
    # O vencedor é aquele com o maior score de qualidade
    winner = max(tournament_contestants, key=lambda rule: rule.quality_score)
    return winner

def find_promising_path_in_dt(dt_model, target_class):
    """
    Finds a path in the Decision Tree leading to a leaf that predicts target_class_leaf
    with good purity or a significant number of samples.
    Returns a list of node indices for the path, or None.
    """
    tree = dt_model.tree_
    best_path = None
    best_leaf_score = -1 # Score can be purity * (n_samples_at_leaf / total_samples_at_root)

    # Find leaf nodes
    leaf_ids = np.where(tree.children_left == -1)[0] # -1 indicates a leaf

    for leaf_id in leaf_ids:
        # Value array at leaf: tree.value[leaf_id] is like [[count_class0, count_class1, ...]]
        class_counts_at_leaf = tree.value[leaf_id][0]
        predicted_class_idx = np.argmax(class_counts_at_leaf)
        
        # Map predicted_class_idx to actual class label
        # dt_model.classes_ should contain the actual class labels [0, 1, ...]
        if predicted_class_idx < len(dt_model.classes_):
            predicted_class_label = dt_model.classes_[predicted_class_idx]
        else:
            continue # Should not happen

        if predicted_class_label == target_class:
            n_samples_at_leaf = tree.n_node_samples[leaf_id]
            if n_samples_at_leaf == 0: continue

            purity = class_counts_at_leaf[predicted_class_idx] / n_samples_at_leaf
            # Simple score: purity * relative sample size (can be refined)
            score = purity * (n_samples_at_leaf / tree.n_node_samples[0]) 

            if score > best_leaf_score:
                best_leaf_score = score
                # Reconstruct path to this leaf
                current_path = []
                curr_node = leaf_id
                # Path reconstruction needs to go upwards, which is not directly stored.
                # Alternative: Recursive DFS to find paths and evaluate leaves.
                
                # Let's do a DFS to find path to this specific leaf_id
                # This is inefficient if done for all leaves. Better to find best leaf then its path.
                # For now, let's find ANY path to a good leaf.
                
                # Simpler DFS approach: traverse and find first good leaf path
                # This is a placeholder for a more robust path finding/selection
                
                # Path reconstruction:
                # To reconstruct the path to leaf_id, we need parent information or to traverse.
                # A simpler way for now: find any path via DFS and check leaf.
                
                # We'll implement a simpler path finder for now.
                # A better approach: iterate all paths to leaves, score them, pick the best.
                # For this first functional version, we'll find one good path.
                
                # Temporary: Find path to the first leaf matching the class
                # This part needs a robust implementation.
                # For a quick version, we can take the path that DecisionTreeClassifier.export_text might show.
                # However, export_text is for visualization.

                # Path reconstruction is tricky. Let's simplify:
                # We will just try to get one path that leads to target_class_leaf with high confidence
                # This is a known hard part of DT rule extraction.
                # For now, we will use a placeholder.
                # A full implementation would traverse from root, find leaf 'leaf_id', and backtrack or store path.
                
                # Placeholder: A real path reconstruction is needed here.
                # This would search from root (node 0) to leaf_id.
                # Due to complexity of robust path finding here, this will be a conceptual placeholder.
                # A proper implementation would be a graph traversal.
                
                # Instead of full path reconstruction for now, let's simplify for this example:
                # We will use export_text as a HINT and try to parse ONE rule.
                # This is NOT robust but gives a starting point.
                # This part is the most complex to make truly general from sklearn's tree struct.

                # A more direct way to get path information is to iterate from root
                # and build paths to all leaves.
                
                # Let's assume we pick this leaf_id and try to build its path
                path_to_leaf = [leaf_id]
                parent_node = -1
                # Find parent (this is where sklearn lacks easy parent pointers)
                # We'd have to search children_left/children_right of all nodes
                # to find which one has leaf_id as a child.
                # This iterative parent search is slow.
                
                # Let's try a DFS to find a suitable path
                # This will be a simplified DFS for one path
                
                # stack for DFS: (node_id, current_path_list)
                dfs_stack = [(0, [])] # Start with root node (0) and empty path
                
                found_path_for_leaf = None
                
                while dfs_stack:
                    curr_nid, path_so_far = dfs_stack.pop()
                    new_path = path_so_far + [curr_nid]
                    
                    if curr_nid == leaf_id: # We found the leaf we were interested in
                        found_path_for_leaf = new_path
                        break # Found path to this specific pre-selected good leaf
                    
                    # If not a leaf, add children to stack
                    if tree.children_left[curr_nid] != -1: # tree.TREE_LEAF is -1
                        dfs_stack.append((tree.children_right[curr_nid], new_path))
                        dfs_stack.append((tree.children_left[curr_nid], new_path))
                
                if found_path_for_leaf:
                    best_path = found_path_for_leaf
                    # We only need one good path for this mutation type
                    return best_path 
    
    return None # No suitable path found

def tournament_selection(population, tournament_size):
    # ... (código como antes) ...
    if not population: return None
    if len(population) < tournament_size: tournament_size = len(population)
    if tournament_size == 0: return None
    tournament = random.sample(population, tournament_size)
    winner = max(tournament, key=lambda ind: ind.fitness)
    return copy.deepcopy(winner)

# def crossover(parent1, parent2, max_depth,
#               attributes, value_ranges, category_values, categorical_features, # Passa info categórica
#               classes, max_rules_per_class, crossover_type="node"):
#     """ Performs crossover, passing feature context to pruning. """
#     # Cria child (passando info categórica)
#     child = Individual( max_rules_per_class, max_depth, attributes, value_ranges, category_values, categorical_features, classes, train_target=[] )
#     for class_label in classes:
#         rules1 = parent1.rules.get(class_label, []); rules2 = parent2.rules.get(class_label, [])
#         child_rules = []
#         # --- Rule-Level Crossover ---
#         if crossover_type == "rule":
#             # ... (lógica como antes) ...
#             if rules1 and rules2:
#                 rule1 = _select_rule_by_tournament(rules1)
#                 rule2 = _select_rule_by_tournament(rules2)
#             len1, len2 = len(rules1), len(rules2); max_len = max(len1, len2)
#             if max_len > 0:
#                  max_possible_point = min(len1, max_rules_per_class);
#                  if max_possible_point > 0:
#                       crossover_point = random.randint(0, max_possible_point); child_rules.extend(copy.deepcopy(rules1[:crossover_point]))
#                       needed_from_p2 = max_rules_per_class - len(child_rules)
#                       if needed_from_p2 > 0 and len2 > 0: child_rules.extend(copy.deepcopy(rules2[:min(needed_from_p2, len2)]))
#                  else: child_rules.extend(copy.deepcopy(rules2[:max_rules_per_class]))
#         # --- Node-Level Crossover ---
#         elif crossover_type == "node":
#             rule1, rule2 = None, None
#             if rules1 and rules2:
#                 rule1 = random.choice(rules1); rule2 = random.choice(rules2)
#                 rule1_copy = copy.deepcopy(rule1); rule2_copy = copy.deepcopy(rule2)
#                 node1 = select_random_node(rule1_copy.root); node2 = select_random_node(rule2_copy.root)
#                 if node1 and node2: swap_subtrees(node1, node2) # Troca inclui feature_type
#                 # Pruning (passando contexto completo)
#                 if get_tree_depth(rule1_copy.root) > max_depth:
#                     logging.debug(f"Pruning rule 1 post-crossover"); rule1_copy.root = prune_tree_to_depth( rule1_copy.root, max_depth, attributes, value_ranges, category_values, categorical_features )
#                 if get_tree_depth(rule2_copy.root) > max_depth:
#                     logging.debug(f"Pruning rule 2 post-crossover"); rule2_copy.root = prune_tree_to_depth( rule2_copy.root, max_depth, attributes, value_ranges, category_values, categorical_features )
#                 child_rules.append(rule1_copy); child_rules.append(rule2_copy)
#             # Preenche slots restantes
#             num_needed = max_rules_per_class - len(child_rules)
#             if num_needed > 0:
#                 rules_to_avoid = [];
#                 if rule1: rules_to_avoid.append(rule1);
#                 if rule2: rules_to_avoid.append(rule2)
#                 potential_rules = [copy.deepcopy(r) for r in rules1 + rules2 if r not in rules_to_avoid]
#                 random.shuffle(potential_rules); child_rules.extend(potential_rules[:num_needed])
#         else: # Fallback
#              logging.warning(f"Unknown crossover type: {crossover_type}. Copying."); child_rules.extend(copy.deepcopy(rules1)); child_rules.extend(copy.deepcopy(rules2)); random.shuffle(child_rules)
#         child.rules[class_label] = child_rules[:max_rules_per_class]
#     child.default_class = parent1.default_class if random.random() < 0.5 else parent2.default_class
#     child.remove_duplicate_rules()
#     return child

# Em ga_operators.py, substitua a função crossover inteira por esta

# def crossover( # último
#     parent1: Individual, parent2: Individual, max_depth: int,
#     attributes: List[str], value_ranges: Dict, category_values: Dict,
#     categorical_features: Set[str], classes: List[Any], max_rules_per_class: int,
#     crossover_type="node"
# ) -> Individual:
#     """
#     Realiza o crossover usando uma estratégia híbrida:
#     1. Prioriza o "transplante" de regras para cobrir fraquezas.
#     2. Usa o crossover de nó para combinar forças.
#     3. Usa o crossover de regra como fallback ou quando especificado.
#     """
#     # Cria um indivíduo filho "em branco"
#     child = Individual(max_rules_per_class, max_depth, attributes, value_ranges,
#                        category_values, categorical_features, classes)
#     child.rules = {c: [] for c in classes}

#     # --- LÓGICA DE CROSSOVER HÍBRIDO ---

#     # Identifica as forças (classes cobertas) de cada pai
#     strengths1 = parent1.covered_classes
#     strengths2 = parent2.covered_classes

#     for class_label in classes:
#         rules1 = parent1.rules.get(class_label, [])
#         rules2 = parent2.rules.get(class_label, [])
#         child_rules_for_class = []

#         # Se o crossover_type for 'rule', usa a lógica antiga de fatiamento de lista.
#         if crossover_type == "rule":
#             len1, len2 = len(rules1), len(rules2)
#             if len1 > 0:
#                 crossover_point = random.randint(0, len1)
#                 child_rules_for_class.extend(copy.deepcopy(rules1[:crossover_point]))
#             needed = max_rules_per_class - len(child_rules_for_class)
#             if needed > 0 and len2 > 0:
#                 child_rules_for_class.extend(copy.deepcopy(rules2[:min(needed, len2)]))

#         # Caso contrário, usa a nova lógica direcionada
#         else:
#             p1_is_strong = class_label in strengths1
#             p2_is_strong = class_label in strengths2

#             # Caso 1: Transplante do Pai 1 (P1 é forte, P2 é fraco)
#             if p1_is_strong and not p2_is_strong:
#                 rules1.sort(key=lambda r: r.quality_score, reverse=True)
#                 child_rules_for_class = copy.deepcopy(rules1[:max_rules_per_class])

#             # Caso 2: Transplante do Pai 2 (P2 é forte, P1 é fraco)
#             elif p2_is_strong and not p1_is_strong:
#                 rules2.sort(key=lambda r: r.quality_score, reverse=True)
#                 child_rules_for_class = copy.deepcopy(rules2[:max_rules_per_class])
            
#             # Caso 3: Ambos fortes (usa node crossover para combinar forças) ou Ambos fracos
#             else:
#                 if rules1 and rules2:
#                     # Seleciona as melhores regras de cada pai para o crossover de nó
#                     rule1 = _select_rule_by_tournament(rules1)
#                     rule2 = _select_rule_by_tournament(rules2)

#                     if rule1 and rule2:
#                         r1_copy, r2_copy = copy.deepcopy(rule1), copy.deepcopy(rule2)
#                         node1, node2 = select_random_node(r1_copy.root), select_random_node(r2_copy.root)
#                         if node1 and node2: swap_subtrees(node1, node2)
                        
#                         # Poda as regras filhas se necessário
#                         if get_tree_depth(r1_copy.root) > max_depth:
#                             r1_copy.root = prune_tree_to_depth(r1_copy.root, max_depth, attributes, value_ranges, category_values, categorical_features)
#                         if get_tree_depth(r2_copy.root) > max_depth:
#                             r2_copy.root = prune_tree_to_depth(r2_copy.root, max_depth, attributes, value_ranges, category_values, categorical_features)
                        
#                         child_rules_for_class.extend([r1_copy, r2_copy])
                
#                 # Preenche os slots restantes com as melhores regras restantes dos pais
#                 num_needed = max_rules_per_class - len(child_rules_for_class)
#                 if num_needed > 0:
#                     potential_rules = rules1 + rules2
#                     potential_rules.sort(key=lambda r: r.quality_score, reverse=True)
                    
#                     for rule in potential_rules:
#                         if len(child_rules_for_class) >= max_rules_per_class: break
#                         if not any(r.to_string() == rule.to_string() for r in child_rules_for_class):
#                             child_rules_for_class.append(copy.deepcopy(rule))

#         child.rules[class_label] = child_rules_for_class

#     # Define a classe padrão e finaliza
#     child.default_class = parent1.default_class if random.random() < 0.5 else parent2.default_class
#     child.remove_duplicate_rules()
#     return child

# Em ga_operators.py, adicione estas DUAS NOVAS FUNÇÕES AUXILIARES
# Elas podem ser colocadas antes da função crossover principal.

# #VERSAO QUE AUMENTOU O G-MEAN
# def _crossover_modo_expansao(parent1, parent2, classes, max_rules_per_class):
#     """
#     MODO EXPANSÃO: Foca em criar um filho com a máxima cobertura de classes,
#     combinando as forças complementares dos pais ("transplante").
#     """
#     child_rules = {c: [] for c in classes}
#     strengths1 = parent1.covered_classes
#     strengths2 = parent2.covered_classes

#     for class_label in classes:
#         rules1 = parent1.rules.get(class_label, [])
#         rules2 = parent2.rules.get(class_label, [])
#         p1_is_strong = class_label in strengths1
#         p2_is_strong = class_label in strengths2

#         # Caso 1: Transplante do Pai 1
#         if p1_is_strong and not p2_is_strong:
#             rules1.sort(key=lambda r: r.quality_score, reverse=True)
#             child_rules[class_label] = copy.deepcopy(rules1[:max_rules_per_class])
        
#         # Caso 2: Transplante do Pai 2
#         elif p2_is_strong and not p1_is_strong:
#             rules2.sort(key=lambda r: r.quality_score, reverse=True)
#             child_rules[class_label] = copy.deepcopy(rules2[:max_rules_per_class])

#         # Caso 3: Ambos fortes ou ambos fracos - pega as melhores regras de ambos
#         else:
#             potential_rules = rules1 + rules2
#             # Ordena todas as regras candidatas pela qualidade
#             potential_rules.sort(key=lambda r: r.quality_score, reverse=True)
#             # Adiciona as melhores regras sem duplicatas
#             unique_rules_added = set()
#             final_rules = []
#             for rule in potential_rules:
#                 if len(final_rules) >= max_rules_per_class: break
#                 rule_str = rule.to_string()
#                 if rule_str not in unique_rules_added:
#                     final_rules.append(copy.deepcopy(rule))
#                     unique_rules_added.add(rule_str)
#             child_rules[class_label] = final_rules
            
#     return child_rules

# def _crossover_modo_refinamento(parent1, parent2, classes, max_rules_per_class, max_depth, attributes, value_ranges, category_values, categorical_features):
#     """
#     MODO REFINAMENTO: Foca em melhorar as classes de pior performance,
#     usando o crossover de nó de forma cirúrgica.
#     """
#     child_rules = {c: [] for c in classes}

#     # Combina todas as regras dos pais e ordena pela qualidade
#     all_parent_rules = {}
#     for class_label in classes:
#         rules = parent1.rules.get(class_label, []) + parent2.rules.get(class_label, [])
#         rules.sort(key=lambda r: r.quality_score, reverse=True)
#         all_parent_rules[class_label] = rules

#     # Herda as melhores regras para a maioria das classes
#     for class_label in classes:
#         if all_parent_rules[class_label]:
#             child_rules[class_label].append(copy.deepcopy(all_parent_rules[class_label][0]))
    
#     # Identifica a classe mais fraca (menor quality_score da melhor regra) para focar
#     weakest_class = None
#     lowest_quality = float('inf')
#     for class_label in classes:
#         if child_rules[class_label]:
#             quality = child_rules[class_label][0].quality_score
#             if quality < lowest_quality:
#                 lowest_quality = quality
#                 weakest_class = class_label
    
#     # Tenta criar uma nova regra para a classe mais fraca via crossover de nó
#     if weakest_class and len(child_rules[weakest_class]) < max_rules_per_class:
#         rules1 = parent1.rules.get(weakest_class, [])
#         rules2 = parent2.rules.get(weakest_class, [])
#         if rules1 and rules2:
#             best_rule1 = _select_rule_by_tournament(rules1)
#             best_rule2 = _select_rule_by_tournament(rules2)
#             if best_rule1 and best_rule2:
#                 # Lógica do crossover de nó que já tínhamos
#                 r1_copy, r2_copy = copy.deepcopy(best_rule1), copy.deepcopy(best_rule2)
#                 node1, node2 = select_random_node(r1_copy.root), select_random_node(r2_copy.root)
#                 if node1 and node2: swap_subtrees(node1, node2)
#                 # ... (poda das regras) ...
#                 child_rules[weakest_class].extend([r1_copy, r2_copy])

#     return child_rules

# Agora, substitua sua função 'crossover' principal por este "Orquestrador"

# #VERSÃO DO CROSSOVER QUE AUMENTO O G-MEAN
# def crossover(
#     parent1: Individual, parent2: Individual, max_depth: int,
#     attributes: List[str], value_ranges: Dict, category_values: Dict,
#     categorical_features: Set[str], classes: List[Any], max_rules_per_class: int,
#     crossover_type="node" # Mantido por compatibilidade
# ) -> Individual:
#     """
#     Orquestrador do Crossover Adaptativo.
#     Verifica a "maturidade" dos pais e escolhe o modo de operação apropriado.
#     """
#     # Cria um indivíduo filho "em branco"
#     child = Individual(max_rules_per_class, max_depth, attributes, value_ranges,
#                        category_values, categorical_features, classes)
#     child.rules = {c: [] for c in classes} # Limpa as regras

#     total_classes = len(classes)
    
#     # --- Lógica de Decisão do Orquestrador ---
    
#     # Verifica se ambos os pais são "completos" (cobrem todas as classes)
#     parent1_is_complete = len(parent1.covered_classes) == total_classes
#     parent2_is_complete = len(parent2.covered_classes) == total_classes

#     # Se o crossover_type for 'rule', usa a lógica antiga como um override
#     if crossover_type == "rule":
#         logging.debug("Crossover: Usando modo 'rule' (override).")
#         # (Lógica do crossover de regra como na sua versão original)
#         for class_label in classes:
#             # ... (seu código de fatiamento de lista de regras aqui)
#             pass # Placeholder para sua lógica de crossover de regra

#     # Se ambos os pais forem completos, entra em MODO REFINAMENTO
#     elif parent1_is_complete and parent2_is_complete:
#         logging.debug("Crossover: Ambos os pais completos. Ativando MODO REFINAMENTO.")
#         child.rules = _crossover_modo_refinamento(
#             parent1, parent2, classes, max_rules_per_class, max_depth,
#             attributes, value_ranges, category_values, categorical_features
#         )
#     # Caso contrário, entra em MODO EXPANSÃO
#     else:
#         logging.debug("Crossover: Pelo menos um pai incompleto. Ativando MODO EXPANSÃO.")
#         child.rules = _crossover_modo_expansao(
#             parent1, parent2, classes, max_rules_per_class
#         )

#     # Finaliza o filho
#     child.default_class = parent1.default_class if random.random() < 0.5 else parent2.default_class
#     child.remove_duplicate_rules()
#     return child

# # --- Mutação (Lógica de Operador/Valor Corrigida para Tipos) ---
# def mutate_individual(
#     individual, mutation_rate, max_depth,
#     attributes, value_ranges, # Contexto numérico
#     category_values, categorical_features, # <<<--- PARÂMETROS RECEBIDOS
#     classes, max_rules_per_class,
#     data, target # Para avaliar performance de regras
#     ):
#     """
#     Mutates an individual in-place, correctly handling operator/value
#     mutation for both numeric and categorical features.
#     """
#     rule_performance = individual.evaluate_rule_performance(data, target)
#     sorted_rules = sorted(rule_performance.items(), key=lambda item: item[1])
#     num_rules_total = individual.count_total_rules()
#     num_rules_to_mutate = max(1, int(num_rules_total * mutation_rate)) if num_rules_total > 0 else 0
#     mutated_indices = set()

#     for i in range(min(num_rules_to_mutate, len(sorted_rules))):
#         (class_label, rule_idx) = sorted_rules[i][0]; rule_tuple = (class_label, rule_idx)
#         if rule_idx >= len(individual.rules[class_label]) or rule_tuple in mutated_indices: continue

#         rule_tree = individual.rules[class_label][rule_idx]
#         mutation_type = random.choice(['operator', 'value', 'subtree', 'from_classifier'])
#         logging.debug(f"Mutating rule {rule_idx} class {class_label} (type: {mutation_type})")

#         try:
#             if mutation_type == 'operator':
#                 node = select_random_node(rule_tree.root)
#                 if node:
#                     if node.is_internal(): node.operator = random.choice(LOGICAL_OPERATORS)
#                     elif node.is_leaf():
#                         # --- LÓGICA CORRIGIDA ---
#                         if node.attribute in categorical_features:
#                             node.operator = random.choice(CATEGORICAL_COMPARISON_OPERATORS)
#                             logging.debug(f"  Mutated categorical operator for {node.attribute} to {node.operator}")
#                         else: # Assume numérico
#                             node.operator = random.choice(NUMERIC_COMPARISON_OPERATORS)
#                             logging.debug(f"  Mutated numeric operator for {node.attribute} to {node.operator}")
#                         # --------------------------

#             elif mutation_type == 'value':
#                 node = select_random_node(rule_tree.root)
#                 if node and node.is_leaf():
#                     attribute = node.attribute
#                     # --- LÓGICA CORRIGIDA ---
#                     if attribute in categorical_features:
#                         possible_values = category_values.get(attribute)
#                         # Garante que há valores para escolher e que o novo valor é diferente (se possível)
#                         if possible_values and len(possible_values) > 1:
#                              current_value = node.value
#                              # Gera um novo valor diferente do atual
#                              new_value = random.choice([val for val in possible_values if val != current_value])
#                              node.value = new_value
#                              logging.debug(f"  Mutated categorical value for {attribute} to {node.value}")
#                         elif possible_values and len(possible_values) == 1:
#                              # Se só existe um valor possível, não há o que mutar
#                              logging.debug(f"  Skipped value mutation for categorical {attribute}: only one possible value.")
#                         else:
#                              logging.warning(f"  Cannot mutate value for categorical {attribute}: no possible values found.")
#                     elif attribute in value_ranges: # É numérico
#                         min_val, max_val = value_ranges[attribute]
#                         if min_val < max_val: # Só muta se houver um range
#                              node.value = random.uniform(min_val, max_val)
#                              logging.debug(f"  Mutated numeric value for {attribute} to {node.value:.4f}")
#                         else:
#                              logging.debug(f"  Skipped value mutation for numeric {attribute}: range is zero.")
#                     else:
#                          logging.warning(f"  Cannot mutate value for attribute {attribute}: not found in categorical or numeric range info.")
#                     # --------------------------

#             elif mutation_type == 'subtree':
#                  new_subtree_depth = random.randint(1, max_depth)
#                  try:
#                      # Chama RuleTree passando contexto categórico
#                      replacement_tree = RuleTree(new_subtree_depth, attributes, value_ranges, category_values, categorical_features)
#                      new_subtree_root = replacement_tree.root
#                  except (ValueError, RuntimeError) as gen_e: logging.error(f"Failed to generate replacement subtree: {gen_e}"); continue

#                  # Seleciona nó para substituir
#                  nodes_with_parent = []; q = [(rule_tree.root, None)];
#                  while q:
#                       curr, parent = q.pop(0)
#                       if curr: nodes_with_parent.append((curr, parent));
#                       if not curr.is_leaf(): q.append((curr.left, curr)); q.append((curr.right, curr))

#                  if nodes_with_parent:
#                       node_to_replace, node_to_replace_parent = random.choice(nodes_with_parent)
#                       # Realiza substituição
#                       if node_to_replace == rule_tree.root: rule_tree.root = new_subtree_root
#                       elif node_to_replace_parent:
#                            if node_to_replace_parent.left == node_to_replace: node_to_replace_parent.left = new_subtree_root
#                            elif node_to_replace_parent.right == node_to_replace: node_to_replace_parent.right = new_subtree_root
#                            else: logging.warning("Subtree replacement failed: child mismatch.")
#                       else: logging.warning("Subtree replacement failed: parent not found.")
#                  else: rule_tree.root = new_subtree_root

#                         # <<< BLOCO NOVO E FUNCIONAL para 'from_classifier' >>>
#             elif mutation_type == 'from_classifier':
#                 logging.debug(f"  Attempting 'from_classifier' mutation for rule {rule_idx} of class {class_label}")
#                 extracted_rule_tree = None
#                 try:
#                     # Etapa A: Pré-processar dados (foco em numéricos para esta versão)
#                     # (all_attributes é 'attributes' no escopo de mutate_individual)
#                     X_np_sample, y_np_sample, numeric_attr_names_dt = _preprocess_data_for_dt(
#                         data, target, attributes, categorical_features, sample_size=200 
#                     )

#                     if X_np_sample.ndim == 1: # Handle case where X_np_sample might be 1D if only 1 feature
#                         X_np_sample = X_np_sample.reshape(-1, 1)

#                     if X_np_sample.shape[0] < 10 or X_np_sample.shape[1] == 0 : # Poucos dados ou nenhuma feature numérica
#                         logging.warning("  'from_classifier': Not enough data or numeric features to train guide DT. Skipping.")
#                     else:
#                         # Etapa B: Treinar DecisionTreeClassifier
#                         dt_guide = DecisionTreeClassifier(
#                             max_depth=5, # Profundidade da árvore guia (simples)
#                             min_samples_leaf=10, 
#                             class_weight='balanced',
#                             random_state=random.randint(0, 100000) # Adicionar aleatoriedade
#                         )
#                         dt_guide.fit(X_np_sample, y_np_sample)

#                         # Etapa C: Encontrar um caminho promissor na árvore treinada
#                         # (target_class_of_rule é class_label)
#                         promising_path_node_indices = find_promising_path_in_dt(dt_guide, class_label)

#                         if promising_path_node_indices:
#                             # Etapa D: Extrair a RuleTree desse caminho
#                             # (attributes_info para o construtor de RuleTree e Node)
#                             base_attributes_info = {
#                                 "all_attributes": attributes, # Lista completa de atributos originais
#                                 "value_ranges": value_ranges,
#                                 "category_values": category_values,
#                                 "categorical_features": categorical_features
#                             }
#                             extracted_rule_tree = _extract_single_rule_from_dt_path(
#                                 dt_guide,
#                                 promising_path_node_indices,
#                                 numeric_attr_names_dt, # Nomes das features numéricas usadas pelo DT
#                                 class_label,
#                                 max_depth, # Profundidade máxima permitida para a regra extraída (pode ser a max_depth do indivíduo)
#                                 base_attributes_info
#                             )
#                         else:
#                             logging.debug(f"  'from_classifier': No promising path found in DT for class {class_label}.")

#                     if extracted_rule_tree and extracted_rule_tree.is_valid_rule():
#                         logging.debug(f"  Successfully generated and replaced rule for class {class_label} using classifier guide.")
#                         individual.rules[class_label][rule_idx] = extracted_rule_tree # Substitui a regra antiga
#                     elif extracted_rule_tree and not extracted_rule_tree.is_valid_rule():
#                         logging.warning(f"  'from_classifier': Extracted rule was invalid for class {class_label}. Original rule kept.")
#                     else: # extracted_rule_tree is None
#                         logging.debug(f"  'from_classifier': No valid rule extracted for class {class_label}. Original rule kept.")
#                         # Opcional: aplicar outra mutação como fallback se esta falhar em gerar uma regra
#                         # Ex: aplicar 'subtree' mutation aqui
#                         # pass 

#                 except ImportError:
#                     logging.warning("  Skipping 'from_classifier' mutation: scikit-learn is not installed.")
#                 except Exception as e_clf:
#                     logging.error(f"  Error during 'from_classifier' mutation: {e_clf}", exc_info=True)

#         except Exception as mut_e:
#              logging.error(f"Error during mutation op ({mutation_type}) rule {rule_idx} class {class_label}: {mut_e}", exc_info=True)

#         mutated_indices.add(rule_tuple)

#     # --- Mutações Estruturais ---
#     if random.random() < (mutation_rate / 2): # Adicionar regra
#         target_class = random.choice(classes)
#         if len(individual.rules.get(target_class, [])) < max_rules_per_class:
#             try: # Chama RuleTree passando contexto categórico
#                  new_rule = RuleTree(max_depth, attributes, value_ranges, category_values, categorical_features)
#                  individual.rules[target_class].append(new_rule); logging.debug(f"Added new random rule to class {target_class}")
#             except Exception as e: logging.error(f"Failed to generate/add new rule: {e}")

#     # Mudar classe padrão
#     if random.random() < (mutation_rate / 2):
#         original_default = individual.default_class; possible_defaults = [c for c in classes if c != original_default]
#         if possible_defaults: individual.default_class = random.choice(possible_defaults); logging.debug(f"Mutated default class from {original_default} to {individual.default_class}")

#     # --- Limpeza e Simplificação ---
#     individual.remove_duplicate_rules()
#     for class_label, rule_list in individual.rules.items():
#          simplified_list = []
#          for i, rule_tree in enumerate(rule_list):
#              try:
#                  # Simplificação agora usa info de tipo dentro de RuleTree
#                  rule_tree.simplify()
#                  simplified_list.append(rule_tree)
#              except Exception as simp_e:
#                   logging.error(f"Error simplifying rule {i} class {class_label}: {simp_e}")
#                   simplified_list.append(rule_tree) # Mantém original
#          individual.rules[class_label] = simplified_list

# ==============================================================================
# --- SEÇÃO DE MUTAÇÃO REFATORADA ---
# ==============================================================================

# Adicione estas funções auxiliares privadas ao seu arquivo ga_operators.py,
# idealmente antes da função principal mutate_individual.

def _mutate_operator(node: Node, categorical_features: Set[str], **kwargs) -> None:
    """Muta o operador de um nó (lógico ou de comparação), respeitando o tipo."""
    try:
        if node.is_internal():
            node.operator = random.choice(LOGICAL_OPERATORS)
            logging.debug(f"  Mutated internal operator to {node.operator}")
        elif node.is_leaf():
            if node.attribute in categorical_features:
                node.operator = random.choice(CATEGORICAL_COMPARISON_OPERATORS)
                logging.debug(f"  Mutated categorical operator for '{node.attribute}' to '{node.operator}'")
            else: # Assume numérico
                node.operator = random.choice(NUMERIC_COMPARISON_OPERATORS)
                logging.debug(f"  Mutated numeric operator for '{node.attribute}' to '{node.operator}'")
    except Exception as e:
        logging.error(f"  Error during operator mutation: {e}", exc_info=False)

def _mutate_value(node: Node, value_ranges: Dict, category_values: Dict, categorical_features: Set[str], **kwargs) -> None:
    """Muta o valor de um nó folha, respeitando o tipo do atributo."""
    try:
        if not node.is_leaf():
            return

        attribute = node.attribute
        if attribute in categorical_features:
            possible_values = list(category_values.get(attribute, []))
            # Garante que há mais de um valor para escolher
            if len(possible_values) > 1:
                # Tenta escolher um valor diferente do atual
                new_value = random.choice([val for val in possible_values if val != node.value] or possible_values)
                node.value = new_value
                logging.debug(f"  Mutated categorical value for '{attribute}' to '{node.value}'")
            else:
                logging.debug(f"  Skipped value mutation for categorical '{attribute}': not enough distinct values.")
        elif attribute in value_ranges: # Assume numérico
            min_val, max_val = value_ranges[attribute]
            if min_val < max_val:
                node.value = random.uniform(min_val, max_val)
                logging.debug(f"  Mutated numeric value for '{attribute}' to {node.value:.4f}")
            else:
                logging.debug(f"  Skipped value mutation for numeric '{attribute}': range is zero.")
        else:
            logging.warning(f"  Cannot mutate value for attribute '{attribute}': not found in value ranges or categories.")
    except Exception as e:
        logging.error(f"  Error during value mutation: {e}", exc_info=False)


def _mutate_subtree(rule_tree: RuleTree, max_depth: int, attributes: List[str], value_ranges: Dict, category_values: Dict, categorical_features: Set[str], **kwargs) -> None:
    """Substitui uma subárvore aleatória da regra por uma nova subárvore gerada aleatoriamente."""
    try:
        # Gera uma nova subárvore para substituição
        new_subtree_depth = random.randint(1, max(1, max_depth - 1))
        replacement_tree = RuleTree(new_subtree_depth, attributes, value_ranges, category_values, categorical_features)
        new_subtree_root = replacement_tree.root

        # Seleciona um nó para ser substituído na árvore original
        nodes_with_parent = []
        q = [(rule_tree.root, None)]
        curr, parent = q.pop(0)
        while q:
            curr, parent = q.pop(0)
            if curr:
                nodes_with_parent.append((curr, parent))
                if not curr.is_leaf():
                    q.append((curr.left, curr))
                    q.append((curr.right, curr))

        if not nodes_with_parent:
            rule_tree.root = new_subtree_root # A árvore estava vazia ou inválida
            return

        node_to_replace, parent = random.choice(nodes_with_parent)

        # Realiza a substituição
        if parent is None: # O nó a ser substituído é a raiz
            rule_tree.root = new_subtree_root
        elif parent.left == node_to_replace:
            parent.left = new_subtree_root
        elif parent.right == node_to_replace:
            parent.right = new_subtree_root
        
        logging.debug(f"  Successfully replaced a subtree.")

    except Exception as e:
        logging.error(f"  Error during subtree mutation: {e}", exc_info=False)

# Em ga_operators.py

# def _mutate_from_classifier(rule_tree: RuleTree, class_label: Any, individual_to_mutate: Individual, **kwargs) -> None:
#     """
#     Realiza uma mutação inteligente guiada pela importância das features de um 
#     Random Forest especialista. Em vez de substituir a regra inteira, ela
#     "transplanta" um atributo de alta importância para dentro de uma condição
#     existente na regra.
#     """
#     try:
#         # --- 0. Extrai o contexto necessário dos kwargs ---
#         data, target = kwargs['data'], kwargs['target']
#         attributes, categorical_features = kwargs['attributes'], kwargs['categorical_features']
#         value_ranges, category_values = kwargs['value_ranges'], kwargs['category_values']
        
#         TOP_N_FEATURES = 5 # Usaremos as 5 features mais importantes
#         RF_PARAMS = {
#             'n_estimators': 20, 'max_depth': 10, 'min_samples_leaf': 10,
#             'random_state': random.randint(0, 10000), 'n_jobs': 1, # n_jobs=1 para evitar problemas em workers paralelos
#             'class_weight': 'balanced'
#         }

#         # --- 1. Lógica do "Focus Training" ---
#         logging.debug(f"  Mutação 'from_classifier' para a classe {class_label}: Iniciando Focus Training...")
#         positive_indices = [i for i, lbl in enumerate(target) if lbl == class_label]
#         if not positive_indices:
#             logging.debug("  Nenhuma instância positiva para o Focus Training. Mutação cancelada.")
#             return

#         negative_indices = [i for i, lbl in enumerate(target) if lbl != class_label]
#         num_negatives = min(len(positive_indices) * 3, len(negative_indices))
#         if num_negatives == 0:
#             logging.debug("  Nenhuma instância negativa para o Focus Training. Mutação cancelada.")
#             return

#         sample_negative_indices = random.sample(negative_indices, num_negatives)
#         focused_indices = positive_indices + sample_negative_indices
#         random.shuffle(focused_indices)

#         X_focused = [data[i] for i in focused_indices]
#         y_focused = [1 if target[i] == class_label else 0 for i in focused_indices]
        
#         X_df_focused = pd.DataFrame(X_focused).apply(pd.to_numeric, errors='coerce').fillna(0)
        
#         # --- 2. Treina o RF especialista e extrai a importância das features ---
#         rf_guide = RandomForestClassifier(**RF_PARAMS)
#         rf_guide.fit(X_df_focused, y_focused)
        
#         importances = rf_guide.feature_importances_
#         feature_names = X_df_focused.columns
#         feature_importance_series = pd.Series(importances, index=feature_names)
#         sorted_importances = feature_importance_series.sort_values(ascending=False)
        
#         top_features = sorted_importances.head(TOP_N_FEATURES).index.tolist()
#         if not top_features:
#             logging.debug("  Não foi possível extrair features importantes. Mutação cancelada.")
#             return
        
#         logging.debug(f"  Top features para a classe {class_label}: {top_features}")

#         # --- 3. "Transplante" Inteligente do Atributo ---
#         # Seleciona um nó folha aleatório da regra para ser modificado
#         node_to_mutate = select_random_node(rule_tree.root)

#         if node_to_mutate and node_to_mutate.is_leaf():
#             # Escolhe um dos atributos de alta importância para o "transplante"
#             new_attribute = random.choice(top_features)
#             original_attribute = node_to_mutate.attribute
            
#             logging.debug(f"  Transplantando atributo: '{original_attribute}' -> '{new_attribute}'")
#             node_to_mutate.attribute = new_attribute

#             # Atualiza o operador e o valor para serem consistentes com o novo atributo
#             if new_attribute in categorical_features:
#                 node_to_mutate.feature_type = 'categorical'
#                 node_to_mutate.operator = random.choice(CATEGORICAL_COMPARISON_OPERATORS)
#                 possible_values = list(category_values.get(new_attribute, ['UNKNOWN']))
#                 node_to_mutate.value = random.choice(possible_values)
#             else: # Assume numérico
#                 node_to_mutate.feature_type = 'numeric'
#                 node_to_mutate.operator = random.choice(NUMERIC_COMPARISON_OPERATORS)
#                 min_val, max_val = value_ranges.get(new_attribute, (0, 1))
#                 node_to_mutate.value = random.uniform(min_val, max_val) if max_val > min_val else min_val
            
#             logging.debug(f"  Nova condição: ({node_to_mutate.attribute} {node_to_mutate.operator} {node_to_mutate.value})")
#         else:
#             logging.debug("  Não foi possível selecionar um nó folha para mutação.")

#     except Exception as e:
#         logging.error(f"  Erro durante a mutação 'from_classifier': {e}", exc_info=True)

# Em ga_operators.py, substitua a função _mutate_from_classifier existente

# def _mutate_from_classifier(rule_tree: RuleTree, class_label: Any, individual_to_mutate: Individual, **kwargs) -> None:
#     """
#     Tenta substituir uma regra inteira por uma nova, extraída de uma 
#     Hoeffding Tree guia treinada em uma amostra do chunk de dados atual.
#     """
#     try:
#         # --- NOVO: Extrai o contexto necessário dos kwargs ---
#         data, target = kwargs['data'], kwargs['target']
#         attributes, categorical_features = kwargs['attributes'], kwargs['categorical_features']
#         value_ranges, category_values = kwargs['value_ranges'], kwargs['category_values']
#         max_depth = kwargs['max_depth']
        
#         # --- REMOVIDO: A antiga chamada ao _preprocess_data_for_dt ---

#         # --- NOVO: Lógica de treino com Hoeffding Tree ---
#         if not data:
#             logging.debug("  'from_classifier': Sem dados para treinar a HT guia. Skipping.")
#             return

#         # 1. Instancia e treina uma Hoeffding Tree leve do river
#         ht_guide = tree.HoeffdingTreeClassifier(
#             grace_period=50,
#             delta=0.01,
#             # Informa à HT quais atributos são categóricos para que ela os trate corretamente
#             nominal_attributes=[attr for attr in attributes if attr in categorical_features]
#         )
        
#         # Pega uma amostra para treinar a árvore guia
#         sample_size = min(len(data), 200) # Usa uma amostra pequena para a mutação ser rápida
#         sample_indices = random.sample(range(len(data)), sample_size)

#         for idx in sample_indices:
#             ht_guide.learn_one(data[idx], target[idx])

#         # 2. Tenta extrair a regra usando nossa função já validada
#         base_attributes_info = {
#             "attributes": attributes, "value_ranges": value_ranges,
#             "category_values": category_values, "categorical_features": categorical_features
#         }
        
#         # Chama a função de extração que já temos neste mesmo arquivo
#         extracted_rt = _extract_single_rule_from_river_ht(
#             ht_guide, class_label, base_attributes_info
#         )

#         # 3. Substitui a regra se a extração foi bem-sucedida
#         if extracted_rt and extracted_rt.is_valid_rule():
#             # Substitui a raiz da regra original pela nova árvore extraída
#             rule_tree.root = extracted_rt.root
#             logging.debug(f"  Sucesso! Regra substituída pela mutação 'from_classifier' para a classe {class_label}.")
#         else:
#             logging.debug(f"  'from_classifier': Nenhuma regra válida foi extraída da HT. Regra original mantida.")

#     except ImportError:
#         logging.warning("  Skipping 'from_classifier' mutation: a biblioteca 'river' não está instalada.")
#     except Exception as e:
#         logging.error(f"  Erro durante a mutação 'from_classifier': {e}", exc_info=True)
# NOVO CÓDIGO COMPLETO para a função em ga_operators.py

def _mutate_from_classifier_xg(rule_tree: RuleTree, class_label: Any, individual_to_mutate: Individual, **kwargs) -> None:
    """
    Realiza uma mutação inteligente guiada pela importância das features de um 
    modelo XGBoost especialista, realizando um "transplante" de atributo.
    """
    try:
        # --- 0. Extrai o contexto necessário dos kwargs ---
        data, target = kwargs['data'], kwargs['target']
        attributes, categorical_features = kwargs['attributes'], kwargs['categorical_features']
        value_ranges, category_values = kwargs['value_ranges'], kwargs['category_values']
        
        TOP_N_FEATURES = 5
        # Parâmetros para um XGBoost leve e rápido, focado em feature importance
        XGB_PARAMS = {
            'n_estimators': 30,       # Um pouco mais de árvores que o RF
            'max_depth': 7,           # Pode ser um pouco mais profundo
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_jobs': 1,              # Essencial para evitar conflitos em paralelismo
            'random_state': random.randint(0, 10000)
        }

        # --- 1. Lógica do "Focus Training" ---
        logging.debug(f"  Mutação 'from_classifier' para a classe {class_label}: Iniciando Focus Training com XGBoost...")
        positive_indices = [i for i, lbl in enumerate(target) if lbl == class_label]
        if not positive_indices:
            logging.debug("  Nenhuma instância positiva para o Focus Training. Mutação cancelada.")
            return

        negative_indices = [i for i, lbl in enumerate(target) if lbl != class_label]
        num_negatives = min(len(positive_indices) * 3, len(negative_indices))
        if num_negatives == 0:
            logging.debug("  Nenhuma instância negativa para o Focus Training. Mutação cancelada.")
            return

        sample_negative_indices = random.sample(negative_indices, num_negatives)
        focused_indices = positive_indices + sample_negative_indices
        random.shuffle(focused_indices)

        X_focused = [data[i] for i in focused_indices]
        y_focused = np.array([1 if target[i] == class_label else 0 for i in focused_indices])
        
        X_df_focused = pd.DataFrame(X_focused).apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # --- 2. Treina o XGBoost especialista e extrai a importância das features ---
        xgb_guide = xgb.XGBClassifier(**XGB_PARAMS)
        xgb_guide.fit(X_df_focused, y_focused)
        
        importances = xgb_guide.feature_importances_
        feature_names = X_df_focused.columns
        feature_importance_series = pd.Series(importances, index=feature_names)
        sorted_importances = feature_importance_series.sort_values(ascending=False)
        
        top_features = sorted_importances.head(TOP_N_FEATURES).index.tolist()
        if not top_features:
            logging.debug("  Não foi possível extrair features importantes do XGBoost. Mutação cancelada.")
            return
        
        logging.debug(f"  Top features (XGBoost) para a classe {class_label}: {top_features}")

        # --- 3. "Transplante" Inteligente do Atributo (lógica idêntica à anterior) ---
        node_to_mutate = select_random_node(rule_tree.root)

        if node_to_mutate and node_to_mutate.is_leaf():
            new_attribute = random.choice(top_features)
            original_attribute = node_to_mutate.attribute
            
            logging.debug(f"  Transplantando atributo: '{original_attribute}' -> '{new_attribute}'")
            node_to_mutate.attribute = new_attribute

            # Atualiza o operador e o valor para serem consistentes com o novo atributo
            if new_attribute in categorical_features:
                node_to_mutate.feature_type = 'categorical'
                node_to_mutate.operator = random.choice(CATEGORICAL_COMPARISON_OPERATORS)
                possible_values = list(category_values.get(new_attribute, ['UNKNOWN']))
                node_to_mutate.value = random.choice(possible_values)
            else: # Assume numérico
                node_to_mutate.feature_type = 'numeric'
                node_to_mutate.operator = random.choice(NUMERIC_COMPARISON_OPERATORS)
                min_val, max_val = value_ranges.get(new_attribute, (0, 1))
                node_to_mutate.value = random.uniform(min_val, max_val) if max_val > min_val else min_val
            
            logging.debug(f"  Nova condição: ({node_to_mutate.attribute} {node_to_mutate.operator} {node_to_mutate.value})")
        else:
            logging.debug("  Não foi possível selecionar um nó folha para mutação.")

    except ImportError:
        logging.warning("  Mutação com XGBoost pulada: a biblioteca 'xgboost' não está instalada.")
    except Exception as e:
        logging.error(f"  Erro durante a mutação 'from_classifier' com XGBoost: {e}", exc_info=True)

def _mutate_from_classifier(rule_tree: RuleTree, class_label: Any, individual_to_mutate: Individual, **kwargs) -> None:
    """Tenta substituir uma regra inteira por uma nova, extraída de uma Decision Tree guia."""
    try:
        # Extrai o contexto necessário dos kwargs
        data, target = kwargs['data'], kwargs['target']
        attributes, categorical_features = kwargs['attributes'], kwargs['categorical_features']
        value_ranges, category_values = kwargs['value_ranges'], kwargs['category_values']
        max_depth = kwargs['max_depth']

        # Prepara dados para a DT (foco em numéricos)
        X_np, y_np, numeric_ft_names = _preprocess_data_for_dt(data, target, attributes, categorical_features, sample_size=200)

        if X_np.shape[0] < 10 or X_np.shape[1] == 0:
            logging.debug("  'from_classifier': Not enough data or numeric features for guide DT. Skipping.")
            return

        # Treina a DT guia
        dt_guide = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, class_weight='balanced', random_state=random.randint(0, 100000))
        dt_guide.fit(X_np, y_np)

        # Encontra um caminho promissor e extrai a regra
        path_indices = find_promising_path_in_dt(dt_guide, class_label)
        if path_indices:
            base_attributes_info = {
                "attributes": attributes, "value_ranges": value_ranges,
                "category_values": category_values, "categorical_features": categorical_features
            }
            extracted_rt = _extract_single_rule_from_dt_path(dt_guide, path_indices, numeric_ft_names, class_label, max_depth, base_attributes_info)
            
            if extracted_rt and extracted_rt.is_valid_rule():
                # Substitui a raiz da regra original pela nova árvore extraída
                rule_tree.root = extracted_rt.root
                logging.debug(f"  Successfully replaced rule with one from classifier guide.")
            else:
                logging.debug("  'from_classifier': Extracted rule was invalid. Original rule kept.")
        else:
            logging.debug(f"  'from_classifier': No promising path found in DT for class {class_label}.")

    except ImportError:
        logging.warning("  Skipping 'from_classifier' mutation: scikit-learn is not installed.")
    except Exception as e:
        logging.error(f"  Error during 'from_classifier' mutation: {e}", exc_info=True)


# # --- Função de Mutação Principal Refatorada ---

# def mutate_individual(
#     individual: Individual,
#     mutation_rate: float,
#     max_depth: int,
#     intelligent_mutation_prob: float,
#     attributes: List[str],
#     value_ranges: Dict,
#     category_values: Dict,
#     categorical_features: Set[str],
#     classes: List[Any],
#     max_rules_per_class: int,
#     data: List[Dict],
#     target: List[Any]
# ) -> None:
#     """
#     Muta um indivíduo, aplicando diferentes estratégias de forma robusta e organizada.
#     Modifica o indivíduo "in-place" (não retorna um novo).
#     """
#     # 1. Avalia as regras para focar a mutação nas de pior desempenho
#     rule_performance = individual.evaluate_rule_performance(data, target)
#     sorted_rules = sorted(rule_performance.items(), key=lambda item: item[1])
#     num_rules_total = individual.count_total_rules()

#     all_rules_with_context = []
#     for class_label, rule_list in individual.rules.items():
#         for rule_idx, rule in enumerate(rule_list):
#             # Armazena a regra e seu "endereço" (classe e índice)
#             all_rules_with_context.append({
#                 "rule_obj": rule,
#                 "class_label": class_label,
#                 "rule_idx": rule_idx
#             })

#     if num_rules_total == 0:
#         logging.debug("Individual has no rules to mutate, attempting structural mutation.")
    
#     num_rules_to_mutate = max(1, int(num_rules_total * mutation_rate)) if num_rules_total > 0 else 0
    
#     # 2. (NOVO) Define as mutações possíveis com base no contexto do chunk de dados
#     possible_mutation_types = ['operator', 'value', 'subtree', 'from_classifier'], 
#     sorted_rules = sorted(all_rules_with_context, key=lambda r: r['rule_obj'].quality_score)
#     num_rules_total = len(sorted_rules)
#     num_rules_to_mutate = max(1, int(num_rules_total * mutation_rate))    
#     # numeric_attributes = set(attributes) - categorical_features
#     # if numeric_attributes and data:
#     #     # Adiciona a opção 'from_classifier' apenas se houver features numéricas.
#     #     # Adicionado duas vezes para manter a probabilidade original da sua implementação.
#     #     possible_mutation_types.extend(['from_classifier', 'from_classifier'])
        
#     # 3. (NOVO) Cria um dispatcher para chamar as funções de mutação
#     mutation_dispatcher = {
#         'operator': _mutate_operator,
#         'value': _mutate_value,
#         'subtree': _mutate_subtree,
#         'from_classifier': _mutate_from_classifier
#     }
#     for i in range(min(num_rules_to_mutate, num_rules_total)):
#         rule_info = sorted_rules[i]
#         rule_to_mutate = rule_info['rule_obj']
#         class_label = rule_info['class_label']
#         rule_idx = rule_info['rule_idx']   
    
#     # 4. Loop principal de mutação, focado nas piores regras
#     mutated_indices = set()
#     for i in range(min(num_rules_to_mutate, len(sorted_rules))):
#         (class_label, rule_idx), _ = sorted_rules[i]
#         rule_tuple = (class_label, rule_idx)
        
#         # Validação para garantir que o índice da regra ainda é válido
#         if rule_idx >= len(individual.rules[class_label]) or rule_tuple in mutated_indices:
#             continue

#         # Seleciona o tipo de mutação a partir da lista de tipos válidos
#         #mutation_type = random.choice(possible_mutation_types)
#         mutation_type = None
#         # Verifica se a mutação inteligente é possível e se deve ser usada
#         if 'from_classifier' in possible_mutation_types and random.random() < intelligent_mutation_prob:
#             mutation_type = 'from_classifier'
#         else:
#             # Se não, escolhe uma das mutações "burras"
#             dumb_mutations = [mtype for mtype in possible_mutation_types if mtype != 'from_classifier']
#             if dumb_mutations:
#                 mutation_type = random.choice(dumb_mutations)

#         if mutation_type is None:
#             continue # Pula se nenhuma mutação for aplicável        
#         logging.debug(f"Mutating rule {rule_idx} class {class_label} (type: {mutation_type})")
        
#         rule_tree_to_mutate = individual.rules[class_label][rule_idx]
#         node_to_mutate = select_random_node(rule_tree_to_mutate.root)

#         # Argumentos para as funções auxiliares
#         context_args = {
#             "node": node_to_mutate, "rule_tree": rule_tree_to_mutate, "class_label": class_label,
#             "individual_to_mutate": individual, "max_depth": max_depth, "attributes": attributes,
#             "value_ranges": value_ranges, "category_values": category_values,
#             "categorical_features": categorical_features, "data": data, "target": target
#         }

#         # Chama a função de mutação apropriada através do dispatcher
#         mutation_function = mutation_dispatcher.get(mutation_type)
#         if mutation_function:
#             mutation_function(**context_args)
        
#         mutated_indices.add(rule_tuple)

#     # 5. Mutações Estruturais (adicionar/remover regras, mudar classe padrão)
#     if random.random() < (mutation_rate / 2):
#         target_class = random.choice(classes)
#         if len(individual.rules.get(target_class, [])) < max_rules_per_class:
#             try:
#                 new_rule = RuleTree(max_depth, attributes, value_ranges, category_values, categorical_features)
#                 individual.rules[target_class].append(new_rule)
#                 logging.debug(f"Added new random rule to class {target_class}")
#             except Exception as e:
#                 logging.error(f"Failed to generate/add new rule: {e}")

#     if random.random() < (mutation_rate / 2):
#         original_default = individual.default_class
#         possible_defaults = [c for c in classes if c != original_default]
#         if possible_defaults:
#             individual.default_class = random.choice(possible_defaults)
#             logging.debug(f"Mutated default class from {original_default} to {individual.default_class}")

#     # 6. Limpeza e Simplificação final do indivíduo
#     individual.remove_duplicate_rules()
#     for class_label, rule_list in individual.rules.items():
#         for i, rule_tree in enumerate(rule_list):
#             try:
#                 rule_tree.simplify()
#             except Exception as e:
#                 logging.error(f"Error simplifying rule {i} class {class_label}: {e}")


def mutate_individual(
    individual: Individual,
    mutation_rate: float,
    max_depth: int,
    intelligent_mutation_prob: float,
    attributes: List[str],
    value_ranges: Dict,
    category_values: Dict,
    categorical_features: Set[str],
    classes: List[Any],
    max_rules_per_class: int,
    data: List[Dict],
    target: List[Any],
    force_gene_therapy: bool = False
) -> None:
    """
    Muta um indivíduo usando uma estratégia de "Terapia Gênica":
    1. Identifica as classes fracas (sem cobertura de regras de qualidade).
    2. Foca os operadores de mutação exclusivamente nas regras dessas classes.
    3. Aplica mutações estruturais (adicionar regra, mudar default).
    4. Realiza a limpeza e simplificação final.
    """
    # --- Passo 1: Coletar e Preparar Regras para Mutação ---
    all_rules_with_context = []
    for class_label, rule_list in individual.rules.items():
        for rule_idx, rule in enumerate(rule_list):
            all_rules_with_context.append({
                "rule_obj": rule, "class_label": class_label, "rule_idx": rule_idx
            })

    if not all_rules_with_context:
        logging.debug("Indivíduo sem regras para mutar, pulando para mutações estruturais.")
    else:
        # # 1. Diagnosticar Fraquezas: identifica as classes que não estão em 'covered_classes'.
        # all_classes_set = set(individual.rules.keys())
        # strong_classes = individual.covered_classes
        # weak_classes = all_classes_set - strong_classes

        # # 2. Foco Cirúrgico: o pool de mutação conterá APENAS regras das classes fracas.
        # rules_to_mutate_pool = [
        #     rule_info for rule_info in all_rules_with_context
        #     if rule_info['class_label'] in weak_classes
        # ]        

        # if not rules_to_mutate_pool:
        #     logging.debug("Nenhuma classe fraca. Mutando as regras de menor qualidade geral.")
        #     all_rules_with_context.sort(key=lambda r: r['rule_obj'].quality_score)
        #     rules_to_mutate_pool = all_rules_with_context

        all_rules_with_context.sort(key=lambda r: r['rule_obj'].quality_score)

        rules_to_mutate_pool = all_rules_with_context

        num_rules_to_mutate = max(1, int(len(rules_to_mutate_pool) * mutation_rate)) if rules_to_mutate_pool else 0


        # --- Passo 2: Definir Mutações Possíveis e o Dispatcher ---
        possible_mutation_types = ['operator', 'value', 'subtree']
        if data and attributes: possible_mutation_types.append('from_classifier')
            
        mutation_dispatcher = {
            'operator': _mutate_operator, 'value': _mutate_value,
            'subtree': _mutate_subtree, 'from_classifier': _mutate_from_classifier_xg
        }
        
        # --- Passo 3: Loop de Mutação sobre o Pool Focado ---
        rules_to_mutate_pool.sort(key=lambda r: r['rule_obj'].quality_score)
        
        for i in range(min(num_rules_to_mutate, len(rules_to_mutate_pool))):
            rule_info = rules_to_mutate_pool[i]
            rule_to_mutate = rule_info['rule_obj']
            class_label = rule_info['class_label']
            rule_idx = rule_info['rule_idx']  # CORRIGIDO: Extrair rule_idx de rule_info

            # (Lógica de seleção do tipo de mutação)
            mutation_type = None

            # if force_gene_therapy and 'from_classifier' in possible_mutation_types:# and random.random() < intelligent_mutation_prob:
            #     mutation_type = 'from_classifier'
            #     ##logging.info(f"  Terapia Gênica Ativa: Forçando mutação 'from_classifier' na regra da classe {class_label}.")
            # else:
            #     other_mutations = [m for m in possible_mutation_types if m != 'from_classifier']
            #     if other_mutations: mutation_type = random.choice(other_mutations)

            use_gene_therapy = (force_gene_therapy or random.random() < intelligent_mutation_prob)
            
            if use_gene_therapy:
                mutation_type = 'from_classifier'
            else:
                mutation_type = random.choice(['operator', 'value', 'subtree'])
            
            logging.debug(f"Mutando regra {rule_idx} da classe {class_label} (Qualidade: {rule_to_mutate.quality_score:.4f}) com tipo: {mutation_type}")

            if mutation_type is None: continue
                
            logging.debug(f"Mutating rule for weak class {class_label} (Quality: {rule_to_mutate.quality_score:.4f}) with type: {mutation_type}")

            # --- Bloco IF/ELIF/ELSE para Execução da Mutação ---
            if mutation_type == 'from_classifier':
                base_info = {"attributes": attributes, "value_ranges": value_ranges, "category_values": category_values, "categorical_features": categorical_features}
                new_rule = _create_specialist_rule(class_label, data, target, base_info)
                if new_rule:
                    individual.rules[class_label][rule_idx] = new_rule
            
            elif mutation_type == 'subtree':
                _mutate_subtree(
                    rule_to_mutate,
                    max_depth,
                    attributes,
                    value_ranges,
                    category_values,
                    categorical_features
                )
            
            else: # Operator ou Value
                node_to_mutate = select_random_node(rule_to_mutate.root)
                if node_to_mutate and node_to_mutate.is_leaf():
                    if mutation_type == 'operator':
                        _mutate_operator(node_to_mutate, categorical_features)  # CORRIGIDO: _mutate_operator em vez de _mutate_operator_node
                    elif mutation_type == 'value':
                        _mutate_value(node_to_mutate, value_ranges, category_values, categorical_features)  # CORRIGIDO: _mutate_value em vez de _mutate_value_node
            
            # (Lógica do dispatcher para chamar a função de mutação correta)
            # node_to_mutate = select_random_node(rule_to_mutate.root)
            # context_args = {
            #     "node": node_to_mutate, "rule_tree": rule_to_mutate, "class_label": class_label,
            #     "individual_to_mutate": individual, "max_depth": max_depth, "attributes": attributes,
            #     "value_ranges": value_ranges, "category_values": category_values,
            #     "categorical_features": categorical_features, "data": data, "target": target
            # }
            # mutation_function = mutation_dispatcher.get(mutation_type)
            # if mutation_function: mutation_function(**context_args)

    # <<< CÓDIGO COMPLETO PARA MUTAÇÕES ESTRUTURAIS >>>
    # --- Passo 4: Mutações Estruturais ---
    if random.random() < (mutation_rate / 2):
        target_class = random.choice(classes)
        if len(individual.rules.get(target_class, [])) < max_rules_per_class:
            try:
                base_info = {
                    "attributes": attributes, "value_ranges": value_ranges,
                    "category_values": category_values, "categorical_features": categorical_features
                }
                new_rule = RuleTree(max_depth=max_depth, **base_info)
                individual.rules[target_class].append(new_rule)
                logging.debug(f"Added new random rule to class {target_class}")
            except Exception as e:
                logging.error(f"Failed to generate/add new rule: {e}")

    if random.random() < (mutation_rate / 2):
        original_default = individual.default_class
        possible_defaults = [c for c in classes if c != original_default]
        if possible_defaults:
            individual.default_class = random.choice(possible_defaults)
            logging.debug(f"Mutated default class from {original_default} to {individual.default_class}")

    # --- Passo 5: Limpeza e Simplificação Final ---
    individual.remove_duplicate_rules()
    for rule_list in individual.rules.values():
        for rule_tree in rule_list:
            try:
                rule_tree.simplify()
            except Exception as e:
                logging.error(f"Error simplifying rule: {e}")

# Em ga_operators.py, substitua a função mutate_individual inteira por esta

# def mutate_individual(
#     individual: Individual,
#     mutation_rate: float,
#     max_depth: int,
#     intelligent_mutation_prob: float,
#     attributes: List[str],
#     value_ranges: Dict,
#     category_values: Dict,
#     categorical_features: Set[str],
#     classes: List[Any],
#     max_rules_per_class: int,
#     data: List[Dict],
#     target: List[Any]
# ) -> None:
#     """
#     Muta um indivíduo, focando nas regras de pior performance (menor quality_score)
#     e usando operadores inteligentes de forma guiada. Modifica o indivíduo "in-place".
#     """
#     # --- Passo 1: Coletar e Ordenar Regras por Qualidade ---
#     all_rules_with_context = []
#     for class_label, rule_list in individual.rules.items():
#         for rule_idx, rule in enumerate(rule_list):
#             all_rules_with_context.append({
#                 "rule_obj": rule,
#                 "class_label": class_label,
#                 "rule_idx": rule_idx
#             })

#     if not all_rules_with_context:
#         logging.debug("Indivíduo sem regras para mutar, pulando para mutações estruturais.")
#     else:
#         # Ordena as regras pela sua qualidade (quality_score), da pior para a melhor.
#         sorted_rules = sorted(all_rules_with_context, key=lambda r: r['rule_obj'].quality_score)
        
#         num_rules_total = len(sorted_rules)
#         num_rules_to_mutate = max(1, int(num_rules_total * mutation_rate))
        
#         # --- Passo 2: Definir Mutações Possíveis e o Dispatcher ---
#         possible_mutation_types = ['operator', 'value', 'subtree']
#         # Verifica se a mutação 'from_classifier' é aplicável (se há dados e features)
#         if data and attributes:
#             possible_mutation_types.append('from_classifier')
            
#         mutation_dispatcher = {
#             'operator': _mutate_operator,
#             'value': _mutate_value,
#             'subtree': _mutate_subtree,
#             'from_classifier': _mutate_from_classifier
#         }
        
#         # --- Passo 3: Loop de Mutação sobre as Piores Regras ---
#         for i in range(min(num_rules_to_mutate, num_rules_total)):
#             # Acessa as informações da regra de forma correta
#             rule_info = sorted_rules[i]
#             rule_to_mutate = rule_info['rule_obj']
#             class_label = rule_info['class_label']
            
#             # Seleciona o tipo de mutação de forma inteligente
#             mutation_type = None
#             if 'from_classifier' in possible_mutation_types and random.random() < intelligent_mutation_prob:
#                 mutation_type = 'from_classifier'
#             else:
#                 other_mutations = [mtype for mtype in possible_mutation_types if mtype != 'from_classifier']
#                 if other_mutations:
#                     mutation_type = random.choice(other_mutations)

#             if mutation_type is None:
#                 continue
                
#             logging.debug(f"Mutating rule for class {class_label} (Quality: {rule_to_mutate.quality_score:.4f}) with type: {mutation_type}")
            
#             # Prepara os argumentos e chama a função de mutação correta
#             node_to_mutate = select_random_node(rule_to_mutate.root)
#             context_args = {
#                 "node": node_to_mutate, "rule_tree": rule_to_mutate, "class_label": class_label,
#                 "individual_to_mutate": individual, "max_depth": max_depth, "attributes": attributes,
#                 "value_ranges": value_ranges, "category_values": category_values,
#                 "categorical_features": categorical_features, "data": data, "target": target
#             }
            
#             mutation_function = mutation_dispatcher.get(mutation_type)
#             if mutation_function:
#                 mutation_function(**context_args)

#     # --- Passo 4: Mutações Estruturais ---
#     if random.random() < (mutation_rate / 2):
#         target_class = random.choice(classes)
#         if len(individual.rules.get(target_class, [])) < max_rules_per_class:
#             try:
#                 base_info = {"attributes": attributes, "value_ranges": value_ranges, "category_values": category_values, "categorical_features": categorical_features}
#                 new_rule = RuleTree(max_depth, **base_info)
#                 individual.rules[target_class].append(new_rule)
#                 logging.debug(f"Added new random rule to class {target_class}")
#             except Exception as e:
#                 logging.error(f"Failed to generate/add new rule: {e}")

#     if random.random() < (mutation_rate / 2):
#         original_default = individual.default_class
#         possible_defaults = [c for c in classes if c != original_default]
#         if possible_defaults:
#             individual.default_class = random.choice(possible_defaults)
#             logging.debug(f"Mutated default class from {original_default} to {individual.default_class}")

#     # --- Passo 5: Limpeza e Simplificação Final ---
#     individual.remove_duplicate_rules()
#     for class_label, rule_list in individual.rules.items():
#         for i, rule_tree in enumerate(rule_list):
#             try:
#                 rule_tree.simplify()
#             except Exception as e:
#                 logging.error(f"Error simplifying rule {i} class {class_label}: {e}")


# ==============================================================================
# --- CROSSOVER BALANCEADO INTELIGENTE (NOVO) ---
# ==============================================================================

def evaluate_rule_quality(rule: RuleTree, data: List[Dict], target: List,
                         class_label: Any) -> float:
    """
    Avalia a qualidade de UMA regra individual com métricas balanceadas.

    Args:
        rule: Regra a ser avaliada
        data: Dados de treino (lista de dicts)
        target: Labels de treino
        class_label: Classe alvo da regra

    Returns:
        Score no intervalo [0.0, 1.0]

    Métricas:
    - Precision (50%): Das instâncias que a regra ativa, quantas são da classe correta?
    - Coverage (30%): Quantas instâncias da classe a regra cobre?
    - Balance (20%): Penaliza regras muito específicas (<1%) ou muito gerais (>90%)
    """
    activated_correct = 0
    activated_total = 0
    target_instances_covered = 0
    target_instances_total = sum(1 for y in target if y == class_label)

    # Se não há instâncias da classe, retorna score 0
    if target_instances_total == 0:
        return 0.0

    # Aplica a regra em cada instância
    for x, y in zip(data, target):
        try:
            activates = rule.evaluate(x)  # Retorna True/False

            if activates:
                activated_total += 1
                if y == class_label:
                    activated_correct += 1
                    target_instances_covered += 1
        except Exception as e:
            # Se houver erro na avaliação, ignora esta instância
            continue

    # 1. PRECISION: Evita regras que ativam errado
    precision = activated_correct / activated_total if activated_total > 0 else 0.0

    # 2. COVERAGE: Recompensa regras que cobrem instâncias da classe
    coverage = target_instances_covered / target_instances_total if target_instances_total > 0 else 0.0

    # 3. BALANCE PENALTY: Penaliza extremos
    activation_rate = activated_total / len(data) if len(data) > 0 else 0.0

    if activation_rate < 0.01:  # Muito específica (< 1% dos dados)
        balance_penalty = 0.5
    elif activation_rate > 0.90:  # Muito geral (> 90% dos dados)
        balance_penalty = 0.7
    else:
        balance_penalty = 1.0  # OK

    # SCORE FINAL: Média ponderada
    score = (0.50 * precision +
             0.30 * coverage +
             0.20 * balance_penalty)

    return score


def get_quality_ratio_adaptive(current_generation: int, max_generations: int,
                               config: Dict = None) -> float:
    """
    Retorna o ratio de qualidade/diversidade baseado na geração atual.

    Estratégia em 3 fases:
    - FASE 1 (Gen 1-20):    50% qualidade + 50% diversidade (EXPLORAÇÃO)
    - FASE 2 (Gen 21-60):   70% qualidade + 30% diversidade (BALANCEADO)
    - FASE 3 (Gen 61-max):  85% qualidade + 15% diversidade (EXPLOITAÇÃO)

    Args:
        current_generation: Geração atual (1-indexed)
        max_generations: Total de gerações configurado
        config: Configurações (opcional, usa defaults se None)

    Returns:
        quality_ratio ∈ [0.5, 0.85]
    """
    # Valores padrão
    phase1_ratio = 0.50
    phase2_ratio = 0.70
    phase3_ratio = 0.85
    phase1_end = 20
    phase2_end = 60

    # Sobrescreve com config se fornecido
    if config:
        phase1_ratio = config.get('balanced_crossover_quality_ratio_phase1', 0.50)
        phase2_ratio = config.get('balanced_crossover_quality_ratio_phase2', 0.70)
        phase3_ratio = config.get('balanced_crossover_quality_ratio_phase3', 0.85)
        phase1_end = config.get('balanced_crossover_phase1_end', 20)
        phase2_end = config.get('balanced_crossover_phase2_end', 60)

    if current_generation <= phase1_end:
        # Fase inicial: Exploração agressiva
        return phase1_ratio
    elif current_generation <= phase2_end:
        # Fase intermediária: Balanceado
        return phase2_ratio
    else:
        # Fase final: Exploitação dominante
        return phase3_ratio


def balanced_crossover(
    parent1: Individual,
    parent2: Individual,
    data: List[Dict],
    target: List,
    max_rules_per_class: int,
    current_generation: int,
    max_generations: int,
    classes: List[Any],
    max_depth: int,
    attributes: List[str],
    value_ranges: Dict,
    category_values: Dict,
    categorical_features: Set[str],
    config: Dict = None,
    **kwargs
) -> Individual:
    """
    Crossover Balanceado Adaptativo: 70% qualidade + 30% diversidade (adaptável).

    Processo:
    1. Avalia TODAS as regras dos pais individualmente
    2. Ranqueia regras por score (melhor → pior)
    3. Seleciona top N% por qualidade (exploitação)
    4. Seleciona (100-N)% aleatórias do restante (exploração)
    5. Preenche slots vazios com regras NOVAS (exploração pura)

    Args:
        parent1, parent2: Pais selecionados
        data, target: Dados de treino do chunk atual (para avaliação)
        max_rules_per_class: Número máximo de regras por classe
        current_generation: Geração atual (para ratio adaptativo)
        max_generations: Total de gerações (para ratio adaptativo)
        classes: Lista de classes do problema
        max_depth, attributes, value_ranges, category_values, categorical_features:
            Parâmetros para criar regras novas se necessário
        config: Configurações (opcional)

    Returns:
        Filho (Individual) com regras balanceadas
    """
    # 1. Determina ratio adaptativo
    quality_ratio = get_quality_ratio_adaptive(current_generation, max_generations, config)
    diversity_ratio = 1.0 - quality_ratio

    logging.debug(f"  Crossover Balanceado (Gen {current_generation}): "
                 f"{quality_ratio:.0%} qualidade + {diversity_ratio:.0%} diversidade")

    # 2. Cria filho vazio
    child = Individual(
        max_rules_per_class=max_rules_per_class,
        max_depth=max_depth,
        attributes=attributes,
        value_ranges=value_ranges,
        category_values=category_values,
        categorical_features=categorical_features,
        classes=classes
    )
    child.rules = {c: [] for c in classes}

    # 3. Para cada classe, cria regras balanceadas
    for class_label in classes:
        # 3a. Coleta regras dos pais
        rules_p1 = parent1.rules.get(class_label, [])
        rules_p2 = parent2.rules.get(class_label, [])

        if not rules_p1 and not rules_p2:
            # Nenhum pai tem regras para esta classe - cria regra nova
            base_attributes_info = {
                'max_depth': max_depth,
                'attributes': attributes,
                'value_ranges': value_ranges,
                'category_values': category_values,
                'categorical_features': categorical_features
            }
            new_rule = _create_specialist_rule(class_label, data, target, base_attributes_info)
            if new_rule:
                child.rules[class_label] = [new_rule]
            continue

        # 3b. Avalia TODAS as regras e ranqueia
        rules_with_scores = []

        for rule in rules_p1:
            try:
                score = evaluate_rule_quality(rule, data, target, class_label)
                rules_with_scores.append((score, rule, 'p1'))
            except Exception as e:
                logging.debug(f"    Erro ao avaliar regra do Pai 1: {e}")
                # Adiciona com score 0 para não perder a regra
                rules_with_scores.append((0.0, rule, 'p1'))

        for rule in rules_p2:
            try:
                score = evaluate_rule_quality(rule, data, target, class_label)
                rules_with_scores.append((score, rule, 'p2'))
            except Exception as e:
                logging.debug(f"    Erro ao avaliar regra do Pai 2: {e}")
                rules_with_scores.append((0.0, rule, 'p2'))

        # Ordena por score (melhor → pior)
        rules_with_scores.sort(key=lambda x: x[0], reverse=True)

        # 3c. Calcula quantas regras por categoria
        n_quality = int(max_rules_per_class * quality_ratio)  # Ex: 70% de 15 = 10
        n_diverse = max_rules_per_class - n_quality           # Ex: 30% de 15 = 5

        # 3d. SELEÇÃO POR QUALIDADE (Top N)
        selected_rules = []
        quality_rules = rules_with_scores[:n_quality]

        for score, rule, parent_id in quality_rules:
            selected_rules.append(copy.deepcopy(rule))

        # 3e. SELEÇÃO POR DIVERSIDADE (Amostra aleatória do restante)
        remaining_rules = rules_with_scores[n_quality:]

        if remaining_rules:
            # Amostra sem reposição do restante
            n_to_sample = min(n_diverse, len(remaining_rules))
            diverse_sample = random.sample(remaining_rules, n_to_sample)

            for score, rule, parent_id in diverse_sample:
                selected_rules.append(copy.deepcopy(rule))

        # 3f. PREENCHIMENTO INTELIGENTE (se ainda faltar regras)
        while len(selected_rules) < max_rules_per_class:
            # Cria regra nova especializada para esta classe
            base_attributes_info = {
                'max_depth': max_depth,
                'attributes': attributes,
                'value_ranges': value_ranges,
                'category_values': category_values,
                'categorical_features': categorical_features
            }
            new_rule = _create_specialist_rule(class_label, data, target, base_attributes_info)

            if new_rule:
                selected_rules.append(new_rule)
            else:
                # Se falhar em criar regra especializada, para de tentar
                break

        # 3g. Atribui regras ao filho
        child.rules[class_label] = selected_rules

        # Log para debug
        n_from_quality = min(n_quality, len(quality_rules))
        n_from_diversity = len(selected_rules) - n_from_quality if len(selected_rules) > n_from_quality else 0
        n_created = max_rules_per_class - len(selected_rules) if len(selected_rules) < max_rules_per_class else 0

        logging.debug(f"    Classe {class_label}: {len(selected_rules)} regras "
                     f"({n_from_quality} qualidade, {n_from_diversity} diversidade, {n_created} criadas)")

    # 4. Finaliza filho
    child.default_class = parent1.default_class if random.random() < 0.5 else parent2.default_class
    child.remove_duplicate_rules()

    return child
