# ga.py (Corrigido - Passando Coeficientes para initialize_population)

import random
import copy
import logging
import numpy as np
import pandas as pd
import multiprocessing
import os
from rule_tree import RuleTree
from rule_node import Node
from river import tree
import time
from typing import Dict, List, Any, Optional
from sklearn.tree import DecisionTreeClassifier
from metrics import calculate_gmean_contextual
# Import necessary components from other modules
from individual import Individual
import fitness as fitness_module
import ga_operators
from sklearn.metrics import f1_score
from utils import calculate_population_diversity
import hill_climbing_v2  # Hill Climbing Hierárquico v2.0


from collections import Counter

# ============================================================================
# SEEDING ADAPTATIVO (Fase 3): Estimativa de Complexidade
# ============================================================================

def estimate_chunk_complexity(train_data: List[Dict], train_target: List) -> tuple:
    """
    Estima a complexidade do chunk usando uma Decision Tree probe rápida.

    ESTRATÉGIA:
    - Treina DT rasa (depth=3) rapidamente
    - Usa accuracy como proxy de complexidade
    - Alta acc → Problema simples (padrões óbvios)
    - Baixa acc → Problema complexo (padrões difíceis)

    Args:
        train_data: Features do chunk
        train_target: Labels do chunk

    Returns:
        (complexity_level, probe_score, profile_params)
        - complexity_level: 'simple', 'medium', 'complex'
        - probe_score: accuracy da DT probe (0.0-1.0)
        - profile_params: dict com parâmetros sugeridos
    """
    df_train = pd.DataFrame(train_data)

    # DT probe rápida para estimativa
    dt_probe = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
    dt_probe.fit(df_train, train_target)
    probe_score = dt_probe.score(df_train, train_target)

    # Perfis adaptativos de seeding
    ADAPTIVE_SEEDING_PROFILES = {
        'simple': {
            'dt_seeding_ratio': 0.8,
            'dt_rule_injection_ratio': 1.0,
            'dt_seeding_depths': [4, 7, 10, 13],
            'description': 'Problema SIMPLES (DT probe ≥90%) - Seeding forte'
        },
        'medium': {
            'dt_seeding_ratio': 0.6,
            'dt_rule_injection_ratio': 0.6,
            'dt_seeding_depths': [5, 8, 10],
            'description': 'Problema MÉDIO (DT probe 75-90%) - Seeding moderado'
        },
        'complex': {
            'dt_seeding_ratio': 0.4,
            'dt_rule_injection_ratio': 0.4,
            'dt_seeding_depths': [3, 5, 7],
            'description': 'Problema COMPLEXO (DT probe <75%) - Seeding suave, exploração máxima'
        }
    }

    # Classifica complexidade baseado em thresholds
    if probe_score >= 0.90:
        complexity_level = 'simple'
    elif probe_score >= 0.75:
        complexity_level = 'medium'
    else:
        complexity_level = 'complex'

    profile = ADAPTIVE_SEEDING_PROFILES[complexity_level]

    logging.info(f"  -> Complexidade estimada: {complexity_level.upper()} (DT probe acc: {probe_score:.3f})")
    logging.info(f"     {profile['description']}")

    return complexity_level, probe_score, profile

# ============================================================================

def _extract_all_rules_from_dt(dt_model: DecisionTreeClassifier, feature_names: List[str], base_attributes_info: Dict) -> Dict[int, List[RuleTree]]:
    """Percorre uma Decision Tree e extrai TODOS os caminhos (regras)."""
    tree = dt_model.tree_
    gene_pool = {c: [] for c in dt_model.classes_}
    def recurse(node_id, conditions):
        if tree.children_left[node_id] == -1:
            predicted_class_idx = np.argmax(tree.value[node_id][0])
            predicted_class = dt_model.classes_[predicted_class_idx]
            if not conditions: return
            if len(conditions) == 1: rule_root = conditions[0]
            else:
                rule_root = Node(operator="AND", left=conditions[0], right=conditions[1])
                for i in range(2, len(conditions)): rule_root = Node(operator="AND", left=rule_root, right=conditions[i])
            rule_tree = RuleTree(max_depth=len(conditions), **base_attributes_info, root_node=rule_root)
            if rule_tree.is_valid_rule(): gene_pool[predicted_class].append(rule_tree)
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

# --- Função Worker para Avaliação Paralela ---
def evaluate_individual_fitness_parallel(worker_args):
    # ... (código como antes - sem alterações) ...
    individual, constant_args = worker_args; 
    individual_repr = f"Ind (Rules: {individual.count_total_rules()})"
    fitness_score, gmean, activation_rate = 0.0, 0.0, 0.0
    try:
        train_data = constant_args['train_data']
        train_target = constant_args['train_target']
        class_weights = constant_args['class_weights']
        regularization_coefficient = constant_args['regularization_coefficient']
        feature_penalty_coefficient = constant_args['feature_penalty_coefficient']
        reference_features = constant_args['reference_features']
        beta = constant_args['beta']
        previous_used_features = constant_args['previous_used_features']
        gamma = constant_args['gamma'] # Original gamma
        operator_penalty_coefficient = constant_args['operator_penalty_coefficient']
        threshold_penalty_coefficient = constant_args['threshold_penalty_coefficient']
        previous_operator_info = constant_args['previous_operator_info']
        operator_change_coefficient = constant_args['operator_change_coefficient'] # Original op_change_coeff
        attributes = constant_args['attributes']
        categorical_features = constant_args['categorical_features']
        reduce_change_penalties = constant_args.get('reduce_change_penalties', False) # <<< EXTRACT THE FLAG, default to False
        gmean_bonus_coefficient_ga = constant_args['gmean_bonus_coefficient']
        class_coverage_coefficient = constant_args['class_coverage_coefficient']


        metrics = fitness_module.calculate_fitness(
            individual, train_data, train_target,
            class_weights,
            regularization_coefficient, feature_penalty_coefficient,
            class_coverage_coefficient, gmean_bonus_coefficient_ga,
            operator_change_coefficient,
            gamma,
            beta,  # Pass original
            reference_features,
            previous_used_features,
            previous_operator_info,
            reduce_change_penalties
        )

        # Extrai os valores para o retorno da função
        fitness_score = metrics.get('fitness', -float('inf'))
        gmean = metrics.get('g_mean', 0.0)

        # --- PASSO 2: Atualização do Estado Interno do Indivíduo (LÓGICA PRESERVADA) ---
        # Esta chamada é crucial para o Crossover Adaptativo e outras lógicas.
        # Ela é executada após o cálculo das métricas principais.
        individual.update_rule_quality_scores(train_data, train_target)

        # --- PASSO 3: Cálculo de Métricas Adicionais (se necessário) ---
        # A taxa de ativação é uma métrica secundária que podemos calcular aqui.
        activation_rate = 0.0
        if train_data:
            activations = sum(1 for inst in train_data if individual._predict_with_activation_check(inst)[1])
            activation_rate = activations / len(train_data) if len(train_data) > 0 else 0.0

        return (fitness_score, gmean, activation_rate)

    except Exception as e:
        logging.error(f"Worker {os.getpid()}: Erro crítico na avaliação do indivíduo: {e}", exc_info=True)
        return (-float('inf'), 0.0, 0.0)

def get_dynamic_mutation_rate(current_best_fitness, current_avg_fitness, current_worst_fitness, base_mutation_rate=0.1, max_mutation_rate=1.0):
    # ... (código como antes) ...
    if abs(current_best_fitness - current_worst_fitness) < 1e-9: return max_mutation_rate
    valid_spread = (current_best_fitness - current_worst_fitness); diversity_metric = (current_best_fitness - current_avg_fitness) / valid_spread if valid_spread > 1e-9 else 0
    mutation_rate = base_mutation_rate + (1.0 - diversity_metric) * (max_mutation_rate - base_mutation_rate); return max(base_mutation_rate, min(max_mutation_rate, mutation_rate))

def get_dynamic_tournament_size(current_best_fitness, current_avg_fitness, current_worst_fitness, min_tournament_size=2, max_tournament_size=5):
    # ... (código como antes) ...
    if abs(current_best_fitness - current_worst_fitness) < 1e-9: return min_tournament_size
    valid_spread = (current_best_fitness - current_worst_fitness); diversity_metric = (current_best_fitness - current_avg_fitness) / valid_spread if valid_spread > 1e-9 else 0
    size = min_tournament_size + int(diversity_metric * (max_tournament_size - min_tournament_size)); return max(min_tournament_size, min(max_tournament_size, size))

def get_dynamic_crossover_type(current_generation, max_generations, transition_point=0.5):
    # ... (código como antes, corrigido) ...
    return "node" if (current_generation / max_generations) < transition_point else "rule"


# --- Inicialização da População (Assinatura e Chamadas Corrigidas) ---
# Em ga.py, substitua a função initialize_population inteira pela versão abaixo.

# Em ga.py, substitua a função initialize_population inteira por esta.

def initialize_population(
    population_size, max_rules_per_class, max_depth,
    attributes, value_ranges, category_values, categorical_features,
    classes, train_data, train_target,
    regularization_coefficient, feature_penalty_coefficient,
    operator_penalty_coefficient, threshold_penalty_coefficient, intelligent_mutation_rate,
    previous_rules_pop=None, best_ever_memory=None,
    best_individual_from_previous_chunk=None,
    performance_label='medium', prev_best_mutant_ratio=0.15,
    reference_features=None, beta=0.0,
    initialization_strategy='default',
    enable_dt_seeding_on_init_config=False,
    dt_seeding_ratio_on_init_config=0.0,
    dt_seeding_sample_size_on_init_config=200,
    # Parâmetros não utilizados diretamente na nova versão, mas mantidos por compatibilidade de chamada
    dt_seeding_rules_to_replace_config=1,
    recovery_aggressive_mutant_ratio_config=0.3,
    historical_reference_dataset=None,
    dt_rule_injection_ratio_config=1.0,  # Seeding Probabilístico: fração de regras a injetar
    enable_adaptive_seeding_config=False,  # Seeding Adaptativo: habilita ajuste automático
    drift_severity='NONE',  # Severidade do drift detectado
    **kwargs
):
    """
    Inicializa a população de forma inteligente, utilizando uma estratégia de 
    "Gene Pool" com "Focus Training" e mesclando fontes (memória, mutantes, etc.)
    com base em uma "receita" definida pela estratégia e performance.
    """
    # --- Funções Auxiliares Internas ---

    # def _build_gene_pool(rules_per_class_in_pool=10, max_attempts_per_rule=100):
    #     """
    #     A "Fábrica de Regras". Treina uma HT especialista para cada classe e
    #     extrai as melhores regras para formar um pool de genes de alta qualidade.
    #     """
    #     logging.info("Construindo o 'Pool de Genes' com a estratégia 'Focus Training'...")
    #     gene_pool = {c: [] for c in classes}
    #     base_attributes_info = {
    #         "attributes": attributes, "value_ranges": value_ranges,
    #         "category_values": category_values, "categorical_features": categorical_features
    #     }

    #     for target_class in classes:
    #         positive_examples = [train_data[i] for i, label in enumerate(train_target) if label == target_class]
    #         if not positive_examples:
    #             logging.warning(f"Nenhuma instância da Classe {target_class} para o Focus Training.")
    #             continue
            
    #         negative_examples = [train_data[i] for i, label in enumerate(train_target) if label != target_class]
    #         num_negatives_to_sample = min(len(positive_examples) * 2, len(negative_examples))
            
    #         if num_negatives_to_sample == 0: continue

    #         focused_x = positive_examples + random.sample(negative_examples, num_negatives_to_sample)
    #         focused_y = [1] * len(positive_examples) + [0] * num_negatives_to_sample
            
    #         ht_specialist = tree.HoeffdingTreeClassifier(
    #             grace_period=50, delta=0.01,
    #             nominal_attributes=[attr for attr in attributes if attr in categorical_features]
    #         )
    #         for i in range(len(focused_x)):
    #             ht_specialist.learn_one(focused_x[i], focused_y[i])

    #         for _ in range(max_attempts_per_rule):
    #             if len(gene_pool[target_class]) >= rules_per_class_in_pool: break
    #             candidate_rule = ga_operators._extract_single_rule_from_river_ht(
    #                 ht_specialist, 1, base_attributes_info # Busca regras para a classe positiva '1'
    #             )
    #             if candidate_rule:
    #                 gene_pool[target_class].append(candidate_rule)
            
    #         logging.debug(f"  -> Pool para Classe {target_class}: {len(gene_pool[target_class])}/{rules_per_class_in_pool} regras extraídas.")
    #     return gene_pool

    # def _build_gene_pool(rules_per_class_in_pool=10, max_attempts_per_class=20):
    #     """
    #     A "Fábrica de Regras" v2. Treina uma Decision Tree (sklearn) especialista 
    #     para cada classe e extrai as melhores regras para o pool de genes.
    #     """
    #     logging.info("Construindo o 'Pool de Genes' com a estratégia 'Focus Training' usando Decision Tree...")
    #     gene_pool = {c: [] for c in classes}
    #     base_attributes_info = {
    #         "attributes": attributes, "value_ranges": value_ranges,
    #         "category_values": category_values, "categorical_features": categorical_features
    #     }

    #     # Parâmetros para a Decision Tree guia
    #     dt_params = {
    #         'max_depth': 5,
    #         'min_samples_leaf': 10,
    #         'class_weight': 'balanced'
    #     }

    #     for target_class in classes:
    #         source_data_for_seeding = train_data
    #         source_target_for_seeding = train_target
    #         data_source_type = "Chunk Atual"

    #         positive_indices_current_chunk = [i for i, label in enumerate(source_data_for_seeding) if label == target_class]

    #         if not positive_indices_current_chunk and historical_reference_dataset:
    #             logging.warning(f"Nenhuma instância da Classe {target_class} no chunk atual. Usando dataset de referência histórico.")
    #             hist_X, hist_y = zip(*historical_reference_dataset)
    #             source_data_for_seeding = list(hist_X)
    #             source_target_for_seeding = list(hist_y)
    #             data_source_type = "Histórico"


    #         # 1. Cria o dataset binário para "Focus Training"
    #         positive_indices = [i for i, label in enumerate(source_data_for_seeding) if label == target_class]
    #         if not positive_indices:
    #             logging.warning(f"Nenhuma instância da Classe {target_class} encontrada (Fonte: {train_data}). Seeding para esta classe será pulado.")
    #             continue
            
    #         negative_indices = [i for i, label in enumerate(train_data) if label != target_class]
    #         num_negatives = min(len(positive_indices) * 2, len(negative_indices))
    #         if num_negatives == 0: continue

    #         focused_indices = positive_indices + random.sample(negative_indices, num_negatives)
            
    #         X_focused = [source_data_for_seeding[i] for i in focused_indices]
    #         y_focused = [1 if source_target_for_seeding[i] == target_class else 0 for i in focused_indices]

    #         # Converte para formato numérico que o sklearn espera
    #         X_df_focused = pd.DataFrame(X_focused).apply(pd.to_numeric, errors='coerce').fillna(0)
            
    #         # 2. Treina a Decision Tree especialista
    #         dt_specialist = DecisionTreeClassifier(random_state=random.randint(0, 10000), **dt_params)
    #         dt_specialist.fit(X_df_focused, y_focused)
            
    #         # 3. Extrai múltiplos caminhos promissores
    #         for _ in range(max_attempts_per_class):
    #             if len(gene_pool[target_class]) >= rules_per_class_in_pool: break
                
    #             # A extração de regras de uma DT do sklearn é mais complexa.
    #             # Usaremos a lógica que já temos no operador de mutação.
    #             # ga_operators precisa ter as funções _find_promising_path_in_dt e _extract_single_rule_from_dt_path
    #             try:
    #                 promising_path = ga_operators.find_promising_path_in_dt(dt_specialist, target_class=1) # Procura a classe positiva
    #                 if promising_path:
    #                     candidate_rule = ga_operators._extract_single_rule_from_dt_path(
    #                         dt_model=dt_specialist,
    #                         path_indices=promising_path,
    #                         feature_names_for_dt=list(X_df_focused.columns),
    #                         target_class_of_rule=target_class, # A classe original
    #                         max_rule_depth_extracted=dt_params['max_depth'],
    #                         base_attributes_info=base_attributes_info
    #                     )
    #                     if candidate_rule:
    #                         gene_pool[target_class].append(candidate_rule)
    #             except Exception as e:
    #                 logging.error(f"Erro ao extrair regra da DT para a classe {target_class}: {e}")
    #                 break # Se a extração falhar, para de tentar para esta classe
                    
    #         logging.debug(f"  -> Pool (DT) para Classe {target_class}: {len(gene_pool[target_class])}/{rules_per_class_in_pool} regras extraídas.")
    #     return gene_pool

# ga.py -> dentro de initialize_population

    def _build_gene_pool(rules_per_class_in_pool=10, max_attempts_per_class=20):
        """
        A "Fábrica de Regras" v2. Treina uma Decision Tree (sklearn) especialista
        para cada classe e extrai as melhores regras para o pool de genes.
        Usa um dataset de referência histórico como fallback se a classe estiver ausente no chunk atual.
        """
        logging.info("Construindo o 'Pool de Genes' com a estratégia 'Focus Training' usando Decision Tree...")
        gene_pool = {c: [] for c in classes}
        base_attributes_info = {
            "attributes": attributes, "value_ranges": value_ranges,
            "category_values": category_values, "categorical_features": categorical_features
        }
        dt_params = {
            'max_depth': 5,
            'min_samples_leaf': 10,
            'class_weight': 'balanced'
        }

        for target_class in classes:
            # --- INÍCIO DA LÓGICA CORRIGIDA E ROBUSTA ---

            # 1. Define a fonte de dados primária como o chunk de treino atual.
            data_source = train_data
            target_source = train_target
            source_name = "Chunk Atual"

            # 2. Verifica se a classe existe na fonte primária (o chunk atual).
            #    A busca é feita na lista de rótulos `target_source`.
            positive_indices = [i for i, label in enumerate(target_source) if label == target_class]

            # 3. Se não encontrar E se houver um dataset histórico, tenta o fallback.
            if not positive_indices and historical_reference_dataset:
                logging.warning(f"Nenhuma instância da Classe {target_class} no chunk atual. Usando dataset de referência histórico.")
                hist_X, hist_y = zip(*historical_reference_dataset)
                data_source = list(hist_X)
                target_source = list(hist_y)
                source_name = "Histórico"
                # RECALCULA os índices positivos usando a nova fonte de dados (histórica).
                positive_indices = [i for i, label in enumerate(target_source) if label == target_class]

            # 4. Checagem final. Se ainda não houver exemplos positivos, pula esta classe.
            if not positive_indices:
                logging.warning(f"Nenhuma instância da Classe {target_class} encontrada (Fonte: {source_name}). Seeding para esta classe será pulado.")
                continue

            # --- FIM DA LÓGICA CORRIGIDA E ROBUSTA ---

            # A partir daqui, o código opera sobre `data_source` e `target_source`, que são as fontes corretas.
            negative_indices = [i for i, label in enumerate(target_source) if label != target_class]
            num_negatives = min(len(positive_indices) * 2, len(negative_indices))
            if num_negatives == 0: continue

            focused_indices = positive_indices + random.sample(negative_indices, num_negatives)

            X_focused = [data_source[i] for i in focused_indices]
            y_focused = [1 if target_source[i] == target_class else 0 for i in focused_indices]

            # Converte para formato numérico que o sklearn espera
            X_df_focused = pd.DataFrame(X_focused).apply(pd.to_numeric, errors='coerce').fillna(0)

            # Treina a Decision Tree especialista
            dt_specialist = DecisionTreeClassifier(random_state=random.randint(0, 10000), **dt_params)
            dt_specialist.fit(X_df_focused, y_focused)

            # Extrai múltiplos caminhos promissores
            for _ in range(max_attempts_per_class):
                if len(gene_pool[target_class]) >= rules_per_class_in_pool: break
                try:
                    promising_path = ga_operators.find_promising_path_in_dt(dt_specialist, target_class=1) # Procura a classe positiva
                    if promising_path:
                        candidate_rule = ga_operators._extract_single_rule_from_dt_path(
                            dt_model=dt_specialist,
                            path_indices=promising_path,
                            feature_names_for_dt=list(X_df_focused.columns),
                            target_class_of_rule=target_class, # A classe original
                            max_rule_depth_extracted=dt_params['max_depth'],
                            base_attributes_info=base_attributes_info
                        )
                        if candidate_rule:
                            gene_pool[target_class].append(candidate_rule)
                except Exception as e:
                    logging.error(f"Erro ao extrair regra da DT para a classe {target_class}: {e}")
                    break

            logging.debug(f"  -> Pool (DT) para Classe {target_class}: {len(gene_pool[target_class])}/{rules_per_class_in_pool} regras extraídas.")
        return gene_pool

    def _create_individual_from_pool(gene_pool, random_fallback_ratio=0.2):
        """
        A "Montadora". Cria um indivíduo pegando regras do pool de genes.
        """
        new_ind = Individual(max_rules_per_class, max_depth, attributes, value_ranges, 
                             category_values, categorical_features, classes, train_target)
        new_ind.rules = {c: [] for c in classes}
        for target_class in classes:
            pool_for_class = gene_pool.get(target_class, [])
            min_rules = max(1, max_rules_per_class // 2)
            num_rules_to_add = random.randint(min_rules, max_rules_per_class)

            for _ in range(num_rules_to_add):
                if random.random() < random_fallback_ratio or not pool_for_class:
                    rule = RuleTree(max_depth, attributes, value_ranges, category_values, categorical_features)
                else:
                    rule = copy.deepcopy(random.choice(pool_for_class))
                new_ind.rules[target_class].append(rule)
        return new_ind

    def _create_mutants(source_individual, num_mutants, mutation_rate):
        """Gera uma lista de mutantes a partir de um indivíduo fonte."""
        mutant_list = []
        if not source_individual or num_mutants <= 0: return mutant_list
        
        for _ in range(num_mutants):
            mutant = copy.deepcopy(source_individual)
            ga_operators.mutate_individual(
                mutant, mutation_rate, max_depth, intelligent_mutation_rate, attributes, value_ranges,
                category_values, categorical_features, classes, max_rules_per_class, train_data, train_target)
            mutant_list.append(mutant)            
        return mutant_list

    # --------------------------------------------------------------------------------
    # --- INÍCIO DA LÓGICA PRINCIPAL ---
    # --------------------------------------------------------------------------------
    
    logging.info(f"Initializing population with strategy '{initialization_strategy}' and performance '{performance_label}'.")
    
    # ETAPA 1: Construir o "Gene Pool" uma única vez, se o seeding estiver habilitado.
    gene_pool = {}
    if enable_dt_seeding_on_init_config:
        gene_pool = _build_gene_pool()

    population = []
    recipe = {}

    # # ETAPA 2: Definir a "Receita" para a composição da população
    # if initialization_strategy == "full_random":
    #     seeding_ratio = dt_seeding_ratio_on_init_config if enable_dt_seeding_on_init_config else 0.0
    #     recipe = {'seeded': seeding_ratio, 'random': 1.0 - seeding_ratio}
    #     #recipe = {'seeded': 0.5, 'random': 0.5}
    # elif initialization_strategy == "diversify":
    #     recipe = {
    #         'seeded': dt_seeding_ratio_on_init_config,
    #         'mutants': recovery_aggressive_mutant_ratio_config,
    #     }
    #     recipe['random'] = 1.0 - sum(recipe.values())
    if initialization_strategy in ['full_random', 'diversify']:
        logging.info("Estratégia de reset ativada. Usando 'Seeding Multi-Profundidade'.")

        # --- SEEDING ADAPTATIVO (Fase 3): Ajusta parâmetros baseado em complexidade ---
        if enable_adaptive_seeding_config:
            logging.info("  -> SEEDING ADAPTATIVO ATIVADO: Estimando complexidade do chunk...")
            complexity_level, probe_score, adaptive_profile = estimate_chunk_complexity(train_data, train_target)

            # Sobrescreve parâmetros fixos com perfil adaptativo
            dt_seeding_ratio_on_init_config = adaptive_profile['dt_seeding_ratio']
            dt_rule_injection_ratio_config = adaptive_profile['dt_rule_injection_ratio']
            dt_seeding_depths = adaptive_profile['dt_seeding_depths']

            # CORREÇÃO: Aumenta seeding intensivo para 85% em drift SEVERE
            if drift_severity == 'SEVERE':
                dt_seeding_ratio_on_init_config = 0.85
                dt_rule_injection_ratio_config = 0.90
                logging.info(f"  -> SEVERE DRIFT DETECTED: Seeding INTENSIVO ativado (85% seeding, 90% injection)")

            logging.info(f"     Parâmetros adaptativos: seeding_ratio={dt_seeding_ratio_on_init_config}, "
                        f"injection_ratio={dt_rule_injection_ratio_config}, depths={dt_seeding_depths}")
        else:
            # Extrai parâmetros de seeding de kwargs (modo manual)
            dt_seeding_depths = kwargs.get('dt_seeding_depths_on_init_config', [5, 10])
            if not dt_seeding_depths:
                dt_seeding_depths = [5, 10]  # Fallback padrão

        if dt_rule_injection_ratio_config < 1.0:
            logging.info(f"  -> Seeding Probabilístico ATIVADO: Injetando {dt_rule_injection_ratio_config*100:.0f}% das regras DT")

        # Define base_attributes_info para uso nas funções de extração
        base_attributes_info = {
            "attributes": attributes,
            "value_ranges": value_ranges,
            "category_values": category_values,
            "categorical_features": categorical_features
        }

        # Define seeded_ratio baseado na configuração (possivelmente adaptada)
        seeded_ratio = dt_seeding_ratio_on_init_config if enable_dt_seeding_on_init_config else 0.0

        population = []
        df_train = pd.DataFrame(train_data)

        # Treinar portfólio de modelos-guia
        generalist_models = {}
        for depth in dt_seeding_depths:
            dt_model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=42, class_weight='balanced')
            dt_model.fit(df_train, train_target)
            generalist_models[depth] = dt_model
        
        # Gerar clones mestres
        master_clones = {}
        for depth, model in generalist_models.items():
            gene_pool = _extract_all_rules_from_dt(model, list(df_train.columns), base_attributes_info)

            # --- Seeding Probabilístico: Seleciona apenas uma fração das regras ---
            if dt_rule_injection_ratio_config < 1.0:
                filtered_gene_pool = {}
                for cls in gene_pool:
                    rules_for_class = gene_pool[cls]
                    num_rules_to_keep = max(1, int(len(rules_for_class) * dt_rule_injection_ratio_config))
                    # Seleção aleatória para manter diversidade
                    filtered_gene_pool[cls] = random.sample(rules_for_class, min(num_rules_to_keep, len(rules_for_class)))
                gene_pool = filtered_gene_pool

            clone = Individual(max_rules_per_class=sum(len(r) for r in gene_pool.values()), max_depth=depth, **base_attributes_info, classes=classes, train_target=train_target, initialize_random_rules=False)
            clone.rules = gene_pool
            master_clones[depth] = clone

        # Construir população híbrida
        num_seeded_individuals = int(population_size * seeded_ratio)
        num_clones_per_depth = num_seeded_individuals // len(dt_seeding_depths) if dt_seeding_depths else 0
        
        for depth, master_clone in master_clones.items():
            for _ in range(num_clones_per_depth):
                population.append(copy.deepcopy(master_clone))
                
        num_randoms = population_size - len(population)
        for _ in range(num_randoms):
            random_ind = Individual(max_rules_per_class=max_rules_per_class, max_depth=max_depth, **base_attributes_info, classes=classes, train_target=train_target, initialize_random_rules=True)
            population.append(random_ind)
            
        logging.info(f"População de reset criada: {len(population)} indivíduos ({len(population) - num_randoms} semeados, {num_randoms} aleatórios).")
        return population    
    else:  # "default"
        if performance_label == 'good':
            recipe = {'seeded': 0.1, 'mutants': 0.15, 'memory': 0.3, 'prev_pop': 0.25, 'random': 0.2}
        elif performance_label == 'medium':
            recipe = {'seeded': 0.2, 'mutants': prev_best_mutant_ratio, 'memory': 0.2, 'prev_pop': 0.2}
            recipe['random'] = 1.0 - sum(recipe.values())
        else:  # 'bad'
            recipe = {'seeded': 0.5, 'mutants': 0.1, 'memory': 0.05, 'prev_pop': 0.05, 'random': 0.3}

        # DRIFT SEVERITY OVERRIDE: Ajusta recipe baseado em severidade do drift
        # Obtém drift_severity de kwargs (passado via run_genetic_algorithm)
        drift_severity = kwargs.get('drift_severity', 'NONE')

        if drift_severity == 'SEVERE':
            # RESET TOTAL: Muito seeding, zero herança
            recipe = {'seeded': 0.6, 'mutants': 0.1, 'memory': 0.0, 'prev_pop': 0.0, 'random': 0.3}
            logging.warning(f"🔴 Recipe adjusted for SEVERE drift: {recipe}")
        elif drift_severity == 'MODERATE':
            # RESET PARCIAL: Aumenta seeding, reduz herança
            recipe = {'seeded': 0.3, 'mutants': 0.15, 'memory': 0.1, 'prev_pop': 0.1, 'random': 0.35}
            logging.info(f"🟡 Recipe adjusted for MODERATE drift: {recipe}")
        elif drift_severity == 'MILD':
            # MANTÉM recipe original
            logging.info(f"🟢 Recipe kept for MILD drift: {recipe}")
        # Se STABLE ou NONE, usa recipe original baseado em performance_label

    # ETAPA 3: Construir a População com base na Receita
    logging.info("Building population based on recipe...")
    
    memory_pool = best_ever_memory or []
    prev_pop_pool = previous_rules_pop or []
    counts = {k: 0 for k in ['elites', 'mutants', 'seeded', 'memory', 'prev_pop', 'random']}

    if best_individual_from_previous_chunk and initialization_strategy != 'full_random':
        population.append(copy.deepcopy(best_individual_from_previous_chunk))
        counts['elites'] = 1
    
    while len(population) < population_size:
        # 1. Mutantes
        num_mutants = int(population_size * recipe.get('mutants', 0.0))
        if counts['mutants'] < num_mutants and best_individual_from_previous_chunk:
            rate = 0.5 if initialization_strategy == 'diversify' else 0.2
            new_mutants = _create_mutants(best_individual_from_previous_chunk, num_mutants - counts['mutants'], rate)
            population.extend(new_mutants)
            counts['mutants'] += len(new_mutants)
        
        # 2. Indivíduos Semeados (a partir do Gene Pool)
        num_seeded = int(population_size * recipe.get('seeded', 0.0))
        if counts['seeded'] < num_seeded and enable_dt_seeding_on_init_config and gene_pool:
            for _ in range(num_seeded - counts['seeded']):
                if len(population) >= population_size: break
                population.append(_create_individual_from_pool(gene_pool))
                counts['seeded'] += 1
        
        # 3. Indivíduos da Memória
        num_memory = int(population_size * recipe.get('memory', 0.0))
        if counts['memory'] < num_memory and memory_pool:
            needed = num_memory - counts['memory']
            selected = random.sample(memory_pool, min(needed, len(memory_pool)))
            population.extend(copy.deepcopy(ind) for ind in selected)
            counts['memory'] += len(selected)

        # 4. Indivíduos da População Anterior
        num_prev = int(population_size * recipe.get('prev_pop', 0.0))
        if counts['prev_pop'] < num_prev and prev_pop_pool:
            needed = num_prev - counts['prev_pop']
            selected = random.sample(prev_pop_pool, min(needed, len(prev_pop_pool)))
            population.extend(copy.deepcopy(ind) for ind in selected)
            counts['prev_pop'] += len(selected)
        
        # 5. Preencher o restante com Aleatórios
        if len(population) < population_size:
            num_random_fill = population_size - len(population)
            for _ in range(num_random_fill):
                population.append(Individual(max_rules_per_class, max_depth, attributes, value_ranges, category_values, categorical_features, classes, train_target))
            counts['random'] += num_random_fill
        
        # Garante que o loop termine, pois todas as categorias foram processadas
        break

    final_population = population[:population_size]
    for ind in final_population:
        ind.fitness = -float('inf')
            
    logging.info(
        f"Initialized population ({len(final_population)} individuals): " 
        f"{counts['elites']} prev_best, {counts['mutants']} mutants, "
        f"{counts['seeded']} seeded, {counts['memory']} from memory, "
        f"{counts['prev_pop']} from prev_pop, {counts['random']} random."
    )
    
    return final_population

# --- Loop Principal do Algoritmo Genético (Chamada para Init Corrigida) ---
def run_genetic_algorithm(
        attributes, value_ranges, classes, class_weights: dict, train_data, train_target,
        categorical_features, category_values,
        class_coverage_coefficient_ga,
        max_rules_per_class, max_depth, # max_depth e max_rules_per_class já são os efetivos/adaptados
        population_size, max_generations, # max_generations já é o efetivo/adaptado
        elitism_rate,
        gmean_bonus_coefficient_ga,
        initial_tournament_size, final_tournament_size,
        regularization_coefficient, feature_penalty_coefficient,
        operator_penalty_coefficient, threshold_penalty_coefficient, intelligent_mutation_rate,
        operator_change_coefficient, gamma,
        previous_rules_pop=None, best_ever_memory=None, performance_label='medium',
        previous_used_features=None, previous_operator_info=None,
        best_individual_from_previous_chunk=None,
        early_stopping_patience=None,
        prev_best_mutant_ratio=0.15, # Usado pela estratégia 'default' para mutantes normais do melhor anterior
        parallel_enabled=True, num_workers=None,
        reduce_change_penalties_flag=False,
        mutation_override_config_ga=None,
        initialization_strategy='full_random',
        enable_dt_seeding_on_init_config_ga=False,
        dt_seeding_ratio_on_init_config_ga=0.0,
        dt_seeding_depths_on_init_config_ga=None,
        dt_seeding_sample_size_on_init_config_ga=200,
        dt_seeding_rules_to_replace_config_ga=1,
        recovery_aggressive_mutant_ratio_config_ga=0.3,
        historical_reference_dataset=None, # Para a estratégia "diversify"
        dt_rule_injection_ratio_config_ga=1.0,  # Seeding Probabilístico
        enable_adaptive_seeding_config_ga=False,  # Seeding Adaptativo
        hc_enable_adaptive=False,  # Hill Climbing adaptativo (legado)
        hc_gmean_threshold=0.90,   # Threshold de G-mean para desabilitar HC (legado)
        hc_hierarchical_enabled=True,  # Hill Climbing Hierárquico v2.0
        stagnation_threshold=15,   # Threshold de estagnação para ativar Hill Climbing
        use_balanced_crossover=False,  # Crossover Balanceado Inteligente
        drift_severity='NONE'  # Severidade do drift detectado (SEVERE, MODERATE, MILD, STABLE, NONE)
       ):
    """
    Runs the genetic algorithm for a single data chunk.
    """
    #logging.info(f"Starting GA run: Pop={population_size}, MaxGen={max_generations}, Elit={elitism_rate:.2f}, Parallel={parallel_enabled}")
    logging.info(f"Starting GA run: Pop={population_size}, MaxGen={max_generations}, Elit={elitism_rate:.2f}, Parallel={parallel_enabled}, InitStrategy={initialization_strategy}") # [ga.py]
    if use_balanced_crossover:
        logging.info(f"  Crossover Balanceado Inteligente ATIVADO (70% qualidade + 30% diversidade)")
    if mutation_override_config_ga and mutation_override_config_ga.get("override_rate") is not None: # [ga.py]
        logging.info(f"  Mutation override: Rate={mutation_override_config_ga['override_rate']} for {mutation_override_config_ga.get('override_generations', 0) or 'all'} generations.") # [ga.py]

    reference_features = fitness_module.compute_features_reference(best_ever_memory);
    beta = fitness_module.get_adaptive_beta(performance_label)
    STAGNATION_THRESHOLD = stagnation_threshold  # Agora configurável via parâmetro

    # --- Chamada Corrigida para initialize_population ---
    population = initialize_population( # [ga.py]
        population_size=population_size, max_rules_per_class=max_rules_per_class, max_depth=max_depth, # max_depth e max_rules_per_class já são os adaptados por main.py [ga.py]
        attributes=attributes, value_ranges=value_ranges, category_values=category_values, categorical_features=categorical_features, # [ga.py]
        classes=classes, train_data=train_data, train_target=train_target, # [ga.py]
        regularization_coefficient=regularization_coefficient, # [ga.py]
        feature_penalty_coefficient=feature_penalty_coefficient, # [ga.py]
        operator_penalty_coefficient=operator_penalty_coefficient, # [ga.py]
        threshold_penalty_coefficient=threshold_penalty_coefficient,
        intelligent_mutation_rate=intelligent_mutation_rate, # [ga.py]
        previous_rules_pop=previous_rules_pop, best_ever_memory=best_ever_memory, # [ga.py]
        best_individual_from_previous_chunk=best_individual_from_previous_chunk,
        performance_label=performance_label,
        prev_best_mutant_ratio=prev_best_mutant_ratio,# [ga.py]
        reference_features=reference_features,
        beta=beta,
        initialization_strategy=initialization_strategy,
        enable_dt_seeding_on_init_config=enable_dt_seeding_on_init_config_ga, # [ga.py]
        dt_seeding_ratio_on_init_config=dt_seeding_ratio_on_init_config_ga, # [ga.py]
        dt_seeding_sample_size_on_init_config=dt_seeding_sample_size_on_init_config_ga, # [ga.py]
        dt_seeding_rules_to_replace_config=dt_seeding_rules_to_replace_config_ga, # [ga.py]
        recovery_aggressive_mutant_ratio_config=recovery_aggressive_mutant_ratio_config_ga,
        dt_seeding_depths_on_init_config=dt_seeding_depths_on_init_config_ga,
        historical_reference_dataset=historical_reference_dataset,          # [ga.py]
        dt_rule_injection_ratio_config=dt_rule_injection_ratio_config_ga,   # Seeding Probabilístico
        enable_adaptive_seeding_config=enable_adaptive_seeding_config_ga,   # Seeding Adaptativo
        drift_severity=drift_severity  # Severidade do drift para ajustar recipes
    )
    # ----------------------------------------------------

    if not population: # [ga.py]
        logging.error("Initialization failed. Aborting GA run.") # [ga.py]
        return None, [], {'best_fitness': [], 'avg_fitness': [], 'std_fitness': [], 'best_gmean': [], 'avg_gmean': [], 'std_gmean': []} # [ga.py]

    # --- Determina Workers ---
    workers_to_use = 0
    if parallel_enabled:
        try:
            available_cores = os.cpu_count()
            if num_workers is None or num_workers <= 0: workers_to_use = available_cores; logging.info(f"Parallel enabled. Using cores: {workers_to_use}")
            else: workers_to_use = min(num_workers, available_cores); logging.info(f"Parallel enabled. Using workers: {workers_to_use} (Available: {available_cores})") # type: ignore
            if workers_to_use <= 1: logging.info("Workers <= 1, defaulting to serial."); parallel_enabled = False; workers_to_use = 0 # type: ignore
        except NotImplementedError: logging.warning("os.cpu_count() not implemented. Disabling parallelism."); parallel_enabled = False; workers_to_use = 0
        except Exception as e: logging.error(f"Error determining worker count: {e}. Disabling parallelism.", exc_info=True); parallel_enabled = False; workers_to_use = 0

    # --- Inicializa estado do loop ---
    num_elite = int(elitism_rate * population_size); best_individual_overall = None; best_fitness_so_far = -float('inf'); no_improvement_count = 0
    history = {'best_fitness': [], 'avg_fitness': [], 'std_fitness': [], 'best_gmean': [], 'avg_gmean': [], 'std_gmean': [], 'diversity': [], 'avg_rule_activation': []}
    generation = 0

    # --- Loop de Gerações ---
    for generation in range(max_generations):
        gen_start_time = time.time()
        fitness_values = []; gmean_values = []
        activation_rates = []
        # Empacota args para worker (já estava correto)
        constant_args_for_worker = {
            'train_data': train_data, 
            'train_target': train_target,
            'classes': classes,
            'class_coverage_coefficient': class_coverage_coefficient_ga,
            'gmean_bonus_coefficient': gmean_bonus_coefficient_ga,
            'regularization_coefficient': regularization_coefficient, 
            'feature_penalty_coefficient': feature_penalty_coefficient, 
            'reference_features': reference_features, 
            'beta': beta, 
            'previous_used_features': previous_used_features, 
            'gamma': gamma, 
            'operator_penalty_coefficient': operator_penalty_coefficient, 
            'threshold_penalty_coefficient': threshold_penalty_coefficient, 
            'previous_operator_info': previous_operator_info, 
            'operator_change_coefficient': operator_change_coefficient, 
            'attributes': attributes, 
            'categorical_features': categorical_features,
            'class_weights': class_weights,
            'reduce_change_penalties': reduce_change_penalties_flag,
            'gmean_bonus_coefficient':gmean_bonus_coefficient_ga
        }

        # --- Avaliação (Paralela ou Serial) ---
        if parallel_enabled and workers_to_use > 1 and len(population) > 1: # type: ignore
            # ... (lógica do pool.map como antes) ...
            logging.debug(f"Gen {generation+1}: Starting parallel fitness evaluation with {workers_to_use} workers.")
            fitness_score, gmean, activation_rate = 0.0, 0.0, 0.0
            map_iterable = [(ind, constant_args_for_worker) for ind in population]
            try:
                with multiprocessing.Pool(processes=workers_to_use) as pool: results = pool.map(evaluate_individual_fitness_parallel, map_iterable)
                valid_results = 0
                for i, individual in enumerate(population):
                    fitness_score, gmean, activation_rate = results[i]
                    individual.fitness = fitness_score
                    fitness_values.append(fitness_score)
                    individual.gmean = gmean
                    gmean_values.append(gmean)
                    activation_rates.append(activation_rate)
                    if fitness_score > -float('inf'): valid_results += 1
                if valid_results < len(population): logging.warning(f"Gen {generation+1}: Only {valid_results}/{len(population)} individuals evaluated successfully in parallel.")
                if valid_results == 0: logging.error(f"Gen {generation+1}: Parallel eval failed for all. Stopping."); break
            except Exception as pool_e: logging.error(f"Error during parallel fitness eval: {pool_e}", exc_info=True); logging.error("Stopping GA run."); break
        else: # Execução Serial
             # ... (lógica serial como antes) ...
            if generation == 0: logging.info(f"Gen {generation+1}: Running fitness evaluation serially.")
            valid_individuals_count = 0
            for individual in population:
                worker_args = (individual, constant_args_for_worker)
                fitness_score, gmean, activation_rate = evaluate_individual_fitness_parallel(worker_args)
                individual.fitness = fitness_score
                individual.gmean = gmean
                fitness_values.append(fitness_score)
                gmean_values.append(gmean)
                activation_rates.append(activation_rate)
                if fitness_score > -float('inf'): valid_individuals_count += 1
            if valid_individuals_count == 0: logging.error(f"Gen {generation+1}: Serial evaluation failed for all. Stopping."); break

        # --- Métricas, Melhor Global, Parada Antecipada, Adaptação ---
        # ... (código como antes) ...
        avg_fitness = np.nanmean(fitness_values); std_fitness = np.nanstd(fitness_values); best_fitness_gen = np.nanmax(fitness_values); worst_fitness_gen = np.nanmin(fitness_values)
        avg_gmean = np.mean(gmean_values); std_gmean = np.std(gmean_values); best_gmean_gen = np.max(gmean_values)
        avg_rule_activation = np.mean(activation_rates) * 100 if activation_rates else 0.0
        diversity_score = calculate_population_diversity(population)
        history['diversity'].append(diversity_score)
        history['best_fitness'].append(best_fitness_gen); 
        history['avg_fitness'].append(avg_fitness); 
        history['std_fitness'].append(std_fitness); 
        history['best_gmean'].append(best_gmean_gen); 
        history['avg_gmean'].append(avg_gmean); 
        history['std_gmean'].append(std_gmean)
        history['avg_rule_activation'].append(avg_rule_activation)

        population.sort(key=lambda ind: ind.fitness, reverse=True); current_best_individual_gen = population[0] if population else None

        rule_activation_rate = 0.0
        if current_best_individual_gen and hasattr(current_best_individual_gen, 'total_predictions_made') and current_best_individual_gen.total_predictions_made > 0:
            total_preds = current_best_individual_gen.total_predictions_made
            defaults_used = current_best_individual_gen.default_class_triggered
            rule_activations = total_preds - defaults_used
            rule_activation_rate = (rule_activations / total_preds) * 100
            # Reseta os contadores para a próxima geração
            current_best_individual_gen.total_predictions_made = 0
            current_best_individual_gen.default_class_triggered = 0

        # if current_best_individual_gen:
        #      if best_individual_overall is None or current_best_individual_gen.fitness > best_individual_overall.fitness: best_individual_overall = copy.deepcopy(current_best_individual_gen); best_fitness_so_far = best_individual_overall.fitness; no_improvement_count = 0
        #      if best_fitness_so_far > -float('inf'): logging.debug(f"Gen {generation+1}: New best fitness: {best_fitness_so_far:.4f}")
        #      else: no_improvement_count += 1 # Incrementa só se não houve melhora
        # if early_stopping_patience is not None and no_improvement_count >= early_stopping_patience: logging.info(f"\nEarly stopping at gen {generation+1}"); break
        if current_best_individual_gen:
            # Verifica se o melhor indivíduo desta geração supera o melhor de todos os tempos.
            if best_individual_overall is None or current_best_individual_gen.fitness > best_individual_overall.fitness:
                # Se sim, atualiza o campeão e zera o contador de estagnação.
                best_individual_overall = copy.deepcopy(current_best_individual_gen)
                best_fitness_so_far = best_individual_overall.fitness
                no_improvement_count = 0
                logging.debug(f"Gen {generation+1}: Novo melhor fitness geral encontrado: {best_fitness_so_far:.4f}")
            else:
                # Se não, incrementa o contador de estagnação.
                no_improvement_count += 1

        # --- EARLY STOPPING ADAPTATIVO EM 3 CAMADAS ---
        # Camada 1: Para rápido se elite é satisfatório (≥88% G-mean) + estagnado (≥15 gens)
        SATISFACTORY_GMEAN = 0.88
        EARLY_PATIENCE_LAYER1 = 15

        current_gmean = best_individual_overall.gmean if best_individual_overall else 0.0

        if current_gmean >= SATISFACTORY_GMEAN and no_improvement_count >= EARLY_PATIENCE_LAYER1:
            logging.info(f"")
            logging.info(f"╔══════════════════════════════════════════════════════════════╗")
            logging.info(f"║ EARLY STOPPING LAYER 1: Elite Satisfatório + Estagnado     ║")
            logging.info(f"╠══════════════════════════════════════════════════════════════╣")
            logging.info(f"║ Elite G-mean:     {current_gmean:.1%} (≥ {SATISFACTORY_GMEAN:.0%})          ║")
            logging.info(f"║ Estagnação:       {no_improvement_count} gerações (≥ {EARLY_PATIENCE_LAYER1})                ║")
            logging.info(f"║ Geração atual:    {generation + 1}                                     ║")
            logging.info(f"║ Decisão:          PARAR (performance satisfatória)         ║")
            logging.info(f"╚══════════════════════════════════════════════════════════════╝")
            logging.info(f"")
            break

        # Camada 2: Para se últimas 30 gerações tiveram melhoria < 0.5%
        MIN_IMPROVEMENT_WINDOW = 30
        MIN_IMPROVEMENT_THRESHOLD = 0.005  # 0.5% G-mean

        if len(history['best_gmean']) >= MIN_IMPROVEMENT_WINDOW and generation + 1 >= MIN_IMPROVEMENT_WINDOW:
            fitness_current = history['best_gmean'][-1]
            fitness_30_gens_ago = history['best_gmean'][-MIN_IMPROVEMENT_WINDOW]
            improvement = fitness_current - fitness_30_gens_ago

            if improvement < MIN_IMPROVEMENT_THRESHOLD:
                logging.info(f"")
                logging.info(f"╔══════════════════════════════════════════════════════════════╗")
                logging.info(f"║ EARLY STOPPING LAYER 2: Melhoria Marginal                   ║")
                logging.info(f"╠══════════════════════════════════════════════════════════════╣")
                logging.info(f"║ G-mean 30 gens atrás: {fitness_30_gens_ago:.4f}                          ║")
                logging.info(f"║ G-mean atual:         {fitness_current:.4f}                          ║")
                logging.info(f"║ Melhoria:              {improvement:.4f} (< {MIN_IMPROVEMENT_THRESHOLD:.4f})          ║")
                logging.info(f"║ Geração atual:         {generation + 1}                              ║")
                logging.info(f"║ Decisão:               PARAR (retorno decrescente severo)  ║")
                logging.info(f"╚══════════════════════════════════════════════════════════════╝")
                logging.info(f"")
                break

        # Camada 3: Stop tradicional (fallback se camadas 1-2 não ativaram)
        if early_stopping_patience is not None and no_improvement_count >= early_stopping_patience:
            logging.info(f"")
            logging.info(f"╔══════════════════════════════════════════════════════════════╗")
            logging.info(f"║ EARLY STOPPING LAYER 3: Estagnação Longa                    ║")
            logging.info(f"╠══════════════════════════════════════════════════════════════╣")
            logging.info(f"║ Estagnação:    {no_improvement_count} gerações (≥ {early_stopping_patience})                    ║")
            logging.info(f"║ Geração atual: {generation + 1}                                     ║")
            logging.info(f"║ Decisão:       PARAR (estagnação prolongada)                ║")
            logging.info(f"╚══════════════════════════════════════════════════════════════╝")
            logging.info(f"")
            break

        valid_fitness = [f for f in fitness_values if np.isfinite(f)]
        if len(valid_fitness) > 1: 
            bfg, afg, wfg = np.max(valid_fitness), np.mean(valid_fitness), np.min(valid_fitness); 
            mutation_rate = get_dynamic_mutation_rate(bfg, afg, wfg); 
        # <<< APPLY MUTATION OVERRIDE IF ACTIVE >>>
            if mutation_override_config_ga and mutation_override_config_ga.get("override_rate") is not None:
                override_gen_limit = mutation_override_config_ga.get("override_generations", 0)
                if override_gen_limit == 0 or generation < override_gen_limit: # 0 significa para todas as gerações do chunk
                    mutation_rate = mutation_override_config_ga["override_rate"]
                    if generation == 0 : logging.debug(f"  Gen {generation+1}: Applying mutation override rate: {mutation_rate}") # [ga.py]
                elif generation == override_gen_limit: # Loga quando o override termina
                    logging.debug(f"  Gen {generation+1}: Mutation override period ended. Reverting to dynamic rate.") # [ga.py]
        # <<< END APPLY MUTATION OVERRIDE >>>            
            max_ts = min(final_tournament_size, population_size); 
            min_ts = min(initial_tournament_size, max_ts); 
            tournament_size = get_dynamic_tournament_size(bfg, afg, wfg, min_ts, max_ts)
        else: mutation_rate, tournament_size = 0.1, initial_tournament_size
        #crossover_type = get_dynamic_crossover_type(generation, max_generations)
        gen_end_time = time.time(); 
        log_message = ( f"\rGen {generation + 1}/{max_generations} - BestFit: {best_fitness_gen:.4f} (G-mean: {best_gmean_gen:.3f}) | AvgFit: {avg_fitness:.4f} (G-mean: {avg_gmean:.3f}) | (Diversity: {diversity_score:.3f}) | (RuleAct: {avg_rule_activation:.1f}) | Mut: {mutation_rate:.3f} | Tourn: {tournament_size} | Stagn: {no_improvement_count} | Time: {gen_end_time - gen_start_time:.2f}s" ); print(log_message, end="")

        # --- Elitismo Híbrido Unificado: Prioriza G-mean com um nicho para o campeão de Fitness ---
        num_elite_slots = int(elitism_rate * population_size)
        new_population = []

        if num_elite_slots > 0 and population:
            # <<< PASSO 1: Identificar os candidatos de elite de ambos os nichos >>>
            
            # Candidato 1: O campeão de Fitness (a população já vem ordenada por fitness da etapa anterior)
            fitness_champion = population[0]
            
            # Candidatos 2: O Top N por G-mean
            # Reclassificamos uma cópia da população por G-mean para encontrar os melhores
            population_sorted_by_gmean = sorted(population, key=lambda ind: ind.gmean, reverse=True)
            top_gmean_individuals = population_sorted_by_gmean[:num_elite_slots]

            # <<< PASSO 2: Unir os candidatos em um pool único, removendo duplicatas >>>
            # Usar um set de IDs ou representações de string para garantir a unicidade dos indivíduos
            candidate_elites = {} # Usamos um dicionário para evitar duplicatas
            
            # Adiciona o campeão de fitness
            candidate_elites[fitness_champion.get_rules_as_string()] = fitness_champion
            
            # Adiciona os campeões de G-mean (duplicatas serão sobrescritas/ignoradas)
            for ind in top_gmean_individuals:
                candidate_elites[ind.get_rules_as_string()] = ind

            # Converte o pool de candidatos únicos de volta para uma lista
            final_candidate_list = list(candidate_elites.values())
            
            # <<< PASSO 3: Classificar o pool de candidatos final pelo G-mean >>>
            final_candidate_list.sort(key=lambda ind: ind.gmean, reverse=True)
            
            # <<< PASSO 4: Selecionar os N melhores do pool final para sobreviver >>>
            final_elites = final_candidate_list[:num_elite_slots]
            
            # Adiciona cópias dos elites à nova população
            new_population.extend(copy.deepcopy(ind) for ind in final_elites)
            
            # <<< PASSO 5: Logging aprimorado para a nova estratégia >>>
            if final_elites:
                logging.info(f"Gen {generation + 1}: Preservando {len(final_elites)} elites do pool híbrido (ordenados por G-mean).")
                for i, elite in enumerate(final_elites):
                    logging.debug(f"  -> Elite #{i+1}: G-mean={elite.gmean:.4f}, Fitness={elite.fitness:.4f}")



        # --- Cria Nova Geração ---
        # ... (código como antes, passando info categórica para crossover e mutate) ...
       
        # # --- Cria Nova Geração ---
        # new_population = []
        
        # # <<< SUBSTITUA SEU BLOCO DE ELITISMO ATUAL POR ESTE >>>

        # # --- Elitismo de Nicho: Preserva o melhor por Fitness E o melhor por G-Mean ---
        # num_elite_slots = int(elitism_rate * population_size)

        # if num_elite_slots > 0 and population:
        #     elites_to_add = []

        #     # 1. Adiciona o campeão de Fitness (a população já está ordenada por fitness)
        #     fitness_champion = population[0]
        #     elites_to_add.append(copy.deepcopy(fitness_champion))
        #     # Adiciona o gmean ao indivíduo para que o log seja completo
        #     fitness_champion.gmean = calculate_gmean_contextual(train_target, [fitness_champion._predict(inst) for inst in train_data], classes)
        #     logging.debug(f"\nGen {generation + 1}: Preservando Campeão de Fitness (Fit: {fitness_champion.fitness:.4f}, G-mean: {fitness_champion.gmean:.4f})")

        #     # 2. Adiciona o campeão de G-Mean, se for diferente e houver espaço
        #     if len(population) > 1 and len(elites_to_add) < num_elite_slots:
                
        #         gmean_champion = max(population, key=lambda ind: ind.gmean)

        #         # Garante que não estamos adicionando o mesmo indivíduo duas vezes
        #         if gmean_champion is not fitness_champion:
        #             elites_to_add.append(copy.deepcopy(gmean_champion))
        #             logging.info(f"Gen {generation + 1}: Preservando Campeão de G-Mean (G-mean: {gmean_champion.gmean:.4f}, Fit: {gmean_champion.fitness:.4f})")
        #         else:
        #             logging.info(f"Gen {generation + 1}: Campeão de Fitness e de G-Mean são o mesmo indivíduo.")
            
        #     new_population.extend(elites_to_add)

        # <<< FIM DA LÓGICA DE ELITISMO DE NICHO >>>
        #         
        # while len(new_population) < population_size:
        #     parent1 = ga_operators.tournament_selection(population, tournament_size)
        #     parent2 = ga_operators.tournament_selection(population, tournament_size)
        #     if parent1 is None or parent2 is None: logging.warning("Selection failed."); break
        #     try: child = ga_operators.crossover( parent1, parent2, max_depth, attributes, value_ranges, category_values, categorical_features, classes, max_rules_per_class)
        #     except Exception as cross_e: logging.error(f"Crossover error: {cross_e}. Skipping.", exc_info=True); continue
        #     try: ga_operators.mutate_individual( child, mutation_rate, max_depth, intelligent_mutation_rate, attributes, value_ranges, category_values, categorical_features, classes, max_rules_per_class, train_data, train_target )
        #     except Exception as mut_e: logging.error(f"Mutation error: {mut_e}. Skipping.", exc_info=True); continue
        #     new_population.append(child)
        # # --- Lógica do "Botão Turbo" (Hill Climbing) ---
        # if no_improvement_count >= STAGNATION_THRESHOLD and population:
        #     logging.info(f"\nEstagnação detectada ({no_improvement_count} gerações)! Ativando Hill Climbing...")
            
        #     # Pega o melhor indivíduo atual para refiná-lo
        #     champion = population[0]
            
        #     # Gera alguns mutantes de refinamento
        #     num_hc_mutants = max(1, int(population_size * 0.1)) # Refina 10% da população
        #     for _ in range(num_hc_mutants):
        #         # Cria uma cópia mutada via Hill Climbing
        #         hc_mutant = ga_operators.mutate_hill_climbing(champion, value_ranges)
                
        #         # Substitui um indivíduo aleatório (não-elite) da nova população
        #         if len(new_population) > num_elite_slots:
        #             idx_to_replace = random.randint(num_elite_slots, len(new_population) - 1)
        #             new_population[idx_to_replace] = hc_mutant

        #     no_improvement_count = 0 # Reseta o contador para dar tempo aos mutantes de agirem
        # --- Lógica de Resgate de Estagnação ---
        force_gene_therapy_flag = False # <<< NOVO: Flag para a terapia gênica
        effective_mutation_rate = mutation_rate
        if no_improvement_count >= STAGNATION_THRESHOLD and population:
            logging.info(f"\nEstagnação detectada ({no_improvement_count} gerações)! Ativando mecanismos de resgate...")
            force_gene_therapy_flag = True # Ativa a terapia para a próxima geração
            effective_mutation_rate = mutation_rate * 2.0

            # --- Hill Climbing Hierárquico v2.0 ---
            elite_gmean = best_individual_overall.gmean if best_individual_overall else 0.0

            # Prepara kwargs para HC hierárquico
            hc_kwargs = {
                'value_ranges': value_ranges,
                'max_depth': max_depth,
                'attributes': attributes,
                'category_values': category_values,
                'categorical_features': categorical_features,
                'classes': classes,
                'max_rules_per_class': max_rules_per_class,
                'train_data': train_data,
                'train_target': train_target
            }

            # Chama HC hierárquico
            hc_variants = hill_climbing_v2.hierarchical_hill_climbing(
                elite=population[0],
                population=population,
                best_ever_memory=best_ever_memory,
                gmean=elite_gmean,
                no_improvement_count=no_improvement_count,
                hc_enable_adaptive=hc_enable_adaptive,
                hc_gmean_threshold=hc_gmean_threshold,
                hc_hierarchical_enabled=hc_hierarchical_enabled,
                **hc_kwargs
            )

            # CORREÇÃO 1: Avalia variantes HC ANTES de injetar
            if hc_variants:
                logging.info(f"     Avaliando {len(hc_variants)} variantes HC geradas...")

                elite_fitness = population[0].fitness
                elite_gmean_val = population[0].gmean

                evaluated_variants = []
                for i, hc_variant in enumerate(hc_variants):
                    # Avalia fitness do variant usando calculate_fitness
                    metrics = fitness_module.calculate_fitness(
                        hc_variant, train_data, train_target,
                        class_weights,
                        regularization_coefficient, feature_penalty_coefficient,
                        class_coverage_coefficient_ga, gmean_bonus_coefficient_ga,
                        operator_change_coefficient,
                        gamma,
                        beta,
                        reference_features,
                        previous_used_features,
                        previous_operator_info,
                        reduce_change_penalties_flag
                    )

                    hc_variant.fitness = metrics['fitness']
                    hc_variant.gmean = metrics['g_mean']  # Chave correta: 'g_mean' com underscore

                    # FASE 2: Tolerância aumentada de 0.5% para 1.5% para melhorar taxa de aprovação HC
                    # Permite exploração de variantes "quase tão boas" com mais diversidade
                    tolerance = 0.015  # 1.5% em G-mean (era 0.5%)
                    fitness_tolerance = tolerance * 2  # Aproximadamente 0.03 em fitness

                    gmean_acceptable = hc_variant.gmean >= (elite_gmean_val - tolerance)
                    fitness_acceptable = hc_variant.fitness >= (elite_fitness - fitness_tolerance)

                    if gmean_acceptable or fitness_acceptable:
                        evaluated_variants.append(hc_variant)

                        # Indica se foi aprovado por ser melhor ou por tolerância
                        if hc_variant.gmean > elite_gmean_val or hc_variant.fitness > elite_fitness:
                            approval_reason = "MELHOR"
                        else:
                            approval_reason = "TOLERÂNCIA"

                        logging.info(
                            f"       ✓ HC variant #{i+1} APROVADO ({approval_reason}): "
                            f"fitness={hc_variant.fitness:.4f} (elite={elite_fitness:.4f}, Δ={hc_variant.fitness-elite_fitness:+.4f}), "
                            f"gmean={hc_variant.gmean:.3f} (elite={elite_gmean_val:.3f}, Δ={hc_variant.gmean-elite_gmean_val:+.3f})"
                        )
                    else:
                        logging.debug(
                            f"       ✗ HC variant #{i+1} REJEITADO: "
                            f"fitness={hc_variant.fitness:.4f} < {elite_fitness-fitness_tolerance:.4f} (tolerância), "
                            f"gmean={hc_variant.gmean:.3f} < {elite_gmean_val-tolerance:.3f} (tolerância)"
                        )

                logging.info(
                    f"     Resultado: {len(evaluated_variants)}/{len(hc_variants)} variantes aprovadas "
                    f"({100*len(evaluated_variants)/len(hc_variants):.1f}% taxa de aprovação)"
                )

                # Insere APENAS variantes aprovadas na população
                # CORREÇÃO: Insere nas primeiras posições após elite para maximizar chance de se tornarem elite
                for i, hc_variant in enumerate(evaluated_variants):
                    # Insere logo após os slots de elite para que compitam diretamente
                    insert_position = num_elite_slots + i
                    if insert_position < len(new_population):
                        new_population[insert_position] = hc_variant
                        logging.debug(f"       HC variant #{i+1} inserido na posição {insert_position}")
                    else:
                        # Se não há espaço, adiciona ao final
                        new_population.append(hc_variant)
                        logging.debug(f"       HC variant #{i+1} adicionado ao final da população")
            else:
                logging.info("     Nenhuma variante HC gerada.")

            # REMOVED: no_improvement_count = 0 # NÃO reseta o contador (permite early stopping funcionar)

        # Preenche o restante da população com filhos (agora cientes da terapia gênica)
        while len(new_population) < population_size:
            parent1 = ga_operators.tournament_selection(population, tournament_size)
            parent2 = ga_operators.tournament_selection(population, tournament_size)
            if parent1 is None or parent2 is None: logging.warning("Selection failed."); break
            try:
                if use_balanced_crossover:
                    # Crossover Balanceado Inteligente: 70% qualidade + 30% diversidade
                    child = ga_operators.balanced_crossover(
                        parent1, parent2,
                        data=train_data, target=train_target,
                        max_rules_per_class=max_rules_per_class,
                        current_generation=generation + 1,
                        max_generations=max_generations,
                        classes=classes,
                        max_depth=max_depth, attributes=attributes,
                        value_ranges=value_ranges, category_values=category_values,
                        categorical_features=categorical_features
                    )
                else:
                    # Crossover tradicional
                    child = ga_operators.crossover( parent1, parent2, max_depth, attributes, value_ranges, category_values, categorical_features, classes, max_rules_per_class)
            except Exception as cross_e:
                logging.error(f"Crossover error: {cross_e}. Skipping.", exc_info=True); continue
            try: 
                # <<< MUDANÇA IMPORTANTE: Passa o flag para a função de mutação >>>
                ga_operators.mutate_individual( 
                    child, effective_mutation_rate, max_depth, intelligent_mutation_rate, attributes, value_ranges, 
                    category_values, categorical_features, classes, max_rules_per_class, train_data, train_target,
                    force_gene_therapy=force_gene_therapy_flag
                )
            except Exception as mut_e: 
                logging.error(f"Mutation error: {mut_e}. Skipping.", exc_info=True); continue
            new_population.append(child)
        population = new_population

        if not population: logging.error(f"Population empty after gen {generation+1}. Stopping."); break

    # --- Fim do Loop ---
    # ... (código como antes) ...
    print(); logging.info(f"GA run finished after {generation + 1} generations.")
    if best_individual_overall is None and population: logging.warning("Best overall not updated, returning best from final pop."); population.sort(key=lambda ind: ind.fitness, reverse=True); best_individual_overall = copy.deepcopy(population[0])
    elif best_individual_overall is None: logging.error("GA run failed to produce any valid best individual.")
    return best_individual_overall, population, history