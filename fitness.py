# fitness.py (Corrigido - UnboundLocalError delta_features)

import numpy as np
import logging
from sklearn.metrics import accuracy_score, f1_score, recall_score
from metrics import calculate_gmean_contextual
from typing import Dict

# DEBUG LAYER 1 - Diagnóstico de Early Stop
DEBUG_LAYER1_FITNESS = True

def debug_print_fitness(msg):
    """Helper para debug logging condicional em fitness"""
    if DEBUG_LAYER1_FITNESS:
        logging.warning(f"[DEBUG L1 FITNESS] {msg}")

# --- Funções Auxiliares (Feature Stability/Memory) ---
def get_all_features_from_memory(best_ever_memory):
    if not best_ever_memory: return []; features_list = [ind.get_used_attributes() for ind in best_ever_memory]; return features_list

def compute_features_reference(best_ever_memory):
    if not best_ever_memory: return set()
    all_feature_sets = get_all_features_from_memory(best_ever_memory) # Atribuição ANTES do uso
    if not all_feature_sets: return set()
    total_inds_in_memory = len(all_feature_sets); feature_counts = {}
    for fset in all_feature_sets:
        for feature in fset: feature_counts[feature] = feature_counts.get(feature, 0) + 1
    usage_threshold = total_inds_in_memory / 2.0; reference_features = set()
    for feature, count in feature_counts.items():
        if count >= usage_threshold: reference_features.add(feature)
    logging.debug(f"Computed reference features from memory: {reference_features}"); return reference_features

def compute_features_distance(used_features, reference_features):
    # ... (código como antes) ...
    if not reference_features: return 0.0; used_features = set(used_features); reference_features = set(reference_features);
    if not used_features and not reference_features: return 0.0; union_set = used_features.union(reference_features); union_size = len(union_set)
    if union_size == 0: return 0.0; symmetric_diff_size = len(used_features.symmetric_difference(reference_features)) # type: ignore
    return symmetric_diff_size / (union_size + 1e-9) # type: ignore

def get_adaptive_beta(performance_label):
    # ... (código como antes) ...
    if performance_label == 'good': return 0.05
    elif performance_label == 'bad': return 0.01
    else: return 0.03 # 'medium'


# --- Função Principal de Cálculo de Fitness (Corrigida Lógica de change_penalty) ---
# def calculate_fitness(
#     individual, data, target,
#     regularization_coefficient, feature_penalty_coefficient, gmean_bonus_coefficient,
#     # Args com Default
#     operator_penalty_coefficient=0.0, threshold_penalty_coefficient=0.0,
#     operator_change_coefficient=0.0, gamma=0.0,
#     reference_features=None, beta=0.0,
#     previous_used_features=None, previous_operator_info=None,
#     attributes=None, categorical_features=None,
#     reduce_change_penalties_active=False,
#     ):
#     """Calculates the fitness score for a given individual."""
#     if not data or not target or len(data) != len(target): logging.warning("Fitness calculation skipped: Invalid data/target."); return -float('inf')
#     if categorical_features is None: categorical_features = set()

#     # 1. Performance
#     accuracy = 0.0
#     accuracy_weight = 0.97
#     g_mean = 0
#     try:
#         predictions = [individual._predict(inst) for inst in data]
#         #Linha accuracy
#         #accuracy = accuracy_score(target, predictions)
#         weighted_f1 = f1_score(target, predictions, labels=individual.classes, average='weighted', zero_division=0)

#         # Linha g-mean
#         present_classes = np.unique(target)
#         #g_mean = geometric_mean_score(target, predictions, average='multiclass')
#         if len(present_classes) < 2:
#             g_mean = accuracy_score(target, predictions)
#             #accuracy = g_mean # São a mesma coisa neste caso
#         else:
#             # 3. Calcular o G-mean apenas sobre as classes presentes.
#             #    Isso evita o UndefinedMetricWarning e calcula um fitness mais justo.
#             g_mean = calculate_gmean_contextual(target, predictions, individual.classes)
#             #g_mean = geometric_mean_score(target, predictions, labels=present_classes, average='multiclass')

#             #accuracy = accuracy_score(target, predictions) # Mantém o cálculo da acurácia para logging.

#     except Exception as e:
#         logging.error(f"Error calculating accuracy during fitness: {e}", exc_info=True)
#         return -float('inf')
    
#     # 2. Complexidade
#     complexity_penalty = 0.0
#     if regularization_coefficient > 0: total_nodes = individual.count_total_nodes(); total_depth_sum = sum(rt.get_depth() for rlist in individual.rules.values() for rt in rlist); complexity_penalty = regularization_coefficient * (total_nodes + total_depth_sum)
#     feature_penalty = 0.0; used_features = individual.get_used_attributes()
#     if feature_penalty_coefficient > 0: feature_penalty = feature_penalty_coefficient * len(used_features)

#     # 3. Estabilidade vs Memória
#     distance_penalty = 0.0
#     if beta > 0.0 and reference_features is not None: distance = compute_features_distance(used_features, reference_features); distance_penalty = beta * distance

#     # --- Determine effective change penalty coefficients ---
#     current_gamma = gamma
#     current_operator_change_coefficient = operator_change_coefficient

#     if reduce_change_penalties_active:
#         logging.debug("Adaptive change penalties: Reducing gamma and operator_change_coefficient.")
#         current_gamma = 0.0  # Or a very small fraction, e.g., gamma * 0.1
#         current_operator_change_coefficient = 0.0 # Or a very small fraction
#         distance_penalty = 0.0
#         feature_penalty = 0.0
#         complexity_penalty = 0.0
#         accuracy_weight = 1.0

#     # --- 4. Mudança vs Chunk Anterior (Contagem Features) - CORRIGIDO ---
#     change_penalty = 0.0
#     if current_gamma > 0.0 and previous_used_features is not None: # Use current_gamma
#         used_features = individual.get_used_attributes() # get_used_attributes should be efficient
#         delta_features = len(used_features) - len(set(previous_used_features))
#         if delta_features > 0:
#             change_penalty = current_gamma * delta_features # Use current_gamma

#     # 5. & 6. Penalidades de Operador/Threshold
#     operator_complexity_penalty = 0.0
#     operator_change_penalty_val = 0.0 
#     calculate_ops_thresholds = (operator_penalty_coefficient > 0 or threshold_penalty_coefficient > 0 or (operator_change_coefficient > 0 and previous_operator_info is not None))
#     if calculate_ops_thresholds:
#         try: logical_ops, comparison_ops, numeric_thresholds, categorical_values_used = individual._collect_ops_and_thresholds()
#         except Exception as collect_e: logging.error(f"Error collecting ops/thresholds: {collect_e}", exc_info=True); logical_ops, comparison_ops, numeric_thresholds, categorical_values_used = [], [], [], []
#         # Penalidade Interna
#         if operator_penalty_coefficient > 0: n_logical_distinct = len(set(logical_ops)); n_comparison_distinct = len(set(comparison_ops)); operator_complexity_penalty += operator_penalty_coefficient * (n_logical_distinct + n_comparison_distinct)
#         if threshold_penalty_coefficient > 0 and len(numeric_thresholds) > 1:
#             try: threshold_range = np.ptp(numeric_thresholds); operator_complexity_penalty += threshold_penalty_coefficient * threshold_range
#             except Exception as ptp_e: logging.warning(f"Could not calculate ptp: {ptp_e}")
#         # Penalidade de Mudança de Operadores
#         if current_operator_change_coefficient > 0.0 and previous_operator_info is not None: # Use current_operator_change_coefficient
#             # ... (existing logic for calculating operator differences) ...
#             # Ensure this part uses current_operator_change_coefficient
#             # Example:
#             # diff_logical_count = len(set(logical_ops).symmetric_difference(old_logical_ops))
#             # diff_comparison_count = len(set(comparison_ops).symmetric_difference(old_comparison_ops))
#             # ...
#             # operator_change_penalty_val = current_operator_change_coefficient * ( diff_logical_count + diff_comparison_count + diff_thresh_mean )
            
#             # Re-implementing this part for clarity using the current_operator_change_coefficient
#             try:
#                 logical_ops, comparison_ops, numeric_thresholds, categorical_values_used = individual._collect_ops_and_thresholds()
#                 old_logical_ops = previous_operator_info.get("logical_ops", set())
#                 old_comparison_ops = previous_operator_info.get("comparison_ops", set())
#                 old_numeric_thresholds = previous_operator_info.get("numeric_thresholds", [])

#                 diff_logical_count = len(set(logical_ops).symmetric_difference(old_logical_ops))
#                 diff_comparison_count = len(set(comparison_ops).symmetric_difference(old_comparison_ops))
                
#                 current_mean_th = np.mean(numeric_thresholds) if numeric_thresholds else 0.0
#                 old_mean_th = np.mean(old_numeric_thresholds) if old_numeric_thresholds else 0.0
#                 diff_thresh_mean = abs(current_mean_th - old_mean_th) # Consider if this part is still desired or too complex

#                 operator_change_penalty_val = current_operator_change_coefficient * (diff_logical_count + diff_comparison_count + diff_thresh_mean)
#             except Exception as collect_e:
#                 logging.error(f"Error collecting ops/thresholds for change penalty: {collect_e}", exc_info=True)
#                 # operator_change_penalty_val remains 0.0

#     # 7. Fitness Final
#     # Ensure you use the calculated operator_change_penalty_val
#     fitness_score = (
#         (g_mean * accuracy_weight)
#         - (0.05 * complexity_penalty)
#         - (0.05 * feature_penalty)
#         - (0.00 * distance_penalty)
#         - (0.00 * change_penalty) # Uses current_gamma effectively
#         - (0.00 * operator_complexity_penalty) # Internal diversity
#         - (0.00 * operator_change_penalty_val) # Change vs previous, uses current_operator_change_coefficient
#     )
#     if np.isnan(fitness_score) or np.isinf(fitness_score):
#         logging.error(f"Fitness NaN/Inf. G-mean: {g_mean}, Penalties: {[complexity_penalty, feature_penalty, distance_penalty, change_penalty, operator_complexity_penalty, operator_change_penalty_val]}. Assigning -inf.")
#         return -float('inf')
#     return fitness_score
# Substitua a função calculate_fitness inteira pela versão abaixo

# Em fitness.py - Corrigir função de fitness

def calculate_fitness(
    individual, data, target,
    # --- Coeficientes principais ---
    class_weights,
    regularization_coefficient,
    feature_penalty_coefficient,
    class_coverage_coefficient,
    gmean_bonus_coefficient,
    # --- Coeficientes para estabilidade (usados para adaptação a drift) ---
    operator_change_coefficient=0.0,
    gamma=0.0,
    beta=0.0, # Penalidade por desvio da memória
    # --- Contexto e flags ---
    reference_features=None,
    previous_used_features=None,
    previous_operator_info=None,
    reduce_change_penalties_active=False,
    # OTIMIZAÇÃO FASE 1.2: Early stopping para indivíduos ruins
    early_stop_threshold=None  # G-mean mínimo esperado (pior elite)
    ) -> dict:
    # **kwargs # Captura argumentos extras não utilizados (como operator_penalty_coefficient)
    # ):
    """
    Calcula o fitness com uma estratégia híbrida: F1-Score como base,
    G-Mean como bônus, e penalidades de complexidade e estabilidade.
    """
    if not data or not target or len(data) != len(target):
        logging.warning("Fitness calculation skipped: Invalid data/target.")
        return {'fitness': -float('inf'), 'g_mean': 0.0, 'weighted_f1': 0.0}

    # OTIMIZAÇÃO FASE 1.2 (CORRIGIDA): Early stopping incremental
    # Avalia apenas 20% dos dados, se ruim descarta (threshold 50% da mediana elite)

    # DEBUG: Contadores estáticos para logging controlado
    if not hasattr(calculate_fitness, 'early_stop_checks'):
        calculate_fitness.early_stop_checks = 0
        calculate_fitness.early_stop_discards = 0

    if early_stop_threshold is not None and early_stop_threshold > 0:
        # Calcula quantos dados representam 20%
        partial_size = max(100, int(len(data) * 0.20))  # Mínimo 100 exemplos

        # DEBUG: Log primeiras verificações
        if calculate_fitness.early_stop_checks < 5:
            debug_print_fitness(f"Early stop check #{calculate_fitness.early_stop_checks + 1}: threshold={early_stop_threshold:.3f}, partial_size={partial_size}/{len(data)}")

        # Avalia apenas os primeiros 20% dos dados
        try:
            partial_data = data[:partial_size]
            partial_target = target[:partial_size]
            partial_predictions = [individual._predict(inst) for inst in partial_data]

            # Calcula G-mean parcial
            partial_gmean = calculate_gmean_contextual(
                partial_target,
                partial_predictions,
                individual.classes
            )

            calculate_fitness.early_stop_checks += 1

            # Early stop: Se G-mean parcial < 50% da mediana elite, descartar
            # Threshold 50% é mais agressivo que 75% anterior
            threshold_50pct = early_stop_threshold * 0.50

            # DEBUG: Log comparação
            if calculate_fitness.early_stop_checks <= 10:
                debug_print_fitness(f"Early stop eval: partial_gmean={partial_gmean:.3f} vs threshold_50%={threshold_50pct:.3f} (should_discard={partial_gmean < threshold_50pct})")

            if partial_gmean < threshold_50pct:
                calculate_fitness.early_stop_discards += 1

                # DEBUG: Log descarte
                if calculate_fitness.early_stop_discards <= 5:
                    debug_print_fitness(f"EARLY STOP DESCARTE #{calculate_fitness.early_stop_discards}: {partial_gmean:.3f} < {threshold_50pct:.3f}")

                logging.debug(f"Early stop: partial_gmean={partial_gmean:.3f} < threshold={threshold_50pct:.3f} (50% of median elite)")
                return {
                    'fitness': -float('inf'),
                    'g_mean': partial_gmean,
                    'weighted_f1': 0.0,
                    'early_stopped': True  # Flag para diagnóstico
                }

            # Se passou do threshold, avaliar os 80% restantes
            remaining_data = data[partial_size:]
            remaining_predictions = [individual._predict(inst) for inst in remaining_data]
            predictions = partial_predictions + remaining_predictions

        except Exception as e:
            # Se falhar early stop, avaliar tudo normalmente
            logging.debug(f"Early stop failed, evaluating full data: {e}")
            predictions = [individual._predict(inst) for inst in data]
    else:
        # Modo normal: Avalia todos os dados de uma vez
        # DEBUG: Log quando early stop não está ativo
        if not hasattr(calculate_fitness, 'no_early_stop_logged'):
            calculate_fitness.no_early_stop_logged = True
            debug_print_fitness(f"Early stop DESATIVADO: threshold={early_stop_threshold}")
        try:
            predictions = [individual._predict(inst) for inst in data]
        except Exception as e:
            logging.error(f"Erro ao calcular predições: {e}", exc_info=False)
            return {'fitness': -float('inf'), 'g_mean': 0.0, 'weighted_f1': 0.0}

    # --- 1. Calcular Métricas de Performance ---
    try:
        g_mean = calculate_gmean_contextual(target, predictions, individual.classes)
        weighted_f1 = f1_score(target, predictions, labels=individual.classes, average='weighted', zero_division=0)
    except Exception as e:
        logging.error(f"Erro ao calcular métricas de performance: {e}", exc_info=False)
        return {'fitness': -float('inf'), 'g_mean': 0.0, 'weighted_f1': 0.0}

    # --- 2. Definir Coeficientes de Penalidade Efetivos ---
    # coverage_bonus = 0.0
    # if class_coverage_coefficient > 0:
    #     # Calcula o recall para cada classe individualmente usando as predições finais
    #     per_class_recall = recall_score(target, predictions, labels=individual.classes, average=None, zero_division=0)
        
    #     # Conta quantas classes tiveram um recall maior que zero
    #     num_classes_with_recall = np.count_nonzero(per_class_recall)
        
    #     # O bônus é proporcional à fração de classes que o indivíduo "não ignorou"
    #     total_classes = len(individual.classes)
    #     coverage_ratio = (num_classes_with_recall / total_classes) if total_classes > 0 else 0.0
        
    #     coverage_bonus = class_coverage_coefficient * coverage_ratio
    #     logging.debug(f"Coverage Bonus: {num_classes_with_recall}/{total_classes} classes com recall > 0 -> Bônus = {coverage_bonus:.4f}")
  
    coverage_bonus = 0.0
    if class_coverage_coefficient > 0 and class_weights:
        per_class_recall = recall_score(target, predictions, labels=individual.classes, average=None, zero_division=0)
        
        sum_of_weights_of_covered_classes = 0.0
        # Itera sobre cada classe e seu recall correspondente
        for i, class_label in enumerate(individual.classes):
            # Se a classe teve pelo menos um acerto (recall > 0)...
            if per_class_recall[i] > 0:
                # ...some o peso daquela classe ao bônus.
                sum_of_weights_of_covered_classes += class_weights.get(class_label, 1.0) # Usa 1.0 como fallback

        # Normaliza o bônus pela soma de todos os pesos possíveis para manter a escala
        total_possible_weight_sum = sum(class_weights.values())
        if total_possible_weight_sum > 0:
            coverage_ratio_weighted = sum_of_weights_of_covered_classes / total_possible_weight_sum
            coverage_bonus = class_coverage_coefficient * coverage_ratio_weighted

            
    current_reg_coeff = regularization_coefficient
    current_feat_coeff = feature_penalty_coefficient
    current_gamma = gamma
    current_beta = beta

    if reduce_change_penalties_active:
        logging.debug("Penalties are temporarily disabled to adapt to drift.")
        current_reg_coeff = 0.0
        current_feat_coeff = 0.0
        current_gamma = 0.0
        current_beta = 0.0
        coverage_bonus=0.0
        # Adicionar outros coeficientes a serem zerados aqui, se necessário

    # --- 3. Calcular Penalidades ---
    
    # a) Penalidade de Complexidade
    total_nodes = individual.count_total_nodes()
    total_rules = individual.count_total_rules()
    complexity_penalty = current_reg_coeff * (total_nodes + 5 * total_rules)

    # b) Penalidade por Número de Features
    used_features = individual.get_used_attributes()
    feature_penalty = current_feat_coeff * len(used_features)

    # c) Penalidade de Instabilidade vs. Memória (Jaccard Distance)
    distance_penalty = 0.0
    if current_beta > 0.0 and reference_features is not None:
        distance = compute_features_distance(used_features, reference_features)
        distance_penalty = current_beta * distance

    # d) Penalidade de Instabilidade vs. Chunk Anterior (Aumento de Features)
    change_penalty = 0.0
    if current_gamma > 0.0 and previous_used_features is not None:
        delta_features = len(used_features - set(previous_used_features))
        if delta_features > 0:
            change_penalty = current_gamma * delta_features

    # --- 4. Calcular Fitness Final ---
    # A fórmula agora é clara: Performance - Penalidades
    
#    performance_score = weighted_f1 + (gmean_bonus_coefficient * g_mean)
#    performance_score = weighted_f1 + (gmean_bonus_coefficient * g_mean) + coverage_bonus
    performance_score = g_mean + (weighted_f1 * gmean_bonus_coefficient) + coverage_bonus
    
    total_penalty = (
        complexity_penalty +
        feature_penalty +
        distance_penalty +
        change_penalty
    )

    # PENALTY_WEIGHT: Controla o balanceamento entre performance e complexidade
    # 0.0 = Foco total em performance (ignora complexidade)
    # 0.1 = Estrategia Suave (10% peso para penalidades) - BALANCEADO
    # 0.3 = Estrategia Moderada (30% peso para penalidades)
    # 0.5 = Estrategia Forte (50% peso para penalidades)
    PENALTY_WEIGHT = 0.1  # Estrategia A (Suave) - Balanceamento performance vs complexidade 

    fitness_score = performance_score - (PENALTY_WEIGHT * total_penalty)

    if np.isnan(fitness_score) or np.isinf(fitness_score):
        logging.error(f"Fitness resulted in NaN/Inf. Performance: {performance_score}, Penalty: {total_penalty}")
        return -float('inf')
        
    return {
        'fitness': fitness_score,
        'g_mean': g_mean,
        'weighted_f1': weighted_f1
    }