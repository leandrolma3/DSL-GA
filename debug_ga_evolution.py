# # debug_ga_evolution.py
# # Script para simular e depurar o fluxo completo de algumas gerações do AG.

# import logging
# import numpy as np
# from sklearn.metrics import classification_report
# import random
# import copy
# from collections import Counter
# from sklearn.metrics import accuracy_score, f1_score
# import itertools
# import matplotlib.pyplot as plt

# # --- Importações do seu projeto ---
# try:
#     import data_handling
#     import ga
#     import fitness
#     import metrics
#     import utils
#     import ga_operators
#     from individual import Individual
#     from rule_tree import RuleTree
#     from river import tree
#     from constants import RANDOM_SEED
# except ImportError as e:
#     print(f"Erro de importação. Certifique-se que o script está no diretório correto: {e}")
#     exit()

# # --- Configuração ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# logger = logging.getLogger("GA_EVOLUTION_DEBUGGER")

# # --- PARÂMETROS DE TESTE ---
# CONFIG_FILE_PATH = 'config.yaml'
# TARGET_EXPERIMENT_ID = 'CovType' # Ou 'PokerHand'
# NUM_GENERATIONS_TO_DEBUG = 5 # Quantas gerações vamos simular após a inicial

# from collections import Counter # Garanta que esta importação esteja presente

# def _calculate_class_weights(target_chunk: list, all_classes: list) -> dict:
#     """
#     Calcula pesos para cada classe com base no inverso de sua frequência.
#     """
#     if not target_chunk:
#         return {c: 1.0 for c in all_classes}
#     n_samples = len(target_chunk)
#     counts = Counter(target_chunk)
#     raw_weights = {}
#     for c in all_classes:
#         count = counts.get(c, 0)
#         raw_weights[c] = n_samples / (count + 1)
#     sum_of_raw_weights = sum(raw_weights.values())
#     num_classes = len(all_classes)
#     normalized_weights = {}
#     if sum_of_raw_weights > 0:
#         for c, rw in raw_weights.items():
#             normalized_weights[c] = (rw / sum_of_raw_weights) * num_classes
#     else:
#         return {c: 1.0 for c in all_classes}
#     return normalized_weights

# def calculate_fitness_with_coverage_bonus(individual, data, target, config_params):
#     """Função de fitness de debug que implementa a nova estratégia."""
    
#     # 1. Calcular Performance Base
#     try:
#         predictions = [individual._predict(inst) for inst in data]
#         weighted_f1 = f1_score(target, individual.classes, predictions, average='weighted', zero_division=0)
#     except Exception:
#         weighted_f1 = 0.0

#     # 2. Calcular Bônus de Cobertura
#     # Atualiza os scores de qualidade e as classes cobertas no indivíduo
#     individual.update_rule_quality_scores(data, target)
#     num_covered = len(individual.covered_classes)
#     total_classes = len(individual.classes)
#     coverage_ratio = (num_covered / total_classes) if total_classes > 0 else 0
    
#     coverage_coeff = config_params.get('class_coverage_coefficient', 0.5)
#     coverage_bonus = coverage_coeff * coverage_ratio

#     # 3. Calcular Penalidades
#     reg_coeff = config_params.get('initial_regularization_coefficient', 0.001)
#     feat_coeff = config_params.get('feature_penalty_coefficient', 0.0005)
    
#     complexity_penalty = reg_coeff * (individual.count_total_nodes() + 5 * individual.count_total_rules())
#     feature_penalty = feat_coeff * len(individual.get_used_attributes())
#     total_penalty = complexity_penalty + feature_penalty
    
#     # 4. Fitness Final
#     fitness_score = weighted_f1 + coverage_bonus - total_penalty
#     return fitness_score

# # --- FUNÇÃO DE ANÁLISE (RELATÓRIO REUTILIZÁVEL) ---
# # Em debug_ga_evolution.py

# def analyze_and_report_population(population, generation_num, train_data, train_target, classes, fitness_args):
#     """Gera um relatório completo sobre o estado atual de uma população."""
#     logger.info(f"\n" + "="*80)
#     logger.info(f"INICIANDO ANÁLISE DA GERAÇÃO {generation_num}")
#     logger.info("="*80)
    
#     g_means, fitness_scores = [], []
    
#     # <<< INÍCIO DA CORREÇÃO >>>
#     for ind in population:
#         # 1. Chama a função de fitness UMA VEZ para obter todas as métricas.
#         metrics_dict = fitness.calculate_fitness(ind, train_data, train_target, **fitness_args)
        
#         # 2. Desempacota os valores do dicionário.
#         score = metrics_dict.get('fitness', -float('inf'))
#         g_mean = metrics_dict.get('g_mean', 0.0)

#         # 3. Adiciona os valores numéricos às listas.
#         g_means.append(g_mean)
#         fitness_scores.append(score)
        
#         # 4. Atribui os valores numéricos corretos aos atributos do indivíduo.
#         ind.gmean = g_mean
#         ind.fitness = score
#     # <<< FIM DA CORREÇÃO >>>

#     print("\n--- ESTATÍSTICAS DA POPULAÇÃO ---")
#     print(f"G-Mean -> Média: {np.mean(g_means):.4f}, Desv. Padrão: {np.std(g_means):.4f}")
#     print(f"G-Mean -> Mínimo: {min(g_means):.4f}, MÁXIMO: {max(g_means):.4f}")
#     print("-" * 20)
#     print(f"Fitness -> Média: {np.mean(fitness_scores):.4f}, Desv. Padrão: {np.std(fitness_scores):.4f}")
#     print(f"Fitness -> Mínimo: {min(fitness_scores):.4f}, Máximo: {max(fitness_scores):.4f}")
#     print(f"Diversidade: {utils.calculate_population_diversity(population):.4f}\n")
    
#     # Encontra o melhor indivíduo por G-mean (se houver algum com gmean > 0)
#     best_gmean_ind = max(population, key=lambda ind: ind.gmean)
    
#     print("\n--- ANÁLISE DO MELHOR INDIVÍDUO (POR G-MEAN) ---")
#     print(f"Melhor G-Mean: {best_gmean_ind.gmean:.4f} (Fitness: {best_gmean_ind.fitness:.4f})")
#     best_predictions = [best_gmean_ind._predict(inst) for inst in train_data]
#     report = classification_report(train_target, best_predictions, labels=classes, zero_division=0, digits=4)
#     print("Relatório de Classificação:")
#     print(report)

# # --- INÍCIO DO SCRIPT DE DEPURAÇÃO ---

# if __name__ == "__main__":
#     logger.info("="*80)
#     logger.info("INICIANDO SCRIPT DE DEPURAÇÃO DO LOOP DO ALGORITMO GENÉTICO")
#     logger.info("="*80)

#     # <<< INÍCIO DO CÓDIGO DE SETUP QUE ESTAVA FALTANDO >>>
#     # --- FASE 1: SETUP (IMITANDO MAIN.PY) ---
#     logger.info("--- FASE 1: CARREGANDO CONFIGURAÇÃO E DADOS ---")
#     config = data_handling.load_full_config(CONFIG_FILE_PATH)
#     if not config: exit()
    
#     ga_params = config['ga_params']
#     fitness_params = config['fitness_params']
#     chunk_size = config['data_params']['chunk_size']
    
#     chunks = data_handling.generate_dataset_chunks(
#         TARGET_EXPERIMENT_ID, chunk_size, 1, chunk_size, CONFIG_FILE_PATH
#     )
    
#     # Converte os dados para os tipos corretos
#     for i in range(len(chunks)):
#         X, y = chunks[i]
#         if not X: continue
#         for inst in X:
#             for k, v in inst.items():
#                 try: inst[k] = float(v)
#                 except (ValueError, TypeError): pass
#         try:
#             chunks[i] = (X, [int(label) for label in y])
#         except (ValueError, TypeError):
#             chunks[i] = (X, y)
    
#     train_data, train_target = chunks[0]
#     all_classes = sorted(list(np.unique(train_target)))
#     attributes = sorted(list(train_data[0].keys()))
#     categorical_features = {a for a in attributes if isinstance(train_data[0].get(a), str)}
#     numeric_features = set(attributes) - categorical_features
#     value_ranges = {a: (min(d[a] for d in train_data), max(d[a] for d in train_data)) for a in numeric_features if train_data}
#     category_values = {a: {d[a] for d in train_data} for a in categorical_features}
    
#     logger.info(f"Dados do chunk 0 carregados. Instâncias: {len(train_data)}. Classes: {all_classes}")
#     # <<< FIM DO CÓDIGO DE SETUP QUE ESTAVA FALTANDO >>>

#     # --- FASE 2: INICIALIZAÇÃO DA POPULAÇÃO ---
#     logger.info("\n" + "--- FASE 2: GERANDO A POPULAÇÃO INICIAL (GERAÇÃO 0) ---")
#     current_population = ga.initialize_population(
#         population_size=ga_params['population_size'],
#         max_rules_per_class=ga_params['max_rules_per_class'],
#         max_depth=ga_params['initial_max_depth'],
#         attributes=attributes,
#         value_ranges=value_ranges,
#         category_values=category_values,
#         categorical_features=categorical_features,
#         classes=all_classes,
#         train_data=train_data,
#         train_target=train_target,
#         regularization_coefficient=fitness_params['initial_regularization_coefficient'],
#         feature_penalty_coefficient=fitness_params['feature_penalty_coefficient'],
#         operator_penalty_coefficient=fitness_params['operator_penalty_coefficient'],
#         threshold_penalty_coefficient=fitness_params['threshold_penalty_coefficient'],
#         intelligent_mutation_rate=ga_params.get('intelligent_mutation_rate', 0.2),
#         enable_dt_seeding_on_init_config=ga_params.get('enable_dt_seeding_on_init', True)
#     )
#     class_weights = _calculate_class_weights(train_target, all_classes)
#     fitness_args = {
#         'gmean_bonus_coefficient': fitness_params.get('gmean_bonus_coefficient', 0.1),
#         'class_coverage_coefficient': fitness_params.get('class_coverage_coefficient', 0.1),
#         'regularization_coefficient': fitness_params.get('initial_regularization_coefficient', 0.001),
#         'feature_penalty_coefficient': fitness_params.get('feature_penalty_coefficient', 0.0005),
#         'class_weights': class_weights
#     }

#     # --- ANÁLISE DA GERAÇÃO 0 ---
#     analyze_and_report_population(current_population, 0, train_data, train_target, all_classes, fitness_args)

#     # --- FASE 3: SIMULANDO A EVOLUÇÃO ---
#     for gen in range(NUM_GENERATIONS_TO_DEBUG):
#         new_population = []
#         population_size = len(current_population)
        
#         # 1. Elitismo de Nicho
#         num_elite_slots = int(ga_params.get('elitism_rate', 0.1) * population_size)
#         if num_elite_slots > 0 and current_population:
#             current_population.sort(key=lambda ind: ind.fitness, reverse=True)
#             fitness_elite = current_population[0]
#             new_population.append(copy.deepcopy(fitness_elite))
#             if len(current_population) > 1:
#                 gmean_elite = max(current_population, key=lambda ind: ind.gmean)
#                 if gmean_elite is not fitness_elite and len(new_population) < num_elite_slots:
#                     new_population.append(copy.deepcopy(gmean_elite))
        
#         # 2. Geração de Filhos
#         tournament_size = ga_params.get('initial_tournament_size', 3)
#         intelligent_mutation_prob = ga_params.get('intelligent_mutation_rate', 0.5)
#         mutation_rate = 0.2
#         max_depth = ga_params.get('initial_max_depth', 4)
#         max_rules = ga_params.get('max_rules_per_class', 3)

#         while len(new_population) < population_size:
#             p1 = ga_operators.tournament_selection(current_population, tournament_size)
#             p2 = ga_operators.tournament_selection(current_population, tournament_size)
#             if not p1 or not p2: continue

#             child = ga_operators.crossover(
#                 p1, p2, max_depth, attributes, value_ranges,
#                 category_values, categorical_features,
#                 all_classes, max_rules, crossover_type="node"
#             )
            
#             ga_operators.mutate_individual(
#                 individual=child,
#                 mutation_rate=mutation_rate,
#                 max_depth=max_depth,
#                 intelligent_mutation_prob=intelligent_mutation_prob,
#                 attributes=attributes,
#                 value_ranges=value_ranges,
#                 category_values=category_values,
#                 categorical_features=categorical_features,
#                 classes=all_classes,
#                 max_rules_per_class=max_rules,
#                 data=train_data,
#                 target=train_target
#             )
#             new_population.append(child)

#         current_population = new_population
        
#         # --- ANÁLISE DA NOVA GERAÇÃO ---
#         analyze_and_report_population(current_population, gen + 1, train_data, train_target, all_classes, fitness_args)

#     logger.info("="*80)
#     logger.info("DEPURAÇÃO DA EVOLUÇÃO CONCLUÍDA.")
#     logger.info("="*80)


# debug_ga_evolution.py (v2 - com Seeding Multi-Profundidade e Loop Evolucionário)

import logging
import numpy as np
from sklearn.metrics import classification_report
import random
import copy
from collections import Counter
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt

# --- Importações de Machine Learning ---
from sklearn.tree import DecisionTreeClassifier

# --- Importações do seu projeto ---
try:
    import data_handling
    import ga
    import fitness
    import metrics
    import utils
    import ga_operators
    from individual import Individual
    from rule_tree import RuleTree
    from node import Node
except ImportError as e:
    print(f"Erro de importação: {e}. Certifique-se que o script está no diretório correto.")
    exit()

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("GA_EVOLUTION_DEBUGGER")

# --- PARÂMETROS DE TESTE ---
CONFIG_FILE_PATH = 'config.yaml'
TARGET_EXPERIMENT_ID = 'CovType'
NUM_GENERATIONS_TO_DEBUG = 10 # Vamos simular mais gerações para ver a evolução

# ==============================================================================
# --- FUNÇÕES AUXILIARES (copiadas dos scripts de debug anteriores) ---
# ==============================================================================

def _load_and_prepare_data(csv_path: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    df = df.loc[:,~df.columns.duplicated()]
    y_data = df.pop('class').tolist()
    X_data_dict = df.to_dict('records')
    all_classes = sorted(list(np.unique(y_data)))
    attributes = list(df.columns)
    base_attributes_info = { "attributes": attributes, "value_ranges": {a: (df[a].min(), df[a].max()) for a in attributes if df[a].dtype in ['int64', 'float64']}, "category_values": {a: set(df[a].unique()) for a in attributes if df[a].dtype == 'object'}, "categorical_features": {a for a in attributes if df[a].dtype == 'object'} }
    return { "X_dict": X_data_dict, "y": y_data, "df": df, "all_classes": all_classes, "base_attributes_info": base_attributes_info }

def _extract_all_rules_from_dt(dt_model: DecisionTreeClassifier, feature_names: List[str], base_attributes_info: Dict) -> Dict[int, List[RuleTree]]:
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

def analyze_and_report_population(population, generation_num, X_data, y_data, all_classes, fitness_args):
    logger.info(f"\n" + "="*80 + f"\nINICIANDO ANÁLISE DA GERAÇÃO {generation_num}" + "\n" + "="*80)
    g_means, fitness_scores = [], []
    for ind in population:
        metrics_dict = fitness.calculate_fitness(ind, X_data, y_data, **fitness_args)
        ind.fitness = metrics_dict.get('fitness', -float('inf'))
        ind.gmean = metrics_dict.get('g_mean', 0.0)
        g_means.append(ind.gmean)
        fitness_scores.append(ind.fitness)
    
    print("\n--- ESTATÍSTICAS DA POPULAÇÃO ---")
    print(f"G-Mean -> Média: {np.mean(g_means):.4f}, Desv. Padrão: {np.std(g_means):.4f}")
    print(f"G-Mean -> Mínimo: {min(g_means):.4f}, MÁXIMO: {max(g_means):.4f}")
    print("-" * 20)
    print(f"Fitness -> Média: {np.mean(fitness_scores):.4f}, Desv. Padrão: {np.std(fitness_scores):.4f}")
    
    best_gmean_ind = max(population, key=lambda ind: ind.gmean)
    print("\n--- ANÁLISE DO MELHOR INDIVÍDUO (POR G-MEAN) ---")
    print(f"Melhor G-Mean: {best_gmean_ind.gmean:.4f} (Fitness: {best_gmean_ind.fitness:.4f})")
    best_predictions = [best_gmean_ind._predict(inst) for inst in X_data]
    print("Relatório de Classificação:")
    print(classification_report(y_data, best_predictions, labels=all_classes, zero_division=0, digits=4))
    return best_gmean_ind.gmean # Retorna o melhor G-mean da geração

# ==============================================================================
# --- SCRIPT DE DEPURAÇÃO PRINCIPAL ---
# ==============================================================================

if __name__ == "__main__":
    logger.info("="*80 + "\nINICIANDO SCRIPT DE DEPURAÇÃO DO LOOP DO ALGORITMO GENÉTICO\n" + "="*80)
    
    # --- FASE 1: SETUP (IMITANDO MAIN.PY) ---
    logger.info("--- FASE 1: CARREGANDO CONFIGURAÇÃO E DADOS ---")
    config = data_handling.load_full_config(CONFIG_FILE_PATH)
    if not config: exit()
    
    data_assets = _load_and_prepare_data(f"stationary_experiment_results_CovPo/{TARGET_EXPERIMENT_ID}/run_1/chunk_data/chunk_0_train.csv")
    X_train_dict, y_train, df_train, all_classes, base_attributes_info = \
        [data_assets[k] for k in ["X_dict", "y", "df", "all_classes", "base_attributes_info"]]
    
    # --- FASE 2: GERANDO A POPULAÇÃO INICIAL (ESTRATÉGIA SEEDING MULTI-PROFUNDIDADE) ---
    logger.info("\n" + "--- FASE 2: GERANDO A POPULAÇÃO INICIAL (GERAÇÃO 0) ---")
    POPULATION_SIZE = 100
    DEPTHS_TO_TRAIN = [4, 7, 10, 13]
    SEEDED_RATIO = 0.60
    
    # Treinar modelos-guia
    generalist_models = {}
    for depth in DEPTHS_TO_TRAIN:
        dt_model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=42, class_weight='balanced')
        dt_model.fit(df_train, y_train)
        generalist_models[depth] = dt_model
    
    # Gerar clones mestres
    master_clones = {}
    for depth, model in generalist_models.items():
        gene_pool = _extract_all_rules_from_dt(model, list(df_train.columns), base_attributes_info)
        clone = Individual(max_rules_per_class=sum(len(r) for r in gene_pool.values()), max_depth=depth, **base_attributes_info, classes=all_classes, train_target=y_train, initialize_random_rules=False)
        clone.rules = gene_pool
        master_clones[depth] = clone

    # Construir população híbrida
    current_population = []
    num_seeded_individuals = int(POPULATION_SIZE * SEEDED_RATIO)
    num_clones_per_depth = num_seeded_individuals // len(DEPTHS_TO_TRAIN)
    for depth, master_clone in master_clones.items():
        for _ in range(num_clones_per_depth):
            current_population.append(copy.deepcopy(master_clone))
    num_randoms = POPULATION_SIZE - len(current_population)
    for _ in range(num_randoms):
        random_ind = Individual(max_rules_per_class=3, max_depth=5, **base_attributes_info, classes=all_classes, train_target=y_train, initialize_random_rules=True)
        current_population.append(random_ind)
    
    # --- ANÁLISE DA GERAÇÃO 0 ---
    ga_params = config['ga_params']
    fitness_params = config['fitness_params']
    class_weights = Counter(y_train) # Simples contagem para pesos
    fitness_args = {
        'class_coverage_coefficient': fitness_params.get('class_coverage_coefficient', 0.2),
        'gmean_bonus_coefficient': fitness_params.get('f1_bonus_coefficient', 0.1),
        'regularization_coefficient': fitness_params.get('initial_regularization_coefficient', 0.001),
        'feature_penalty_coefficient': fitness_params.get('feature_penalty_coefficient', 0.0005),
        'class_weights': class_weights
    }
    
    history_best_gmean = []
    best_gmean_gen0 = analyze_and_report_population(current_population, 0, X_train_dict, y_train, all_classes, fitness_args)
    history_best_gmean.append(best_gmean_gen0)

    # --- FASE 3: SIMULANDO A EVOLUÇÃO ---
    for gen in range(NUM_GENERATIONS_TO_DEBUG):
        new_population = []
        
        # 1. Elitismo Híbrido Unificado (Lógica do ga.py)
        num_elite_slots = int(ga_params.get('elitism_rate', 0.1) * POPULATION_SIZE)
        if num_elite_slots > 0:
            fitness_champion = max(current_population, key=lambda ind: ind.fitness)
            population_sorted_by_gmean = sorted(current_population, key=lambda ind: ind.gmean, reverse=True)
            top_gmean_individuals = population_sorted_by_gmean[:num_elite_slots]
            candidate_elites = {ind.get_rules_as_string(): ind for ind in top_gmean_individuals}
            candidate_elites[fitness_champion.get_rules_as_string()] = fitness_champion
            final_candidate_list = sorted(list(candidate_elites.values()), key=lambda ind: ind.gmean, reverse=True)
            final_elites = final_candidate_list[:num_elite_slots]
            new_population.extend(copy.deepcopy(ind) for ind in final_elites)
        
        # 2. Geração de Filhos
        tournament_size = ga_params.get('initial_tournament_size', 3)
        while len(new_population) < POPULATION_SIZE:
            p1 = ga_operators.tournament_selection(current_population, tournament_size)
            p2 = ga_operators.tournament_selection(current_population, tournament_size)
            if not p1 or not p2: continue
            child = ga_operators.crossover(p1, p2, max_depth=ga_params.get('initial_max_depth', 5), max_rules_per_class=ga_params.get('max_rules_per_class', 3), **base_attributes_info, classes=all_classes)
            ga_operators.mutate_individual(individual=child, mutation_rate=0.2, intelligent_mutation_prob=0.5, data=X_train_dict, target=y_train, **base_attributes_info, classes=all_classes, max_rules_per_class=ga_params.get('max_rules_per_class', 3), max_depth=ga_params.get('initial_max_depth', 5))
            new_population.append(child)

        current_population = new_population
        
        # ANÁLISE DA NOVA GERAÇÃO
        best_gmean_gen = analyze_and_report_population(current_population, gen + 1, X_train_dict, y_train, all_classes, fitness_args)
        history_best_gmean.append(best_gmean_gen)

    # --- FASE 4: PLOT DA EVOLUÇÃO ---
    logger.info("\n" + "="*80 + "\nDEPURAÇÃO DA EVOLUÇÃO CONCLUÍDA. Plotando resultados..." + "\n" + "="*80)
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(history_best_gmean)), history_best_gmean, marker='o', linestyle='-', label='Melhor G-mean da Geração')
    plt.title('Evolução do Melhor G-mean por Geração', fontsize=16)
    plt.xlabel('Geração', fontsize=12)
    plt.ylabel('G-mean Score', fontsize=12)
    plt.xticks(range(len(history_best_gmean)))
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.show()