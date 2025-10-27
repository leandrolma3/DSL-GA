# # debug_population_generator.py
# # Script para gerar e analisar em profundidade uma população inicial.

# import logging
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix
# import itertools
# import random
# import matplotlib.pyplot as plt
# from typing import Dict, Any, Optional
# from collections import Counter

# # Adicione esta função auxiliar no topo do script
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

# # --- Importações do seu projeto ---
# try:
#     import data_handling
#     import ga
#     from individual import Individual
#     from utils import calculate_population_diversity
#     import metrics
#     import fitness
#     from constants import RANDOM_SEED
# except ImportError as e:
#     print(f"Erro de importação. Certifique-se que o script está no diretório correto: {e}")
#     exit()

# # --- Configuração ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# logger = logging.getLogger("POPULATION_DEBUGGER")

# # --- PARÂMETROS DE TESTE ---
# CONFIG_FILE_PATH = 'config.yaml'
# TARGET_EXPERIMENT_ID = 'CovType' # Ou 'PokerHand'
# # Para este teste, vamos simular uma situação de "drift" para forçar o seeding
# FORCED_INITIALIZATION_STRATEGY = 'full_random'
# FORCED_PERFORMANCE_LABEL = 'bad'

# # --- INÍCIO DO SCRIPT ---

# if __name__ == "__main__":
#     print("\n" + "="*80)
#     logger.info("INICIANDO SCRIPT DE ANÁLISE DA POPULAÇÃO INICIAL")
#     print("="*80)

#     # --- FASE 1: SETUP (IMITANDO MAIN.PY) ---
#     logger.info("--- FASE 1: CARREGANDO CONFIGURAÇÃO E DADOS ---")
#     config = data_handling.load_full_config(CONFIG_FILE_PATH)
#     if not config: exit()
    
#     chunk_size = config['data_params']['chunk_size']
#     chunks = data_handling.generate_dataset_chunks(TARGET_EXPERIMENT_ID, chunk_size, 1, chunk_size, CONFIG_FILE_PATH)
    
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
    
#     logger.info(f"Dados do chunk 0 carregados para '{TARGET_EXPERIMENT_ID}'.")

#     # --- FASE 2: GERAÇÃO DA POPULAÇÃO ---
#     logger.info("\n" + f"--- FASE 2: GERANDO POPULAÇÃO INICIAL (Estratégia Forçada: '{FORCED_INITIALIZATION_STRATEGY}') ---")
#     random.seed(RANDOM_SEED)
#     np.random.seed(RANDOM_SEED)
    
#     ga_params = config['ga_params']
#     fitness_params = config['fitness_params']

#     # <<< CORREÇÃO: Chamada completa para ga.initialize_population >>>
#     initial_population = ga.initialize_population(
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
#         initialization_strategy=FORCED_INITIALIZATION_STRATEGY,
#         performance_label=FORCED_PERFORMANCE_LABEL,
#         enable_dt_seeding_on_init_config=True, # Força o seeding para o teste
#         dt_seeding_ratio_on_init_config=ga_params.get('dt_seeding_ratio_on_init', 0.5)
#     )
    
#     # --- FASE 3: AVALIAÇÃO E ANÁLISE DA POPULAÇÃO ---
#     logger.info("\n" + "--- FASE 3: AVALIANDO E ANALISANDO A QUALIDADE DA POPULAÇÃO GERADA ---")
    
#     g_means = []
#     fitness_scores = []
#     # Calcula os pesos das classes primeiro
#     class_weights = _calculate_class_weights(train_target, all_classes) # Supondo que você adicione a função auxiliar

#     fitness_args = {
#         'regularization_coefficient': fitness_params.get('initial_regularization_coefficient', 0.001),
#         'feature_penalty_coefficient': fitness_params.get('feature_penalty_coefficient', 0.0005),
#         'class_coverage_coefficient': fitness_params.get('class_coverage_coefficient', 0.8), # Adicionado
#         'gmean_bonus_coefficient': fitness_params.get('gmean_bonus_coefficient', 0.3),     # Adicionado
#         'class_weights': class_weights # Adicionado
#     }    
#     # fitness_args = {
#     #     'gmean_bonus_coefficient': fitness_params.get('gmean_bonus_coefficient', 0.1),
#     #     'regularization_coefficient': fitness_params.get('initial_regularization_coefficient', 0.001),
#     #     'feature_penalty_coefficient': fitness_params.get('feature_penalty_coefficient', 0.0005),
#     #     'operator_penalty_coefficient': fitness_params.get('operator_penalty_coefficient', 0.0),
#     #     'threshold_penalty_coefficient': fitness_params.get('threshold_penalty_coefficient', 0.0),
#     #     'operator_change_coefficient': fitness_params.get('operator_change_coefficient', 0.0),
#     #     'gamma': fitness_params.get('gamma', 0.0)
#     # }

#     for ind in initial_population:
#         predictions = [ind._predict(inst) for inst in train_data]
#         g_mean = metrics.calculate_gmean_contextual(train_target, predictions, all_classes)
#         score = fitness.calculate_fitness(ind, train_data, train_target, **fitness_args)
#         g_means.append(g_mean)
#         fitness_scores.append(score)
#         ind.fitness = score

#     # --- 3.1: Estatísticas Gerais ---
#     print("\n--- ESTATÍSTICAS DA POPULAÇÃO INICIAL ---")
#     print(f"G-Mean -> Média: {np.mean(g_means):.4f}, Desv. Padrão: {np.std(g_means):.4f}")
#     print(f"G-Mean -> Mínimo: {min(g_means):.4f}, MÁXIMO: {max(g_means):.4f}  <--- (ESTE É O VALOR MAIS IMPORTANTE)")
#     print("-" * 20)
#     print(f"Fitness -> Média: {np.mean(fitness_scores):.4f}, Desv. Padrão: {np.std(fitness_scores):.4f}")
#     print(f"Fitness -> Mínimo: {min(fitness_scores):.4f}, Máximo: {max(fitness_scores):.4f}")
    
#     diversity = calculate_population_diversity(initial_population)
#     print(f"\nDiversidade da População: {diversity:.4f}")

#     # --- 3.2: Análise Profunda do Melhor Indivíduo ---
#     print("\n--- ANÁLISE DO MELHOR INDIVÍDUO DA POPULAÇÃO INICIAL ---")
#     best_initial_ind = initial_population[np.argmax(g_means)]
    
#     print(f"Melhor G-Mean encontrado: {max(g_means):.4f}")
#     print(f"Fitness correspondente: {best_initial_ind.fitness:.4f}")
#     print("\nRegras do Melhor Indivíduo:")
#     print(best_initial_ind.get_rules_as_string())
    
#     logger.info("Calculando relatório de classificação para o melhor indivíduo...")
#     best_predictions = [best_initial_ind._predict(inst) for inst in train_data]
#     report = classification_report(train_target, best_predictions, labels=all_classes, zero_division=0, digits=4)
#     print("\nRelatório de Classificação do Melhor Indivíduo:")
#     print(report)

#     # --- 3.3: Visualização ---
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#     axes[0].hist(g_means, bins=20, color='skyblue', edgecolor='black')
#     axes[0].set_title('Distribuição do G-Mean na População Inicial')
#     axes[0].set_xlabel('G-Mean Score')
#     axes[0].set_ylabel('Contagem de Indivíduos')

#     axes[1].hist(fitness_scores, bins=20, color='salmon', edgecolor='black')
#     axes[1].set_title('Distribuição do Fitness na População Inicial')
#     axes[1].set_xlabel('Fitness Score')
    
#     fig.tight_layout()
#     plt.show()

#     print("\n" + "="*80)
#     logger.info("DEPURAÇÃO DA POPULAÇÃO INICIAL CONCLUÍDA.")
#     print("="*80)

# debug_population_generator.py (v2 - Estratégia Híbrida + Análise Detalhada)

# debug_population_generator.py (v5 - Estratégia "Seeding Multi-Profundidade")

import os
import logging
import argparse
import pandas as pd
import numpy as np
import copy
import random
from collections import Counter
from typing import Dict, List, Any

import matplotlib.pyplot as plt

# --- Importações de Machine Learning ---
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# --- Importações do Projeto GBML ---
try:
    from individual import Individual
    from rule_tree import RuleTree
    from node import Node
    import metrics
except ImportError as e:
    print(f"Erro de importação: {e}. Certifique-se que o script está na pasta raiz do projeto.")
    exit()

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("MultiDepthSeedingDebugger")


# ==============================================================================
# --- FUNÇÕES AUXILIARES (sem alterações) ---
# ==============================================================================

def _load_and_prepare_data(csv_path: str) -> Dict[str, Any]:
    logger.info(f"Carregando e preparando dados de: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.loc[:,~df.columns.duplicated()]
    y_data = df.pop('class').tolist()
    X_data_dict = df.to_dict('records')
    all_classes = sorted(list(np.unique(y_data)))
    attributes = list(df.columns)
    base_attributes_info = { "attributes": attributes, "value_ranges": {a: (df[a].min(), df[a].max()) for a in attributes if df[a].dtype in ['int64', 'float64']}, "category_values": {a: set(df[a].unique()) for a in attributes if df[a].dtype == 'object'}, "categorical_features": {a for a in attributes if df[a].dtype == 'object'} }
    logger.info(f"Dados carregados: {len(y_data)} instâncias, {len(all_classes)} classes.")
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

# ==============================================================================
# --- SCRIPT PRINCIPAL ---
# ==============================================================================

def main(chunk_path: str):
    # --- Parâmetros da População e da Estratégia ---
    POPULATION_SIZE = 100
    # Define o portfólio de profundidades para as árvores-guia
    DEPTHS_TO_TRAIN = [4, 7, 10, 13]
    # Proporção da população que será semeada com indivíduos inteligentes (clones)
    SEEDED_RATIO = 0.60 # 60% clones, 40% aleatórios

    # --- Etapa 1: Preparação ---
    logger.info("--- FASE 1: Carregando Dados ---")
    data_assets = _load_and_prepare_data(chunk_path)
    X_train_dict, y_train, df_train, all_classes, base_attributes_info = \
        [data_assets[k] for k in ["X_dict", "y", "df", "all_classes", "base_attributes_info"]]

    # --- Etapa 2: Treinamento do Portfólio de Modelos-Guia ---
    logger.info(f"\n--- FASE 2: Treinando Portfólio de {len(DEPTHS_TO_TRAIN)} Modelos-Guia com Diferentes Profundidades ---")
    
    generalist_models = {}
    for depth in DEPTHS_TO_TRAIN:
        dt_model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=42, class_weight='balanced')
        dt_model.fit(df_train, y_train)
        generalist_models[depth] = dt_model
        logger.info(f"  - Modelo Generalista com max_depth={depth} treinado.")

    # --- Etapa 3: Geração dos Indivíduos-Semente (Clones de Múltiplas Profundidades) ---
    logger.info("\n--- FASE 3: Gerando Indivíduos-Semente a partir do Portfólio de Modelos ---")
    master_clones = {}
    for depth, model in generalist_models.items():
        gene_pool = _extract_all_rules_from_dt(model, list(df_train.columns), base_attributes_info)
        total_rules = sum(len(r) for r in gene_pool.values())
        
        clone = Individual(
            max_rules_per_class=total_rules, max_depth=depth,
            **base_attributes_info, classes=all_classes, train_target=y_train,
            initialize_random_rules=False
        )
        clone.rules = gene_pool
        master_clones[depth] = clone
    logger.info(f"  - {len(master_clones)} clones mestres (um para cada profundidade) criados com sucesso.")

    # --- Etapa 4: Construção da População Híbrida Final ---
    logger.info("\n--- FASE 4: Construção da População Híbrida Final (Seeding Multi-Profundidade) ---")
    population = []
    
    num_seeded_individuals = int(POPULATION_SIZE * SEEDED_RATIO)
    num_clones_per_depth = num_seeded_individuals // len(DEPTHS_TO_TRAIN)
    
    for depth, master_clone in master_clones.items():
        # Adiciona N cópias do clone mestre de cada profundidade
        for _ in range(num_clones_per_depth):
            population.append(copy.deepcopy(master_clone))
    logger.info(f"  - {len(population)} indivíduos clonados de diferentes profundidades adicionados.")
    
    # Preenche com aleatórios
    num_randoms = POPULATION_SIZE - len(population)
    for _ in range(num_randoms):
        random_ind = Individual(max_rules_per_class=3, max_depth=5, **base_attributes_info, classes=all_classes, train_target=y_train, initialize_random_rules=True)
        population.append(random_ind)
    logger.info(f"  - {num_randoms} Indivíduos Aleatórios adicionados.")
    logger.info(f"População final criada com {len(population)} indivíduos.")

    # --- Etapa 5: Análise da Nova População Híbrida ---
    logger.info("\n--- FASE 5: Análise da Qualidade da População 'Seeding Multi-Profundidade' ---")
    g_mean_scores = []
    for ind in population:
        predictions = [ind._predict(inst) for inst in X_train_dict]
        g_mean = metrics.calculate_gmean_contextual(y_train, predictions, all_classes)
        g_mean_scores.append(g_mean)
        ind.gmean = g_mean

    print("\n--- ESTATÍSTICAS GERAIS DA POPULAÇÃO INICIAL HÍBRIDA ---")
    print(f"  - MÁXIMO G-mean: {np.max(g_mean_scores):.4f}")
    print(f"  - MÉDIO G-mean:  {np.mean(g_mean_scores):.4f}")
    print(f"  - MÍNIMO G-mean: {np.min(g_mean_scores):.4f}")
    print(f"  - Desvio Padrão: {np.std(g_mean_scores):.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(g_mean_scores, bins=25, color='skyblue', edgecolor='black')
    plt.title('Distribuição do G-Mean (Estratégia Seeding Multi-Profundidade)', fontsize=16)
    plt.xlabel('G-Mean Score', fontsize=12)
    plt.ylabel('Número de Indivíduos', fontsize=12)
    plt.axvline(np.mean(g_mean_scores), color='red', linestyle='dashed', linewidth=2, label=f'Média: {np.mean(g_mean_scores):.2f}')
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.show()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Gerar e analisar uma população com a estratégia 'Seeding Multi-Profundidade'.")
#     parser.add_argument("chunk_path", type=str, help="Caminho para o arquivo CSV de um chunk de treino.")
#     args = parser.parse_args()
#     main(args.chunk_path)

if __name__ == '__main__':
    # Usaremos um caminho fixo para simplificar, mas a estrutura com argparse é mantida comentada
    # parser = argparse.ArgumentParser(description="Gerar e analisar uma população inicial híbrida.")
    # parser.add_argument("chunk_path", type=str, help="Caminho para o arquivo CSV de um chunk de treino.")
    # args = parser.parse_args()
    # main(args.chunk_path)

    # Para executar, apenas defina o caminho para o seu arquivo de treino aqui
    CHUNK_FILE_PATH = "stationary_experiment_results_CovPo/CovType/run_1/chunk_data/chunk_0_train.csv"
    main(CHUNK_FILE_PATH)