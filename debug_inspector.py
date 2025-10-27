# debug_full_ga_loop.py
# Script para simular e depurar um ciclo completo de uma geração do AG.

import logging
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import random

# --- Importações do seu projeto ---
try:
    import data_handling
    import ga
    import fitness
    import metrics
    from individual import Individual
    from constants import RANDOM_SEED
except ImportError as e:
    print(f"Erro de importação. Certifique-se que o script está no diretório correto: {e}")
    exit()

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("GA_LOOP_DEBUGGER")

# --- PARÂMETROS DE TESTE ---
CONFIG_FILE_PATH = 'config.yaml'
# Use o dataset que está apresentando o problema.
TARGET_EXPERIMENT_ID = 'CovType' # Ou 'PokerHand'

# --- INÍCIO DO SCRIPT DE DEPURAÇÃO ---

if __name__ == "__main__":
    print("\n" + "="*80)
    logger.info("INICIANDO SCRIPT DE DEPURAÇÃO DO LOOP DO ALGORITMO GENÉTICO")
    print("="*80)

    # --- FASE 1: SETUP (IMITANDO MAIN.PY) ---
    logger.info("--- FASE 1: CARREGANDO CONFIGURAÇÃO E DADOS ---")
    config = data_handling.load_full_config(CONFIG_FILE_PATH)
    if not config: exit()
    
    chunk_size = config['data_params']['chunk_size']
    chunks = data_handling.generate_dataset_chunks(TARGET_EXPERIMENT_ID, chunk_size, 1, chunk_size, CONFIG_FILE_PATH)
    
    # Converte os dados para os tipos corretos
    for i in range(len(chunks)):
        X, y = chunks[i]
        if not X: continue
        for inst in X:
            for k, v in inst.items():
                try: inst[k] = float(v)
                except (ValueError, TypeError): pass
        try:
            chunks[i] = (X, [int(label) for label in y])
        except (ValueError, TypeError):
            chunks[i] = (X, y)
    
    train_data, train_target = chunks[0]
    all_classes = sorted(list(np.unique(train_target)))
    attributes = sorted(list(train_data[0].keys()))
    categorical_features = {a for a in attributes if isinstance(train_data[0][a], str)}
    numeric_features = set(attributes) - categorical_features
    value_ranges = {a: (min(d[a] for d in train_data), max(d[a] for d in train_data)) for a in numeric_features}
    category_values = {a: {d[a] for d in train_data} for a in categorical_features}
    
    logger.info(f"Dados do chunk 0 carregados. Instâncias: {len(train_data)}. Classes: {all_classes}")

# --- FASE 2: INICIALIZAÇÃO DA POPULAÇÃO (IMITANDO GA.PY) ---
    logger.info("\n" + "--- FASE 2: GERANDO A POPULAÇÃO INICIAL ---")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # <<< INÍCIO DA CORREÇÃO >>>
    # A chamada agora inclui todos os parâmetros necessários que foram apontados pelo erro.

    initial_population = ga.initialize_population(
        # Parâmetros de estrutura
        population_size=config['ga_params']['population_size'],
        max_rules_per_class=config['ga_params']['max_rules_per_class'],
        max_depth=config['ga_params']['initial_max_depth'],
        
        # Parâmetros de dados
        attributes=attributes,
        value_ranges=value_ranges,
        category_values=category_values,
        categorical_features=categorical_features,
        classes=all_classes,
        train_data=train_data,                 # <<< ADICIONADO (estava faltando)
        train_target=train_target,
        
        # Parâmetros de fitness
        regularization_coefficient=config['fitness_params']['initial_regularization_coefficient'],
        feature_penalty_coefficient=config['fitness_params']['feature_penalty_coefficient'],
        operator_penalty_coefficient=config['fitness_params']['operator_penalty_coefficient'], # <<< ADICIONADO
        threshold_penalty_coefficient=config['fitness_params']['threshold_penalty_coefficient'], # <<< ADICIONADO
        intelligent_mutation_rate=config['ga_params']['intelligent_mutation_rate'],
        initialization_strategy='diversify',
        performance_label='bad',
        enable_dt_seeding_on_init_config=True, # Força o seeding para o teste
        dt_seeding_ratio_on_init_config=0.5
    )
    logger.info(f"População inicial com {len(initial_population)} indivíduos gerada.")

    # --- FASE 3: SIMULAÇÃO DE UMA GERAÇÃO (SERIAL E COM LOGS) ---
    logger.info("\n" + "--- FASE 3: SIMULANDO A AVALIAÇÃO DA 1ª GERAÇÃO (MODO SERIAL) ---")
    
    evaluated_population = []
    
    for i, ind in enumerate(initial_population):
        logger.info(f"\n--- Avaliando Indivíduo {i} ---")
        
        # 1. Gerar predições
        predictions = []
        rule_activations = 0
        for inst in train_data:
            prediction, activated = ind._predict_with_activation_check(inst)
            predictions.append(prediction)
            if activated:
                rule_activations += 1
        
        activation_rate = (rule_activations / len(train_data)) * 100 if train_data else 0
        logger.info(f"  Taxa de Ativação de Regras (RuleAct): {activation_rate:.2f}%")

        # 2. Calcular G-Mean
        g_mean = metrics.calculate_gmean_contextual(train_target, predictions, all_classes)
        logger.info(f"  G-Mean Calculado: {g_mean:.4f}")
        
        # 3. Calcular Fitness
        # fitness_score = fitness.calculate_fitness(
        #     ind, train_data, train_target,
        #     regularization_coefficient=config['fitness_params']['initial_regularization_coefficient'],
        #     feature_penalty_coefficient=config['fitness_params']['feature_penalty_coefficient']
        # )
        fitness_score = fitness.calculate_fitness(
            ind, train_data, train_target,
            regularization_coefficient=config['fitness_params']['initial_regularization_coefficient'],
            feature_penalty_coefficient=config['fitness_params']['feature_penalty_coefficient'],
            gmean_bonus_coefficient=config['fitness_params']['gmean_bonus_coefficient'] # Argumento adicionado
        )        
        logger.info(f"  Fitness Final Calculado: {fitness_score:.4f}")
        
        ind.fitness = fitness_score
        evaluated_population.append(ind)

    # --- FASE 4: ANÁLISE DO MELHOR INDIVÍDUO ---
    logger.info("\n" + "--- FASE 4: ANÁLISE DO MELHOR INDIVÍDUO DA GERAÇÃO ---")
    
    if not evaluated_population:
        logger.error("Nenhum indivíduo foi avaliado.")
        exit()

    # Ordena a população pelo fitness para encontrar o melhor
    evaluated_population.sort(key=lambda x: x.fitness, reverse=True)
    best_individual = evaluated_population[0]
    
    logger.info(f"Melhor Fitness Encontrado: {best_individual.fitness:.4f}")
    print("\nRegras do Melhor Indivíduo:")
    print(best_individual.get_rules_as_string())

    # --- FASE 5: ANÁLISE PROFUNDA DA PERFORMANCE DO MELHOR INDIVÍDUO ---
    logger.info("\n" + "--- FASE 5: ANÁLISE DETALHADA DAS PREDIÇÕES DO MELHOR INDIVÍDUO ---")
    
    final_predictions = [best_individual._predict(inst) for inst in train_data]
    final_gmean = metrics.calculate_gmean_contextual(train_target, final_predictions, all_classes)
    
    logger.info(f"G-Mean recalculado para o melhor indivíduo: {final_gmean:.4f}")
    
    print("\nDistribuição das Predições do Melhor Indivíduo:")
    print(dict(Counter(final_predictions)))
    
    print("\nMatriz de Confusão (Linhas=Verdadeiro, Colunas=Predito):")
    cm = confusion_matrix(train_target, final_predictions, labels=all_classes)
    print(cm)
    
    print("\nRelatório de Classificação Detalhado:")
    report = classification_report(train_target, final_predictions, labels=all_classes, zero_division=0)
    print(report)
    print("\n" + "="*80)
    logger.info("DEPURAÇÃO CONCLUÍDA. Verifique a coluna 'recall' no relatório acima.")
    print("="*80)