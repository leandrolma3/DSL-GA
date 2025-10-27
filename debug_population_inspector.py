# debug_population_inspector.py
# SCRIPT PARA ANÁLISE PROFUNDA DA GERAÇÃO 0
import logging
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from collections import Counter

# --- Importações do seu projeto ---
try:
    import data_handling
    import ga
    from individual import Individual
    from utils import calculate_population_diversity
    import metrics
    import fitness
    from constants import RANDOM_SEED
except ImportError as e:
    print(f"Erro de importação: {e}. Certifique-se de que os arquivos do projeto estão acessíveis.")
    exit()

logging.root.handlers = []

# 2. Configurar o logging do zero com o nível e formato desejados.
#    Direcionamos a saída para 'sys.stdout' para garantir que apareça na célula do notebook.
logging.basicConfig(
    level=logging.INFO,  # Defina o nível desejado: DEBUG, INFO, WARNING, etc.
    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)

# 3. Teste para confirmar que a configuração foi aplicada.
logging.info("O sistema de logging foi reconfigurado com sucesso para o Colab!")
logging.warning("Este é um aviso de teste.")

# --- Configuração ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# logger = logging.getLogger("PopulationInspector")

# --- PARÂMETROS DE TESTE ---
CONFIG_FILE_PATH = 'config.yaml'
TARGET_EXPERIMENT_ID = 'CovType'

if __name__ == "__main__":
    print("\n" + "="*80)
    logging.info("INICIANDO SCRIPT DE INSPEÇÃO DA POPULAÇÃO INICIAL")
    print("="*80)

    # --- FASE 1: SETUP ---
    logging.info("--- FASE 1: Carregando configuração e dados ---")
    config = data_handling.load_full_config(CONFIG_FILE_PATH)
    if not config: exit()

    ga_params = config['ga_params']
    fitness_params = config['fitness_params']
    chunk_size = config['data_params']['chunk_size']

    chunks = data_handling.generate_dataset_chunks(
        TARGET_EXPERIMENT_ID, chunk_size, 2, chunk_size * 2, CONFIG_FILE_PATH
    )
    train_data, train_target = chunks[0]
    all_classes = sorted(list(np.unique(train_target)))
    attributes = sorted(list(train_data[0].keys()))
    categorical_features = {a for a in attributes if isinstance(train_data[0].get(a), str)}
    value_ranges = {a: (min(d[a] for d in train_data), max(d[a] for d in train_data)) for a in (set(attributes) - categorical_features) if train_data}
    category_values = {a: {d[a] for d in train_data} for a in categorical_features}

    logging.info(f"Dados do chunk 0 carregados para '{TARGET_EXPERIMENT_ID}'.")

    # --- FASE 2: GERAÇÃO DA POPULAÇÃO ---
    logging.info("\n" + "--- FASE 2: Gerando população inicial com a estratégia 'default' ---")
    initial_population = ga.initialize_population(
        population_size=ga_params['population_size'],
        max_rules_per_class=ga_params['max_rules_per_class'],
        max_depth=ga_params['initial_max_depth'],
        attributes=attributes, value_ranges=value_ranges, category_values=category_values,
        categorical_features=categorical_features, classes=all_classes,
        train_data=train_data, train_target=train_target,
        # Passando todos os coeficientes para consistência
        regularization_coefficient=fitness_params['initial_regularization_coefficient'],
        feature_penalty_coefficient=fitness_params['feature_penalty_coefficient'],
        operator_penalty_coefficient=fitness_params.get('operator_penalty_coefficient', 0.0),
        threshold_penalty_coefficient=fitness_params.get('threshold_penalty_coefficient', 0.0),
        intelligent_mutation_rate=ga_params.get('intelligent_mutation_rate', 0.2),
        initialization_strategy='default', performance_label='medium',
        enable_dt_seeding_on_init_config=ga_params.get('enable_dt_seeding_on_init', True),
        dt_seeding_ratio_on_init_config=ga_params.get('dt_seeding_ratio_on_init', 0.2)
    )

    # --- FASE 3: ANÁLISE DETALHADA ---
    logging.info("\n" + "--- FASE 3: Analisando a composição e qualidade da Geração 0 ---")

    population_stats = []
    for i, ind in enumerate(initial_population):
        # Calcula a cobertura de classes (quantas classes têm pelo menos 1 regra)
        covered_classes_count = len([c for c, r_list in ind.rules.items() if r_list])

        # Calcula o G-mean inicial
        predictions = [ind._predict(inst) for inst in train_data]
        g_mean = metrics.calculate_gmean_contextual(train_target, predictions, all_classes)
        ind.gmean = g_mean # Salva para referência

        population_stats.append({
            'individual': i,
            'total_rules': ind.count_total_rules(),
            'class_coverage': covered_classes_count,
            'total_nodes': ind.count_total_nodes(),
            'features_used': len(ind.get_used_attributes()),
            'g_mean': g_mean
        })

    df_stats = pd.DataFrame(population_stats)

    print("\n--- RESUMO ESTATÍSTICO DA POPULAÇÃO INICIAL ---")
    print(df_stats.describe())

    # --- Visualização da Cobertura de Classes ---
    # plt.figure(figsize=(10, 6))
    # coverage_counts = df_stats['class_coverage'].value_counts().sort_index()
    # coverage_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    # plt.title('Distribuição da Cobertura de Classes na População Inicial', fontsize=16)
    # plt.xlabel('Número de Classes com Regras', fontsize=12)
    # plt.ylabel('Número de Indivíduos', fontsize=12)
    # plt.xticks(rotation=0)
    # plt.grid(axis='y', linestyle='--')
    # plt.show()

    # --- Análise do Melhor Indivíduo ---
    print("\n--- ANÁLISE DO MELHOR INDIVÍDUO DA GERAÇÃO 0 (POR G-MEAN) ---")
    if not df_stats.empty:
        best_initial_idx = df_stats['g_mean'].idxmax()
        best_individual = initial_population[best_initial_idx]

        print(f"Melhor Indivíduo: #{best_initial_idx} com G-Mean: {df_stats.loc[best_initial_idx, 'g_mean']:.4f}")
        print(f"  - Total de Regras: {best_individual.count_total_rules()}")
        print(f"  - Cobertura de Classes: {df_stats.loc[best_initial_idx, 'class_coverage']} / {len(all_classes)}")
        print("\nRegras do Melhor Indivíduo:")
        print(best_individual.get_rules_as_string())

        best_predictions = [best_individual._predict(inst) for inst in train_data]
        print("\nRelatório de Classificação:")
        print(classification_report(train_target, best_predictions, labels=all_classes, zero_division=0, digits=4))
    else:
        print("Nenhuma estatística pôde ser gerada.")

    print("\n" + "="*80)
    logging.info("INSPEÇÃO CONCLUÍDA.")
    print("="*80)