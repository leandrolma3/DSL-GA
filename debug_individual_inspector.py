import pickle
import logging
import pandas as pd
import numpy as np
# NOVO CÓDIGO CORRIGIDO
from sklearn.metrics import classification_report, confusion_matrix
import os

# --- Importações do seu projeto ---
try:
    import data_handling
    import ga # Importa o módulo ga para acessar a função de avaliação
    from individual import Individual
    from rule_tree import RuleTree
    from metrics import calculate_gmean_contextual
except ImportError as e:
    print(f"Erro de importação: {e}. Verifique o PYTHONPATH.")
    exit()

from collections import Counter # Garanta que esta importação esteja presente


def inspect_performance(individual: Individual, data: list, target: list, all_classes: list, dataset_name: str):
    """
    Executa uma análise completa de um indivíduo em um conjunto de dados e imprime um relatório.
    """
    logger.info(f"--- INSPECIONANDO PERFORMANCE NO CONJUNTO DE DADOS: {dataset_name} ---")
    
    if not data:
        logger.warning("Conjunto de dados está vazio. Análise pulada.")
        return

    # 1. Realiza as predições e verifica a ativação das regras
    predictions = []
    rule_activations = 0
    for inst in data:
        prediction, activated = individual._predict_with_activation_check(inst)
        predictions.append(prediction)
        if activated:
            rule_activations += 1
    
    # 2. Calcula e exibe as métricas
    g_mean = calculate_gmean_contextual(target, predictions, all_classes)
    activation_rate = (rule_activations / len(data)) * 100
    
    print(f"\nResultados para o conjunto '{dataset_name}':")
    print("-" * 40)
    print(f"  - G-Mean Calculado: {g_mean:.4f}")
    print(f"  - Taxa de Ativação de Regras: {activation_rate:.2f}% ({rule_activations}/{len(data)} instâncias)")
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(target, predictions, labels=all_classes, zero_division=0, digits=4))
    
    print("Matriz de Confusão (Linhas=Verdadeiro, Colunas=Predito):")
    print(confusion_matrix(target, predictions, labels=all_classes))
    print("-" * 40 + "\n")


def _calculate_class_weights(target_chunk: list, all_classes: list) -> dict:
    """
    Calcula pesos para cada classe com base no inverso de sua frequência.
    """
    if not target_chunk:
        return {c: 1.0 for c in all_classes}
    n_samples = len(target_chunk)
    counts = Counter(target_chunk)
    raw_weights = {}
    for c in all_classes:
        count = counts.get(c, 0)
        raw_weights[c] = n_samples / (count + 1)
    sum_of_raw_weights = sum(raw_weights.values())
    num_classes = len(all_classes)
    normalized_weights = {}
    if sum_of_raw_weights > 0:
        for c, rw in raw_weights.items():
            normalized_weights[c] = (rw / sum_of_raw_weights) * num_classes
    else:
        return {c: 1.0 for c in all_classes}
    return normalized_weights


# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("GmeanComparator")

# --- PARÂMETROS ---
RUN_DIRECTORY = r'C:\Users\LeandroAlmeida\Downloads\DSL-AG-hybrid\real_and_stationary_experiment_results\chunk_data' # Ajuste este caminho
CONFIG_FILE_PATH = 'config.yaml'

if __name__ == "__main__":
    logger.info("--- INICIANDO SCRIPT DE COMPARAÇÃO DE CÁLCULO DE G-MEAN ---")
    
    # 1. Carregar o indivíduo campeão salvo
    pickle_path = os.path.join(RUN_DIRECTORY, 'best_individuals.pkl')
    try:
        with open(pickle_path, 'rb') as f:
            best_individual = pickle.load(f)[-1]
        logger.info(f"Indivíduo campeão carregado de {pickle_path}")
    except Exception as e:
        logger.error(f"Erro ao carregar {pickle_path}: {e}"); exit()

    # 2. Carregar o chunk de dados de treino exato
    train_csv_path = os.path.join(RUN_DIRECTORY, 'chunk_0_train.csv')
    try:
        df_train = pd.read_csv(train_csv_path)
        train_target = df_train.pop('class').tolist()
        train_data = df_train.to_dict('records')
        all_classes = sorted(list(np.unique(train_target)))
        logger.info(f"Chunk de treino carregado de {train_csv_path}")
    except Exception as e:
        logger.error(f"Erro ao carregar {train_csv_path}: {e}"); exit()

    TEST_CHUNK_CSV = os.path.join(RUN_DIRECTORY, 'chunk_1_test.csv')
    # 3. Carregar o chunk de dados de TESTE
    try:
        df_test = pd.read_csv(TEST_CHUNK_CSV)
        test_target = df_test.pop('class').tolist()
        test_data = df_test.to_dict('records')
        logger.info(f"Chunk de TESTE carregado de {TEST_CHUNK_CSV}")
    except Exception as e:
        logger.error(f"Erro ao carregar dados de teste: {e}"); exit()

    # 4. Executar a inspeção em ambos os conjuntos de dados
    inspect_performance(best_individual, train_data, train_target, all_classes, "Treino (chunk 0)")
    inspect_performance(best_individual, test_data, test_target, all_classes, "Teste (chunk 1)")
    
    logger.info("--- INSPEÇÃO CONCLUÍDA ---")

    # 3. Replicar as condições da avaliação do AG
    logger.info("Replicando os argumentos de avaliação do algoritmo genético...")
    config = data_handling.load_full_config(CONFIG_FILE_PATH)
    ga_p = config['ga_params']
    fit_p = config['fitness_params']
    class_weights = _calculate_class_weights(train_target, all_classes)
    constant_args_for_worker = {
        'train_data': train_data, 
        'train_target': train_target,
        'classes': all_classes,
        'class_weights': class_weights,
        'regularization_coefficient': fit_p.get('initial_regularization_coefficient', 0.0),
        'feature_penalty_coefficient': fit_p.get('feature_penalty_coefficient', 0.0),
        'class_coverage_coefficient': fit_p.get('class_coverage_coefficient', 0.0),
        'gmean_bonus_coefficient': fit_p.get('gmean_bonus_coefficient', 0.0),
        'reference_features': set(), 'beta': 0.0, 'previous_used_features': None,
        'gamma': 0.0, 'operator_penalty_coefficient': 0.0, 'threshold_penalty_coefficient': 0.0,
        'previous_operator_info': None, 'operator_change_coefficient': 0.0,
        'attributes': list(df_train.columns), 'categorical_features': set(),
        'reduce_change_penalties_flag': False
    }

    # --- O TESTE DE COMPARAÇÃO ---
    print("\n" + "="*80)
    logger.info("EXECUTANDO COMPARAÇÃO DIRETA DOS MÉTODOS DE CÁLCULO")
    print("="*80)

    # MÉTODO 1: Usando a função de avaliação do worker do AG
    logger.info("Calculando G-mean usando a função do worker 'evaluate_individual_fitness_parallel'...")
    worker_args = (best_individual, constant_args_for_worker)
    _, gmean_from_ga_worker, _ = ga.evaluate_individual_fitness_parallel(worker_args)
    print(f"  -> G-Mean (cálculo do AG 'ao vivo'): {gmean_from_ga_worker:.4f}")

    # MÉTODO 2: Usando o cálculo de controle direto (como no inspetor anterior)
    logger.info("Calculando G-mean usando o método de controle direto...")
    predictions_control = [best_individual._predict(inst) for inst in train_data]
    gmean_control = calculate_gmean_contextual(train_target, predictions_control, all_classes)
    print(f"  -> G-Mean (cálculo do Inspetor 'controle'): {gmean_control:.4f}")
    
    print("="*80)

    # --- Análise e Diagnóstico Final ---
    if np.isclose(gmean_from_ga_worker, gmean_control) and gmean_control > 0:
        logger.info("DIAGNÓSTICO: SUCESSO! O cálculo dentro do AG está funcionando. O problema anterior foi resolvido.")
        logger.info("O próximo passo é rodar o main.py completo e observar a evolução do G-mean nos logs de geração.")
    elif not np.isclose(gmean_from_ga_worker, gmean_control):
        logger.error("DIAGNÓSTICO: FALHA! Encontramos o bug. O cálculo de G-mean dentro do worker do AG está produzindo um resultado diferente do cálculo de controle.")
        logger.error("A causa provável está na função 'evaluate_individual_fitness_parallel' em 'ga.py' ou nos argumentos que ela está recebendo.")
    else: # Ambos são zero
        logger.warning("DIAGNÓSTICO: Ambos os cálculos resultaram em zero. Isso significa que o indivíduo campeão, na verdade, não tem um G-mean positivo no dataset de treino.")
        logger.warning("Isso aponta para um problema de estagnação mais profundo ou um desbalanço extremo nos dados do chunk que impede o G-mean positivo.")