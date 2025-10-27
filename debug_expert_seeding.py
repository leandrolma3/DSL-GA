# debug_expert_seeding.py (v10 - Estratégia de "Clonagem" de Decision Tree)

import os
import logging
import argparse
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Any

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

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("DecisionTreeCloningDebugger")


# ==============================================================================
# --- FUNÇÕES AUXILIARES ---
# ==============================================================================

def _load_and_prepare_data(csv_path: str) -> Dict[str, Any]:
    """Carrega um chunk de um arquivo CSV e prepara todos os ativos de dados necessários."""
    logger.info(f"Carregando e preparando dados de: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.loc[:,~df.columns.duplicated()]
    y_data = df.pop('class').tolist()
    X_data_dict = df.to_dict('records')

    all_classes = sorted(list(np.unique(y_data)))
    attributes = list(df.columns)
    
    logger.info(f"Dados carregados: {len(y_data)} instâncias, {len(all_classes)} classes.")
    
    # Prepara o dicionário de informações base para a criação de RuleTree e Individual
    base_attributes_info = {
        "attributes": attributes,
        "value_ranges": {a: (df[a].min(), df[a].max()) for a in attributes if df[a].dtype in ['int64', 'float64']},
        "category_values": {a: set(df[a].unique()) for a in attributes if df[a].dtype == 'object'},
        "categorical_features": {a for a in attributes if df[a].dtype == 'object'}
    }
    
    return {
        "X_dict": X_data_dict, "y": y_data, "df": df,
        "all_classes": all_classes, "base_attributes_info": base_attributes_info
    }

def _evaluate_and_report(model: Any, X_df: pd.DataFrame, y_true: List, all_classes: List, model_type: str) -> float:
    """Gera predições, calcula o G-mean e imprime um relatório de performance."""
    if model_type == 'gbml':
        predictions = [model._predict(inst) for inst in X_df.to_dict('records')]
    else: # Decision Tree
        predictions = model.predict(X_df)
            
    gmean = metrics.calculate_gmean_contextual(y_true, predictions, all_classes)
    
    print(f"\nMétricas Globais:")
    print(f"  - G-mean: {gmean:.4f}")
    
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, predictions, labels=all_classes, zero_division=0, digits=4))
    return gmean

def _extract_all_rules_from_dt(dt_model: DecisionTreeClassifier, feature_names: List[str], base_attributes_info: Dict) -> Dict[int, List[RuleTree]]:
    """Percorre uma Decision Tree e extrai TODOS os caminhos (regras)."""
    tree = dt_model.tree_
    gene_pool = {c: [] for c in dt_model.classes_}

    def recurse(node_id, conditions):
        # Se for uma folha, constrói a regra e a adiciona ao gene_pool
        if tree.children_left[node_id] == -1:
            predicted_class_idx = np.argmax(tree.value[node_id][0])
            predicted_class = dt_model.classes_[predicted_class_idx]

            # Ignora regras vazias (caso a árvore seja apenas um nó raiz)
            if not conditions: return

            # Constrói a árvore de regras a partir das condições acumuladas
            if len(conditions) == 1:
                rule_root = conditions[0]
            else:
                rule_root = Node(operator="AND", left=conditions[0], right=conditions[1])
                for i in range(2, len(conditions)):
                    rule_root = Node(operator="AND", left=rule_root, right=conditions[i])
            
            rule_tree = RuleTree(max_depth=len(conditions), **base_attributes_info, root_node=rule_root)
            if rule_tree.is_valid_rule():
                gene_pool[predicted_class].append(rule_tree)
            return

        # Se for um nó interno, continua a recursão para os filhos
        feature_idx = tree.feature[node_id]
        feature = feature_names[feature_idx]
        threshold = tree.threshold[node_id]
        
        # Ramo da Esquerda (condição <= threshold)
        left_cond = Node(attribute=feature, operator="<=", value=threshold, feature_type='numeric')
        recurse(tree.children_left[node_id], conditions + [left_cond])
        
        # Ramo da Direita (condição > threshold)
        right_cond = Node(attribute=feature, operator=">", value=threshold, feature_type='numeric')
        recurse(tree.children_right[node_id], conditions + [right_cond])

    recurse(0, [])
    return gene_pool

# ==============================================================================
# --- SCRIPT PRINCIPAL ---
# ==============================================================================

def main(chunk_path: str):
    # --- Etapa 1: Preparação e Carregamento de Dados ---
    logger.info("--- Etapa 1: Preparação e Carregamento de Dados ---")
    data_assets = _load_and_prepare_data(chunk_path)
    X_train_dict, y_train, df_train, all_classes, base_attributes_info = \
        [data_assets[k] for k in ["X_dict", "y", "df", "all_classes", "base_attributes_info"]]

    # --- Etapa 2: Treinamento e Avaliação do Modelo de Referência (Decision Tree) ---
    logger.info("\n--- Etapa 2: Treinamento e Avaliação do Modelo de Referência (Decision Tree) ---")
    
    dt_max_depth = 10 
    dt_model = DecisionTreeClassifier(
        max_depth=dt_max_depth,
        min_samples_leaf=5, 
        random_state=42, 
        class_weight='balanced'
    )
    dt_model.fit(df_train, y_train)
    
    logger.info("Performance da Árvore de Decisão Original (no treino):")
    dt_gmean = _evaluate_and_report(dt_model, df_train, y_train, all_classes, 'dt')

    # --- Etapa 3: "Clonagem" da Árvore para o Indivíduo GBML ---
    logger.info("\n--- Etapa 3: Clonagem da Árvore para o Indivíduo GBML ---")
    
    gene_pool = _extract_all_rules_from_dt(dt_model, list(df_train.columns), base_attributes_info)
    total_rules_extracted = sum(len(rules) for rules in gene_pool.values())
    logger.info(f"Total de {total_rules_extracted} regras extraídas da Decision Tree.")

    cloned_individual = Individual(
        max_rules_per_class=total_rules_extracted, # Garante espaço para todas as regras
        max_depth=dt_max_depth,
        **base_attributes_info, 
        classes=all_classes,
        train_target=y_train,
        initialize_random_rules=False # Para definir a mesma default_class
    )
    cloned_individual.rules = gene_pool # Atribui diretamente o gene pool completo
    logger.info("Indivíduo GBML clonado a partir da árvore.")

    # --- Etapa 4: Validação da Clonagem ---
    logger.info("\n--- Etapa 4: Validação da Clonagem ---")
    logger.info("Avaliando a performance do Indivíduo GBML clonado (deve ser idêntica à original)...")
    gbml_gmean = _evaluate_and_report(cloned_individual, df_train, y_train, all_classes, 'gbml')

    # --- Etapa 5: Relatório Comparativo Final ---
    print("\n\n" + "#"*80)
    logger.info("Etapa 5: Relatório Comparativo Final da Clonagem")
    print("#"*80)

    df_comp_data = {
        'Modelo': [
            '1. Árvore de Decisão (Original)', 
            '2. Indivíduo GBML (Clonado)'
        ],
        'G-mean (no Treino)': [
            f"{dt_gmean:.4f}", 
            f"{gbml_gmean:.4f}"
        ],
        'Sucesso na Clonagem?': [
            '---',
            'SIM ✅' if np.isclose(dt_gmean, gbml_gmean) else 'NÃO ❌'
        ]
    }
    df_comp = pd.DataFrame(df_comp_data)
    print(df_comp.to_string(index=False))
    
    if not np.isclose(dt_gmean, gbml_gmean):
        logger.warning("Os G-means não são idênticos. O bug está na lógica de predição/desempate do 'Individual' ou na extração das regras.")
    else:
        logger.info("Sucesso! O indivíduo clonado replica perfeitamente o comportamento da Árvore de Decisão. A estratégia é viável.")
    print("#"*80)

    # # --- Etapa 6: Análise Forense de Instância Única ---
    # logger.info("\n\n" + "#"*80)
    # logger.info("Etapa 6: Análise Forense de Instância Única")
    # logger.info("Rastreando uma instância para encontrar a divergência...")
    # print("#"*80)

    # # 6.1. Seleciona uma instância que a DT original acerta
    # dt_predictions = dt_model.predict(df_train)
    # correct_indices = [i for i, (pred, true) in enumerate(zip(dt_predictions, y_train)) if pred == true]
    
    # if not correct_indices:
    #     logger.warning("A DT original não acertou nenhuma predição. Análise forense não é possível.")
    #     return
        
    # test_idx = correct_indices[0] # Pega a primeira instância que a DT acertou
    # test_instance = X_train_dict[test_idx]
    # true_label = y_train[test_idx]
    
    # logger.info(f"Instância de teste selecionada (índice: {test_idx}), classe verdadeira: {true_label}")
    
    # # 6.2. Rastreia a instância na DT original
    # leaf_id = dt_model.apply(df_train.iloc[[test_idx]])[0]
    # logger.info(f"A DT original classificou esta instância corretamente, terminando na folha de ID: {leaf_id}")
    
    # # 6.3. Executa a análise forense no Indivíduo Clonado
    # logger.info("\n--- Análise Forense no Indivíduo Clonado ---")
    
    # # Itera sobre as regras da classe que DEVERIA ser predita
    # logger.info(f"Verificando as regras clonadas para a classe correta ({true_label})...")
    # rules_to_check = cloned_individual.rules.get(true_label, [])
    # if not rules_to_check:
    #     logger.warning("Nenhuma regra foi clonada para a classe correta!")
    
    # found_firing_rule = False
    # for i, rule in enumerate(rules_to_check):
    #     print(f"\n--- Checando Regra {i} da Classe {true_label} ---")
    #     # Usa o novo método verboso
    #     if rule.evaluate_with_debug(test_instance):
    #         found_firing_rule = True
    #         logger.info(f"!!! SUCESSO: A Regra {i} da Classe {true_label} foi ativada corretamente. !!!")

    # if not found_firing_rule:
    #     logger.error(f"!!! FALHA: Nenhuma das regras clonadas para a classe correta ({true_label}) foi ativada. O problema está na avaliação da regra. !!!")
        
    # logger.info("\n--- Verificando a Predição Final do Indivíduo Clonado ---")
    # final_prediction = cloned_individual._predict(test_instance)
    # logger.info(f"A predição final do Indivíduo Clonado para a instância foi: {final_prediction}")

    # if final_prediction != true_label:
    #     logger.error(f"DIVERGÊNCIA CONFIRMADA: A DT previu {true_label}, mas o clone previu {final_prediction}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para validar a clonagem de uma Decision Tree para um Indivíduo GBML.")
    parser.add_argument("chunk_path", type=str, help="Caminho para o arquivo CSV de um chunk de treino.")
    args = parser.parse_args()
    main(args.chunk_path)