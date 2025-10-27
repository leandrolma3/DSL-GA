# debug_run_analyzer.py (v2 - com RF e XGBoost)

import os
import pickle
import logging
import argparse
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Any, Tuple

# --- Importações de Machine Learning ---
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier # <<< ADICIONADO
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, recall_score

# Tenta importar XGBoost, mas não quebra se não estiver instalado
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# --- Importações do Projeto GBML ---
try:
    from individual import Individual
    from rule_tree import RuleTree
    import metrics
except ImportError as e:
    print(f"Erro de importação: {e}. Certifique-se que o script está na pasta raiz do projeto.")
    exit()

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("RunAnalyzer")


# ==============================================================================
# --- FUNÇÕES AUXILIARES DE ANÁLISE (com adição de _display_feature_importances) ---
# ==============================================================================

def _format_confusion_matrix(y_true: List, y_pred: List, labels: List) -> str:
    # (sem alterações)
    try:
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        df_cm = pd.DataFrame(matrix, index=labels, columns=labels)
        df_cm.index.name = 'True Label'
        df_cm.columns.name = 'Predicted Label'
        return df_cm.to_string()
    except Exception as e:
        return f"Could not generate confusion matrix: {e}"

def _analyze_performance(model: Any, X_data: List[Dict], y_data: List, all_classes: List, model_type: str) -> Tuple[float, float]:
    """Executa e exibe uma análise de performance, retornando G-mean e F1-score."""
    # (sem alterações na lógica interna, apenas adicionado um retorno)
    if not X_data:
        logger.warning("Conjunto de dados vazio. Análise de performance pulada.")
        return 0.0, 0.0

    X_df = pd.DataFrame(X_data)
    
    if model_type == 'gbml':
        predictions = [model._predict(inst) for inst in X_data]
    elif model_type in ['dt', 'rf', 'xgb']:
        predictions = model.predict(X_df)
    else:
        logger.error(f"Tipo de modelo desconhecido: {model_type}")
        return 0.0, 0.0

    gmean_official = metrics.calculate_gmean_contextual(y_data, predictions, all_classes)
    f1_weighted = f1_score(y_data, predictions, average='weighted', zero_division=0)
    
    print(f"\nMétricas Globais:")
    print(f"  - G-mean: {gmean_official:.4f}")
    print(f"  - F1-Score Ponderado: {f1_weighted:.4f}")
    print(f"  - Acurácia: {accuracy_score(y_data, predictions):.4f}")

    print("\nMatriz de Confusão:")
    print(_format_confusion_matrix(y_data, predictions, all_classes))

    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_data, predictions, labels=all_classes, zero_division=0, digits=4))
    
    return gmean_official, f1_weighted


def _analyze_gbml_rule_activation(individual: Individual, X_data: List[Dict], y_data: List) -> None:
    # (sem alterações)
    rule_stats = {}
    for class_label, rule_list in individual.rules.items():
        for i, rule in enumerate(rule_list):
            rule_id = f"Classe {class_label}, Regra {i}"
            rule_stats[rule_id] = {'total': 0, 'correct': 0, 'text': rule.to_string()}

    default_class_activations = {'total': 0, 'correct': 0}

    for i, instance in enumerate(X_data):
        true_label = y_data[i]
        activated_rules_info = []
        for class_label, rule_list in individual.rules.items():
            for rule_idx, rule in enumerate(rule_list):
                if rule.evaluate(instance):
                    specificity = rule.count_nodes()
                    rule_id = f"Classe {class_label}, Regra {rule_idx}"
                    activated_rules_info.append((class_label, specificity, rule_id))
                    break 

        if activated_rules_info:
            best_rule = max(activated_rules_info, key=lambda item: item[1])
            prediction, _, winning_rule_id = best_rule
            
            rule_stats[winning_rule_id]['total'] += 1
            if prediction == true_label:
                rule_stats[winning_rule_id]['correct'] += 1
        else:
            prediction = individual.default_class
            default_class_activations['total'] += 1
            if prediction == true_label:
                default_class_activations['correct'] += 1

    print("\nTabela de Ativação e Qualidade das Regras:")
    df_data = []
    for rule_id, stats in rule_stats.items():
        precision = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        df_data.append([rule_id, stats['total'], stats['correct'], f"{precision:.2f}%", stats['text']])
    
    df = pd.DataFrame(df_data, columns=['ID da Regra', 'Ativações Totais', 'Ativações Corretas', 'Precisão', 'Regra'])
    print(df.to_string())
    print("\nAtivação da Classe Padrão:")
    default_prec = (default_class_activations['correct'] / default_class_activations['total'] * 100) if default_class_activations['total'] > 0 else 0
    print(f"  - Ativada {default_class_activations['total']} vezes com {default_prec:.2f}% de precisão.")


def _display_feature_importances(model: Any, feature_names: List[str], top_n: int = 15) -> None:
    """Exibe as features mais importantes para modelos de ensemble."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(top_n)
        print(f"\nTop {top_n} Features Mais Importantes:")
        print(feature_importance_df.to_string(index=False))
    else:
        print("\nO modelo não possui o atributo 'feature_importances_'.")


# ==============================================================================
# --- SCRIPT PRINCIPAL (MODIFICADO) ---
# ==============================================================================

def main(run_directory: str):
    # (Lógica de busca de arquivos inalterada)
    chunk_data_dir = os.path.join(run_directory, "chunk_data")
    if not os.path.isdir(chunk_data_dir):
        logger.error(f"Diretório 'chunk_data' não encontrado em: {run_directory}")
        return
    individual_files = [f for f in os.listdir(chunk_data_dir) if f.startswith('best_individual') and f.endswith('.pkl')]
    individual_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if not individual_files:
        logger.error(f"Nenhum arquivo de indivíduo (.pkl) encontrado em {chunk_data_dir}")
        return

    for individual_file in individual_files:
        chunk_idx = int(individual_file.split('_')[-1].split('.')[0])
        
        print("\n" + "="*80)
        logger.info(f"Analisando Transição: Treino no Chunk {chunk_idx} -> Teste no Chunk {chunk_idx + 1}")
        print("="*80)

        # --- Etapa 1: Carregamento dos Ativos ---
        # (Lógica de carregamento inalterada)
        try:
            logger.info("Etapa 1: Carregando Ativos...")
            with open(os.path.join(chunk_data_dir, individual_file), 'rb') as f:
                gbml_individual = pickle.load(f)
            if chunk_idx == 0:
                train_csv_path = os.path.join(chunk_data_dir, 'chunk_0_train.csv')
            else:
                train_csv_path = os.path.join(chunk_data_dir, f'chunk_{chunk_idx}_test.csv')
            test_csv_path = os.path.join(chunk_data_dir, f'chunk_{chunk_idx + 1}_test.csv')
            df_train = pd.read_csv(train_csv_path)
            df_test = pd.read_csv(test_csv_path)
            y_train = df_train.pop('class').tolist()
            X_train = df_train.to_dict('records')
            y_test = df_test.pop('class').tolist()
            X_test = df_test.to_dict('records')
            all_classes = sorted(list(np.unique(y_train + y_test)))
            logger.info("Ativos carregados com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar ativos para a transição {chunk_idx}: {e}")
            continue

        # --- Etapa 2: Análise Exaustiva do Indivíduo GBML ---
        # (Seção inalterada)
        print("\n\n" + "#"*80)
        logger.info(f"Etapa 2: Análise Exaustiva do Indivíduo GBML (Treinado no Chunk {chunk_idx})")
        print("#"*80)
        # ... (código de apresentação, performance em treino/teste e ativação de regras como antes)
        print("\n\n--- 2.3. Performance no CHUNK DE TREINO (Chunk {chunk_idx}) ---")
        gbml_gmean_test, gbml_f1_test = _analyze_performance(gbml_individual, X_train, y_train, all_classes, 'gbml')

        print("\n\n--- 2.3. Performance no CHUNK DE TESTE (Chunk {chunk_idx+1}) ---")
        gbml_gmean_test, gbml_f1_test = _analyze_performance(gbml_individual, X_test, y_test, all_classes, 'gbml')

        # --- Etapa 3: Análise de Modelos de Baseline (MODIFICADA) ---
        print("\n\n" + "#"*80)
        logger.info(f"Etapa 3: Análise de Modelos de Baseline (Treinados no Chunk {chunk_idx})")
        print("#"*80)
        
        # 3.1 Treinamento dos modelos
        logger.info("--- 3.1. Treinando modelos de baseline... ---")
        dt_model = DecisionTreeClassifier(max_depth=7, min_samples_leaf=10, random_state=42, class_weight='balanced')
        dt_model.fit(df_train, y_train)
        
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=42, class_weight='balanced', n_jobs=-1)
        rf_model.fit(df_train, y_train)
        
        if XGB_AVAILABLE:
            print("\n\n--- 3.4. Análise do XGBoost (XGB) ---")
            
            # --- INÍCIO DA CORREÇÃO ---
            logger.info("Ajustando rótulos para o XGBoost (de 1-7 para 0-6)...")
            # Cria novas listas de rótulos apenas para o XGBoost
            y_train_xgb = [label - 1 for label in y_train]
            y_test_xgb = [label - 1 for label in y_test]
            # Cria a lista de todas as classes no formato esperado pelo XGBoost
            all_classes_xgb = [c - 1 for c in all_classes]

            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss', random_state=42)
            xgb_model.fit(df_train, y_train_xgb)
        logger.info("Treinamento concluído.")

        # 3.2 Análise da Árvore de Decisão
        print("\n\n--- 3.2. Análise da Árvore de Decisão (DT) ---")
        print("Regras (export_text):")
        #print(export_text(dt_model, feature_names=list(df_train.columns)))
        print("\nPerformance da DT no CHUNK DE TREINO:")
        dt_gmean_test, dt_f1_test = _analyze_performance(dt_model, X_train, y_train, all_classes, 'dt')

        print("\nPerformance da DT no CHUNK DE TESTE:")
        dt_gmean_test, dt_f1_test = _analyze_performance(dt_model, X_test, y_test, all_classes, 'dt')
        
        # 3.3 Análise do Random Forest
        print("\n\n--- 3.3. Análise do Random Forest (RF) ---")
        _display_feature_importances(rf_model, list(df_train.columns))
        print("\nPerformance do RF no CHUNK DE TREINO:")
        rf_gmean_test, rf_f1_test = _analyze_performance(rf_model, X_train, y_train, all_classes, 'rf')

        print("\nPerformance do RF no CHUNK DE TESTE:")
        rf_gmean_test, rf_f1_test = _analyze_performance(rf_model, X_test, y_test, all_classes, 'rf')

        # 3.4 Análise do XGBoost
        if XGB_AVAILABLE:

            print("\n\n--- 3.4. Análise do XGBoost (XGB) ---")
            _display_feature_importances(xgb_model, list(df_train.columns))
            print("\nPerformance do XGB no CHUNK DE TREINO:")
            xgb_gmean_test, xgb_f1_test = _analyze_performance(xgb_model, X_train, y_train_xgb, all_classes_xgb, 'xgb')

            print("\nPerformance do XGB no CHUNK DE TESTE:")
            xgb_gmean_test, xgb_f1_test = _analyze_performance(xgb_model, X_test, y_test_xgb, all_classes_xgb, 'xgb')
        else:
            logger.warning("XGBoost não está instalado. Análise pulada.")
            xgb_gmean_test, xgb_f1_test = 0.0, 0.0

        # --- Etapa 4: Relatório Final Comparativo (MODIFICADA) ---
        print("\n\n" + "#"*80)
        logger.info(f"Etapa 4: Relatório Final Comparativo da Transição {chunk_idx} -> {chunk_idx+1}")
        print("#"*80)

        model_names = ['Indivíduo GBML', 'Árvore de Decisão', 'Random Forest']
        gmean_scores = [gbml_gmean_test, dt_gmean_test, rf_gmean_test]
        f1_scores = [gbml_f1_test, dt_f1_test, rf_f1_test]

        if XGB_AVAILABLE:
            model_names.append('XGBoost')
            gmean_scores.append(xgb_gmean_test)
            f1_scores.append(xgb_f1_test)

        winner_idx = np.argmax(gmean_scores)
        winners = [''] * len(model_names)
        winners[winner_idx] = '<-- VENCEDOR'

        df_comp_data = {
            'Modelo': model_names,
            'G-mean (Teste)': [f"{s:.4f}" for s in gmean_scores],
            'F1-Score (Teste)': [f"{s:.4f}" for s in f1_scores],
            'Vencedor (G-mean)': winners
        }
        df_comp = pd.DataFrame(df_comp_data)
        print(df_comp.to_string(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analisador de Transição de Chunks para o GBML, com benchmarks de RF e XGBoost.")
    parser.add_argument("run_directory", type=str, help="Caminho para o diretório de uma execução específica (ex: 'results/CovType/run_1').")
    args = parser.parse_args()
    main(args.run_directory)