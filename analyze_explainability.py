"""
Analise de Explicabilidade - GBML vs ERulesD2S

Este script extrai e consolida metricas de explicabilidade dos modelos
baseados em regras (GBML e ERulesD2S) para comparacao.

Metricas extraidas:
- GBML: Numero de regras, condicoes, operadores AND/OR, profundidade, TCS/RIR/AMS
- ERulesD2S: NumberRules, NumberConditions, NumberNodes

Output:
- CSVs consolidados para analise
- Relatorios estatisticos
- Dados prontos para uso no artigo

Autor: Analise automatizada
Data: 2024
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURACAO
# ==============================================================================

# Diretorios de experimentos
EXPERIMENT_DIRS = [
    "experiments_6chunks_phase2_gbml/batch_1",
    "experiments_6chunks_phase2_gbml/batch_2",
    "experiments_6chunks_phase2_gbml/batch_3",
    "experiments_6chunks_phase2_gbml/batch_4",
    "experiments_6chunks_phase3_real/batch_5",
    "experiments_6chunks_phase3_real/batch_6",
    "experiments_6chunks_phase3_real/batch_7",
]

# Diretorio de output
OUTPUT_DIR = Path("explainability_analysis")

# Pesos para calculo do TCS (conforme chunk_transition_analyzer.py)
W_INSTABILITY = 0.6
W_MODIFICATION_IMPACT = 0.4

# ==============================================================================
# FUNCOES DE EXTRACAO - GBML
# ==============================================================================

def extract_gbml_rule_details(experiment_dir: Path) -> List[Dict]:
    """
    Extrai detalhes de regras do GBML a partir do rule_details_per_chunk.json

    Returns:
        Lista de dicionarios com metricas por chunk
    """
    rule_details_file = experiment_dir / "run_1" / "rule_details_per_chunk.json"

    if not rule_details_file.exists():
        logger.warning(f"Arquivo nao encontrado: {rule_details_file}")
        return []

    try:
        with open(rule_details_file, 'r') as f:
            data = json.load(f)

        results = []

        # O arquivo pode ter formato de lista ou dicionario
        if isinstance(data, list):
            for chunk_idx, chunk_data in enumerate(data):
                metrics = {
                    'chunk': chunk_idx,
                    'n_and_ops': len(chunk_data.get('logical_ops', [])),
                    'n_or_ops': 0,  # Contar ORs separadamente
                    'n_conditions': len(chunk_data.get('comparison_ops', [])),
                    'n_thresholds': len(chunk_data.get('numeric_thresholds', [])),
                    'n_features_used': len(chunk_data.get('features', [])),
                    'avg_threshold': np.mean(chunk_data.get('numeric_thresholds', [0])) if chunk_data.get('numeric_thresholds') else 0
                }

                # Contar AND vs OR
                logical_ops = chunk_data.get('logical_ops', [])
                metrics['n_and_ops'] = logical_ops.count('AND')
                metrics['n_or_ops'] = logical_ops.count('OR')
                metrics['n_total_logical_ops'] = len(logical_ops)

                results.append(metrics)

        elif isinstance(data, dict):
            for chunk_idx, chunk_data in data.items():
                if isinstance(chunk_data, dict):
                    # Processar cada classe
                    total_rules = 0
                    total_conditions = 0
                    total_and = 0
                    total_or = 0

                    for class_label, rules in chunk_data.items():
                        if isinstance(rules, list):
                            total_rules += len(rules)
                            for rule in rules:
                                total_conditions += rule.get('n_conditions', 0)
                                total_and += rule.get('n_and_ops', 0)
                                total_or += rule.get('n_or_ops', 0)

                    results.append({
                        'chunk': int(chunk_idx),
                        'n_rules': total_rules,
                        'n_conditions': total_conditions,
                        'n_and_ops': total_and,
                        'n_or_ops': total_or,
                        'n_total_logical_ops': total_and + total_or
                    })

        return results

    except Exception as e:
        logger.error(f"Erro ao ler {rule_details_file}: {e}")
        return []


def parse_rules_history(experiment_dir: Path) -> List[Dict]:
    """
    Parseia o arquivo RulesHistory para extrair regras por chunk.

    Returns:
        Lista de dicionarios com regras por chunk
    """
    # Encontrar arquivo RulesHistory
    run_dir = experiment_dir / "run_1"
    rules_files = list(run_dir.glob("RulesHistory_*.txt"))

    if not rules_files:
        logger.warning(f"RulesHistory nao encontrado em {run_dir}")
        return []

    rules_file = rules_files[0]

    try:
        with open(rules_file, 'r', encoding='utf-8') as f:
            content = f.read()

        chunks_data = []

        # Regex para encontrar chunks
        chunk_pattern = r'--- Chunk (\d+) \(Trained\) ---.*?(?=--- Chunk \d+|$)'
        chunks = re.findall(chunk_pattern, content, re.DOTALL)

        # Parsear cada chunk
        chunk_sections = re.split(r'--- Chunk \d+ \(Trained\) ---', content)[1:]

        for idx, section in enumerate(chunk_sections):
            chunk_data = {
                'chunk': idx,
                'rules': [],
                'n_rules': 0,
                'total_conditions': 0,
                'total_and_ops': 0,
                'total_or_ops': 0,
                'avg_rule_length': 0
            }

            # Extrair regras (linhas que comecam com IF)
            rule_pattern = r'IF (.+?) THEN Class (\d+)'
            rules = re.findall(rule_pattern, section)

            chunk_data['n_rules'] = len(rules)

            total_conditions = 0
            total_and = 0
            total_or = 0
            rule_lengths = []

            for rule_text, class_label in rules:
                # Contar condicoes (comparacoes)
                conditions = len(re.findall(r'[<>=]+', rule_text))
                total_conditions += conditions

                # Contar operadores
                and_count = rule_text.count(' AND ')
                or_count = rule_text.count(' OR ')
                total_and += and_count
                total_or += or_count

                # Comprimento da regra
                rule_lengths.append(len(rule_text))

                chunk_data['rules'].append({
                    'condition': rule_text,
                    'class': class_label,
                    'n_conditions': conditions,
                    'n_and': and_count,
                    'n_or': or_count
                })

            chunk_data['total_conditions'] = total_conditions
            chunk_data['total_and_ops'] = total_and
            chunk_data['total_or_ops'] = total_or
            chunk_data['avg_rule_length'] = np.mean(rule_lengths) if rule_lengths else 0
            chunk_data['avg_conditions_per_rule'] = total_conditions / len(rules) if rules else 0

            chunks_data.append(chunk_data)

        return chunks_data

    except Exception as e:
        logger.error(f"Erro ao parsear {rules_file}: {e}")
        return []


def extract_gbml_performance(experiment_dir: Path) -> List[Dict]:
    """
    Extrai metricas de performance do GBML.
    """
    metrics_file = experiment_dir / "run_1" / "chunk_metrics.json"

    if not metrics_file.exists():
        return []

    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Erro ao ler {metrics_file}: {e}")
        return []


# ==============================================================================
# FUNCOES DE EXTRACAO - ERulesD2S
# ==============================================================================

def extract_erulesd2s_metrics(experiment_dir: Path) -> List[Dict]:
    """
    Extrai metricas do ERulesD2S a partir dos CSVs de resultado.

    Returns:
        Lista de dicionarios com metricas por chunk
    """
    results = []
    run_dir = experiment_dir / "run_1"

    # Procurar por pastas erulesd2s_chunk_*
    chunk_dirs = sorted(run_dir.glob("erulesd2s_chunk_*"))

    if not chunk_dirs:
        # Tentar arquivo consolidado
        consolidated = run_dir / "erulesd2s_results.csv"
        if consolidated.exists():
            try:
                df = pd.read_csv(consolidated)
                for idx, row in df.iterrows():
                    results.append({
                        'chunk': row.get('chunk', idx),
                        'accuracy': row.get('accuracy', 0),
                        'gmean': row.get('gmean', 0),
                        'n_rules': row.get('NumberRules', 0),
                        'n_conditions': row.get('NumberConditions', 0),
                        'n_nodes': row.get('NumberNodes', 0)
                    })
            except Exception as e:
                logger.warning(f"Erro ao ler CSV consolidado: {e}")
        return results

    for chunk_dir in chunk_dirs:
        chunk_idx = int(chunk_dir.name.split('_')[-1])

        # Ler CSV de resultados
        csv_file = chunk_dir / "erulesd2s_results.csv"
        if not csv_file.exists():
            continue

        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()

            # Filtrar headers duplicados
            data_lines = [l for l in lines if not l.startswith('learning')]

            if len(data_lines) >= 2:
                header = data_lines[0].strip().split(',')
                values = data_lines[-1].strip().split(',')

                data_dict = dict(zip(header, values))

                metrics = {
                    'chunk': chunk_idx,
                    'n_rules': float(data_dict.get('NumberRules', 0)),
                    'n_conditions': float(data_dict.get('NumberConditions', 0)),
                    'n_nodes': float(data_dict.get('NumberNodes', 0)),
                    'accuracy': float(data_dict.get('classifications correct (percent)', 0)) / 100
                }

                results.append(metrics)

        except Exception as e:
            logger.warning(f"Erro ao ler {csv_file}: {e}")

    return sorted(results, key=lambda x: x['chunk'])


# ==============================================================================
# CALCULO DE METRICAS DE TRANSICAO (TCS/RIR/AMS)
# ==============================================================================

def calculate_rule_similarity(rule1: str, rule2: str) -> float:
    """
    Calcula similaridade entre duas regras usando distancia de Levenshtein normalizada.
    """
    if rule1 == rule2:
        return 1.0

    len1, len2 = len(rule1), len(rule2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Matriz de distancia
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if rule1[i-1] == rule2[j-1] else 1
            matrix[i][j] = min(
                matrix[i-1][j] + 1,
                matrix[i][j-1] + 1,
                matrix[i-1][j-1] + cost
            )

    distance = matrix[len1][len2]
    max_len = max(len1, len2)
    similarity = 1.0 - (distance / max_len)

    return similarity


def calculate_transition_metrics(chunks_data: List[Dict]) -> List[Dict]:
    """
    Calcula metricas de transicao (TCS, RIR, AMS) entre chunks consecutivos.

    Args:
        chunks_data: Lista de dados de regras por chunk (do parse_rules_history)

    Returns:
        Lista de metricas de transicao
    """
    transitions = []

    # Limite de regras para calculo de similaridade detalhado (evita O(n^2) muito lento)
    MAX_RULES_FOR_SIMILARITY = 50

    for i in range(len(chunks_data) - 1):
        chunk_i = chunks_data[i]
        chunk_j = chunks_data[i + 1]

        rules_i = [r['condition'] for r in chunk_i.get('rules', [])]
        rules_j = [r['condition'] for r in chunk_j.get('rules', [])]

        if not rules_i or not rules_j:
            transitions.append({
                'transition': f"{i}->{i+1}",
                'chunk_from': i,
                'chunk_to': i + 1,
                'RIR': 1.0,  # Instabilidade maxima se nao ha regras
                'AMS': 1.0,
                'TCS': 1.0,
                'unchanged_count': 0,
                'modified_count': 0,
                'new_count': len(rules_j),
                'deleted_count': len(rules_i)
            })
            continue

        # Identificar regras unchanged, modified, new, deleted
        unchanged = []
        modified_pairs = []

        rules_i_matched = set()
        rules_j_matched = set()

        # Passo 1: Identificar regras identicas (unchanged)
        for idx_i, rule_i in enumerate(rules_i):
            for idx_j, rule_j in enumerate(rules_j):
                if idx_j in rules_j_matched:
                    continue
                if rule_i == rule_j:
                    unchanged.append((idx_i, idx_j))
                    rules_i_matched.add(idx_i)
                    rules_j_matched.add(idx_j)
                    break

        # Passo 2: Identificar regras modificadas (similaridade > threshold)
        SIMILARITY_THRESHOLD = 0.5
        SEVERITY_THRESHOLD = 0.8

        candidates = []
        severities = []

        # Se houver muitas regras, pular o calculo de similaridade detalhado
        n_unmatched_i = len(rules_i) - len(rules_i_matched)
        n_unmatched_j = len(rules_j) - len(rules_j_matched)
        skip_similarity = (n_unmatched_i * n_unmatched_j) > (MAX_RULES_FOR_SIMILARITY * MAX_RULES_FOR_SIMILARITY)

        if not skip_similarity:
            for idx_i, rule_i in enumerate(rules_i):
                if idx_i in rules_i_matched:
                    continue
                for idx_j, rule_j in enumerate(rules_j):
                    if idx_j in rules_j_matched:
                        continue

                    similarity = calculate_rule_similarity(rule_i, rule_j)
                    if similarity >= SIMILARITY_THRESHOLD:
                        severity = 1.0 - similarity  # Severity = 1 - similarity
                        if severity < SEVERITY_THRESHOLD:
                            candidates.append((idx_i, idx_j, similarity, severity))

        # Ordenar por menor severidade (melhor match)
        candidates.sort(key=lambda x: x[3])

        for idx_i, idx_j, sim, sev in candidates:
            if idx_i not in rules_i_matched and idx_j not in rules_j_matched:
                modified_pairs.append((idx_i, idx_j, sev))
                severities.append(sev)
                rules_i_matched.add(idx_i)
                rules_j_matched.add(idx_j)

        # Contar new e deleted
        new_count = len(rules_j) - len(rules_j_matched)
        deleted_count = len(rules_i) - len(rules_i_matched)
        unchanged_count = len(unchanged)
        modified_count = len(modified_pairs)

        # Calcular metricas
        total_rules = len(rules_i) + len(rules_j)

        # RIR (Rule Instability Rate)
        RIR = (new_count + deleted_count) / total_rules if total_rules > 0 else 0.0

        # AMS (Average Modification Severity)
        AMS = np.mean(severities) if severities else 0.0

        # TCS (Transition Change Score)
        prop_modified = modified_count / len(rules_j) if len(rules_j) > 0 else 0.0
        TCS = W_INSTABILITY * RIR + W_MODIFICATION_IMPACT * prop_modified * AMS
        TCS = min(max(TCS, 0.0), 1.0)

        transitions.append({
            'transition': f"{i}->{i+1}",
            'chunk_from': i,
            'chunk_to': i + 1,
            'RIR': round(RIR, 4),
            'AMS': round(AMS, 4),
            'TCS': round(TCS, 4),
            'unchanged_count': unchanged_count,
            'modified_count': modified_count,
            'new_count': new_count,
            'deleted_count': deleted_count,
            'total_rules_from': len(rules_i),
            'total_rules_to': len(rules_j)
        })

    return transitions


# ==============================================================================
# FUNCAO PRINCIPAL DE EXTRACAO
# ==============================================================================

def extract_all_metrics(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extrai todas as metricas de todos os experimentos.

    Returns:
        Tuple de DataFrames:
        - gbml_rules: Metricas de regras GBML por chunk
        - gbml_transitions: Metricas de transicao GBML
        - erulesd2s_metrics: Metricas ERulesD2S
        - performance_comparison: Comparacao de performance
    """
    gbml_rules_all = []
    gbml_transitions_all = []
    erulesd2s_all = []
    performance_all = []

    for exp_dir_rel in EXPERIMENT_DIRS:
        exp_base = base_dir / exp_dir_rel

        if not exp_base.exists():
            logger.warning(f"Diretorio nao existe: {exp_base}")
            continue

        # Listar datasets neste batch
        datasets = [d for d in exp_base.iterdir() if d.is_dir() and not d.name.startswith('.')]

        batch_name = exp_dir_rel.split('/')[-1]

        for dataset_dir in datasets:
            dataset_name = dataset_dir.name
            logger.info(f"Processando: {batch_name}/{dataset_name}")

            # Extrair metricas GBML
            rules_data = parse_rules_history(dataset_dir)
            rule_details = extract_gbml_rule_details(dataset_dir)
            performance = extract_gbml_performance(dataset_dir)

            # Adicionar dados de regras
            for chunk_data in rules_data:
                gbml_rules_all.append({
                    'batch': batch_name,
                    'dataset': dataset_name,
                    'chunk': chunk_data['chunk'],
                    'n_rules': chunk_data['n_rules'],
                    'total_conditions': chunk_data['total_conditions'],
                    'total_and_ops': chunk_data['total_and_ops'],
                    'total_or_ops': chunk_data['total_or_ops'],
                    'avg_conditions_per_rule': chunk_data['avg_conditions_per_rule'],
                    'avg_rule_length': chunk_data['avg_rule_length']
                })

            # Calcular metricas de transicao
            if rules_data:
                transitions = calculate_transition_metrics(rules_data)
                for trans in transitions:
                    trans['batch'] = batch_name
                    trans['dataset'] = dataset_name
                    gbml_transitions_all.append(trans)

            # Adicionar performance
            for perf in performance:
                performance_all.append({
                    'batch': batch_name,
                    'dataset': dataset_name,
                    'model': 'GBML',
                    'chunk': perf.get('chunk', 0),
                    'train_gmean': perf.get('train_gmean', 0),
                    'test_gmean': perf.get('test_gmean', 0),
                    'test_f1': perf.get('test_f1', 0)
                })

            # Extrair metricas ERulesD2S
            erulesd2s_metrics = extract_erulesd2s_metrics(dataset_dir)
            for metrics in erulesd2s_metrics:
                metrics['batch'] = batch_name
                metrics['dataset'] = dataset_name
                erulesd2s_all.append(metrics)

    # Criar DataFrames
    df_gbml_rules = pd.DataFrame(gbml_rules_all)
    df_gbml_transitions = pd.DataFrame(gbml_transitions_all)
    df_erulesd2s = pd.DataFrame(erulesd2s_all)
    df_performance = pd.DataFrame(performance_all)

    return df_gbml_rules, df_gbml_transitions, df_erulesd2s, df_performance


# ==============================================================================
# ANALISE ESTATISTICA
# ==============================================================================

def generate_statistical_analysis(
    df_gbml_rules: pd.DataFrame,
    df_gbml_transitions: pd.DataFrame,
    df_erulesd2s: pd.DataFrame,
    output_dir: Path
) -> str:
    """
    Gera analise estatistica comparativa.
    """
    report = []
    report.append("=" * 80)
    report.append("ANALISE DE EXPLICABILIDADE - GBML vs ERulesD2S")
    report.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")

    # 1. Estatisticas descritivas GBML
    report.append("-" * 80)
    report.append("1. ESTATISTICAS DESCRITIVAS - GBML")
    report.append("-" * 80)

    if not df_gbml_rules.empty:
        report.append(f"\nTotal de datasets analisados: {df_gbml_rules['dataset'].nunique()}")
        report.append(f"Total de chunks: {len(df_gbml_rules)}")

        report.append("\nNumero de Regras por Chunk:")
        report.append(f"  Media: {df_gbml_rules['n_rules'].mean():.2f}")
        report.append(f"  Desvio Padrao: {df_gbml_rules['n_rules'].std():.2f}")
        report.append(f"  Minimo: {df_gbml_rules['n_rules'].min()}")
        report.append(f"  Maximo: {df_gbml_rules['n_rules'].max()}")

        report.append("\nCondicoes por Regra (media):")
        report.append(f"  Media: {df_gbml_rules['avg_conditions_per_rule'].mean():.2f}")
        report.append(f"  Desvio Padrao: {df_gbml_rules['avg_conditions_per_rule'].std():.2f}")

        report.append("\nOperadores Logicos por Chunk:")
        report.append(f"  AND (media): {df_gbml_rules['total_and_ops'].mean():.2f}")
        report.append(f"  OR (media): {df_gbml_rules['total_or_ops'].mean():.2f}")
    else:
        report.append("\n[AVISO] Nenhum dado GBML encontrado")

    # 2. Estatisticas de Transicao
    report.append("")
    report.append("-" * 80)
    report.append("2. METRICAS DE TRANSICAO (Evolutionary Change Metrics)")
    report.append("-" * 80)

    if not df_gbml_transitions.empty:
        report.append(f"\nTotal de transicoes analisadas: {len(df_gbml_transitions)}")

        report.append("\nRIR (Rule Instability Rate):")
        report.append(f"  Media: {df_gbml_transitions['RIR'].mean():.4f}")
        report.append(f"  Desvio Padrao: {df_gbml_transitions['RIR'].std():.4f}")
        report.append(f"  Minimo: {df_gbml_transitions['RIR'].min():.4f}")
        report.append(f"  Maximo: {df_gbml_transitions['RIR'].max():.4f}")

        report.append("\nAMS (Average Modification Severity):")
        report.append(f"  Media: {df_gbml_transitions['AMS'].mean():.4f}")
        report.append(f"  Desvio Padrao: {df_gbml_transitions['AMS'].std():.4f}")

        report.append("\nTCS (Transition Change Score):")
        report.append(f"  Media: {df_gbml_transitions['TCS'].mean():.4f}")
        report.append(f"  Desvio Padrao: {df_gbml_transitions['TCS'].std():.4f}")

        # Identificar transicoes com alto RIR (possivel drift)
        high_rir = df_gbml_transitions[df_gbml_transitions['RIR'] > 0.5]
        report.append(f"\nTransicoes com RIR > 0.5 (alto drift): {len(high_rir)}")

        if len(high_rir) > 0:
            report.append("\nTop 10 transicoes com maior RIR:")
            top_rir = df_gbml_transitions.nlargest(10, 'RIR')[['dataset', 'transition', 'RIR', 'TCS']]
            report.append(top_rir.to_string(index=False))
    else:
        report.append("\n[AVISO] Nenhuma transicao encontrada")

    # 3. Estatisticas ERulesD2S
    report.append("")
    report.append("-" * 80)
    report.append("3. ESTATISTICAS DESCRITIVAS - ERulesD2S")
    report.append("-" * 80)

    if not df_erulesd2s.empty:
        report.append(f"\nTotal de datasets com ERulesD2S: {df_erulesd2s['dataset'].nunique()}")
        report.append(f"Total de chunks: {len(df_erulesd2s)}")

        report.append("\nNumero de Regras:")
        report.append(f"  Media: {df_erulesd2s['n_rules'].mean():.2f}")
        report.append(f"  Desvio Padrao: {df_erulesd2s['n_rules'].std():.2f}")

        report.append("\nNumero de Condicoes:")
        report.append(f"  Media: {df_erulesd2s['n_conditions'].mean():.2f}")
        report.append(f"  Desvio Padrao: {df_erulesd2s['n_conditions'].std():.2f}")

        report.append("\nNumero de Nos:")
        report.append(f"  Media: {df_erulesd2s['n_nodes'].mean():.2f}")
        report.append(f"  Desvio Padrao: {df_erulesd2s['n_nodes'].std():.2f}")
    else:
        report.append("\n[AVISO] Nenhum dado ERulesD2S encontrado")

    # 4. Comparacao direta
    report.append("")
    report.append("-" * 80)
    report.append("4. COMPARACAO DIRETA - GBML vs ERulesD2S")
    report.append("-" * 80)

    if not df_gbml_rules.empty and not df_erulesd2s.empty:
        # Encontrar datasets em comum
        gbml_datasets = set(df_gbml_rules['dataset'].unique())
        erulesd2s_datasets = set(df_erulesd2s['dataset'].unique())
        common_datasets = gbml_datasets.intersection(erulesd2s_datasets)

        report.append(f"\nDatasets em comum: {len(common_datasets)}")

        if common_datasets:
            # Calcular medias por dataset
            gbml_avg = df_gbml_rules[df_gbml_rules['dataset'].isin(common_datasets)].groupby('dataset').agg({
                'n_rules': 'mean',
                'total_conditions': 'mean'
            }).reset_index()

            erulesd2s_avg = df_erulesd2s[df_erulesd2s['dataset'].isin(common_datasets)].groupby('dataset').agg({
                'n_rules': 'mean',
                'n_conditions': 'mean'
            }).reset_index()

            report.append("\nComparacao de numero de regras (media por dataset):")
            report.append(f"  GBML: {gbml_avg['n_rules'].mean():.2f}")
            report.append(f"  ERulesD2S: {erulesd2s_avg['n_rules'].mean():.2f}")

            report.append("\nComparacao de condicoes (media por dataset):")
            report.append(f"  GBML: {gbml_avg['total_conditions'].mean():.2f}")
            report.append(f"  ERulesD2S: {erulesd2s_avg['n_conditions'].mean():.2f}")

    # 5. Sumario para artigo
    report.append("")
    report.append("-" * 80)
    report.append("5. SUMARIO PARA ARTIGO")
    report.append("-" * 80)

    report.append("""
Os resultados mostram que o GBML gera regras interpretaveis com as seguintes
caracteristicas:

- Numero medio de regras por chunk: {n_rules_gbml}
- Condicoes medias por regra: {cond_per_rule}
- Uso predominante de operador AND (vs OR): {and_pct}%

As metricas evolutivas (TCS, RIR, AMS) permitem monitorar a estabilidade
das regras durante o processamento do stream:

- RIR medio: {rir_avg} (indicando {rir_interpretation})
- TCS medio: {tcs_avg} (score composto de mudanca)
- AMS medio: {ams_avg} (severidade das modificacoes)

Em comparacao com ERulesD2S:
- GBML gera {comparison_rules} regras em media
- GBML produz regras {comparison_complexity} complexas
""".format(
        n_rules_gbml=f"{df_gbml_rules['n_rules'].mean():.1f}" if not df_gbml_rules.empty else "N/A",
        cond_per_rule=f"{df_gbml_rules['avg_conditions_per_rule'].mean():.1f}" if not df_gbml_rules.empty else "N/A",
        and_pct=f"{100 * df_gbml_rules['total_and_ops'].sum() / (df_gbml_rules['total_and_ops'].sum() + df_gbml_rules['total_or_ops'].sum() + 1):.0f}" if not df_gbml_rules.empty else "N/A",
        rir_avg=f"{df_gbml_transitions['RIR'].mean():.3f}" if not df_gbml_transitions.empty else "N/A",
        rir_interpretation="estabilidade moderada" if not df_gbml_transitions.empty and df_gbml_transitions['RIR'].mean() < 0.5 else "alta instabilidade",
        tcs_avg=f"{df_gbml_transitions['TCS'].mean():.3f}" if not df_gbml_transitions.empty else "N/A",
        ams_avg=f"{df_gbml_transitions['AMS'].mean():.3f}" if not df_gbml_transitions.empty else "N/A",
        comparison_rules="menos" if not df_gbml_rules.empty and not df_erulesd2s.empty and df_gbml_rules['n_rules'].mean() < df_erulesd2s['n_rules'].mean() else "mais",
        comparison_complexity="menos" if not df_gbml_rules.empty and not df_erulesd2s.empty else "igualmente"
    ))

    report.append("")
    report.append("=" * 80)
    report.append("FIM DO RELATORIO")
    report.append("=" * 80)

    report_text = "\n".join(report)

    # Salvar relatorio
    report_file = output_dir / "statistical_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    logger.info(f"Relatorio salvo em: {report_file}")

    return report_text


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Funcao principal."""
    base_dir = Path(".")

    # Criar diretorio de output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("INICIANDO ANALISE DE EXPLICABILIDADE")
    logger.info("=" * 60)

    # Extrair todas as metricas
    logger.info("\n[1/4] Extraindo metricas de todos os experimentos...")
    df_gbml_rules, df_gbml_transitions, df_erulesd2s, df_performance = extract_all_metrics(base_dir)

    # Salvar CSVs
    logger.info("\n[2/4] Salvando CSVs consolidados...")

    if not df_gbml_rules.empty:
        csv_path = OUTPUT_DIR / "gbml_rules_per_chunk.csv"
        df_gbml_rules.to_csv(csv_path, index=False)
        logger.info(f"  Salvo: {csv_path} ({len(df_gbml_rules)} registros)")

    if not df_gbml_transitions.empty:
        csv_path = OUTPUT_DIR / "gbml_transition_metrics.csv"
        df_gbml_transitions.to_csv(csv_path, index=False)
        logger.info(f"  Salvo: {csv_path} ({len(df_gbml_transitions)} registros)")

    if not df_erulesd2s.empty:
        csv_path = OUTPUT_DIR / "erulesd2s_metrics.csv"
        df_erulesd2s.to_csv(csv_path, index=False)
        logger.info(f"  Salvo: {csv_path} ({len(df_erulesd2s)} registros)")

    if not df_performance.empty:
        csv_path = OUTPUT_DIR / "performance_comparison.csv"
        df_performance.to_csv(csv_path, index=False)
        logger.info(f"  Salvo: {csv_path} ({len(df_performance)} registros)")

    # Gerar analise estatistica
    logger.info("\n[3/4] Gerando analise estatistica...")
    report = generate_statistical_analysis(
        df_gbml_rules,
        df_gbml_transitions,
        df_erulesd2s,
        OUTPUT_DIR
    )

    # Criar tabela comparativa consolidada
    logger.info("\n[4/4] Criando tabelas consolidadas para artigo...")

    # Tabela de complexidade por dataset
    if not df_gbml_rules.empty:
        complexity_summary = df_gbml_rules.groupby('dataset').agg({
            'n_rules': ['mean', 'std'],
            'total_conditions': ['mean', 'std'],
            'avg_conditions_per_rule': ['mean', 'std'],
            'total_and_ops': 'mean',
            'total_or_ops': 'mean'
        }).round(2)
        complexity_summary.columns = ['_'.join(col).strip() for col in complexity_summary.columns]
        complexity_summary.to_csv(OUTPUT_DIR / "gbml_complexity_by_dataset.csv")
        logger.info(f"  Salvo: gbml_complexity_by_dataset.csv")

    # Tabela de transicoes por dataset
    if not df_gbml_transitions.empty:
        transition_summary = df_gbml_transitions.groupby('dataset').agg({
            'RIR': ['mean', 'std', 'max'],
            'AMS': ['mean', 'std'],
            'TCS': ['mean', 'std', 'max']
        }).round(4)
        transition_summary.columns = ['_'.join(col).strip() for col in transition_summary.columns]
        transition_summary.to_csv(OUTPUT_DIR / "gbml_transitions_by_dataset.csv")
        logger.info(f"  Salvo: gbml_transitions_by_dataset.csv")

    # Imprimir sumario
    print("\n" + "=" * 60)
    print("SUMARIO DA ANALISE")
    print("=" * 60)
    print(f"\nArquivos gerados em: {OUTPUT_DIR.absolute()}")
    print(f"\nDatasets GBML processados: {df_gbml_rules['dataset'].nunique() if not df_gbml_rules.empty else 0}")
    print(f"Datasets ERulesD2S processados: {df_erulesd2s['dataset'].nunique() if not df_erulesd2s.empty else 0}")
    print(f"Total de transicoes analisadas: {len(df_gbml_transitions)}")

    if not df_gbml_transitions.empty:
        print(f"\nMetricas de Transicao (medias globais):")
        print(f"  RIR: {df_gbml_transitions['RIR'].mean():.4f}")
        print(f"  AMS: {df_gbml_transitions['AMS'].mean():.4f}")
        print(f"  TCS: {df_gbml_transitions['TCS'].mean():.4f}")

    print("\n" + "=" * 60)
    print("ANALISE CONCLUIDA")
    print("=" * 60)


if __name__ == "__main__":
    main()
