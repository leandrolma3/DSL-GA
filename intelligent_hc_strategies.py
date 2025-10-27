# intelligent_hc_strategies.py
"""
Estratégias Inteligentes de Hill Climbing

PROPÓSITO: Implementa 3 estratégias inteligentes que usam Decision Trees
           para gerar variantes de alta qualidade do elite.

ESTRATÉGIAS:
1. Error-Focused DT Rules: Extrai regras DT dos erros e substitui piores regras
2. Ensemble Boosting: Treina DT com pesos nos erros (boosting manual)
3. Guided Mutation: Muta regras focando em features importantes da DT

AUTOR: Claude Code
DATA: 2025-10-13
"""

import logging
import copy
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from individual import Individual
from dt_rule_extraction import (
    extract_rules_from_tree,
    rank_rules_by_error_coverage,
    convert_dt_rule_to_ruletree,
    filter_rules_by_class
)

logger = logging.getLogger("intelligent_hc")


# ============================================================================
# ESTRATÉGIA 1: ERROR-FOCUSED DT RULES
# ============================================================================

def error_focused_dt_rules(elite: Individual,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           n_variants: int = 5,
                           **kwargs) -> List[Individual]:
    """
    ESTRATÉGIA 1: Extrai regras DT dos erros do elite e substitui piores regras.

    Processo:
    1. Identifica erros do elite (regiões onde ele falha)
    2. Treina DT profunda nos erros (max_depth=10)
    3. Extrai TODAS as regras DT via root→leaf paths
    4. Ranqueia regras DT por cobertura de erros
    5. Substitui 20-30% piores regras do elite por regras DT top

    Args:
        elite: Melhor indivíduo atual
        X_train: DataFrame com features de treino
        y_train: Series com labels de treino
        n_variants: Número de variantes a gerar
        **kwargs: Parâmetros adicionais (max_depth, attributes, etc.)

    Returns:
        Lista de variantes do elite
    """
    try:
        # 1. Identifica erros do elite
        y_pred = []
        for idx in range(len(X_train)):
            instance = X_train.iloc[idx].to_dict()
            pred = elite._predict(instance)
            y_pred.append(pred)

        y_pred = np.array(y_pred)
        error_mask = (y_pred != y_train.values)

        X_errors = X_train[error_mask]
        y_errors = y_train[error_mask]

        if len(X_errors) < 10:
            logger.info(f"       error_focused_dt_rules: Elite tem apenas {len(X_errors)} erros - não aplicável")
            return []

        error_rate = len(X_errors) / len(X_train)
        logger.info(f"       error_focused_dt_rules: Elite erra em {len(X_errors)}/{len(X_train)} ({error_rate:.1%})")

        # 2. Treina DT profunda nos erros
        dt_error = DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=5,
            min_samples_split=10,
            class_weight='balanced',
            random_state=random.randint(0, 10000)
        )
        dt_error.fit(X_errors, y_errors)

        dt_acc = dt_error.score(X_errors, y_errors)
        logger.info(f"       DT nos erros: acc={dt_acc:.3f}, depth={dt_error.get_depth()}, nodes={dt_error.tree_.node_count}")

        # 3. Extrai regras DT
        feature_names = list(X_train.columns)
        classes = list(elite.classes)

        dt_rules = extract_rules_from_tree(dt_error, feature_names, classes)

        if not dt_rules:
            logger.warning(f"       error_focused_dt_rules: Nenhuma regra extraída da DT")
            return []

        # 4. Ranqueia regras DT por cobertura de erros
        dt_rules_ranked = rank_rules_by_error_coverage(dt_rules, X_errors, y_errors)

        if not dt_rules_ranked:
            logger.warning(f"       error_focused_dt_rules: Nenhuma regra DT ranqueada")
            return []

        logger.info(f"       Top-3 DT rules: scores={[f'{r.score:.3f}' for r in dt_rules_ranked[:3]]}")

        # 5. Gera variantes substituindo piores regras do elite
        variants = []

        for i in range(n_variants):
            variant = copy.deepcopy(elite)

            # Para cada classe, substitui piores regras
            for cls in variant.classes:
                if cls not in variant.rules or not variant.rules[cls]:
                    continue

                # Filtra regras DT desta classe
                dt_rules_cls = filter_rules_by_class(dt_rules_ranked, cls)

                if not dt_rules_cls:
                    continue

                # Substitui 20-30% piores regras (varia por variante)
                replacement_ratio = random.uniform(0.20, 0.30)
                n_replace = max(1, int(len(variant.rules[cls]) * replacement_ratio))

                # Identifica piores regras do variant (usando quality_score)
                rules_with_score = [(idx, rule.quality_score) for idx, rule in enumerate(variant.rules[cls])]
                rules_with_score.sort(key=lambda x: x[1])  # Ordena crescente por score

                worst_indices = [idx for idx, score in rules_with_score[:n_replace]]

                # Substitui por top DT rules
                dt_rules_to_use = dt_rules_cls[:n_replace]

                for idx, dt_rule in zip(worst_indices, dt_rules_to_use):
                    # Converte DTRule para RuleTree
                    rule_tree = convert_dt_rule_to_ruletree(
                        dt_rule=dt_rule,
                        max_depth=kwargs.get('max_depth', 10),
                        attributes=kwargs['attributes'],
                        value_ranges=kwargs['value_ranges'],
                        category_values=kwargs.get('category_values', {}),
                        categorical_features=kwargs.get('categorical_features', set())
                    )

                    if rule_tree and rule_tree.is_valid_rule():
                        variant.rules[cls][idx] = rule_tree

            variants.append(variant)

        logger.info(f"       error_focused_dt_rules: {len(variants)} variantes geradas")

        return variants

    except Exception as e:
        logger.error(f"       error_focused_dt_rules falhou: {e}", exc_info=True)
        return []


# ============================================================================
# ESTRATÉGIA 2: ENSEMBLE BOOSTING
# ============================================================================

def ensemble_boosting(elite: Individual,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     n_variants: int = 4,
                     **kwargs) -> List[Individual]:
    """
    ESTRATÉGIA 2: Treina DT com pesos nos erros (boosting manual).

    Processo:
    1. Calcula pesos: erros do elite têm peso 3×
    2. Treina DT com sample_weight (boosting manual)
    3. Extrai regras DT boosted
    4. Cria híbridos: 70% elite + 30% DT boosted

    Args:
        elite: Melhor indivíduo atual
        X_train: DataFrame com features de treino
        y_train: Series com labels de treino
        n_variants: Número de variantes a gerar
        **kwargs: Parâmetros adicionais

    Returns:
        Lista de variantes do elite
    """
    try:
        # 1. Calcula pesos baseados em erros do elite
        y_pred = []
        for idx in range(len(X_train)):
            instance = X_train.iloc[idx].to_dict()
            pred = elite._predict(instance)
            y_pred.append(pred)

        y_pred = np.array(y_pred)
        sample_weights = np.ones(len(X_train))
        sample_weights[y_pred != y_train.values] = 3.0  # Erros têm peso 3×

        error_count = (y_pred != y_train.values).sum()
        logger.info(f"       ensemble_boosting: {error_count} erros com peso 3×")

        variants = []

        for i in range(n_variants):
            # 2. Treina DT com pesos (profundidade varia)
            dt_depth = random.choice([6, 8, 10, 12])

            dt_boosted = DecisionTreeClassifier(
                max_depth=dt_depth,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=random.randint(0, 10000)
            )
            dt_boosted.fit(X_train, y_train, sample_weight=sample_weights)

            dt_acc = dt_boosted.score(X_train, y_train)
            logger.info(f"       DT boosted #{i+1}: depth={dt_depth}, acc={dt_acc:.3f}")

            # 3. Extrai regras DT boosted
            feature_names = list(X_train.columns)
            classes = list(elite.classes)

            dt_rules = extract_rules_from_tree(dt_boosted, feature_names, classes)

            if not dt_rules:
                continue

            # Ranqueia regras por fitness no dataset completo
            dt_rules_ranked = rank_rules_by_error_coverage(dt_rules, X_train, y_train)

            # 4. Cria híbrido: 70% elite + 30% DT boosted
            variant = copy.deepcopy(elite)

            for cls in variant.classes:
                if cls not in variant.rules or not variant.rules[cls]:
                    continue

                # Filtra regras DT desta classe
                dt_rules_cls = filter_rules_by_class(dt_rules_ranked, cls)

                if not dt_rules_cls:
                    continue

                # Substitui 30% regras aleatórias por DT boosted
                n_replace = max(1, int(len(variant.rules[cls]) * 0.30))
                replace_indices = random.sample(range(len(variant.rules[cls])), n_replace)

                dt_rules_to_use = dt_rules_cls[:n_replace]

                for idx, dt_rule in zip(replace_indices, dt_rules_to_use):
                    rule_tree = convert_dt_rule_to_ruletree(
                        dt_rule=dt_rule,
                        max_depth=kwargs.get('max_depth', 10),
                        attributes=kwargs['attributes'],
                        value_ranges=kwargs['value_ranges'],
                        category_values=kwargs.get('category_values', {}),
                        categorical_features=kwargs.get('categorical_features', set())
                    )

                    if rule_tree and rule_tree.is_valid_rule():
                        variant.rules[cls][idx] = rule_tree

            variants.append(variant)

        logger.info(f"       ensemble_boosting: {len(variants)} variantes geradas")

        return variants

    except Exception as e:
        logger.error(f"       ensemble_boosting falhou: {e}", exc_info=True)
        return []


# ============================================================================
# ESTRATÉGIA 3: GUIDED MUTATION
# ============================================================================

def guided_mutation(elite: Individual,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   n_variants: int = 4,
                   **kwargs) -> List[Individual]:
    """
    ESTRATÉGIA 3: Muta regras focando em features importantes da DT.

    Processo:
    1. Treina DT nos erros do elite
    2. Extrai importância de features (feature_importances_)
    3. Muta 40% das regras do elite focando nas top-5 features
    4. 70% mutações guiadas + 30% mutações aleatórias (diversidade)

    Args:
        elite: Melhor indivíduo atual
        X_train: DataFrame com features de treino
        y_train: Series com labels de treino
        n_variants: Número de variantes a gerar
        **kwargs: Parâmetros adicionais

    Returns:
        Lista de variantes do elite
    """
    try:
        # 1. Identifica erros e treina DT
        y_pred = []
        for idx in range(len(X_train)):
            instance = X_train.iloc[idx].to_dict()
            pred = elite._predict(instance)
            y_pred.append(pred)

        y_pred = np.array(y_pred)
        error_mask = (y_pred != y_train.values)

        X_errors = X_train[error_mask]
        y_errors = y_train[error_mask]

        if len(X_errors) < 10:
            logger.info(f"       guided_mutation: Elite tem apenas {len(X_errors)} erros - não aplicável")
            return []

        dt_error = DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            random_state=random.randint(0, 10000)
        )
        dt_error.fit(X_errors, y_errors)

        # 2. Extrai importância de features
        feature_importance = dt_error.feature_importances_
        feature_names = list(X_train.columns)

        # Top-5 features mais importantes
        top_indices = np.argsort(feature_importance)[-5:]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = [feature_importance[i] for i in top_indices]

        logger.info(f"       guided_mutation: Top-5 features: {top_features}")
        logger.info(f"       importâncias: {[f'{imp:.3f}' for imp in top_importances]}")

        variants = []

        for i in range(n_variants):
            variant = copy.deepcopy(elite)

            # 3. Muta 40% das regras do variant
            total_rules = sum(len(rules) for rules in variant.rules.values())
            n_mutate = max(1, int(total_rules * 0.40))

            mutations_done = 0

            for cls in variant.classes:
                if cls not in variant.rules or not variant.rules[cls]:
                    continue

                for rule_idx, rule in enumerate(variant.rules[cls]):
                    if mutations_done >= n_mutate:
                        break

                    # 70% chance de mutação guiada, 30% aleatória
                    if random.random() < 0.7:
                        # Mutação guiada: foca em top features
                        _mutate_rule_guided(rule, top_features, X_train, kwargs)
                    else:
                        # Mutação aleatória (diversidade)
                        _mutate_rule_random(rule, feature_names, X_train, kwargs)

                    mutations_done += 1

            variants.append(variant)

        logger.info(f"       guided_mutation: {len(variants)} variantes geradas")

        return variants

    except Exception as e:
        logger.error(f"       guided_mutation falhou: {e}", exc_info=True)
        return []


def _mutate_rule_guided(rule, top_features: List[str], X_train: pd.DataFrame, kwargs: Dict):
    """
    Muta uma regra focando em top features importantes.

    Estratégia: Substitui ou adiciona condição usando top feature
    """
    from rule_tree import Node
    from constants import NUMERIC_COMPARISON_OPERATORS

    # Escolhe top feature aleatória
    feature = random.choice(top_features)

    # Calcula threshold baseado em percentis dos dados
    percentile = random.choice([25, 50, 75])
    threshold = X_train[feature].quantile(percentile / 100.0)

    # Escolhe operador
    operator = random.choice(NUMERIC_COMPARISON_OPERATORS)

    # Cria novo nó folha
    new_node = Node(
        attribute=feature,
        operator=operator,
        value=threshold,
        feature_type='numeric'
    )

    # Tenta substituir nó folha aleatório da regra
    _replace_random_leaf(rule.root, new_node)


def _mutate_rule_random(rule, feature_names: List[str], X_train: pd.DataFrame, kwargs: Dict):
    """
    Muta uma regra aleatoriamente (para diversidade).
    """
    from rule_tree import Node
    from constants import NUMERIC_COMPARISON_OPERATORS

    # Escolhe feature aleatória
    feature = random.choice(feature_names)

    # Threshold aleatório no range
    min_val, max_val = X_train[feature].min(), X_train[feature].max()
    threshold = random.uniform(min_val, max_val)

    # Operador aleatório
    operator = random.choice(NUMERIC_COMPARISON_OPERATORS)

    new_node = Node(
        attribute=feature,
        operator=operator,
        value=threshold,
        feature_type='numeric'
    )

    _replace_random_leaf(rule.root, new_node)


def _replace_random_leaf(node, new_node):
    """
    Substitui um nó folha aleatório na árvore.
    """
    if node is None:
        return

    if node.is_leaf():
        # Substitui atributos do nó atual
        node.attribute = new_node.attribute
        node.operator = new_node.operator
        node.value = new_node.value
        node.feature_type = new_node.feature_type
        return

    # Navega recursivamente (50% chance cada lado)
    if random.random() < 0.5 and node.left:
        _replace_random_leaf(node.left, new_node)
    elif node.right:
        _replace_random_leaf(node.right, new_node)


# ============================================================================
# TESTE DO MÓDULO
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
    )

    logger.info("=== Teste do Módulo intelligent_hc_strategies.py ===")
    logger.info("✓ Módulo carregado com sucesso")
    logger.info("  Estratégias disponíveis:")
    logger.info("    1. error_focused_dt_rules")
    logger.info("    2. ensemble_boosting")
    logger.info("    3. guided_mutation")
