# dt_rule_extraction.py
"""
Decision Tree Rule Extraction Module

PROPÓSITO: Extrai regras inteligentes de Decision Trees para uso no Hill Climbing.

COMPONENTES:
1. Extração de regras via root→leaf paths
2. Ranqueamento de regras por cobertura × accuracy
3. Conversão de regras DT para RuleTrees

AUTOR: Claude Code
DATA: 2025-10-13
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.tree import DecisionTreeClassifier, _tree
from rule_tree import RuleTree, Node

logger = logging.getLogger("dt_rule_extraction")


# ============================================================================
# ESTRUTURA DE REGRA DT
# ============================================================================

class DTRule:
    """
    Representa uma regra extraída de uma Decision Tree.

    Uma regra DT é um caminho root→leaf com:
    - conditions: Lista de (feature, operator, threshold)
    - predicted_class: Classe predita pela folha
    - confidence: Confiança da predição (pureza da folha)
    - n_samples: Número de amostras na folha
    """

    def __init__(self, conditions: List[Tuple[str, str, float]],
                 predicted_class: Any, confidence: float, n_samples: int):
        """
        Args:
            conditions: [(feature, operator, threshold), ...]
                       operator: '<=' ou '>'
            predicted_class: Classe predita
            confidence: Confiança (0-1)
            n_samples: Número de amostras na folha
        """
        self.conditions = conditions
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.n_samples = n_samples

        # Métricas de avaliação (calculadas depois)
        self.coverage = 0.0  # % de instâncias que ativam a regra
        self.accuracy = 0.0  # % de acertos quando ativa
        self.score = 0.0     # coverage × accuracy

    def __repr__(self):
        cond_str = " AND ".join([f"{feat} {op} {val:.3f}" for feat, op, val in self.conditions])
        return f"IF {cond_str} THEN class={self.predicted_class} (conf={self.confidence:.2f}, n={self.n_samples})"

    def match(self, X: pd.DataFrame) -> np.ndarray:
        """
        Verifica quais instâncias ativam esta regra.

        Args:
            X: DataFrame com features

        Returns:
            Boolean array indicando quais instâncias ativam a regra
        """
        mask = np.ones(len(X), dtype=bool)

        for feature, operator, threshold in self.conditions:
            if feature not in X.columns:
                logger.warning(f"Feature '{feature}' não encontrada no DataFrame. Regra não aplicável.")
                return np.zeros(len(X), dtype=bool)

            if operator == '<=':
                mask &= (X[feature] <= threshold)
            elif operator == '>':
                mask &= (X[feature] > threshold)
            else:
                logger.warning(f"Operador desconhecido: {operator}")
                return np.zeros(len(X), dtype=bool)

        return mask

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediz classes para instâncias que ativam a regra.

        Args:
            X: DataFrame com features

        Returns:
            Array de classes preditas (só para instâncias que ativam)
        """
        mask = self.match(X)
        predictions = np.full(len(X), None, dtype=object)
        predictions[mask] = self.predicted_class
        return predictions


# ============================================================================
# EXTRAÇÃO DE REGRAS DA DECISION TREE
# ============================================================================

def extract_rules_from_tree(dt_model: DecisionTreeClassifier,
                            feature_names: List[str],
                            classes: List[Any]) -> List[DTRule]:
    """
    Extrai todas as regras de uma Decision Tree como paths root→leaf.

    Cada path da raiz até uma folha vira uma regra DTRule.

    Args:
        dt_model: Decision Tree treinada
        feature_names: Lista de nomes das features
        classes: Lista de classes possíveis

    Returns:
        Lista de DTRule objects
    """
    tree = dt_model.tree_
    rules = []

    def recurse(node: int, conditions: List[Tuple[str, str, float]]):
        """
        Recursivamente percorre a árvore extraindo regras.

        Args:
            node: ID do nó atual
            conditions: Condições acumuladas no caminho até aqui
        """
        # Folha: cria regra completa
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            class_id = np.argmax(tree.value[node][0])
            predicted_class = classes[class_id]

            # Calcula confiança (pureza da folha)
            total_samples = tree.value[node][0].sum()
            class_samples = tree.value[node][0][class_id]
            confidence = class_samples / total_samples if total_samples > 0 else 0.0

            n_samples = int(tree.n_node_samples[node])

            rule = DTRule(
                conditions=conditions.copy(),
                predicted_class=predicted_class,
                confidence=confidence,
                n_samples=n_samples
            )
            rules.append(rule)
            return

        # Nó interno: bifurca
        feature_id = tree.feature[node]
        threshold = tree.threshold[node]
        feature_name = feature_names[feature_id]

        # Left child: feature <= threshold
        left_conditions = conditions.copy()
        left_conditions.append((feature_name, '<=', threshold))
        recurse(tree.children_left[node], left_conditions)

        # Right child: feature > threshold
        right_conditions = conditions.copy()
        right_conditions.append((feature_name, '>', threshold))
        recurse(tree.children_right[node], right_conditions)

    # Inicia recursão na raiz
    recurse(0, [])

    logger.info(f"  Extração DT: {len(rules)} regras extraídas de {tree.node_count} nós")

    return rules


# ============================================================================
# RANQUEAMENTO DE REGRAS
# ============================================================================

def rank_rules_by_error_coverage(rules: List[DTRule],
                                 X_errors: pd.DataFrame,
                                 y_errors: pd.Series) -> List[DTRule]:
    """
    Ranqueia regras DT por quanto elas cobrem erros corretamente.

    Métrica: score = coverage × accuracy
    - coverage: % de erros que a regra ativa
    - accuracy: % de acertos quando regra ativa nos erros

    Args:
        rules: Lista de DTRule objects
        X_errors: DataFrame com features das instâncias onde elite errou
        y_errors: Series com labels corretos das instâncias onde elite errou

    Returns:
        Lista de DTRule objects ranqueadas (melhor → pior)
    """
    if len(X_errors) == 0:
        logger.warning("  Ranqueamento: X_errors vazio, retornando lista vazia")
        return []

    ranked = []

    for rule in rules:
        # Aplicar regra nos erros
        mask = rule.match(X_errors)

        # Se regra não ativa em nenhum erro, pula
        if mask.sum() == 0:
            rule.coverage = 0.0
            rule.accuracy = 0.0
            rule.score = 0.0
            continue

        # Calcular coverage
        rule.coverage = mask.sum() / len(X_errors)

        # Calcular accuracy (quando regra ativa, % que acerta)
        predictions = np.full(len(X_errors), rule.predicted_class, dtype=object)
        correct = (predictions[mask] == y_errors[mask]).sum()
        rule.accuracy = correct / mask.sum() if mask.sum() > 0 else 0.0

        # Score combinado
        rule.score = rule.coverage * rule.accuracy

        ranked.append(rule)

    # Ordenar decrescente por score
    ranked.sort(key=lambda r: r.score, reverse=True)

    # Log estatísticas
    if ranked:
        logger.info(f"  Ranqueamento: {len(ranked)} regras avaliadas")
        logger.info(f"    Top-3 scores: {[f'{r.score:.3f}' for r in ranked[:3]]}")
        logger.info(f"    Top-1: coverage={ranked[0].coverage:.2%}, accuracy={ranked[0].accuracy:.2%}")

    return ranked


def rank_rules_by_fitness(rules: List[DTRule],
                         X: pd.DataFrame,
                         y: pd.Series) -> List[DTRule]:
    """
    Ranqueia regras DT por fitness (coverage × accuracy) no dataset completo.

    Similar a rank_rules_by_error_coverage, mas usado para:
    - Avaliar regras no dataset completo (não só erros)
    - Identificar regras ruins do elite para substituição

    Args:
        rules: Lista de DTRule objects
        X: DataFrame com features
        y: Series com labels

    Returns:
        Lista de DTRule objects ranqueadas (melhor → pior)
    """
    return rank_rules_by_error_coverage(rules, X, y)


# ============================================================================
# CONVERSÃO DE REGRAS DT PARA RULETREE
# ============================================================================

def convert_dt_rule_to_ruletree(dt_rule: DTRule,
                                max_depth: int,
                                attributes: List[str],
                                value_ranges: Dict[str, Tuple[float, float]],
                                category_values: Dict[str, set],
                                categorical_features: set) -> RuleTree:
    """
    Converte uma DTRule extraída da DT para um RuleTree do sistema GA.

    Process:
    1. Cria RuleTree vazio
    2. Constrói árvore AND das condições
    3. Retorna RuleTree pronto para uso no Individual

    Args:
        dt_rule: Regra extraída da DT
        max_depth: Profundidade máxima permitida
        attributes: Lista de atributos
        value_ranges: Ranges de valores numéricos
        category_values: Valores categóricos possíveis
        categorical_features: Set de features categóricas

    Returns:
        RuleTree compatível com Individual
    """
    # Cria RuleTree vazio (sem inicializar aleatoriamente)
    rule_tree = RuleTree(
        max_depth=max_depth,
        attributes=attributes,
        value_ranges=value_ranges,
        category_values=category_values,
        categorical_features=categorical_features
    )

    # Se não há condições, cria regra que sempre ativa (folha simples)
    if not dt_rule.conditions:
        logger.warning("  Conversão: Regra DT sem condições, criando regra trivial")
        # Cria nó folha simples
        rule_tree.root = Node(
            attribute=attributes[0] if attributes else "dummy",
            operator='>=',
            value=float('-inf'),  # Sempre ativa (qualquer valor >= -inf)
            feature_type='numeric'
        )
        return rule_tree

    # Constrói árvore de condições em cascata (AND implícito)
    # Exemplo: [(f1, '<=', 0.5), (f2, '>', 0.3)] → (f1 <= 0.5) AND (f2 > 0.3)

    def build_and_tree(conditions: List[Tuple[str, str, float]], depth: int = 0) -> Node:
        """
        Constrói árvore AND recursivamente.

        Estratégia: Converte lista de condições em árvore binária balanceada
        usando operador AND como nós internos.
        """
        if not conditions:
            return None

        # Caso base: apenas 1 condição → folha
        if len(conditions) == 1:
            feature, operator, threshold = conditions[0]

            # Determina feature_type baseado em se está em categorical_features
            is_categorical = feature in categorical_features
            feature_type = 'categorical' if is_categorical else 'numeric'

            return Node(
                attribute=feature,
                operator=operator,
                value=threshold,
                feature_type=feature_type
            )

        # Divide condições ao meio para balancear árvore
        mid = len(conditions) // 2
        left_conditions = conditions[:mid]
        right_conditions = conditions[mid:]

        # Cria nó AND interno
        and_node = Node(
            operator='AND',
            attribute=None,
            value=None
        )

        # Constrói subárvores recursivamente
        and_node.left = build_and_tree(left_conditions, depth + 1)
        and_node.right = build_and_tree(right_conditions, depth + 1)

        return and_node

    # Constrói árvore a partir das condições
    rule_tree.root = build_and_tree(dt_rule.conditions)

    # Valida regra construída
    if not rule_tree.is_valid_rule():
        logger.warning(f"  Conversão: Regra DT convertida é inválida: {dt_rule}")
        return None

    return rule_tree


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def filter_rules_by_class(rules: List[DTRule], target_class: Any) -> List[DTRule]:
    """
    Filtra regras que predem uma classe específica.

    Args:
        rules: Lista de DTRule objects
        target_class: Classe desejada

    Returns:
        Lista de DTRule que predem target_class
    """
    filtered = [rule for rule in rules if rule.predicted_class == target_class]
    logger.debug(f"  Filtro: {len(filtered)}/{len(rules)} regras predem classe {target_class}")
    return filtered


def filter_rules_by_confidence(rules: List[DTRule], min_confidence: float = 0.5) -> List[DTRule]:
    """
    Filtra regras por confiança mínima.

    Args:
        rules: Lista de DTRule objects
        min_confidence: Confiança mínima (0-1)

    Returns:
        Lista de DTRule com confidence >= min_confidence
    """
    filtered = [rule for rule in rules if rule.confidence >= min_confidence]
    logger.debug(f"  Filtro: {len(filtered)}/{len(rules)} regras com confiança >= {min_confidence:.2f}")
    return filtered


def print_rule_statistics(rules: List[DTRule]):
    """
    Imprime estatísticas sobre lista de regras.

    Args:
        rules: Lista de DTRule objects
    """
    if not rules:
        logger.info("  Estatísticas: Nenhuma regra disponível")
        return

    logger.info(f"  Estatísticas de {len(rules)} regras:")
    logger.info(f"    Confiança média: {np.mean([r.confidence for r in rules]):.3f}")
    logger.info(f"    Confiança min/max: {np.min([r.confidence for r in rules]):.3f} / {np.max([r.confidence for r in rules]):.3f}")
    logger.info(f"    Samples médio: {np.mean([r.n_samples for r in rules]):.1f}")
    logger.info(f"    Complexidade média: {np.mean([len(r.conditions) for r in rules]):.1f} condições/regra")


# ============================================================================
# TESTE DO MÓDULO
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
    )

    logger.info("=== Teste do Módulo dt_rule_extraction.py ===")

    # Gera dataset sintético para teste
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Converte para DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    y_train_s = pd.Series(y_train)
    y_test_s = pd.Series(y_test)

    logger.info(f"\n✓ Dataset gerado: {X_train.shape[0]} treino, {X_test.shape[0]} teste")

    # Treina Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train_df, y_train_s)

    dt_acc = dt.score(X_test_df, y_test_s)
    logger.info(f"✓ Decision Tree treinada: accuracy={dt_acc:.3f}, depth={dt.get_depth()}, nodes={dt.tree_.node_count}")

    # Extrai regras
    logger.info(f"\n--- Extraindo Regras ---")
    rules = extract_rules_from_tree(dt, feature_names, classes=[0, 1])

    # Estatísticas
    print_rule_statistics(rules)

    # Ranqueia regras
    logger.info(f"\n--- Ranqueando Regras (dataset completo) ---")
    rules_ranked = rank_rules_by_error_coverage(rules, X_train_df, y_train_s)

    logger.info(f"\n--- Top-5 Regras ---")
    for i, rule in enumerate(rules_ranked[:5], 1):
        logger.info(f"  #{i}: score={rule.score:.3f}, coverage={rule.coverage:.2%}, acc={rule.accuracy:.2%}")
        logger.info(f"       {rule}")

    # Testa conversão para RuleTree
    logger.info(f"\n--- Teste de Conversão para RuleTree ---")

    value_ranges = {feat: (X_train[:, i].min(), X_train[:, i].max())
                   for i, feat in enumerate(feature_names)}

    for i, rule in enumerate(rules_ranked[:3], 1):
        logger.info(f"\n  Convertendo regra #{i}:")
        logger.info(f"    Original: {rule}")

        try:
            rule_tree = convert_dt_rule_to_ruletree(
                dt_rule=rule,
                max_depth=10,
                attributes=feature_names,
                value_ranges=value_ranges,
                category_values={},
                categorical_features=set()
            )

            if rule_tree and rule_tree.is_valid_rule():
                logger.info(f"    ✓ Conversão bem-sucedida: {rule_tree.count_nodes()} nós")
                logger.info(f"    RuleTree: {rule_tree.to_string()}")
            else:
                logger.warning(f"    ✗ Conversão falhou: regra inválida")

        except Exception as e:
            logger.error(f"    ✗ Erro na conversão: {e}")

    logger.info(f"\n=== Teste Completo ===")
