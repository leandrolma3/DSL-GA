# test_rule_extraction.py
# Script autônomo para testar a extração de regras de uma Hoeffding Tree do River.

import random
import logging
import math
from typing import List, Dict, Set, Any, Optional

# Dependências externas (necessário ter river instalado)
from river import tree, datasets
from river.tree.nodes import branch

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s')
logger = logging.getLogger("RuleExtractorTest")

# --- Bloco 1: Mini-versões das Classes e Constantes do seu Projeto ---
# Copiamos as definições essenciais aqui para que o script seja autônomo.

LOGICAL_OPERATORS = ["AND", "OR"]
NUMERIC_COMPARISON_OPERATORS = ["<", ">", "<=", ">="]
CATEGORICAL_COMPARISON_OPERATORS = ["==", "!="]
ALL_COMPARISON_OPERATORS = NUMERIC_COMPARISON_OPERATORS + CATEGORICAL_COMPARISON_OPERATORS

class Node:
    """Mini-versão da classe Node para este teste."""
    def __init__(self, attribute=None, operator=None, value=None, left=None, right=None, feature_type=None):
        self.attribute = attribute
        self.operator = operator
        self.value = value
        self.left = left
        self.right = right
        self.feature_type = feature_type

    def is_leaf(self):
        return self.attribute is not None and self.operator in ALL_COMPARISON_OPERATORS and self.left is None and self.right is None

    def is_internal(self):
        return self.operator in LOGICAL_OPERATORS and self.left is not None and self.right is not None

class RuleTree:
    """Mini-versão da classe RuleTree para este teste."""
    def __init__(self, max_depth: int, attributes: List, **kwargs):
        self.max_depth = max_depth
        self.attributes = attributes
        self.root: Optional[Node] = None

    def is_valid_rule(self):
        # Validação simplificada para o teste
        return self.root is not None

    def to_string(self):
        return self._node_to_string(self.root)

    def _node_to_string(self, node):
        if node is None: return "None"
        if node.is_leaf():
            value_str = f"{node.value:.4f}" if isinstance(node.value, (int, float)) else str(node.value)
            return f"({node.attribute} {node.operator} {value_str})"
        elif node.is_internal():
            left_expr = self._node_to_string(node.left)
            right_expr = self._node_to_string(node.right)
            return f"({left_expr} {node.operator} {right_expr})"
        return "(Unknown Node Type)"

# --- Bloco 2: A Nova Função de Extração de Regras ---

def _extract_single_rule_from_river_ht(
    ht_model: tree.HoeffdingTreeClassifier,
    target_class: Any,
    base_attributes_info: Dict
) -> Optional[RuleTree]:
    """
    Navega por um caminho em uma Hoeffding Tree do river e o traduz
    para uma única RuleTree do nosso framework.
    """
    if not hasattr(ht_model, '_root'):
        logger.error("Modelo Hoeffding Tree não parece ter um nó raiz '_root'.")
        return None

    try:
        current_node = ht_model._root
        conditions: List[Node] = []
        path_string = "ROOT"

        # 1. Navega pela árvore enquanto for um nó de divisão (galho)
        while isinstance(current_node, branch.NumericBinaryBranch):
            feature = current_node.feature
            threshold = current_node.threshold
            
            # 2. Decisão de Caminho Inteligente
            #    Escolhe o filho que tem a maior contagem de exemplos da 'target_class'.
            stats_child_0 = current_node.children[0].stats.get(target_class, 0)
            stats_child_1 = current_node.children[1].stats.get(target_class, 0)

            if stats_child_0 >= stats_child_1:
                chosen_child_index = 0
                operator = '<='
                path_string += " -> LEFT"
            else:
                chosen_child_index = 1
                operator = '>'
                path_string += " -> RIGHT"

            # 3. Cria nosso nó de condição e o adiciona à lista
            condition_node = Node(
                attribute=feature,
                operator=operator,
                value=threshold,
                feature_type='numeric' # Assumindo numérico, pois é um NumericBinaryBranch
            )
            conditions.append(condition_node)
            
            # 4. Desce para o próximo nó na árvore
            current_node = current_node.children[chosen_child_index]

        logger.info(f"Caminho percorrido na árvore: {path_string}")

        # 5. Validação da Folha Final
        # 'current_node' é agora uma folha (ex: LeafNaiveBayesAdaptive)
        leaf_prediction = max(current_node.stats, key=current_node.stats.get) if current_node.stats else None
        if leaf_prediction != target_class:
            logger.warning(f"O caminho escolhido levou a uma folha que prevê a classe {leaf_prediction}, mas buscávamos a classe {target_class}. A regra pode não ser útil.")
            # Continuamos mesmo assim, a regra ainda pode ser um bom material genético

        # 6. Monta a RuleTree final a partir da lista de condições
        if not conditions:
            logger.warning("Nenhuma condição foi extraída, a árvore é apenas uma folha.")
            return None

        # Une todas as condições com 'AND'
        if len(conditions) == 1:
            rule_root = conditions[0]
        else:
            rule_root = Node(operator="AND", left=conditions[0], right=conditions[1])
            for i in range(2, len(conditions)):
                new_root = Node(operator="AND", left=rule_root, right=conditions[i])
                rule_root = new_root

        # Cria o objeto RuleTree final
        final_rule_tree = RuleTree(max_depth=len(conditions), **base_attributes_info)
        final_rule_tree.root = rule_root

        return final_rule_tree if final_rule_tree.is_valid_rule() else None
    
    except Exception as e:
        logger.error(f"Ocorreu um erro inesperado durante a extração da regra: {e}", exc_info=True)
        return None


# --- Bloco 3: O Teste em si ---

if __name__ == "__main__":
    # 1. Treina uma Hoeffding Tree (exatamente como no teste anterior)
    logger.info("Inicializando Hoeffding Tree com parâmetros para incentivar splits...")
    ht = tree.HoeffdingTreeClassifier(grace_period=50, delta=0.01, split_criterion='info_gain')

    dataset = datasets.synth.SEA(seed=42)
    NUM_INSTANCES_TO_TRAIN = 5000
    logger.info(f"Treinando a Hoeffding Tree do river com {NUM_INSTANCES_TO_TRAIN} exemplos...")
    for x, y in dataset.take(NUM_INSTANCES_TO_TRAIN):
        ht.learn_one(x, y)
    logger.info("Treinamento concluído.")

    # 2. Prepara os argumentos para nossa nova função de extração
    target_class_to_find = 1 # Para o dataset SEA, vamos procurar regras para a classe 1
    
    # Simula o dicionário de contexto que nosso AG teria
    attributes_info = {
        "attributes": [0, 1, 2], # Nomes dos atributos do SEA
        "value_ranges": {0: (0, 10), 1: (0, 10), 2: (0, 10)},
        "category_values": {},
        "categorical_features": set()
    }

    logger.info(f"\nTentando extrair uma regra para a classe '{target_class_to_find}'...")
    
    # 3. Chama a função e analisa o resultado
    extracted_rule_tree = _extract_single_rule_from_river_ht(ht, target_class_to_find, attributes_info)

    print("\n" + "="*50)
    print("RESULTADO DA EXTRAÇÃO")
    print("="*50)
    if extracted_rule_tree:
        print("✅ Sucesso! Regra extraída:")
        print(f"   IF {extracted_rule_tree.to_string()} THEN Class {target_class_to_find}")
    else:
        print("❌ Falha. Nenhuma regra válida foi extraída.")
    print("="*50)