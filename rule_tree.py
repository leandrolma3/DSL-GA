# rule_tree.py (Implementando Uso de Tipos de Features)

import random
import logging
import math
from rule_node import Node
from typing import Dict, List, Set, Any, Union
from constants import (
    NUMERIC_COMPARISON_OPERATORS,
    CATEGORICAL_COMPARISON_OPERATORS,
    ALL_COMPARISON_OPERATORS,
    LOGICAL_OPERATORS
)

class RuleTree:
    """
    Representa uma regra de classificação como uma árvore binária.
    """
    def __init__(self, max_depth,
                 attributes, value_ranges,
                 category_values, categorical_features, 
                 root_node=None):
        """
        Inicializa uma RuleTree.
        """
        self.attributes = attributes
        self.value_ranges = value_ranges
        self.category_values = category_values
        self.categorical_features = categorical_features
        self.max_depth = max_depth
        
        # <<< NOVO: Adiciona o atributo para o score de qualidade da regra >>>
        self.quality_score = 0.0

        # --- LÓGICA DO CONSTRUTOR CORRIGIDA ---
        if root_node:
            # Se um nó raiz foi fornecido (como na clonagem), use-o.
            self.root = root_node
        else:
            # Se não, e APENAS se não, gere uma nova árvore aleatória.
            if not self.attributes:
                raise ValueError("A lista de atributos não pode estar vazia para gerar uma nova árvore.")
            try:
                self.root = self._generate_random_tree(
                    max_depth=self.max_depth,
                    attributes=self.attributes,
                    value_ranges=self.value_ranges,
                    category_values=self.category_values,
                    categorical_features=self.categorical_features
                )
            except Exception as e:
                logging.error(f"Erro durante a geração inicial da RuleTree: {e}", exc_info=True)
                raise RuntimeError("Falha ao gerar a estrutura inicial da RuleTree.") from e

        # Validação final
        if not self.is_valid_rule():
             error_msg = f"Erro Interno: a RuleTree resultante é inválida: {self.to_string()}"
             logging.critical(error_msg)
             raise RuntimeError(error_msg)
        
        
        # if root_node:
        #     self.root = root_node
        # else:
        #     if not self.attributes:
        #         raise ValueError("A lista de atributos não pode estar vazia para gerar uma nova árvore.")
        #     self.root = self._generate_random_tree(self.max_depth, self.attributes, self.value_ranges, self.category_values, self.categorical_features)

        # # Gera a árvore passando todo o contexto necessário
        # try:
        #     self.root = self._generate_random_tree(
        #         max_depth=self.max_depth,
        #         attributes=self.attributes,
        #         value_ranges=self.value_ranges,
        #         category_values=self.category_values,         # Passa adiante
        #         categorical_features=self.categorical_features # Passa adiante
        #     )
        # except Exception as e:
        #     logging.error(f"Error during initial RuleTree generation: {e}", exc_info=True)
        #     raise RuntimeError("Failed to generate initial RuleTree structure.") from e

        # # Validação final (deve passar se geração for correta)
        # if not self.is_valid_rule():
        #      error_msg = f"Internal Error: _generate_random_tree produced invalid tree: {self.to_string()}"; logging.critical(error_msg); raise RuntimeError(error_msg)

    # def _generate_random_tree(self, max_depth, attributes, value_ranges, category_values, categorical_features, current_depth=0):
    #     """ Recursively generates a random subtree, passing feature context. """
    #     if current_depth >= max_depth or not attributes:
    #         # Passa contexto para geração da folha
    #         return self._generate_leaf_node(attributes, value_ranges, category_values, categorical_features)
    #     else:
    #         # Chance de gerar folha antes da profundidade máxima
    #         if random.random() < 0.2 and current_depth > 0:
    #              return self._generate_leaf_node(attributes, value_ranges, category_values, categorical_features)

    #         operator = random.choice(LOGICAL_OPERATORS); node = Node(operator=operator)
    #         # Passa contexto nas chamadas recursivas
    #         node.left = self._generate_random_tree(max_depth, attributes, value_ranges, category_values, categorical_features, current_depth + 1)
    #         node.right = self._generate_random_tree(max_depth, attributes, value_ranges, category_values, categorical_features, current_depth + 1)
    #         if node.left is None or node.right is None:
    #              logging.error("Child node None during generation. Falling back to leaf.")
    #              return self._generate_leaf_node(attributes, value_ranges, category_values, categorical_features)
    #         return node

    def evaluate_with_debug(self, instance: Dict) -> bool:
        """Executa a avaliação da regra, imprimindo cada passo para depuração."""
        print(f"\n--- INICIANDO ANÁLISE FORENSE DA REGRA: {self.to_string()} ---")
        return self._evaluate_node_with_debug(self.root, instance, indent=0)

    def _evaluate_node_with_debug(self, node: Node, instance: Dict, indent: int) -> bool:
        """Método recursivo de avaliação com prints detalhados."""
        prefix = "  " * indent
        if node.is_leaf():
            attr = node.attribute
            instance_val = instance.get(attr)
            node_val = node.value
            op = node.operator
            
            print(f"{prefix}AVALIANDO FOLHA: ({attr} {op} {node_val})")
            print(f"{prefix}  - Valor na Instância: {instance_val}")
            
            # Lógica de avaliação idêntica à original
            result = self.evaluate_leaf_node(node, instance) # Supondo que você refatore a lógica de folha para um método
            
            print(f"{prefix}  --> RESULTADO: {result}")
            return result

        elif node.is_internal():
            print(f"{prefix}AVALIANDO NÓ INTERNO: {node.operator}")
            
            left_result = self._evaluate_node_with_debug(node.left, instance, indent + 1)
            
            # Lógica de curto-circuito
            if node.operator == "AND" and not left_result:
                print(f"{prefix}--> RESULTADO (Curto-circuito AND): False")
                return False
            if node.operator == "OR" and left_result:
                print(f"{prefix}--> RESULTADO (Curto-circuito OR): True")
                return True
                
            right_result = self._evaluate_node_with_debug(node.right, instance, indent + 1)
            
            if node.operator == "AND":
                final_result = left_result and right_result
            else: # OR
                final_result = left_result or right_result
            
            print(f"{prefix}--> RESULTADO FINAL ({node.operator}): {final_result}")
            return final_result
        return False

    def evaluate_leaf_node(self, node: Node, instance: Dict) -> bool:
        """
        Avalia um único nó folha (uma condição) em relação a uma instância, tratando os tipos de features.
        Este é um método auxiliar para evaluate() e evaluate_with_debug().
        """
        if node.attribute not in instance:
            logging.debug(f"Atributo '{node.attribute}' não encontrado na instância durante a avaliação.")
            return False

        instance_value = instance.get(node.attribute)
        node_value = node.value
        feature_type = node.feature_type

        # Trata casos onde a instância possui um valor ausente para o atributo
        if instance_value is None:
            if node.operator == '!=': return node_value is not None
            if node.operator == '==': return node_value is None
            return False # Comparações como <, > com None são Falsas

        # --- Lógica para features CATEGÓRICAS ---
        if feature_type == 'categorical':
            op = node.operator
            # Compara como strings para evitar problemas de tipo (ex: 0 vs '0')
            str_instance_val = str(instance_value)
            str_node_val = str(node_value)
            if op == "==": return str_instance_val == str_node_val
            elif op == "!=": return str_instance_val != str_node_val
            else:
                logging.warning(f"Operador inválido '{op}' para a feature categórica '{node.attribute}'.")
                return False
        
        # --- Lógica para features NUMÉRICAS ---
        elif feature_type == 'numeric':
            op = node.operator
            try:
                # Converte ambos os valores para float para uma comparação segura
                num_instance_value = float(instance_value)
                num_node_value = float(node_value)

                if op == "<": return num_instance_value < num_node_value
                elif op == ">": return num_instance_value > num_node_value
                elif op == "<=": return num_instance_value <= num_node_value
                elif op == ">=": return num_instance_value >= num_node_value
                # Trata a igualdade para floats com uma tolerância
                elif op == "==": return math.isclose(num_instance_value, num_node_value)
                elif op == "!=": return not math.isclose(num_instance_value, num_node_value)
                else:
                    logging.warning(f"Operador numérico desconhecido '{op}'.")
                    return False
            except (ValueError, TypeError):
                # Isso pode acontecer se um valor como 'N/A' estiver em uma coluna numérica
                logging.warning(f"Não foi possível converter os valores para float para a comparação numérica no atributo '{node.attribute}'. Valor da Instância: '{instance_value}', Valor do Nó: '{node_value}'.")
                return False
        
        else:
             logging.error(f"Tipo de feature desconhecido '{feature_type}' para o nó.")
             return False

    def _generate_random_tree(self, max_depth, attributes, value_ranges, category_values, categorical_features, current_depth=0):
        """
        Gera uma subárvore aleatória de forma mais estruturada, favorecendo 'AND'
        para criar regras mais específicas e menos tautológicas.
        """
        # Condição de parada: profundidade máxima ou sem atributos
        if current_depth >= max_depth or not attributes:
            return self._generate_leaf_node(attributes, value_ranges, category_values, categorical_features)
        
        # Chance de criar uma folha antes de atingir a profundidade máxima
        # Aumentamos a chance se a profundidade for > 1
        prob_leaf = 0.4 if current_depth > 0 else 0.1
        if random.random() < prob_leaf:
            return self._generate_leaf_node(attributes, value_ranges, category_values, categorical_features)

        # <<< LÓGICA DE GERAÇÃO MAIS INTELIGENTE >>>
        # Para evitar regras excessivamente genéricas, vamos favorecer o operador 'AND'.
        # O operador 'OR' só será usado com menor probabilidade.
        # Regras como (A > 1 AND B < 5) são específicas.
        # Regras como (A > 1 OR B < 5) são genéricas.
        
        if random.random() < 0.8:  # 80% de chance de ser AND
            operator = "AND"
        else:
            operator = "OR"
            
        node = Node(operator=operator)
        
        # Gera os filhos recursivamente
        node.left = self._generate_random_tree(max_depth, attributes, value_ranges, category_values, categorical_features, current_depth + 1)
        node.right = self._generate_random_tree(max_depth, attributes, value_ranges, category_values, categorical_features, current_depth + 1)
        
        # Garante que os filhos foram criados com sucesso
        if node.left is None or node.right is None:
             return self._generate_leaf_node(attributes, value_ranges, category_values, categorical_features)
             
        return node

    # # --- Geração de Folhas (LÓGICA ATUALIZADA PARA USAR TIPOS) ---
    # def _generate_leaf_node(self, attributes, value_ranges, category_values, categorical_features):
    #     """ Generates a random leaf node, choosing operator/value based on feature type. """
    #     if not attributes: logging.error("CRITICAL: _generate_leaf_node called empty attrs."); return Node(attribute="error_attr", operator="<=", value=0, feature_type='numeric')

    #     attribute = random.choice(attributes)
    #     operator, value, feature_type = None, None, None

    #     # --- LÓGICA DE DECISÃO BASEADA NO TIPO ---
    #     if attribute in categorical_features:
    #         feature_type = 'categorical'
    #         operator = random.choice(CATEGORICAL_COMPARISON_OPERATORS) # <<<--- Usa operadores categóricos
    #         possible_values = category_values.get(attribute)
    #         # Garante que o conjunto não está vazio ou contém apenas None
    #         valid_cat_values = [v for v in possible_values if v is not None] if possible_values else []
    #         if valid_cat_values:
    #             value = random.choice(valid_cat_values) # <<<--- Escolhe um valor da categoria
    #         else: # Fallback se não houver valores válidos conhecidos
    #             logging.warning(f"No valid possible values for categorical attr '{attribute}'. Using default 'UNKNOWN_CAT'.")
    #             value = "UNKNOWN_CAT"
    #             operator = "!=" # Força != para ter alguma chance de ser útil
    #     else: # Assume numérico
    #         feature_type = 'numeric'
    #         operator = random.choice(NUMERIC_COMPARISON_OPERATORS) # <<<--- Usa operadores numéricos
    #         min_val, max_val = value_ranges.get(attribute, (0, 0))
    #         if min_val >= max_val: value = min_val
    #         else: value = random.uniform(min_val, max_val)
    #     # -----------------------------------------

    #     leaf_node = Node(attribute=attribute, operator=operator, value=value, feature_type=feature_type)

    #     # Validação da estrutura gerada
    #     is_node_valid = (leaf_node.attribute is not None and
    #                      leaf_node.operator in ALL_COMPARISON_OPERATORS and # Checa se operador é válido no geral
    #                      leaf_node.value is not None and # Valor não pode ser None aqui
    #                      leaf_node.feature_type in ['numeric', 'categorical'])
    #     # Validação adicional: operador é compatível com o tipo?
    #     if feature_type == 'numeric' and operator not in NUMERIC_COMPARISON_OPERATORS:
    #          logging.error(f"Internal inconsistency: Numeric feature '{attribute}' assigned categorical operator '{operator}'. Fixing.")
    #          operator = random.choice(NUMERIC_COMPARISON_OPERATORS) # Corrige operador
    #          leaf_node.operator = operator # Atualiza nó
    #          is_node_valid = False # Sinaliza para log crítico abaixo se necessário, embora corrigido
    #     elif feature_type == 'categorical' and operator not in CATEGORICAL_COMPARISON_OPERATORS:
    #          logging.error(f"Internal inconsistency: Categorical feature '{attribute}' assigned numeric operator '{operator}'. Fixing.")
    #          operator = random.choice(CATEGORICAL_COMPARISON_OPERATORS) # Corrige operador
    #          leaf_node.operator = operator # Atualiza nó
    #          is_node_valid = False # Sinaliza para log crítico abaixo se necessário, embora corrigido

    #     if not is_node_valid:
    #          # Se mesmo após a correção acima ainda for inválido (não deveria), loga crítico
    #          logging.critical(f"Internal Error: Generated leaf node still invalid after checks: {self._node_to_string(leaf_node)}")
    #          # Fallback extremo
    #          return Node(attribute=attribute, operator="<=", value=0, feature_type='numeric')

    #     return leaf_node
    def _generate_leaf_node(self, attributes, value_ranges, category_values, categorical_features):
        """ Gera uma folha aleatória, garantindo que o range de valor seja válido. """
        if not attributes:
            # Fallback de segurança se for chamado sem atributos
            return Node(attribute="error_attr", operator="<=", value=0, feature_type='numeric')

        attribute = random.choice(attributes)
        
        # --- Lógica de decisão baseada no tipo ---
        if attribute in categorical_features:
            feature_type = 'categorical'
            operator = random.choice(CATEGORICAL_COMPARISON_OPERATORS)
            possible_values = list(category_values.get(attribute, []))
            
            if possible_values:
                value = random.choice(possible_values)
            else: # Fallback se não houver valores
                value = "UNKNOWN_CAT"
                operator = "!="
        else: # Assume numérico
            feature_type = 'numeric'
            operator = random.choice(NUMERIC_COMPARISON_OPERATORS)
            min_val, max_val = value_ranges.get(attribute, (0, 0))

            # <<< MUDANÇA IMPORTANTE: Evita ranges inválidos ou muito pequenos >>>
            if (max_val - min_val) < 1e-6: # Se o range for praticamente zero
                value = min_val
            else:
                # Gera um valor que não seja exatamente o mínimo ou máximo para criar alguma seletividade
                value = random.uniform(min_val + 0.1 * (max_val - min_val), 
                                       max_val - 0.1 * (max_val - min_val))
        
        return Node(attribute=attribute, operator=operator, value=value, feature_type=feature_type)

    # --- Validação ---
    def _is_valid_node(self, node):
        """ Recursively checks node validity, including feature type and operator compatibility. """
        if node is None: return False
        if node.is_leaf():
            # Folha válida tem atributo, operador, valor e tipo consistentes
            is_struct_valid = (node.attribute is not None and node.operator in ALL_COMPARISON_OPERATORS and node.value is not None and node.feature_type in ['numeric', 'categorical'])
            if not is_struct_valid: return False
            # Verifica compatibilidade operador/tipo
            if node.feature_type == 'numeric' and node.operator not in NUMERIC_COMPARISON_OPERATORS: return False
            if node.feature_type == 'categorical' and node.operator not in CATEGORICAL_COMPARISON_OPERATORS: return False
            return True # Se passou em tudo
        elif node.is_internal():
            # Interno válido tem operador lógico e DOIS filhos VÁLIDOS
            return (node.operator in LOGICAL_OPERATORS and self._is_valid_node(node.left) and self._is_valid_node(node.right))
        else: return False # Não é folha nem interno

    def is_valid_rule(self):
        if self.root is None: return False
        return self._is_valid_node(self.root)

    # --- Avaliação (Atualizada para Tipos) ---
    def evaluate(self, instance):
        """ Evaluates the rule tree for an instance, handling feature types. """
        try: return self._evaluate_node(self.root, instance)
        except Exception as e: logging.error(f"Eval exception: {e}. Rule: {self.to_string()}", exc_info=False); return False # Log simplificado

    # --- AVALIAÇÃO DE FOLHA ATUALIZADA ---
    def _evaluate_node(self, node, instance):
        """ Recursively evaluates a node, handling numeric/categorical types in leaves. """
        if node is None: raise ValueError("Evaluation encountered None node.")

        if node.is_leaf():
            if node.attribute not in instance: logging.debug(f"Attr '{node.attribute}' not in instance."); return False
            instance_value = instance[node.attribute]; node_value = node.value; feature_type = node.feature_type

            # Trata instância com valor None
            if instance_value is None:
                 # Apenas comparações de (des)igualdade com None são significativas
                 if node.operator == '!=': return node.value is not None
                 if node.operator == '==': return node.value is None
                 return False # <, >, <=, >= com None são falsos

            # --- LÓGICA DE COMPARAÇÃO BASEADA NO TIPO ---
            if feature_type == 'categorical':
                op = node.operator
                # Compara como strings para evitar problemas de tipo (ex: 0 vs '0')
                str_instance_val = str(instance_value)
                str_node_val = str(node_value)
                if op == "==": return str_instance_val == str_node_val
                elif op == "!=": return str_instance_val != str_node_val
                else:
                    # Este warning não deveria mais ocorrer se a geração/mutação for correta
                    logging.warning(f"Invalid operator '{op}' during eval for categorical '{node.attribute}'.")
                    return False
            elif feature_type == 'numeric':
                op = node.operator
                # Operadores == e != devem ser tratados primeiro para evitar erro de float()
                if op == "==":
                    try: return math.isclose(float(instance_value), float(node_value))
                    except (ValueError, TypeError): return False # Comparação impossível
                elif op == "!=":
                    try: return not math.isclose(float(instance_value), float(node_value))
                    except (ValueError, TypeError): return True # Se não podem ser comparados como float, são diferentes
                # Tenta conversão para operadores numéricos restantes
                try:
                    num_instance_value = float(instance_value); num_node_value = float(node_value)
                    if op == "<": return num_instance_value < num_node_value
                    elif op == ">": return num_instance_value > num_node_value
                    elif op == "<=": return num_instance_value <= num_node_value
                    elif op == ">=": return num_instance_value >= num_node_value
                    else: logging.warning(f"Unknown numeric operator '{op}'."); return False
                except (ValueError, TypeError) as e:
                    logging.warning(f"Numeric eval failed for attr '{node.attribute}': {e}. Inst='{instance_value}', Node='{node_value}'."); return False
            else:
                 logging.error(f"Unknown feature_type '{feature_type}' for node: {self._node_to_string(node)}"); return False
            # -------------------------------------------

        elif node.is_internal():
             # Avaliação de nó interno (lógica como antes)
            left_result = self._evaluate_node(node.left, instance); op = node.operator
            if op == "AND": return left_result and self._evaluate_node(node.right, instance)
            elif op == "OR": return left_result or self._evaluate_node(node.right, instance)
            else: logging.error(f"Unknown logical operator: {op}"); return False
        else: raise ValueError("Node is neither leaf nor internal.")

    # --- Métodos de Contagem e Profundidade ---
    def count_nodes(self): return self._count_nodes_recursive(self.root)
    def _count_nodes_recursive(self, node):
        if node is None: return 0
        return 1 + self._count_nodes_recursive(node.left) + self._count_nodes_recursive(node.right)
    def get_depth(self): return self._get_depth_recursive(self.root)
    def _get_depth_recursive(self, node):
         if node is None: return 0;
         if node.is_leaf(): return 1
         return 1 + max(self._get_depth_recursive(node.left), self._get_depth_recursive(node.right))

    # --- Simplificação (Atualizada para Tipos) ---
    def simplify(self):
        """ Attempts to simplify the rule tree using various rules, respecting feature types. """
        if self.root is None: return
        new_root = self._simplify_node(self.root)
        if new_root is not None and self._is_valid_node(new_root): self.root = new_root
        elif new_root is None: logging.warning(f"Simplification resulted in None root. Reverting for: {self._node_to_string(self.root)}")
        else: logging.warning(f"Simplification resulted in invalid root. Reverting.")

    def _simplify_node(self, node):
        """ Recursively simplifies subtree, checking feature types. """
        if node is None or node.is_leaf(): return node
        node.left = self._simplify_node(node.left); node.right = self._simplify_node(node.right)
        # Reparo local...
        if node.operator == "AND":
            if node.left is None or node.right is None: return None
        elif node.operator == "OR":
            if node.left is None and node.right is None: return None
            elif node.left is None: return node.right
            elif node.right is None: return node.left

        # --- Regra 1: Redundância Direta ---
        if node.left and node.right and self._nodes_equal(node.left, node.right): logging.debug(f"Simplify Rule 1 (X op X): ..."); return node.left

        # --- Regra 2: Absorção ---
        op = node.operator; left = node.left; right = node.right
        if op == "AND":
             if right.is_internal() and right.operator == "OR" and (self._nodes_equal(left, right.left) or self._nodes_equal(left, right.right)): return left
             if left.is_internal() and left.operator == "OR" and (self._nodes_equal(right, left.left) or self._nodes_equal(right, left.right)): return right
        elif op == "OR":
             if right.is_internal() and right.operator == "AND" and (self._nodes_equal(left, right.left) or self._nodes_equal(left, right.right)): return left
             if left.is_internal() and left.operator == "AND" and (self._nodes_equal(right, left.left) or self._nodes_equal(right, left.right)): return right

        # --- Regra 3: Intervalo Numérico (Verifica Tipo) ---
        # Garante que filhos existem e são folhas ANTES de acessar atributos
        if (node.left and node.right and
            node.left.is_leaf() and node.right.is_leaf() and
            node.left.attribute == node.right.attribute and
            # <<<--- USA node.feature_type ARMAZENADO --->>>
            node.left.feature_type == 'numeric'): # Implícito que node.right também é

            attr = node.left.attribute; op1, v1 = node.left.operator, node.left.value; op2, v2 = node.right.operator, node.right.value; parent_op = node.operator; simplified_node = None
            try:
                f_v1 = float(v1); f_v2 = float(v2)
                # ... (lógica de comparação numérica como antes) ...
                if parent_op == "AND":
                    if op1 in ["<", "<="] and op2 in ["<", "<="]: new_v = min(f_v1, f_v2); new_op = "<" if op1 == "<" or op2 == "<" else "<="; simplified_node = Node(attribute=attr, operator=new_op, value=new_v, feature_type='numeric')
                    elif op1 in [">", ">="] and op2 in [">", ">="]: new_v = max(f_v1, f_v2); new_op = ">" if op1 == ">" or op2 == ">" else ">="; simplified_node = Node(attribute=attr, operator=new_op, value=new_v, feature_type='numeric')
                elif parent_op == "OR":
                    if op1 in ["<", "<="] and op2 in ["<", "<="]: new_v = max(f_v1, f_v2); new_op = "<=" if op1 == "<=" or op2 == "<=" else "<"; simplified_node = Node(attribute=attr, operator=new_op, value=new_v, feature_type='numeric')
                    elif op1 in [">", ">="] and op2 in [">", ">="]: new_v = min(f_v1, f_v2); new_op = ">=" if op1 == ">=" or op2 == ">=" else ">"; simplified_node = Node(attribute=attr, operator=new_op, value=new_v, feature_type='numeric')

                if simplified_node: logging.debug(f"Simplify Rule 3 (Numeric): {self._node_to_string(node)} -> {self._node_to_string(simplified_node)}"); return simplified_node
            except (ValueError, TypeError): pass # Ignora se não for numérico

        return node

    # --- Outras Funções ---
    def _nodes_equal(self, node1, node2):
        if node1 is None and node2 is None: return True
        if node1 is None or node2 is None: return False
        return self._node_to_string(node1) == self._node_to_string(node2)
    def to_string(self):
            return self._node_to_string(self.root)

    def _node_to_string(self, node):
        if node is None:
            return "None" # Handles case where the root or a child is None

        if node.is_leaf():
            # Formats leaf nodes: (attribute operator value)
            # Handles numeric vs categorical formatting for value
            value_str = f"{node.value:.4f}" if isinstance(node.value, (int, float)) and node.feature_type=='numeric' else str(node.value)
            # Ensures attribute and operator are included
            return f"({node.attribute} {node.operator} {value_str})"

        elif node.is_internal():
            # Recursively formats internal nodes: (left_expr operator right_expr)
            left_expr = self._node_to_string(node.left)
            right_expr = self._node_to_string(node.right)
            # Includes the operator and parentheses
            return f"({left_expr} {node.operator} {right_expr})"
        else:
            # Fallback for unexpected node types
            return "(Unknown Node Type)"