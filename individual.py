# individual.py (Modificado para info categórica e coleta de ops/vals)

import random
import copy
import logging
import numpy as np
# Removido: from sklearn.metrics import accuracy_score (agora em fitness.py)
from typing import Dict, List, Set, Any, Union # Adicione Union
from collections import Counter
# Importa RuleTree e constantes (operadores)
from rule_tree import RuleTree
from constants import LOGICAL_OPERATORS, ALL_COMPARISON_OPERATORS # Importa novas listas se necessário

# Helper functions (get_tree_depth, compute_features_distance) movidas para utils.py/fitness.py

class Individual:
    """
    Represents a candidate solution in the GA population.
    Consists of rules (RuleTrees) and handles fitness calculation context.
    Now includes awareness of categorical features.
    """
    def __init__(self, max_rules_per_class, max_depth,
                 attributes, value_ranges, # Ranges para numéricas
                 category_values, categorical_features, # <<<--- NOVOS PARÂMETROS
                 classes, train_target=None, initialize_random_rules: bool = True):
        """
        Initializes a new Individual with randomly generated rules.

        Args:
            max_rules_per_class (int): Max rules per class label.
            max_depth (int): Max depth for each RuleTree.
            attributes (list): List of ALL attribute names.
            value_ranges (dict): Dict {attr: (min, max)} for NUMERIC attributes.
            category_values (dict): Dict {cat_attr: set(values)} for CATEGORICAL attributes.
            categorical_features (set): Set of categorical attribute names.
            classes (list): List of unique class labels.
            train_target (list, optional): Target values for determining default class.
        """
        self.COVERAGE_PRECISION_THRESHOLD = 0.10
        self.max_rules_per_class = max_rules_per_class
        self.max_depth = max_depth
        self.gmean = 0.0
        # Armazena as informações sobre features
        self.attributes = attributes
        self.value_ranges = value_ranges
        self.category_values = category_values
        self.categorical_features = categorical_features
        # ---
        self.classes = classes
        self.fitness = 0.0
        self.creation_chunk_index = -1
        self.rules = {class_label: [] for class_label in self.classes}
        self.covered_classes: Set[Any] = set()


        # # Determina classe padrão
        # if train_target and len(train_target) > 0:
        #     unique_classes, counts = np.unique(train_target, return_counts=True)
        #     self.default_class = unique_classes[np.argmax(counts)]
        # elif self.classes:
        #     self.default_class = random.choice(self.classes)
        # else:
        #     self.default_class = None
        #     logging.warning("No classes provided for default class setting.")

        if train_target and len(train_target) > 0:
            self.default_class = Counter(train_target).most_common(1)[0][0]
        elif self.classes:
            self.default_class = random.choice(self.classes)
        else:
            self.default_class = None

        # --- MUDANÇA 2: Geração de regras aleatórias agora é condicional ---
        if initialize_random_rules:
            if not self.attributes:
                logging.warning("Individual set to initialize with random rules, but no attributes provided.")
            else:
                for class_label in self.classes:
                    # <<< CORREÇÃO SUTIL: O número de regras deve ser no máximo o permitido >>>
                    num_rules = random.randint(1, self.max_rules_per_class)
                    for _ in range(num_rules):
                        try:
                            rule_tree = RuleTree(max_depth=self.max_depth, attributes=self.attributes, value_ranges=self.value_ranges, category_values=self.category_values, categorical_features=self.categorical_features)
                            if rule_tree.is_valid_rule():
                                self.rules[class_label].append(rule_tree)
                        except (ValueError, RuntimeError) as e:
                            logging.error(f"Error generating RuleTree for class {class_label}: {e}.")
                            break
            self.remove_duplicate_rules()

        # # Gera regras iniciais aleatórias
        # if not self.attributes:
        #     logging.warning(f"Individual initialized with no attributes. No rules generated.")
        # else:
        #     for class_label in self.classes:
        #         num_rules = random.randint(1, max(1, self.max_rules_per_class))
        #         for _ in range(num_rules):
        #             try:
        #                 # >>>>> CHAMA RuleTree COM NOVOS PARÂMETROS <<<<<
        #                 # Assume que RuleTree.__init__ será atualizado para aceitá-los
        #                 rule_tree = RuleTree(
        #                     max_depth=self.max_depth,
        #                     attributes=self.attributes,
        #                     value_ranges=self.value_ranges,
        #                     category_values=self.category_values, # Passa valores categóricos
        #                     categorical_features=self.categorical_features # Passa nomes categóricos
        #                 )
        #                 # Adiciona apenas se for válida (RuleTree agora garante isso na construção)
        #                 # A verificação aqui é uma dupla checagem opcional.
        #                 if rule_tree.is_valid_rule():
        #                      self.rules[class_label].append(rule_tree)
        #                 else:
        #                      # Isso não deveria acontecer se RuleTree.__init__ for robusto
        #                      logging.error(f"CRITICAL: RuleTree constructor failed validation for class {class_label}. Skipping rule.")
        #             except (ValueError, RuntimeError) as e:
        #                  # Captura erros da construção de RuleTree (ex: atributos vazios, falha interna)
        #                  logging.error(f"Error generating RuleTree for class {class_label}: {e}. Stopping rule generation for this class.")
        #                  break # Para de gerar regras para esta classe se houver erro fundamental


        #self.remove_duplicate_rules() # Mantém limpeza de duplicatas

    def update_rule_quality_scores(self, data: List[Dict], target: List[Any]) -> None:
            """
            Calcula a precisão de cada regra individualmente, atualiza o 'quality_score'
            em cada RuleTree e, ao final, povoa o conjunto 'covered_classes'.
            """
            if not data or not target:
                return

            # Primeiro, calcula a precisão para cada regra individual
            for class_label, rule_list in self.rules.items():
                for rule in rule_list:
                    activations = 0
                    correct_activations = 0
                    
                    for i, instance in enumerate(data):
                        try:
                            if rule.evaluate(instance):
                                activations += 1
                                if target[i] == class_label:
                                    correct_activations += 1
                        except Exception:
                            continue
                    
                    precision = (correct_activations / activations) if activations > 0 else 0.0
                    rule.quality_score = precision
            
            # <<< NOVO: Após avaliar todas as regras, identifica as classes cobertas >>>
            self.covered_classes.clear() # Limpa o conjunto para a nova avaliação
            for class_label, rule_list in self.rules.items():
                # Uma classe é considerada "coberta" se PELO MENOS UMA de suas regras
                # tem uma qualidade (precisão) acima do nosso limiar.
                if any(rule.quality_score > self.COVERAGE_PRECISION_THRESHOLD for rule in rule_list):
                    self.covered_classes.add(class_label)
            
            logging.debug(f"Análise de cobertura do indivíduo concluída. Classes cobertas: {self.covered_classes}")


    def get_rules_as_string(self) -> str:
        """
        Concatena as representações em string de todas as regras do indivíduo
        para uma representação única. As regras são ordenadas para consistência.
        """
        full_str_parts = []
        # Ordena por classe e depois por regra para garantir uma representação estável
        for class_label in sorted(self.rules.keys()):
            rule_list = self.rules[class_label]
            for rule_tree in sorted(rule_list, key=lambda r: r.to_string()):
                full_str_parts.append(f"C{class_label}:{rule_tree.to_string()}")
        
        return " | ".join(full_str_parts)
    # --- Avaliação e Predição ---

    # calculate_fitness foi movido para fitness.py

# Em individual.py, substitua AMBOS os métodos de predição por estes.
# Em individual.py - Melhorar predição
    # def _predict(self, instance):
    #     predictions = []
        
    #     for class_label in self.rules:
    #         for rule in self.rules[class_label]:
    #             if rule.evaluate(instance):
    #                 predictions.append(class_label)
    #                 break
        
    #     if not predictions:
    #         return self.default_class
     
    #     # Retornar classe mais frequente no conjunto de treinamento
    #     return max(set(predictions), key=predictions.count)
   
    # Retornar classe mais frequente no conjunto de treinamento
    # return max(set(predictions), key=predictions.count)

    def _predict(self, instance):
        """
        Prevê o rótulo da classe para uma única instância usando a lógica de desempate por especificidade.
        """
        prediction, _ = self._predict_with_activation_check(instance)
        return prediction

    def _predict_with_activation_check(self, instance: Dict) -> tuple[Any, bool]:
        """
        Avalia as regras e retorna a predição e se uma regra foi ativada.
        O desempate é feito pela regra mais específica (maior número de nós).
        """
        # <<< LÓGICA DE DESEMPATE COMPLETAMENTE REFEITA >>>
        
        # Em vez de apenas a classe, armazenamos uma tupla: (classe, especificidade)
        activated_rules_info = []

        for class_label in sorted(self.rules.keys()):
            rule_list = self.rules.get(class_label, [])
            for rule in rule_list:
                try:
                    if rule.evaluate(instance):
                        # Mede a especificidade da regra (número de nós)
                        specificity = rule.count_nodes()
                        activated_rules_info.append((class_label, specificity))
                        # Continua para a próxima classe
                        break 
                except Exception as e:
                    logging.error(f"Error evaluating rule {rule.to_string()} for class {class_label}: {e}", exc_info=True)
                    continue

        # Se uma ou mais regras foram ativadas
        if activated_rules_info:
            rule_was_activated = True
            
            # --- A NOVA LÓGICA DE DESEMPATE ---
            # Encontra a regra com a maior especificidade (mais nós)
            best_rule = max(activated_rules_info, key=lambda item: item[1])
            prediction = best_rule[0] # A predição é a classe da regra mais específica
            # -----------------------------------

        # Se nenhuma regra foi ativada, usa a classe padrão
        else:
            rule_was_activated = False
            prediction = self.default_class
            
        return prediction, rule_was_activated

    # def _predict(self, instance):
    #     """
    #     Predicts the class label for a single instance using the individual's rules.
    #     (A lógica interna depende de rule.evaluate(), que será modificada em RuleTree)
    #     """
    #     # ... (lógica de predição como antes - sem alterações aqui) ...
    #     activated_classes = []
    #     sorted_classes = sorted(self.rules.keys())
    #     for class_label in sorted_classes:
    #         rule_list = self.rules.get(class_label, []) # Use get for safety
    #         for rule in rule_list:
    #             try:
    #                 # A chamada evaluate() permanece a mesma aqui, mas a implementação
    #                 # interna em RuleTree será modificada para lidar com tipos.
    #                 if rule.evaluate(instance):
    #                     activated_classes.append(class_label)

    #                     break # Assume primeira regra que dispara vence para a classe
    #             except Exception as e:
    #                 logging.error(f"Error evaluating rule {rule.to_string()} for class {class_label}: {e}", exc_info=True)
    #                 continue # Pula regra com erro

    #     if len(activated_classes) == 1: return activated_classes[0]
    #     elif len(activated_classes) > 1:
    #         logging.debug(f"Tie between classes: {activated_classes}. Resolving with min().")
    #         try: return min(activated_classes)
    #         except TypeError: return self.default_class # Fallback se classes não comparáveis
    #     else: return self.default_class

    # --- Coleta de Informações das Regras ---

    def get_used_attributes(self):
        """Collects the set of unique attributes used in the leaf nodes of all rules."""
        # ... (lógica como antes - sem alterações) ...
        used_attributes = set()
        for rule_list in self.rules.values():
            for rule_tree in rule_list:
                self._collect_attributes_recursive(rule_tree.root, used_attributes)
        return used_attributes

    def _collect_attributes_recursive(self, node, used_attributes_set):
        """Recursively traverses a subtree to collect attributes from leaf nodes."""
        # ... (lógica como antes - sem alterações) ...
        if node is None: return
        if node.is_leaf():
            if node.attribute is not None: used_attributes_set.add(node.attribute)
        elif node.is_internal():
            self._collect_attributes_recursive(node.left, used_attributes_set)
            self._collect_attributes_recursive(node.right, used_attributes_set)


    # --- Coleta de Operadores e Valores (Atualizada) ---

    def _collect_ops_and_thresholds(self):
        """
        Collects operators and distinguishes numeric thresholds from categorical values.

        Returns:
            tuple: (list_logical_ops, list_comparison_ops, list_numeric_thresholds, list_categorical_values)
        """
        logical_ops = []
        comparison_ops = []
        numeric_thresholds = []
        categorical_values_used = [] # Guarda os valores usados em comparações categóricas

        for rule_list in self.rules.values():
            for rule_tree in rule_list:
                # Passa o set de features categóricas para o helper recursivo
                self._collect_ops_and_thresholds_node(
                    rule_tree.root, logical_ops, comparison_ops,
                    numeric_thresholds, categorical_values_used, # Passa as novas listas
                    self.categorical_features # Passa o set de nomes categóricos
                )

        return logical_ops, comparison_ops, numeric_thresholds, categorical_values_used

    def _collect_ops_and_thresholds_node(self, node, logical_ops, comparison_ops,
                                          numeric_thresholds, categorical_values_used, # Novas listas
                                          categorical_features): # Set de nomes categóricos
        """
        Recursively traverses a node to collect operators, numeric thresholds,
        and categorical values.

        Args:
            node (Node): Current node.
            logical_ops (list): List to append logical operators.
            comparison_ops (list): List to append comparison operators (both types).
            numeric_thresholds (list): List to append numeric threshold values.
            categorical_values_used (list): List to append categorical values used.
            categorical_features (set): Set of categorical attribute names.
        """
        if node is None:
            return

        if node.is_leaf():
            # Adiciona o operador de comparação usado (==, !=, <, >, etc.)
            if node.operator in ALL_COMPARISON_OPERATORS: # Usa lista completa aqui
                 comparison_ops.append(node.operator)

            # Verifica se o atributo é categórico ou numérico
            if node.attribute in categorical_features:
                if node.value is not None:
                    categorical_values_used.append(node.value) # Adiciona valor à lista categórica
            else: # Assume numérico
                if node.value is not None:
                    try:
                        # Tenta converter para float para garantir que é numérico
                        numeric_val = float(node.value)
                        numeric_thresholds.append(numeric_val) # Adiciona à lista numérica
                    except (ValueError, TypeError):
                        # Se não for conversível, loga mas não adiciona a nenhuma lista
                        logging.warning(f"Leaf node for presumed numeric attribute '{node.attribute}' has non-numeric value: '{node.value}'. Ignoring for threshold stats.")

        elif node.is_internal():
            if node.operator in LOGICAL_OPERATORS:
                logical_ops.append(node.operator)
            # Recurse passando todas as listas e o set de features categóricas
            self._collect_ops_and_thresholds_node(node.left, logical_ops, comparison_ops, numeric_thresholds, categorical_values_used, categorical_features)
            self._collect_ops_and_thresholds_node(node.right, logical_ops, comparison_ops, numeric_thresholds, categorical_values_used, categorical_features)


    # --- Outros Métodos ---

    def evaluate_rule_performance(self, data, target):
        """Evaluates the performance (accuracy) of each individual rule tree."""
        # ... (lógica como antes - sem alterações aqui) ...
        rule_performance = {};
        if not data: return rule_performance
        for class_label, rule_list in self.rules.items():
            for idx, rule in enumerate(rule_list):
                rule_fired_count = 0; rule_correct_count = 0
                for instance, true_label in zip(data, target):
                    try:
                        prediction_fires = rule.evaluate(instance)
                        if prediction_fires:
                            rule_fired_count += 1
                            if true_label == class_label: rule_correct_count += 1
                    except Exception as e: logging.error(f"Error evaluating rule {idx} class {class_label} for perf check: {e}"); continue
                accuracy = rule_correct_count / (rule_fired_count + 1e-9)
                rule_performance[(class_label, idx)] = accuracy
        return rule_performance


    def remove_duplicate_rules(self):
        """Removes duplicate rules within each class list based on string representation."""
        # ... (lógica como antes - sem alterações) ...
        for class_label in self.rules:
            unique_rules = []; seen_rule_strings = set()
            for rule in self.rules[class_label]:
                rule_str = rule.to_string()
                if rule_str not in seen_rule_strings:
                    seen_rule_strings.add(rule_str); unique_rules.append(rule)
                else: logging.debug(f"Removing duplicate rule class {class_label}: {rule_str}")
            self.rules[class_label] = unique_rules


    def count_total_rules(self):
        """Counts the total number of rules across all classes."""
        # ... (lógica como antes - sem alterações) ...
        return sum(len(rule_list) for rule_list in self.rules.values())

    def count_total_nodes(self):
         """Counts the total number of nodes across all rules."""
         # ... (lógica como antes - sem alterações) ...
         return sum(rt.count_nodes() for rlist in self.rules.values() for rt in rlist)

    def count_total_conditionals(self):
        """Counts the total number of conditions (leaf nodes) across all rules."""
        # ... (lógica como antes, usando count_total_nodes) ...
        return self.count_total_nodes()

    def __str__(self):
        """Provides a string summary of the individual."""
        # ... (lógica como antes - sem alterações) ...
        rule_summary = [];
        for class_label, rule_list in self.rules.items():
             rule_summary.append(f"  Class {class_label}: {len(rule_list)} rules")
        rules_str = "\n".join(rule_summary)
        return (f"Individual:\n" f"  Fitness: {self.fitness:.4f}\n" f"  Default Class: {self.default_class}\n" f"  Rules:\n{rules_str}")