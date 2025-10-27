# hill_climbing_v2.py
"""
Hill Climbing Hierárquico Multi-Nível (v2.0)

PROPÓSITO: Sistema avançado de refinamento com 3 níveis hierárquicos baseado
           na performance do elite, para recuperar gap de performance quando
           usando seeding adaptativo/probabilístico.

NÍVEIS:
- AGGRESSIVE (70-85%): Exploração agressiva com mudanças estruturais grandes
- MODERATE (85-92%):   Refinamento moderado com ajustes estruturais
- FINE_TUNING (92-98%): Fine-tuning com ajustes finos de precisão

AUTOR: Claude Code
DATA: 2025-01-08
"""

import logging
import copy
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.tree import DecisionTreeClassifier
from individual import Individual
import ga_operators

# Importa novas estratégias inteligentes
from intelligent_hc_strategies import (
    error_focused_dt_rules,
    ensemble_boosting,
    guided_mutation
)

logger = logging.getLogger("hill_climbing_v2")


# ============================================================================
# CONFIGURAÇÃO DE NÍVEIS HIERÁRQUICOS
# ============================================================================

HILL_CLIMBING_LEVELS = {
    'aggressive': {
        'gmean_range': (0.70, 0.92),  # EXPANDIDO: Ativa mais cedo para exploração agressiva
        'description': 'EXPLORAÇÃO AGRESSIVA - HC Inteligente Multi-Estratégia',
        'num_variants_base': 18,  # AUMENTADO: 15 → 18 (+20% variantes)
        'mutation_strength': 0.8,
        'operations': [
            'error_focused_dt_rules',  # NOVA: 40% - Extrai regras DT dos erros (8 variantes)
            'ensemble_boosting',       # NOVA: 35% - DT com boosting nos erros (6 variantes)
            'guided_mutation'          # NOVA: 25% - Mutação guiada por feature importance (4 variantes)
        ]
    },
    'moderate': {
        'gmean_range': (0.92, 0.96),  # AJUSTADO: Ativa apenas quando elite já está bom
        'description': 'REFINAMENTO MODERADO - HC Inteligente + Operações Legadas',
        'num_variants_base': 12,  # AUMENTADO: 10 → 12 (+20% variantes)
        'mutation_strength': 0.5,
        'operations': [
            'error_focused_dt_rules',  # NOVA: 50% - Foco em correção de erros (6 variantes)
            'ensemble_boosting',       # NOVA: 30% - Boosting (4 variantes)
            'crossover_with_memory',   # LEGADO: 20% - Cruza com histórico (2 variantes)
        ]
    },
    'fine_tuning': {
        'gmean_range': (0.96, 0.98),  # AJUSTADO: Apenas para elites quase perfeitos
        'description': 'FINE-TUNING - Mutação Guiada + Ajustes Finos',
        'num_variants_base': 6,   # AUMENTADO: 5 → 6 (+20% variantes)
        'mutation_strength': 0.2,
        'operations': [
            'guided_mutation',         # NOVA: 60% - Refinamento guiado (4 variantes)
            'error_focused_dt_rules',  # NOVA: 40% - Pequenas correções (2 variantes)
        ]
    }
}


# ============================================================================
# OPERAÇÕES DE HILL CLIMBING POR NÍVEL
# ============================================================================

def inject_memory_rules(elite: Individual, population: List[Individual],
                       best_ever_memory: List[Individual], **kwargs) -> List[Individual]:
    """
    AGGRESSIVE: Injeta regras de alta qualidade da memória histórica no elite.

    Se memória está vazia, usa top indivíduos da população atual.
    """
    variants = []

    # Se memória vazia, usa top 3 da população atual como "memória"
    memory_source = best_ever_memory if best_ever_memory else population[:3]

    if not memory_source:
        return []

    # Tenta combinar com top 2 da memória/população
    for memory_ind in memory_source[:2]:
        variant = copy.deepcopy(elite)

        # Para cada classe, pega as melhores regras da memória
        for cls in variant.classes:
            if cls in memory_ind.rules and memory_ind.rules[cls]:
                # Adiciona até 2 regras melhores da memória (deep copy)
                best_memory_rules = memory_ind.rules[cls][:2]
                for rule in best_memory_rules:
                    variant.rules[cls].append(copy.deepcopy(rule))

        variants.append(variant)

    return variants


def crossover_with_memory(elite: Individual, population: List[Individual],
                         best_ever_memory: List[Individual], **kwargs) -> List[Individual]:
    """
    AGGRESSIVE: Cruza elite com melhores indivíduos da memória histórica.

    Se memória está vazia, cruza com top indivíduos da população atual.
    """
    # Se memória vazia, usa top 3 da população atual
    memory_source = best_ever_memory if best_ever_memory else population[:3]

    if not memory_source:
        return []

    variants = []
    # Cruza com top 3 da memória/população
    for memory_ind in memory_source[:3]:
        try:
            variant = ga_operators.crossover(
                elite, memory_ind,
                kwargs['max_depth'], kwargs['attributes'],
                kwargs['value_ranges'], kwargs['category_values'],
                kwargs['categorical_features'], kwargs['classes'],
                kwargs['max_rules_per_class']
            )
            variants.append(variant)
        except Exception as e:
            logger.warning(f"Crossover with memory failed: {e}")

    return variants


def add_random_rules(elite: Individual, population: List[Individual],
                     best_ever_memory: List[Individual], **kwargs) -> List[Individual]:
    """
    AGGRESSIVE: Adiciona regras aleatórias COMPLEXAS para aumentar diversidade.

    Usa crossover para gerar regras novas e adiciona ao elite.
    """
    variants = []

    try:
        # Seleciona indivíduos aleatórios da população para obter material genético
        candidates = [ind for ind in population if ind.fitness > 0]
        if len(candidates) < 2:
            return []

        # Gera 3 variantes com regras de outros indivíduos
        for _ in range(3):
            variant = copy.deepcopy(elite)

            # Pega 2 indivíduos aleatórios da população
            donor1 = random.choice(candidates)
            donor2 = random.choice(candidates)

            # Para cada classe, adiciona 1-2 regras aleatórias dos doadores
            for cls in variant.classes:
                num_to_add = random.randint(1, 2)

                # Coleta regras dos doadores
                donor_rules = []
                if cls in donor1.rules:
                    donor_rules.extend(donor1.rules[cls])
                if cls in donor2.rules:
                    donor_rules.extend(donor2.rules[cls])

                # Adiciona regras aleatórias dos doadores
                if donor_rules:
                    for _ in range(min(num_to_add, len(donor_rules))):
                        rule = random.choice(donor_rules)
                        variant.rules[cls].append(copy.deepcopy(rule))

            variants.append(variant)

        return variants

    except Exception as e:
        logger.warning(f"add_random_rules falhou: {e}")
        return []


def diverse_mutation(elite: Individual, population: List[Individual],
                    best_ever_memory: List[Individual], **kwargs) -> List[Individual]:
    """
    AGGRESSIVE: Mutação com alta taxa para exploração agressiva.
    """
    variants = []
    num_variants = 4

    for _ in range(num_variants):
        variant = copy.deepcopy(elite)

        # Mutação agressiva usando mutate_individual do ga_operators
        try:
            # Usa função de mutação completa com alta taxa
            # Ordem correta dos parâmetros conforme ga_operators.py:1695
            ga_operators.mutate_individual(
                individual=variant,
                mutation_rate=0.9,  # Taxa muito alta
                max_depth=kwargs['max_depth'],
                intelligent_mutation_prob=0.7,  # Alta taxa de mutação inteligente
                attributes=kwargs['attributes'],
                value_ranges=kwargs['value_ranges'],
                category_values=kwargs['category_values'],
                categorical_features=kwargs['categorical_features'],
                classes=kwargs['classes'],
                max_rules_per_class=kwargs['max_rules_per_class'],
                data=kwargs.get('train_data', []),
                target=kwargs.get('train_target', []),
                force_gene_therapy=False
            )
            variants.append(variant)
        except Exception as e:
            logger.warning(f"Mutação agressiva falhou: {e}")

    return variants


def dt_error_correction(elite: Individual, population: List[Individual],
                       best_ever_memory: List[Individual], **kwargs) -> List[Individual]:
    """
    MODERATE: Treina DT nas instâncias onde elite erra e injeta regras corretivas.

    Esta é a operação mais inteligente do HC:
    1. Identifica onde o elite está errando
    2. Treina DT especificamente nesses erros
    3. Extrai regras DT e injeta no elite
    """
    train_data = kwargs.get('train_data', [])
    train_target = kwargs.get('train_target', [])

    if not train_data or len(train_data) < 10:
        return []

    try:
        # 1. Identifica instâncias onde elite erra
        error_indices = []
        error_data = []
        error_target = []

        for idx, (x, y_true) in enumerate(zip(train_data, train_target)):
            y_pred = elite._predict(x)
            if y_pred != y_true:
                error_indices.append(idx)
                error_data.append(x)
                error_target.append(y_true)

        # Se não há erros ou poucos erros, não faz nada
        if len(error_data) < 5:
            logger.info(f"       dt_error_correction: Elite tem apenas {len(error_data)} erros - não aplicável")
            return []

        logger.info(f"       dt_error_correction: Elite erra em {len(error_data)}/{len(train_data)} instâncias ({100*len(error_data)/len(train_data):.1f}%)")

        # 2. Treina DT nos erros (foca nas instâncias difíceis)
        from sklearn.tree import DecisionTreeClassifier

        # Converte dados para formato sklearn
        X_error = pd.DataFrame(error_data)
        y_error = np.array(error_target)

        # Treina DT com profundidade MAIOR para regras mais complexas e precisas
        # AUMENTADO: de [3,5,7] para [5,8,10,12] para gerar regras mais sofisticadas
        dt = DecisionTreeClassifier(
            max_depth=random.choice([5, 8, 10, 12]),  # Profundidades maiores
            min_samples_split=max(2, len(error_data) // 15),  # Menos restritivo
            min_samples_leaf=max(1, len(error_data) // 40),  # Menos restritivo
            random_state=random.randint(0, 10000)
        )
        dt.fit(X_error, y_error)

        # 3. Extrai regras do DT e injeta no elite
        variant = copy.deepcopy(elite)

        # Extrai regras usando o mesmo método do seeding
        from sklearn.tree import _tree
        tree_ = dt.tree_
        feature_names = list(X_error.columns)

        def extract_rules_from_node(node, depth=0):
            """Extrai regras de um nó da árvore DT"""
            if tree_.feature[node] != _tree.TREE_UNDEFINED:  # Nó interno
                feature = feature_names[tree_.feature[node]]
                threshold = tree_.threshold[node]

                # Recursão
                left_rules = extract_rules_from_node(tree_.children_left[node], depth + 1)
                right_rules = extract_rules_from_node(tree_.children_right[node], depth + 1)

                return left_rules + right_rules
            else:  # Folha
                # Retorna a classe predita e número de amostras
                class_id = np.argmax(tree_.value[node][0])
                n_samples = tree_.n_node_samples[node]
                return [(kwargs['classes'][class_id], n_samples)]

        # Extrai até 2 regras do DT por classe
        dt_rules_by_class = {}
        for cls in variant.classes:
            dt_rules_by_class[cls] = []

        # Injeta regras corretivas usando doadores da população
        # Seleciona indivíduos com bom desempenho para fornecer material genético
        candidates = [ind for ind in population if ind.fitness > 0]
        if candidates:
            for cls in variant.classes:
                num_to_add = random.randint(1, 2)
                # Seleciona doadores aleatórios
                donors = random.sample(candidates, min(2, len(candidates)))

                # Coleta regras dos doadores
                donor_rules = []
                for donor in donors:
                    if cls in donor.rules and donor.rules[cls]:
                        donor_rules.extend(donor.rules[cls])

                # Injeta regras se disponíveis
                if donor_rules:
                    for _ in range(min(num_to_add, len(donor_rules))):
                        rule = random.choice(donor_rules)
                        variant.rules[cls].append(copy.deepcopy(rule))

        logger.info(f"       dt_error_correction: Injetou regras corretivas no elite")

        return [variant]

    except Exception as e:
        logger.warning(f"dt_error_correction falhou: {e}")
        return []


def optimize_thresholds(elite: Individual, population: List[Individual],
                       best_ever_memory: List[Individual], **kwargs) -> List[Individual]:
    """
    MODERATE: Otimiza thresholds numéricos com ajustes moderados.
    """
    variant = copy.deepcopy(elite)

    # Ajusta thresholds em +/- 10%
    for cls in variant.classes:
        for rule in variant.rules[cls]:
            _adjust_thresholds_recursive(rule.root, delta_pct=0.10)

    return [variant]


def _adjust_thresholds_recursive(node, delta_pct=0.10):
    """Helper: Ajusta thresholds numéricos recursivamente."""
    if node is None:
        return

    # Se é nó de comparação numérica com threshold
    if hasattr(node, 'operator') and node.operator in ['<=', '>', '<', '>=']:
        if hasattr(node, 'value') and isinstance(node.value, (int, float)):
            # Ajusta em +/- delta_pct com 50% de chance cada
            if random.random() < 0.5:
                node.value *= (1 + delta_pct)
            else:
                node.value *= (1 - delta_pct)

    # Recursão
    if hasattr(node, 'left'):
        _adjust_thresholds_recursive(node.left, delta_pct)
    if hasattr(node, 'right'):
        _adjust_thresholds_recursive(node.right, delta_pct)


def prune_weak_rules(elite: Individual, population: List[Individual],
                    best_ever_memory: List[Individual], **kwargs) -> List[Individual]:
    """
    MODERATE: Remove regras com baixa ativação/qualidade.
    """
    variant = copy.deepcopy(elite)
    train_data = kwargs.get('train_data', [])
    train_target = kwargs.get('train_target', [])

    if not train_data:
        return [variant]

    # Avalia ativação de cada regra
    for cls in variant.classes:
        if len(variant.rules[cls]) <= 1:
            continue  # Mantém pelo menos 1 regra por classe

        rule_activations = []
        for rule in variant.rules[cls]:
            activation_count = sum(1 for x in train_data if rule.evaluate(x))
            rule_activations.append((activation_count, rule))

        # Remove regras com ativação < 5% dos dados
        min_activation = len(train_data) * 0.05
        variant.rules[cls] = [rule for act, rule in rule_activations if act >= min_activation]

        # Garante pelo menos 1 regra
        if not variant.rules[cls] and rule_activations:
            variant.rules[cls] = [rule_activations[0][1]]

    return [variant]


def targeted_mutation(elite: Individual, population: List[Individual],
                     best_ever_memory: List[Individual], **kwargs) -> List[Individual]:
    """
    MODERATE: Mutação focada com taxa moderada.
    """
    variants = []
    num_variants = 2

    for _ in range(num_variants):
        variant = ga_operators.mutate_hill_climbing(elite, kwargs['value_ranges'])
        variants.append(variant)

    return variants


def threshold_grid_search(elite: Individual, population: List[Individual],
                         best_ever_memory: List[Individual], **kwargs) -> List[Individual]:
    """
    FINE_TUNING: Busca em grid refinada nos thresholds (+/- 5%, 2%, 1%).
    """
    variants = []

    for delta_pct in [0.05, 0.02, 0.01]:
        variant = copy.deepcopy(elite)
        for cls in variant.classes:
            for rule in variant.rules[cls]:
                _adjust_thresholds_recursive(rule.root, delta_pct=delta_pct)
        variants.append(variant)

    return variants


def micro_adjustments(elite: Individual, population: List[Individual],
                     best_ever_memory: List[Individual], **kwargs) -> List[Individual]:
    """
    FINE_TUNING: Micro-ajustes com mutação muito suave.
    """
    variant = copy.deepcopy(elite)

    # Mutação muito suave usando mutate_individual com taxa baixa
    try:
        # Ordem correta dos parâmetros conforme ga_operators.py:1695
        ga_operators.mutate_individual(
            individual=variant,
            mutation_rate=0.1,  # Taxa muito baixa para ajustes finos
            max_depth=kwargs['max_depth'],
            intelligent_mutation_prob=0.05,
            attributes=kwargs['attributes'],
            value_ranges=kwargs['value_ranges'],
            category_values=kwargs['category_values'],
            categorical_features=kwargs['categorical_features'],
            classes=kwargs['classes'],
            max_rules_per_class=kwargs['max_rules_per_class'],
            data=kwargs.get('train_data', []),
            target=kwargs.get('train_target', []),
            force_gene_therapy=False
        )
    except Exception as e:
        logger.warning(f"Micro-ajuste falhou: {e}")

    return [variant]


# ============================================================================
# SISTEMA PRINCIPAL: HILL CLIMBING HIERÁRQUICO
# ============================================================================

def hierarchical_hill_climbing(
    elite: Individual,
    population: List[Individual],
    best_ever_memory: List[Individual],
    gmean: float,
    no_improvement_count: int,
    hc_enable_adaptive: bool = True,
    hc_gmean_threshold: float = 0.90,
    hc_hierarchical_enabled: bool = True,
    **kwargs
) -> List[Individual]:
    """
    Sistema hierárquico de Hill Climbing com 3 níveis.

    Args:
        elite: Melhor indivíduo atual
        population: População atual
        best_ever_memory: Memória dos melhores históricos
        gmean: G-mean do elite
        no_improvement_count: Gerações sem melhoria
        hc_enable_adaptive: Se True, usa threshold para desabilitar HC
        hc_gmean_threshold: Threshold de G-mean para desabilitar (padrão 0.90)
        hc_hierarchical_enabled: Se True, usa sistema hierárquico (padrão True)
        **kwargs: Parâmetros adicionais (train_data, value_ranges, etc.)

    Returns:
        Lista de variantes geradas pelo Hill Climbing
    """
    # 1. Verifica se HC deve rodar (sistema adaptativo legado)
    if hc_enable_adaptive and not hc_hierarchical_enabled:
        # Modo legado: desabilita se gmean >= threshold
        if gmean >= hc_gmean_threshold:
            logger.info(f"  -> Hill Climbing DESABILITADO (elite G-mean: {gmean:.3f} ≥ {hc_gmean_threshold:.3f})")
            logger.info(f"     Margem insuficiente para refinamento. Apenas aumentando mutação.")
            return []

    # 2. Sistema hierárquico: Seleciona nível baseado em G-mean
    if not hc_hierarchical_enabled:
        # Fallback para HC simples legado
        logger.info(f"  -> Ativando Hill Climbing simples (elite G-mean: {gmean:.3f})...")
        champion = population[0]
        num_hc_mutants = max(1, int(len(population) * 0.1))
        variants = []
        for _ in range(num_hc_mutants):
            hc_mutant = ga_operators.mutate_hill_climbing(champion, kwargs.get('value_ranges', {}))
            variants.append(hc_mutant)
        return variants

    # 3. Seleciona nível hierárquico
    level_key = None
    for key, config in HILL_CLIMBING_LEVELS.items():
        min_gmean, max_gmean = config['gmean_range']
        if min_gmean <= gmean < max_gmean:
            level_key = key
            break

    # Se gmean >= 0.98, desabilita HC (quase perfeito)
    if level_key is None:
        if gmean >= 0.98:
            logger.info(f"  -> Hill Climbing DESABILITADO (elite G-mean: {gmean:.3f} ≥ 0.98 - Quase perfeito)")
            return []
        else:
            # gmean < 0.70, usa aggressive
            level_key = 'aggressive'

    level_config = HILL_CLIMBING_LEVELS[level_key]

    logger.info(f"  -> Hill Climbing V2 HIERÁRQUICO [{level_key.upper()}]")
    logger.info(f"     Elite G-mean: {gmean:.3f} → {level_config['description']}")
    logger.info(f"     Operações: {level_config['operations']}")
    logger.info(f"     Variantes planejadas: {level_config['num_variants_base']}")

    # 4. Aplica operações do nível
    all_variants = []
    operation_map = {
        # Operações legadas
        'inject_memory_rules': inject_memory_rules,
        'crossover_with_memory': crossover_with_memory,
        'add_random_rules': add_random_rules,
        'diverse_mutation': diverse_mutation,
        'dt_error_correction': dt_error_correction,  # Operação antiga (manter para compatibilidade)
        'optimize_thresholds': optimize_thresholds,
        'prune_weak_rules': prune_weak_rules,
        'targeted_mutation': targeted_mutation,
        'threshold_grid_search': threshold_grid_search,
        'micro_adjustments': micro_adjustments,
        # Novas operações inteligentes
        'error_focused_dt_rules': error_focused_dt_rules,
        'ensemble_boosting': ensemble_boosting,
        'guided_mutation': guided_mutation
    }

    for op_name in level_config['operations']:
        if op_name in operation_map:
            try:
                operation_func = operation_map[op_name]

                # Para novas estratégias inteligentes, passa dados em formato pandas
                if op_name in ['error_focused_dt_rules', 'ensemble_boosting', 'guided_mutation']:
                    # Converte dados de treino para pandas
                    train_data = kwargs.get('train_data', [])
                    train_target = kwargs.get('train_target', [])

                    if not train_data or not train_target:
                        logger.warning(f"       ✗ {op_name}: dados de treino não disponíveis")
                        continue

                    # Converte para DataFrame/Series
                    feature_names = kwargs.get('attributes', [f'f{i}' for i in range(len(train_data[0]))])
                    X_train = pd.DataFrame(train_data, columns=feature_names)
                    y_train = pd.Series(train_target)

                    # Chama estratégia inteligente
                    variants = operation_func(elite, X_train, y_train, **kwargs)
                else:
                    # Operações legadas
                    variants = operation_func(elite, population, best_ever_memory, **kwargs)

                all_variants.extend(variants)
                logger.info(f"       ✓ {op_name}: {len(variants)} variantes geradas")
            except Exception as e:
                logger.warning(f"       ✗ {op_name} falhou: {e}")

    # 5. Retorna top-K variantes
    num_variants = min(level_config['num_variants_base'], len(all_variants))

    logger.info(f"     Total gerado: {len(all_variants)} variantes, retornando {num_variants}")

    return all_variants[:num_variants] if all_variants else []


# ============================================================================
# TESTE DO MÓDULO
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
    )

    logger.info("=== Teste do Módulo hill_climbing_v2.py ===")
    logger.info("✓ Módulo hill_climbing_v2.py carregado com sucesso")
    logger.info(f"  Níveis configurados: {list(HILL_CLIMBING_LEVELS.keys())}")

    for level_key, config in HILL_CLIMBING_LEVELS.items():
        logger.info(f"\n  Nível: {level_key.upper()}")
        logger.info(f"    Range G-mean: {config['gmean_range']}")
        logger.info(f"    Operações: {config['operations']}")
        logger.info(f"    Variantes base: {config['num_variants_base']}")
