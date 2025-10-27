# gbml_evaluator.py
"""
Wrapper do GBML Framework para Interface Unificada de Avaliação

PROPÓSITO: Adaptar o sistema GBML existente para usar a mesma interface
           de avaliação que os modelos River, garantindo comparação justa.

AUTOR: Claude Code
DATA: 2025-01-06
"""

import logging
import copy
import pickle
import os
from typing import List, Dict, Any
from shared_evaluation import ChunkEvaluator, calculate_shared_metrics

# Importações do GBML
import ga
from individual import Individual

logger = logging.getLogger("gbml_evaluator")


# ============================================================================
# AVALIADOR GBML
# ============================================================================

class GBMLEvaluator(ChunkEvaluator):
    """
    Adapta o framework GBML para a interface unificada ChunkEvaluator.

    NOTA: GBML já trabalha nativamente com chunks, então esta adaptação
          é mais simples que a do River.
    """

    def __init__(
        self,
        model_name: str,
        classes: List,
        attributes: List[str],
        value_ranges: Dict[str, tuple],
        category_values: Dict[str, set],
        categorical_features: set,
        ga_params: Dict,
        fitness_params: Dict,
        memory_params: Dict,
        parallelism_params: Dict,
        output_dir: str = None
    ):
        """
        Args:
            model_name: Nome do modelo (e.g., 'GBML')
            classes: Lista de classes possíveis
            attributes: Lista de nomes de atributos
            value_ranges: Dict {attr: (min, max)} para atributos numéricos
            category_values: Dict {attr: set(values)} para atributos categóricos
            categorical_features: Set de nomes de atributos categóricos
            ga_params: Parâmetros do algoritmo genético
            fitness_params: Parâmetros da função de fitness
            memory_params: Parâmetros de memória
            parallelism_params: Parâmetros de paralelização
            output_dir: Diretório para salvar melhores indivíduos (opcional)
        """
        super().__init__(model_name, classes)

        self.attributes = attributes
        self.value_ranges = value_ranges
        self.category_values = category_values
        self.categorical_features = categorical_features

        self.ga_params = ga_params
        self.fitness_params = fitness_params
        self.memory_params = memory_params
        self.parallelism_params = parallelism_params

        # Diretório para salvamento
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self.logger.info(f"Salvamento de indivíduos ativado em: {output_dir}")

        # Estado do GBML
        self.best_individual = None
        self.best_ever_memory = []
        self.previous_best_individual = None
        self.previous_used_features = None
        self.previous_operator_info = None

        self.logger.info(f"GBMLEvaluator inicializado para classes: {classes}")

    def train_on_chunk(self, X_train: List[Dict], y_train: List, chunk_index: int = None) -> Dict[str, Any]:
        """
        Treina o GBML em um chunk executando o loop evolucionário.

        Args:
            X_train: Features do chunk de treino
            y_train: Labels do chunk de treino

        Returns:
            Dicionário com métricas de treino
        """
        self.logger.info(f"Iniciando evolução GA em {len(X_train)} instâncias...")

        # Calcula class_weights baseado na distribuição do chunk
        from collections import Counter
        class_counts = Counter(y_train)
        total = len(y_train)
        class_weights = {cls: total / (len(self.classes) * count)
                        for cls, count in class_counts.items()}
        # Normaliza para que a soma seja igual ao número de classes
        weight_sum = sum(class_weights.values())
        class_weights = {cls: (w / weight_sum) * len(self.classes)
                        for cls, w in class_weights.items()}

        try:
            # Executa o loop evolucionário do GBML
            # Esta é a chamada principal que treina o modelo
            best_individual, final_population, ga_history = ga.run_genetic_algorithm(
                attributes=self.attributes,
                value_ranges=self.value_ranges,
                classes=self.classes,
                class_weights=class_weights,
                train_data=X_train,
                train_target=y_train,
                categorical_features=self.categorical_features,
                category_values=self.category_values,
                # Parâmetros do GA
                population_size=self.ga_params['population_size'],
                max_generations=self.ga_params['max_generations'],
                max_rules_per_class=self.ga_params['max_rules_per_class'],
                max_depth=self.ga_params.get('max_depth', self.ga_params.get('initial_max_depth', 5)),
                elitism_rate=self.ga_params['elitism_rate'],
                intelligent_mutation_rate=self.ga_params['intelligent_mutation_rate'],
                initial_tournament_size=self.ga_params['initial_tournament_size'],
                final_tournament_size=self.ga_params['final_tournament_size'],
                early_stopping_patience=self.ga_params['early_stopping_patience'],
                # Parâmetros de fitness
                regularization_coefficient=self.fitness_params['initial_regularization_coefficient'],
                feature_penalty_coefficient=self.fitness_params['feature_penalty_coefficient'],
                operator_penalty_coefficient=self.fitness_params['operator_penalty_coefficient'],
                threshold_penalty_coefficient=self.fitness_params['threshold_penalty_coefficient'],
                operator_change_coefficient=self.fitness_params['operator_change_coefficient'],
                gamma=self.fitness_params['gamma'],
                gmean_bonus_coefficient_ga=self.fitness_params.get('gmean_bonus_coefficient', 0.0),
                class_coverage_coefficient_ga=self.fitness_params.get('class_coverage_coefficient', 0.0),
                # Estado anterior (para adaptação incremental)
                previous_rules_pop=None,
                best_ever_memory=self.best_ever_memory,
                best_individual_from_previous_chunk=self.previous_best_individual,
                previous_used_features=self.previous_used_features,
                previous_operator_info=self.previous_operator_info,
                # Parâmetros adicionais
                parallel_enabled=self.parallelism_params.get('enabled', True),
                num_workers=self.parallelism_params.get('num_workers', None),
                performance_label='medium',  # Será ajustado dinamicamente em produção
                reduce_change_penalties_flag=False,  # Será ajustado se houver drift
                # Parâmetros de seeding (Robust Seeding v4.0)
                initialization_strategy=self.ga_params.get('initialization_strategy', 'full_random'),
                enable_dt_seeding_on_init_config_ga=self.ga_params.get('enable_dt_seeding_on_init', False),
                dt_seeding_ratio_on_init_config_ga=self.ga_params.get('dt_seeding_ratio_on_init', 0.0),
                dt_seeding_depths_on_init_config_ga=self.ga_params.get('dt_seeding_depths_on_init', [5, 10]),
                dt_seeding_sample_size_on_init_config_ga=self.ga_params.get('dt_seeding_sample_size_on_init', 200),
                dt_seeding_rules_to_replace_config_ga=self.ga_params.get('dt_seeding_rules_to_replace_per_class', 1),
                recovery_aggressive_mutant_ratio_config_ga=self.ga_params.get('recovery_aggressive_mutant_ratio', 0.0),
                mutation_override_config_ga=self.ga_params.get('mutation_override', None),
                historical_reference_dataset=None,  # Pode ser adicionado para usar dados históricos
                # Seeding Probabilístico
                dt_rule_injection_ratio_config_ga=self.ga_params.get('dt_rule_injection_ratio', 1.0),
                # Seeding Adaptativo
                enable_adaptive_seeding_config_ga=self.ga_params.get('enable_adaptive_seeding', False),
                # Hill Climbing Adaptativo (legado)
                hc_enable_adaptive=self.ga_params.get('hc_enable_adaptive', False),
                hc_gmean_threshold=self.ga_params.get('hc_gmean_threshold', 0.90),
                # Hill Climbing Hierárquico v2.0
                hc_hierarchical_enabled=self.ga_params.get('hc_hierarchical_enabled', True),
                stagnation_threshold=self.ga_params.get('stagnation_threshold', 15)
            )

            # Atualiza estado interno
            self.best_individual = best_individual

            # Atualiza best_ever_memory manualmente (já que run_genetic_algorithm não retorna)
            # Adiciona o melhor indivíduo atual à memória
            if best_individual is not None:
                self.best_ever_memory.append(copy.deepcopy(best_individual))
                # Ordena por fitness e mantém apenas os melhores
                self.best_ever_memory.sort(key=lambda ind: ind.fitness, reverse=True)
                max_memory = self.memory_params.get('max_memory_size', 20)
                self.best_ever_memory = self.best_ever_memory[:max_memory]

                # Salva o melhor indivíduo do chunk em pickle (se output_dir configurado)
                if self.output_dir and chunk_index is not None:
                    individual_filename = f"best_individual_chunk_{chunk_index}.pkl"
                    individual_path = os.path.join(self.output_dir, individual_filename)
                    try:
                        with open(individual_path, 'wb') as f:
                            pickle.dump(best_individual, f)
                        self.logger.info(f"✓ Melhor indivíduo do chunk {chunk_index} salvo em: {individual_path}")
                    except Exception as e:
                        self.logger.error(f"✗ Erro ao salvar indivíduo do chunk {chunk_index}: {e}")

            self.previous_best_individual = copy.deepcopy(best_individual)
            self.previous_used_features = best_individual.get_used_attributes()

            # Coleta info de operadores
            try:
                l_ops, c_ops, num_thresh, cat_vals = best_individual._collect_ops_and_thresholds()
                self.previous_operator_info = {
                    "logical_ops": set(l_ops),
                    "comparison_ops": set(c_ops),
                    "numeric_thresholds": num_thresh,
                    "categorical_values": cat_vals
                }
            except Exception as e:
                self.logger.warning(f"Não foi possível coletar info de operadores: {e}")
                self.previous_operator_info = None

            # Retorna métricas de treino
            train_metrics = {
                'train_fitness': best_individual.fitness,
                'train_gmean': best_individual.gmean,
                'n_generations': len(ga_history) if ga_history else 0
            }

            self.logger.info(
                f"✓ Evolução concluída - Fitness: {best_individual.fitness:.4f}, "
                f"G-mean: {best_individual.gmean:.4f}"
            )

            return train_metrics

        except Exception as e:
            self.logger.error(f"✗ Erro durante treinamento do GBML: {e}", exc_info=True)
            raise

    def test_on_chunk(self, X_test: List[Dict], y_test: List) -> Dict[str, float]:
        """
        Testa o melhor indivíduo do GBML em um chunk.

        Args:
            X_test: Features do chunk de teste
            y_test: Labels verdadeiros

        Returns:
            Dicionário com métricas de teste
        """
        if self.best_individual is None:
            raise RuntimeError("Modelo não foi treinado ainda! Chame train_on_chunk() primeiro.")

        # Faz predições usando o melhor indivíduo
        y_pred = [self.best_individual._predict(x) for x in X_test]

        # Calcula métricas padronizadas
        metrics = calculate_shared_metrics(y_test, y_pred, self.classes)

        return metrics

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o estado atual do GBML.

        Returns:
            Dicionário com informações do modelo
        """
        if self.best_individual is None:
            return {'model_type': 'GBML', 'status': 'not_trained'}

        return {
            'model_type': 'GBML',
            'n_rules': self.best_individual.count_total_rules(),
            'n_nodes': self.best_individual.count_total_nodes(),
            'n_features_used': len(self.best_individual.get_used_attributes()),
            'memory_size': len(self.best_ever_memory)
        }


# ============================================================================
# FACTORY PARA CRIAR GBML A PARTIR DE CONFIG
# ============================================================================

def create_gbml_from_config(
    config: Dict,
    attributes: List[str],
    value_ranges: Dict,
    category_values: Dict,
    categorical_features: set,
    classes: List,
    output_dir: str = None
) -> GBMLEvaluator:
    """
    Cria um GBMLEvaluator a partir de um dicionário de configuração.

    Args:
        config: Dicionário de configuração (carregado do config.yaml)
        attributes: Lista de nomes de atributos
        value_ranges: Ranges de valores numéricos
        category_values: Valores categóricos possíveis
        categorical_features: Set de features categóricas
        classes: Lista de classes
        output_dir: Diretório para salvar melhores indivíduos (opcional)

    Returns:
        GBMLEvaluator configurado
    """
    return GBMLEvaluator(
        model_name='GBML',
        classes=classes,
        attributes=attributes,
        value_ranges=value_ranges,
        category_values=category_values,
        categorical_features=categorical_features,
        ga_params=config.get('ga_params', {}),
        fitness_params=config.get('fitness_params', {}),
        memory_params=config.get('memory_params', {}),
        parallelism_params=config.get('parallelism', {}),
        output_dir=output_dir
    )


# ============================================================================
# TESTE DO MÓDULO
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
    )

    logger.info("=== Teste do Módulo gbml_evaluator.py ===")

    # Este teste requer os módulos do GBML estarem funcionais
    # Em produção, usar create_gbml_from_config() com config.yaml real

    logger.info("✓ Módulo gbml_evaluator.py carregado com sucesso")
    logger.info("  Para teste completo, use compare_gbml_vs_river.py")
