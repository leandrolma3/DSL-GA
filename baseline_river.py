# baseline_river.py
"""
Implementação de Modelos Baseline do River para Comparação com GBML

PROPÓSITO: Avaliar modelos clássicos de data stream learning usando
           a MESMA divisão de dados e metodologia do GBML.

MODELOS IMPLEMENTADOS:
- Hoeffding Adaptive Tree (HAT)
- Adaptive Random Forest (ARF)
- Streaming Random Patches (SRP)
- ADWIN Bagging
- Leveraging Bagging

AUTOR: Claude Code
DATA: 2025-01-06
"""

import logging
import numpy as np
from typing import List, Dict, Any
from shared_evaluation import ChunkEvaluator, calculate_shared_metrics

# Importações do River
try:
    from river import tree, ensemble, drift
    from river.base import Classifier
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logging.error(
        "River não está instalado! Instale com: pip install river"
    )

logger = logging.getLogger("baseline_river")


# ============================================================================
# AVALIADOR GENÉRICO PARA MODELOS RIVER
# ============================================================================

class RiverEvaluator(ChunkEvaluator):
    """
    Avaliador que adapta modelos River (incremental) para trabalhar com chunks.

    IMPORTANTE: River trabalha instância-por-instância (incremental learning).
                Esta classe adapta para trabalhar com chunks mantendo
                compatibilidade com GBML.
    """

    def __init__(self, model: Classifier, model_name: str, classes: List):
        """
        Args:
            model: Instância do modelo River (deve implementar learn_one e predict_one)
            model_name: Nome do modelo para identificação
            classes: Lista de classes possíveis
        """
        super().__init__(model_name, classes)

        if not RIVER_AVAILABLE:
            raise ImportError("River não está instalado!")

        self.model = model
        self.logger.info(f"Modelo River '{model_name}' inicializado")

    def train_on_chunk(self, X_train: List[Dict], y_train: List) -> Dict[str, Any]:
        """
        Treina modelo River em um chunk usando learn_one sequencialmente.

        Args:
            X_train: Lista de dicionários com features
            y_train: Lista de labels

        Returns:
            Dicionário vazio (River não retorna métricas de treino por padrão)
        """
        if len(X_train) != len(y_train):
            raise ValueError(f"Tamanhos incompatíveis: X={len(X_train)}, y={len(y_train)}")

        # Treina instância por instância (modo incremental do River)
        for x, y in zip(X_train, y_train):
            # Garante que x é dict e y é escalar
            if not isinstance(x, dict):
                raise TypeError(f"X deve ser dict, recebido: {type(x)}")

            try:
                self.model.learn_one(x, y)
            except Exception as e:
                self.logger.error(f"Erro no learn_one: {e}")
                raise

        self.logger.debug(f"Treinamento concluído: {len(X_train)} instâncias")
        return {}

    def test_on_chunk(self, X_test: List[Dict], y_test: List) -> Dict[str, float]:
        """
        Testa modelo River em um chunk usando predict_one sequencialmente.

        Args:
            X_test: Lista de dicionários com features
            y_test: Lista de labels verdadeiros

        Returns:
            Dicionário com métricas de teste
        """
        if len(X_test) != len(y_test):
            raise ValueError(f"Tamanhos incompatíveis: X={len(X_test)}, y={len(y_test)}")

        y_pred = []

        # Prediz instância por instância
        for x in X_test:
            if not isinstance(x, dict):
                raise TypeError(f"X deve ser dict, recebido: {type(x)}")

            try:
                # predict_one retorna um dict {class: proba} para alguns modelos
                # ou apenas a classe para outros
                prediction = self.model.predict_one(x)

                # Se retornar dict de probabilidades, pega a classe com maior prob
                if isinstance(prediction, dict):
                    prediction = max(prediction.items(), key=lambda item: item[1])[0]

                y_pred.append(prediction)

            except Exception as e:
                self.logger.error(f"Erro no predict_one: {e}")
                # Fallback: prediz classe majoritária ou primeira classe
                y_pred.append(self.classes[0] if self.classes else 0)

        # Calcula métricas padronizadas
        metrics = calculate_shared_metrics(y_test, y_pred, self.classes)

        return metrics

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo River.

        Returns:
            Dicionário com informações do modelo
        """
        info = {
            'model_type': type(self.model).__name__
        }

        # Tenta obter informações específicas de cada tipo de modelo
        try:
            # Para modelos baseados em árvore
            if hasattr(self.model, 'n_nodes'):
                info['n_nodes'] = getattr(self.model, 'n_nodes', None)

            # Para ensembles
            if hasattr(self.model, 'n_models'):
                info['n_models'] = getattr(self.model, 'n_models', None)
            elif hasattr(self.model, 'models'):  # Lista de modelos
                info['n_models'] = len(self.model.models) if self.model.models else 0

        except Exception as e:
            self.logger.warning(f"Não foi possível extrair info do modelo: {e}")

        return info


# ============================================================================
# FACTORY PARA MODELOS RIVER PRÉ-CONFIGURADOS
# ============================================================================

def create_river_model(model_name: str, classes: List, **kwargs) -> RiverEvaluator:
    """
    Factory function para criar modelos River pré-configurados.

    Args:
        model_name: Nome do modelo ('HAT', 'ARF', 'SRP', etc.)
        classes: Lista de classes do problema
        **kwargs: Parâmetros adicionais para o modelo

    Returns:
        RiverEvaluator configurado com o modelo especificado

    Raises:
        ValueError: Se model_name não for reconhecido
    """
    if not RIVER_AVAILABLE:
        raise ImportError("River não está instalado!")

    model_name_upper = model_name.upper()

    # Configurações padrão para cada modelo
    if model_name_upper == 'HAT':
        # API atualizada do River - parâmetros corretos
        model = tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=kwargs.get('grace_period', 200),
            delta=kwargs.get('delta', 1e-7),  # Mudou de split_confidence para delta
            tau=kwargs.get('tau', 0.05),       # Mudou de tie_threshold para tau
            leaf_prediction=kwargs.get('leaf_prediction', 'nba'),
            **{k: v for k, v in kwargs.items() if k not in [
                'grace_period', 'delta', 'tau', 'leaf_prediction'
            ]}
        )

    elif model_name_upper == 'ARF':
        # River 0.15+ moveu ARF de ensemble para forest
        try:
            from river.forest import ARFClassifier
        except ImportError:
            # Fallback para versões antigas (< 0.15)
            try:
                from river.ensemble import AdaptiveRandomForestClassifier as ARFClassifier
            except ImportError:
                raise ImportError(
                    "ARF não encontrado. Instale River >= 0.15 ou use outro modelo."
                )

        model = ARFClassifier(
            n_models=kwargs.get('n_models', 10),
            max_features=kwargs.get('max_features', 'sqrt'),
            lambda_value=kwargs.get('lambda_value', 6),
            drift_detector=drift.ADWIN(delta=kwargs.get('adwin_delta', 0.001)),
            **{k: v for k, v in kwargs.items() if k not in [
                'n_models', 'max_features', 'lambda_value', 'adwin_delta'
            ]}
        )

    elif model_name_upper == 'SRP':
        model = ensemble.SRPClassifier(
            n_models=kwargs.get('n_models', 10),
            subspace_size=kwargs.get('subspace_size', 0.6),
            drift_detector=drift.ADWIN(delta=kwargs.get('adwin_delta', 0.001)),
            **{k: v for k, v in kwargs.items() if k not in [
                'n_models', 'subspace_size', 'adwin_delta'
            ]}
        )

    elif model_name_upper in ['ADWIN_BAGGING', 'ADWIN_BAG']:
        model = ensemble.ADWINBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(),
            n_models=kwargs.get('n_models', 10),
            **{k: v for k, v in kwargs.items() if k != 'n_models'}
        )

    elif model_name_upper in ['LEVERAGING_BAGGING', 'LEV_BAG']:
        model = ensemble.LeveragingBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(),
            n_models=kwargs.get('n_models', 10),
            **{k: v for k, v in kwargs.items() if k != 'n_models'}
        )

    else:
        available = ['HAT', 'ARF', 'SRP', 'ADWIN_BAGGING', 'LEVERAGING_BAGGING']
        raise ValueError(
            f"Modelo '{model_name}' não reconhecido. "
            f"Modelos disponíveis: {available}"
        )

    logger.info(f"Modelo River '{model_name}' criado com sucesso")
    return RiverEvaluator(model, model_name, classes)


# ============================================================================
# FUNÇÃO PARA EXECUTAR MÚLTIPLOS MODELOS RIVER
# ============================================================================

def run_river_baselines(
    chunks: List[tuple],
    classes: List,
    model_names: List[str] = None,
    evaluation_period: int = None,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Executa múltiplos modelos River nos mesmos chunks.

    Args:
        chunks: Lista de chunks gerados por load_or_generate_chunks()
        classes: Lista de classes do problema
        model_names: Lista de modelos a executar (default: todos)
        evaluation_period: Período de avaliação (para compatibilidade)
        **model_kwargs: Parâmetros adicionais para os modelos

    Returns:
        Dicionário com resultados de cada modelo
    """
    if model_names is None:
        model_names = ['HAT', 'ARF', 'SRP', 'ADWIN_BAGGING']

    results = {}

    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Executando modelo: {model_name}")
        logger.info(f"{'='*60}")

        try:
            # Cria avaliador
            evaluator = create_river_model(model_name, classes, **model_kwargs)

            # Executa avaliação prequential
            df_results = evaluator.evaluate_prequential(
                chunks,
                evaluation_period=evaluation_period
            )

            results[model_name] = df_results

            # Log de resumo
            avg_acc = df_results['accuracy'].mean()
            avg_f1 = df_results['f1_weighted'].mean()
            avg_gmean = df_results['gmean'].mean()

            logger.info(f"\n✓ {model_name} - Resultados:")
            logger.info(f"  Accuracy média: {avg_acc:.4f}")
            logger.info(f"  F1 média:       {avg_f1:.4f}")
            logger.info(f"  G-mean média:   {avg_gmean:.4f}")

        except Exception as e:
            logger.error(f"✗ Erro ao executar {model_name}: {e}")
            results[model_name] = None

    return results


# ============================================================================
# TESTE DO MÓDULO
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
    )

    if not RIVER_AVAILABLE:
        logger.error("River não está disponível. Instale com: pip install river")
        exit(1)

    logger.info("=== Teste do Módulo baseline_river.py ===")

    # Cria dados fictícios para teste
    fake_chunks = [
        (
            [{'x1': np.random.rand(), 'x2': np.random.rand()} for _ in range(100)],
            [np.random.randint(0, 2) for _ in range(100)]
        )
        for _ in range(5)
    ]

    classes = [0, 1]

    # Testa criação de modelos
    try:
        hat = create_river_model('HAT', classes)
        logger.info(f"✓ HAT criado: {type(hat.model).__name__}")

        arf = create_river_model('ARF', classes, n_models=5)
        logger.info(f"✓ ARF criado com 5 modelos")

    except Exception as e:
        logger.error(f"✗ Erro ao criar modelos: {e}")

    logger.info("\n=== Módulo baseline_river.py pronto para uso ===")
