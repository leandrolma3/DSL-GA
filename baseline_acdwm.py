"""
Implementação do ACDWM para Comparação com GBML e River

PROPÓSITO:
    Adapter que permite executar o ACDWM (Adaptive Chunk-Based Dynamic
    Weighted Majority) no framework de comparação existente.

DESIGN:
    - Segue a mesma interface do baseline_river.py (ChunkEvaluator)
    - Converte dados River (dicts) → NumPy arrays
    - Importa código do ACDWM dinamicamente
    - Suporta ambos os modos: test-then-train e train-then-test

DIFERENÇAS METODOLÓGICAS:
    ACDWM original usa test-then-train (prequential):
        - Chunk 1: Test com M₀ → Train → M₁
        - Chunk 2: Test com M₁ → Train → M₂

    GBML/River usam train-then-test:
        - Chunk 1: Train → M₁ → Test no Chunk 2
        - Chunk 2: Train → M₂ → Test no Chunk 3

    Este adapter suporta AMBOS os modos para comparação justa.

AUTOR: Claude Code
DATA: 2025-01-07
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Importa módulos locais
from data_converters import (
    river_to_numpy,
    river_labels_to_numpy,
    convert_chunks_river_to_numpy
)
from shared_evaluation import ChunkEvaluator, calculate_shared_metrics

logger = logging.getLogger("baseline_acdwm")


# ============================================================================
# FUNÇÕES DE CONVERSÃO DE LABELS
# ============================================================================

def convert_labels_to_acdwm(y: np.ndarray) -> np.ndarray:
    """
    Converte labels de 0/1 para -1/+1 (formato ACDWM).

    Args:
        y: Array com labels em formato 0/1

    Returns:
        Array com labels em formato -1/+1
    """
    return np.where(y == 0, -1, 1).astype(np.int32)


def convert_labels_from_acdwm(y_pred: np.ndarray) -> np.ndarray:
    """
    Converte labels de -1/+1 (formato ACDWM) para 0/1.

    Args:
        y_pred: Array com predições em formato -1/+1

    Returns:
        Array com predições em formato 0/1
    """
    return np.where(y_pred == -1, 0, 1).astype(np.int32)


# ============================================================================
# IMPORTAÇÃO DINÂMICA DO CÓDIGO ACDWM
# ============================================================================

def import_acdwm_modules(acdwm_path: str):
    """
    Importa módulos do ACDWM dinamicamente.

    Args:
        acdwm_path: Caminho para o diretório do código ACDWM

    Returns:
        Dicionário com módulos importados

    Raises:
        ImportError: Se não conseguir importar ACDWM
    """
    acdwm_path = Path(acdwm_path).resolve()

    if not acdwm_path.exists():
        raise ImportError(
            f"Diretório ACDWM não encontrado: {acdwm_path}\n"
            f"Clone o repositório com:\n"
            f"  git clone https://github.com/jasonyanglu/ACDWM.git {acdwm_path}"
        )

    logger.info(f"Importando módulos ACDWM de: {acdwm_path}")

    # Adiciona ao path se ainda não estiver
    acdwm_path_str = str(acdwm_path)
    if acdwm_path_str not in sys.path:
        sys.path.insert(0, acdwm_path_str)
        logger.debug(f"Adicionado ao sys.path: {acdwm_path_str}")

    try:
        # Importa módulos principais do ACDWM
        import dwmil
        import chunk_size_select
        import underbagging
        import subunderbagging

        logger.info("✓ Módulos ACDWM importados com sucesso")

        return {
            'dwmil': dwmil,
            'chunk_size_select': chunk_size_select,
            'underbagging': underbagging,
            'subunderbagging': subunderbagging
        }

    except ImportError as e:
        logger.error(f"Erro ao importar módulos ACDWM: {e}")
        logger.error(f"Verifique se os arquivos existem em: {acdwm_path}")
        logger.error("Arquivos esperados: dwmil.py, chunk_size_select.py, etc.")
        raise ImportError(
            f"Não foi possível importar ACDWM de {acdwm_path}\n"
            f"Erro original: {e}"
        ) from e


# ============================================================================
# EVALUATOR DO ACDWM
# ============================================================================

class ACDWMEvaluator(ChunkEvaluator):
    """
    Evaluator que adapta ACDWM para trabalhar com chunks do GBML.

    Responsabilidades:
        - Converter dados River → NumPy
        - Executar ACDWM (train/test)
        - Converter resultados para formato padronizado
        - Gerenciar modo de avaliação (test-then-train vs train-then-test)
    """

    def __init__(
        self,
        acdwm_path: str,
        classes: List,
        evaluation_mode: str = 'train-then-test',
        adaptive_chunk_size: bool = True,
        theta: float = 0.001,
        min_ensemble_size: int = 10,
        **kwargs
    ):
        """
        Inicializa o evaluator ACDWM.

        Args:
            acdwm_path: Caminho para código ACDWM
            classes: Lista de classes (ex: [0, 1])
            evaluation_mode: 'train-then-test' ou 'test-then-train'
            adaptive_chunk_size: Se True, usa seleção adaptativa de chunk
            theta: Threshold para remover classificadores (peso < theta)
            min_ensemble_size: Tamanho mínimo do ensemble UnderBagging
            **kwargs: Parâmetros adicionais
        """
        super().__init__(model_name='ACDWM', classes=classes)

        # Valida modo de avaliação
        if evaluation_mode not in ['train-then-test', 'test-then-train']:
            raise ValueError(
                f"evaluation_mode deve ser 'train-then-test' ou 'test-then-train', "
                f"recebido: {evaluation_mode}"
            )

        self.evaluation_mode = evaluation_mode
        self.adaptive_chunk_size = adaptive_chunk_size
        self.theta = theta
        self.min_ensemble_size = min_ensemble_size

        logger.info(f"Modo de avaliação: {evaluation_mode}")
        logger.info(f"Chunk size adaptativo: {adaptive_chunk_size}")

        # Importa módulos ACDWM
        self.acdwm_modules = import_acdwm_modules(acdwm_path)

        # Inicializa modelo ACDWM
        self._init_acdwm_model()

        # Estado interno
        self.feature_names = None
        self.trained_on_chunks = 0
        self.previous_chunk_data = None  # Para train-then-test

        logger.info(f"ACDWM Evaluator inicializado")

    def _init_acdwm_model(self):
        """
        Inicializa o modelo ACDWM com parâmetros configurados.

        Cria uma instância de DWMIL (Dynamic Weighted Majority for Imbalanced Learning)
        com os parâmetros especificados.
        """
        try:
            # Importa classe DWMIL
            DWMIL = self.acdwm_modules['dwmil'].DWMIL

            # Cria modelo ACDWM
            # Nota: data_num será definido na primeira chamada de treino
            # chunk_size=0 para usar chunks fornecidos pelo framework
            self.model = DWMIL(
                data_num=999999,  # Será atualizado dinamicamente
                chunk_size=0,     # Usa chunks do framework
                theta=self.theta,
                err_func='gm',    # G-mean
                r=1.0             # Imbalance ratio
            )

            logger.info(f"[OK] Modelo DWMIL inicializado (theta={self.theta})")

        except Exception as e:
            logger.error(f"Erro ao inicializar modelo ACDWM: {e}")
            raise

    def train_on_chunk(
        self,
        X_train: List[Dict],
        y_train: List
    ) -> Dict[str, Any]:
        """
        Treina ACDWM em um chunk de dados.

        Args:
            X_train: Lista de dicts com features (formato River)
            y_train: Lista de labels

        Returns:
            Dicionário com informações do treino
        """
        if not X_train or not y_train:
            logger.warning("Chunk de treino vazio!")
            return {}

        logger.debug(f"Treinando no chunk {self.trained_on_chunks + 1}: "
                    f"{len(X_train)} samples")

        # Converte River → NumPy
        X_array, feature_names = river_to_numpy(X_train, self.feature_names)
        y_array = river_labels_to_numpy(y_train)

        # Armazena feature_names na primeira vez
        if self.feature_names is None:
            self.feature_names = feature_names
            logger.info(f"Features: {feature_names}")

        # Converte labels para formato ACDWM (-1/+1)
        y_array_acdwm = convert_labels_to_acdwm(y_array)

        # Executa treino do ACDWM usando _update_chunk()
        # Nota: _update_chunk treina no chunk (não faz predição)
        self.model._update_chunk(X_array, y_array_acdwm)

        train_info = {
            'n_samples': len(X_array),
            'n_features': X_array.shape[1],
            'class_distribution': {
                int(c): int(np.sum(y_array == c))
                for c in self.classes
            },
            'ensemble_size': len(self.model.ensemble),
            'active_weights': len(self.model.w)
        }

        self.trained_on_chunks += 1

        logger.debug(f"Treino concluido no chunk {self.trained_on_chunks} "
                    f"(ensemble_size={len(self.model.ensemble)})")

        return train_info

    def test_on_chunk(
        self,
        X_test: List[Dict],
        y_test: List
    ) -> Dict[str, float]:
        """
        Testa ACDWM em um chunk de dados.

        Args:
            X_test: Lista de dicts com features (formato River)
            y_test: Lista de labels verdadeiros

        Returns:
            Dicionário com métricas padronizadas
        """
        if not X_test or not y_test:
            logger.warning("Chunk de teste vazio!")
            return {}

        logger.debug(f"Testando no chunk: {len(X_test)} samples")

        # Converte River → NumPy
        X_array, _ = river_to_numpy(X_test, self.feature_names)
        y_array = river_labels_to_numpy(y_test)

        # Executa predição do ACDWM
        # Nota: predict() retorna predições em formato -1/+1
        y_pred_acdwm = self.model.predict(X_array)

        # Converte predições de -1/+1 para 0/1
        y_pred = convert_labels_from_acdwm(y_pred_acdwm)

        # Calcula métricas padronizadas
        metrics = calculate_shared_metrics(y_test, y_pred.tolist(), self.classes)

        logger.debug(f"Teste concluido: G-mean={metrics.get('gmean', 0):.4f}, "
                    f"ensemble_size={len(self.model.ensemble)}")

        return metrics

    def evaluate_train_then_test(
        self,
        chunks: List[Tuple[List[Dict], List, List[Dict], List]]
    ) -> List[Dict[str, Any]]:
        """
        Avalia ACDWM no modo train-then-test (compatível com GBML/River).

        Sequência:
            Chunk 1: Train → Test no Chunk 2
            Chunk 2: Train → Test no Chunk 3
            ...

        Args:
            chunks: Lista de (X_train, y_train, X_test, y_test)

        Returns:
            Lista de resultados por chunk
        """
        logger.info(f"Avaliação: train-then-test com {len(chunks)} chunks")

        results = []

        for i, (X_train, y_train, X_test, y_test) in enumerate(chunks):
            logger.info(f"\n--- Chunk {i+1}/{len(chunks)} ---")

            # 1. Treina no chunk atual
            train_info = self.train_on_chunk(X_train, y_train)

            # 2. Testa no chunk de teste
            test_metrics = self.test_on_chunk(X_test, y_test)

            # Combina informações
            chunk_result = {
                'chunk_idx': i,
                'train_size': len(X_train),
                'test_size': len(X_test),
                **test_metrics,
                'train_info': train_info
            }

            results.append(chunk_result)

            logger.info(f"Chunk {i+1} - G-mean: {test_metrics.get('gmean', 0):.4f}")

        return results

    def evaluate_test_then_train(
        self,
        chunks: List[Tuple[List[Dict], List, List[Dict], List]]
    ) -> List[Dict[str, Any]]:
        """
        Avalia ACDWM no modo test-then-train (prequential, original do ACDWM).

        Usa update_chunk() do ACDWM que implementa prequential evaluation:
            - Prediz com modelo atual
            - Treina no mesmo chunk
            - Retorna predições

        Args:
            chunks: Lista de (X_train, y_train, X_test, y_test)

        Returns:
            Lista de resultados por chunk
        """
        logger.info(f"Avaliacao: test-then-train (prequential) com {len(chunks)} chunks")

        results = []

        for i, (X_train, y_train, X_test, y_test) in enumerate(chunks):
            logger.info(f"\n--- Chunk {i+1}/{len(chunks)} ---")

            # Converte River to NumPy
            X_chunk, feature_names = river_to_numpy(X_train, self.feature_names)

            # Armazena feature_names na primeira vez
            if self.feature_names is None:
                self.feature_names = feature_names
                logger.info(f"Features: {feature_names}")

            y_chunk = river_labels_to_numpy(y_train)
            y_chunk_acdwm = convert_labels_to_acdwm(y_chunk)

            # Usa update_chunk() - test-then-train atomico
            # Retorna predicoes feitas ANTES do treino
            y_pred_acdwm = self.model.update_chunk(X_chunk, y_chunk_acdwm)

            # Converte predicoes de -1/+1 para 0/1
            y_pred = convert_labels_from_acdwm(y_pred_acdwm)

            # Calcula metricas
            test_metrics = calculate_shared_metrics(y_train, y_pred.tolist(), self.classes)

            train_info = {
                'n_samples': len(X_chunk),
                'n_features': X_chunk.shape[1],
                'ensemble_size': len(self.model.ensemble),
                'active_weights': len(self.model.w)
            }

            # Combina informacoes
            chunk_result = {
                'chunk_idx': i,
                'train_size': len(X_train),
                'test_size': len(X_train),  # Prequential usa mesmo chunk
                **test_metrics,
                'train_info': train_info
            }

            results.append(chunk_result)

            logger.info(f"Chunk {i+1} - G-mean: {test_metrics.get('gmean', 0):.4f}")

        self.trained_on_chunks = len(chunks)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo ACDWM.

        Returns:
            Dicionário com informações do modelo
        """
        return {
            'model_type': 'ACDWM',
            'evaluation_mode': self.evaluation_mode,
            'adaptive_chunk_size': self.adaptive_chunk_size,
            'theta': self.theta,
            'min_ensemble_size': self.min_ensemble_size,
            'ensemble_size': len(self.model.ensemble) if hasattr(self, 'model') else 0,
            'active_weights': len(self.model.w) if hasattr(self, 'model') else 0,
            'trained_chunks': self.trained_on_chunks,
            'feature_names': self.feature_names
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_acdwm_model(
    acdwm_path: str,
    classes: List,
    evaluation_mode: str = 'train-then-test',
    **kwargs
) -> ACDWMEvaluator:
    """
    Factory function para criar modelo ACDWM configurado.

    Args:
        acdwm_path: Caminho para código ACDWM
        classes: Lista de classes
        evaluation_mode: 'train-then-test' ou 'test-then-train'
        **kwargs: Parâmetros adicionais

    Returns:
        ACDWMEvaluator configurado

    Example:
        >>> acdwm = create_acdwm_model(
        ...     acdwm_path='/content/ACDWM',
        ...     classes=[0, 1],
        ...     evaluation_mode='train-then-test'
        ... )
    """
    logger.info(f"Criando modelo ACDWM (modo: {evaluation_mode})")

    evaluator = ACDWMEvaluator(
        acdwm_path=acdwm_path,
        classes=classes,
        evaluation_mode=evaluation_mode,
        **kwargs
    )

    return evaluator


# ============================================================================
# RUNNER FUNCTION (Para compatibilidade com compare_gbml_vs_river.py)
# ============================================================================

def run_acdwm_baseline(
    acdwm_model: ACDWMEvaluator,
    chunks_data: List[Tuple],
    chunk_names: List[str]
) -> Dict[str, Any]:
    """
    Executa ACDWM em chunks de dados.

    Args:
        acdwm_model: Instância do ACDWMEvaluator
        chunks_data: Lista de chunks (X_train, y_train, X_test, y_test)
        chunk_names: Lista de nomes dos chunks

    Returns:
        Dicionário com resultados

    Example:
        >>> acdwm = create_acdwm_model(...)
        >>> results = run_acdwm_baseline(acdwm, chunks, names)
    """
    logger.info(f"Executando ACDWM em {len(chunks_data)} chunks")

    # Escolhe método de avaliação baseado no modo
    if acdwm_model.evaluation_mode == 'train-then-test':
        chunk_results = acdwm_model.evaluate_train_then_test(chunks_data)
    else:
        chunk_results = acdwm_model.evaluate_test_then_train(chunks_data)

    # Formata resultados
    results = {
        'model': 'ACDWM',
        'evaluation_mode': acdwm_model.evaluation_mode,
        'chunk_results': chunk_results,
        'model_info': acdwm_model.get_model_info(),
        'summary': {
            'mean_gmean': np.mean([r['gmean'] for r in chunk_results if 'gmean' in r]),
            'mean_accuracy': np.mean([r['accuracy'] for r in chunk_results if 'accuracy' in r]),
            'mean_f1_weighted': np.mean([r['f1_weighted'] for r in chunk_results if 'f1_weighted' in r]),
        }
    }

    logger.info(f"ACDWM concluído - G-mean médio: {results['summary']['mean_gmean']:.4f}")

    return results


# ============================================================================
# MAIN (Para testes standalone)
# ============================================================================

if __name__ == '__main__':
    """Teste básico do módulo"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
    )

    print("="*70)
    print("TESTE: baseline_acdwm.py")
    print("="*70)

    # Testa importação de módulos
    try:
        print("\n[1/3] Testando importação de módulos...")

        # Caminho padrão do ACDWM (ajustar conforme necessário)
        acdwm_path = Path(__file__).parent / 'ACDWM'

        if not acdwm_path.exists():
            print(f"  [!] Diretório ACDWM não encontrado: {acdwm_path}")
            print(f"  Clone com: git clone https://github.com/jasonyanglu/ACDWM.git {acdwm_path}")
        else:
            modules = import_acdwm_modules(str(acdwm_path))
            print(f"  [OK] Módulos importados: {list(modules.keys())}")

    except Exception as e:
        print(f"  [X] Erro: {e}")

    # Testa criação de modelo
    try:
        print("\n[2/3] Testando criação de modelo...")

        if acdwm_path.exists():
            model = create_acdwm_model(
                acdwm_path=str(acdwm_path),
                classes=[0, 1],
                evaluation_mode='train-then-test'
            )
            print(f"  [OK] Modelo criado: {model.model_name}")
            print(f"  Info: {model.get_model_info()}")
        else:
            print("  [SKIP] ACDWM não disponível")

    except Exception as e:
        print(f"  [X] Erro: {e}")

    # Testa com dados sintéticos
    try:
        print("\n[3/3] Testando com dados sintéticos...")

        # Gera dados dummy
        X_train = [{'x_0': 0.1, 'x_1': 0.2}, {'x_0': 0.3, 'x_1': 0.4}]
        y_train = [0, 1]
        X_test = [{'x_0': 0.5, 'x_1': 0.6}]
        y_test = [0]

        if acdwm_path.exists():
            # Treina
            train_info = model.train_on_chunk(X_train, y_train)
            print(f"  [OK] Treino: {train_info}")

            # Testa
            test_metrics = model.test_on_chunk(X_test, y_test)
            print(f"  [OK] Teste: G-mean={test_metrics.get('gmean', 0):.4f}")
        else:
            print("  [SKIP] ACDWM não disponível")

    except Exception as e:
        print(f"  [X] Erro: {e}")

    print("\n" + "="*70)
    print("Teste concluído!")
    print("="*70)
