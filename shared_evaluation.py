# shared_evaluation.py
"""
Módulo Unificado de Avaliação para GBML e River

PROPÓSITO: Garantir que ambos os frameworks usem EXATAMENTE os mesmos dados,
           divisões e metodologia de avaliação para comparação científica válida.

AUTOR: Claude Code
DATA: 2025-01-06
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, f1_score
from metrics import calculate_gmean_contextual

# Configuração de logging
logger = logging.getLogger("shared_evaluation")


# ============================================================================
# PARTE 1: GERENCIAMENTO DE DADOS CENTRALIZADO
# ============================================================================

class ChunkCache:
    """
    Gerencia cache de chunks para garantir reprodutibilidade e evitar
    recomputação desnecessária.
    """

    def __init__(self, cache_dir: str = "chunks_cache"):
        """
        Args:
            cache_dir: Diretório para armazenar chunks em cache
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"ChunkCache inicializado em: {self.cache_dir}")

    def get_cache_path(self, stream_name: str, chunk_size: int,
                       num_chunks: int, seed: int) -> str:
        """Gera path único para combinação de parâmetros"""
        filename = f"{stream_name}_cs{chunk_size}_nc{num_chunks}_seed{seed}.pkl"
        return os.path.join(self.cache_dir, filename)

    def exists(self, stream_name: str, chunk_size: int,
               num_chunks: int, seed: int) -> bool:
        """Verifica se chunks já estão em cache"""
        return os.path.exists(self.get_cache_path(stream_name, chunk_size,
                                                   num_chunks, seed))

    def save(self, chunks: List[Tuple[List[Dict], List]],
             stream_name: str, chunk_size: int, num_chunks: int, seed: int):
        """Salva chunks em cache"""
        cache_path = self.get_cache_path(stream_name, chunk_size, num_chunks, seed)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(chunks, f)
            logger.info(f"✓ Chunks salvos em cache: {cache_path}")
        except Exception as e:
            logger.error(f"✗ Erro ao salvar cache: {e}")

    def load(self, stream_name: str, chunk_size: int,
             num_chunks: int, seed: int) -> Optional[List[Tuple[List[Dict], List]]]:
        """Carrega chunks do cache"""
        cache_path = self.get_cache_path(stream_name, chunk_size, num_chunks, seed)
        try:
            with open(cache_path, 'rb') as f:
                chunks = pickle.load(f)
            logger.info(f"✓ Chunks carregados do cache: {cache_path}")
            return chunks
        except Exception as e:
            logger.error(f"✗ Erro ao carregar cache: {e}")
            return None


def load_or_generate_chunks(
    stream_name: str,
    chunk_size: int,
    num_chunks: int,
    max_instances: int,
    config_path: str,
    seed: int = 42,
    force_regenerate: bool = False,
    use_cache: bool = True
) -> List[Tuple[List[Dict], List]]:
    """
    Função central para carregar ou gerar chunks.

    GARANTE: Mesmos chunks para GBML e River quando chamada com mesmos parâmetros.

    Args:
        stream_name: Nome do stream (e.g., 'SEA_Abrupt_Simple')
        chunk_size: Tamanho de cada chunk
        num_chunks: Número de chunks a gerar
        max_instances: Máximo de instâncias
        config_path: Caminho para config.yaml
        seed: Seed para reprodutibilidade
        force_regenerate: Se True, força regeneração mesmo com cache
        use_cache: Se False, não usa cache (útil para debug)

    Returns:
        Lista de chunks: [(X_chunk, y_chunk), ...]
    """
    from data_handling import generate_dataset_chunks

    cache = ChunkCache() if use_cache else None

    # Tenta carregar do cache primeiro
    if cache and not force_regenerate:
        if cache.exists(stream_name, chunk_size, num_chunks, seed):
            logger.info(f"[CACHE HIT] Carregando chunks para '{stream_name}' (seed={seed})")
            cached_chunks = cache.load(stream_name, chunk_size, num_chunks, seed)
            if cached_chunks is not None:
                return cached_chunks
            logger.warning("Cache corrompido, regenerando...")

    # Gera chunks se não estiver em cache
    logger.info(f"[CACHE MISS] Gerando chunks para '{stream_name}' (seed={seed})")

    # CRÍTICO: Define seed ANTES de gerar para garantir reprodutibilidade
    np.random.seed(seed)

    chunks = generate_dataset_chunks(
        stream_or_dataset_name=stream_name,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        max_instances=max_instances,
        config_path=config_path,
        run_number=seed  # Usa seed como run_number para consistência
    )

    # Salva no cache para reutilização
    if cache:
        cache.save(chunks, stream_name, chunk_size, num_chunks, seed)

    logger.info(f"✓ Total de {len(chunks)} chunks gerados/carregados para '{stream_name}'")
    return chunks


# ============================================================================
# PARTE 2: INTERFACE ABSTRATA DE AVALIAÇÃO
# ============================================================================

class ChunkEvaluator(ABC):
    """
    Classe abstrata que define a interface comum para avaliadores.

    GBML e River devem implementar esta interface para garantir
    avaliação consistente.
    """

    def __init__(self, model_name: str, classes: List):
        """
        Args:
            model_name: Nome do modelo (para logging/relatórios)
            classes: Lista de classes possíveis
        """
        self.model_name = model_name
        self.classes = classes
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{model_name}]")

    @abstractmethod
    def train_on_chunk(self, X_train: List[Dict], y_train: List) -> Dict[str, Any]:
        """
        Treina o modelo em um chunk de dados.

        Args:
            X_train: Features do chunk de treino
            y_train: Labels do chunk de treino

        Returns:
            Dicionário com métricas de treino (opcional)
        """
        pass

    @abstractmethod
    def test_on_chunk(self, X_test: List[Dict], y_test: List) -> Dict[str, float]:
        """
        Testa o modelo em um chunk de dados.

        Args:
            X_test: Features do chunk de teste
            y_test: Labels do chunk de teste

        Returns:
            Dicionário com métricas: {'accuracy', 'f1', 'gmean', ...}
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o estado atual do modelo.

        Returns:
            Dicionário com info relevante (e.g., número de regras, árvores, etc.)
        """
        pass

    def evaluate_prequential(
        self,
        chunks: List[Tuple[List[Dict], List]],
        evaluation_period: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Avaliação prequential (test-then-train) padrão.

        METODOLOGIA:
        - Treina no chunk i
        - Testa no chunk i+1
        - (Opcional) Avalia periodicamente dentro do chunk de teste

        Args:
            chunks: Lista de chunks gerados por load_or_generate_chunks()
            evaluation_period: Se fornecido, avalia a cada k instâncias

        Returns:
            DataFrame com resultados por chunk
        """
        results = []

        for i in range(len(chunks) - 1):
            train_chunk_idx = i
            test_chunk_idx = i + 1

            X_train, y_train = chunks[train_chunk_idx]
            X_test, y_test = chunks[test_chunk_idx]

            self.logger.info(f"[PREQUENTIAL] Treinando no chunk {train_chunk_idx} ({len(X_train)} instâncias)...")
            # Tenta passar chunk_index se o método aceitar (para salvamento de indivíduos)
            try:
                train_metrics = self.train_on_chunk(X_train, y_train, chunk_index=train_chunk_idx)
            except TypeError:
                # Método não aceita chunk_index (ex: River models)
                train_metrics = self.train_on_chunk(X_train, y_train)

            self.logger.info(f"[PREQUENTIAL] Testando no chunk {test_chunk_idx} ({len(X_test)} instâncias)...")
            test_metrics = self.test_on_chunk(X_test, y_test)

            # Obtém informações do modelo
            model_info = self.get_model_info()

            # Combina todas as métricas (registra chunk de TESTE como referência)
            chunk_result = {
                'train_chunk': train_chunk_idx,
                'test_chunk': test_chunk_idx,
                'chunk': test_chunk_idx,  # Mantém 'chunk' para compatibilidade (refere-se ao teste)
                'model': self.model_name,
                **test_metrics,
                **model_info
            }

            results.append(chunk_result)

            self.logger.info(
                f"[RESULTADO] Treinou chunk {train_chunk_idx} → Testou chunk {test_chunk_idx} | "
                f"Acc: {test_metrics.get('accuracy', 0):.4f}, "
                f"F1: {test_metrics.get('f1_weighted', 0):.4f}, "
                f"G-mean: {test_metrics.get('gmean', 0):.4f}"
            )

        return pd.DataFrame(results)


# ============================================================================
# PARTE 3: FUNÇÕES UTILITÁRIAS COMPARTILHADAS
# ============================================================================

def calculate_shared_metrics(
    y_true: List,
    y_pred: List,
    classes: List
) -> Dict[str, float]:
    """
    Calcula métricas padronizadas para comparação.

    GARANTE: Mesmas métricas calculadas de forma idêntica para GBML e River.

    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        classes: Lista de todas as classes possíveis

    Returns:
        Dicionário com métricas padronizadas
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, labels=classes,
                                average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, labels=classes,
                            average='macro', zero_division=0),
        'gmean': calculate_gmean_contextual(y_true, y_pred, classes)
    }


def validate_chunks_consistency(
    chunks: List[Tuple[List[Dict], List]]
) -> bool:
    """
    Valida se os chunks estão consistentes (sem NaN, tipos corretos, etc.)

    Args:
        chunks: Lista de chunks a validar

    Returns:
        True se válidos, False caso contrário
    """
    try:
        for i, (X, y) in enumerate(chunks):
            # Verifica se não está vazio
            if not X or not y:
                logger.error(f"Chunk {i} está vazio!")
                return False

            # Verifica se tamanhos batem
            if len(X) != len(y):
                logger.error(f"Chunk {i}: len(X)={len(X)} != len(y)={len(y)}")
                return False

            # Verifica se não há None em y
            if None in y or any(pd.isna(label) for label in y):
                logger.error(f"Chunk {i} contém labels None/NaN")
                return False

        logger.info(f"✓ Validação de {len(chunks)} chunks passou!")
        return True

    except Exception as e:
        logger.error(f"✗ Erro na validação de chunks: {e}")
        return False


# ============================================================================
# PARTE 4: ANÁLISE COMPARATIVA
# ============================================================================

def compare_results(
    results_gbml: pd.DataFrame,
    results_river: pd.DataFrame,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compara resultados de GBML e River lado a lado.

    Args:
        results_gbml: DataFrame de resultados do GBML
        results_river: DataFrame de resultados do River
        save_path: Se fornecido, salva tabela comparativa

    Returns:
        DataFrame com comparação lado a lado
    """
    # Mescla baseado no chunk
    comparison = pd.merge(
        results_gbml,
        results_river,
        on='chunk',
        suffixes=('_gbml', '_river')
    )

    # Calcula diferenças
    for metric in ['accuracy', 'f1_weighted', 'gmean']:
        if f'{metric}_gbml' in comparison.columns and f'{metric}_river' in comparison.columns:
            comparison[f'{metric}_diff'] = (
                comparison[f'{metric}_gbml'] - comparison[f'{metric}_river']
            )

    # Salva se solicitado
    if save_path:
        comparison.to_csv(save_path, index=False)
        logger.info(f"✓ Comparação salva em: {save_path}")

    return comparison


if __name__ == "__main__":
    # Teste básico do módulo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
    )

    logger.info("=== Teste do Módulo shared_evaluation.py ===")

    # Teste 1: Cache
    cache = ChunkCache()
    logger.info(f"✓ ChunkCache inicializado em: {cache.cache_dir}")

    # Teste 2: Validação de chunks fictícios
    fake_chunks = [
        ([{'x1': 1.0, 'x2': 2.0}], [0]),
        ([{'x1': 1.5, 'x2': 2.5}], [1])
    ]
    is_valid = validate_chunks_consistency(fake_chunks)
    logger.info(f"✓ Validação de chunks: {'PASSOU' if is_valid else 'FALHOU'}")

    logger.info("=== Módulo shared_evaluation.py pronto para uso ===")
