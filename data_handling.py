# data_handling.py (Versão Final Consolidada e Compatível com main.py)

import yaml
import numpy as np
import river
from river import stream, datasets, base
import itertools
import logging
import os
import importlib
from typing import Dict, List, Tuple, Any, Iterable, Union

try:
    from custom_generators import AssetNegotiation
except ImportError:
    AssetNegotiation = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("data_handling_final")

CUSTOM_GENERATORS = {'custom_generators.AssetNegotiation': AssetNegotiation}
#CONFIG: Dict | None = None
CONFIG: Union[Dict, None] = None
# --- Funções Auxiliares para o main.py (Reintegradas e Mantidas) ---

def load_full_config(config_path: str = 'config.yaml') -> Union[Dict, None]:
    """Carrega o arquivo de configuração YAML completo e o armazena globalmente."""
    global CONFIG
    if CONFIG is None:
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        try:
            with open(config_path, 'r') as f:
                CONFIG = yaml.safe_load(f)
            logger.info(f"Full configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading/parsing {config_path}: {e}")
            CONFIG = None
    return CONFIG

def get_stream_definition(stream_name: str, config: Dict) -> Union[Dict, None]:
    """Obtém a definição de um stream específico da seção 'experimental_streams'."""
    try:
        return config['experimental_streams'][stream_name]
    except KeyError:
        logger.error(f"Stream definition for '{stream_name}' not found in 'experimental_streams'.")
        return None

def get_concept_for_chunk(stream_definition: Dict, chunk_index: int) -> Tuple[Union[str, None], Union[str, None], float]:
    """Determina o(s) conceito(s) ativo(s) para um determinado chunk."""
    # ... (Esta função, se necessária pelo main.py, deve ser mantida da v8)
    concept_sequence = stream_definition.get('concept_sequence', [])
    drift_type = stream_definition.get('drift_type', 'abrupt')
    gradual_width = stream_definition.get('gradual_drift_width_chunks', 0)
    if not concept_sequence: return None, None, 0.0
    current_chunk_start = 0; previous_concept_id = None
    for i, stage in enumerate(concept_sequence):
        concept_id = str(stage['concept_id']); duration = stage['duration_chunks']
        current_chunk_end = current_chunk_start + duration
        if current_chunk_start <= chunk_index < current_chunk_end:
            if drift_type == 'gradual' and i > 0 and gradual_width > 0 and previous_concept_id is not None:
                transition_start_chunk = current_chunk_start
                transition_end_chunk = current_chunk_start + gradual_width
                if transition_start_chunk <= chunk_index < transition_end_chunk:
                    chunk_pos_in_window = chunk_index - transition_start_chunk
                    mixture_prob_b = (chunk_pos_in_window + 0.5) / gradual_width
                    return str(previous_concept_id), concept_id, max(0.0, min(1.0, mixture_prob_b))
            return concept_id, None, 0.0
        previous_concept_id = concept_id; current_chunk_start = current_chunk_end
    last_concept_id = str(concept_sequence[-1]['concept_id'])
    return last_concept_id, None, 0.0


# --- LÓGICA DE GERAÇÃO DE DADOS (Versão Final e Validada) ---

def get_class_from_string(class_path: str):
    if class_path in CUSTOM_GENERATORS:
        return CUSTOM_GENERATORS[class_path]
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import class '{class_path}'.") from e

class StreamFactory:
    def __init__(self, config: Dict):
        self.config = config
        self.datasets_map = config.get('drift_analysis', {}).get('datasets', {})

    def _get_base_iterator(self, family_name: str, params_override: Dict = None) -> Iterable[Tuple[Dict, Any]]:
        family_config = self.datasets_map.get(family_name)
        if 'loader' in family_config and family_config['loader'] == 'local_csv':
            path, target = family_config.get('source_path'), family_config.get('target_column')
            return stream.iter_csv(path, target=target)
        class_path = family_config.get('class')
        GeneratorClass = get_class_from_string(class_path)
        base_params = family_config.get('params', {})
        if params_override: base_params.update(params_override)
        return GeneratorClass(**base_params)

    def _apply_noise(self, iterator: Iterable, noise_config: Dict) -> Iterable[Tuple[Dict, Any]]:
        noise_level, ratio = noise_config.get('noise_level', 0.1), noise_config.get('attribute_ratio', 0.45)
        first, noisy_features = True, []
        for x, y in iterator:
            if first:
                numeric_features = [k for k, v in x.items() if isinstance(v, (int, float))]
                num_to_affect = int(len(numeric_features) * ratio)
                if num_to_affect > 0: noisy_features = np.random.choice(numeric_features, size=num_to_affect, replace=False)
                first = False
            for feat in noisy_features:
                if feat in x: x[feat] += np.random.normal(0, noise_level)
            yield x, y

# --- FUNÇÃO PRINCIPAL (COM A INTERFACE CORRETA) ---
def generate_dataset_chunks(
    stream_or_dataset_name: str,
    chunk_size: int,
    num_chunks: int,
    max_instances: int,
    config_path: str,
    run_number: int = 1, # Adicionando os parâmetros que faltavam com valores padrão
    results_dir: str = "results"
    ) -> List[Tuple[List[Dict], List]]:
    """
    Função pública e unificada para gerar dados em blocos.
    Compatível com a chamada do main.py.
    """
    config = load_full_config(config_path)
    if config is None:
        raise ValueError("Configuration could not be loaded.")
        
    factory = StreamFactory(config)
    stream_config = config.get('experimental_streams', {}).get(stream_or_dataset_name)
    if not stream_config:
        raise ValueError(f"Stream '{stream_or_dataset_name}' not found in 'experimental_streams'.")

    logger.info(f"--- Generating stream: '{stream_or_dataset_name}' ---")

    is_drift_simulation = 'concept_sequence' in stream_config
    iterators = []
    
    if is_drift_simulation:
        logger.info("Mode: Drift Simulation")
        sequence, family = stream_config['concept_sequence'], stream_config['dataset_type']
        for i, stage in enumerate(sequence):
            concept_id = stage['concept_id']
            concept_params = factory.datasets_map.get(family, {}).get('concepts', {}).get(concept_id)
            if concept_params is None: raise ValueError(f"Concept '{concept_id}' not defined for family '{family}'.")
            
            duration = stage['duration_chunks'] * chunk_size
            drift_type = stage.get('drift_type_override', stream_config.get('drift_type', 'abrupt'))
            width = stream_config.get('gradual_drift_width_chunks', 0) * chunk_size

            if drift_type == 'gradual' and i > 0 and width > 0:
                prev_stage = sequence[i-1]
                prev_params = factory.datasets_map.get(family, {}).get('concepts', {}).get(prev_stage['concept_id'])
                iter_a = iter(factory._get_base_iterator(family, prev_params))
                iter_b = iter(factory._get_base_iterator(family, concept_params))
                transition_iter = (next(iter_b) if np.random.rand() < ((k + 1) / width) else next(iter_a) for k in range(width))
                iterators.append(transition_iter)
                duration -= width
            
            if duration > 0:
                iterators.append(itertools.islice(factory._get_base_iterator(family, concept_params), duration))
    else: # Modo Estacionário
        logger.info("Mode: Stationary")
        family, params = stream_config['dataset_type'], stream_config.get('params_override')
        total_instances = chunk_size * num_chunks
        base_iterator = factory._get_base_iterator(family, params)
        iterators.append(itertools.islice(base_iterator, total_instances))

    full_stream_iterator = itertools.chain(*iterators)

    if stream_config.get('noise_config', {}).get('enabled', False):
        full_stream_iterator = factory._apply_noise(full_stream_iterator, stream_config['noise_config'])

    chunks = []
    while True:
        chunk_slice = list(itertools.islice(full_stream_iterator, chunk_size))
        if not chunk_slice: break
        X_chunk, y_chunk = zip(*chunk_slice)
        chunks.append((list(X_chunk), list(y_chunk)))
        if len(chunks) == num_chunks: break
    
    logger.info(f"Generation for '{stream_or_dataset_name}' complete. Total chunks: {len(chunks)}.")
    return chunks