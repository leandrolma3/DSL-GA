# test_all_streams.py (Versão Final com Verificação de Arquivos Locais)

import yaml
import numpy as np
import logging
import os

from data_handling import generate_dataset_chunks

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("Universal_Stream_Validator")

def validate_chunks(stream_name: str, chunks: list):
    # ... (a função de validação continua exatamente a mesma da versão anterior)
    logger.info(f"--- Running Validations for '{stream_name}' ---")

    if not chunks or not chunks[0][0]:
        logger.error("[VALIDATION FAILED] No chunks or instances were generated.")
        return False
    logger.info(f"[Validation] Generated {len(chunks)} chunks.")

    first_chunk_x, first_chunk_y = chunks[0]
    if not (isinstance(first_chunk_x, list) and isinstance(first_chunk_y, list)):
        logger.error("[VALIDATION FAILED] Chunk structure is not a tuple of two lists.")
        return False

    if not isinstance(first_chunk_x[0], dict):
        logger.error(f"[VALIDATION FAILED] Instance type is not dict, but {type(first_chunk_x[0])}.")
        return False

    logger.info("[Validation] Base structure and types are correct.")
    logger.info(f"--- Validations for '{stream_name}' PASSED ---")
    return True

def main():
    """Carrega o config e testa dinamicamente todos os streams definidos."""
    CONFIG_FILE = 'config.yaml'
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        experimental_streams = config.get('experimental_streams', {})
        datasets_definitions = config.get('drift_analysis', {}).get('datasets', {})
    except Exception as e:
        logger.error(f"Failed to load or parse {CONFIG_FILE}: {e}")
        return

    total_streams = len(experimental_streams)
    logger.info(f"Found {total_streams} streams to validate.")
    failed_streams = []
    skipped_streams = []
    
    NUM_CHUNKS_TO_TEST = 2
    CHUNK_SIZE_TO_TEST = 100

    for i, (stream_name, stream_config) in enumerate(experimental_streams.items()):
        print("\n" + "="*80)
        logger.info(f"TESTING STREAM [{i+1}/{total_streams}]: '{stream_name}'")
        print("="*80)
        
        # <<< MUDANÇA PRINCIPAL AQUI >>>
        # Verifica se o stream é do tipo 'local_csv' e se o arquivo existe
        dataset_type = stream_config.get('dataset_type')
        definition = datasets_definitions.get(dataset_type, {})
        
        if definition.get('loader') == 'local_csv':
            path = definition.get('source_path')
            if not path or not os.path.exists(path):
                logger.warning(f"[SKIPPED] Test for '{stream_name}' was skipped because the data file was not found at: {path}")
                skipped_streams.append(stream_name)
                continue # Pula para o próximo stream
        # <<< FIM DA MUDANÇA >>>

        try:
            chunks = generate_dataset_chunks(
                stream_name=stream_name,
                chunk_size=CHUNK_SIZE_TO_TEST,
                num_chunks=NUM_CHUNKS_TO_TEST,
                config_path=CONFIG_FILE
            )
            
            if not validate_chunks(stream_name, chunks):
                failed_streams.append(stream_name)

        except Exception as e:
            logger.exception(f"FATAL ERROR during generation of stream '{stream_name}'.")
            failed_streams.append(stream_name)

    # Relatório final
    print("\n" + "="*80)
    logger.info("COMPREHENSIVE TEST SUITE FINISHED")
    print("="*80)
    
    if not failed_streams and not skipped_streams:
        logger.info(f"✅ SUCCESS: All {total_streams} streams were generated and validated successfully!")
    else:
        if skipped_streams:
            logger.warning(f"🟡 SKIPPED: {len(skipped_streams)} streams were skipped due to missing local data files:")
            for name in skipped_streams:
                logger.warning(f"  - {name}")
        if failed_streams:
            logger.error(f"❌ FAILED: {len(failed_streams)} streams encountered fatal errors during generation:")
            for name in failed_streams:
                logger.error(f"  - {name}")

if __name__ == '__main__':
    main()