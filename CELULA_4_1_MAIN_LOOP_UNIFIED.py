# =============================================================================
# CELULA 4.1: Loop Principal de Execucao (Unified COM PENALTY)
# =============================================================================
# SUBSTITUA a CELULA 4.1 original por este codigo
# Loop adaptado para unified_chunks com suporte a com/sem penalidade
#
# LOGICA:
# - Para configs SEM penalty: executa EGIS + todos modelos comparativos
# - Para configs COM penalty: carrega EGIS_Penalty + reutiliza modelos comparativos
# =============================================================================

import time
import numpy as np
from pathlib import Path

print("="*70)
print("CELULA 4.1: EXECUCAO TODOS OS MODELOS (UNIFIED COM PENALTY)")
print("="*70)

# Lista para armazenar todos os resultados
ALL_RESULTS = []

# Rastrear quais datasets ja tiveram modelos comparativos executados
# (para evitar executar duas vezes - uma vez basta, pois os dados sao os mesmos)
EXECUTED_COMPARATIVE = set()

total_start = time.time()
dataset_count = 0

# Contar total de datasets
total_datasets = 0
for chunk_size_name in CHUNK_SIZES_TO_RUN:
    for batch_name in BATCHES.get(chunk_size_name, []):
        datasets = get_datasets_for_batch(chunk_size_name, batch_name)
        total_datasets += len(datasets)

print(f"Total de datasets a processar: {total_datasets}")
print(f"Chunk sizes: {CHUNK_SIZES_TO_RUN}")
print(f"Modelos: {MODELS_TO_RUN}")

# =============================================================================
# LOOP PRINCIPAL: chunk_size -> batch -> dataset -> modelo
# =============================================================================

for chunk_size_name in CHUNK_SIZES_TO_RUN:
    config = CHUNK_CONFIGS[chunk_size_name]
    chunk_size = config['size']
    is_penalty = config['penalty']
    egis_model_name = config['egis_model_name']
    base_chunk_size = chunk_size_name.replace('_penalty', '')

    penalty_str = "(COM penalty)" if is_penalty else "(SEM penalty)"

    print(f"\n{'#'*80}")
    print(f"# CHUNK SIZE: {chunk_size_name} {penalty_str}")
    print(f"# Tamanho: {chunk_size} instancias/chunk")
    print(f"# EGIS Model: {egis_model_name}")
    print(f"{'#'*80}")

    batches = BATCHES.get(chunk_size_name, [])

    for batch_name in batches:
        datasets = get_datasets_for_batch(chunk_size_name, batch_name)

        print(f"\n{'='*80}")
        print(f"BATCH: {batch_name} | CONFIG: {chunk_size_name} | DATASETS: {len(datasets)}")
        print(f"{'='*80}")

        for dataset_name in datasets:
            dataset_count += 1
            print(f"\n[{dataset_count}/{total_datasets}] {dataset_name}")

            # Chave para rastrear execucao de modelos comparativos
            comparative_key = f"{base_chunk_size}_{batch_name}_{dataset_name}"

            # Path para salvar resultados (sempre na pasta sem penalty para comparativos)
            dataset_results_dir = get_results_path(chunk_size_name, batch_name, dataset_name, 'COMPARATIVE')

            # ================================================================
            # Carregar chunks de unified_chunks
            # ================================================================
            X_chunks, y_chunks = load_chunks_from_unified(chunk_size_name, dataset_name)
            chunks_available = X_chunks is not None

            if chunks_available:
                n_classes = len(np.unique(np.concatenate(y_chunks)))
                actual_chunk_size = len(X_chunks[0]) if X_chunks else chunk_size
                print(f"  Chunks: {len(X_chunks)} | Samples/chunk: ~{actual_chunk_size} | Classes: {n_classes}")
            else:
                n_classes = 0
                actual_chunk_size = chunk_size
                print(f"  [AVISO] Chunks nao encontrados")

            # ================================================================
            # 1. EGIS (GBML ou GBML_Penalty) - Sempre carrega
            # ================================================================
            egis = load_egis_results(chunk_size_name, batch_name, dataset_name)

            if egis:
                ALL_RESULTS.append({
                    'chunk_size': chunk_size_name,
                    'batch': batch_name,
                    'dataset': dataset_name,
                    'model': egis_model_name,  # 'GBML' ou 'GBML_Penalty'
                    'gmean': egis['gmean'],
                    'f1': egis.get('f1', 0.0),
                    'accuracy': 0.0,
                    'status': 'OK',
                    'penalty': is_penalty,
                })
                print(f"  {egis_model_name}: {egis['gmean']:.4f} (OK)")
            else:
                ALL_RESULTS.append({
                    'chunk_size': chunk_size_name,
                    'batch': batch_name,
                    'dataset': dataset_name,
                    'model': egis_model_name,
                    'gmean': 0.0,
                    'f1': 0.0,
                    'accuracy': 0.0,
                    'status': 'NOT_FOUND',
                    'penalty': is_penalty,
                })
                print(f"  {egis_model_name}: 0.0000 (NOT_FOUND)")

            # ================================================================
            # MODELOS COMPARATIVOS
            # Se ja executamos para este dataset (na versao sem penalty),
            # apenas reutilizamos os resultados
            # ================================================================
            already_executed = comparative_key in EXECUTED_COMPARATIVE

            if is_penalty and already_executed:
                print(f"  [INFO] Reutilizando resultados comparativos de {base_chunk_size}")

            # ================================================================
            # 2. CDCMS - Carrega de cache (ja executado separadamente)
            # ================================================================
            if 'CDCMS' in MODELS_TO_RUN:
                cdcms = load_cdcms_results(chunk_size_name, batch_name, dataset_name)

                if cdcms:
                    ALL_RESULTS.append({
                        'chunk_size': chunk_size_name,
                        'batch': batch_name,
                        'dataset': dataset_name,
                        'model': 'CDCMS',
                        'gmean': cdcms['gmean'],
                        'accuracy': 0.0,
                        'status': 'OK',
                        'penalty': is_penalty,
                    })
                    print(f"  CDCMS: {cdcms['gmean']:.4f} (cached)")
                elif is_multiclass_dataset(dataset_name):
                    ALL_RESULTS.append({
                        'chunk_size': chunk_size_name,
                        'batch': batch_name,
                        'dataset': dataset_name,
                        'model': 'CDCMS',
                        'gmean': 0.0,
                        'accuracy': 0.0,
                        'status': 'N/A',
                        'penalty': is_penalty,
                        'note': f'Multiclass ({MULTICLASS_DATASETS.get(dataset_name, "?")} classes)'
                    })
                    print(f"  CDCMS: N/A (multiclass)")
                else:
                    ALL_RESULTS.append({
                        'chunk_size': chunk_size_name,
                        'batch': batch_name,
                        'dataset': dataset_name,
                        'model': 'CDCMS',
                        'gmean': 0.0,
                        'accuracy': 0.0,
                        'status': 'NOT_FOUND',
                        'penalty': is_penalty,
                    })
                    print(f"  CDCMS: 0.0000 (NOT_FOUND)")

            # ================================================================
            # 3. ROSE_Original - Executa ou carrega cache
            # ================================================================
            if 'ROSE_Original' in MODELS_TO_RUN:
                # Se ja executou na versao sem penalty, carregar cache
                if is_penalty or already_executed:
                    cached = load_existing_model_results(chunk_size_name, batch_name, dataset_name, 'ROSE_Original')
                    if cached:
                        ALL_RESULTS.append({
                            'chunk_size': chunk_size_name,
                            'batch': batch_name,
                            'dataset': dataset_name,
                            'model': 'ROSE_Original',
                            'gmean': cached['gmean'],
                            'accuracy': 0.0,
                            'status': 'CACHED',
                            'penalty': is_penalty,
                        })
                        print(f"  ROSE_Original: {cached['gmean']:.4f} (cached)")
                    else:
                        # Nao tem cache, precisa executar
                        if chunks_available and not is_penalty:
                            try:
                                X_all = np.vstack(X_chunks)
                                y_all = np.concatenate(y_chunks)

                                arff_dir = dataset_results_dir / "rose_arff"
                                arff_file = arff_dir / f"{dataset_name}.arff"
                                create_arff_file(X_all, y_all, arff_file, relation_name=dataset_name)

                                rose_output = dataset_results_dir / "rose_original_output"
                                success, rose_results = run_rose_original(
                                    arff_file, rose_output,
                                    n_classes=n_classes,
                                    timeout=MODEL_TIMEOUT['ROSE_Original']
                                )

                                gmean = rose_results.get('gmean', 0.0)
                                ALL_RESULTS.append({
                                    'chunk_size': chunk_size_name,
                                    'batch': batch_name,
                                    'dataset': dataset_name,
                                    'model': 'ROSE_Original',
                                    'gmean': gmean,
                                    'accuracy': rose_results.get('accuracy', 0.0),
                                    'status': 'OK' if success else 'FAILED',
                                    'penalty': is_penalty,
                                })
                                print(f"  ROSE_Original: {gmean:.4f} ({'OK' if success else 'FAILED'})")

                            except Exception as e:
                                ALL_RESULTS.append({
                                    'chunk_size': chunk_size_name,
                                    'batch': batch_name,
                                    'dataset': dataset_name,
                                    'model': 'ROSE_Original',
                                    'gmean': 0.0,
                                    'accuracy': 0.0,
                                    'status': 'ERROR',
                                    'penalty': is_penalty,
                                })
                                print(f"  ROSE_Original: 0.0000 (ERROR)")
                        else:
                            ALL_RESULTS.append({
                                'chunk_size': chunk_size_name,
                                'batch': batch_name,
                                'dataset': dataset_name,
                                'model': 'ROSE_Original',
                                'gmean': 0.0,
                                'accuracy': 0.0,
                                'status': 'NO_DATA',
                                'penalty': is_penalty,
                            })
                            print(f"  ROSE_Original: 0.0000 (NO_DATA)")
                else:
                    # Primeira execucao (sem penalty)
                    if chunks_available:
                        try:
                            X_all = np.vstack(X_chunks)
                            y_all = np.concatenate(y_chunks)

                            arff_dir = dataset_results_dir / "rose_arff"
                            arff_file = arff_dir / f"{dataset_name}.arff"
                            create_arff_file(X_all, y_all, arff_file, relation_name=dataset_name)

                            rose_output = dataset_results_dir / "rose_original_output"
                            success, rose_results = run_rose_original(
                                arff_file, rose_output,
                                n_classes=n_classes,
                                timeout=MODEL_TIMEOUT['ROSE_Original']
                            )

                            gmean = rose_results.get('gmean', 0.0)
                            ALL_RESULTS.append({
                                'chunk_size': chunk_size_name,
                                'batch': batch_name,
                                'dataset': dataset_name,
                                'model': 'ROSE_Original',
                                'gmean': gmean,
                                'accuracy': rose_results.get('accuracy', 0.0),
                                'status': 'OK' if success else 'FAILED',
                                'penalty': is_penalty,
                            })
                            print(f"  ROSE_Original: {gmean:.4f} ({'OK' if success else 'FAILED'})")

                        except Exception as e:
                            ALL_RESULTS.append({
                                'chunk_size': chunk_size_name,
                                'batch': batch_name,
                                'dataset': dataset_name,
                                'model': 'ROSE_Original',
                                'gmean': 0.0,
                                'accuracy': 0.0,
                                'status': 'ERROR',
                                'penalty': is_penalty,
                            })
                            print(f"  ROSE_Original: 0.0000 (ERROR)")
                    else:
                        ALL_RESULTS.append({
                            'chunk_size': chunk_size_name,
                            'batch': batch_name,
                            'dataset': dataset_name,
                            'model': 'ROSE_Original',
                            'gmean': 0.0,
                            'accuracy': 0.0,
                            'status': 'NO_DATA',
                            'penalty': is_penalty,
                        })
                        print(f"  ROSE_Original: 0.0000 (NO_DATA)")

            # ================================================================
            # 4. ROSE_ChunkEval - Similar ao ROSE_Original
            # ================================================================
            if 'ROSE_ChunkEval' in MODELS_TO_RUN:
                if is_penalty or already_executed:
                    cached = load_existing_model_results(chunk_size_name, batch_name, dataset_name, 'ROSE_ChunkEval')
                    if cached:
                        ALL_RESULTS.append({
                            'chunk_size': chunk_size_name,
                            'batch': batch_name,
                            'dataset': dataset_name,
                            'model': 'ROSE_ChunkEval',
                            'gmean': cached['gmean'],
                            'accuracy': 0.0,
                            'status': 'CACHED',
                            'penalty': is_penalty,
                        })
                        print(f"  ROSE_ChunkEval: {cached['gmean']:.4f} (cached)")
                    elif chunks_available and not is_penalty:
                        try:
                            arff_dir = dataset_results_dir / "rose_arff"
                            arff_file = arff_dir / f"{dataset_name}.arff"

                            if not arff_file.exists():
                                X_all = np.vstack(X_chunks)
                                y_all = np.concatenate(y_chunks)
                                create_arff_file(X_all, y_all, arff_file, relation_name=dataset_name)

                            rose_output = dataset_results_dir / "rose_chunk_eval_output"
                            success, rose_results = run_rose_chunk_eval(
                                arff_file, rose_output,
                                n_classes=n_classes,
                                chunk_size=actual_chunk_size,
                                n_chunks=len(X_chunks),
                                timeout=MODEL_TIMEOUT['ROSE_ChunkEval']
                            )

                            gmean = rose_results.get('gmean', 0.0)
                            ALL_RESULTS.append({
                                'chunk_size': chunk_size_name,
                                'batch': batch_name,
                                'dataset': dataset_name,
                                'model': 'ROSE_ChunkEval',
                                'gmean': gmean,
                                'accuracy': rose_results.get('accuracy', 0.0),
                                'status': 'OK' if success else 'FAILED',
                                'penalty': is_penalty,
                            })
                            print(f"  ROSE_ChunkEval: {gmean:.4f} ({'OK' if success else 'FAILED'})")
                        except Exception as e:
                            ALL_RESULTS.append({
                                'chunk_size': chunk_size_name,
                                'batch': batch_name,
                                'dataset': dataset_name,
                                'model': 'ROSE_ChunkEval',
                                'gmean': 0.0,
                                'status': 'ERROR',
                                'penalty': is_penalty,
                            })
                            print(f"  ROSE_ChunkEval: 0.0000 (ERROR)")
                    else:
                        ALL_RESULTS.append({
                            'chunk_size': chunk_size_name,
                            'batch': batch_name,
                            'dataset': dataset_name,
                            'model': 'ROSE_ChunkEval',
                            'gmean': 0.0,
                            'status': 'NO_DATA',
                            'penalty': is_penalty,
                        })
                        print(f"  ROSE_ChunkEval: 0.0000 (NO_DATA)")
                else:
                    if chunks_available:
                        try:
                            arff_dir = dataset_results_dir / "rose_arff"
                            arff_file = arff_dir / f"{dataset_name}.arff"

                            if not arff_file.exists():
                                X_all = np.vstack(X_chunks)
                                y_all = np.concatenate(y_chunks)
                                create_arff_file(X_all, y_all, arff_file, relation_name=dataset_name)

                            rose_output = dataset_results_dir / "rose_chunk_eval_output"
                            success, rose_results = run_rose_chunk_eval(
                                arff_file, rose_output,
                                n_classes=n_classes,
                                chunk_size=actual_chunk_size,
                                n_chunks=len(X_chunks),
                                timeout=MODEL_TIMEOUT['ROSE_ChunkEval']
                            )

                            gmean = rose_results.get('gmean', 0.0)
                            ALL_RESULTS.append({
                                'chunk_size': chunk_size_name,
                                'batch': batch_name,
                                'dataset': dataset_name,
                                'model': 'ROSE_ChunkEval',
                                'gmean': gmean,
                                'accuracy': rose_results.get('accuracy', 0.0),
                                'status': 'OK' if success else 'FAILED',
                                'penalty': is_penalty,
                            })
                            print(f"  ROSE_ChunkEval: {gmean:.4f} ({'OK' if success else 'FAILED'})")
                        except Exception as e:
                            ALL_RESULTS.append({
                                'chunk_size': chunk_size_name,
                                'batch': batch_name,
                                'dataset': dataset_name,
                                'model': 'ROSE_ChunkEval',
                                'gmean': 0.0,
                                'status': 'ERROR',
                                'penalty': is_penalty,
                            })
                            print(f"  ROSE_ChunkEval: 0.0000 (ERROR)")
                    else:
                        ALL_RESULTS.append({
                            'chunk_size': chunk_size_name,
                            'batch': batch_name,
                            'dataset': dataset_name,
                            'model': 'ROSE_ChunkEval',
                            'gmean': 0.0,
                            'status': 'NO_DATA',
                            'penalty': is_penalty,
                        })
                        print(f"  ROSE_ChunkEval: 0.0000 (NO_DATA)")

            # ================================================================
            # 5. River Models (HAT, ARF, SRP)
            # ================================================================
            for river_model in ['HAT', 'ARF', 'SRP']:
                if river_model in MODELS_TO_RUN:
                    if is_penalty or already_executed:
                        cached = load_existing_model_results(chunk_size_name, batch_name, dataset_name, river_model)
                        if cached:
                            ALL_RESULTS.append({
                                'chunk_size': chunk_size_name,
                                'batch': batch_name,
                                'dataset': dataset_name,
                                'model': river_model,
                                'gmean': cached['gmean'],
                                'status': 'CACHED',
                                'penalty': is_penalty,
                            })
                            print(f"  {river_model}: {cached['gmean']:.4f} (cached)")
                        elif chunks_available and not is_penalty:
                            try:
                                success, results = run_river_model(
                                    river_model, X_chunks, y_chunks,
                                    timeout=MODEL_TIMEOUT[river_model]
                                )
                                gmean = results.get('gmean', 0.0)
                                ALL_RESULTS.append({
                                    'chunk_size': chunk_size_name,
                                    'batch': batch_name,
                                    'dataset': dataset_name,
                                    'model': river_model,
                                    'gmean': gmean,
                                    'status': 'OK' if success else 'FAILED',
                                    'penalty': is_penalty,
                                })
                                print(f"  {river_model}: {gmean:.4f} ({'OK' if success else 'FAILED'})")
                            except Exception as e:
                                ALL_RESULTS.append({
                                    'chunk_size': chunk_size_name,
                                    'batch': batch_name,
                                    'dataset': dataset_name,
                                    'model': river_model,
                                    'gmean': 0.0,
                                    'status': 'ERROR',
                                    'penalty': is_penalty,
                                })
                                print(f"  {river_model}: 0.0000 (ERROR)")
                        else:
                            ALL_RESULTS.append({
                                'chunk_size': chunk_size_name,
                                'batch': batch_name,
                                'dataset': dataset_name,
                                'model': river_model,
                                'gmean': 0.0,
                                'status': 'NO_DATA',
                                'penalty': is_penalty,
                            })
                            print(f"  {river_model}: 0.0000 (NO_DATA)")
                    else:
                        if chunks_available:
                            try:
                                success, results = run_river_model(
                                    river_model, X_chunks, y_chunks,
                                    timeout=MODEL_TIMEOUT[river_model]
                                )
                                gmean = results.get('gmean', 0.0)
                                ALL_RESULTS.append({
                                    'chunk_size': chunk_size_name,
                                    'batch': batch_name,
                                    'dataset': dataset_name,
                                    'model': river_model,
                                    'gmean': gmean,
                                    'status': 'OK' if success else 'FAILED',
                                    'penalty': is_penalty,
                                })
                                print(f"  {river_model}: {gmean:.4f} ({'OK' if success else 'FAILED'})")
                            except Exception as e:
                                ALL_RESULTS.append({
                                    'chunk_size': chunk_size_name,
                                    'batch': batch_name,
                                    'dataset': dataset_name,
                                    'model': river_model,
                                    'gmean': 0.0,
                                    'status': 'ERROR',
                                    'penalty': is_penalty,
                                })
                                print(f"  {river_model}: 0.0000 (ERROR)")
                        else:
                            ALL_RESULTS.append({
                                'chunk_size': chunk_size_name,
                                'batch': batch_name,
                                'dataset': dataset_name,
                                'model': river_model,
                                'gmean': 0.0,
                                'status': 'NO_DATA',
                                'penalty': is_penalty,
                            })
                            print(f"  {river_model}: 0.0000 (NO_DATA)")

            # ================================================================
            # 6. ACDWM
            # ================================================================
            if 'ACDWM' in MODELS_TO_RUN:
                if is_penalty or already_executed:
                    cached = load_existing_model_results(chunk_size_name, batch_name, dataset_name, 'ACDWM')
                    if cached:
                        ALL_RESULTS.append({
                            'chunk_size': chunk_size_name,
                            'batch': batch_name,
                            'dataset': dataset_name,
                            'model': 'ACDWM',
                            'gmean': cached['gmean'],
                            'status': 'CACHED',
                            'penalty': is_penalty,
                        })
                        print(f"  ACDWM: {cached['gmean']:.4f} (cached)")
                    elif chunks_available and not is_penalty:
                        try:
                            success, results = run_acdwm(X_chunks, y_chunks, timeout=MODEL_TIMEOUT['ACDWM'])
                            gmean = results.get('gmean', 0.0)
                            ALL_RESULTS.append({
                                'chunk_size': chunk_size_name,
                                'batch': batch_name,
                                'dataset': dataset_name,
                                'model': 'ACDWM',
                                'gmean': gmean,
                                'status': 'OK' if success else 'FAILED',
                                'penalty': is_penalty,
                            })
                            print(f"  ACDWM: {gmean:.4f} ({'OK' if success else 'FAILED'})")
                        except Exception as e:
                            ALL_RESULTS.append({
                                'chunk_size': chunk_size_name,
                                'batch': batch_name,
                                'dataset': dataset_name,
                                'model': 'ACDWM',
                                'gmean': 0.0,
                                'status': 'ERROR',
                                'penalty': is_penalty,
                            })
                            print(f"  ACDWM: 0.0000 (ERROR)")
                    else:
                        ALL_RESULTS.append({
                            'chunk_size': chunk_size_name,
                            'batch': batch_name,
                            'dataset': dataset_name,
                            'model': 'ACDWM',
                            'gmean': 0.0,
                            'status': 'NO_DATA',
                            'penalty': is_penalty,
                        })
                        print(f"  ACDWM: 0.0000 (NO_DATA)")
                else:
                    if chunks_available:
                        try:
                            success, results = run_acdwm(X_chunks, y_chunks, timeout=MODEL_TIMEOUT['ACDWM'])
                            gmean = results.get('gmean', 0.0)
                            ALL_RESULTS.append({
                                'chunk_size': chunk_size_name,
                                'batch': batch_name,
                                'dataset': dataset_name,
                                'model': 'ACDWM',
                                'gmean': gmean,
                                'status': 'OK' if success else 'FAILED',
                                'penalty': is_penalty,
                            })
                            print(f"  ACDWM: {gmean:.4f} ({'OK' if success else 'FAILED'})")
                        except Exception as e:
                            ALL_RESULTS.append({
                                'chunk_size': chunk_size_name,
                                'batch': batch_name,
                                'dataset': dataset_name,
                                'model': 'ACDWM',
                                'gmean': 0.0,
                                'status': 'ERROR',
                                'penalty': is_penalty,
                            })
                            print(f"  ACDWM: 0.0000 (ERROR)")
                    else:
                        ALL_RESULTS.append({
                            'chunk_size': chunk_size_name,
                            'batch': batch_name,
                            'dataset': dataset_name,
                            'model': 'ACDWM',
                            'gmean': 0.0,
                            'status': 'NO_DATA',
                            'penalty': is_penalty,
                        })
                        print(f"  ACDWM: 0.0000 (NO_DATA)")

            # ================================================================
            # 7. ERulesD2S (apenas cache)
            # ================================================================
            if 'ERulesD2S' in MODELS_TO_RUN:
                cached = load_existing_model_results(chunk_size_name, batch_name, dataset_name, 'ERulesD2S')
                if cached:
                    ALL_RESULTS.append({
                        'chunk_size': chunk_size_name,
                        'batch': batch_name,
                        'dataset': dataset_name,
                        'model': 'ERulesD2S',
                        'gmean': cached['gmean'],
                        'status': 'CACHED',
                        'penalty': is_penalty,
                    })
                    print(f"  ERulesD2S: {cached['gmean']:.4f} (cached)")
                else:
                    ALL_RESULTS.append({
                        'chunk_size': chunk_size_name,
                        'batch': batch_name,
                        'dataset': dataset_name,
                        'model': 'ERulesD2S',
                        'gmean': 0.0,
                        'status': 'NOT_RUN',
                        'penalty': is_penalty,
                    })
                    print(f"  ERulesD2S: 0.0000 (NOT_RUN)")

            # Marcar como executado (para configs sem penalty)
            if not is_penalty:
                EXECUTED_COMPARATIVE.add(comparative_key)

# =============================================================================
# RESUMO DA EXECUCAO
# =============================================================================
total_duration = time.time() - total_start

print("\n" + "="*80)
print("EXECUCAO CONCLUIDA!")
print("="*80)
print(f"Duracao total: {total_duration/60:.1f} minutos")
print(f"Datasets processados: {dataset_count}")
print(f"Total de resultados: {len(ALL_RESULTS)}")

# Estatisticas por modelo e penalty
print("\n" + "-"*50)
print("ESTATISTICAS POR MODELO:")
print("-"*50)

df_temp = pd.DataFrame(ALL_RESULTS)

for model in ['GBML', 'GBML_Penalty', 'CDCMS', 'ROSE_Original', 'ROSE_ChunkEval', 'HAT', 'ARF', 'SRP', 'ACDWM', 'ERulesD2S']:
    if model in df_temp['model'].values:
        model_df = df_temp[df_temp['model'] == model]
        ok_df = model_df[model_df['status'].isin(['OK', 'CACHED'])]
        avg_gmean = ok_df['gmean'].mean() if len(ok_df) > 0 else 0
        print(f"  {model:15s}: {len(ok_df):3d} OK | Media G-Mean: {avg_gmean:.4f}")

print("="*80)
