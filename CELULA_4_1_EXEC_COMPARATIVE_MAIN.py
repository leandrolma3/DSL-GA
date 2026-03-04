# =============================================================================
# CELULA 4.1: Executar TODOS os Modelos em TODOS os Experimentos (UNIFIED)
# =============================================================================
# SUBSTITUA a CELULA 4.1 original por este codigo
# Suporta experimentos unified (chunk_500, chunk_1000 com/sem penalty)
# Carrega EGIS e CDCMS de cache, executa demais modelos
# =============================================================================

ALL_RESULTS = []

print("=" * 80)
print("INICIO DA EXECUCAO - TODOS OS MODELOS EM TODOS OS EXPERIMENTOS")
print("=" * 80)
print(f"Inicio: {datetime.now()}")
print(f"Experimentos: {EXPERIMENTS_TO_RUN}")
print(f"Modelos a executar: {MODELS_TO_RUN}")
print(f"Usar cache: {USE_CACHE}")
print("=" * 80)

total_start = time.time()

for exp_name in EXPERIMENTS_TO_RUN:
    config = EXPERIMENT_CONFIGS[exp_name]
    chunk_size = config['chunk_size']
    data_source = config.get('data_source', 'legacy')
    is_penalty = config.get('penalty_weight', 0.0) > 0
    reuse_from = config.get('reuse_comparative_from', None)

    print(f"\n{'#' * 80}")
    print(f"# EXPERIMENTO: {exp_name}")
    print(f"# chunk_size: {chunk_size}")
    print(f"# penalty_weight: {config.get('penalty_weight', 0.0)}")
    print(f"# data_source: {data_source}")
    if reuse_from:
        print(f"# Reusando comparativos de: {reuse_from}")
    print(f"# {config['description']}")
    print(f"{'#' * 80}")

    # =========================================================================
    # DETECTAR DATASETS DISPONIVEIS
    # =========================================================================
    if data_source == 'unified':
        datasets_info = detect_available_datasets_unified(exp_name, config)
        # Converter para formato padrao
        available = [(info['batch'], info['dataset'], info['data_path']) for info in datasets_info]
    else:
        available = detect_available_datasets(exp_name, config)

    print(f"\nDatasets disponiveis: {len(available)}")

    # =========================================================================
    # PROCESSAR CADA DATASET
    # =========================================================================
    for idx, item in enumerate(available):
        if data_source == 'unified':
            batch_name, dataset_name, data_path = item
            dataset_dir = get_dataset_output_dir(config, batch_name, dataset_name)
        else:
            batch_name, dataset_name, dataset_dir = item
            data_path = dataset_dir

        print(f"\n[{idx+1}/{len(available)}] {batch_name}/{dataset_name}")

        # =====================================================================
        # CARREGAR CHUNKS DE DADOS
        # =====================================================================
        if data_source == 'unified':
            data_dir = config.get('data_dir', 'chunk_500')
            X_chunks, y_chunks = load_chunks_from_unified(data_dir, dataset_name)
        else:
            X_chunks, y_chunks = load_chunks_from_gbml(dataset_dir)

        if X_chunks is None:
            print(f"  AVISO: Chunks nao encontrados")
            continue

        n_classes = len(np.unique(np.concatenate(y_chunks)))
        is_multiclass = n_classes > 2
        print(f"  Chunks: {len(X_chunks)} | Samples/chunk: ~{len(X_chunks[0])} | Classes: {n_classes}")

        # =====================================================================
        # CARREGAR EGIS (de cache)
        # =====================================================================
        if data_source == 'unified':
            results_dir = config.get('results_dir', 'chunk_500')
            egis_model_name = config.get('egis_model_name', 'EGIS')

            egis_results = load_egis_results(results_dir, batch_name, dataset_name, egis_model_name)

            if egis_results:
                ALL_RESULTS.append({
                    'experiment': exp_name,
                    'batch': batch_name,
                    'dataset': dataset_name,
                    'model': egis_model_name,
                    'gmean': egis_results['gmean'],
                    'status': 'CACHED',
                    'chunk_size': chunk_size,
                    'penalty': is_penalty
                })
                print(f"  {egis_model_name}: {egis_results['gmean']:.4f} (cached)")
            else:
                ALL_RESULTS.append({
                    'experiment': exp_name,
                    'batch': batch_name,
                    'dataset': dataset_name,
                    'model': egis_model_name,
                    'gmean': 0.0,
                    'status': 'NOT_FOUND',
                    'chunk_size': chunk_size,
                    'penalty': is_penalty
                })
                print(f"  {egis_model_name}: NOT_FOUND")

        # =====================================================================
        # CARREGAR CDCMS (de cache)
        # =====================================================================
        if data_source == 'unified':
            if is_multiclass:
                # CDCMS nao suporta multiclasse
                ALL_RESULTS.append({
                    'experiment': exp_name,
                    'batch': batch_name,
                    'dataset': dataset_name,
                    'model': 'CDCMS',
                    'gmean': 0.0,
                    'status': 'N/A (multiclass)',
                    'chunk_size': chunk_size,
                    'penalty': is_penalty
                })
                print(f"  CDCMS: N/A (multiclass)")
            else:
                cdcms_results = load_cdcms_results(results_dir, batch_name, dataset_name)

                if cdcms_results:
                    ALL_RESULTS.append({
                        'experiment': exp_name,
                        'batch': batch_name,
                        'dataset': dataset_name,
                        'model': 'CDCMS',
                        'gmean': cdcms_results['gmean'],
                        'status': 'CACHED',
                        'chunk_size': chunk_size,
                        'penalty': is_penalty
                    })
                    print(f"  CDCMS: {cdcms_results['gmean']:.4f} (cached)")
                else:
                    ALL_RESULTS.append({
                        'experiment': exp_name,
                        'batch': batch_name,
                        'dataset': dataset_name,
                        'model': 'CDCMS',
                        'gmean': 0.0,
                        'status': 'NOT_FOUND',
                        'chunk_size': chunk_size,
                        'penalty': is_penalty
                    })
                    print(f"  CDCMS: NOT_FOUND")

        # =====================================================================
        # EXECUTAR MODELOS COMPARATIVOS
        # =====================================================================
        for model_name in MODELS_TO_RUN:

            # -----------------------------------------------------------------
            # SE PENALTY: Reusar resultados de modelos comparativos
            # -----------------------------------------------------------------
            if reuse_from and model_name not in ['EGIS', 'EGIS_Penalty', 'CDCMS']:
                # Buscar resultado da versao sem penalty
                base_config = EXPERIMENT_CONFIGS.get(reuse_from, config)
                base_results_dir = base_config.get('results_dir', results_dir.replace('_penalty', ''))

                cached = load_existing_comparative_results(
                    base_results_dir, batch_name, dataset_name, model_name
                )

                if cached:
                    ALL_RESULTS.append({
                        'experiment': exp_name,
                        'batch': batch_name,
                        'dataset': dataset_name,
                        'model': model_name,
                        'gmean': cached['gmean'],
                        'status': 'REUSED',
                        'chunk_size': chunk_size,
                        'penalty': is_penalty
                    })
                    print(f"  {model_name}: {cached['gmean']:.4f} (reused from {reuse_from})")
                    continue

            # -----------------------------------------------------------------
            # VERIFICAR CACHE
            # -----------------------------------------------------------------
            if USE_CACHE and model_name not in FORCE_RERUN:
                if data_source == 'unified':
                    cached = load_existing_comparative_results(
                        config.get('results_dir', 'chunk_500').replace('_penalty', ''),
                        batch_name, dataset_name, model_name
                    )
                else:
                    cached = load_existing_model_results(dataset_dir, model_name)

                if cached:
                    ALL_RESULTS.append({
                        'experiment': exp_name,
                        'batch': batch_name,
                        'dataset': dataset_name,
                        'model': model_name,
                        'gmean': cached['gmean'],
                        'status': 'CACHED',
                        'chunk_size': chunk_size,
                        'penalty': is_penalty
                    })
                    print(f"  {model_name}: {cached['gmean']:.4f} (cached)")
                    continue

            # -----------------------------------------------------------------
            # VERIFICAR SE MODELO E APLICAVEL
            # -----------------------------------------------------------------
            if model_name == 'ACDWM' and is_multiclass:
                ALL_RESULTS.append({
                    'experiment': exp_name,
                    'batch': batch_name,
                    'dataset': dataset_name,
                    'model': model_name,
                    'gmean': 0.0,
                    'status': 'N/A (multiclass)',
                    'chunk_size': chunk_size,
                    'penalty': is_penalty
                })
                print(f"  {model_name}: N/A (multiclass)")
                continue

            # -----------------------------------------------------------------
            # EXECUTAR MODELO
            # -----------------------------------------------------------------
            try:
                # Determinar diretorio de saida
                output_dir = get_dataset_output_dir(config, batch_name, dataset_name)
                run_dir = output_dir / "run_1"
                run_dir.mkdir(parents=True, exist_ok=True)

                if model_name == 'ROSE_Original':
                    X_all = np.vstack(X_chunks)
                    y_all = np.concatenate(y_chunks)
                    arff_dir = run_dir / "rose_arff"
                    arff_file = arff_dir / f"{dataset_name}.arff"
                    create_arff_file(X_all, y_all, arff_file, relation_name=dataset_name)

                    rose_output = run_dir / "rose_original_output"
                    success, results = run_rose_original(
                        arff_file, rose_output, n_classes=n_classes,
                        timeout=MODEL_TIMEOUT.get('ROSE_Original', 600)
                    )
                    gmean = results.get('gmean', 0.0) if success else 0.0
                    status = 'OK' if success else 'FAILED'

                    # Copiar resultado para run_1/
                    if success:
                        import shutil
                        src = rose_output / "rose_original_results.csv"
                        dst = run_dir / "rose_original_results.csv"
                        if src.exists():
                            shutil.copy(src, dst)

                elif model_name == 'ROSE_ChunkEval':
                    X_all = np.vstack(X_chunks)
                    y_all = np.concatenate(y_chunks)
                    arff_dir = run_dir / "rose_arff"
                    arff_file = arff_dir / f"{dataset_name}.arff"
                    if not arff_file.exists():
                        create_arff_file(X_all, y_all, arff_file, relation_name=dataset_name)

                    rose_output = run_dir / "rose_chunk_eval_output"
                    success, results = run_rose_chunk_eval(
                        arff_file, rose_output, n_classes=n_classes,
                        chunk_size=chunk_size,
                        timeout=MODEL_TIMEOUT.get('ROSE_ChunkEval', 600)
                    )
                    gmean = results.get('gmean', 0.0) if success else 0.0
                    status = 'OK' if success else 'FAILED'

                    # Copiar resultado para run_1/
                    if success:
                        import shutil
                        src = rose_output / "rose_chunk_eval_results.csv"
                        dst = run_dir / "rose_chunk_eval_results.csv"
                        if src.exists():
                            shutil.copy(src, dst)

                elif model_name in ['HAT', 'ARF', 'SRP']:
                    results = run_river_model(
                        model_name, X_chunks, y_chunks,
                        timeout=MODEL_TIMEOUT.get(model_name, 300)
                    )
                    gmean = results.get('gmean', 0.0)
                    status = 'OK' if 'error' not in results else 'FAILED'

                    # Salvar resultados por chunk
                    if 'chunk_results' in results:
                        save_model_results(output_dir, model_name, results)

                elif model_name == 'ACDWM':
                    results = run_acdwm(
                        X_chunks, y_chunks,
                        acdwm_path=ACDWM_DIR,
                        timeout=MODEL_TIMEOUT.get('ACDWM', 600)
                    )
                    gmean = results.get('gmean', 0.0)
                    status = 'OK' if 'error' not in results else 'FAILED'

                    if 'chunk_results' in results:
                        save_model_results(output_dir, model_name, results)

                elif model_name == 'ERulesD2S':
                    # ERulesD2S - tenta cache primeiro, depois executa se habilitado
                    if ERULESD2S_ENABLED:
                        results = run_erulesd2s(
                            X_chunks, y_chunks, output_dir, dataset_name,
                            chunk_size=chunk_size,
                            timeout=MODEL_TIMEOUT.get('ERulesD2S', 1800)
                        )
                        gmean = results.get('gmean', 0.0)
                        status = 'OK' if 'error' not in results else f"FAILED: {results.get('error', '')[:20]}"
                    else:
                        gmean = 0.0
                        status = 'SKIPPED'
                else:
                    gmean = 0.0
                    status = 'UNKNOWN_MODEL'

                ALL_RESULTS.append({
                    'experiment': exp_name,
                    'batch': batch_name,
                    'dataset': dataset_name,
                    'model': model_name,
                    'gmean': gmean,
                    'status': status,
                    'chunk_size': chunk_size,
                    'penalty': is_penalty
                })
                print(f"  {model_name}: {gmean:.4f} ({status})")

            except Exception as e:
                ALL_RESULTS.append({
                    'experiment': exp_name,
                    'batch': batch_name,
                    'dataset': dataset_name,
                    'model': model_name,
                    'gmean': 0.0,
                    'status': f'ERROR: {str(e)[:30]}',
                    'chunk_size': chunk_size,
                    'penalty': is_penalty
                })
                print(f"  {model_name}: 0.0000 (ERROR: {str(e)[:30]})")

total_time = time.time() - total_start
print(f"\n{'=' * 80}")
print(f"EXECUCAO COMPLETA!")
print(f"Tempo total: {total_time/60:.1f} minutos")
print(f"Total de resultados: {len(ALL_RESULTS)}")
print(f"{'=' * 80}")

# =============================================================================
# RESUMO POR EXPERIMENTO
# =============================================================================
print("\nRESUMO POR EXPERIMENTO:")
print("-" * 60)

df_temp = pd.DataFrame(ALL_RESULTS)
for exp_name in df_temp['experiment'].unique():
    exp_df = df_temp[df_temp['experiment'] == exp_name]
    ok_count = len(exp_df[exp_df['status'].isin(['OK', 'CACHED', 'REUSED'])])
    na_count = len(exp_df[exp_df['status'].str.contains('N/A')])
    fail_count = len(exp_df) - ok_count - na_count

    print(f"\n{exp_name}:")
    print(f"  Total: {len(exp_df)} | OK/Cached: {ok_count} | N/A: {na_count} | Failed: {fail_count}")

    # Media por modelo
    for model in exp_df['model'].unique():
        model_df = exp_df[exp_df['model'] == model]
        ok_df = model_df[model_df['status'].isin(['OK', 'CACHED', 'REUSED'])]
        if len(ok_df) > 0:
            print(f"    {model:15s}: {ok_df['gmean'].mean():.4f} (n={len(ok_df)})")
