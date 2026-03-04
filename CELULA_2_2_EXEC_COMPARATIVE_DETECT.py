# =============================================================================
# CELULA 2.2: Auto-detectar Datasets Disponiveis (UNIFIED + LEGACY)
# =============================================================================
# SUBSTITUA a CELULA 2.2 original por este codigo
# Suporta deteccao de datasets em unified_chunks e diretorios legados
# =============================================================================

from pathlib import Path
from collections import defaultdict

def detect_available_datasets(exp_name, config):
    """
    Detecta quais datasets estao disponiveis para um experimento.

    Suporta dois modos:
    1. UNIFIED: Verifica unified_chunks + experiments_unified
    2. LEGACY: Verifica estrutura antiga (run_1/chunk_data)

    Returns:
        Lista de (batch_name, dataset_name, dataset_dir) para datasets disponiveis
    """
    available = []
    data_source = config.get('data_source', 'legacy')

    for batch_name, batch_info in config['batches'].items():
        base_dir = Path(WORK_DIR) / batch_info['base_dir']

        if not base_dir.exists():
            continue

        for dataset_name in batch_info['datasets']:
            dataset_dir = base_dir / dataset_name

            if data_source == 'unified':
                # MODO UNIFIED: Verificar dados em unified_chunks
                data_dir = config.get('data_dir', 'chunk_500')
                data_path = UNIFIED_CHUNKS_DIR / data_dir / dataset_name

                # Verificar se tem chunks de dados
                has_data = data_path.exists() and any(data_path.glob('chunk_*.csv'))

                # Verificar se tem resultados EGIS
                results_dir = config.get('results_dir', data_dir)
                egis_metrics = EXPERIMENTS_UNIFIED_DIR / results_dir / batch_name / dataset_name / 'run_1' / 'chunk_metrics.json'
                has_egis = egis_metrics.exists()

                if has_data and has_egis:
                    available.append((batch_name, dataset_name, dataset_dir))
                elif has_data and not has_egis:
                    # Dataset tem dados mas nao tem EGIS ainda
                    # Pode adicionar com flag indicando que EGIS esta faltando
                    pass

            else:
                # MODO LEGACY: Verificar estrutura antiga
                run_dir = dataset_dir / 'run_1'
                chunk_data_dir = run_dir / 'chunk_data'

                if chunk_data_dir.exists() and any(chunk_data_dir.glob('chunk_*_test.csv')):
                    available.append((batch_name, dataset_name, dataset_dir))

    return available


def detect_available_datasets_unified(exp_name, config):
    """
    Versao especifica para experimentos unified.
    Retorna informacoes adicionais sobre status de EGIS e CDCMS.

    Returns:
        Lista de dicts com informacoes detalhadas
    """
    available = []
    data_dir = config.get('data_dir', 'chunk_500')
    results_dir = config.get('results_dir', data_dir)

    for batch_name, batch_info in config['batches'].items():
        for dataset_name in batch_info['datasets']:
            # Verificar dados
            data_path = UNIFIED_CHUNKS_DIR / data_dir / dataset_name
            has_data = data_path.exists() and any(data_path.glob('chunk_*.csv'))

            if not has_data:
                continue

            # Verificar EGIS
            egis_path = EXPERIMENTS_UNIFIED_DIR / results_dir / batch_name / dataset_name / 'run_1' / 'chunk_metrics.json'
            has_egis = egis_path.exists()

            # Verificar CDCMS (sempre na pasta sem penalty)
            base_results_dir = results_dir.replace('_penalty', '')
            cdcms_path = EXPERIMENTS_UNIFIED_DIR / base_results_dir / batch_name / dataset_name / 'cdcms_results' / 'chunk_metrics.json'
            has_cdcms = cdcms_path.exists()

            # Verificar se e multiclasse
            is_multiclass = dataset_name in MULTICLASS_DATASETS

            available.append({
                'batch': batch_name,
                'dataset': dataset_name,
                'data_path': data_path,
                'has_egis': has_egis,
                'has_cdcms': has_cdcms,
                'is_multiclass': is_multiclass,
                'base_dir': batch_info['base_dir']
            })

    return available


# =============================================================================
# MOSTRAR RESUMO DOS DATASETS DISPONIVEIS
# =============================================================================
print("=" * 80)
print("DATASETS DISPONIVEIS POR EXPERIMENTO")
print("=" * 80)

for exp_name in EXPERIMENTS_TO_RUN:
    config = EXPERIMENT_CONFIGS[exp_name]
    data_source = config.get('data_source', 'legacy')

    print(f"\n{'='*60}")
    print(f"{exp_name}")
    print(f"  Fonte: {data_source}")
    print(f"  chunk_size: {config['chunk_size']}")
    print(f"  penalty_weight: {config.get('penalty_weight', 0.0)}")
    print(f"  {config['description']}")
    print(f"{'='*60}")

    if data_source == 'unified':
        # Usar deteccao detalhada para unified
        datasets_info = detect_available_datasets_unified(exp_name, config)

        # Agrupar por batch
        by_batch = defaultdict(list)
        for info in datasets_info:
            by_batch[info['batch']].append(info)

        total = 0
        egis_ok = 0
        cdcms_ok = 0
        multiclass = 0

        for batch, items in sorted(by_batch.items()):
            print(f"\n  {batch}: {len(items)} datasets")

            for item in items:
                egis_str = "EGIS:OK" if item['has_egis'] else "EGIS:--"
                cdcms_str = "CDCMS:OK" if item['has_cdcms'] else ("CDCMS:N/A" if item['is_multiclass'] else "CDCMS:--")
                mc_str = " [MULTICLASS]" if item['is_multiclass'] else ""

                print(f"    {item['dataset']}: {egis_str}, {cdcms_str}{mc_str}")

                total += 1
                if item['has_egis']:
                    egis_ok += 1
                if item['has_cdcms']:
                    cdcms_ok += 1
                if item['is_multiclass']:
                    multiclass += 1

        print(f"\n  RESUMO: {total} datasets, EGIS={egis_ok}, CDCMS={cdcms_ok}, Multiclass={multiclass}")

    else:
        # Usar deteccao simples para legacy
        available = detect_available_datasets(exp_name, config)

        by_batch = defaultdict(list)
        for batch, dataset, _ in available:
            by_batch[batch].append(dataset)

        for batch, datasets in sorted(by_batch.items()):
            print(f"  {batch}: {len(datasets)} datasets")

        print(f"\n  TOTAL: {len(available)} datasets")

print("\n" + "=" * 80)
