# =============================================================================
# CDCMS - Exclusao de Datasets Multiclasse
# =============================================================================
# O CDCMS.CIL foi projetado APENAS para classificacao BINARIA.
#
# Referencia: Paper "The Value of Diversity for Dealing with Concept Drift
#             in Class-Imbalanced Data Streams" (Chiu & Minku, IEEE DSAA 2025)
#
# Citacao do paper (Secao V.A):
#   "Covtype and INSECTS were originally multi-class problems. They have been
#    adapted into several versions of binary classification problems for this
#    study as shown in the supplementary material."
#
# Data: 2026-01-26
# =============================================================================

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

print("="*70)
print("CDCMS - GESTAO DE DATASETS MULTICLASSE")
print("="*70)

# =============================================================================
# 1. Lista de datasets multiclasse (detectados automaticamente)
# =============================================================================

MULTICLASS_DATASETS = {
    # LED - 10 classes (digitos 0-9)
    'LED_Abrupt_Simple': {'num_classes': 10, 'reason': 'LED generator - 10 digit classes'},
    'LED_Gradual_Simple': {'num_classes': 10, 'reason': 'LED generator - 10 digit classes'},
    'LED_Stationary': {'num_classes': 10, 'reason': 'LED generator - 10 digit classes'},

    # WAVEFORM - 3 classes
    'WAVEFORM_Abrupt_Simple': {'num_classes': 3, 'reason': 'WAVEFORM generator - 3 wave types'},
    'WAVEFORM_Gradual_Simple': {'num_classes': 3, 'reason': 'WAVEFORM generator - 3 wave types'},
    'WAVEFORM_Stationary': {'num_classes': 3, 'reason': 'WAVEFORM generator - 3 wave types'},

    # CovType - 7 classes (forest cover types)
    'CovType': {'num_classes': 7, 'reason': 'Forest cover type - 7 vegetation classes'},

    # Shuttle - 7 classes
    'Shuttle': {'num_classes': 7, 'reason': 'Shuttle dataset - 7 radiator classes'},

    # RBF_Stationary - 4 classes
    'RBF_Stationary': {'num_classes': 4, 'reason': 'RBF with 4 centroids/classes'},
}

# =============================================================================
# 2. Funcao para detectar se dataset e multiclasse
# =============================================================================

def is_multiclass_dataset(dataset_name: str) -> bool:
    """Verifica se o dataset e multiclasse (nao suportado pelo CDCMS)."""
    return dataset_name in MULTICLASS_DATASETS


def get_multiclass_info(dataset_name: str) -> dict:
    """Retorna informacoes sobre dataset multiclasse."""
    if dataset_name in MULTICLASS_DATASETS:
        return MULTICLASS_DATASETS[dataset_name]
    return None


def detect_num_classes(dataset_path: Path) -> int:
    """Detecta numero de classes a partir dos arquivos CSV."""
    chunks = list(dataset_path.glob("chunk_*.csv"))
    if not chunks:
        return -1

    df = pd.read_csv(chunks[0])
    class_col = df.columns[-1]
    return df[class_col].nunique()


# =============================================================================
# 3. Funcao para gerar resultado "N/A" para multiclasse
# =============================================================================

def generate_na_result(dataset_name: str, chunk_size_name: str, batch_name: str) -> dict:
    """
    Gera resultado marcado como N/A para datasets multiclasse.
    """
    info = MULTICLASS_DATASETS.get(dataset_name, {})

    return {
        'dataset': dataset_name,
        'chunk_size_name': chunk_size_name,
        'batch': batch_name,
        'classifier': 'CDCMS_CIL_GMean',
        'status': 'NOT_APPLICABLE',
        'reason': 'Multiclass dataset - CDCMS.CIL supports only binary classification',
        'num_classes': info.get('num_classes', 'Unknown'),
        'detail': info.get('reason', 'Multiclass dataset'),
        # Metricas como None/NaN
        'avg_prequential_gmean': None,
        'avg_prequential_f1': None,
        'avg_holdout_gmean': None,
        'avg_holdout_f1': None,
        'execution_time_seconds': 0,
        'note': 'CDCMS.CIL designed for binary classification only (Chiu & Minku, 2025)'
    }


# =============================================================================
# 4. Gerar relatorio de exclusao
# =============================================================================

def generate_exclusion_report(output_dir: Path) -> pd.DataFrame:
    """
    Gera relatorio CSV com todos os datasets excluidos.
    """
    records = []

    for dataset_name, info in MULTICLASS_DATASETS.items():
        records.append({
            'dataset': dataset_name,
            'num_classes': info['num_classes'],
            'reason': info['reason'],
            'cdcms_applicable': False,
            'egis_applicable': True,  # EGIS suporta multiclasse
            'note': 'Excluded from CDCMS comparison'
        })

    df = pd.DataFrame(records)

    # Salvar CSV
    output_file = output_dir / 'cdcms_excluded_datasets.csv'
    df.to_csv(output_file, index=False)

    print(f"\n[OK] Relatorio de exclusao salvo: {output_file}")

    return df


# =============================================================================
# 5. Atualizar funcao run_cdcms_on_dataset para verificar multiclasse
# =============================================================================

def should_skip_dataset(dataset_name: str, verbose: bool = True) -> bool:
    """
    Verifica se dataset deve ser pulado (multiclasse).
    Retorna True se deve pular.
    """
    if is_multiclass_dataset(dataset_name):
        if verbose:
            info = MULTICLASS_DATASETS[dataset_name]
            print(f"  [SKIP] {dataset_name} - Multiclasse ({info['num_classes']} classes)")
            print(f"         CDCMS.CIL suporta apenas classificacao binaria")
        return True
    return False


# =============================================================================
# 6. Exibir resumo
# =============================================================================

print("\nDatasets MULTICLASSE (excluidos do CDCMS):")
print("-" * 50)

for dataset, info in sorted(MULTICLASS_DATASETS.items()):
    print(f"  {dataset:30s} | {info['num_classes']:2d} classes | {info['reason']}")

print("-" * 50)
print(f"Total: {len(MULTICLASS_DATASETS)} datasets excluidos")

print("\n" + "="*70)
print("IMPORTANTE: Estes datasets serao marcados como 'N/A' nos resultados")
print("            do CDCMS, pois o algoritmo foi projetado apenas para")
print("            classificacao binaria (2 classes).")
print("="*70)

print("\nFuncoes disponiveis:")
print("  - is_multiclass_dataset(dataset_name)")
print("  - should_skip_dataset(dataset_name)")
print("  - generate_na_result(dataset_name, chunk_size, batch)")
print("  - generate_exclusion_report(output_dir)")
