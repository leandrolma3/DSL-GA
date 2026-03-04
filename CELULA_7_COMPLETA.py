# =============================================================================
# PARTE 7 COMPLETA: Testar CDCMS com Datasets do unified_chunks
# =============================================================================
# Este arquivo substitui toda a PARTE 7 do Setup_CDCMS_CIL.ipynb
# Cole o conteudo de cada secao em celulas separadas no Colab
# =============================================================================

# =============================================================================
# CELULA 7.1: Setup do Google Drive e Caminhos
# =============================================================================

from pathlib import Path
import json

print("="*70)
print("SETUP DO GOOGLE DRIVE")
print("="*70)

# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Caminho base no Drive
DRIVE_BASE = Path('/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid')

# Verificar se existe
if DRIVE_BASE.exists():
    print(f"[OK] Caminho base: {DRIVE_BASE}")
else:
    print(f"[ERRO] Caminho nao encontrado!")
    # Listar o que existe em Othercomputers
    other = Path('/content/drive/Othercomputers')
    if other.exists():
        print("\nConteudo de /content/drive/Othercomputers:")
        for d in other.iterdir():
            print(f"  - {d.name}")

# Diretorios
UNIFIED_CHUNKS_DIR = DRIVE_BASE / 'unified_chunks'
DEFAULT_CHUNK_SIZE = 'chunk_2000'  # 6 chunks - mais rapido
CHUNKS_DIR = UNIFIED_CHUNKS_DIR / DEFAULT_CHUNK_SIZE

# Verificar dados
if CHUNKS_DIR.exists():
    AVAILABLE_DATASETS = sorted([d.name for d in CHUNKS_DIR.iterdir()
                                  if d.is_dir() and not d.name.startswith('.')
                                  and not d.name.endswith('_backup')])
    print(f"[OK] {len(AVAILABLE_DATASETS)} datasets disponiveis")
else:
    AVAILABLE_DATASETS = []
    print("[ERRO] Diretorio de chunks nao encontrado")

# Carregar metadata
metadata_file = UNIFIED_CHUNKS_DIR / 'metadata.json'
if metadata_file.exists():
    with open(metadata_file) as f:
        METADATA = json.load(f)
    print(f"[OK] Metadata: {len(METADATA['datasets'])} datasets")

print("\n[OK] Variaveis: DRIVE_BASE, CHUNKS_DIR, AVAILABLE_DATASETS")


# =============================================================================
# CELULA 7.2: Funcao para Converter CSV para ARFF e Executar CDCMS
# =============================================================================

import subprocess
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

def run_cdcms_on_dataset(dataset_name, chunks_dir=None, chunk_size='chunk_2000',
                          classifier="CDCMS_CIL_GMean", timeout=600):
    """
    Carrega dataset do Drive, converte para ARFF e executa CDCMS.
    Nao copia arquivos - le diretamente do Drive.
    """
    if chunks_dir is None:
        chunks_dir = UNIFIED_CHUNKS_DIR

    dataset_path = Path(chunks_dir) / chunk_size / dataset_name

    if not dataset_path.exists():
        print(f"[ERRO] Dataset nao encontrado: {dataset_path}")
        return None

    # Listar chunks
    chunks = sorted(dataset_path.glob("chunk_*.csv"),
                    key=lambda x: int(x.stem.split('_')[1]))

    print(f"\nDataset: {dataset_name}")
    print(f"  Chunks: {len(chunks)}")

    # Carregar e concatenar (le do Drive, nao copia)
    all_data = []
    for chunk_file in chunks:
        df = pd.read_csv(chunk_file)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"  Instancias: {len(combined_df)}")
    print(f"  Features: {len(combined_df.columns)-1}")

    # Detectar classes
    class_col = combined_df.columns[-1]
    unique_classes = sorted(combined_df[class_col].unique())
    print(f"  Classes: {unique_classes}")

    # Criar ARFF temporario no Colab (nao no Drive)
    arff_file = TEST_DIR / f"{dataset_name}.arff"

    with open(arff_file, 'w') as f:
        f.write(f"@relation {dataset_name}\n\n")

        for col in combined_df.columns[:-1]:
            if combined_df[col].dtype in ['int64', 'float64']:
                f.write(f"@attribute {col} numeric\n")
            else:
                unique_vals = sorted(combined_df[col].unique())
                vals_str = ",".join(str(v) for v in unique_vals)
                f.write(f"@attribute {col} {{{vals_str}}}\n")

        class_str = ",".join(str(int(c)) for c in unique_classes)
        f.write(f"@attribute class {{{class_str}}}\n\n")

        f.write("@data\n")
        for _, row in combined_df.iterrows():
            values = ",".join(str(v) for v in row)
            f.write(f"{values}\n")

    # Executar CDCMS
    output_file = RESULTS_DIR / f"{dataset_name}_{classifier}.csv"

    # Classpath simples (CDCMS_JAR + MOA_DEPS_JAR)
    classpath = f"{CDCMS_JAR}:{MOA_DEPS_JAR}"

    cmd = [
        "java", "-Xmx4g",
        "-cp", f"{classpath}:{TEST_DIR}",
        "CDCMSEvaluator",
        str(arff_file),
        str(output_file),
        classifier
    ]

    print(f"  Executando {classifier}...")
    start = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        duration = time.time() - start

        if output_file.exists() and output_file.stat().st_size > 0:
            result_df = pd.read_csv(output_file)
            final_accuracy = result_df['accuracy'].iloc[-1]

            print(f"  [OK] Accuracy: {final_accuracy:.4f} | Tempo: {duration:.1f}s")

            return {
                'dataset': dataset_name,
                'classifier': classifier,
                'chunk_size': chunk_size,
                'instances': len(combined_df),
                'features': len(combined_df.columns)-1,
                'classes': len(unique_classes),
                'accuracy': final_accuracy,
                'time_seconds': duration
            }
        else:
            print(f"  [ERRO] Execucao falhou")
            if result.stderr:
                errors = [l for l in result.stderr.split('\n')
                          if 'WARNING' not in l and l.strip()][:3]
                for e in errors:
                    print(f"    {e[:80]}")
            return None

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Excedeu {timeout}s")
        return None
    except Exception as e:
        print(f"  [ERRO] {e}")
        return None

print("[OK] Funcao run_cdcms_on_dataset carregada")


# =============================================================================
# CELULA 7.3: Executar em Datasets de Teste
# =============================================================================

print("="*70)
print("EXECUTAR CDCMS EM DATASETS DE TESTE")
print("="*70)

# Datasets para teste inicial (representativos de cada tipo)
TEST_DATASETS = [
    'SINE_Abrupt_Simple',         # 2 features, binario, drift abrupto
    'SEA_Gradual_Simple_Slow',    # 3 features, binario, drift gradual
    'HYPERPLANE_Abrupt_Simple',   # 10 features, binario, drift abrupto
    'AGRAWAL_Abrupt_Chain_Long',  # 9 features, binario, drift em cadeia
    'RBF_Abrupt_Severe',          # 10 features, binario, drift severo
]

results = []

for dataset in TEST_DATASETS:
    if dataset in AVAILABLE_DATASETS:
        result = run_cdcms_on_dataset(dataset, UNIFIED_CHUNKS_DIR)
        if result:
            results.append(result)
    else:
        print(f"\n[SKIP] {dataset} nao encontrado")

# Resumo
if results:
    print("\n" + "="*70)
    print("RESUMO DOS RESULTADOS")
    print("="*70)

    results_df = pd.DataFrame(results)
    print(results_df[['dataset', 'instances', 'features', 'accuracy', 'time_seconds']].to_string(index=False))

    # Salvar
    summary_file = RESULTS_DIR / 'cdcms_test_summary.csv'
    results_df.to_csv(summary_file, index=False)
    print(f"\nSalvo em: {summary_file}")


# =============================================================================
# CELULA 7.4: Executar em TODOS os Datasets
# =============================================================================

print("="*70)
print("EXECUTAR CDCMS EM TODOS OS DATASETS")
print("="*70)

# Excluir datasets problematicos
EXCLUDE_PATTERNS = ['_backup', 'IntelLabSensors']  # IntelLab tem NaN

datasets_to_run = [d for d in AVAILABLE_DATASETS
                   if not any(p in d for p in EXCLUDE_PATTERNS)]

print(f"Datasets a executar: {len(datasets_to_run)}")
print("(Isso pode demorar alguns minutos)")

all_results = []

for i, dataset in enumerate(datasets_to_run, 1):
    print(f"\n[{i}/{len(datasets_to_run)}]", end="")
    result = run_cdcms_on_dataset(dataset, UNIFIED_CHUNKS_DIR)
    if result:
        all_results.append(result)

# Resumo final
if all_results:
    print("\n" + "="*70)
    print("RESUMO FINAL - TODOS OS DATASETS")
    print("="*70)

    final_df = pd.DataFrame(all_results)

    # Estatisticas
    print(f"\nDatasets executados: {len(final_df)}")
    print(f"Accuracy media: {final_df['accuracy'].mean():.4f}")
    print(f"Accuracy mediana: {final_df['accuracy'].median():.4f}")
    print(f"Tempo total: {final_df['time_seconds'].sum():.1f}s")

    # Top 5 melhores
    print("\nTop 5 melhores:")
    top5 = final_df.nlargest(5, 'accuracy')[['dataset', 'accuracy']]
    print(top5.to_string(index=False))

    # Top 5 piores
    print("\nTop 5 piores:")
    bottom5 = final_df.nsmallest(5, 'accuracy')[['dataset', 'accuracy']]
    print(bottom5.to_string(index=False))

    # Salvar completo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = RESULTS_DIR / f'cdcms_all_results_{timestamp}.csv'
    final_df.to_csv(final_file, index=False)
    print(f"\nResultados salvos em: {final_file}")


# =============================================================================
# CELULA 7.5: Comparar por Tipo de Drift
# =============================================================================

if all_results:
    print("\n" + "="*70)
    print("ANALISE POR TIPO DE DRIFT")
    print("="*70)

    final_df = pd.DataFrame(all_results)

    # Extrair tipo de drift do nome
    def get_drift_type(name):
        if 'Stationary' in name:
            return 'Stationary'
        elif 'Abrupt' in name:
            return 'Abrupt'
        elif 'Gradual' in name:
            return 'Gradual'
        else:
            return 'Other'

    final_df['drift_type'] = final_df['dataset'].apply(get_drift_type)

    # Agrupar por tipo
    drift_summary = final_df.groupby('drift_type').agg({
        'accuracy': ['mean', 'std', 'count'],
        'time_seconds': 'mean'
    }).round(4)

    print("\nAccuracy por tipo de drift:")
    print(drift_summary.to_string())
