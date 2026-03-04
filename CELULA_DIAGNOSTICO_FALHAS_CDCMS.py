# =============================================================================
# CELULA DIAGNOSTICO: Investigar falhas do CDCMS
# =============================================================================
# Executar para entender porque alguns datasets falharam
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess

print("="*70)
print("DIAGNOSTICO: DATASETS QUE FALHARAM NO CDCMS")
print("="*70)

# Lista de datasets que falharam
FAILED_DATASETS = [
    ('LED_Abrupt_Simple', 'chunk_500', 'batch_2'),
    ('LED_Gradual_Simple', 'chunk_500', 'batch_2'),
    ('WAVEFORM_Abrupt_Simple', 'chunk_500', 'batch_2'),
    ('WAVEFORM_Gradual_Simple', 'chunk_500', 'batch_2'),
    ('CovType', 'chunk_500', 'batch_3'),
    ('LED_Stationary', 'chunk_500', 'batch_3'),
    ('RBF_Stationary', 'chunk_500', 'batch_3'),
    ('Shuttle', 'chunk_500', 'batch_3'),
    ('WAVEFORM_Stationary', 'chunk_500', 'batch_3'),
]

# =============================================================================
# 1. Analisar cada dataset
# =============================================================================
print("\n" + "="*70)
print("1. ANALISE DOS DADOS")
print("="*70)

diagnostics = []

for dataset_name, chunk_size, batch in FAILED_DATASETS:
    data_path = UNIFIED_CHUNKS_DIR / chunk_size / dataset_name

    print(f"\n--- {dataset_name} ---")

    if not data_path.exists():
        print(f"  [ERRO] Path nao existe: {data_path}")
        continue

    # Carregar primeiro chunk para analise
    chunks = sorted(data_path.glob("chunk_*.csv"))
    if not chunks:
        print(f"  [ERRO] Nenhum chunk encontrado")
        continue

    # Carregar todos os chunks
    all_data = []
    for chunk_file in chunks:
        df = pd.read_csv(chunk_file)
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    # Analise
    class_col = combined.columns[-1]
    unique_classes = combined[class_col].unique()
    num_classes = len(unique_classes)

    print(f"  Instancias: {len(combined)}")
    print(f"  Features: {len(combined.columns)-1}")
    print(f"  Coluna classe: '{class_col}'")
    print(f"  Tipo classe: {combined[class_col].dtype}")
    print(f"  Num classes: {num_classes}")
    print(f"  Classes: {sorted(unique_classes)[:10]}{'...' if num_classes > 10 else ''}")

    # Verificar se tem NaN
    nan_count = combined.isna().sum().sum()
    if nan_count > 0:
        print(f"  [AVISO] NaN encontrados: {nan_count}")

    # Verificar valores da classe
    class_sample = combined[class_col].head(5).tolist()
    print(f"  Amostra classe: {class_sample}")

    diagnostics.append({
        'dataset': dataset_name,
        'batch': batch,
        'instances': len(combined),
        'features': len(combined.columns)-1,
        'class_col': class_col,
        'class_dtype': str(combined[class_col].dtype),
        'num_classes': num_classes,
        'classes': sorted(unique_classes),
        'has_nan': nan_count > 0
    })

# =============================================================================
# 2. Resumo dos padroes
# =============================================================================
print("\n" + "="*70)
print("2. PADRAO IDENTIFICADO")
print("="*70)

# Agrupar por numero de classes
multiclass = [d for d in diagnostics if d['num_classes'] > 2]
binary = [d for d in diagnostics if d['num_classes'] == 2]

print(f"\nDatasets MULTICLASSE (>2 classes): {len(multiclass)}")
for d in multiclass:
    print(f"  - {d['dataset']}: {d['num_classes']} classes")

print(f"\nDatasets BINARIOS (2 classes): {len(binary)}")
for d in binary:
    print(f"  - {d['dataset']}: classes {d['classes']}")

# =============================================================================
# 3. Teste detalhado com um dataset
# =============================================================================
print("\n" + "="*70)
print("3. TESTE DETALHADO COM LED_Abrupt_Simple")
print("="*70)

test_dataset = 'LED_Abrupt_Simple'
test_path = UNIFIED_CHUNKS_DIR / 'chunk_500' / test_dataset

if test_path.exists():
    # Carregar dados
    chunks = sorted(test_path.glob("chunk_*.csv"))
    all_data = []
    for chunk_file in chunks:
        df = pd.read_csv(chunk_file)
        all_data.append(df)
    combined = pd.concat(all_data, ignore_index=True)

    class_col = combined.columns[-1]
    print(f"Classes originais: {sorted(combined[class_col].unique())}")
    print(f"Tipo: {combined[class_col].dtype}")

    # Criar ARFF de teste
    test_arff = TEMP_ARFF_DIR / 'test_led_debug.arff'

    # Usar a funcao create_arff_from_dataframe
    success = create_arff_from_dataframe(combined, test_arff, test_dataset)

    if success:
        print(f"\n[OK] ARFF criado: {test_arff}")

        # Mostrar header do ARFF
        print("\nHeader do ARFF:")
        with open(test_arff) as f:
            for i, line in enumerate(f):
                if i < 15:
                    print(f"  {line.rstrip()}")
                elif '@data' in line.lower():
                    print(f"  {line.rstrip()}")
                    # Mostrar primeiras 3 linhas de dados
                    for j in range(3):
                        data_line = next(f, None)
                        if data_line:
                            print(f"  {data_line.rstrip()[:80]}...")
                    break

        # Tentar executar CDCMS
        print("\n--- Executando CDCMS ---")

        output_file = TEST_DIR / 'test_led_output.csv'
        if output_file.exists():
            output_file.unlink()

        classpath = f"{CDCMS_JAR}:{MOA_DEPS_JAR}"
        cmd = [
            "java", "-Xmx4g",
            "-cp", f"{classpath}:{TEST_DIR}",
            "CDCMSEvaluator",
            str(test_arff),
            str(output_file),
            "CDCMS_CIL_GMean"
        ]

        print(f"Comando: java CDCMSEvaluator {test_arff.name} ...")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        print(f"\nReturn code: {result.returncode}")

        if result.stdout:
            print(f"\nSTDOUT:")
            for line in result.stdout.strip().split('\n')[:20]:
                print(f"  {line}")

        if result.stderr:
            print(f"\nSTDERR (filtrado):")
            for line in result.stderr.strip().split('\n'):
                if 'Exception' in line or 'Error' in line or 'error' in line.lower():
                    print(f"  {line[:100]}")

        if output_file.exists():
            size = output_file.stat().st_size
            print(f"\nArquivo de saida: {size} bytes")
        else:
            print(f"\n[ERRO] Arquivo de saida NAO criado!")

# =============================================================================
# 4. Conclusao
# =============================================================================
print("\n" + "="*70)
print("4. CONCLUSAO E HIPOTESE")
print("="*70)

print("""
HIPOTESE PRINCIPAL: CDCMS_CIL_GMean pode ter problemas com datasets multiclasse.

Observacoes:
- LED: 10 classes (digitos 0-9)
- WAVEFORM: 3 classes
- CovType: 7 classes (tipos de cobertura florestal)
- Shuttle: 7 classes
- RBF_Stationary: Possivelmente multiclasse ou alta dimensionalidade

SOLUCAO PROPOSTA:
1. Verificar se CDCMS_CIL (sem GMean) funciona melhor com multiclasse
2. Ou: Usar apenas CDCMS para datasets binarios
3. Ou: Investigar se ha versao do CDCMS para multiclasse
""")
