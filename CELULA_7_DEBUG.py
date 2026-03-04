# =============================================================================
# CELULA 7 DEBUG: Diagnosticar problema na execucao do CDCMS
# =============================================================================
# Execute esta celula para identificar o problema exato
# =============================================================================

import subprocess
import time
import pandas as pd
from pathlib import Path

print("="*70)
print("DEBUG: DIAGNOSTICAR PROBLEMA NA EXECUCAO DO CDCMS")
print("="*70)

# =============================================================================
# 1. Verificar variaveis
# =============================================================================
print("\n--- 1. VERIFICAR VARIAVEIS ---")

variables_ok = True

# Verificar se variaveis existem
try:
    print(f"CDCMS_JAR: {CDCMS_JAR}")
    print(f"  Existe: {CDCMS_JAR.exists()}")
except NameError:
    print("[ERRO] CDCMS_JAR nao definido!")
    variables_ok = False

try:
    print(f"MOA_DEPS_JAR: {MOA_DEPS_JAR}")
    print(f"  Existe: {MOA_DEPS_JAR.exists()}")
except NameError:
    print("[ERRO] MOA_DEPS_JAR nao definido!")
    variables_ok = False

try:
    print(f"TEST_DIR: {TEST_DIR}")
    print(f"  Existe: {TEST_DIR.exists()}")
except NameError:
    print("[ERRO] TEST_DIR nao definido!")
    variables_ok = False

try:
    print(f"TEMP_ARFF_DIR: {TEMP_ARFF_DIR}")
    print(f"  Existe: {TEMP_ARFF_DIR.exists()}")
except NameError:
    print("[ERRO] TEMP_ARFF_DIR nao definido!")
    # Criar se nao existir
    TEMP_ARFF_DIR = Path('/content/cdcms_temp_arff')
    TEMP_ARFF_DIR.mkdir(exist_ok=True)
    print(f"  [CRIADO] {TEMP_ARFF_DIR}")

try:
    print(f"UNIFIED_CHUNKS_DIR: {UNIFIED_CHUNKS_DIR}")
    print(f"  Existe: {UNIFIED_CHUNKS_DIR.exists()}")
except NameError:
    print("[ERRO] UNIFIED_CHUNKS_DIR nao definido!")
    variables_ok = False

# =============================================================================
# 2. Verificar CDCMSEvaluator.class
# =============================================================================
print("\n--- 2. VERIFICAR CDCMSEvaluator.class ---")

evaluator_class = TEST_DIR / 'CDCMSEvaluator.class'
if evaluator_class.exists():
    print(f"[OK] {evaluator_class}")
    print(f"     Tamanho: {evaluator_class.stat().st_size} bytes")
else:
    print(f"[ERRO] CDCMSEvaluator.class NAO ENCONTRADO!")
    print(f"       Esperado em: {evaluator_class}")
    print("       Execute a CELULA 5.1 CORRIGIDA primeiro!")
    variables_ok = False

# =============================================================================
# 3. Testar com dataset simples
# =============================================================================
print("\n--- 3. TESTAR COM UM DATASET ---")

dataset_name = "SEA_Abrupt_Simple"
chunk_size_name = "chunk_500"

# Verificar se o dataset existe
data_path = UNIFIED_CHUNKS_DIR / chunk_size_name / dataset_name
print(f"Dataset path: {data_path}")
print(f"  Existe: {data_path.exists()}")

if data_path.exists():
    # Listar chunks
    chunks = sorted(data_path.glob("chunk_*.csv"))
    print(f"  Chunks encontrados: {len(chunks)}")

    if chunks:
        # Mostrar primeiro chunk
        print(f"  Primeiro: {chunks[0].name}")
        print(f"  Ultimo: {chunks[-1].name}")

# =============================================================================
# 4. Carregar e converter dados
# =============================================================================
print("\n--- 4. CARREGAR E CONVERTER DADOS ---")

if data_path.exists():
    # Carregar chunks
    chunks = sorted(data_path.glob("chunk_*.csv"),
                   key=lambda x: int(x.stem.split('_')[1]))

    all_data = []
    for chunk_file in chunks:
        df = pd.read_csv(chunk_file)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    print(f"Dados carregados:")
    print(f"  Instancias: {len(combined_df)}")
    print(f"  Colunas: {list(combined_df.columns)}")
    print(f"  Classes: {sorted(combined_df[combined_df.columns[-1]].unique())}")

    # Criar ARFF
    arff_file = TEMP_ARFF_DIR / f"{dataset_name}_debug.arff"

    print(f"\nCriando ARFF: {arff_file}")

    with open(arff_file, 'w') as f:
        f.write(f"@relation {dataset_name}\n\n")

        for col in combined_df.columns[:-1]:
            f.write(f"@attribute {col} numeric\n")

        class_col = combined_df.columns[-1]
        unique_classes = sorted(combined_df[class_col].unique())
        class_str = ",".join(str(int(c)) for c in unique_classes)
        f.write(f"@attribute class {{{class_str}}}\n\n")

        f.write("@data\n")
        for _, row in combined_df.iterrows():
            values = ",".join(str(v) for v in row)
            f.write(f"{values}\n")

    print(f"  [OK] ARFF criado")
    print(f"  Tamanho: {arff_file.stat().st_size} bytes")

    # Mostrar primeiras linhas do ARFF
    print(f"\n  Primeiras 15 linhas do ARFF:")
    with open(arff_file) as f:
        for i, line in enumerate(f):
            if i < 15:
                print(f"    {line.rstrip()}")
            else:
                break

# =============================================================================
# 5. Executar CDCMS com debug completo
# =============================================================================
print("\n--- 5. EXECUTAR CDCMS COM DEBUG ---")

if data_path.exists() and evaluator_class.exists():
    output_file = TEST_DIR / f"{dataset_name}_debug_output.csv"

    # Remover arquivo anterior se existir
    if output_file.exists():
        output_file.unlink()

    # Montar classpath
    classpath = f"{CDCMS_JAR}:{MOA_DEPS_JAR}"

    # Comando
    cmd = [
        "java", "-Xmx4g",
        "-cp", f"{classpath}:{TEST_DIR}",
        "CDCMSEvaluator",
        str(arff_file),
        str(output_file),
        "CDCMS_CIL_GMean"
    ]

    print("Comando:")
    print(f"  {' '.join(cmd)}")

    print(f"\nArquivos:")
    print(f"  ARFF input:  {arff_file}")
    print(f"    Existe: {arff_file.exists()}")
    print(f"  CSV output:  {output_file}")

    print("\nExecutando...")
    start_time = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300
    )

    duration = time.time() - start_time

    print(f"\n--- RESULTADO DA EXECUCAO ---")
    print(f"Return code: {result.returncode}")
    print(f"Tempo: {duration:.1f}s")

    print(f"\nSTDOUT:")
    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")
    else:
        print("  (vazio)")

    print(f"\nSTDERR:")
    if result.stderr:
        # Filtrar warnings comuns
        for line in result.stderr.strip().split('\n'):
            if 'WARNING' not in line or 'ARPACK' in line or 'Error' in line or 'Exception' in line:
                print(f"  {line}")
    else:
        print("  (vazio)")

    print(f"\nArquivo de saida:")
    if output_file.exists():
        size = output_file.stat().st_size
        print(f"  [OK] Existe! Tamanho: {size} bytes")

        if size > 0:
            # Mostrar primeiras linhas
            print(f"\n  Primeiras 10 linhas:")
            with open(output_file) as f:
                for i, line in enumerate(f):
                    if i < 10:
                        print(f"    {line.rstrip()}")
                    else:
                        break

            # Contar linhas
            with open(output_file) as f:
                num_lines = sum(1 for _ in f)
            print(f"\n  Total de linhas: {num_lines}")
        else:
            print("  [ERRO] Arquivo vazio!")
    else:
        print(f"  [ERRO] Arquivo NAO existe!")
        print(f"         Esperado: {output_file}")

else:
    print("[ERRO] Nao foi possivel executar o teste")
    if not data_path.exists():
        print(f"  - Dataset nao encontrado: {data_path}")
    if not evaluator_class.exists():
        print(f"  - CDCMSEvaluator.class nao encontrado")

# =============================================================================
# 6. Resumo
# =============================================================================
print("\n" + "="*70)
print("RESUMO DO DIAGNOSTICO")
print("="*70)

if output_file.exists() and output_file.stat().st_size > 0:
    print("\n[OK] CDCMS executou com sucesso!")
    print("     O problema pode estar em outra parte do codigo.")
else:
    print("\n[ERRO] CDCMS nao gerou saida.")
    print("\nPossiveis causas:")
    print("  1. Erro no Java (verifique STDERR acima)")
    print("  2. Problema com o arquivo ARFF")
    print("  3. Classpath incorreto")
    print("  4. Timeout")
