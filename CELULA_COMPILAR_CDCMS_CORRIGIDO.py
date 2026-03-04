# =============================================================================
# COMPILAR CDCMS.CIL - VERSAO CORRIGIDA
# =============================================================================
# Correcoes:
# 1. Usar src/main/java (nao src/test/java)
# 2. Compilar APENAS as classes CDCMS (nao todo o MOA)
# 3. Ignorar dependencias de teste (junit, difflib)
# =============================================================================

import subprocess
from pathlib import Path
import shutil
import time

print("="*70)
print("COMPILAR CDCMS.CIL - VERSAO CORRIGIDA")
print("="*70)

# =============================================================================
# PASSO 0: Definir diretorios
# =============================================================================
WORK_DIR = Path('/content')
CDCMS_REPO_DIR = WORK_DIR / 'CDCMS_CIL_src' / 'CDCMS.CIL'
ROSE_JARS_DIR = WORK_DIR / 'rose_jars'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

MOA_DEPS_JAR = ROSE_JARS_DIR / 'MOA-dependencies.jar'
SIZEOFAG_JAR = ROSE_JARS_DIR / 'sizeofag-1.0.4.jar'

# Verificar pre-requisitos
print("\n--- Verificar pre-requisitos ---")
for path, name in [(MOA_DEPS_JAR, "MOA-dependencies.jar"), (CDCMS_REPO_DIR, "CDCMS.CIL repo")]:
    if path.exists():
        print(f"[OK] {name}")
    else:
        print(f"[ERRO] {name} nao encontrado!")

# =============================================================================
# PASSO 1: Identificar estrutura correta
# =============================================================================
print("\n--- PASSO 1: Identificar estrutura do repositorio ---")

# O codigo fonte esta em Implementation/moa/src/main/java
SRC_MAIN_DIR = CDCMS_REPO_DIR / 'Implementation' / 'moa' / 'src' / 'main' / 'java'

if SRC_MAIN_DIR.exists():
    print(f"[OK] Diretorio fonte: {SRC_MAIN_DIR}")

    # Listar arquivos CDCMS especificos
    cdcms_meta_dir = SRC_MAIN_DIR / 'moa' / 'classifiers' / 'meta'
    cdcms_drift_dir = SRC_MAIN_DIR / 'moa' / 'classifiers' / 'core' / 'driftdetection'

    print("\nArquivos em moa/classifiers/meta/:")
    cdcms_files = []
    if cdcms_meta_dir.exists():
        for f in cdcms_meta_dir.glob("*.java"):
            print(f"  {f.name}")
            cdcms_files.append(f)

    print("\nArquivos em moa/classifiers/core/driftdetection/:")
    if cdcms_drift_dir.exists():
        for f in cdcms_drift_dir.glob("*.java"):
            if "GMean" in f.name or "CDCMS" in f.name or "DDM" in f.name or "PMAUC" in f.name:
                print(f"  {f.name}")
                cdcms_files.append(f)

    print(f"\nTotal de arquivos CDCMS a compilar: {len(cdcms_files)}")
else:
    print(f"[ERRO] Diretorio fonte nao encontrado: {SRC_MAIN_DIR}")
    cdcms_files = []

# =============================================================================
# PASSO 2: Compilar APENAS classes CDCMS
# =============================================================================
print("\n--- PASSO 2: Compilar classes CDCMS ---")

BUILD_DIR = WORK_DIR / 'cdcms_build'
if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)
BUILD_DIR.mkdir()

if cdcms_files and MOA_DEPS_JAR.exists():
    # Criar arquivo com lista de fontes
    sources_file = BUILD_DIR / 'sources.txt'
    with open(sources_file, 'w') as f:
        for jf in cdcms_files:
            f.write(str(jf) + '\n')

    print(f"Compilando {len(cdcms_files)} arquivos...")

    # Compilar com warnings suprimidos
    compile_cmd = [
        "javac",
        "-d", str(BUILD_DIR),
        "-cp", str(MOA_DEPS_JAR),
        "-source", "11",
        "-target", "11",
        "-Xlint:none",  # Suprimir todos os warnings
        f"@{sources_file}"
    ]

    result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)

    if result.returncode == 0:
        print("[OK] Compilacao bem-sucedida!")
    else:
        print(f"[INFO] Compilacao com erros (returncode={result.returncode})")
        # Mostrar erros especificos
        if result.stderr:
            errors = [l for l in result.stderr.split('\n') if 'error:' in l.lower()]
            print(f"Erros encontrados: {len(errors)}")
            for e in errors[:5]:
                print(f"  {e[:80]}")

    # Verificar classes geradas
    class_files = list(BUILD_DIR.rglob("*.class"))
    print(f"\nClasses geradas: {len(class_files)}")

    # Listar classes CDCMS
    for cf in class_files:
        if "CDCMS" in cf.name or "GMean" in cf.name:
            print(f"  [OK] {cf.name}")
else:
    class_files = []
    print("[SKIP] Pre-requisitos nao atendidos")

# =============================================================================
# PASSO 3: Criar JAR
# =============================================================================
print("\n--- PASSO 3: Criar JAR ---")

CDCMS_RECOMPILED_JAR = CDCMS_JARS_DIR / 'cdcms_cil_recompiled.jar'

if class_files:
    print(f"Criando JAR com {len(class_files)} classes...")

    jar_cmd = ["jar", "cf", str(CDCMS_RECOMPILED_JAR), "-C", str(BUILD_DIR), "."]
    result = subprocess.run(jar_cmd, capture_output=True, text=True, timeout=60)

    if CDCMS_RECOMPILED_JAR.exists():
        print(f"[OK] JAR criado: {CDCMS_RECOMPILED_JAR.stat().st_size/1024:.1f} KB")

        # Verificar conteudo
        verify = subprocess.run(
            f'jar tf "{CDCMS_RECOMPILED_JAR}" | grep -E "CDCMS|GMean"',
            shell=True, capture_output=True, text=True
        )
        if verify.stdout.strip():
            print("\nClasses no JAR:")
            for line in verify.stdout.strip().split('\n')[:10]:
                print(f"  {line}")
    else:
        print(f"[ERRO] {result.stderr}")
else:
    print("[SKIP] Nenhuma classe para empacotar")

# =============================================================================
# PASSO 4: Testar execucao
# =============================================================================
print("\n--- PASSO 4: Testar execucao ---")

# Criar arquivo ARFF de teste
TEST_DIR.mkdir(exist_ok=True)
test_arff = TEST_DIR / 'test_data.arff'

test_arff_content = """@relation test_imbalanced
@attribute a1 numeric
@attribute a2 numeric
@attribute a3 numeric
@attribute class {0,1}
@data
"""

import random
random.seed(42)
data_lines = []
for i in range(2000):
    a1 = random.gauss(0, 1) + (0 if random.random() < 0.9 else 2)
    a2 = random.gauss(0, 1) + (0 if random.random() < 0.9 else 2)
    a3 = random.gauss(0, 1)
    cls = 0 if random.random() < 0.9 else 1
    data_lines.append(f"{a1:.4f},{a2:.4f},{a3:.4f},{cls}")

with open(test_arff, 'w') as f:
    f.write(test_arff_content + '\n'.join(data_lines))

print(f"[OK] Arquivo teste: {test_arff.stat().st_size} bytes")

# Testar
if CDCMS_RECOMPILED_JAR.exists() and MOA_DEPS_JAR.exists():
    classpath = f"{CDCMS_RECOMPILED_JAR}:{MOA_DEPS_JAR}"

    # Teste 1: WriteCommandLineTemplate
    print("\n  Teste 1: WriteCommandLineTemplate")
    test1_cmd = [
        "java", "-Xmx2g",
        "-cp", classpath,
        "moa.DoTask",
        "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean"
    ]

    result1 = subprocess.run(test1_cmd, capture_output=True, text=True, timeout=30)
    output1 = result1.stdout + result1.stderr

    if "Exception" not in output1 and result1.returncode == 0:
        print("    [OK] Classe reconhecida!")
        for line in output1.strip().split('\n')[:3]:
            if line.strip():
                print(f"    {line[:70]}")
    else:
        print("    [FALHA]")
        for line in output1.strip().split('\n')[:5]:
            if line.strip():
                print(f"    {line[:80]}")

    # Teste 2: Execucao real
    print("\n  Teste 2: EvaluateInterleavedTestThenTrain")
    output_file = TEST_DIR / 'cdcms_recompiled_output.csv'

    test_arff_abs = str(test_arff.resolve())
    output_file_abs = str(output_file.resolve())

    task_string = f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff_abs}) -l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2) -f 500 -d {output_file_abs}"

    exec_cmd = [
        "java", "-Xmx4g",
        "-cp", classpath,
        "moa.DoTask",
        task_string
    ]

    # Adicionar javaagent se existir
    if SIZEOFAG_JAR.exists():
        exec_cmd.insert(2, f"-javaagent:{SIZEOFAG_JAR}")

    print(f"    Executando...")
    start = time.time()

    try:
        result2 = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=180)
        duration = time.time() - start

        print(f"    Tempo: {duration:.1f}s")
        print(f"    Return code: {result2.returncode}")

        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"    [SUCESSO] Arquivo criado: {output_file.stat().st_size} bytes")

            with open(output_file) as f:
                lines = f.readlines()
            print(f"    Linhas: {len(lines)}")
            if lines:
                print(f"    Header: {lines[0].strip()[:60]}...")
        else:
            print(f"    [FALHA] Arquivo nao criado")
            if result2.stderr:
                for line in result2.stderr.strip().split('\n')[:5]:
                    print(f"    {line[:80]}")
            if result2.stdout:
                for line in result2.stdout.strip().split('\n')[:3]:
                    print(f"    {line[:80]}")

    except subprocess.TimeoutExpired:
        print("    [ERRO] Timeout!")
    except Exception as e:
        print(f"    [ERRO] {e}")
else:
    print("[SKIP] JAR nao disponivel")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO")
print("="*70)

output_file = TEST_DIR / 'cdcms_recompiled_output.csv'
if output_file.exists() and output_file.stat().st_size > 0:
    print("\n*** SUCESSO! CDCMS.CIL recompilado funciona! ***")
    print(f"\nUse este JAR: {CDCMS_RECOMPILED_JAR}")
else:
    print("\n[RESULTADO] Verifique os testes acima")

    # Se compilou mas nao executou, pode ser problema de dependencia
    if CDCMS_RECOMPILED_JAR.exists():
        print("\nO JAR foi criado mas a execucao falhou.")
        print("Isso pode indicar dependencias faltando em tempo de execucao.")
        print("\nTente o proximo script: CELULA_USAR_CDCMS_ORIGINAL_COM_MOA_MAVEN.py")
