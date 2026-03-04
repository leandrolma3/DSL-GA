# =============================================================================
# SOLUCAO DEFINITIVA: Recompilar CDCMS.CIL do Fonte
# =============================================================================
# O problema identificado: CDCMS.CIL foi compilado contra uma versao diferente
# do MOA. A solucao e recompilar do codigo fonte usando o MOA-dependencies.jar
# =============================================================================

import subprocess
import urllib.request
from pathlib import Path
import shutil
import time
import os

print("="*70)
print("SOLUCAO DEFINITIVA: Recompilar CDCMS.CIL")
print("="*70)

# =============================================================================
# PASSO 0: Definir diretorios
# =============================================================================
WORK_DIR = Path('/content')
CDCMS_SRC_DIR = WORK_DIR / 'CDCMS_CIL_src'
ROSE_JARS_DIR = WORK_DIR / 'rose_jars'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

# Criar diretorios
for d in [CDCMS_SRC_DIR, ROSE_JARS_DIR, CDCMS_JARS_DIR, TEST_DIR]:
    d.mkdir(exist_ok=True)

# =============================================================================
# PASSO 1: Baixar MOA-dependencies.jar do ROSE (se nao existir)
# =============================================================================
print("\n--- PASSO 1: Verificar MOA-dependencies.jar ---")

MOA_DEPS_JAR = ROSE_JARS_DIR / 'MOA-dependencies.jar'
SIZEOFAG_JAR = ROSE_JARS_DIR / 'sizeofag-1.0.4.jar'

for jar_name, jar_path in [("MOA-dependencies.jar", MOA_DEPS_JAR), ("sizeofag-1.0.4.jar", SIZEOFAG_JAR)]:
    if jar_path.exists() and jar_path.stat().st_size > 1000000:  # > 1MB
        print(f"[OK] {jar_name}: {jar_path.stat().st_size/(1024*1024):.1f} MB")
    else:
        print(f"Baixando {jar_name}...")
        url = f"https://github.com/canoalberto/ROSE/raw/master/{jar_name}"
        try:
            urllib.request.urlretrieve(url, jar_path)
            print(f"[OK] Baixado: {jar_path.stat().st_size/(1024*1024):.1f} MB")
        except Exception as e:
            print(f"[ERRO] {e}")

# =============================================================================
# PASSO 2: Clonar repositorio CDCMS.CIL
# =============================================================================
print("\n--- PASSO 2: Clonar repositorio CDCMS.CIL ---")

CDCMS_REPO_DIR = CDCMS_SRC_DIR / 'CDCMS.CIL'

if CDCMS_REPO_DIR.exists():
    print(f"[OK] Repositorio ja existe: {CDCMS_REPO_DIR}")
else:
    print("Clonando https://github.com/michaelchiucw/CDCMS.CIL ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/michaelchiucw/CDCMS.CIL.git"],
        cwd=str(CDCMS_SRC_DIR),
        capture_output=True, text=True, timeout=120
    )
    if CDCMS_REPO_DIR.exists():
        print(f"[OK] Clonado com sucesso")
    else:
        print(f"[ERRO] Falha ao clonar: {result.stderr}")

# =============================================================================
# PASSO 3: Identificar arquivos fonte CDCMS
# =============================================================================
print("\n--- PASSO 3: Identificar arquivos fonte ---")

CDCMS_SOURCE_FILES = []
if CDCMS_REPO_DIR.exists():
    # Procurar por arquivos .java
    for java_file in CDCMS_REPO_DIR.rglob("*.java"):
        CDCMS_SOURCE_FILES.append(java_file)

    print(f"Arquivos .java encontrados: {len(CDCMS_SOURCE_FILES)}")

    # Mostrar estrutura de pastas
    src_dirs = set()
    for f in CDCMS_SOURCE_FILES:
        rel_path = f.relative_to(CDCMS_REPO_DIR)
        if len(rel_path.parts) > 1:
            src_dirs.add(rel_path.parts[0])

    print(f"Diretorios fonte: {sorted(src_dirs)}")

    # Procurar especificamente por CDCMS_CIL_GMean
    gmean_files = [f for f in CDCMS_SOURCE_FILES if "GMean" in f.name or "CDCMS" in f.name]
    print(f"\nArquivos CDCMS/GMean:")
    for f in gmean_files[:10]:
        print(f"  {f.relative_to(CDCMS_REPO_DIR)}")
else:
    print("[ERRO] Repositorio nao encontrado")

# =============================================================================
# PASSO 4: Compilar CDCMS contra MOA-dependencies.jar
# =============================================================================
print("\n--- PASSO 4: Compilar CDCMS ---")

BUILD_DIR = CDCMS_SRC_DIR / 'build'
BUILD_DIR.mkdir(exist_ok=True)

if CDCMS_SOURCE_FILES and MOA_DEPS_JAR.exists():
    # Encontrar diretorio src
    src_candidates = list(CDCMS_REPO_DIR.glob("**/moa/classifiers/meta/*.java"))

    if src_candidates:
        # Determinar raiz do codigo fonte
        src_root = src_candidates[0].parent.parent.parent.parent  # subir de moa/classifiers/meta
        print(f"Raiz do codigo fonte: {src_root}")

        # Listar todos os .java a compilar
        java_files = list(src_root.rglob("*.java"))
        print(f"Arquivos a compilar: {len(java_files)}")

        # Criar arquivo com lista de arquivos
        sources_file = BUILD_DIR / 'sources.txt'
        with open(sources_file, 'w') as f:
            for jf in java_files:
                f.write(str(jf) + '\n')

        # Compilar
        print("\nCompilando...")
        compile_cmd = [
            "javac",
            "-d", str(BUILD_DIR),
            "-cp", str(MOA_DEPS_JAR),
            "-source", "11",
            "-target", "11",
            "-Xlint:-deprecation",
            "-Xlint:-unchecked",
            f"@{sources_file}"
        ]

        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("[OK] Compilacao bem-sucedida!")

            # Contar classes compiladas
            class_files = list(BUILD_DIR.rglob("*.class"))
            print(f"Classes geradas: {len(class_files)}")

            # Verificar se CDCMS_CIL_GMean foi gerada
            gmean_class = list(BUILD_DIR.rglob("CDCMS_CIL_GMean.class"))
            if gmean_class:
                print(f"[OK] CDCMS_CIL_GMean.class encontrada: {gmean_class[0]}")
            else:
                print("[AVISO] CDCMS_CIL_GMean.class NAO encontrada")
        else:
            print(f"[ERRO] Compilacao falhou")
            # Mostrar erros
            errors = result.stderr.split('\n')
            unique_errors = set()
            for line in errors:
                if "error:" in line.lower():
                    # Extrair tipo de erro
                    error_type = line.split("error:")[-1].strip()[:60]
                    unique_errors.add(error_type)

            print(f"\nErros unicos ({len(unique_errors)}):")
            for err in list(unique_errors)[:10]:
                print(f"  - {err}")
    else:
        print("[ERRO] Arquivos fonte CDCMS nao encontrados na estrutura esperada")
else:
    print("[SKIP] Pre-requisitos nao atendidos")

# =============================================================================
# PASSO 5: Criar JAR com classes compiladas
# =============================================================================
print("\n--- PASSO 5: Criar JAR ---")

CDCMS_RECOMPILED_JAR = CDCMS_JARS_DIR / 'cdcms_cil_recompiled.jar'

class_files = list(BUILD_DIR.rglob("*.class"))
if class_files:
    print(f"Criando JAR com {len(class_files)} classes...")

    # Criar JAR
    jar_cmd = [
        "jar", "cf", str(CDCMS_RECOMPILED_JAR),
        "-C", str(BUILD_DIR), "."
    ]

    result = subprocess.run(jar_cmd, capture_output=True, text=True, timeout=60)

    if CDCMS_RECOMPILED_JAR.exists():
        print(f"[OK] JAR criado: {CDCMS_RECOMPILED_JAR.stat().st_size/1024:.1f} KB")

        # Verificar conteudo
        verify_cmd = f'jar tf "{CDCMS_RECOMPILED_JAR}" | grep -E "CDCMS|GMean"'
        verify = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)
        if verify.stdout.strip():
            print("\nClasses CDCMS no JAR:")
            for line in verify.stdout.strip().split('\n')[:10]:
                print(f"  {line}")
    else:
        print(f"[ERRO] Falha ao criar JAR: {result.stderr}")
else:
    print("[SKIP] Nenhuma classe compilada")

# =============================================================================
# PASSO 6: Testar execucao
# =============================================================================
print("\n--- PASSO 6: Testar execucao ---")

# Criar arquivo ARFF de teste
test_arff = TEST_DIR / 'test_data.arff'
test_arff_content = """@relation test_imbalanced
@attribute a1 numeric
@attribute a2 numeric
@attribute a3 numeric
@attribute class {0,1}
@data
"""
# Gerar dados desbalanceados (90% classe 0, 10% classe 1)
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

print(f"[OK] Arquivo teste criado: {test_arff.stat().st_size} bytes")

# Testar execucao
if CDCMS_RECOMPILED_JAR.exists() and MOA_DEPS_JAR.exists():
    output_file = TEST_DIR / 'cdcms_recompiled_output.csv'

    classpath = f"{CDCMS_RECOMPILED_JAR}:{MOA_DEPS_JAR}"

    test_arff_abs = str(test_arff.resolve())
    output_file_abs = str(output_file.resolve())

    # Testar primeiro com WriteCommandLineTemplate
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
        print("    [OK] WriteCommandLineTemplate funcionou!")
        for line in output1.strip().split('\n')[:3]:
            print(f"    {line[:70]}")
    else:
        print("    [FALHA]")
        for line in output1.strip().split('\n')[:3]:
            if line.strip():
                print(f"    {line[:70]}")

    # Teste 2: Execucao real
    print("\n  Teste 2: EvaluateInterleavedTestThenTrain")

    task_string = f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff_abs}) -l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2) -f 500 -d {output_file_abs}"

    exec_cmd = [
        "java", "-Xmx4g",
        "-javaagent:" + str(SIZEOFAG_JAR),
        "-cp", classpath,
        "moa.DoTask",
        task_string
    ]

    print(f"    Executando...")
    start = time.time()

    try:
        result2 = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=180)
        duration = time.time() - start

        print(f"    Tempo: {duration:.1f}s")
        print(f"    Return code: {result2.returncode}")

        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"    [SUCESSO] Arquivo criado: {output_file.stat().st_size} bytes")

            # Mostrar conteudo
            with open(output_file) as f:
                lines = f.readlines()
            print(f"    Linhas: {len(lines)}")
            if lines:
                print(f"    Header: {lines[0].strip()[:60]}...")
                if len(lines) > 1:
                    print(f"    Primeira linha: {lines[1].strip()[:60]}...")
        else:
            print(f"    [FALHA] Arquivo nao criado")
            if result2.stderr:
                for line in result2.stderr.strip().split('\n')[:5]:
                    print(f"    {line[:70]}")

    except subprocess.TimeoutExpired:
        print("    [ERRO] Timeout!")
    except Exception as e:
        print(f"    [ERRO] {e}")

else:
    print("[SKIP] Pre-requisitos nao atendidos")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO")
print("="*70)

success = False
if CDCMS_RECOMPILED_JAR.exists():
    output_file = TEST_DIR / 'cdcms_recompiled_output.csv'
    if output_file.exists() and output_file.stat().st_size > 0:
        success = True
        print(f"\n*** SUCESSO! CDCMS.CIL recompilado funciona! ***")
        print(f"\nArquivos gerados:")
        print(f"  - JAR: {CDCMS_RECOMPILED_JAR}")
        print(f"  - Output: {output_file}")
        print(f"\nPara usar no notebook principal:")
        print(f'  CDCMS_JAR = "{CDCMS_RECOMPILED_JAR}"')

if not success:
    print("\n[FALHA] Recompilacao nao resolveu o problema")
    print("\nProximos passos:")
    print("  1. Verificar erros de compilacao acima")
    print("  2. Contatar autores do CDCMS.CIL para JAR funcional")
    print("  3. Verificar se existe pom.xml no repositorio CDCMS.CIL")
