# =============================================================================
# SOLUCAO COMPLETA CDCMS.CIL - TUDO EM UMA CELULA
# =============================================================================
# Este script faz TUDO:
# 1. Baixa MOA-dependencies.jar do ROSE
# 2. Clona e compila CDCMS.CIL do fonte (TODOS os arquivos)
# 3. Testa execucao
# =============================================================================

import subprocess
from pathlib import Path
import shutil
import time
import urllib.request

print("="*70)
print("SOLUCAO COMPLETA CDCMS.CIL")
print("="*70)

# =============================================================================
# CONFIGURACAO
# =============================================================================
WORK_DIR = Path('/content')
ROSE_JARS_DIR = WORK_DIR / 'rose_jars'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
CDCMS_SRC_DIR = WORK_DIR / 'CDCMS_CIL_src'
BUILD_DIR = WORK_DIR / 'cdcms_build'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

# Criar diretorios
for d in [ROSE_JARS_DIR, CDCMS_JARS_DIR, CDCMS_SRC_DIR, BUILD_DIR, TEST_DIR]:
    d.mkdir(exist_ok=True)

# =============================================================================
# PASSO 1: Baixar MOA-dependencies.jar
# =============================================================================
print("\n--- PASSO 1: MOA-dependencies.jar ---")

MOA_DEPS_JAR = ROSE_JARS_DIR / 'MOA-dependencies.jar'
SIZEOFAG_JAR = ROSE_JARS_DIR / 'sizeofag-1.0.4.jar'

for jar_name, jar_path, min_size in [
    ("MOA-dependencies.jar", MOA_DEPS_JAR, 50*1024*1024),
    ("sizeofag-1.0.4.jar", SIZEOFAG_JAR, 1000)
]:
    if jar_path.exists() and jar_path.stat().st_size > min_size:
        print(f"[OK] {jar_name}: {jar_path.stat().st_size/(1024*1024):.1f} MB")
    else:
        print(f"Baixando {jar_name}...")
        url = f"https://github.com/canoalberto/ROSE/raw/master/{jar_name}"
        try:
            urllib.request.urlretrieve(url, jar_path)
            print(f"[OK] Baixado: {jar_path.stat().st_size/(1024*1024):.2f} MB")
        except Exception as e:
            print(f"[ERRO] {e}")

# =============================================================================
# PASSO 2: Clonar repositorio CDCMS.CIL
# =============================================================================
print("\n--- PASSO 2: Clonar CDCMS.CIL ---")

CDCMS_REPO_DIR = CDCMS_SRC_DIR / 'CDCMS.CIL'

if CDCMS_REPO_DIR.exists():
    print(f"[OK] Repositorio existe")
else:
    print("Clonando...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/michaelchiucw/CDCMS.CIL.git"],
        cwd=str(CDCMS_SRC_DIR),
        capture_output=True, text=True, timeout=120
    )
    if CDCMS_REPO_DIR.exists():
        print("[OK] Clonado")
    else:
        print(f"[ERRO] {result.stderr}")

# =============================================================================
# PASSO 3: Compilar TODOS os arquivos Java
# =============================================================================
print("\n--- PASSO 3: Compilar codigo fonte ---")

SRC_MAIN_DIR = CDCMS_REPO_DIR / 'Implementation' / 'moa' / 'src' / 'main' / 'java'

if SRC_MAIN_DIR.exists() and MOA_DEPS_JAR.exists():
    # Limpar build anterior
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir()

    # Listar TODOS os arquivos .java (nao so CDCMS)
    all_java_files = list(SRC_MAIN_DIR.rglob("*.java"))
    print(f"Total de arquivos .java: {len(all_java_files)}")

    # Criar arquivo com lista de fontes
    sources_file = BUILD_DIR / 'sources.txt'
    with open(sources_file, 'w') as f:
        for jf in all_java_files:
            f.write(str(jf) + '\n')

    # Compilar
    print("Compilando (pode demorar)...")
    compile_cmd = [
        "javac",
        "-d", str(BUILD_DIR),
        "-cp", str(MOA_DEPS_JAR),
        "-source", "11",
        "-target", "11",
        "-Xlint:none",
        "-encoding", "UTF-8",
        f"@{sources_file}"
    ]

    start = time.time()
    result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=600)
    compile_time = time.time() - start

    print(f"Tempo de compilacao: {compile_time:.1f}s")
    print(f"Return code: {result.returncode}")

    # Contar classes geradas
    class_files = list(BUILD_DIR.rglob("*.class"))
    print(f"Classes geradas: {len(class_files)}")

    # Verificar se CDCMS foi compilado
    cdcms_classes = [f for f in class_files if "CDCMS" in f.name]
    if cdcms_classes:
        print(f"\nClasses CDCMS compiladas:")
        for cf in cdcms_classes[:8]:
            print(f"  [OK] {cf.name}")
    else:
        print("\n[AVISO] Nenhuma classe CDCMS compilada")

        # Mostrar erros
        if result.stderr:
            errors = [l for l in result.stderr.split('\n') if 'error:' in l.lower()]
            print(f"\nErros de compilacao ({len(errors)}):")
            # Agrupar por tipo de erro
            error_types = {}
            for e in errors:
                if "cannot find symbol" in e:
                    error_types["cannot find symbol"] = error_types.get("cannot find symbol", 0) + 1
                elif "package" in e and "does not exist" in e:
                    error_types["package does not exist"] = error_types.get("package does not exist", 0) + 1
                else:
                    error_types["outro"] = error_types.get("outro", 0) + 1

            for et, count in error_types.items():
                print(f"  {et}: {count}")
else:
    class_files = []
    print("[ERRO] Pre-requisitos nao atendidos")

# =============================================================================
# PASSO 4: Criar JAR
# =============================================================================
print("\n--- PASSO 4: Criar JAR ---")

CDCMS_COMPILED_JAR = CDCMS_JARS_DIR / 'cdcms_cil_compiled.jar'

if class_files:
    print(f"Empacotando {len(class_files)} classes...")

    jar_cmd = ["jar", "cf", str(CDCMS_COMPILED_JAR), "-C", str(BUILD_DIR), "."]
    result = subprocess.run(jar_cmd, capture_output=True, text=True, timeout=60)

    if CDCMS_COMPILED_JAR.exists():
        print(f"[OK] JAR criado: {CDCMS_COMPILED_JAR.stat().st_size/1024:.1f} KB")
    else:
        print(f"[ERRO] {result.stderr}")
else:
    print("[SKIP] Nenhuma classe compilada")

    # Alternativa: tentar usar JAR do repositorio se existir
    print("\nProcurando JARs pre-compilados no repositorio...")
    for jar_file in CDCMS_REPO_DIR.rglob("*.jar"):
        print(f"  Encontrado: {jar_file}")
        if "cdcms" in jar_file.name.lower() or "moa" in jar_file.name.lower():
            # Copiar
            dest = CDCMS_JARS_DIR / jar_file.name
            shutil.copy(jar_file, dest)
            print(f"  -> Copiado para {dest}")

# =============================================================================
# PASSO 5: Testar execucao
# =============================================================================
print("\n--- PASSO 5: Testar execucao ---")

# Encontrar JAR CDCMS para testar
CDCMS_JAR = None
for candidate in [
    CDCMS_COMPILED_JAR,
    CDCMS_JARS_DIR / 'cdcms_cil.jar',
    CDCMS_JARS_DIR / 'cdcms_cil_clean.jar'
]:
    if candidate.exists():
        CDCMS_JAR = candidate
        break

# Criar arquivo de teste
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
    a2 = random.gauss(0, 1)
    a3 = random.gauss(0, 1)
    cls = 0 if random.random() < 0.9 else 1
    data_lines.append(f"{a1:.4f},{a2:.4f},{a3:.4f},{cls}")

with open(test_arff, 'w') as f:
    f.write(test_arff_content + '\n'.join(data_lines))

print(f"[OK] Arquivo teste: {test_arff}")

if CDCMS_JAR and MOA_DEPS_JAR.exists():
    print(f"\nUsando JAR: {CDCMS_JAR.name}")

    classpath = f"{CDCMS_JAR}:{MOA_DEPS_JAR}"

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
        test1_ok = True
    else:
        print("    [FALHA]")
        for line in output1.strip().split('\n')[:5]:
            if line.strip():
                print(f"    {line[:80]}")
        test1_ok = False

    # Teste 2: Execucao real
    if test1_ok:
        print("\n  Teste 2: EvaluateInterleavedTestThenTrain")
        output_file = TEST_DIR / 'cdcms_output.csv'

        task_string = f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff}) -l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2) -f 500 -d {output_file}"

        exec_cmd = [
            "java", "-Xmx4g",
            "-cp", classpath,
            "moa.DoTask",
            task_string
        ]

        if SIZEOFAG_JAR.exists():
            exec_cmd.insert(2, f"-javaagent:{SIZEOFAG_JAR}")

        print("    Executando...")
        start = time.time()

        try:
            result2 = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=300)
            duration = time.time() - start

            print(f"    Tempo: {duration:.1f}s")

            if output_file.exists() and output_file.stat().st_size > 0:
                print(f"    [SUCESSO] {output_file.stat().st_size} bytes")

                with open(output_file) as f:
                    lines = f.readlines()
                print(f"    Linhas: {len(lines)}")
                if lines:
                    print(f"    Header: {lines[0].strip()[:60]}...")
            else:
                print("    [FALHA] Arquivo nao criado")
                if result2.stderr:
                    for line in result2.stderr.strip().split('\n')[:5]:
                        print(f"    {line[:80]}")

        except subprocess.TimeoutExpired:
            print("    [TIMEOUT]")
        except Exception as e:
            print(f"    [ERRO] {e}")
else:
    print("[SKIP] JAR CDCMS nao disponivel")

# =============================================================================
# RESUMO FINAL
# =============================================================================
print("\n" + "="*70)
print("RESUMO FINAL")
print("="*70)

# Verificar resultado
output_file = TEST_DIR / 'cdcms_output.csv'
if output_file.exists() and output_file.stat().st_size > 0:
    print("\n*** SUCESSO! CDCMS.CIL FUNCIONANDO! ***")
    print(f"\nJAR: {CDCMS_JAR}")
    print(f"Output: {output_file}")
else:
    print("\n[FALHA] CDCMS.CIL ainda nao funciona")

    # Diagnostico adicional
    print("\n--- Diagnostico ---")

    # Verificar se o repositorio tem instrucoes de build
    readme = CDCMS_REPO_DIR / 'README.md'
    if readme.exists():
        print("\nConteudo do README.md:")
        with open(readme) as f:
            content = f.read()
        print(content[:2000])

    # Verificar se tem pom.xml em algum lugar
    pom_files = list(CDCMS_REPO_DIR.rglob("pom.xml"))
    if pom_files:
        print(f"\nArquivos pom.xml encontrados:")
        for pf in pom_files:
            print(f"  {pf.relative_to(CDCMS_REPO_DIR)}")

            # Ler versao MOA
            with open(pf) as f:
                pom_content = f.read()
            import re
            moa_match = re.search(r'<artifactId>moa</artifactId>\s*<version>([^<]+)', pom_content)
            if moa_match:
                print(f"    -> MOA version: {moa_match.group(1)}")

    # Listar estrutura do repositorio
    print("\nEstrutura do repositorio:")
    for item in sorted(CDCMS_REPO_DIR.iterdir()):
        if item.is_dir():
            print(f"  [DIR] {item.name}/")
        else:
            print(f"  [FILE] {item.name}")

    print("\n--- Proximos passos ---")
    print("1. Verificar se existe pom.xml e compilar com Maven")
    print("2. Contatar autores: https://github.com/michaelchiucw/CDCMS.CIL/issues")
    print("3. Verificar releases: https://github.com/michaelchiucw/CDCMS.CIL/releases")
