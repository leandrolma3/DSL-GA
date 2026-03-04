# =============================================================================
# COMPILAR CDCMS.CIL - SOLUCAO FINAL COMPLETA
# =============================================================================
# Combina:
# 1. MOA-dependencies.jar do ROSE (tem nz.ac.waikato.cms.locator)
# 2. JARs do ND4J do Maven (tem org.nd4j.*)
# =============================================================================

import subprocess
from pathlib import Path
import shutil
import time
import urllib.request

print("="*70)
print("COMPILAR CDCMS.CIL - SOLUCAO FINAL")
print("="*70)

WORK_DIR = Path('/content')
ROSE_JARS_DIR = WORK_DIR / 'rose_jars'
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'  # JARs do Maven (ND4J)
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
CDCMS_SRC_DIR = WORK_DIR / 'CDCMS_CIL_src'
BUILD_DIR = WORK_DIR / 'cdcms_build'

for d in [ROSE_JARS_DIR, CDCMS_JARS_DIR, BUILD_DIR]:
    d.mkdir(exist_ok=True)

# =============================================================================
# PASSO 1: Verificar MOA-dependencies.jar
# =============================================================================
print("\n--- PASSO 1: Verificar MOA-dependencies.jar ---")

MOA_DEPS_JAR = ROSE_JARS_DIR / 'MOA-dependencies.jar'

if MOA_DEPS_JAR.exists() and MOA_DEPS_JAR.stat().st_size > 50*1024*1024:
    print(f"[OK] MOA-dependencies.jar: {MOA_DEPS_JAR.stat().st_size/(1024*1024):.1f} MB")
else:
    print("Baixando MOA-dependencies.jar...")
    url = "https://github.com/canoalberto/ROSE/raw/master/MOA-dependencies.jar"
    urllib.request.urlretrieve(url, MOA_DEPS_JAR)
    print(f"[OK] Baixado: {MOA_DEPS_JAR.stat().st_size/(1024*1024):.1f} MB")

# =============================================================================
# PASSO 2: Verificar JARs do ND4J (do Maven)
# =============================================================================
print("\n--- PASSO 2: Verificar JARs do ND4J ---")

nd4j_jars = list(DEPS_DIR.glob("nd4j*.jar"))
print(f"JARs ND4J encontrados: {len(nd4j_jars)}")

if len(nd4j_jars) == 0:
    print("[ERRO] JARs do ND4J nao encontrados!")
    print("Execute primeiro: CELULA_MAVEN_DEPS_COMPLETO.py")
else:
    for j in nd4j_jars[:5]:
        print(f"  - {j.name}")

# =============================================================================
# PASSO 3: Montar classpath COMBINADO
# =============================================================================
print("\n--- PASSO 3: Montar classpath ---")

# MOA-dependencies.jar PRIMEIRO (tem prioridade)
# Depois todos os JARs do Maven (incluindo ND4J)
all_maven_jars = list(DEPS_DIR.glob("*.jar"))

# Remover moa-*.jar do Maven para evitar conflito
maven_jars_filtered = [j for j in all_maven_jars if not j.name.startswith('moa-')]

classpath_parts = [str(MOA_DEPS_JAR)] + [str(j) for j in maven_jars_filtered]
classpath = ":".join(classpath_parts)

print(f"MOA-dependencies.jar: 1 JAR")
print(f"JARs do Maven: {len(maven_jars_filtered)}")
print(f"Total no classpath: {len(classpath_parts)} JARs")

# =============================================================================
# PASSO 4: Verificar repositorio
# =============================================================================
print("\n--- PASSO 4: Verificar repositorio ---")

CDCMS_REPO = CDCMS_SRC_DIR / 'CDCMS.CIL'
SRC_DIR = CDCMS_REPO / 'Implementation' / 'moa' / 'src' / 'main' / 'java'

if SRC_DIR.exists():
    all_java = list(SRC_DIR.rglob("*.java"))
    print(f"[OK] Arquivos Java: {len(all_java)}")
else:
    print("[ERRO] Codigo fonte nao encontrado")
    all_java = []

# Corrigir bug se necessario
META_DIR = SRC_DIR / 'moa' / 'classifiers' / 'meta'
problematic_file = META_DIR / 'CDCMS_CIL_GMean_OSUS.java'
correct_file = META_DIR / 'CDCMS_GMean_OSUS.java'

if problematic_file.exists():
    shutil.copy(problematic_file, correct_file)
    problematic_file.unlink()
    print("[OK] Bug corrigido")
    all_java = list(SRC_DIR.rglob("*.java"))

# =============================================================================
# PASSO 5: Compilar
# =============================================================================
print("\n--- PASSO 5: Compilar ---")

if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)
BUILD_DIR.mkdir()

if all_java and MOA_DEPS_JAR.exists() and len(nd4j_jars) > 0:
    # Criar arquivo de fontes
    sources_file = BUILD_DIR / 'sources.txt'
    with open(sources_file, 'w') as f:
        for jf in all_java:
            f.write(str(jf) + '\n')

    print(f"Compilando {len(all_java)} arquivos...")
    print(f"Classpath: {len(classpath_parts)} JARs")

    compile_cmd = [
        "javac",
        "-d", str(BUILD_DIR),
        "-cp", classpath,
        "-source", "1.8",
        "-target", "1.8",
        "-Xlint:none",
        "-encoding", "UTF-8",
        f"@{sources_file}"
    ]

    start = time.time()
    result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=600)
    compile_time = time.time() - start

    print(f"\nTempo: {compile_time:.1f}s")
    print(f"Return code: {result.returncode}")

    class_files = list(BUILD_DIR.rglob("*.class"))
    print(f"Classes geradas: {len(class_files)}")

    if result.returncode == 0 and len(class_files) > 0:
        print("\n[OK] Compilacao bem-sucedida!")

        cdcms_classes = [c for c in class_files if 'CDCMS' in c.name]
        print(f"\nClasses CDCMS: {len(cdcms_classes)}")
        for c in cdcms_classes:
            print(f"  - {c.name}")
    else:
        print("\n[ERRO] Compilacao falhou")

        # Identificar pacotes faltantes
        missing = set()
        for line in result.stderr.split('\n'):
            if 'package' in line and 'does not exist' in line:
                import re
                match = re.search(r'package ([^\s]+) does not exist', line)
                if match:
                    missing.add(match.group(1))

        if missing:
            print(f"\nPacotes ainda faltando:")
            for pkg in sorted(missing):
                print(f"  - {pkg}")
        else:
            # Mostrar outros erros
            errors = [l for l in result.stderr.split('\n') if 'error:' in l][:10]
            for e in errors:
                print(f"  {e[:100]}")
else:
    print("[ERRO] Pre-requisitos nao atendidos")
    class_files = []

# =============================================================================
# PASSO 6: Criar JAR
# =============================================================================
print("\n--- PASSO 6: Criar JAR ---")

CDCMS_JAR = CDCMS_JARS_DIR / 'cdcms_cil_final.jar'

if class_files:
    print(f"Empacotando {len(class_files)} classes...")

    jar_cmd = ["jar", "cf", str(CDCMS_JAR), "-C", str(BUILD_DIR), "."]
    subprocess.run(jar_cmd, capture_output=True, timeout=120)

    if CDCMS_JAR.exists():
        print(f"[OK] JAR criado: {CDCMS_JAR.name}")
        print(f"     Tamanho: {CDCMS_JAR.stat().st_size/(1024*1024):.1f} MB")
else:
    print("[SKIP] Nenhuma classe para empacotar")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO")
print("="*70)

if CDCMS_JAR.exists() and CDCMS_JAR.stat().st_size > 100000:
    print(f"\n*** COMPILACAO BEM-SUCEDIDA! ***")
    print(f"\nArquivos criados:")
    print(f"  JAR: {CDCMS_JAR}")
    print(f"\nPara executar, use classpath:")
    print(f"  {CDCMS_JAR}:{MOA_DEPS_JAR}")
else:
    print("\n[FALHA] JAR nao foi criado")
