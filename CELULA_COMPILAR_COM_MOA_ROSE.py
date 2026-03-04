# =============================================================================
# COMPILAR CDCMS.CIL USANDO MOA-dependencies.jar DO ROSE
# =============================================================================
# O MOA-dependencies.jar e um UBER JAR que contem TUDO:
# - MOA, Weka, nz.ac.waikato.cms.locator, etc.
# Este e o UNICO JAR necessario para compilar!
# =============================================================================

import subprocess
from pathlib import Path
import shutil
import time
import urllib.request

print("="*70)
print("COMPILAR CDCMS.CIL COM MOA-dependencies.jar")
print("="*70)

WORK_DIR = Path('/content')
ROSE_JARS_DIR = WORK_DIR / 'rose_jars'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
CDCMS_SRC_DIR = WORK_DIR / 'CDCMS_CIL_src'
BUILD_DIR = WORK_DIR / 'cdcms_build'

for d in [ROSE_JARS_DIR, CDCMS_JARS_DIR, BUILD_DIR]:
    d.mkdir(exist_ok=True)

# =============================================================================
# PASSO 1: Baixar MOA-dependencies.jar do ROSE
# =============================================================================
print("\n--- PASSO 1: Baixar MOA-dependencies.jar ---")

MOA_DEPS_JAR = ROSE_JARS_DIR / 'MOA-dependencies.jar'
SIZEOFAG_JAR = ROSE_JARS_DIR / 'sizeofag-1.0.4.jar'

for jar_name, jar_path, min_size in [
    ("MOA-dependencies.jar", MOA_DEPS_JAR, 50*1024*1024),  # 50MB
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

# Verificar que MOA-dependencies.jar tem as classes necessarias
if MOA_DEPS_JAR.exists():
    result = subprocess.run(
        f'jar tf "{MOA_DEPS_JAR}" | grep -i "locator/ClassCache"',
        shell=True, capture_output=True, text=True
    )
    if result.stdout:
        print(f"\n[OK] ClassCache encontrado no MOA-dependencies.jar:")
        print(f"  {result.stdout.strip()}")
    else:
        print("\n[AVISO] ClassCache nao encontrado")

# =============================================================================
# PASSO 2: Verificar repositorio CDCMS.CIL
# =============================================================================
print("\n--- PASSO 2: Verificar CDCMS.CIL ---")

CDCMS_REPO = CDCMS_SRC_DIR / 'CDCMS.CIL'
SRC_DIR = CDCMS_REPO / 'Implementation' / 'moa' / 'src' / 'main' / 'java'

if not SRC_DIR.exists():
    print("[ERRO] Codigo fonte nao encontrado!")
    print("Execute primeiro a celula que clona o repositorio")
else:
    all_java = list(SRC_DIR.rglob("*.java"))
    print(f"[OK] Arquivos Java: {len(all_java)}")

# =============================================================================
# PASSO 3: Corrigir bug do arquivo com nome errado
# =============================================================================
print("\n--- PASSO 3: Corrigir bug ---")

META_DIR = SRC_DIR / 'moa' / 'classifiers' / 'meta'
problematic_file = META_DIR / 'CDCMS_CIL_GMean_OSUS.java'
correct_file = META_DIR / 'CDCMS_GMean_OSUS.java'

if problematic_file.exists():
    shutil.copy(problematic_file, correct_file)
    problematic_file.unlink()
    print("[OK] Bug corrigido (arquivo renomeado)")
elif correct_file.exists():
    print("[OK] Bug ja foi corrigido")
else:
    print("[INFO] Arquivos nao encontrados (pode ja estar OK)")

# =============================================================================
# PASSO 4: Compilar usando MOA-dependencies.jar
# =============================================================================
print("\n--- PASSO 4: Compilar ---")

if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)
BUILD_DIR.mkdir()

if MOA_DEPS_JAR.exists() and SRC_DIR.exists():
    all_java = list(SRC_DIR.rglob("*.java"))
    print(f"Arquivos Java: {len(all_java)}")

    # Criar arquivo de fontes
    sources_file = BUILD_DIR / 'sources.txt'
    with open(sources_file, 'w') as f:
        for jf in all_java:
            f.write(str(jf) + '\n')

    # Compilar usando APENAS MOA-dependencies.jar
    print(f"\nCompilando com MOA-dependencies.jar ({MOA_DEPS_JAR.stat().st_size/(1024*1024):.1f} MB)...")

    compile_cmd = [
        "javac",
        "-d", str(BUILD_DIR),
        "-cp", str(MOA_DEPS_JAR),  # APENAS este JAR!
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
        if result.stderr:
            # Mostrar erros
            errors = [l for l in result.stderr.split('\n') if l.strip()][:20]
            for e in errors:
                print(f"  {e[:100]}")
else:
    print("[ERRO] Pre-requisitos nao atendidos")
    class_files = []

# =============================================================================
# PASSO 5: Criar JAR
# =============================================================================
print("\n--- PASSO 5: Criar JAR ---")

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
    print(f"\nJAR: {CDCMS_JAR}")
    print(f"MOA-deps: {MOA_DEPS_JAR}")
    print(f"\nPara usar:")
    print(f"  classpath = '{CDCMS_JAR}:{MOA_DEPS_JAR}'")
else:
    print("\n[FALHA] JAR nao foi criado")
    print("\nVerifique os erros de compilacao acima")
