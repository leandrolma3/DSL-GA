# =============================================================================
# CELULA 4.1: Compilar codigo fonte (V2)
# =============================================================================

import subprocess
import time
import shutil
from pathlib import Path

print("="*70)
print("COMPILAR CDCMS.CIL")
print("="*70)

# Definir diretorios
WORK_DIR = Path('/content')
CDCMS_SRC_DIR = WORK_DIR / 'CDCMS_CIL_src'
CDCMS_REPO = CDCMS_SRC_DIR / 'CDCMS.CIL'
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'
BUILD_DIR = WORK_DIR / 'cdcms_build'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'

# Limpar build anterior
if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)
BUILD_DIR.mkdir()
CDCMS_JARS_DIR.mkdir(exist_ok=True)

# Montar classpath (sem moa*.jar externo)
all_jars = list(DEPS_DIR.glob("*.jar"))
non_moa_jars = [j for j in all_jars if 'moa-' not in j.name.lower()]
classpath = ":".join(str(j) for j in non_moa_jars)

print(f"JARs no classpath: {len(non_moa_jars)}")

# Listar arquivos Java
SRC_DIR = CDCMS_REPO / 'Implementation' / 'moa' / 'src' / 'main' / 'java'
all_java = list(SRC_DIR.rglob("*.java"))
print(f"Arquivos Java: {len(all_java)}")

# Corrigir bug do arquivo com nome errado (se ainda existir)
META_DIR = SRC_DIR / 'moa' / 'classifiers' / 'meta'
problematic_file = META_DIR / 'CDCMS_CIL_GMean_OSUS.java'
correct_file = META_DIR / 'CDCMS_GMean_OSUS.java'

if problematic_file.exists():
    print("\n[FIX] Corrigindo bug do arquivo com nome errado...")
    shutil.copy(problematic_file, correct_file)
    problematic_file.unlink()
    print("[OK] Bug corrigido")
    # Atualizar lista
    all_java = list(SRC_DIR.rglob("*.java"))

# Criar arquivo de fontes
sources_file = BUILD_DIR / 'sources.txt'
with open(sources_file, 'w') as f:
    for jf in all_java:
        f.write(str(jf) + '\n')

# Compilar
print("\nCompilando (pode demorar)...")

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

# Contar classes
class_files = list(BUILD_DIR.rglob("*.class"))
print(f"Classes geradas: {len(class_files)}")

if result.returncode == 0:
    print("\n[OK] Compilacao bem-sucedida!")

    # Listar classes CDCMS
    cdcms_classes = [c for c in class_files if 'CDCMS' in c.name]
    print(f"Classes CDCMS: {len(cdcms_classes)}")

    # Criar JAR
    print("\nCriando JAR...")
    CDCMS_JAR = CDCMS_JARS_DIR / 'cdcms_cil_final.jar'
    jar_cmd = ["jar", "cf", str(CDCMS_JAR), "-C", str(BUILD_DIR), "."]
    jar_result = subprocess.run(jar_cmd, capture_output=True, text=True)

    if CDCMS_JAR.exists():
        print(f"[OK] JAR criado: {CDCMS_JAR.name}")
        print(f"Tamanho: {CDCMS_JAR.stat().st_size/(1024*1024):.1f} MB")
    else:
        print("[ERRO] Falha ao criar JAR")

else:
    print("\n[ERRO] Compilacao falhou")
    print("\n--- ERROS (primeiros 30) ---")
    error_lines = [l for l in result.stderr.split('\n') if l.strip()]
    for line in error_lines[:30]:
        print(line[:100])

    # Identificar pacotes faltantes
    print("\n--- Pacotes faltantes ---")
    missing = set()
    for line in result.stderr.split('\n'):
        if 'package' in line and 'does not exist' in line:
            parts = line.split('package ')
            if len(parts) > 1:
                pkg = parts[1].split(' ')[0]
                missing.add(pkg)

    for pkg in sorted(missing):
        print(f"  - {pkg}")
