# =============================================================================
# CELULA 4.1: Compilar codigo fonte
# =============================================================================

import subprocess
import time
import shutil
from pathlib import Path

print("="*70)
print("COMPILAR CDCMS.CIL")
print("="*70)

# Definir diretorios (caso nao estejam definidos)
WORK_DIR = Path('/content')
CDCMS_SRC_DIR = WORK_DIR / 'CDCMS_CIL_src'
CDCMS_REPO = CDCMS_SRC_DIR / 'CDCMS.CIL'
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'
BUILD_DIR = WORK_DIR / 'cdcms_build'

# Limpar build anterior
if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)
BUILD_DIR.mkdir()

# Montar classpath (sem moa*.jar externo)
all_jars = list(DEPS_DIR.glob("*.jar"))
non_moa_jars = [j for j in all_jars if 'moa-' not in j.name.lower()]
classpath = ":".join(str(j) for j in non_moa_jars)

print(f"JARs no classpath: {len(non_moa_jars)}")
for j in non_moa_jars:
    print(f"  - {j.name}")

# Listar arquivos Java
SRC_DIR = CDCMS_REPO / 'Implementation' / 'moa' / 'src' / 'main' / 'java'
all_java = list(SRC_DIR.rglob("*.java"))
print(f"\nArquivos Java: {len(all_java)}")

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
    print(f"\nClasses CDCMS: {len(cdcms_classes)}")
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
            # Extrair nome do pacote
            parts = line.split('package ')
            if len(parts) > 1:
                pkg = parts[1].split(' ')[0]
                missing.add(pkg)

    for pkg in sorted(missing)[:15]:
        print(f"  - {pkg}")
