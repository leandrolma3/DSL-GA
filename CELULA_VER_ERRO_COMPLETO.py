# =============================================================================
# VER ERRO COMPLETO DE COMPILACAO
# =============================================================================

import subprocess
from pathlib import Path
import shutil

print("="*70)
print("VER ERRO COMPLETO DE COMPILACAO")
print("="*70)

WORK_DIR = Path('/content')
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'
BUILD_DIR = WORK_DIR / 'cdcms_full_build'

if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)
BUILD_DIR.mkdir()

CDCMS_REPO = WORK_DIR / 'CDCMS_CIL_src' / 'CDCMS.CIL'
SRC_DIR = CDCMS_REPO / 'Implementation' / 'moa' / 'src' / 'main' / 'java'

# Montar classpath
all_jars = list(DEPS_DIR.glob("*.jar"))
classpath = ":".join(str(j) for j in all_jars)

print(f"JARs no classpath: {len(all_jars)}")

# Listar arquivos Java
all_java = list(SRC_DIR.rglob("*.java"))
print(f"Arquivos Java: {len(all_java)}")

# Criar arquivo de fontes
sources_file = BUILD_DIR / 'sources.txt'
with open(sources_file, 'w') as f:
    for jf in all_java:
        f.write(str(jf) + '\n')

# Compilar e capturar TODOS os erros
print("\nCompilando...")

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

result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=600)

print(f"Return code: {result.returncode}")

# Mostrar STDERR completo
print("\n" + "="*70)
print("STDERR COMPLETO:")
print("="*70)
print(result.stderr)

# Contar classes
class_files = list(BUILD_DIR.rglob("*.class"))
print(f"\nClasses geradas: {len(class_files)}")

# Se tiver classes, mostrar quais
if class_files:
    print("\nClasses CDCMS:")
    for c in class_files:
        if 'CDCMS' in c.name:
            print(f"  {c.name}")
