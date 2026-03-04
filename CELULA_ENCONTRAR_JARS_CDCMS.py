# =============================================================================
# ENCONTRAR: JARs e dependencias no repositorio CDCMS.CIL
# =============================================================================
# Verificar se o repositorio tem:
# - JARs pre-compilados
# - Diretorio lib/ com dependencias
# - Instrucoes de build especificas
# =============================================================================

import subprocess
from pathlib import Path
import shutil

print("="*70)
print("ENCONTRAR JARs NO REPOSITORIO CDCMS.CIL")
print("="*70)

WORK_DIR = Path('/content')
CDCMS_REPO = WORK_DIR / 'CDCMS_CIL_src' / 'CDCMS.CIL'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'

CDCMS_JARS_DIR.mkdir(exist_ok=True)

# =============================================================================
# PASSO 1: Listar TODA a estrutura do repositorio
# =============================================================================
print("\n--- PASSO 1: Estrutura completa do repositorio ---")

if CDCMS_REPO.exists():
    print(f"Repositorio: {CDCMS_REPO}\n")

    # Listar recursivamente
    all_files = []
    all_dirs = []

    for item in CDCMS_REPO.rglob("*"):
        if item.is_file():
            all_files.append(item)
        elif item.is_dir() and '.git' not in str(item):
            all_dirs.append(item)

    print(f"Total de arquivos: {len(all_files)}")
    print(f"Total de diretorios: {len(all_dirs)}")

    # Mostrar diretorios principais
    print("\nDiretorios (excluindo .git):")
    for d in sorted(all_dirs)[:30]:
        rel = d.relative_to(CDCMS_REPO)
        if len(rel.parts) <= 3:  # Apenas primeiros 3 niveis
            print(f"  {rel}/")
else:
    print("[ERRO] Repositorio nao encontrado")
    all_files = []

# =============================================================================
# PASSO 2: Procurar JARs
# =============================================================================
print("\n--- PASSO 2: Procurar arquivos JAR ---")

jar_files = [f for f in all_files if f.suffix == '.jar']
print(f"Arquivos .jar encontrados: {len(jar_files)}")

for jar in jar_files:
    rel = jar.relative_to(CDCMS_REPO)
    size = jar.stat().st_size
    print(f"  {rel} ({size/1024:.1f} KB)")

    # Copiar para cdcms_jars
    dest = CDCMS_JARS_DIR / jar.name
    shutil.copy(jar, dest)
    print(f"    -> Copiado para {dest.name}")

# =============================================================================
# PASSO 3: Procurar diretorios lib/
# =============================================================================
print("\n--- PASSO 3: Procurar diretorios lib/ ---")

lib_dirs = [d for d in all_dirs if d.name == 'lib']
print(f"Diretorios 'lib' encontrados: {len(lib_dirs)}")

for lib_dir in lib_dirs:
    rel = lib_dir.relative_to(CDCMS_REPO)
    print(f"\n  {rel}/")

    # Listar conteudo
    for item in lib_dir.iterdir():
        if item.is_file():
            print(f"    {item.name} ({item.stat().st_size/1024:.1f} KB)")

            # Copiar JARs
            if item.suffix == '.jar':
                dest = CDCMS_JARS_DIR / item.name
                shutil.copy(item, dest)

# =============================================================================
# PASSO 4: Procurar arquivos de configuracao/build
# =============================================================================
print("\n--- PASSO 4: Arquivos de build ---")

build_files = ['pom.xml', 'build.gradle', 'build.xml', 'Makefile', '.classpath', 'MANIFEST.MF']

for bf in build_files:
    found = [f for f in all_files if f.name == bf]
    if found:
        print(f"\n{bf}:")
        for f in found:
            rel = f.relative_to(CDCMS_REPO)
            print(f"  {rel}")

            # Ler conteudo se for pom.xml
            if bf == 'pom.xml':
                with open(f) as file:
                    content = file.read()

                # Procurar dependencias
                import re
                deps = re.findall(r'<dependency>.*?</dependency>', content, re.DOTALL)
                if deps:
                    print(f"    Dependencias ({len(deps)}):")
                    for dep in deps[:5]:
                        artifact = re.search(r'<artifactId>([^<]+)', dep)
                        version = re.search(r'<version>([^<]+)', dep)
                        if artifact:
                            v = version.group(1) if version else "?"
                            print(f"      - {artifact.group(1)}: {v}")

# =============================================================================
# PASSO 5: Verificar se existe target/ com JARs compilados
# =============================================================================
print("\n--- PASSO 5: Procurar diretorios target/ ---")

target_dirs = [d for d in all_dirs if d.name == 'target']
print(f"Diretorios 'target' encontrados: {len(target_dirs)}")

for target_dir in target_dirs:
    rel = target_dir.relative_to(CDCMS_REPO)
    print(f"\n  {rel}/")

    jars = list(target_dir.glob("*.jar"))
    for jar in jars:
        print(f"    {jar.name} ({jar.stat().st_size/1024:.1f} KB)")

        # Copiar
        dest = CDCMS_JARS_DIR / jar.name
        shutil.copy(jar, dest)

# =============================================================================
# PASSO 6: Ler README para instrucoes
# =============================================================================
print("\n--- PASSO 6: Instrucoes do README ---")

readme_files = [f for f in all_files if f.name.lower() in ['readme.md', 'readme.txt', 'readme']]

for readme in readme_files:
    rel = readme.relative_to(CDCMS_REPO)
    print(f"\n{rel}:")

    with open(readme) as f:
        content = f.read()

    # Procurar secoes relevantes
    lines = content.split('\n')
    in_relevant_section = False

    for i, line in enumerate(lines):
        lower_line = line.lower()
        if any(kw in lower_line for kw in ['build', 'compile', 'install', 'run', 'usage', 'how to']):
            in_relevant_section = True
            print(f"\n  [Linha {i+1}] {line}")
        elif in_relevant_section and line.strip():
            print(f"  {line[:80]}")
            if i > 0 and not lines[i-1].strip():  # Secao terminou
                in_relevant_section = False

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO")
print("="*70)

print(f"\nJARs copiados para {CDCMS_JARS_DIR}/:")
for jar in CDCMS_JARS_DIR.glob("*.jar"):
    print(f"  {jar.name} ({jar.stat().st_size/1024:.1f} KB)")

if not list(CDCMS_JARS_DIR.glob("*.jar")):
    print("  (nenhum JAR encontrado)")
    print("\n[NOTA] O repositorio CDCMS.CIL aparentemente nao inclui JARs pre-compilados")
    print("Sera necessario compilar do fonte ou contatar os autores")
