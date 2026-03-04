# =============================================================================
# CELULA 1.4 CORRIGIDA: Baixar MOA Framework COMPLETO
# =============================================================================
# Correções:
# 1. Threshold corrigido: 40MB (não 50MB) - ZIP do SourceForge tem ~48MB
# 2. Download do Weka explícito como fallback
# 3. Verificação mais robusta do conteúdo do ZIP
# =============================================================================

print("="*60)
print("BAIXANDO MOA FRAMEWORK (VERSAO CORRIGIDA)")
print("="*60)

import os
import subprocess
from pathlib import Path

# Diretorio MOA
MOA_DIR = WORK_DIR / 'moa'
MOA_DIR.mkdir(exist_ok=True)

MOA_JAR = MOA_DIR / 'moa.jar'
MOA_LIB_DIR = MOA_DIR / 'lib'
WEKA_JAR = MOA_DIR / 'weka.jar'

# =============================================================================
# ESTRATÉGIA 1: Download do MOA Release completo do SourceForge
# =============================================================================
print("\n--- Estratégia 1: MOA Release do SourceForge ---")

moa_zip = MOA_DIR / 'moa-release.zip'

# URLs alternativas para SourceForge (mirrors diferentes)
SF_URLS = [
    "https://downloads.sourceforge.net/project/moa-datastream/MOA/2018%20June/moa-release-2018.6.0-bin.zip",
    "https://netcologne.dl.sourceforge.net/project/moa-datastream/MOA/2018%20June/moa-release-2018.6.0-bin.zip",
    "https://kent.dl.sourceforge.net/project/moa-datastream/MOA/2018%20June/moa-release-2018.6.0-bin.zip",
]

downloaded_release = False

for url in SF_URLS:
    if downloaded_release:
        break

    mirror = url.split('/')[2]
    print(f"\nTentando mirror: {mirror}...")

    # Usar wget com timeout e retry
    result = subprocess.run(
        f'wget -q --timeout=60 --tries=2 -O "{moa_zip}" "{url}"',
        shell=True, capture_output=True, text=True
    )

    if moa_zip.exists():
        size_mb = moa_zip.stat().st_size / (1024*1024)
        print(f"  Tamanho: {size_mb:.1f} MB")

        # CORREÇÃO: Threshold de 40MB (não 50MB)
        # O ZIP real tem aproximadamente 48MB
        if size_mb > 40:
            # Verificar se é um ZIP válido
            check_zip = subprocess.run(
                f'unzip -t "{moa_zip}" 2>&1 | head -5',
                shell=True, capture_output=True, text=True
            )

            if 'No errors' in check_zip.stdout or 'moa.jar' in check_zip.stdout:
                print("  [OK] ZIP válido! Extraindo...")

                # Extrair
                subprocess.run(
                    f'cd "{MOA_DIR}" && unzip -q -o "{moa_zip}"',
                    shell=True
                )

                # Procurar moa.jar
                moa_found = list(MOA_DIR.rglob('moa.jar'))
                if moa_found:
                    # Copiar para local padrão
                    subprocess.run(f'cp "{moa_found[0]}" "{MOA_JAR}"', shell=True)
                    print(f"  [OK] moa.jar copiado")

                    # Procurar e copiar lib/
                    lib_found = list(MOA_DIR.rglob('lib'))
                    for lib_path in lib_found:
                        if lib_path.is_dir() and list(lib_path.glob('*.jar')):
                            MOA_LIB_DIR.mkdir(exist_ok=True)
                            subprocess.run(f'cp -r "{lib_path}"/* "{MOA_LIB_DIR}/"', shell=True)
                            n_jars = len(list(MOA_LIB_DIR.glob('*.jar')))
                            print(f"  [OK] lib/ copiado ({n_jars} JARs)")
                            downloaded_release = True
                            break
                else:
                    print("  [X] moa.jar não encontrado no ZIP")
            else:
                print(f"  [X] ZIP inválido ou corrompido")
        else:
            print(f"  [X] Arquivo pequeno demais (provavelmente página HTML)")

        if not downloaded_release:
            moa_zip.unlink()

if not downloaded_release:
    print("\n[AVISO] Download do release completo falhou.")

# =============================================================================
# ESTRATÉGIA 2: Download dos JARs individuais do Maven Central
# =============================================================================
if not MOA_JAR.exists():
    print("\n--- Estratégia 2: Download do Maven Central ---")

    MAVEN_JARS = [
        ("https://repo1.maven.org/maven2/nz/ac/waikato/cms/moa/moa/2018.6.0/moa-2018.6.0.jar", "moa.jar"),
    ]

    for url, name in MAVEN_JARS:
        dest = MOA_DIR / name
        print(f"  Baixando {name}...")
        subprocess.run(f'curl -L -s -o "{dest}" "{url}"', shell=True)
        if dest.exists():
            print(f"  [OK] {name} ({dest.stat().st_size / (1024*1024):.1f} MB)")

# =============================================================================
# ESTRATÉGIA 3: Baixar Weka explicitamente (CRÍTICO para CDCMS.CIL)
# =============================================================================
if not WEKA_JAR.exists() and not (MOA_LIB_DIR.exists() and list(MOA_LIB_DIR.glob('weka*.jar'))):
    print("\n--- Estratégia 3: Download do Weka (necessário para CDCMS.CIL) ---")

    # Weka stable 3.8.3 (compatível com MOA 2018.6.0)
    WEKA_URL = "https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka/weka-stable/3.8.3/weka-stable-3.8.3.jar"

    print(f"  Baixando weka-stable-3.8.3.jar...")
    subprocess.run(f'curl -L -s -o "{WEKA_JAR}" "{WEKA_URL}"', shell=True)

    if WEKA_JAR.exists():
        size_mb = WEKA_JAR.stat().st_size / (1024*1024)
        print(f"  [OK] weka.jar ({size_mb:.1f} MB)")

        # Verificar se contém classes de clustering
        check = subprocess.run(
            f'jar tf "{WEKA_JAR}" 2>/dev/null | grep "weka/clusterers/EM.class"',
            shell=True, capture_output=True, text=True
        )
        if check.stdout.strip():
            print(f"  [OK] weka.clusterers.EM encontrado!")
        else:
            print(f"  [AVISO] weka.clusterers.EM não encontrado")

# =============================================================================
# VERIFICAÇÃO FINAL
# =============================================================================
print("\n" + "="*60)
print("VERIFICAÇÃO FINAL")
print("="*60)

print(f"\nmoa.jar: {MOA_JAR.exists()}")
if MOA_JAR.exists():
    print(f"  Tamanho: {MOA_JAR.stat().st_size / (1024*1024):.1f} MB")

print(f"\nlib/: {MOA_LIB_DIR.exists()}")
if MOA_LIB_DIR.exists():
    lib_jars = list(MOA_LIB_DIR.glob('*.jar'))
    print(f"  JARs: {len(lib_jars)}")

    # Verificar JARs importantes
    important = ['weka', 'sizeofag', 'javacliparser']
    for name in important:
        found = [j for j in lib_jars if name in j.name.lower()]
        status = "[OK]" if found else "[X]"
        print(f"  {status} {name}: {found[0].name if found else 'não encontrado'}")

print(f"\nweka.jar (backup): {WEKA_JAR.exists()}")
if WEKA_JAR.exists():
    print(f"  Tamanho: {WEKA_JAR.stat().st_size / (1024*1024):.1f} MB")

# Verificar classes críticas para CDCMS.CIL
print("\n--- Verificando classes críticas para CDCMS.CIL ---")

all_jars = [MOA_JAR] if MOA_JAR.exists() else []
if MOA_LIB_DIR.exists():
    all_jars.extend(MOA_LIB_DIR.glob('*.jar'))
if WEKA_JAR.exists():
    all_jars.append(WEKA_JAR)

critical_classes = [
    "weka/clusterers/EM.class",
    "weka/clusterers/SimpleKMeans.class",
    "weka/filters/unsupervised/attribute/Remove.class",
    "moa/tasks/EvaluateInterleavedTestThenTrain.class",
]

for cls in critical_classes:
    found = False
    for jar in all_jars:
        check = subprocess.run(
            f'jar tf "{jar}" 2>/dev/null | grep "{cls}"',
            shell=True, capture_output=True, text=True
        )
        if check.stdout.strip():
            found = True
            break

    status = "[OK]" if found else "[X] FALTANDO"
    class_name = cls.replace('/', '.').replace('.class', '')
    print(f"  {status} {class_name}")

if not MOA_LIB_DIR.exists() and not WEKA_JAR.exists():
    print("\n[ERRO CRÍTICO] Nem lib/ nem weka.jar estão disponíveis!")
    print("O CDCMS.CIL precisa do Weka para clustering.")
    print("Tente executar esta célula novamente ou baixe manualmente.")
