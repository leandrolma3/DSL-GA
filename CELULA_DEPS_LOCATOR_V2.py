# =============================================================================
# BAIXAR DEPENDENCIA FALTANTE: nz.ac.waikato.cms.locator (V2)
# =============================================================================

import subprocess
from pathlib import Path
import urllib.request
import ssl

print("="*70)
print("BAIXAR: nz.ac.waikato.cms.locator")
print("="*70)

WORK_DIR = Path('/content')
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# =============================================================================
# Tentar varias versoes do jclasslocator do fracpete
# =============================================================================
print("\n--- Tentando jclasslocator (fracpete) ---")

# Versoes conhecidas do jclasslocator
versions = ['0.0.18', '0.0.17', '0.0.16', '0.0.15', '0.0.14', '0.0.13', '0.0.12', '0.0.11', '0.0.10']

downloaded = False
for version in versions:
    name = f"jclasslocator-{version}.jar"
    url = f"https://repo1.maven.org/maven2/com/github/fracpete/jclasslocator/{version}/jclasslocator-{version}.jar"
    path = DEPS_DIR / name

    if path.exists() and path.stat().st_size > 1000:
        print(f"[OK] {name} ja existe")
        downloaded = True
        break

    try:
        print(f"Tentando v{version}...", end=" ", flush=True)
        with urllib.request.urlopen(url, context=ssl_context, timeout=30) as response:
            data = response.read()
            with open(path, 'wb') as f:
                f.write(data)

        if path.exists() and path.stat().st_size > 1000:
            print(f"OK ({path.stat().st_size/1024:.0f} KB)")
            downloaded = True
            break
        else:
            print("vazio")
            path.unlink(missing_ok=True)
    except urllib.error.HTTPError as e:
        print(f"404")
    except Exception as e:
        print(f"erro")

# =============================================================================
# Se nao encontrou, tentar classlocator de waikato via Maven Central
# =============================================================================
if not downloaded:
    print("\n--- Tentando classlocator (waikato) ---")

    waikato_versions = ['0.0.15', '0.0.14', '0.0.13', '0.0.12']

    for version in waikato_versions:
        name = f"classlocator-{version}.jar"
        # Tentar no Maven Central (algumas versoes estao la)
        url = f"https://repo1.maven.org/maven2/nz/ac/waikato/cms/classlocator/{version}/classlocator-{version}.jar"
        path = DEPS_DIR / name

        try:
            print(f"Tentando classlocator v{version}...", end=" ", flush=True)
            with urllib.request.urlopen(url, context=ssl_context, timeout=30) as response:
                data = response.read()
                with open(path, 'wb') as f:
                    f.write(data)

            if path.exists() and path.stat().st_size > 1000:
                print(f"OK ({path.stat().st_size/1024:.0f} KB)")
                downloaded = True
                break
            else:
                print("vazio")
                path.unlink(missing_ok=True)
        except:
            print("falhou")

# =============================================================================
# Se ainda nao encontrou, extrair do weka-dev que ja temos
# =============================================================================
if not downloaded:
    print("\n--- Verificando se weka-dev contem locator ---")

    weka_jar = DEPS_DIR / 'weka-dev-3.9.2.jar'
    if weka_jar.exists():
        result = subprocess.run(
            f'jar tf "{weka_jar}" | grep -i "locator"',
            shell=True, capture_output=True, text=True
        )
        if result.stdout:
            print("Classes locator encontradas no weka-dev:")
            for line in result.stdout.strip().split('\n')[:10]:
                print(f"  {line}")
            print("\n[INFO] O weka-dev pode ja conter as classes necessarias")
            downloaded = True

# =============================================================================
# Verificar resultado
# =============================================================================
print("\n" + "="*50)
print("RESULTADO")
print("="*50)

locator_jars = [j for j in DEPS_DIR.glob("*.jar") if 'locator' in j.name.lower()]

if locator_jars:
    print("\n[OK] JAR do locator encontrado:")
    for j in locator_jars:
        print(f"  - {j.name}")

        # Verificar classes
        result = subprocess.run(
            f'jar tf "{j}" | grep -E "ClassCache|locator" | head -5',
            shell=True, capture_output=True, text=True
        )
        if result.stdout:
            print("  Classes:")
            for line in result.stdout.strip().split('\n'):
                print(f"    {line}")

    print("\n[OK] Execute: CELULA_4_1_COMPILAR_V2.py")

elif downloaded:
    print("\n[INFO] Locator pode estar embutido no weka-dev")
    print("Execute: CELULA_4_1_COMPILAR_V2.py para verificar")

else:
    print("\n[ERRO] Nao foi possivel baixar")
    print("\nOpcao alternativa - usar MOA oficial que ja tem tudo:")
    print("!pip install moa-jar")
