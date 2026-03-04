# =============================================================================
# BAIXAR DEPENDENCIA FALTANTE: nz.ac.waikato.cms.locator
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
MAVEN_DIR = WORK_DIR / 'maven_locator'

MAVEN_DIR.mkdir(exist_ok=True)

# O pacote nz.ac.waikato.cms.locator vem de:
# groupId: nz.ac.waikato.cms
# artifactId: classlocator

# =============================================================================
# METODO 1: Tentar via Maven
# =============================================================================
print("\n--- Metodo 1: Maven ---")

pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>temp</groupId>
    <artifactId>locator</artifactId>
    <version>1.0</version>

    <repositories>
        <repository>
            <id>waikato</id>
            <url>https://maven.cms.waikato.ac.nz/content/groups/public/</url>
        </repository>
    </repositories>

    <dependencies>
        <dependency>
            <groupId>nz.ac.waikato.cms</groupId>
            <artifactId>classlocator</artifactId>
            <version>0.0.14</version>
        </dependency>
    </dependencies>
</project>
'''

pom_path = MAVEN_DIR / 'pom.xml'
with open(pom_path, 'w') as f:
    f.write(pom_content)

result = subprocess.run(
    ["mvn", "dependency:copy-dependencies", f"-DoutputDirectory={DEPS_DIR}"],
    cwd=str(MAVEN_DIR),
    capture_output=True, text=True, timeout=120
)

if result.returncode == 0:
    print("[OK] Maven baixou classlocator")
else:
    print("[AVISO] Maven falhou, tentando download direto...")

# =============================================================================
# METODO 2: Download direto de varias fontes
# =============================================================================
print("\n--- Metodo 2: Download direto ---")

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

urls_to_try = [
    # Waikato Maven repo
    ("classlocator-0.0.14.jar",
     "https://maven.cms.waikato.ac.nz/content/groups/public/nz/ac/waikato/cms/classlocator/0.0.14/classlocator-0.0.14.jar"),

    # Versao mais antiga
    ("classlocator-0.0.12.jar",
     "https://maven.cms.waikato.ac.nz/content/groups/public/nz/ac/waikato/cms/classlocator/0.0.12/classlocator-0.0.12.jar"),

    # Fracpete version (alternativa)
    ("jclasslocator-0.0.19.jar",
     "https://repo1.maven.org/maven2/com/github/fracpete/jclasslocator/0.0.19/jclasslocator-0.0.19.jar"),
]

downloaded = False
for name, url in urls_to_try:
    path = DEPS_DIR / name
    if path.exists() and path.stat().st_size > 1000:
        print(f"[OK] {name} ja existe ({path.stat().st_size/1024:.0f} KB)")
        downloaded = True
        break

    try:
        print(f"Tentando {name}...", end=" ", flush=True)
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
    except Exception as e:
        print(f"falhou: {str(e)[:30]}")

# =============================================================================
# Verificar
# =============================================================================
print("\n--- Verificar ---")

jars = list(DEPS_DIR.glob("*.jar"))
locator_jars = [j for j in jars if 'locator' in j.name.lower()]

if locator_jars:
    print("[OK] JAR do locator encontrado:")
    for j in locator_jars:
        print(f"  - {j.name} ({j.stat().st_size/1024:.0f} KB)")

    # Verificar conteudo
    jar_path = locator_jars[0]
    result = subprocess.run(
        f'jar tf "{jar_path}" | grep -i "locator.*class$" | head -5',
        shell=True, capture_output=True, text=True
    )
    if result.stdout:
        print("\nClasses no JAR:")
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")

    print("\n[OK] Execute novamente CELULA_4_1_COMPILAR_V2.py")
else:
    print("[ERRO] Nao foi possivel baixar o classlocator")
    print("\nTente manualmente:")
    print("!wget -O /content/cdcms_all_deps/classlocator.jar \\")
    print("  'https://maven.cms.waikato.ac.nz/content/groups/public/nz/ac/waikato/cms/classlocator/0.0.14/classlocator-0.0.14.jar'")
