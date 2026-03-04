# =============================================================================
# CELULA 3.2: Baixar TODAS as dependencias necessarias
# =============================================================================

import urllib.request
import ssl
from pathlib import Path

print("="*70)
print("BAIXAR TODAS AS DEPENDENCIAS")
print("="*70)

WORK_DIR = Path('/content')
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'
DEPS_DIR.mkdir(exist_ok=True)

# Criar contexto SSL
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Lista COMPLETA de dependencias
all_jars = [
    # === WEKA ===
    ("weka-dev-3.9.2.jar",
     "https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka/weka-dev/3.9.2/weka-dev-3.9.2.jar"),

    # === ND4J (Numerical Computing) ===
    ("nd4j-api-1.0.0-beta7.jar",
     "https://repo1.maven.org/maven2/org/nd4j/nd4j-api/1.0.0-beta7/nd4j-api-1.0.0-beta7.jar"),
    ("nd4j-native-api-1.0.0-beta7.jar",
     "https://repo1.maven.org/maven2/org/nd4j/nd4j-native-api/1.0.0-beta7/nd4j-native-api-1.0.0-beta7.jar"),
    ("nd4j-native-1.0.0-beta7.jar",
     "https://repo1.maven.org/maven2/org/nd4j/nd4j-native/1.0.0-beta7/nd4j-native-1.0.0-beta7.jar"),
    ("nd4j-buffer-1.0.0-beta7.jar",
     "https://repo1.maven.org/maven2/org/nd4j/nd4j-buffer/1.0.0-beta7/nd4j-buffer-1.0.0-beta7.jar"),
    ("nd4j-context-1.0.0-beta7.jar",
     "https://repo1.maven.org/maven2/org/nd4j/nd4j-context/1.0.0-beta7/nd4j-context-1.0.0-beta7.jar"),

    # === Apache Commons Math ===
    ("commons-math3-3.6.1.jar",
     "https://repo1.maven.org/maven2/org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar"),

    # === Waikato Locator ===
    ("classlocator-0.0.12.jar",
     "https://repo1.maven.org/maven2/nz/ac/waikato/cms/jenern/classlocator/0.0.12/classlocator-0.0.12.jar"),

    # === MEKA ===
    ("meka-1.9.2.jar",
     "https://repo1.maven.org/maven2/nz/ac/waikato/cms/meka/meka/1.9.2/meka-1.9.2.jar"),

    # === Outras dependencias ===
    ("sizeofag-1.0.4.jar",
     "https://repo1.maven.org/maven2/com/github/fracpete/sizeofag/1.0.4/sizeofag-1.0.4.jar"),
    ("jama-1.0.3.jar",
     "https://repo1.maven.org/maven2/gov/nist/math/jama/1.0.3/jama-1.0.3.jar"),
    ("bcprov-jdk15on-1.59.jar",
     "https://repo1.maven.org/maven2/org/bouncycastle/bcprov-jdk15on/1.59/bcprov-jdk15on-1.59.jar"),
    ("commons-compress-1.16.1.jar",
     "https://repo1.maven.org/maven2/org/apache/commons/commons-compress/1.16.1/commons-compress-1.16.1.jar"),
    ("mtj-1.0.4.jar",
     "https://repo1.maven.org/maven2/com/googlecode/matrix-toolkits-java/mtj/1.0.4/mtj-1.0.4.jar"),
    ("netlib-core-1.1.2.jar",
     "https://repo1.maven.org/maven2/com/github/fommil/netlib/core/1.1.2/core-1.1.2.jar"),
    ("arpack_combined_all-0.1.jar",
     "https://repo1.maven.org/maven2/net/sourceforge/f2j/arpack_combined_all/0.1/arpack_combined_all-0.1.jar"),

    # === Dependencias do ND4J ===
    ("guava-25.1-jre.jar",
     "https://repo1.maven.org/maven2/com/google/guava/guava/25.1-jre/guava-25.1-jre.jar"),
    ("javacpp-1.5.jar",
     "https://repo1.maven.org/maven2/org/bytedeco/javacpp/1.5/javacpp-1.5.jar"),
    ("jackson-databind-2.9.9.jar",
     "https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-databind/2.9.9/jackson-databind-2.9.9.jar"),
    ("jackson-core-2.9.9.jar",
     "https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-core/2.9.9/jackson-core-2.9.9.jar"),
    ("jackson-annotations-2.9.9.jar",
     "https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-annotations/2.9.9/jackson-annotations-2.9.9.jar"),
    ("slf4j-api-1.7.26.jar",
     "https://repo1.maven.org/maven2/org/slf4j/slf4j-api/1.7.26/slf4j-api-1.7.26.jar"),
    ("slf4j-simple-1.7.26.jar",
     "https://repo1.maven.org/maven2/org/slf4j/slf4j-simple/1.7.26/slf4j-simple-1.7.26.jar"),
]

downloaded = 0
failed = 0
skipped = 0

for name, url in all_jars:
    path = DEPS_DIR / name
    if path.exists() and path.stat().st_size > 1000:
        size_kb = path.stat().st_size / 1024
        print(f"[OK] {name} ({size_kb:.0f} KB) - existe")
        skipped += 1
    else:
        try:
            print(f"Baixando {name}...", end=" ", flush=True)
            with urllib.request.urlopen(url, context=ssl_context, timeout=120) as response:
                data = response.read()
                with open(path, 'wb') as f:
                    f.write(data)

            if path.exists() and path.stat().st_size > 1000:
                size_kb = path.stat().st_size / 1024
                print(f"OK ({size_kb:.0f} KB)")
                downloaded += 1
            else:
                print("FALHA")
                failed += 1
        except Exception as e:
            print(f"ERRO: {str(e)[:40]}")
            failed += 1

# Resumo
print(f"\n" + "="*50)
print(f"RESUMO")
print(f"="*50)
jars = list(DEPS_DIR.glob("*.jar"))
total_size = sum(j.stat().st_size for j in jars)
print(f"Baixados agora: {downloaded}")
print(f"Ja existiam: {skipped}")
print(f"Falhas: {failed}")
print(f"\nTotal de JARs: {len(jars)}")
print(f"Tamanho total: {total_size/(1024*1024):.1f} MB")

if failed == 0:
    print("\n[OK] Todas as dependencias prontas!")
else:
    print(f"\n[AVISO] {failed} dependencias falharam")
