# =============================================================================
# CELULA 3.3: Baixar dependencias que falharam (URLs corrigidas)
# =============================================================================

import urllib.request
import ssl
from pathlib import Path

print("="*70)
print("BAIXAR DEPENDENCIAS FALTANTES")
print("="*70)

WORK_DIR = Path('/content')
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'
DEPS_DIR.mkdir(exist_ok=True)

# Criar contexto SSL
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Dependencias com URLs corrigidas
missing_jars = [
    # ND4J Buffer - nome correto do artifact
    ("nd4j-buffer-1.0.0-beta7.jar",
     "https://repo1.maven.org/maven2/org/nd4j/nd4j-buffer/1.0.0-beta7/nd4j-buffer-1.0.0-beta7.jar"),

    # ND4J Context - nome correto
    ("nd4j-context-1.0.0-beta7.jar",
     "https://repo1.maven.org/maven2/org/nd4j/nd4j-context/1.0.0-beta7/nd4j-context-1.0.0-beta7.jar"),

    # Waikato classlocator - groupId correto
    ("classlocator-0.0.14.jar",
     "https://repo1.maven.org/maven2/nz/ac/waikato/cms/jenern/classlocator/0.0.14/classlocator-0.0.14.jar"),

    # MEKA - versao correta
    ("meka-1.9.3.jar",
     "https://repo1.maven.org/maven2/nz/ac/waikato/cms/meka/meka/1.9.3/meka-1.9.3.jar"),

    # ND4J common (dependencia adicional)
    ("nd4j-common-1.0.0-beta7.jar",
     "https://repo1.maven.org/maven2/org/nd4j/nd4j-common/1.0.0-beta7/nd4j-common-1.0.0-beta7.jar"),

    # JavaCPP Presets (dependencia do ND4J)
    ("openblas-0.3.5-1.5-linux-x86_64.jar",
     "https://repo1.maven.org/maven2/org/bytedeco/openblas/0.3.5-1.5/openblas-0.3.5-1.5-linux-x86_64.jar"),

    # Flatbuffers (usado pelo ND4J)
    ("flatbuffers-java-1.10.0.jar",
     "https://repo1.maven.org/maven2/com/google/flatbuffers/flatbuffers-java/1.10.0/flatbuffers-java-1.10.0.jar"),

    # Protobuf
    ("protobuf-java-3.5.1.jar",
     "https://repo1.maven.org/maven2/com/google/protobuf/protobuf-java/3.5.1/protobuf-java-3.5.1.jar"),
]

downloaded = 0
failed = 0
failed_list = []

for name, url in missing_jars:
    path = DEPS_DIR / name
    if path.exists() and path.stat().st_size > 1000:
        size_kb = path.stat().st_size / 1024
        print(f"[OK] {name} ({size_kb:.0f} KB) - existe")
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
                print("FALHA (vazio)")
                failed += 1
                failed_list.append(name)
        except Exception as e:
            print(f"ERRO: {str(e)[:50]}")
            failed += 1
            failed_list.append(name)

# Resumo
print(f"\n" + "="*50)
print(f"RESUMO")
print(f"="*50)
jars = list(DEPS_DIR.glob("*.jar"))
total_size = sum(j.stat().st_size for j in jars)
print(f"Baixados agora: {downloaded}")
print(f"Falhas: {failed}")
print(f"\nTotal de JARs: {len(jars)}")
print(f"Tamanho total: {total_size/(1024*1024):.1f} MB")

if failed > 0:
    print(f"\nFalharam:")
    for f in failed_list:
        print(f"  - {f}")
else:
    print("\n[OK] Todas as dependencias faltantes baixadas!")
