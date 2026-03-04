# =============================================================================
# CELULA 3.4: Verificar e baixar dependencias do Maven Central
# =============================================================================

import urllib.request
import ssl
import json
from pathlib import Path

print("="*70)
print("VERIFICAR DEPENDENCIAS NO MAVEN CENTRAL")
print("="*70)

WORK_DIR = Path('/content')
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

def search_maven(query):
    """Busca artefato no Maven Central"""
    url = f"https://search.maven.org/solrsearch/select?q={query}&rows=5&wt=json"
    try:
        with urllib.request.urlopen(url, context=ssl_context, timeout=30) as response:
            data = json.loads(response.read().decode())
            docs = data.get('response', {}).get('docs', [])
            return docs
    except Exception as e:
        print(f"Erro na busca: {e}")
        return []

# Buscar artefatos faltantes
print("\n--- Buscando nd4j-buffer ---")
results = search_maven("a:nd4j-buffer")
for r in results[:3]:
    print(f"  {r.get('g')}:{r.get('a')}:{r.get('latestVersion')}")

print("\n--- Buscando nd4j-context ---")
results = search_maven("a:nd4j-context")
for r in results[:3]:
    print(f"  {r.get('g')}:{r.get('a')}:{r.get('latestVersion')}")

print("\n--- Buscando classlocator ---")
results = search_maven("a:classlocator")
for r in results[:3]:
    print(f"  {r.get('g')}:{r.get('a')}:{r.get('latestVersion')}")

print("\n--- Buscando meka ---")
results = search_maven("g:nz.ac.waikato.cms.meka")
for r in results[:3]:
    print(f"  {r.get('g')}:{r.get('a')}:{r.get('latestVersion')}")

print("\n--- Buscando waikato locator ---")
results = search_maven("nz.ac.waikato.cms.locator")
for r in results[:5]:
    print(f"  {r.get('g')}:{r.get('a')}:{r.get('latestVersion')}")

# Agora tentar URLs alternativas baseadas em versoes mais recentes
print("\n" + "="*70)
print("TENTANDO URLs ALTERNATIVAS")
print("="*70)

alternative_jars = [
    # ND4J versao mais recente que tem buffer/context
    ("nd4j-api-1.0.0-M2.1.jar",
     "https://repo1.maven.org/maven2/org/nd4j/nd4j-api/1.0.0-M2.1/nd4j-api-1.0.0-M2.1.jar"),

    # Locator - tentar weka-stable que tem o locator embutido
    ("weka-stable-3.8.6.jar",
     "https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka/weka-stable/3.8.6/weka-stable-3.8.6.jar"),

    # MEKA versao mais antiga que existe
    ("meka-1.9.1.jar",
     "https://repo1.maven.org/maven2/nz/ac/waikato/cms/meka/meka/1.9.1/meka-1.9.1.jar"),

    # Locator alternativo
    ("jclasslocator-0.0.19.jar",
     "https://repo1.maven.org/maven2/com/github/fracpete/jclasslocator/0.0.19/jclasslocator-0.0.19.jar"),
]

downloaded = 0
failed = 0

for name, url in alternative_jars:
    path = DEPS_DIR / name
    if path.exists() and path.stat().st_size > 1000:
        print(f"[OK] {name} - existe")
    else:
        try:
            print(f"Baixando {name}...", end=" ", flush=True)
            with urllib.request.urlopen(url, context=ssl_context, timeout=120) as response:
                data = response.read()
                with open(path, 'wb') as f:
                    f.write(data)
            if path.exists() and path.stat().st_size > 1000:
                print(f"OK ({path.stat().st_size/1024:.0f} KB)")
                downloaded += 1
            else:
                print("FALHA")
                failed += 1
        except Exception as e:
            print(f"ERRO: {str(e)[:50]}")
            failed += 1

# Resumo
print(f"\n" + "="*50)
jars = list(DEPS_DIR.glob("*.jar"))
print(f"Total de JARs: {len(jars)}")
print(f"Tamanho total: {sum(j.stat().st_size for j in jars)/(1024*1024):.1f} MB")
