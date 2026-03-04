# =============================================================================
# BAIXAR DEPENDENCIAS FALTANTES ESPECIFICAS
# =============================================================================
# Packages faltando:
# - Jama
# - nz.ac.waikato.cms.locator (jclasslocator)
# - org.nd4j.linalg.* (ND4J)
# =============================================================================

import subprocess
from pathlib import Path
import urllib.request
import shutil
import time

print("="*70)
print("BAIXAR DEPENDENCIAS FALTANTES")
print("="*70)

WORK_DIR = Path('/content')
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'
BUILD_DIR = WORK_DIR / 'cdcms_full_build'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'

DEPS_DIR.mkdir(exist_ok=True)

if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)
BUILD_DIR.mkdir()

# =============================================================================
# PASSO 1: Baixar dependencias faltantes
# =============================================================================
print("\n--- PASSO 1: Baixar dependencias faltantes ---")

missing_deps = [
    # Jama - matrix library (usado por Weka/MOA)
    ("Jama-1.0.3.jar", "https://repo1.maven.org/maven2/gov/nist/math/jama/1.0.3/jama-1.0.3.jar"),

    # jclasslocator - URL correta
    ("jclasslocator-0.0.12.jar", "https://repo1.maven.org/maven2/com/github/fracpete/jclasslocator/0.0.12/jclasslocator-0.0.12.jar"),

    # ND4J core
    ("nd4j-api-1.0.0-beta7.jar", "https://repo1.maven.org/maven2/org/nd4j/nd4j-api/1.0.0-beta7/nd4j-api-1.0.0-beta7.jar"),
    ("nd4j-buffer-1.0.0-beta7.jar", "https://repo1.maven.org/maven2/org/nd4j/nd4j-buffer/1.0.0-beta7/nd4j-buffer-1.0.0-beta7.jar"),
    ("nd4j-context-1.0.0-beta7.jar", "https://repo1.maven.org/maven2/org/nd4j/nd4j-context/1.0.0-beta7/nd4j-context-1.0.0-beta7.jar"),
    ("nd4j-common-1.0.0-beta7.jar", "https://repo1.maven.org/maven2/org/nd4j/nd4j-common/1.0.0-beta7/nd4j-common-1.0.0-beta7.jar"),

    # ND4J native backend
    ("nd4j-native-api-1.0.0-beta7.jar", "https://repo1.maven.org/maven2/org/nd4j/nd4j-native-api/1.0.0-beta7/nd4j-native-api-1.0.0-beta7.jar"),
    ("nd4j-native-1.0.0-beta7.jar", "https://repo1.maven.org/maven2/org/nd4j/nd4j-native/1.0.0-beta7/nd4j-native-1.0.0-beta7.jar"),

    # JavaCPP presets
    ("openblas-0.3.7-1.5.2.jar", "https://repo1.maven.org/maven2/org/bytedeco/openblas/0.3.7-1.5.2/openblas-0.3.7-1.5.2.jar"),
]

for name, url in missing_deps:
    path = DEPS_DIR / name
    if path.exists() and path.stat().st_size > 1000:
        print(f"  [OK] {name} (existe)")
        continue

    try:
        print(f"  Baixando {name}...")
        urllib.request.urlretrieve(url, path)
        if path.exists():
            size = path.stat().st_size
            print(f"       {size/1024:.1f} KB" if size < 1024*1024 else f"       {size/(1024*1024):.1f} MB")
    except Exception as e:
        print(f"  [ERRO] {name}: {e}")

# =============================================================================
# PASSO 2: Copiar JARs do Maven (se existirem)
# =============================================================================
print("\n--- PASSO 2: Copiar JARs do Maven ---")

maven_deps = WORK_DIR / 'temp_maven' / 'deps'
if maven_deps.exists():
    maven_jars = list(maven_deps.glob("*.jar"))
    print(f"JARs disponiveis do Maven: {len(maven_jars)}")

    # Copiar todos para DEPS_DIR
    copied = 0
    for jar in maven_jars:
        dest = DEPS_DIR / jar.name
        if not dest.exists():
            shutil.copy(jar, dest)
            copied += 1

    print(f"Copiados: {copied}")

# =============================================================================
# PASSO 3: Listar todos os JARs disponiveis
# =============================================================================
print("\n--- PASSO 3: JARs disponiveis ---")

all_jars = list(DEPS_DIR.glob("*.jar"))
total_size = sum(j.stat().st_size for j in all_jars)
print(f"Total: {len(all_jars)} JARs, {total_size/(1024*1024):.1f} MB")

# Verificar se temos os pacotes necessarios
required_packages = ['jama', 'nd4j', 'jclasslocator']
for pkg in required_packages:
    found = [j for j in all_jars if pkg.lower() in j.name.lower()]
    if found:
        print(f"  [OK] {pkg}: {[f.name for f in found]}")
    else:
        print(f"  [X] {pkg}: NAO ENCONTRADO")

# =============================================================================
# PASSO 4: Compilar
# =============================================================================
print("\n--- PASSO 4: Compilar ---")

CDCMS_REPO = WORK_DIR / 'CDCMS_CIL_src' / 'CDCMS.CIL'
SRC_DIR = CDCMS_REPO / 'Implementation' / 'moa' / 'src' / 'main' / 'java'

classpath = ":".join(str(j) for j in all_jars)

if SRC_DIR.exists():
    all_java = list(SRC_DIR.rglob("*.java"))
    print(f"Arquivos Java: {len(all_java)}")

    sources_file = BUILD_DIR / 'sources.txt'
    with open(sources_file, 'w') as f:
        for jf in all_java:
            f.write(str(jf) + '\n')

    print(f"Compilando com {len(all_jars)} JARs no classpath...")

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
    print(f"Tempo: {time.time()-start:.1f}s")

    class_files = list(BUILD_DIR.rglob("*.class"))
    print(f"Classes geradas: {len(class_files)}")

    if len(class_files) == 0:
        # Analisar erros restantes
        errors = result.stderr.split('\n')
        error_lines = [e for e in errors if 'error:' in e.lower()]
        print(f"\nErros restantes: {len(error_lines)}")

        # Packages ainda faltando
        import re
        missing = set()
        for e in errors:
            if 'package' in e and 'does not exist' in e:
                pkg = re.search(r'package ([^\s]+) does not exist', e)
                if pkg:
                    missing.add(pkg.group(1))

        if missing:
            print(f"\nPackages ainda faltando ({len(missing)}):")
            for p in sorted(missing):
                print(f"  {p}")

        # Mostrar alguns erros
        print("\nExemplos de erros:")
        for e in error_lines[:10]:
            print(f"  {e[:100]}")
    else:
        # Sucesso!
        cdcms_classes = [c for c in class_files if 'CDCMS' in c.name]
        print(f"\nClasses CDCMS: {len(cdcms_classes)}")

# =============================================================================
# PASSO 5: Criar JAR e testar
# =============================================================================
print("\n--- PASSO 5: Criar JAR ---")

class_files = list(BUILD_DIR.rglob("*.class"))

if class_files:
    cdcms_jar = CDCMS_JARS_DIR / 'cdcms_cil_complete.jar'

    jar_cmd = ["jar", "cf", str(cdcms_jar), "-C", str(BUILD_DIR), "."]
    subprocess.run(jar_cmd, capture_output=True, timeout=120)

    if cdcms_jar.exists():
        print(f"[OK] JAR: {cdcms_jar.stat().st_size/(1024*1024):.1f} MB")

        # Testar
        test_cp = f"{cdcms_jar}:{classpath}"

        test_cmd = [
            "java", "-Xmx2g",
            "-cp", test_cp,
            "moa.DoTask",
            "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean"
        ]

        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr

        if "Exception" not in output and result.returncode == 0:
            print("\n[SUCESSO] CDCMS.CIL FUNCIONA!")
            for line in output.strip().split('\n')[:3]:
                print(f"  {line[:70]}")

            # Teste completo
            print("\nExecutando teste completo...")
            TEST_DIR = WORK_DIR / 'cdcms_test_output'
            TEST_DIR.mkdir(exist_ok=True)

            test_arff = TEST_DIR / 'test.arff'
            with open(test_arff, 'w') as f:
                f.write("@relation test\n@attribute a1 numeric\n@attribute a2 numeric\n@attribute class {0,1}\n@data\n")
                import random
                random.seed(42)
                for _ in range(1000):
                    f.write(f"{random.gauss(0,1):.4f},{random.gauss(0,1):.4f},{0 if random.random()<0.9 else 1}\n")

            output_file = TEST_DIR / 'result.csv'

            exec_cmd = [
                "java", "-Xmx4g", "-cp", test_cp, "moa.DoTask",
                f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff}) -l (moa.classifiers.meta.CDCMS_CIL_GMean) -f 500 -d {output_file}"
            ]

            result = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=300)

            if output_file.exists() and output_file.stat().st_size > 0:
                print(f"[SUCESSO TOTAL] Output: {output_file.stat().st_size} bytes")
            else:
                print("[FALHA] Execucao real")
        else:
            print("\n[FALHA] Teste")
            for line in output.strip().split('\n')[:5]:
                print(f"  {line[:80]}")
else:
    print("[SKIP] Nenhuma classe")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO")
print("="*70)

class_files = list(BUILD_DIR.rglob("*.class"))
print(f"\nClasses compiladas: {len(class_files)}")
print(f"JARs de dependencia: {len(all_jars)}")

if len(class_files) == 0:
    print("\n[STATUS] Compilacao ainda falha")
    print("\nO projeto CDCMS.CIL tem dependencias muito complexas.")
    print("A melhor opcao e contatar os autores.")
