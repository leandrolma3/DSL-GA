# =============================================================================
# COMPILAR CDCMS.CIL COM TODAS AS DEPENDENCIAS
# =============================================================================
# Baseado no pom.xml, precisamos de:
# - weka-dev: 3.9.2
# - sizeofag: 1.0.4
# - jclasslocator: 0.0.12
# - meka: 1.9.2
# - nd4j-native-platform: 1.0.0-beta7
# - commons-math3, jfreechart, etc.
# =============================================================================

import subprocess
from pathlib import Path
import urllib.request
import shutil
import time

print("="*70)
print("COMPILAR CDCMS.CIL COM TODAS AS DEPENDENCIAS")
print("="*70)

WORK_DIR = Path('/content')
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'
BUILD_DIR = WORK_DIR / 'cdcms_full_build'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

DEPS_DIR.mkdir(exist_ok=True)
CDCMS_JARS_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)
BUILD_DIR.mkdir()

# =============================================================================
# PASSO 1: Baixar TODAS as dependencias do Maven
# =============================================================================
print("\n--- PASSO 1: Baixar dependencias ---")

# Lista completa baseada no pom.xml
dependencies = [
    # Weka
    ("weka-dev-3.9.2.jar", "https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka/weka-dev/3.9.2/weka-dev-3.9.2.jar"),

    # Fracpete tools
    ("sizeofag-1.0.4.jar", "https://repo1.maven.org/maven2/com/github/fracpete/sizeofag/1.0.4/sizeofag-1.0.4.jar"),
    ("jclasslocator-0.0.12.jar", "https://repo1.maven.org/maven2/com/github/fracpete/jclasslocator/0.0.12/jclasslocator-0.0.12.jar"),

    # Meka
    ("meka-1.9.2.jar", "https://repo1.maven.org/maven2/net/sf/meka/meka/1.9.2/meka-1.9.2.jar"),

    # Apache Commons
    ("commons-math3-3.6.1.jar", "https://repo1.maven.org/maven2/org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar"),

    # JFreeChart
    ("jfreechart-1.0.19.jar", "https://repo1.maven.org/maven2/org/jfree/jfreechart/1.0.19/jfreechart-1.0.19.jar"),
    ("jcommon-1.0.23.jar", "https://repo1.maven.org/maven2/org/jfree/jcommon/1.0.23/jcommon-1.0.23.jar"),

    # Javacpp (para ND4J)
    ("javacpp-1.5.2.jar", "https://repo1.maven.org/maven2/org/bytedeco/javacpp/1.5.2/javacpp-1.5.2.jar"),

    # MOA core (versao antiga para referencia de API)
    ("moa-2019.05.0.jar", "https://repo1.maven.org/maven2/nz/ac/waikato/cms/moa/moa/2019.05.0/moa-2019.05.0.jar"),
]

print(f"Baixando {len(dependencies)} dependencias...")

for name, url in dependencies:
    path = DEPS_DIR / name
    if path.exists() and path.stat().st_size > 10000:
        print(f"  [OK] {name} (existe)")
        continue

    try:
        print(f"  Baixando {name}...")
        urllib.request.urlretrieve(url, path)
        if path.exists():
            print(f"       {path.stat().st_size/(1024*1024):.1f} MB")
    except Exception as e:
        print(f"  [ERRO] {name}: {e}")

# Listar JARs baixados
print(f"\nJARs em {DEPS_DIR}/:")
jars = list(DEPS_DIR.glob("*.jar"))
total_size = sum(j.stat().st_size for j in jars)
print(f"  Total: {len(jars)} JARs, {total_size/(1024*1024):.1f} MB")

# =============================================================================
# PASSO 2: Montar classpath completo
# =============================================================================
print("\n--- PASSO 2: Montar classpath ---")

classpath = ":".join(str(j) for j in jars)
print(f"Classpath com {len(jars)} JARs")

# =============================================================================
# PASSO 3: Compilar TODOS os arquivos Java
# =============================================================================
print("\n--- PASSO 3: Compilar ---")

CDCMS_REPO = WORK_DIR / 'CDCMS_CIL_src' / 'CDCMS.CIL'
SRC_DIR = CDCMS_REPO / 'Implementation' / 'moa' / 'src' / 'main' / 'java'

if SRC_DIR.exists():
    all_java = list(SRC_DIR.rglob("*.java"))
    print(f"Arquivos Java: {len(all_java)}")

    # Criar arquivo de fontes
    sources_file = BUILD_DIR / 'sources.txt'
    with open(sources_file, 'w') as f:
        for jf in all_java:
            f.write(str(jf) + '\n')

    print("Compilando...")
    compile_cmd = [
        "javac",
        "-d", str(BUILD_DIR),
        "-cp", classpath,
        "-source", "1.8",
        "-target", "1.8",
        "-Xlint:none",
        "-encoding", "UTF-8",
        "-J-Xmx4g",  # Mais memoria para javac
        f"@{sources_file}"
    ]

    start = time.time()
    result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=600)
    compile_time = time.time() - start

    print(f"Tempo: {compile_time:.1f}s")
    print(f"Return code: {result.returncode}")

    # Contar classes
    class_files = list(BUILD_DIR.rglob("*.class"))
    print(f"Classes geradas: {len(class_files)}")

    if len(class_files) == 0 and result.stderr:
        # Analisar erros
        errors = result.stderr.split('\n')
        error_count = len([e for e in errors if 'error:' in e.lower()])
        print(f"\nErros de compilacao: {error_count}")

        # Agrupar por tipo
        error_types = {}
        missing_packages = set()

        for e in errors:
            if 'package' in e and 'does not exist' in e:
                import re
                pkg = re.search(r'package ([^\s]+) does not exist', e)
                if pkg:
                    missing_packages.add(pkg.group(1))
                error_types['package not found'] = error_types.get('package not found', 0) + 1
            elif 'cannot find symbol' in e:
                error_types['cannot find symbol'] = error_types.get('cannot find symbol', 0) + 1
            elif 'error:' in e.lower():
                error_types['outro'] = error_types.get('outro', 0) + 1

        print("\nTipos de erro:")
        for et, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {et}: {count}")

        if missing_packages:
            print(f"\nPackages faltando ({len(missing_packages)}):")
            for pkg in sorted(missing_packages)[:15]:
                print(f"  {pkg}")

    elif len(class_files) > 0:
        # Verificar classes CDCMS
        cdcms_classes = [c for c in class_files if 'CDCMS' in c.name]
        print(f"\nClasses CDCMS: {len(cdcms_classes)}")
        for c in cdcms_classes[:5]:
            print(f"  {c.name}")

# =============================================================================
# PASSO 4: Criar JAR se houver classes
# =============================================================================
print("\n--- PASSO 4: Criar JAR ---")

class_files = list(BUILD_DIR.rglob("*.class"))

if class_files:
    cdcms_jar = CDCMS_JARS_DIR / 'cdcms_cil_full.jar'

    jar_cmd = ["jar", "cf", str(cdcms_jar), "-C", str(BUILD_DIR), "."]
    subprocess.run(jar_cmd, capture_output=True, timeout=120)

    if cdcms_jar.exists():
        print(f"[OK] JAR criado: {cdcms_jar.stat().st_size/(1024*1024):.1f} MB")

        # Testar
        print("\nTestando...")

        # Classpath: JAR compilado + dependencias
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
            print("[SUCESSO] CDCMS.CIL funciona!")
            for line in output.strip().split('\n')[:3]:
                print(f"  {line[:70]}")
        else:
            print("[FALHA]")
            for line in output.strip().split('\n')[:5]:
                print(f"  {line[:80]}")
else:
    print("[SKIP] Nenhuma classe compilada")

# =============================================================================
# PASSO 5: Alternativa - usar Maven para resolver dependencias
# =============================================================================
print("\n--- PASSO 5: Alternativa Maven ---")

if not class_files:
    print("Tentando resolver dependencias via Maven...")

    # Criar pom.xml temporario para baixar dependencias
    temp_pom_dir = WORK_DIR / 'temp_maven'
    temp_pom_dir.mkdir(exist_ok=True)

    pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>temp</groupId>
    <artifactId>temp</artifactId>
    <version>1.0</version>

    <dependencies>
        <dependency>
            <groupId>nz.ac.waikato.cms.weka</groupId>
            <artifactId>weka-dev</artifactId>
            <version>3.9.2</version>
        </dependency>
        <dependency>
            <groupId>nz.ac.waikato.cms.moa</groupId>
            <artifactId>moa</artifactId>
            <version>2019.05.0</version>
        </dependency>
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-core</artifactId>
            <version>1.0.0-beta7</version>
        </dependency>
    </dependencies>
</project>
"""

    pom_path = temp_pom_dir / 'pom.xml'
    with open(pom_path, 'w') as f:
        f.write(pom_content)

    print("Baixando dependencias via Maven (pode demorar)...")

    result = subprocess.run(
        ["mvn", "dependency:copy-dependencies", "-DoutputDirectory=deps", "-q"],
        cwd=str(temp_pom_dir),
        capture_output=True, text=True,
        timeout=600
    )

    deps_from_maven = temp_pom_dir / 'deps'
    if deps_from_maven.exists():
        maven_jars = list(deps_from_maven.glob("*.jar"))
        print(f"JARs baixados via Maven: {len(maven_jars)}")

        # Copiar para deps dir
        for jar in maven_jars:
            dest = DEPS_DIR / jar.name
            if not dest.exists():
                shutil.copy(jar, dest)

        # Tentar compilar novamente
        print("\nRecompilando com dependencias do Maven...")

        jars = list(DEPS_DIR.glob("*.jar"))
        classpath = ":".join(str(j) for j in jars)

        result = subprocess.run(
            ["javac", "-d", str(BUILD_DIR), "-cp", classpath,
             "-source", "1.8", "-target", "1.8", "-Xlint:none",
             f"@{sources_file}"],
            capture_output=True, text=True, timeout=600
        )

        class_files = list(BUILD_DIR.rglob("*.class"))
        print(f"Classes: {len(class_files)}")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO FINAL")
print("="*70)

class_files = list(BUILD_DIR.rglob("*.class"))
jars_created = list(CDCMS_JARS_DIR.glob("*.jar"))

print(f"\nClasses compiladas: {len(class_files)}")
print(f"JARs criados: {len(jars_created)}")

for jar in jars_created:
    print(f"  {jar.name} ({jar.stat().st_size/(1024*1024):.1f} MB)")

if len(class_files) == 0:
    print("\n[CONCLUSAO] Compilacao falhou - dependencias complexas")
    print("\nO CDCMS.CIL usa bibliotecas de deep learning (ND4J/DL4J)")
    print("que tem dependencias nativas complexas.")
    print("\n[ACAO RECOMENDADA]")
    print("Abrir issue pedindo JAR pre-compilado:")
    print("  https://github.com/michaelchiucw/CDCMS.CIL/issues/new")
    print("\nSugestao de mensagem:")
    print('  "Could you please provide a pre-compiled JAR file?')
    print('   I am trying to run CDCMS_CIL_GMean with MOA but')
    print('   compilation fails due to complex dependencies."')
