# =============================================================================
# COMPILAR CDCMS.CIL COM MAVEN
# =============================================================================
# O jeito correto de compilar projetos Java com dependencias e usando Maven
# Este script:
# 1. Instala Maven se necessario
# 2. Encontra pom.xml no repositorio
# 3. Compila com mvn package
# 4. Testa o JAR resultante
# =============================================================================

import subprocess
from pathlib import Path
import shutil
import time

print("="*70)
print("COMPILAR CDCMS.CIL COM MAVEN")
print("="*70)

WORK_DIR = Path('/content')
CDCMS_SRC_DIR = WORK_DIR / 'CDCMS_CIL_src'
CDCMS_REPO_DIR = CDCMS_SRC_DIR / 'CDCMS.CIL'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
ROSE_JARS_DIR = WORK_DIR / 'rose_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

CDCMS_JARS_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

# =============================================================================
# PASSO 1: Verificar/Instalar Maven
# =============================================================================
print("\n--- PASSO 1: Verificar Maven ---")

result = subprocess.run(["mvn", "--version"], capture_output=True, text=True)
if result.returncode == 0:
    print("[OK] Maven instalado")
    for line in result.stdout.strip().split('\n')[:2]:
        print(f"  {line}")
else:
    print("Instalando Maven...")
    subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "maven"], capture_output=True)

    result = subprocess.run(["mvn", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print("[OK] Maven instalado")
    else:
        print("[ERRO] Falha ao instalar Maven")

# =============================================================================
# PASSO 2: Clonar repositorio se necessario
# =============================================================================
print("\n--- PASSO 2: Verificar repositorio ---")

if not CDCMS_REPO_DIR.exists():
    print("Clonando repositorio...")
    CDCMS_SRC_DIR.mkdir(exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/michaelchiucw/CDCMS.CIL.git"],
        cwd=str(CDCMS_SRC_DIR),
        capture_output=True, text=True, timeout=120
    )

if CDCMS_REPO_DIR.exists():
    print(f"[OK] Repositorio: {CDCMS_REPO_DIR}")
else:
    print("[ERRO] Repositorio nao encontrado")

# =============================================================================
# PASSO 3: Encontrar pom.xml
# =============================================================================
print("\n--- PASSO 3: Encontrar pom.xml ---")

pom_files = list(CDCMS_REPO_DIR.rglob("pom.xml"))
print(f"Arquivos pom.xml encontrados: {len(pom_files)}")

for pom in pom_files:
    rel_path = pom.relative_to(CDCMS_REPO_DIR)
    print(f"  {rel_path}")

    # Ler conteudo
    with open(pom) as f:
        content = f.read()

    # Extrair info
    import re

    artifact = re.search(r'<artifactId>([^<]+)</artifactId>', content)
    version = re.search(r'<version>([^<]+)</version>', content)
    moa_dep = re.search(r'<artifactId>moa</artifactId>\s*<version>([^<]+)</version>', content)

    if artifact:
        print(f"    Artifact: {artifact.group(1)}")
    if version:
        print(f"    Version: {version.group(1)}")
    if moa_dep:
        print(f"    MOA dependency: {moa_dep.group(1)}")

# =============================================================================
# PASSO 4: Compilar com Maven
# =============================================================================
print("\n--- PASSO 4: Compilar com Maven ---")

# Procurar pom.xml mais proximo da raiz
main_pom = None
for pom in sorted(pom_files, key=lambda p: len(p.parts)):
    main_pom = pom
    break

if main_pom:
    pom_dir = main_pom.parent
    print(f"Diretorio do projeto: {pom_dir}")

    # Executar mvn package
    print("\nExecutando: mvn clean package -DskipTests")
    print("(Isso pode demorar alguns minutos...)\n")

    start = time.time()

    result = subprocess.run(
        ["mvn", "clean", "package", "-DskipTests", "-q"],
        cwd=str(pom_dir),
        capture_output=True,
        text=True,
        timeout=600
    )

    build_time = time.time() - start
    print(f"Tempo de build: {build_time:.1f}s")
    print(f"Return code: {result.returncode}")

    if result.returncode == 0:
        print("[OK] Build bem-sucedido!")

        # Procurar JARs gerados
        target_dir = pom_dir / 'target'
        if target_dir.exists():
            jars = list(target_dir.glob("*.jar"))
            print(f"\nJARs gerados em target/:")
            for jar in jars:
                print(f"  {jar.name} ({jar.stat().st_size/1024:.1f} KB)")

                # Copiar para cdcms_jars
                dest = CDCMS_JARS_DIR / jar.name
                shutil.copy(jar, dest)
                print(f"    -> Copiado para {dest}")
    else:
        print("[ERRO] Build falhou")
        if result.stdout:
            # Mostrar ultimas linhas do output
            lines = result.stdout.strip().split('\n')
            print("\nOutput (ultimas 20 linhas):")
            for line in lines[-20:]:
                print(f"  {line[:80]}")

        if result.stderr:
            print("\nErros:")
            for line in result.stderr.strip().split('\n')[-10:]:
                print(f"  {line[:80]}")
else:
    print("[ERRO] Nenhum pom.xml encontrado")

    # Tentar abordagem alternativa - criar pom.xml minimo
    print("\n--- Alternativa: Criar pom.xml minimo ---")

    impl_dir = CDCMS_REPO_DIR / 'Implementation' / 'moa'
    if impl_dir.exists():
        minimal_pom = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.cdcms</groupId>
    <artifactId>cdcms-cil</artifactId>
    <version>1.0</version>
    <packaging>jar</packaging>

    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>

    <dependencies>
        <dependency>
            <groupId>nz.ac.waikato.cms.moa</groupId>
            <artifactId>moa</artifactId>
            <version>2020.07.1</version>
        </dependency>
    </dependencies>

    <build>
        <sourceDirectory>src/main/java</sourceDirectory>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-jar-plugin</artifactId>
                <version>3.2.0</version>
            </plugin>
        </plugins>
    </build>
</project>
"""
        pom_path = impl_dir / 'pom.xml'
        with open(pom_path, 'w') as f:
            f.write(minimal_pom)
        print(f"[OK] Criado {pom_path}")

        # Tentar compilar
        print("\nCompilando com pom.xml minimo...")
        result = subprocess.run(
            ["mvn", "clean", "package", "-DskipTests", "-q"],
            cwd=str(impl_dir),
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print("[OK] Build bem-sucedido!")
            target_dir = impl_dir / 'target'
            if target_dir.exists():
                for jar in target_dir.glob("*.jar"):
                    dest = CDCMS_JARS_DIR / jar.name
                    shutil.copy(jar, dest)
                    print(f"  JAR copiado: {dest}")
        else:
            print(f"[FALHA] Return code: {result.returncode}")
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-10:]:
                    print(f"  {line[:80]}")

# =============================================================================
# PASSO 5: Testar JAR resultante
# =============================================================================
print("\n--- PASSO 5: Testar JAR ---")

# Encontrar JAR
CDCMS_JAR = None
for jar in sorted(CDCMS_JARS_DIR.glob("*.jar"), key=lambda p: p.stat().st_size, reverse=True):
    if "cdcms" in jar.name.lower() or "cil" in jar.name.lower():
        CDCMS_JAR = jar
        break

# MOA
MOA_DEPS_JAR = ROSE_JARS_DIR / 'MOA-dependencies.jar'

if CDCMS_JAR and MOA_DEPS_JAR.exists():
    print(f"JAR CDCMS: {CDCMS_JAR.name} ({CDCMS_JAR.stat().st_size/1024:.1f} KB)")
    print(f"MOA: {MOA_DEPS_JAR.name}")

    classpath = f"{CDCMS_JAR}:{MOA_DEPS_JAR}"

    # Teste
    print("\nTeste: WriteCommandLineTemplate")
    test_cmd = [
        "java", "-Xmx2g",
        "-cp", classpath,
        "moa.DoTask",
        "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean"
    ]

    result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr

    if "Exception" not in output and result.returncode == 0:
        print("[SUCESSO] CDCMS.CIL reconhecido!")
        for line in output.strip().split('\n')[:3]:
            print(f"  {line[:70]}")

        # Teste completo
        print("\nTeste completo: EvaluateInterleavedTestThenTrain")

        # Criar arquivo de teste
        test_arff = TEST_DIR / 'test_data.arff'
        test_content = """@relation test
@attribute a1 numeric
@attribute a2 numeric
@attribute class {0,1}
@data
"""
        import random
        random.seed(42)
        for i in range(1000):
            a1 = random.gauss(0, 1)
            a2 = random.gauss(0, 1)
            cls = 0 if random.random() < 0.9 else 1
            test_content += f"{a1:.4f},{a2:.4f},{cls}\n"

        with open(test_arff, 'w') as f:
            f.write(test_content)

        output_file = TEST_DIR / 'cdcms_maven_output.csv'

        exec_cmd = [
            "java", "-Xmx4g",
            "-cp", classpath,
            "moa.DoTask",
            f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff}) -l (moa.classifiers.meta.CDCMS_CIL_GMean) -f 500 -d {output_file}"
        ]

        result = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=180)

        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"[SUCESSO TOTAL] Output: {output_file.stat().st_size} bytes")
        else:
            print("[FALHA] Execucao real falhou")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[:5]:
                    print(f"  {line[:80]}")
    else:
        print("[FALHA] Classe nao reconhecida")
        for line in output.strip().split('\n')[:5]:
            print(f"  {line[:80]}")
else:
    print("[SKIP] JARs nao disponiveis")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO")
print("="*70)

output_file = TEST_DIR / 'cdcms_maven_output.csv'
if output_file.exists() and output_file.stat().st_size > 0:
    print("\n*** SUCESSO! CDCMS.CIL FUNCIONANDO! ***")
else:
    print("\n[STATUS] Verifique os passos acima")

    print("\nJARs em cdcms_jars/:")
    for jar in CDCMS_JARS_DIR.glob("*.jar"):
        print(f"  {jar.name} ({jar.stat().st_size/1024:.1f} KB)")
