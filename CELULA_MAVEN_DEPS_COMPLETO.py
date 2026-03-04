# =============================================================================
# BAIXAR TODAS AS DEPENDENCIAS VIA MAVEN (TRANSITIVAS)
# =============================================================================
# O Maven resolve automaticamente TODAS as dependencias transitivas.
# Isso e necessario porque ND4J tem muitas sub-dependencias.
# =============================================================================

import subprocess
from pathlib import Path
import shutil
import time

print("="*70)
print("BAIXAR DEPENDENCIAS VIA MAVEN (COMPLETO)")
print("="*70)

WORK_DIR = Path('/content')
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'
MAVEN_DIR = WORK_DIR / 'maven_resolver'

# Limpar e recriar
if MAVEN_DIR.exists():
    shutil.rmtree(MAVEN_DIR)
MAVEN_DIR.mkdir()
DEPS_DIR.mkdir(exist_ok=True)

# =============================================================================
# PASSO 1: Criar pom.xml com TODAS as dependencias do CDCMS.CIL
# =============================================================================
print("\n--- PASSO 1: Criar pom.xml ---")

# Baseado no pom.xml do CDCMS.CIL em:
# Implementation/moa/pom.xml

pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>temp</groupId>
    <artifactId>cdcms-deps</artifactId>
    <version>1.0</version>
    <packaging>jar</packaging>

    <properties>
        <maven.compiler.source>1.8</maven.compiler.source>
        <maven.compiler.target>1.8</maven.compiler.target>
    </properties>

    <dependencies>
        <!-- Weka -->
        <dependency>
            <groupId>nz.ac.waikato.cms.weka</groupId>
            <artifactId>weka-dev</artifactId>
            <version>3.9.2</version>
        </dependency>

        <!-- SizeOfAg -->
        <dependency>
            <groupId>com.github.fracpete</groupId>
            <artifactId>sizeofag</artifactId>
            <version>1.0.4</version>
        </dependency>

        <!-- Apache Commons Math -->
        <dependency>
            <groupId>org.apache.commons</groupId>
            <artifactId>commons-math3</artifactId>
            <version>3.6.1</version>
        </dependency>

        <!-- ND4J - o pacote completo com todas dependencias nativas -->
        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>nd4j-native-platform</artifactId>
            <version>1.0.0-beta7</version>
        </dependency>

        <!-- DeepLearning4J core (inclui ND4J) -->
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-core</artifactId>
            <version>1.0.0-beta7</version>
        </dependency>

        <!-- MEKA -->
        <dependency>
            <groupId>net.sf.meka</groupId>
            <artifactId>meka</artifactId>
            <version>1.9.2</version>
        </dependency>

        <!-- JFreeChart -->
        <dependency>
            <groupId>org.jfree</groupId>
            <artifactId>jfreechart</artifactId>
            <version>1.0.19</version>
        </dependency>

    </dependencies>

</project>
'''

pom_path = MAVEN_DIR / 'pom.xml'
with open(pom_path, 'w') as f:
    f.write(pom_content)

print(f"[OK] pom.xml criado: {pom_path}")

# =============================================================================
# PASSO 2: Executar Maven dependency:copy-dependencies
# =============================================================================
print("\n--- PASSO 2: Executar Maven ---")
print("Isso pode demorar varios minutos (baixando ~200+ JARs)...")
print()

start = time.time()

# Executar Maven com output visivel
result = subprocess.run(
    [
        "mvn",
        "dependency:copy-dependencies",
        f"-DoutputDirectory={DEPS_DIR}",
        "-DincludeScope=runtime",
        "-Dhttps.protocols=TLSv1.2"
    ],
    cwd=str(MAVEN_DIR),
    capture_output=True,
    text=True,
    timeout=1800  # 30 minutos
)

duration = time.time() - start
print(f"Tempo: {duration:.0f}s")

# Verificar resultado
if result.returncode == 0:
    print("[OK] Maven executado com sucesso!")
else:
    print("[AVISO] Maven retornou erro")
    # Mostrar erros relevantes
    for line in result.stdout.split('\n'):
        if '[ERROR]' in line or 'BUILD' in line:
            print(f"  {line}")

# =============================================================================
# PASSO 3: Contar JARs baixados
# =============================================================================
print("\n--- PASSO 3: Verificar JARs ---")

jars = list(DEPS_DIR.glob("*.jar"))
total_size = sum(j.stat().st_size for j in jars)

print(f"JARs baixados: {len(jars)}")
print(f"Tamanho total: {total_size/(1024*1024):.1f} MB")

# Verificar pacotes criticos
critical_packages = {
    'nd4j': False,
    'commons-math': False,
    'weka': False,
    'meka': False,
    'jama': False,
}

for jar in jars:
    name = jar.name.lower()
    for pkg in critical_packages:
        if pkg in name:
            critical_packages[pkg] = True

print("\nPacotes criticos:")
for pkg, found in critical_packages.items():
    status = "[OK]" if found else "[X]"
    print(f"  {status} {pkg}")

# =============================================================================
# PASSO 4: Se Maven falhou, mostrar instrucoes alternativas
# =============================================================================
if len(jars) < 50:
    print("\n" + "="*70)
    print("[AVISO] Poucos JARs baixados")
    print("="*70)
    print("\nSe o Maven falhou, execute manualmente no Colab:")
    print()
    print("!mvn dependency:copy-dependencies \\")
    print(f"    -DoutputDirectory={DEPS_DIR} \\")
    print(f"    -f {pom_path}")
    print()
    print("Ou tente reinstalar Maven:")
    print("!apt-get install -y maven")
else:
    print("\n" + "="*70)
    print("[OK] Dependencias baixadas com sucesso!")
    print("="*70)
    print(f"\nExecute agora: CELULA_4_1_COMPILAR_V2.py")
