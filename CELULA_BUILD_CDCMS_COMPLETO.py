# =============================================================================
# BUILD COMPLETO: CDCMS.CIL (Fork do MOA)
# =============================================================================
# DESCOBERTA: O CDCMS.CIL e um FORK do MOA, nao um plugin!
# Por isso tem 782 arquivos Java - e o MOA inteiro modificado.
# Precisamos buildar o projeto completo usando Maven.
# =============================================================================

import subprocess
from pathlib import Path
import shutil
import time
import urllib.request

print("="*70)
print("BUILD COMPLETO: CDCMS.CIL (Fork do MOA)")
print("="*70)

WORK_DIR = Path('/content')
CDCMS_SRC_DIR = WORK_DIR / 'CDCMS_CIL_src'
CDCMS_REPO = CDCMS_SRC_DIR / 'CDCMS.CIL'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

CDCMS_JARS_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

# =============================================================================
# PASSO 1: Verificar Maven
# =============================================================================
print("\n--- PASSO 1: Verificar Maven ---")

result = subprocess.run(["mvn", "--version"], capture_output=True, text=True)
if result.returncode != 0:
    print("Instalando Maven...")
    subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "maven"], capture_output=True)

result = subprocess.run(["mvn", "--version"], capture_output=True, text=True)
if result.returncode == 0:
    print("[OK] Maven disponivel")
else:
    print("[ERRO] Maven nao disponivel")

# =============================================================================
# PASSO 2: Clonar repositorio (clone limpo)
# =============================================================================
print("\n--- PASSO 2: Clonar repositorio ---")

if CDCMS_REPO.exists():
    print(f"[OK] Repositorio existe: {CDCMS_REPO}")
else:
    CDCMS_SRC_DIR.mkdir(exist_ok=True)
    print("Clonando...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/michaelchiucw/CDCMS.CIL.git"],
        cwd=str(CDCMS_SRC_DIR),
        capture_output=True, text=True, timeout=120
    )
    if CDCMS_REPO.exists():
        print("[OK] Clonado")

# =============================================================================
# PASSO 3: Analisar estrutura do projeto Maven
# =============================================================================
print("\n--- PASSO 3: Estrutura Maven ---")

impl_dir = CDCMS_REPO / 'Implementation'
moa_dir = impl_dir / 'moa'

if impl_dir.exists():
    print(f"Implementation/: {impl_dir}")

    # Ler pom.xml principal
    pom_path = impl_dir / 'pom.xml'
    if pom_path.exists():
        with open(pom_path) as f:
            pom_content = f.read()

        print("\npom.xml principal:")
        # Procurar modules
        import re
        modules = re.findall(r'<module>([^<]+)</module>', pom_content)
        print(f"  Modulos: {modules}")

        # Procurar packaging
        packaging = re.search(r'<packaging>([^<]+)</packaging>', pom_content)
        if packaging:
            print(f"  Packaging: {packaging.group(1)}")

# =============================================================================
# PASSO 4: Build com Maven (apenas modulo moa)
# =============================================================================
print("\n--- PASSO 4: Build do modulo MOA ---")

if moa_dir.exists():
    print(f"Diretorio: {moa_dir}")

    # Verificar pom.xml do modulo moa
    moa_pom = moa_dir / 'pom.xml'
    if moa_pom.exists():
        with open(moa_pom) as f:
            moa_pom_content = f.read()

        # Verificar se tem parent
        has_parent = '<parent>' in moa_pom_content
        print(f"  Tem parent: {has_parent}")

        if has_parent:
            print("\n  [NOTA] Modulo depende do parent, buildando projeto inteiro...")
            build_dir = impl_dir
        else:
            build_dir = moa_dir

        # Executar Maven
        print(f"\n  Executando: mvn clean package -DskipTests -pl moa -am")
        print("  (Isso pode demorar varios minutos...)\n")

        start = time.time()

        # Tentar buildar apenas o modulo moa
        result = subprocess.run(
            ["mvn", "clean", "package", "-DskipTests", "-pl", "moa", "-am", "-q"],
            cwd=str(impl_dir),
            capture_output=True, text=True,
            timeout=900  # 15 minutos
        )

        build_time = time.time() - start
        print(f"  Tempo: {build_time:.1f}s")
        print(f"  Return code: {result.returncode}")

        if result.returncode == 0:
            print("  [OK] Build bem-sucedido!")
        else:
            print("  [ERRO] Build falhou")

            # Mostrar erro
            output = result.stdout + result.stderr
            lines = output.split('\n')

            # Procurar mensagens de erro
            for i, line in enumerate(lines):
                if '[ERROR]' in line:
                    print(f"    {line[:80]}")

            # Se falhou por dependencia ciclica, tentar build diferente
            if "cyclic" in output.lower():
                print("\n  Tentando build alternativo (install primeiro)...")

                result2 = subprocess.run(
                    ["mvn", "install", "-DskipTests", "-N", "-q"],
                    cwd=str(impl_dir),
                    capture_output=True, text=True, timeout=300
                )

                if result2.returncode == 0:
                    result3 = subprocess.run(
                        ["mvn", "package", "-DskipTests", "-q"],
                        cwd=str(moa_dir),
                        capture_output=True, text=True, timeout=600
                    )
                    if result3.returncode == 0:
                        print("  [OK] Build alternativo funcionou!")
                        result = result3

        # Procurar JARs gerados
        target_dir = moa_dir / 'target'
        if target_dir.exists():
            print(f"\n  JARs em target/:")
            for jar in target_dir.glob("*.jar"):
                print(f"    {jar.name} ({jar.stat().st_size/(1024*1024):.1f} MB)")
                # Copiar
                dest = CDCMS_JARS_DIR / jar.name
                shutil.copy(jar, dest)
                print(f"      -> Copiado")

# =============================================================================
# PASSO 5: Se Maven falhou, compilar manualmente
# =============================================================================
print("\n--- PASSO 5: Verificar resultado ---")

jars = list(CDCMS_JARS_DIR.glob("*.jar"))
cdcms_jar = None

for jar in jars:
    if jar.stat().st_size > 100000:  # > 100KB
        print(f"[OK] JAR encontrado: {jar.name} ({jar.stat().st_size/(1024*1024):.1f} MB)")
        cdcms_jar = jar
        break

if not cdcms_jar:
    print("[AVISO] Nenhum JAR gerado pelo Maven")
    print("\nTentando compilacao manual completa...")

    SRC_DIR = moa_dir / 'src' / 'main' / 'java'
    BUILD_DIR = WORK_DIR / 'cdcms_manual_build'

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir()

    # Baixar dependencias necessarias
    deps_dir = WORK_DIR / 'moa_deps'
    deps_dir.mkdir(exist_ok=True)

    deps = [
        ("weka-dev-3.9.2.jar", "https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka/weka-dev/3.9.2/weka-dev-3.9.2.jar"),
        ("sizeofag-1.0.4.jar", "https://repo1.maven.org/maven2/com/github/fracpete/sizeofag/1.0.4/sizeofag-1.0.4.jar"),
    ]

    print("\n  Baixando dependencias...")
    for name, url in deps:
        path = deps_dir / name
        if not path.exists():
            try:
                urllib.request.urlretrieve(url, path)
                print(f"    [OK] {name}")
            except:
                print(f"    [ERRO] {name}")

    # Montar classpath
    classpath = ":".join(str(j) for j in deps_dir.glob("*.jar"))

    if SRC_DIR.exists() and classpath:
        # Listar todos os arquivos Java
        all_java = list(SRC_DIR.rglob("*.java"))
        print(f"\n  Arquivos Java: {len(all_java)}")

        # Criar arquivo de fontes
        sources_file = BUILD_DIR / 'sources.txt'
        with open(sources_file, 'w') as f:
            for jf in all_java:
                f.write(str(jf) + '\n')

        print("  Compilando (pode demorar)...")
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
        print(f"  Tempo: {time.time()-start:.1f}s")

        class_files = list(BUILD_DIR.rglob("*.class"))
        print(f"  Classes: {len(class_files)}")

        if class_files:
            # Criar JAR
            cdcms_jar = CDCMS_JARS_DIR / 'cdcms_cil_manual.jar'
            jar_cmd = ["jar", "cf", str(cdcms_jar), "-C", str(BUILD_DIR), "."]
            subprocess.run(jar_cmd, capture_output=True, timeout=120)

            if cdcms_jar.exists():
                print(f"  [OK] JAR criado: {cdcms_jar.stat().st_size/(1024*1024):.1f} MB")

# =============================================================================
# PASSO 6: Testar
# =============================================================================
print("\n--- PASSO 6: Testar ---")

if cdcms_jar and cdcms_jar.exists():
    # O JAR do CDCMS e self-contained (fork do MOA)
    # Nao precisa de MOA-dependencies.jar separado

    print(f"Testando: {cdcms_jar.name}")

    test_cmd = [
        "java", "-Xmx2g",
        "-cp", str(cdcms_jar),
        "moa.DoTask"
    ]

    result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)

    if result.returncode == 0 or "Usage:" in result.stdout or "Available tasks" in result.stdout:
        print("[OK] moa.DoTask funciona!")

        # Testar classe CDCMS
        print("\nTestando CDCMS_CIL_GMean...")
        test2_cmd = [
            "java", "-Xmx2g",
            "-cp", str(cdcms_jar),
            "moa.DoTask",
            "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean"
        ]

        result2 = subprocess.run(test2_cmd, capture_output=True, text=True, timeout=30)
        output2 = result2.stdout + result2.stderr

        if "Exception" not in output2:
            print("[SUCESSO] CDCMS_CIL_GMean reconhecido!")
            for line in output2.strip().split('\n')[:3]:
                print(f"  {line[:70]}")
        else:
            print("[FALHA]")
            for line in output2.strip().split('\n')[:5]:
                print(f"  {line[:80]}")
    else:
        print("[FALHA] moa.DoTask nao funciona")
        print(result.stderr[:500] if result.stderr else result.stdout[:500])
else:
    print("[SKIP] Nenhum JAR disponivel")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO")
print("="*70)

print(f"\nJARs em {CDCMS_JARS_DIR}/:")
for jar in CDCMS_JARS_DIR.glob("*.jar"):
    print(f"  {jar.name} ({jar.stat().st_size/(1024*1024):.1f} MB)")

if not list(CDCMS_JARS_DIR.glob("*.jar")):
    print("  (nenhum)")
    print("\n[CONCLUSAO] O build do CDCMS.CIL e complexo e requer:")
    print("  1. Todas as dependencias Maven resolvidas")
    print("  2. Build multi-modulo sem referencias ciclicas")
    print("\n[RECOMENDACAO] Contatar autores para JAR pre-compilado:")
    print("  https://github.com/michaelchiucw/CDCMS.CIL/issues")
