# =============================================================================
# ENCONTRAR MOA CORRETO PARA CDCMS.CIL
# =============================================================================
# O CDCMS.CIL usa MOA 2018.6.1-SNAPSHOT
# Vamos encontrar a versao release correspondente
# =============================================================================

import subprocess
from pathlib import Path
import urllib.request
import time

print("="*70)
print("ENCONTRAR MOA CORRETO PARA CDCMS.CIL")
print("="*70)

WORK_DIR = Path('/content')
MOA_DIR = WORK_DIR / 'moa_versions'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

MOA_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

# =============================================================================
# PASSO 1: Tentar diferentes formatos de versao MOA
# =============================================================================
print("\n--- PASSO 1: Tentar versoes MOA do Maven ---")

# Diferentes formatos de versao que podem existir
moa_versions_to_try = [
    # Formato mais antigo
    ("2018.6.0", "https://repo1.maven.org/maven2/nz/ac/waikato/cms/moa/moa/2018.6.0/moa-2018.6.0.jar"),
    ("2018.6.1", "https://repo1.maven.org/maven2/nz/ac/waikato/cms/moa/moa/2018.6.1/moa-2018.6.1.jar"),
    # Formato com zero
    ("2018.06.0", "https://repo1.maven.org/maven2/nz/ac/waikato/cms/moa/moa/2018.06.0/moa-2018.06.0.jar"),
    # Versoes proximas
    ("2019.05.0", "https://repo1.maven.org/maven2/nz/ac/waikato/cms/moa/moa/2019.05.0/moa-2019.05.0.jar"),
    ("2017.06", "https://repo1.maven.org/maven2/nz/ac/waikato/cms/moa/moa/2017.06/moa-2017.06.jar"),
]

found_moa = None

for version, url in moa_versions_to_try:
    jar_path = MOA_DIR / f'moa-{version}.jar'

    if jar_path.exists() and jar_path.stat().st_size > 1000000:
        print(f"[OK] moa-{version}.jar ja existe ({jar_path.stat().st_size/(1024*1024):.1f} MB)")
        found_moa = jar_path
        break

    print(f"Tentando moa-{version}...")
    try:
        # Timeout curto para verificar se existe
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                print(f"  [EXISTE] Baixando...")
                urllib.request.urlretrieve(url, jar_path)
                if jar_path.exists() and jar_path.stat().st_size > 1000000:
                    print(f"  [OK] {jar_path.stat().st_size/(1024*1024):.1f} MB")
                    found_moa = jar_path
                    break
    except urllib.error.HTTPError as e:
        print(f"  [NAO EXISTE] {e.code}")
    except Exception as e:
        print(f"  [ERRO] {e}")

# =============================================================================
# PASSO 2: Baixar MOA Release do GitHub
# =============================================================================
print("\n--- PASSO 2: Baixar MOA Release do GitHub ---")

if not found_moa:
    # MOA releases no GitHub tem formato diferente
    # https://github.com/Waikato/moa/releases
    github_releases = [
        ("2019.05.0", "https://github.com/Waikato/moa/releases/download/2019.05.0/moa-release-2019.05.0.jar"),
        ("2018.06.0", "https://github.com/Waikato/moa/releases/download/2018.06.0/moa-release-2018.06.0.jar"),
    ]

    for version, url in github_releases:
        jar_path = MOA_DIR / f'moa-release-{version}.jar'

        if jar_path.exists() and jar_path.stat().st_size > 10000000:
            print(f"[OK] moa-release-{version}.jar ja existe")
            found_moa = jar_path
            break

        print(f"Tentando GitHub release {version}...")
        try:
            # Usar wget que segue redirects melhor
            result = subprocess.run(
                ["wget", "-q", "--timeout=30", "-O", str(jar_path), url],
                capture_output=True, timeout=120
            )
            if jar_path.exists() and jar_path.stat().st_size > 10000000:
                print(f"  [OK] {jar_path.stat().st_size/(1024*1024):.1f} MB")
                found_moa = jar_path
                break
            else:
                if jar_path.exists():
                    jar_path.unlink()  # Remover arquivo incompleto
                print(f"  [FALHA]")
        except Exception as e:
            print(f"  [ERRO] {e}")

# =============================================================================
# PASSO 3: Usar moa-pom do Maven (JAR menor com dependencias)
# =============================================================================
print("\n--- PASSO 3: Verificar moa-pom no Maven ---")

if not found_moa:
    # MOA no Maven pode estar como 'moa-pom' ou outro artifact
    pom_url = "https://repo1.maven.org/maven2/nz/ac/waikato/cms/moa/"

    print("Listando versoes disponiveis no Maven...")
    try:
        # Baixar index
        result = subprocess.run(
            ["curl", "-s", pom_url],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout:
            # Extrair versoes
            import re
            versions = re.findall(r'href="(\d{4}\.[^/"]+)/"', result.stdout)
            print(f"Versoes encontradas: {versions[:10]}")

            # Tentar baixar versao mais proxima de 2018
            for v in sorted(versions, reverse=True):
                if '2018' in v or '2019' in v:
                    jar_url = f"{pom_url}{v}/moa-{v}.jar"
                    jar_path = MOA_DIR / f'moa-{v}.jar'
                    print(f"  Tentando {v}...")
                    try:
                        urllib.request.urlretrieve(jar_url, jar_path)
                        if jar_path.exists() and jar_path.stat().st_size > 100000:
                            print(f"    [OK] {jar_path.stat().st_size/(1024*1024):.1f} MB")
                            found_moa = jar_path
                            break
                    except:
                        pass
    except Exception as e:
        print(f"[ERRO] {e}")

# =============================================================================
# PASSO 4: Usar MOA-dependencies.jar do ROSE como fallback
# =============================================================================
print("\n--- PASSO 4: Fallback - MOA-dependencies.jar ---")

ROSE_MOA = WORK_DIR / 'rose_jars' / 'MOA-dependencies.jar'

if not found_moa and ROSE_MOA.exists():
    print(f"Usando MOA-dependencies.jar do ROSE como fallback")
    print(f"  (Versao mais recente, pode nao ser compativel)")
    found_moa = ROSE_MOA

# =============================================================================
# PASSO 5: Listar JARs disponiveis
# =============================================================================
print("\n--- PASSO 5: JARs MOA disponiveis ---")

print(f"\nEm {MOA_DIR}/:")
for jar in MOA_DIR.glob("*.jar"):
    print(f"  {jar.name} ({jar.stat().st_size/(1024*1024):.1f} MB)")

if found_moa:
    print(f"\n[SELECIONADO] {found_moa.name}")

# =============================================================================
# PASSO 6: Compilar classes CDCMS
# =============================================================================
print("\n--- PASSO 6: Compilar classes CDCMS ---")

CDCMS_REPO = WORK_DIR / 'CDCMS_CIL_src' / 'CDCMS.CIL'
SRC_DIR = CDCMS_REPO / 'Implementation' / 'moa' / 'src' / 'main' / 'java'
BUILD_DIR = WORK_DIR / 'cdcms_build_final'

# Weka
WEKA_JAR = None
for weka in MOA_DIR.glob("weka*.jar"):
    WEKA_JAR = weka
    break

if not WEKA_JAR:
    # Baixar Weka 3.9.2 (conforme pom.xml do CDCMS)
    weka_path = MOA_DIR / 'weka-dev-3.9.2.jar'
    weka_url = "https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka/weka-dev/3.9.2/weka-dev-3.9.2.jar"
    print("Baixando weka-dev-3.9.2.jar...")
    try:
        urllib.request.urlretrieve(weka_url, weka_path)
        WEKA_JAR = weka_path
        print(f"[OK] {weka_path.stat().st_size/(1024*1024):.1f} MB")
    except Exception as e:
        print(f"[ERRO] {e}")

if found_moa and SRC_DIR.exists():
    # Limpar build anterior
    import shutil
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir()

    # Encontrar APENAS arquivos CDCMS
    cdcms_files = []
    for pattern in ['**/CDCMS*.java', '**/DDM_GMean.java', '**/PMAUC*.java']:
        cdcms_files.extend(SRC_DIR.glob(pattern))

    print(f"\nArquivos CDCMS: {len(cdcms_files)}")

    if cdcms_files:
        # Montar classpath
        classpath = str(found_moa)
        if WEKA_JAR and WEKA_JAR.exists():
            classpath += f":{WEKA_JAR}"

        # Criar arquivo de fontes
        sources_file = BUILD_DIR / 'sources.txt'
        with open(sources_file, 'w') as f:
            for jf in cdcms_files:
                f.write(str(jf) + '\n')

        print(f"Compilando contra {found_moa.name}...")

        compile_cmd = [
            "javac",
            "-d", str(BUILD_DIR),
            "-cp", classpath,
            "-source", "1.8",
            "-target", "1.8",
            "-Xlint:none",
            f"@{sources_file}"
        ]

        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)

        class_files = list(BUILD_DIR.rglob("*.class"))
        print(f"Classes compiladas: {len(class_files)}")

        if class_files:
            # Criar JAR
            CDCMS_JAR = CDCMS_JARS_DIR / 'cdcms_cil_final.jar'
            jar_cmd = ["jar", "cf", str(CDCMS_JAR), "-C", str(BUILD_DIR), "."]
            subprocess.run(jar_cmd, capture_output=True, timeout=60)

            if CDCMS_JAR.exists():
                print(f"[OK] JAR criado: {CDCMS_JAR.name} ({CDCMS_JAR.stat().st_size/1024:.1f} KB)")
        else:
            print("[ERRO] Compilacao falhou")
            if result.stderr:
                errors = [l for l in result.stderr.split('\n') if 'error:' in l]
                error_types = {}
                for e in errors:
                    if "cannot find symbol" in e:
                        error_types["cannot find symbol"] = error_types.get("cannot find symbol", 0) + 1
                    elif "package" in e and "does not exist" in e:
                        error_types["package not found"] = error_types.get("package not found", 0) + 1
                    else:
                        error_types["outro"] = error_types.get("outro", 0) + 1

                print(f"Tipos de erro:")
                for et, count in error_types.items():
                    print(f"  {et}: {count}")

                # Mostrar exemplos
                print("\nExemplos de erros:")
                for e in errors[:5]:
                    print(f"  {e[:100]}")

# =============================================================================
# PASSO 7: Testar
# =============================================================================
print("\n--- PASSO 7: Testar ---")

CDCMS_JAR = CDCMS_JARS_DIR / 'cdcms_cil_final.jar'

if CDCMS_JAR.exists() and found_moa:
    classpath = f"{CDCMS_JAR}:{found_moa}"
    if WEKA_JAR and WEKA_JAR.exists():
        classpath += f":{WEKA_JAR}"

    print(f"Classpath: {CDCMS_JAR.name}:{found_moa.name}")

    test_cmd = [
        "java", "-Xmx2g",
        "-cp", classpath,
        "moa.DoTask",
        "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean"
    ]

    result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr

    if "Exception" not in output and result.returncode == 0:
        print("[SUCESSO] Classe reconhecida!")
    else:
        print("[FALHA]")
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

print(f"\nMOA encontrado: {found_moa.name if found_moa else 'NENHUM'}")
print(f"CDCMS JAR: {CDCMS_JAR if CDCMS_JAR.exists() else 'NAO COMPILADO'}")
