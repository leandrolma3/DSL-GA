# =============================================================================
# SOLUCAO: Usar MOA 2018.06.0 (versao correta para CDCMS.CIL)
# =============================================================================
# DESCOBERTA: O pom.xml do CDCMS.CIL especifica MOA 2018.6.1-SNAPSHOT
# Isso significa que precisamos usar MOA 2018.06.0 (release correspondente)
# =============================================================================

import subprocess
from pathlib import Path
import urllib.request
import time

print("="*70)
print("SOLUCAO: MOA 2018.06.0 (versao correta para CDCMS.CIL)")
print("="*70)

WORK_DIR = Path('/content')
MOA_2018_DIR = WORK_DIR / 'moa_2018'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

MOA_2018_DIR.mkdir(exist_ok=True)
CDCMS_JARS_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

# =============================================================================
# PASSO 1: Baixar MOA 2018.06.0 do Maven Central
# =============================================================================
print("\n--- PASSO 1: Baixar MOA 2018.06.0 ---")

MOA_VERSION = "2018.06.0"
MOA_JAR = MOA_2018_DIR / f'moa-{MOA_VERSION}.jar'

# URL do Maven Central
MOA_URL = f"https://repo1.maven.org/maven2/nz/ac/waikato/cms/moa/moa/{MOA_VERSION}/moa-{MOA_VERSION}.jar"

if MOA_JAR.exists() and MOA_JAR.stat().st_size > 1000000:
    print(f"[OK] moa-{MOA_VERSION}.jar ja existe ({MOA_JAR.stat().st_size/(1024*1024):.1f} MB)")
else:
    print(f"Baixando moa-{MOA_VERSION}.jar de Maven Central...")
    print(f"URL: {MOA_URL}")
    try:
        urllib.request.urlretrieve(MOA_URL, MOA_JAR)
        if MOA_JAR.exists():
            print(f"[OK] Baixado: {MOA_JAR.stat().st_size/(1024*1024):.1f} MB")
        else:
            print("[ERRO] Download falhou")
    except Exception as e:
        print(f"[ERRO] {e}")

# =============================================================================
# PASSO 2: Baixar Weka compativel (3.8.1 - versao da epoca)
# =============================================================================
print("\n--- PASSO 2: Baixar Weka 3.8.1 ---")

WEKA_VERSION = "3.8.1"
WEKA_JAR = MOA_2018_DIR / f'weka-stable-{WEKA_VERSION}.jar'
WEKA_URL = f"https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka/weka-stable/{WEKA_VERSION}/weka-stable-{WEKA_VERSION}.jar"

if WEKA_JAR.exists() and WEKA_JAR.stat().st_size > 1000000:
    print(f"[OK] weka-stable-{WEKA_VERSION}.jar ja existe ({WEKA_JAR.stat().st_size/(1024*1024):.1f} MB)")
else:
    print(f"Baixando weka-stable-{WEKA_VERSION}.jar...")
    try:
        urllib.request.urlretrieve(WEKA_URL, WEKA_JAR)
        if WEKA_JAR.exists():
            print(f"[OK] Baixado: {WEKA_JAR.stat().st_size/(1024*1024):.1f} MB")
    except Exception as e:
        print(f"[ERRO] {e}")
        # Tentar versao alternativa
        for alt_version in ["3.8.2", "3.8.3", "3.8.0"]:
            alt_url = f"https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka/weka-stable/{alt_version}/weka-stable-{alt_version}.jar"
            alt_jar = MOA_2018_DIR / f'weka-stable-{alt_version}.jar'
            try:
                print(f"Tentando weka-stable-{alt_version}...")
                urllib.request.urlretrieve(alt_url, alt_jar)
                if alt_jar.exists():
                    WEKA_JAR = alt_jar
                    print(f"[OK] Baixado: {alt_jar.stat().st_size/(1024*1024):.1f} MB")
                    break
            except:
                continue

# =============================================================================
# PASSO 3: Obter cdcms_cil.jar original
# =============================================================================
print("\n--- PASSO 3: Obter cdcms_cil.jar ---")

CDCMS_JAR = None

# Verificar se ja existe algum JAR CDCMS
for candidate in ['cdcms_cil.jar', 'cdcms_cil_clean.jar', 'cdcms_cil_minimal.jar']:
    path = CDCMS_JARS_DIR / candidate
    if path.exists():
        CDCMS_JAR = path
        print(f"[OK] Encontrado: {path.name} ({path.stat().st_size/1024:.1f} KB)")
        break

if not CDCMS_JAR:
    # Verificar se foi compilado antes
    for compiled in WORK_DIR.rglob("cdcms*.jar"):
        if compiled.stat().st_size > 10000:
            CDCMS_JAR = compiled
            print(f"[OK] Encontrado: {compiled}")
            break

if not CDCMS_JAR:
    print("[AVISO] Nenhum JAR CDCMS encontrado")
    print("Vamos tentar compilar as classes CDCMS contra MOA 2018...")

    # Tentar compilar apenas os arquivos CDCMS
    CDCMS_REPO = WORK_DIR / 'CDCMS_CIL_src' / 'CDCMS.CIL'
    SRC_DIR = CDCMS_REPO / 'Implementation' / 'moa' / 'src' / 'main' / 'java'

    if SRC_DIR.exists() and MOA_JAR.exists():
        BUILD_DIR = WORK_DIR / 'cdcms_build_2018'
        BUILD_DIR.mkdir(exist_ok=True)

        # Pegar APENAS os arquivos CDCMS (nao todos os 782)
        cdcms_files = []
        for pattern in ['**/CDCMS*.java', '**/DDM_GMean.java', '**/PMAUC*.java']:
            cdcms_files.extend(SRC_DIR.glob(pattern))

        print(f"\nArquivos CDCMS a compilar: {len(cdcms_files)}")
        for f in cdcms_files:
            print(f"  {f.name}")

        if cdcms_files:
            # Criar arquivo de fontes
            sources_file = BUILD_DIR / 'sources.txt'
            with open(sources_file, 'w') as f:
                for jf in cdcms_files:
                    f.write(str(jf) + '\n')

            # Classpath com MOA 2018 e Weka
            classpath = str(MOA_JAR)
            if WEKA_JAR.exists():
                classpath += f":{WEKA_JAR}"

            print(f"\nCompilando contra MOA {MOA_VERSION}...")
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
                CDCMS_JAR = CDCMS_JARS_DIR / 'cdcms_cil_2018.jar'
                jar_cmd = ["jar", "cf", str(CDCMS_JAR), "-C", str(BUILD_DIR), "."]
                subprocess.run(jar_cmd, capture_output=True, timeout=60)

                if CDCMS_JAR.exists():
                    print(f"[OK] JAR criado: {CDCMS_JAR.name} ({CDCMS_JAR.stat().st_size/1024:.1f} KB)")
            else:
                print("[ERRO] Compilacao falhou")
                if result.stderr:
                    errors = [l for l in result.stderr.split('\n') if 'error:' in l][:5]
                    for e in errors:
                        print(f"  {e[:80]}")

# =============================================================================
# PASSO 4: Testar com MOA 2018
# =============================================================================
print("\n--- PASSO 4: Testar com MOA 2018.06.0 ---")

if CDCMS_JAR and CDCMS_JAR.exists() and MOA_JAR.exists():
    # Montar classpath
    classpath = f"{CDCMS_JAR}:{MOA_JAR}"
    if WEKA_JAR.exists():
        classpath += f":{WEKA_JAR}"

    print(f"Classpath:")
    print(f"  1. {CDCMS_JAR.name}")
    print(f"  2. {MOA_JAR.name}")
    if WEKA_JAR.exists():
        print(f"  3. {WEKA_JAR.name}")

    # Teste 1: Verificar se classe e reconhecida
    print("\n  Teste 1: WriteCommandLineTemplate")
    test1_cmd = [
        "java", "-Xmx2g",
        "-cp", classpath,
        "moa.DoTask",
        "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean"
    ]

    result1 = subprocess.run(test1_cmd, capture_output=True, text=True, timeout=30)
    output1 = result1.stdout + result1.stderr

    if "Exception" not in output1 and result1.returncode == 0:
        print("    [SUCESSO] Classe reconhecida!")
        for line in output1.strip().split('\n')[:3]:
            if line.strip():
                print(f"    {line[:70]}")
        test1_ok = True
    else:
        print("    [FALHA]")
        for line in output1.strip().split('\n')[:5]:
            if line.strip():
                print(f"    {line[:80]}")
        test1_ok = False

    # Teste 2: Execucao real
    if test1_ok:
        print("\n  Teste 2: EvaluateInterleavedTestThenTrain")

        # Criar arquivo de teste
        test_arff = TEST_DIR / 'test_data.arff'
        test_content = """@relation test_imbalanced
@attribute a1 numeric
@attribute a2 numeric
@attribute a3 numeric
@attribute class {0,1}
@data
"""
        import random
        random.seed(42)
        for i in range(2000):
            a1 = random.gauss(0, 1) + (0 if random.random() < 0.9 else 2)
            a2 = random.gauss(0, 1)
            a3 = random.gauss(0, 1)
            cls = 0 if random.random() < 0.9 else 1
            test_content += f"{a1:.4f},{a2:.4f},{a3:.4f},{cls}\n"

        with open(test_arff, 'w') as f:
            f.write(test_content)

        output_file = TEST_DIR / 'cdcms_moa2018_output.csv'

        task_string = f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff}) -l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2) -f 500 -d {output_file}"

        exec_cmd = [
            "java", "-Xmx4g",
            "-cp", classpath,
            "moa.DoTask",
            task_string
        ]

        print("    Executando...")
        start = time.time()

        try:
            result2 = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=300)
            duration = time.time() - start

            print(f"    Tempo: {duration:.1f}s")

            if output_file.exists() and output_file.stat().st_size > 0:
                print(f"    [SUCESSO] Arquivo criado: {output_file.stat().st_size} bytes")

                with open(output_file) as f:
                    lines = f.readlines()
                print(f"    Linhas: {len(lines)}")
                if lines:
                    print(f"    Header: {lines[0].strip()[:60]}...")
            else:
                print("    [FALHA] Arquivo nao criado")
                if result2.stderr:
                    for line in result2.stderr.strip().split('\n')[:5]:
                        print(f"    {line[:80]}")

        except subprocess.TimeoutExpired:
            print("    [TIMEOUT]")
        except Exception as e:
            print(f"    [ERRO] {e}")
else:
    print("[SKIP] JARs nao disponiveis")

# =============================================================================
# PASSO 5: Se falhar, tentar com JAR original + dependencias extras
# =============================================================================
print("\n--- PASSO 5: Alternativas ---")

output_file = TEST_DIR / 'cdcms_moa2018_output.csv'
if not (output_file.exists() and output_file.stat().st_size > 0):
    print("Tentando com dependencias extras do MOA 2018...")

    # MOA 2018 precisa de algumas dependencias extras
    extra_deps = [
        ("commons-math3", "3.6.1", "org/apache/commons/commons-math3"),
        ("jfreechart", "1.0.19", "org/jfree/jfreechart"),
    ]

    for name, version, path in extra_deps:
        jar_path = MOA_2018_DIR / f'{name}-{version}.jar'
        if not jar_path.exists():
            url = f"https://repo1.maven.org/maven2/{path}/{version}/{name}-{version}.jar"
            try:
                print(f"  Baixando {name}-{version}.jar...")
                urllib.request.urlretrieve(url, jar_path)
            except:
                pass

    # Montar classpath completo
    all_jars = list(MOA_2018_DIR.glob("*.jar"))
    if CDCMS_JAR:
        classpath_full = str(CDCMS_JAR) + ":" + ":".join(str(j) for j in all_jars)

        print(f"\n  Testando com classpath completo ({len(all_jars)+1} JARs)...")

        test_cmd = [
            "java", "-Xmx2g",
            "-cp", classpath_full,
            "moa.DoTask",
            "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean"
        ]

        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr

        if "Exception" not in output:
            print("    [SUCESSO] Com dependencias extras!")
        else:
            print("    [FALHA]")

# =============================================================================
# RESUMO FINAL
# =============================================================================
print("\n" + "="*70)
print("RESUMO FINAL")
print("="*70)

output_file = TEST_DIR / 'cdcms_moa2018_output.csv'
if output_file.exists() and output_file.stat().st_size > 0:
    print("\n*** SUCESSO! CDCMS.CIL FUNCIONANDO COM MOA 2018! ***")
    print(f"\nConfiguracao que funciona:")
    print(f"  MOA: moa-{MOA_VERSION}.jar")
    if WEKA_JAR.exists():
        print(f"  Weka: {WEKA_JAR.name}")
    print(f"  CDCMS: {CDCMS_JAR.name if CDCMS_JAR else 'N/A'}")
else:
    print("\n[STATUS] CDCMS.CIL ainda nao funciona")
    print("\nArquivos baixados em moa_2018/:")
    for f in MOA_2018_DIR.glob("*.jar"):
        print(f"  {f.name} ({f.stat().st_size/(1024*1024):.1f} MB)")

    print("\nProximo passo: Usar o cdcms_cil.jar ORIGINAL do projeto")
    print("(Pode ser necessario obter o JAR compilado diretamente dos autores)")
