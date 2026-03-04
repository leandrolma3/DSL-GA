# =============================================================================
# DESCOBRIR: Qual versao MOA o CDCMS.CIL original usa
# =============================================================================
# Analisar o cdcms_cil.jar original para descobrir contra qual MOA foi compilado
# Depois baixar essa versao especifica do Maven
# =============================================================================

import subprocess
from pathlib import Path
import urllib.request
import re
import zipfile
import tempfile

print("="*70)
print("DESCOBRIR VERSAO MOA DO CDCMS.CIL")
print("="*70)

WORK_DIR = Path('/content')
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
MOA_JARS_DIR = WORK_DIR / 'moa_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

MOA_JARS_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

# Encontrar JAR CDCMS original
CDCMS_JAR = None
for candidate in ['cdcms_cil.jar', 'cdcms_cil_clean.jar', 'cdcms_cil_minimal.jar']:
    path = CDCMS_JARS_DIR / candidate
    if path.exists():
        CDCMS_JAR = path
        break

# =============================================================================
# PASSO 1: Analisar MANIFEST e estrutura do JAR
# =============================================================================
print("\n--- PASSO 1: Analisar JAR original ---")

if CDCMS_JAR:
    print(f"JAR encontrado: {CDCMS_JAR.name} ({CDCMS_JAR.stat().st_size/1024:.1f} KB)")

    # Extrair MANIFEST
    result = subprocess.run(
        f'unzip -p "{CDCMS_JAR}" META-INF/MANIFEST.MF 2>/dev/null',
        shell=True, capture_output=True, text=True
    )
    if result.stdout:
        print("\nMANIFEST.MF:")
        for line in result.stdout.strip().split('\n')[:15]:
            print(f"  {line}")

    # Procurar por pom.properties (se for JAR Maven)
    result2 = subprocess.run(
        f'unzip -l "{CDCMS_JAR}" 2>/dev/null | grep -i pom',
        shell=True, capture_output=True, text=True
    )
    if result2.stdout:
        print("\nArquivos POM encontrados:")
        print(result2.stdout)

    # Listar classes MOA no JAR para identificar versao
    result3 = subprocess.run(
        f'jar tf "{CDCMS_JAR}" 2>/dev/null | grep "moa/core/.*\\.class" | head -10',
        shell=True, capture_output=True, text=True
    )
    if result3.stdout:
        print("\nClasses MOA incluidas:")
        for line in result3.stdout.strip().split('\n')[:5]:
            print(f"  {line}")
else:
    print("[ERRO] Nenhum JAR CDCMS encontrado")

# =============================================================================
# PASSO 2: Verificar versoes MOA disponiveis no Maven
# =============================================================================
print("\n--- PASSO 2: Versoes MOA disponiveis ---")

# Versoes conhecidas do MOA
MOA_VERSIONS = [
    "2022.12.0",  # Mais recente
    "2022.07.01",
    "2021.07.0",
    "2020.07.1",
    "2019.05.0",
    "2018.06.0",
    "2017.06",
]

print("Versoes MOA conhecidas:")
for v in MOA_VERSIONS:
    print(f"  - {v}")

# =============================================================================
# PASSO 3: Baixar e testar cada versao do MOA
# =============================================================================
print("\n--- PASSO 3: Testar versoes MOA ---")

# Usar o cdcms_cil_minimal.jar ou similar
CDCMS_MINIMAL = CDCMS_JARS_DIR / 'cdcms_cil_minimal.jar'
CDCMS_CLEAN = CDCMS_JARS_DIR / 'cdcms_cil_clean.jar'
CDCMS_ORIGINAL = CDCMS_JARS_DIR / 'cdcms_cil.jar'

CDCMS_TO_TEST = None
for jar in [CDCMS_MINIMAL, CDCMS_CLEAN, CDCMS_ORIGINAL]:
    if jar.exists():
        CDCMS_TO_TEST = jar
        print(f"Usando: {jar.name}")
        break

if not CDCMS_TO_TEST:
    print("[ERRO] Nenhum JAR CDCMS disponivel")
else:
    working_versions = []

    for moa_version in MOA_VERSIONS[:4]:  # Testar as 4 mais recentes
        print(f"\n  Testando MOA {moa_version}...")

        moa_jar = MOA_JARS_DIR / f'moa-{moa_version}.jar'

        # Baixar se nao existir
        if not moa_jar.exists():
            # URL do Maven Central
            url = f"https://repo1.maven.org/maven2/nz/ac/waikato/cms/moa/moa/{moa_version}/moa-{moa_version}.jar"
            print(f"    Baixando de Maven Central...")

            try:
                urllib.request.urlretrieve(url, moa_jar)
                if moa_jar.exists():
                    print(f"    [OK] {moa_jar.stat().st_size/(1024*1024):.1f} MB")
                else:
                    print(f"    [FALHA] Download falhou")
                    continue
            except Exception as e:
                print(f"    [ERRO] {e}")
                continue

        # Testar
        classpath = f"{CDCMS_TO_TEST}:{moa_jar}"

        test_cmd = [
            "java", "-Xmx2g",
            "-cp", classpath,
            "moa.DoTask",
            "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean"
        ]

        try:
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
            output = result.stdout + result.stderr

            if "Exception" not in output and result.returncode == 0:
                print(f"    [SUCESSO] MOA {moa_version} funciona!")
                working_versions.append((moa_version, moa_jar))
                # Mostrar output
                for line in output.strip().split('\n')[:2]:
                    if line.strip():
                        print(f"      {line[:60]}")
            else:
                # Identificar tipo de erro
                if "NoClassDefFoundError" in output:
                    error_type = "NoClassDefFoundError"
                elif "ClassNotFoundException" in output:
                    error_type = "ClassNotFoundException"
                elif "not an instance" in output:
                    error_type = "not an instance of moa.tasks"
                else:
                    error_type = "Outro erro"
                print(f"    [FALHA] {error_type}")

        except subprocess.TimeoutExpired:
            print(f"    [TIMEOUT]")
        except Exception as e:
            print(f"    [ERRO] {e}")

    # =============================================================================
    # PASSO 4: Testar com Weka adicional
    # =============================================================================
    if not working_versions:
        print("\n--- PASSO 4: Testar com Weka adicional ---")
        print("MOA standalone precisa de Weka. Baixando...")

        # Baixar Weka
        weka_versions = ["3.9.6", "3.9.5", "3.9.4"]
        weka_jar = None

        for weka_v in weka_versions:
            weka_path = MOA_JARS_DIR / f'weka-stable-{weka_v}.jar'
            if weka_path.exists():
                weka_jar = weka_path
                break

            url = f"https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka/weka-stable/{weka_v}/weka-stable-{weka_v}.jar"
            try:
                print(f"  Baixando weka-stable-{weka_v}.jar...")
                urllib.request.urlretrieve(url, weka_path)
                if weka_path.exists() and weka_path.stat().st_size > 1000000:
                    weka_jar = weka_path
                    print(f"  [OK] {weka_path.stat().st_size/(1024*1024):.1f} MB")
                    break
            except:
                continue

        if weka_jar:
            # Testar cada MOA com Weka
            for moa_version in MOA_VERSIONS[:4]:
                moa_jar = MOA_JARS_DIR / f'moa-{moa_version}.jar'
                if not moa_jar.exists():
                    continue

                print(f"\n  Testando MOA {moa_version} + Weka...")

                classpath = f"{CDCMS_TO_TEST}:{moa_jar}:{weka_jar}"

                test_cmd = [
                    "java", "-Xmx2g",
                    "-cp", classpath,
                    "moa.DoTask",
                    "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean"
                ]

                try:
                    result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
                    output = result.stdout + result.stderr

                    if "Exception" not in output and result.returncode == 0:
                        print(f"    [SUCESSO] MOA {moa_version} + Weka funciona!")
                        working_versions.append((moa_version, moa_jar, weka_jar))
                    else:
                        if "not an instance" in output:
                            print(f"    [FALHA] Incompatibilidade de versao")
                        else:
                            print(f"    [FALHA]")
                except:
                    pass

    # =============================================================================
    # RESUMO
    # =============================================================================
    print("\n" + "="*70)
    print("RESUMO")
    print("="*70)

    if working_versions:
        print(f"\n*** ENCONTRADA VERSAO COMPATIVEL! ***")
        print(f"\nVersao: MOA {working_versions[0][0]}")
        print(f"JAR: {working_versions[0][1]}")

        if len(working_versions[0]) > 2:
            print(f"Weka: {working_versions[0][2]}")

        # Criar comando de teste completo
        print("\n--- Executar teste completo ---")

        moa_jar = working_versions[0][1]
        classpath = f"{CDCMS_TO_TEST}:{moa_jar}"
        if len(working_versions[0]) > 2:
            classpath += f":{working_versions[0][2]}"

        # Criar arquivo de teste
        test_arff = TEST_DIR / 'test_data.arff'
        test_arff_content = """@relation test_imbalanced
@attribute a1 numeric
@attribute a2 numeric
@attribute a3 numeric
@attribute class {0,1}
@data
"""
        import random
        random.seed(42)
        data_lines = []
        for i in range(2000):
            a1 = random.gauss(0, 1) + (0 if random.random() < 0.9 else 2)
            a2 = random.gauss(0, 1) + (0 if random.random() < 0.9 else 2)
            a3 = random.gauss(0, 1)
            cls = 0 if random.random() < 0.9 else 1
            data_lines.append(f"{a1:.4f},{a2:.4f},{a3:.4f},{cls}")

        with open(test_arff, 'w') as f:
            f.write(test_arff_content + '\n'.join(data_lines))

        output_file = TEST_DIR / f'cdcms_moa_{working_versions[0][0]}_output.csv'

        task_string = f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff}) -l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2) -f 500 -d {output_file}"

        exec_cmd = [
            "java", "-Xmx4g",
            "-cp", classpath,
            "moa.DoTask",
            task_string
        ]

        print(f"Executando teste completo...")
        import time
        start = time.time()

        result = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=180)

        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"[SUCESSO TOTAL] Arquivo criado: {output_file.stat().st_size} bytes")
            print(f"Tempo: {time.time()-start:.1f}s")

            with open(output_file) as f:
                lines = f.readlines()
            print(f"Linhas: {len(lines)}")
        else:
            print(f"[FALHA] Execucao real falhou")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[:5]:
                    print(f"  {line[:80]}")

    else:
        print("\n[FALHA] Nenhuma versao MOA compativel encontrada")
        print("\nPossiveis causas:")
        print("  1. CDCMS.CIL foi compilado com MOA modificado/customizado")
        print("  2. Dependencias adicionais necessarias")
        print("\nProximos passos:")
        print("  1. Verificar se ha instrucoes de build no README do CDCMS.CIL")
        print("  2. Contatar autores: https://github.com/michaelchiucw/CDCMS.CIL/issues")
        print("  3. Verificar se ha releases no GitHub com JAR pre-compilado")
