# =============================================================================
# ABORDAGEM ALTERNATIVA: Usar MOA Oficial do Waikato
# =============================================================================
# O MOA-dependencies.jar do ROSE pode ser uma versão customizada/incompleta
# Esta célula baixa e testa com o MOA oficial
# =============================================================================

import subprocess
import urllib.request
from pathlib import Path
import time

print("="*70)
print("ABORDAGEM ALTERNATIVA: MOA Oficial do Waikato")
print("="*70)

# Diretório para MOA oficial
moa_official_dir = WORK_DIR / 'moa_official'
moa_official_dir.mkdir(exist_ok=True)

# =============================================================================
# PASSO 1: Baixar MOA Release do GitHub oficial
# =============================================================================
print("\n--- PASSO 1: Baixar MOA Oficial ---")

# MOA 2022.07.01 - versão mais recente estável
MOA_VERSION = "2022.07.01"
MOA_JAR_NAME = f"moa-release-{MOA_VERSION}.jar"
MOA_URL = f"https://github.com/Waikato/moa/releases/download/{MOA_VERSION}/{MOA_JAR_NAME}"

moa_jar_path = moa_official_dir / MOA_JAR_NAME

if moa_jar_path.exists():
    print(f"[OK] {MOA_JAR_NAME} já existe ({moa_jar_path.stat().st_size/(1024*1024):.1f} MB)")
else:
    print(f"Baixando {MOA_JAR_NAME}...")
    try:
        # Usar wget que é mais confiável para arquivos grandes
        wget_cmd = f'wget -q --show-progress -O "{moa_jar_path}" "{MOA_URL}"'
        result = subprocess.run(wget_cmd, shell=True, timeout=300)

        if moa_jar_path.exists() and moa_jar_path.stat().st_size > 10*1024*1024:
            print(f"[OK] Baixado: {moa_jar_path.stat().st_size/(1024*1024):.1f} MB")
        else:
            print("[ERRO] Download falhou ou arquivo muito pequeno")
            # Tentar URL alternativa
            print("Tentando URL alternativa...")
            ALT_URL = "https://sourceforge.net/projects/moa-datastream/files/MOA/2022%20July/moa-release-2022.07.01.jar"
            wget_cmd = f'wget -q -O "{moa_jar_path}" "{ALT_URL}"'
            subprocess.run(wget_cmd, shell=True, timeout=300)
    except Exception as e:
        print(f"[ERRO] {e}")

# =============================================================================
# PASSO 2: Verificar conteúdo do MOA oficial
# =============================================================================
print("\n--- PASSO 2: Verificar MOA Oficial ---")

if moa_jar_path.exists():
    print(f"Tamanho: {moa_jar_path.stat().st_size/(1024*1024):.1f} MB")

    # Verificar se contém as tasks necessárias
    tasks_check = subprocess.run(
        f'jar tf "{moa_jar_path}" 2>/dev/null | grep -E "EvaluateInterleavedTestThenTrain|WriteCommandLineTemplate|DoTask"',
        shell=True, capture_output=True, text=True
    )

    if tasks_check.stdout.strip():
        print("\nTasks encontradas:")
        for line in tasks_check.stdout.strip().split('\n')[:10]:
            print(f"  {line}")
    else:
        print("[AVISO] Tasks não encontradas - pode ser JAR errado")

    # Verificar classes Weka
    weka_check = subprocess.run(
        f'jar tf "{moa_jar_path}" 2>/dev/null | grep "weka/clusterers" | head -5',
        shell=True, capture_output=True, text=True
    )

    if weka_check.stdout.strip():
        print("\nClasses Weka clusterers:")
        for line in weka_check.stdout.strip().split('\n'):
            print(f"  {line}")
    else:
        print("\n[AVISO] Classes Weka clusterers não encontradas")

# =============================================================================
# PASSO 3: Testar moa.DoTask com MOA oficial
# =============================================================================
print("\n--- PASSO 3: Testar moa.DoTask ---")

if moa_jar_path.exists():
    # Testar sem argumentos
    test_cmd = f'java -Xmx1g -cp "{moa_jar_path}" moa.DoTask 2>&1 | head -20'
    result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=30)

    output = result.stdout + result.stderr
    print("Saída de moa.DoTask:")
    for line in output.strip().split('\n')[:10]:
        print(f"  {line[:80]}")

# =============================================================================
# PASSO 4: Testar CDCMS.CIL com MOA oficial
# =============================================================================
print("\n--- PASSO 4: Testar CDCMS.CIL com MOA Oficial ---")

CDCMS_CLEAN_JAR = CDCMS_JARS_DIR / 'cdcms_cil_clean.jar'

if moa_jar_path.exists() and CDCMS_CLEAN_JAR.exists():
    # Classpath: CDCMS primeiro, depois MOA oficial
    classpath = f"{CDCMS_CLEAN_JAR}:{moa_jar_path}"
    print(f"Classpath: {CDCMS_CLEAN_JAR.name}:{moa_jar_path.name}")

    # Teste 1: WriteCommandLineTemplate
    print("\n  Teste 1: WriteCommandLineTemplate")
    test1_cmd = f'java -Xmx2g -cp "{classpath}" moa.DoTask "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean" 2>&1'
    result = subprocess.run(test1_cmd, shell=True, capture_output=True, text=True, timeout=30)

    output = result.stdout + result.stderr
    if "Exception" not in output:
        print("    [OK] WriteCommandLineTemplate funcionou!")
        for line in output.strip().split('\n')[:5]:
            print(f"    {line[:70]}")
    else:
        print("    [FALHA] WriteCommandLineTemplate")
        for line in output.strip().split('\n')[:3]:
            if "Exception" in line or "Error" in line:
                print(f"    {line[:70]}")

    # Teste 2: Execução real
    print("\n  Teste 2: Execução real com EvaluateInterleavedTestThenTrain")
    output_file = TEST_DIR / 'cdcms_moa_oficial_output.csv'

    test_arff_abs = str(test_arff.resolve())
    output_file_abs = str(output_file.resolve())

    task_string = f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff_abs}) -l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2) -f 500 -d {output_file_abs}"

    exec_cmd = [
        "java", "-Xmx4g",
        "-cp", classpath,
        "moa.DoTask",
        task_string
    ]

    print(f"    Executando...")
    start = time.time()

    try:
        result = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=180)
        duration = time.time() - start

        print(f"    Tempo: {duration:.1f}s")
        print(f"    Return code: {result.returncode}")

        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"    [SUCESSO] Arquivo criado: {output_file.stat().st_size} bytes")

            # Mostrar conteúdo
            with open(output_file) as f:
                lines = f.readlines()
            print(f"    Linhas: {len(lines)}")
            if lines:
                print(f"    Header: {lines[0].strip()[:60]}...")
        else:
            print(f"    [FALHA] Arquivo não criado")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[:5]:
                    print(f"    {line[:70]}")

    except subprocess.TimeoutExpired:
        print("    [ERRO] Timeout!")
    except Exception as e:
        print(f"    [ERRO] {e}")

elif not moa_jar_path.exists():
    print("[SKIP] MOA oficial não disponível")
elif not CDCMS_CLEAN_JAR.exists():
    print("[SKIP] cdcms_cil_clean.jar não encontrado")

# =============================================================================
# PASSO 5: Se ainda falhar, testar ordem inversa do classpath
# =============================================================================
print("\n--- PASSO 5: Testar com ordem inversa do classpath ---")

if moa_jar_path.exists() and CDCMS_CLEAN_JAR.exists():
    # MOA primeiro, CDCMS depois
    classpath_inv = f"{moa_jar_path}:{CDCMS_CLEAN_JAR}"
    print(f"Classpath (invertido): {moa_jar_path.name}:{CDCMS_CLEAN_JAR.name}")

    output_file_inv = TEST_DIR / 'cdcms_moa_oficial_inv_output.csv'
    output_file_inv_abs = str(output_file_inv.resolve())

    task_string_inv = f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff_abs}) -l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2) -f 500 -d {output_file_inv_abs}"

    exec_cmd_inv = [
        "java", "-Xmx4g",
        "-cp", classpath_inv,
        "moa.DoTask",
        task_string_inv
    ]

    print(f"    Executando...")
    start = time.time()

    try:
        result = subprocess.run(exec_cmd_inv, capture_output=True, text=True, timeout=180)
        duration = time.time() - start

        print(f"    Tempo: {duration:.1f}s")
        print(f"    Return code: {result.returncode}")

        if output_file_inv.exists() and output_file_inv.stat().st_size > 0:
            print(f"    [SUCESSO] Arquivo criado: {output_file_inv.stat().st_size} bytes")
        else:
            print(f"    [FALHA] Arquivo não criado")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[:3]:
                    print(f"    {line[:70]}")

    except Exception as e:
        print(f"    [ERRO] {e}")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO DA ABORDAGEM MOA OFICIAL")
print("="*70)

results = []

# Verificar resultados
for name, path in [
    ("MOA Oficial (CDCMS primeiro)", TEST_DIR / 'cdcms_moa_oficial_output.csv'),
    ("MOA Oficial (MOA primeiro)", TEST_DIR / 'cdcms_moa_oficial_inv_output.csv')
]:
    if path.exists() and path.stat().st_size > 0:
        results.append((name, path))
        print(f"[OK] {name}: {path.stat().st_size} bytes")
    else:
        print(f"[X] {name}")

if results:
    print(f"\n*** SUCESSO: {results[0][0]} funcionou! ***")
    print(f"\nPróximo passo: Atualizar o notebook para usar MOA oficial ao invés de MOA-dependencies.jar do ROSE")
else:
    print("\n[FALHA] Nenhuma abordagem com MOA oficial funcionou")
    print("\nPróximos passos:")
    print("  1. Verificar se CDCMS.CIL foi compilado contra versão compatível do MOA")
    print("  2. Recompilar CDCMS.CIL usando o mesmo MOA release")
    print("  3. Contatar autores do CDCMS.CIL para JAR funcional")
