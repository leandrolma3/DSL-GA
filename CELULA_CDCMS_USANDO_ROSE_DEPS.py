# =============================================================================
# CELULA CORRIGIDA: CDCMS.CIL usando MOA-dependencies do ROSE
# =============================================================================
# Baseado na analise do Execute_All_Comparative_Models.ipynb e rose_wrapper.py
#
# INSIGHT CHAVE:
# - ROSE funciona porque usa apenas 2 JARs: ROSE-1.0.jar + MOA-dependencies.jar
# - MOA-dependencies.jar (64.6 MB) e um UBER JAR com TUDO bundled (Weka, MOA, etc.)
# - Nos tinhamos 53+ JARs fragmentados causando conflitos
#
# SOLUCAO:
# - Usar cdcms_cil.jar + MOA-dependencies.jar (do ROSE)
# - Isso elimina conflitos de classpath e garante que Weka clusterers existam
# =============================================================================

import subprocess
import time
from pathlib import Path

print("="*70)
print("CDCMS.CIL - USANDO MOA-dependencies.jar DO ROSE")
print("="*70)

# =============================================================================
# PASSO 1: Verificar JARs necessarios
# =============================================================================
print("\n--- Verificando JARs ---")

# Caminhos
CDCMS_JAR = CDCMS_JARS_DIR / 'cdcms_cil.jar'
ROSE_MOA_DEPS = WORK_DIR / 'rose_jars' / 'MOA-dependencies.jar'
SIZEOFAG_JAR = WORK_DIR / 'rose_jars' / 'sizeofag-1.0.4.jar'

# Verificar existencia
cdcms_ok = CDCMS_JAR.exists()
moa_deps_ok = ROSE_MOA_DEPS.exists()
sizeofag_ok = SIZEOFAG_JAR.exists()

print(f"cdcms_cil.jar: {cdcms_ok}")
if cdcms_ok:
    print(f"  Tamanho: {CDCMS_JAR.stat().st_size / (1024*1024):.2f} MB")

print(f"MOA-dependencies.jar: {moa_deps_ok}")
if moa_deps_ok:
    print(f"  Tamanho: {ROSE_MOA_DEPS.stat().st_size / (1024*1024):.2f} MB")

print(f"sizeofag-1.0.4.jar: {sizeofag_ok}")

# =============================================================================
# PASSO 2: Baixar JARs do ROSE se necessario
# =============================================================================
if not moa_deps_ok:
    print("\n--- Baixando JARs do ROSE ---")

    rose_dir = WORK_DIR / 'rose_jars'
    rose_dir.mkdir(exist_ok=True)

    base_url = "https://github.com/canoalberto/ROSE/raw/master"
    jars_to_download = [
        "MOA-dependencies.jar",
        "sizeofag-1.0.4.jar"
    ]

    for jar in jars_to_download:
        jar_path = rose_dir / jar
        if not jar_path.exists():
            url = f"{base_url}/{jar}"
            print(f"Baixando {jar}...")
            subprocess.run(f'wget -q -O "{jar_path}" "{url}"', shell=True)
            if jar_path.exists():
                print(f"  [OK] {jar} ({jar_path.stat().st_size/(1024*1024):.1f} MB)")
            else:
                print(f"  [ERRO] Falha ao baixar {jar}")
        else:
            print(f"[OK] {jar} ja existe")

    # Atualizar flags
    moa_deps_ok = ROSE_MOA_DEPS.exists()
    sizeofag_ok = SIZEOFAG_JAR.exists()

# =============================================================================
# PASSO 3: Verificar que CDCMS.CIL esta compilado
# =============================================================================
if not cdcms_ok:
    print("\n[AVISO] cdcms_cil.jar nao encontrado!")
    print("Execute a celula de compilacao do CDCMS.CIL primeiro.")
else:
    # Verificar classes principais
    check = subprocess.run(
        f'jar tf "{CDCMS_JAR}" | grep "CDCMS_CIL_GMean.class$"',
        shell=True, capture_output=True, text=True
    )
    if check.stdout.strip():
        print(f"\n[OK] Classe CDCMS_CIL_GMean encontrada")
    else:
        print(f"\n[AVISO] CDCMS_CIL_GMean.class nao encontrada no JAR!")

# =============================================================================
# PASSO 4: Executar CDCMS.CIL
# =============================================================================
if cdcms_ok and moa_deps_ok:
    print("\n" + "="*70)
    print("EXECUTANDO CDCMS.CIL")
    print("="*70)

    # Output
    output_file = TEST_DIR / 'cdcms_rose_deps_output.csv'
    log_file = TEST_DIR / 'cdcms_rose_deps_log.txt'

    # Limpar arquivo anterior
    if output_file.exists():
        output_file.unlink()

    # Classpath: cdcms PRIMEIRO, depois MOA-dependencies
    # (cdcms_cil.jar contem as classes CDCMS que queremos)
    # (MOA-dependencies.jar contem MOA + Weka bundled)
    classpath = f"{CDCMS_JAR}:{ROSE_MOA_DEPS}"

    # Caminhos absolutos
    test_arff_abs = str(test_arff.resolve())
    output_file_abs = str(output_file.resolve())

    print(f"\nClasspath: 2 JARs (igual ao ROSE!)")
    print(f"  1. {CDCMS_JAR.name}")
    print(f"  2. {ROSE_MOA_DEPS.name}")

    # Construir comando no MESMO formato do ROSE
    # Parametros CDCMS.CIL: -s ensembleSize, -t timeStepsInterval, -c numClasses
    ensemble_size = 10
    time_steps = 500
    num_classes = 2
    eval_frequency = 500

    # Construir task string (formato ROSE)
    task_parts = [
        "EvaluateInterleavedTestThenTrain",
        "-s", f"(ArffFileStream -f {test_arff_abs})",
        "-l", f"(moa.classifiers.meta.CDCMS_CIL_GMean -s {ensemble_size} -t {time_steps} -c {num_classes})",
        "-f", str(eval_frequency),
        "-d", output_file_abs
    ]
    task_string = " ".join(task_parts)

    # Construir comando Java
    cmd = ["java", "-Xmx4g"]

    # Adicionar sizeofag se disponivel (como ROSE faz)
    if sizeofag_ok:
        cmd.append(f"-javaagent:{SIZEOFAG_JAR}")

    cmd.extend(["-cp", classpath, "moa.DoTask", task_string])

    print(f"\nLearner: CDCMS_CIL_GMean")
    print(f"  Ensemble size: {ensemble_size}")
    print(f"  Time steps: {time_steps}")
    print(f"  Num classes: {num_classes}")
    print(f"  Eval frequency: {eval_frequency}")
    print(f"\nUsando sizeofag: {sizeofag_ok}")
    print(f"\nExecutando...")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        duration = time.time() - start_time

        # Salvar log completo
        with open(log_file, 'w') as f:
            f.write(f"Duration: {duration:.1f}s\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Classpath: {classpath}\n\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.write(f"--- STDOUT ---\n{result.stdout}\n\n")
            f.write(f"--- STDERR ---\n{result.stderr}\n")

        print(f"\nTempo: {duration:.1f}s")
        print(f"Return code: {result.returncode}")

        # Mostrar saidas
        if result.stdout.strip():
            print(f"\nSTDOUT (primeiras linhas):")
            for line in result.stdout.strip().split('\n')[:5]:
                print(f"  {line[:100]}")

        if result.stderr.strip():
            print(f"\nSTDERR (primeiras linhas):")
            for line in result.stderr.strip().split('\n')[:5]:
                print(f"  {line[:100]}")

    except subprocess.TimeoutExpired:
        print("[ERRO] Timeout!")
    except Exception as e:
        print(f"[ERRO] {e}")

# =============================================================================
# PASSO 5: Verificar resultado
# =============================================================================
print("\n" + "="*70)
print("VERIFICANDO RESULTADO")
print("="*70)

if output_file.exists() and output_file.stat().st_size > 0:
    print(f"\n[SUCESSO] Arquivo criado: {output_file}")
    print(f"  Tamanho: {output_file.stat().st_size} bytes")

    # Mostrar conteudo
    with open(output_file) as f:
        lines = f.readlines()
    print(f"  Linhas: {len(lines)}")
    if lines:
        print(f"\n  Header: {lines[0].strip()[:100]}")
        if len(lines) > 1:
            print(f"  Ultima linha: {lines[-1].strip()[:100]}")
else:
    print(f"\n[FALHA] Arquivo nao criado ou vazio")
    print(f"  Verifique o log: {log_file}")

    # Mostrar arquivos no diretorio
    if TEST_DIR.exists():
        print(f"\n  Arquivos em {TEST_DIR.name}/:")
        for f in TEST_DIR.iterdir():
            print(f"    - {f.name}")

# =============================================================================
# ALTERNATIVA: Se ainda falhar, testar CDCMS_CIL base (sem GMean)
# =============================================================================
if not (output_file.exists() and output_file.stat().st_size > 0):
    print("\n" + "="*70)
    print("ALTERNATIVA: Testando CDCMS_CIL base (sem GMean)")
    print("="*70)

    output_file_base = TEST_DIR / 'cdcms_base_output.csv'

    task_parts_base = [
        "EvaluateInterleavedTestThenTrain",
        "-s", f"(ArffFileStream -f {test_arff_abs})",
        "-l", f"(moa.classifiers.meta.CDCMS_CIL -s {ensemble_size} -t {time_steps})",
        "-f", str(eval_frequency),
        "-d", str(output_file_base.resolve())
    ]
    task_string_base = " ".join(task_parts_base)

    cmd_base = ["java", "-Xmx4g"]
    if sizeofag_ok:
        cmd_base.append(f"-javaagent:{SIZEOFAG_JAR}")
    cmd_base.extend(["-cp", classpath, "moa.DoTask", task_string_base])

    print("Executando CDCMS_CIL (versao base)...")
    result_base = subprocess.run(cmd_base, capture_output=True, text=True, timeout=180)

    print(f"Return code: {result_base.returncode}")
    if output_file_base.exists() and output_file_base.stat().st_size > 0:
        print(f"[OK] CDCMS_CIL base funcionou!")
    else:
        print(f"[X] CDCMS_CIL base tambem falhou")
        if result_base.stderr:
            print(f"STDERR: {result_base.stderr[:300]}")
