# =============================================================================
# TESTE DIRETO DO CDCMS.CIL
# =============================================================================
# O timeout no teste anterior indica que a task ESTA funcionando!
# Vamos executar com timeout maior e argumentos corretos
# =============================================================================

import subprocess
from pathlib import Path
import time

print("="*70)
print("TESTE DIRETO DO CDCMS.CIL")
print("="*70)

WORK_DIR = Path('/content')
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

TEST_DIR.mkdir(exist_ok=True)

CDCMS_JAR = CDCMS_JARS_DIR / 'cdcms_cil_final.jar'

# Criar classpath sem MOA externo
all_jars = list(DEPS_DIR.glob("*.jar"))
non_moa_jars = [j for j in all_jars if 'moa' not in j.name.lower()]
classpath = str(CDCMS_JAR) + ":" + ":".join(str(j) for j in non_moa_jars)

print(f"JAR: {CDCMS_JAR.name}")
print(f"Dependencias: {len(non_moa_jars)}")

# =============================================================================
# PASSO 1: Criar arquivo de teste
# =============================================================================
print("\n--- PASSO 1: Criar arquivo de teste ---")

test_arff = TEST_DIR / 'test_imbalanced.arff'
with open(test_arff, 'w') as f:
    f.write("@relation test_imbalanced\n")
    f.write("@attribute a1 numeric\n")
    f.write("@attribute a2 numeric\n")
    f.write("@attribute a3 numeric\n")
    f.write("@attribute class {0,1}\n")
    f.write("@data\n")

    import random
    random.seed(42)
    for i in range(3000):
        # Dados com drift simulado
        if i < 1000:
            a1 = random.gauss(0, 1)
            a2 = random.gauss(0, 1)
        elif i < 2000:
            a1 = random.gauss(2, 1)  # drift
            a2 = random.gauss(2, 1)
        else:
            a1 = random.gauss(-1, 1)  # outro drift
            a2 = random.gauss(-1, 1)

        a3 = random.gauss(0, 1)
        # 90% classe 0, 10% classe 1 (imbalanced)
        cls = 0 if random.random() < 0.9 else 1
        f.write(f"{a1:.4f},{a2:.4f},{a3:.4f},{cls}\n")

print(f"[OK] Arquivo criado: {test_arff}")
print(f"     Instancias: 3000")

# =============================================================================
# PASSO 2: Executar CDCMS_CIL_GMean
# =============================================================================
print("\n--- PASSO 2: Executar CDCMS_CIL_GMean ---")

output_file = TEST_DIR / 'cdcms_cil_gmean_output.csv'

# Comando completo
task_string = f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff}) -l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2) -f 500 -d {output_file}"

cmd = [
    "java", "-Xmx4g",
    "-cp", classpath,
    "moa.DoTask",
    task_string
]

print(f"Task: {task_string[:80]}...")
print("\nExecutando (timeout: 5 minutos)...")

start = time.time()

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    duration = time.time() - start

    print(f"\nTempo: {duration:.1f}s")
    print(f"Return code: {result.returncode}")

    # Verificar output
    if output_file.exists():
        size = output_file.stat().st_size
        print(f"\n[OK] Arquivo de saida criado: {size} bytes")

        with open(output_file) as f:
            lines = f.readlines()

        print(f"Linhas: {len(lines)}")

        if lines:
            print(f"\nHeader: {lines[0].strip()}")
            if len(lines) > 1:
                print(f"Linha 1: {lines[1].strip()[:80]}")
            if len(lines) > 2:
                print(f"Linha 2: {lines[2].strip()[:80]}")

        # Verificar se tem metricas validas
        if len(lines) > 1:
            print("\n*** SUCESSO! CDCMS.CIL FUNCIONANDO! ***")
    else:
        print("\n[AVISO] Arquivo de saida nao criado")

        # Mostrar stderr
        if result.stderr:
            print("\nSTDERR:")
            for line in result.stderr.strip().split('\n')[:10]:
                print(f"  {line[:80]}")

        if result.stdout:
            print("\nSTDOUT:")
            for line in result.stdout.strip().split('\n')[:10]:
                print(f"  {line[:80]}")

except subprocess.TimeoutExpired:
    duration = time.time() - start
    print(f"\n[TIMEOUT] Execucao demorou mais de 5 minutos ({duration:.1f}s)")
    print("O algoritmo pode estar rodando mas muito lento")

    # Verificar se criou arquivo parcial
    if output_file.exists():
        print(f"\nArquivo parcial existe: {output_file.stat().st_size} bytes")

except Exception as e:
    print(f"\n[ERRO] {e}")

# =============================================================================
# PASSO 3: Se falhou, tentar classificador mais simples
# =============================================================================
print("\n--- PASSO 3: Testar com classificador simples (NaiveBayes) ---")

if not (output_file.exists() and output_file.stat().st_size > 100):
    output_nb = TEST_DIR / 'naivebayes_output.csv'

    task_nb = f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff}) -l moa.classifiers.bayes.NaiveBayes -f 500 -d {output_nb}"

    cmd_nb = ["java", "-Xmx2g", "-cp", classpath, "moa.DoTask", task_nb]

    print("Testando NaiveBayes para verificar se o MOA funciona...")

    try:
        result_nb = subprocess.run(cmd_nb, capture_output=True, text=True, timeout=60)

        if output_nb.exists() and output_nb.stat().st_size > 0:
            print(f"[OK] NaiveBayes funciona! Output: {output_nb.stat().st_size} bytes")
            print("\nO MOA esta funcionando. O problema pode ser especifico do CDCMS.")
        else:
            print("[FALHA] NaiveBayes tambem falhou")
            if result_nb.stderr:
                for line in result_nb.stderr.strip().split('\n')[:5]:
                    print(f"  {line[:80]}")

    except subprocess.TimeoutExpired:
        print("[TIMEOUT] NaiveBayes tambem deu timeout")

# =============================================================================
# RESUMO FINAL
# =============================================================================
print("\n" + "="*70)
print("RESUMO FINAL")
print("="*70)

if output_file.exists() and output_file.stat().st_size > 100:
    print("\n*** CDCMS.CIL FUNCIONANDO! ***")
    print(f"\nJAR: {CDCMS_JAR}")
    print(f"Output: {output_file}")
    print(f"\nPara usar em seus experimentos:")
    print(f"  Classpath: cdcms_cil_final.jar + dependencias (sem outro moa*.jar)")
else:
    print("\n[STATUS] Verificar logs acima para diagnostico")
