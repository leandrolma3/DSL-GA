# =============================================================================
# TESTAR CDCMS.CIL SEM CONFLITO DE MOA
# =============================================================================
# O cdcms_cil_final.jar JA CONTEM o MOA (e um fork)
# Nao podemos ter outro moa*.jar no classpath
# =============================================================================

import subprocess
from pathlib import Path
import time

print("="*70)
print("TESTAR CDCMS.CIL SEM CONFLITO DE MOA")
print("="*70)

WORK_DIR = Path('/content')
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

TEST_DIR.mkdir(exist_ok=True)

CDCMS_JAR = CDCMS_JARS_DIR / 'cdcms_cil_final.jar'

# =============================================================================
# PASSO 1: Criar classpath SEM moa*.jar
# =============================================================================
print("\n--- PASSO 1: Criar classpath limpo ---")

all_jars = list(DEPS_DIR.glob("*.jar"))
print(f"Total de JARs em deps: {len(all_jars)}")

# Filtrar - remover qualquer jar que contenha "moa" no nome
non_moa_jars = [j for j in all_jars if 'moa' not in j.name.lower()]
print(f"JARs sem MOA: {len(non_moa_jars)}")

# Mostrar quais foram removidos
moa_jars = [j for j in all_jars if 'moa' in j.name.lower()]
print(f"\nJARs MOA removidos do classpath:")
for j in moa_jars:
    print(f"  {j.name}")

# Classpath: CDCMS primeiro (contem MOA), depois dependencias
classpath = str(CDCMS_JAR) + ":" + ":".join(str(j) for j in non_moa_jars)

print(f"\nClasspath: cdcms_cil_final.jar + {len(non_moa_jars)} dependencias")

# =============================================================================
# PASSO 2: Testar moa.DoTask
# =============================================================================
print("\n--- PASSO 2: Testar moa.DoTask ---")

test1 = subprocess.run(
    ["java", "-Xmx2g", "-cp", classpath, "moa.DoTask"],
    capture_output=True, text=True, timeout=30
)

output1 = test1.stdout + test1.stderr
print("Saida:")
for line in output1.strip().split('\n')[:10]:
    print(f"  {line}")

# =============================================================================
# PASSO 3: Listar tasks disponiveis
# =============================================================================
print("\n--- PASSO 3: Listar tasks ---")

# Tentar listar tasks
test2 = subprocess.run(
    ["java", "-Xmx2g", "-cp", classpath, "moa.DoTask", "-h"],
    capture_output=True, text=True, timeout=30
)

output2 = test2.stdout + test2.stderr
if "task" in output2.lower():
    print("Tasks disponiveis:")
    for line in output2.strip().split('\n'):
        if 'task' in line.lower() or 'evaluate' in line.lower():
            print(f"  {line[:80]}")

# =============================================================================
# PASSO 4: Testar WriteCommandLineTemplate
# =============================================================================
print("\n--- PASSO 4: Testar WriteCommandLineTemplate ---")

test3 = subprocess.run(
    ["java", "-Xmx2g", "-cp", classpath, "moa.DoTask",
     "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean"],
    capture_output=True, text=True, timeout=30
)

output3 = test3.stdout + test3.stderr

if "Exception" not in output3 and test3.returncode == 0:
    print("[OK] CDCMS_CIL_GMean reconhecido!")
    for line in output3.strip().split('\n')[:5]:
        print(f"  {line[:70]}")
else:
    print("[FALHA]")
    for line in output3.strip().split('\n')[:5]:
        print(f"  {line[:80]}")

# =============================================================================
# PASSO 5: Testar EvaluateInterleavedTestThenTrain
# =============================================================================
print("\n--- PASSO 5: Testar EvaluateInterleavedTestThenTrain ---")

# Primeiro verificar se a task existe
test4 = subprocess.run(
    ["java", "-Xmx2g", "-cp", classpath, "moa.DoTask",
     "EvaluateInterleavedTestThenTrain"],
    capture_output=True, text=True, timeout=30
)

output4 = test4.stdout + test4.stderr

if "not an instance" in output4 or "Class not found" in output4:
    print("[AVISO] Task EvaluateInterleavedTestThenTrain nao funciona")

    # Tentar descobrir quais tasks existem
    print("\nTentando outras tasks...")

    for task in ["EvaluatePrequential", "EvaluateModel", "LearnModel"]:
        test_task = subprocess.run(
            ["java", "-Xmx2g", "-cp", classpath, "moa.DoTask", task],
            capture_output=True, text=True, timeout=10
        )
        output_task = test_task.stdout + test_task.stderr
        if "not an instance" not in output_task and "Class not found" not in output_task:
            print(f"  [OK] {task} funciona!")
        else:
            print(f"  [X] {task}")
else:
    print("[INFO] Task disponivel")
    for line in output4.strip().split('\n')[:5]:
        print(f"  {line[:80]}")

# =============================================================================
# PASSO 6: Tentar execucao com task diferente ou via GUI
# =============================================================================
print("\n--- PASSO 6: Tentativas alternativas ---")

# Criar arquivo de teste
test_arff = TEST_DIR / 'test_data.arff'
with open(test_arff, 'w') as f:
    f.write("@relation test\n@attribute a1 numeric\n@attribute a2 numeric\n@attribute class {0,1}\n@data\n")
    import random
    random.seed(42)
    for _ in range(1000):
        a1 = random.gauss(0, 1)
        a2 = random.gauss(0, 1)
        cls = 0 if random.random() < 0.9 else 1
        f.write(f"{a1:.4f},{a2:.4f},{cls}\n")

output_file = TEST_DIR / 'cdcms_output.csv'

# Tentar diferentes formatos de task
task_formats = [
    # Formato 1: Task completa como argumento unico
    f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff}) -l moa.classifiers.meta.CDCMS_CIL_GMean -f 500",

    # Formato 2: Usando nome completo da task
    f"moa.tasks.EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff}) -l moa.classifiers.meta.CDCMS_CIL_GMean",

    # Formato 3: EvaluatePrequential (alternativa)
    f"EvaluatePrequential -s (ArffFileStream -f {test_arff}) -l moa.classifiers.meta.CDCMS_CIL_GMean -f 500",
]

for i, task_str in enumerate(task_formats, 1):
    print(f"\n  Tentativa {i}:")
    print(f"    {task_str[:60]}...")

    test = subprocess.run(
        ["java", "-Xmx4g", "-cp", classpath, "moa.DoTask", task_str],
        capture_output=True, text=True, timeout=120
    )

    output = test.stdout + test.stderr

    if test.returncode == 0 and "Exception" not in output:
        print(f"    [SUCESSO!]")
        break
    else:
        error_line = [l for l in output.split('\n') if 'Exception' in l or 'Error' in l]
        if error_line:
            print(f"    [FALHA] {error_line[0][:60]}")
        else:
            print(f"    [FALHA] Return code: {test.returncode}")

# =============================================================================
# PASSO 7: Verificar versao do MOA no JAR compilado
# =============================================================================
print("\n--- PASSO 7: Verificar versao MOA ---")

# Extrair MANIFEST
manifest = subprocess.run(
    f'unzip -p "{CDCMS_JAR}" META-INF/MANIFEST.MF 2>/dev/null',
    shell=True, capture_output=True, text=True
)

if manifest.stdout:
    print("MANIFEST.MF:")
    for line in manifest.stdout.strip().split('\n')[:10]:
        print(f"  {line}")

# Verificar classes de tasks no JAR
tasks_check = subprocess.run(
    f'jar tf "{CDCMS_JAR}" | grep -i "tasks.*class$" | head -20',
    shell=True, capture_output=True, text=True
)

if tasks_check.stdout:
    print("\nClasses de tasks no JAR:")
    for line in tasks_check.stdout.strip().split('\n')[:10]:
        print(f"  {line}")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO")
print("="*70)

print(f"""
JAR compilado: {CDCMS_JAR}
Tamanho: {CDCMS_JAR.stat().st_size/(1024*1024):.1f} MB
Classes CDCMS: 30

Status: Compilacao OK, mas as tasks do MOA 2018.06 podem ter
        interface diferente das versoes mais recentes.

O cdcms_cil_final.jar contem MOA 2018.06 que e mais antigo.
As tasks podem ter nomes ou parametros diferentes.
""")
