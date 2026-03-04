# =============================================================================
# CORRIGIR BUG E COMPILAR CDCMS.CIL
# =============================================================================
# Bug encontrado: O arquivo CDCMS_CIL_GMean_OSUS.java contem uma classe
# chamada CDCMS_GMean_OSUS (falta "_CIL" no nome da classe)
# Solucao: Renomear o arquivo para corresponder ao nome da classe
# =============================================================================

import subprocess
from pathlib import Path
import shutil
import time

print("="*70)
print("CORRIGIR BUG E COMPILAR CDCMS.CIL")
print("="*70)

WORK_DIR = Path('/content')
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'
BUILD_DIR = WORK_DIR / 'cdcms_full_build'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

CDCMS_JARS_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)
BUILD_DIR.mkdir()

CDCMS_REPO = WORK_DIR / 'CDCMS_CIL_src' / 'CDCMS.CIL'
SRC_DIR = CDCMS_REPO / 'Implementation' / 'moa' / 'src' / 'main' / 'java'

# =============================================================================
# PASSO 1: Corrigir o bug - renomear arquivo
# =============================================================================
print("\n--- PASSO 1: Corrigir bug no codigo fonte ---")

problematic_file = SRC_DIR / 'moa' / 'classifiers' / 'meta' / 'CDCMS_CIL_GMean_OSUS.java'
correct_filename = SRC_DIR / 'moa' / 'classifiers' / 'meta' / 'CDCMS_GMean_OSUS.java'

if problematic_file.exists():
    print(f"Arquivo com bug: {problematic_file.name}")

    # Verificar se ja foi renomeado
    if correct_filename.exists():
        print(f"[OK] Arquivo ja foi corrigido anteriormente")
    else:
        # Renomear
        shutil.copy(problematic_file, correct_filename)
        print(f"[OK] Copiado para: {correct_filename.name}")

        # Remover arquivo original para evitar conflito
        problematic_file.unlink()
        print(f"[OK] Arquivo original removido")
elif correct_filename.exists():
    print(f"[OK] Arquivo ja corrigido: {correct_filename.name}")
else:
    print("[AVISO] Arquivo nao encontrado")

# =============================================================================
# PASSO 2: Compilar
# =============================================================================
print("\n--- PASSO 2: Compilar ---")

all_jars = list(DEPS_DIR.glob("*.jar"))
classpath = ":".join(str(j) for j in all_jars)

print(f"JARs no classpath: {len(all_jars)}")

all_java = list(SRC_DIR.rglob("*.java"))
print(f"Arquivos Java: {len(all_java)}")

sources_file = BUILD_DIR / 'sources.txt'
with open(sources_file, 'w') as f:
    for jf in all_java:
        f.write(str(jf) + '\n')

print("Compilando...")

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
compile_time = time.time() - start

print(f"Tempo: {compile_time:.1f}s")
print(f"Return code: {result.returncode}")

class_files = list(BUILD_DIR.rglob("*.class"))
print(f"Classes geradas: {len(class_files)}")

if result.returncode != 0:
    print("\n[ERRO] Compilacao falhou")
    print(result.stderr[:1000])
else:
    print("\n[OK] Compilacao bem-sucedida!")

    # Contar classes CDCMS
    cdcms_classes = [c for c in class_files if 'CDCMS' in c.name]
    print(f"\nClasses CDCMS: {len(cdcms_classes)}")
    for c in cdcms_classes:
        print(f"  {c.name}")

# =============================================================================
# PASSO 3: Criar JAR
# =============================================================================
print("\n--- PASSO 3: Criar JAR ---")

if class_files:
    cdcms_jar = CDCMS_JARS_DIR / 'cdcms_cil_final.jar'

    jar_cmd = ["jar", "cf", str(cdcms_jar), "-C", str(BUILD_DIR), "."]
    subprocess.run(jar_cmd, capture_output=True, timeout=120)

    if cdcms_jar.exists():
        print(f"[OK] JAR criado: {cdcms_jar.name}")
        print(f"     Tamanho: {cdcms_jar.stat().st_size/(1024*1024):.1f} MB")
else:
    cdcms_jar = None
    print("[SKIP] Nenhuma classe para empacotar")

# =============================================================================
# PASSO 4: Testar
# =============================================================================
print("\n--- PASSO 4: Testar ---")

if cdcms_jar and cdcms_jar.exists():
    test_cp = f"{cdcms_jar}:{classpath}"

    # Teste 1: moa.DoTask
    print("\n  Teste 1: moa.DoTask")
    test1 = subprocess.run(
        ["java", "-Xmx2g", "-cp", test_cp, "moa.DoTask"],
        capture_output=True, text=True, timeout=30
    )

    if "Usage" in test1.stdout or "tasks" in test1.stdout.lower():
        print("    [OK] moa.DoTask funciona!")
    else:
        print("    [INFO] moa.DoTask retornou:")
        print(f"    {(test1.stdout + test1.stderr)[:200]}")

    # Teste 2: WriteCommandLineTemplate
    print("\n  Teste 2: WriteCommandLineTemplate para CDCMS_CIL_GMean")
    test2 = subprocess.run(
        ["java", "-Xmx2g", "-cp", test_cp, "moa.DoTask",
         "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean"],
        capture_output=True, text=True, timeout=30
    )

    output2 = test2.stdout + test2.stderr

    if "Exception" not in output2 and test2.returncode == 0:
        print("    [OK] CDCMS_CIL_GMean reconhecido!")
        for line in output2.strip().split('\n')[:3]:
            if line.strip():
                print(f"    {line[:70]}")
    else:
        print("    [FALHA]")
        for line in output2.strip().split('\n')[:5]:
            print(f"    {line[:80]}")

    # Teste 3: Execucao real
    print("\n  Teste 3: EvaluateInterleavedTestThenTrain")

    # Criar arquivo de teste
    test_arff = TEST_DIR / 'test_data.arff'
    with open(test_arff, 'w') as f:
        f.write("@relation test\n@attribute a1 numeric\n@attribute a2 numeric\n@attribute class {0,1}\n@data\n")
        import random
        random.seed(42)
        for _ in range(2000):
            a1 = random.gauss(0, 1) + (0 if random.random() < 0.9 else 2)
            a2 = random.gauss(0, 1)
            cls = 0 if random.random() < 0.9 else 1
            f.write(f"{a1:.4f},{a2:.4f},{cls}\n")

    output_file = TEST_DIR / 'cdcms_final_output.csv'

    task_string = f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff}) -l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2) -f 500 -d {output_file}"

    print("    Executando (pode demorar)...")
    start = time.time()

    test3 = subprocess.run(
        ["java", "-Xmx4g", "-cp", test_cp, "moa.DoTask", task_string],
        capture_output=True, text=True, timeout=300
    )

    duration = time.time() - start
    print(f"    Tempo: {duration:.1f}s")

    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"    [SUCESSO] Output: {output_file.stat().st_size} bytes")

        # Mostrar conteudo
        with open(output_file) as f:
            lines = f.readlines()
        print(f"    Linhas: {len(lines)}")
        if lines:
            print(f"    Header: {lines[0].strip()[:60]}...")
            if len(lines) > 1:
                print(f"    Dados: {lines[1].strip()[:60]}...")
    else:
        print("    [FALHA] Arquivo nao criado")
        output3 = test3.stdout + test3.stderr
        for line in output3.strip().split('\n')[:10]:
            print(f"    {line[:80]}")

# =============================================================================
# RESUMO FINAL
# =============================================================================
print("\n" + "="*70)
print("RESUMO FINAL")
print("="*70)

output_file = TEST_DIR / 'cdcms_final_output.csv'

if output_file.exists() and output_file.stat().st_size > 0:
    print("\n" + "*"*50)
    print("*** SUCESSO! CDCMS.CIL ESTA FUNCIONANDO! ***")
    print("*"*50)
    print(f"\nJAR: {cdcms_jar}")
    print(f"Output: {output_file}")
    print(f"\nPara usar no seu experimento:")
    print(f'  CDCMS_JAR = "{cdcms_jar}"')
    print(f'  DEPS_DIR = "{DEPS_DIR}"')
else:
    class_files = list(BUILD_DIR.rglob("*.class"))
    print(f"\nClasses compiladas: {len(class_files)}")

    if class_files:
        print("\n[STATUS] Compilacao OK, mas execucao falhou")
        print("Verifique os erros acima")
    else:
        print("\n[STATUS] Compilacao ainda falhou")
