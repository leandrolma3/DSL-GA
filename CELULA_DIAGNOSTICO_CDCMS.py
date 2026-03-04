# =============================================================================
# CELULA DE DIAGNÓSTICO DETALHADO PARA CDCMS.CIL
# =============================================================================
# Execute esta célula APÓS a célula 3.1 (criação do dataset)
# Objetivo: Identificar exatamente onde está o problema
# =============================================================================

import subprocess
import time
from pathlib import Path

print("="*70)
print("DIAGNÓSTICO DETALHADO - CDCMS.CIL_GMean")
print("="*70)

# Usar caminhos absolutos
test_arff_abs = str(test_arff.resolve())
output_base = TEST_DIR / 'diag'
output_base.mkdir(exist_ok=True)

# =============================================================================
# TESTE 1: Verificar se a classe CDCMS_CIL_GMean pode ser carregada
# =============================================================================
print("\n" + "="*70)
print("TESTE 1: Carregamento da classe CDCMS_CIL_GMean")
print("="*70)

# Construir classpath (cdcms_cil.jar PRIMEIRO)
classpath_parts = [str(CDCMS_JAR)]
deps_dir = CDCMS_MOA_DIR / 'deps'
if deps_dir.exists():
    for jar in sorted(deps_dir.glob('*.jar')):
        classpath_parts.append(str(jar))
if MOA_JAR.exists():
    classpath_parts.append(str(MOA_JAR))

full_classpath = ':'.join(classpath_parts)
print(f"Classpath: {len(classpath_parts)} JARs (cdcms_cil.jar primeiro)")

# Tentar listar opções do classificador
test_cmd = [
    "java", "-Xmx2g", "-cp", full_classpath,
    "moa.DoTask", "WriteCommandLineTemplate",
    "-l", "moa.classifiers.meta.CDCMS_CIL_GMean"
]

print(f"\nComando: java ... moa.DoTask WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean")
result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)

print(f"Return code: {result.returncode}")
if result.stdout.strip():
    print(f"STDOUT:\n{result.stdout[:1000]}")
if result.stderr.strip():
    print(f"STDERR:\n{result.stderr[:1000]}")

# =============================================================================
# TESTE 2: Usar nome COMPLETO da tarefa
# =============================================================================
print("\n" + "="*70)
print("TESTE 2: Usar nome completo da tarefa (moa.tasks.EvaluateInterleavedTestThenTrain)")
print("="*70)

output_file_2 = output_base / 'test2_output.csv'

# CORREÇÃO: Usar nome completo da tarefa
learner = "moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500"
stream = f"moa.streams.ArffFileStream -f {test_arff_abs}"

# Usar formato com aspas
task_str = f'"moa.tasks.EvaluateInterleavedTestThenTrain -l ({learner}) -s ({stream}) -f 500 -d {output_file_2}"'

cmd_str = f'java -Xmx4g -cp "{full_classpath}" moa.DoTask {task_str}'
print(f"\nComando:\n{cmd_str[:200]}...")

result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, timeout=120)

print(f"\nReturn code: {result.returncode}")
if result.stdout.strip():
    print(f"STDOUT:\n{result.stdout[:500]}")
if result.stderr.strip():
    print(f"STDERR:\n{result.stderr[:500]}")

if output_file_2.exists():
    print(f"\n[OK] Arquivo criado: {output_file_2.stat().st_size} bytes")
else:
    print(f"\n[X] Arquivo NÃO criado")

# =============================================================================
# TESTE 3: Tentar com CDCMS_CIL (sem GMean)
# =============================================================================
print("\n" + "="*70)
print("TESTE 3: Testar CDCMS_CIL (versão base, sem GMean)")
print("="*70)

output_file_3 = output_base / 'test3_output.csv'

learner_base = "moa.classifiers.meta.CDCMS_CIL -s 10 -t 500"
task_str_3 = f'"moa.tasks.EvaluateInterleavedTestThenTrain -l ({learner_base}) -s ({stream}) -f 500 -d {output_file_3}"'

cmd_str_3 = f'java -Xmx4g -cp "{full_classpath}" moa.DoTask {task_str_3}'
print(f"\nComando:\n{cmd_str_3[:200]}...")

result = subprocess.run(cmd_str_3, shell=True, capture_output=True, text=True, timeout=120)

print(f"\nReturn code: {result.returncode}")
if result.stderr.strip():
    print(f"STDERR:\n{result.stderr[:500]}")

if output_file_3.exists():
    print(f"\n[OK] Arquivo criado: {output_file_3.stat().st_size} bytes")
else:
    print(f"\n[X] Arquivo NÃO criado")

# =============================================================================
# TESTE 4: Verificar dependências do CDCMS_CIL_GMean
# =============================================================================
print("\n" + "="*70)
print("TESTE 4: Verificar imports/dependências do CDCMS_CIL_GMean")
print("="*70)

# Usar javap para ver as dependências
javap_cmd = f'javap -cp "{full_classpath}" -verbose moa.classifiers.meta.CDCMS_CIL_GMean 2>&1 | head -50'
result = subprocess.run(javap_cmd, shell=True, capture_output=True, text=True, timeout=30)

if result.stdout.strip():
    print("Informações da classe:")
    for line in result.stdout.split('\n')[:30]:
        if line.strip():
            print(f"  {line}")

# =============================================================================
# TESTE 5: Verificar se há conflito de versões do MOA
# =============================================================================
print("\n" + "="*70)
print("TESTE 5: Verificar JARs com classes MOA duplicadas")
print("="*70)

# Procurar JARs que contenham moa.DoTask
print("JARs contendo 'moa/DoTask':")
for jar_path in classpath_parts[:20]:  # Primeiros 20
    jar = Path(jar_path)
    if jar.exists():
        check = subprocess.run(
            f'jar tf "{jar}" 2>/dev/null | grep "moa/DoTask"',
            shell=True, capture_output=True, text=True
        )
        if check.stdout.strip():
            print(f"  [!] {jar.name}")

# =============================================================================
# TESTE 6: Testar com formato de comando diferente (sem parênteses externos)
# =============================================================================
print("\n" + "="*70)
print("TESTE 6: Formato alternativo do comando")
print("="*70)

output_file_6 = output_base / 'test6_output.csv'

# Formato: passar task como string simples
cmd_6 = [
    "java", "-Xmx4g", "-cp", full_classpath,
    "moa.DoTask",
    "moa.tasks.EvaluateInterleavedTestThenTrain",
    "-l", f"(moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500)",
    "-s", f"(moa.streams.ArffFileStream -f {test_arff_abs})",
    "-f", "500",
    "-d", str(output_file_6)
]

print(f"Comando: java ... moa.DoTask moa.tasks.EvaluateInterleavedTestThenTrain -l (...) -s (...) -f 500 -d ...")

result = subprocess.run(cmd_6, capture_output=True, text=True, timeout=120)

print(f"\nReturn code: {result.returncode}")
if result.stderr.strip():
    print(f"STDERR:\n{result.stderr[:500]}")

if output_file_6.exists():
    print(f"\n[OK] Arquivo criado: {output_file_6.stat().st_size} bytes")
else:
    print(f"\n[X] Arquivo NÃO criado")

# =============================================================================
# TESTE 7: Verificar se há problema com o número de classes
# =============================================================================
print("\n" + "="*70)
print("TESTE 7: Testar CDCMS_CIL_GMean com parâmetro numClasses")
print("="*70)

output_file_7 = output_base / 'test7_output.csv'

# CDCMS_CIL_GMean tem parâmetro numClasses
cmd_7 = [
    "java", "-Xmx4g", "-cp", full_classpath,
    "moa.DoTask",
    "moa.tasks.EvaluateInterleavedTestThenTrain",
    "-l", f"(moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2)",  # -c para numClasses
    "-s", f"(moa.streams.ArffFileStream -f {test_arff_abs})",
    "-f", "500",
    "-d", str(output_file_7)
]

print(f"Comando: ... CDCMS_CIL_GMean -s 10 -t 500 -c 2 ...")

result = subprocess.run(cmd_7, capture_output=True, text=True, timeout=120)

print(f"\nReturn code: {result.returncode}")
if result.stderr.strip():
    print(f"STDERR:\n{result.stderr[:500]}")

if output_file_7.exists():
    print(f"\n[OK] Arquivo criado: {output_file_7.stat().st_size} bytes")
    # Mostrar conteúdo
    with open(output_file_7) as f:
        lines = f.readlines()
    print(f"Linhas: {len(lines)}")
    if lines:
        print(f"Header: {lines[0].strip()[:100]}")
else:
    print(f"\n[X] Arquivo NÃO criado")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO DOS TESTES")
print("="*70)

tests = [
    ("Teste 1 - Carregamento classe", "Ver output acima"),
    ("Teste 2 - Nome completo tarefa", output_file_2),
    ("Teste 3 - CDCMS_CIL base", output_file_3),
    ("Teste 6 - Formato alternativo", output_file_6),
    ("Teste 7 - Com numClasses", output_file_7),
]

for name, path in tests:
    if isinstance(path, Path):
        status = "[OK]" if path.exists() and path.stat().st_size > 0 else "[X]"
        size = f"({path.stat().st_size} bytes)" if path.exists() else ""
        print(f"{status} {name} {size}")
    else:
        print(f"[?] {name}: {path}")
