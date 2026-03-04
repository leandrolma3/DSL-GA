# =============================================================================
# CELULA 3.2: DIAGNÓSTICO AVANÇADO - Múltiplas Abordagens
# =============================================================================
# Testa diferentes formatos de comando para identificar o que funciona
# =============================================================================

import subprocess
import time
from pathlib import Path

print("="*70)
print("DIAGNÓSTICO AVANÇADO - CDCMS.CIL")
print("="*70)

# Caminhos
test_arff_abs = str(test_arff.resolve())
diag_dir = TEST_DIR / 'diagnostico'
diag_dir.mkdir(exist_ok=True)

# Construir classpath (ORDEM: cdcms -> weka/lib -> moa -> deps)
classpath_parts = []
if CDCMS_JAR.exists():
    classpath_parts.append(str(CDCMS_JAR))
if MOA_LIB_DIR.exists():
    for jar in sorted(MOA_LIB_DIR.glob('*.jar')):
        classpath_parts.append(str(jar))
if MOA_JAR.exists():
    classpath_parts.append(str(MOA_JAR))
# NÃO incluir deps/ para evitar conflitos
# deps_dir = CDCMS_MOA_DIR / 'deps'
# if deps_dir.exists():
#     for jar in sorted(deps_dir.glob('*.jar')):
#         classpath_parts.append(str(jar))

full_classpath = ':'.join(classpath_parts)
print(f"Classpath: {len(classpath_parts)} JARs (SEM deps/ para evitar conflitos)")

# =============================================================================
# TESTE 1: Comando como STRING ÚNICA (formato oficial MOA)
# =============================================================================
print("\n" + "="*70)
print("TESTE 1: Comando como string única (formato oficial MOA)")
print("="*70)

output_1 = diag_dir / 'test1_output.csv'

# Formato oficial: java -cp ... moa.DoTask "Task -l learner -s stream ..."
task_str = f'EvaluateInterleavedTestThenTrain -l trees.HoeffdingTree -s (ArffFileStream -f {test_arff_abs}) -f 500 -d {output_1}'

cmd_1 = f'java -Xmx4g -cp "{full_classpath}" moa.DoTask "{task_str}"'

print(f"Comando: java ... moa.DoTask \"{task_str[:50]}...\"")
start = time.time()
result = subprocess.run(cmd_1, shell=True, capture_output=True, text=True, timeout=60)
duration = time.time() - start

print(f"Tempo: {duration:.1f}s, Return: {result.returncode}")
if result.stderr.strip():
    print(f"STDERR: {result.stderr[:200]}")
print(f"Output: {'[OK]' if output_1.exists() and output_1.stat().st_size > 0 else '[X]'}")

# =============================================================================
# TESTE 2: CDCMS_CIL base (sem GMean) - String única
# =============================================================================
print("\n" + "="*70)
print("TESTE 2: CDCMS_CIL base (sem GMean)")
print("="*70)

output_2 = diag_dir / 'test2_output.csv'

task_str = f'EvaluateInterleavedTestThenTrain -l (meta.CDCMS_CIL -s 10 -t 500) -s (ArffFileStream -f {test_arff_abs}) -f 500 -d {output_2}'

cmd_2 = f'java -Xmx4g -cp "{full_classpath}" moa.DoTask "{task_str}"'

print(f"Comando: ... CDCMS_CIL (sem GMean) ...")
start = time.time()
result = subprocess.run(cmd_2, shell=True, capture_output=True, text=True, timeout=120)
duration = time.time() - start

print(f"Tempo: {duration:.1f}s, Return: {result.returncode}")
if result.stderr.strip():
    print(f"STDERR: {result.stderr[:300]}")
print(f"Output: {'[OK]' if output_2.exists() and output_2.stat().st_size > 0 else '[X]'}")

# =============================================================================
# TESTE 3: CDCMS_CIL_GMean - String única com nome curto
# =============================================================================
print("\n" + "="*70)
print("TESTE 3: CDCMS_CIL_GMean com nome curto")
print("="*70)

output_3 = diag_dir / 'test3_output.csv'

# Usar nome curto: meta.CDCMS_CIL_GMean (sem moa.classifiers.)
task_str = f'EvaluateInterleavedTestThenTrain -l (meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2) -s (ArffFileStream -f {test_arff_abs}) -f 500 -d {output_3}'

cmd_3 = f'java -Xmx4g -cp "{full_classpath}" moa.DoTask "{task_str}"'

print(f"Comando: ... meta.CDCMS_CIL_GMean ...")
start = time.time()
result = subprocess.run(cmd_3, shell=True, capture_output=True, text=True, timeout=120)
duration = time.time() - start

print(f"Tempo: {duration:.1f}s, Return: {result.returncode}")
if result.stderr.strip():
    print(f"STDERR: {result.stderr[:300]}")
print(f"Output: {'[OK]' if output_3.exists() and output_3.stat().st_size > 0 else '[X]'}")

# =============================================================================
# TESTE 4: Verificar se o problema é AutoClassDiscovery
# =============================================================================
print("\n" + "="*70)
print("TESTE 4: Verificar AutoClassDiscovery")
print("="*70)

# Tentar carregar a classe diretamente
check_cmd = f'''java -Xmx2g -cp "{full_classpath}" -c "
public class Test {{
    public static void main(String[] args) {{
        try {{
            Class<?> cls = Class.forName(\\"moa.classifiers.meta.CDCMS_CIL_GMean\\");
            Object obj = cls.getDeclaredConstructor().newInstance();
            System.out.println(\\"OK: Classe instanciada com sucesso\\");
        }} catch (Exception e) {{
            System.out.println(\\"ERRO: \\" + e.getMessage());
            e.printStackTrace();
        }}
    }}
}}"
'''

# Alternativa: usar jshell
jshell_cmd = f'''echo "
import moa.classifiers.meta.CDCMS_CIL_GMean;
try {{
    CDCMS_CIL_GMean learner = new CDCMS_CIL_GMean();
    System.out.println(\\"OK: Classe instanciada\\");
}} catch (Exception e) {{
    System.out.println(\\"ERRO: \\" + e);
    e.printStackTrace();
}}
/exit
" | jshell --class-path "{full_classpath}" 2>&1 | head -20
'''

print("Tentando instanciar CDCMS_CIL_GMean via jshell...")
result = subprocess.run(jshell_cmd, shell=True, capture_output=True, text=True, timeout=30)
print(result.stdout[:500] if result.stdout else "Sem output")
if result.stderr:
    print(f"STDERR: {result.stderr[:200]}")

# =============================================================================
# TESTE 5: CDCMS_CIL_GMean com clusterer EM explícito
# =============================================================================
print("\n" + "="*70)
print("TESTE 5: CDCMS_CIL_GMean com clusterer explícito")
print("="*70)

output_5 = diag_dir / 'test5_output.csv'

# Tentar passar o clusterer explicitamente (-w 0 para EM)
task_str = f'EvaluateInterleavedTestThenTrain -l (meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2 -w 0) -s (ArffFileStream -f {test_arff_abs}) -f 500 -d {output_5}'

cmd_5 = f'java -Xmx4g -cp "{full_classpath}" moa.DoTask "{task_str}"'

print(f"Comando: ... CDCMS_CIL_GMean -w 0 (EM) ...")
start = time.time()
result = subprocess.run(cmd_5, shell=True, capture_output=True, text=True, timeout=120)
duration = time.time() - start

print(f"Tempo: {duration:.1f}s, Return: {result.returncode}")
if result.stderr.strip():
    print(f"STDERR: {result.stderr[:300]}")
print(f"Output: {'[OK]' if output_5.exists() and output_5.stat().st_size > 0 else '[X]'}")

# =============================================================================
# TESTE 6: WriteCommandLineTemplate para ver parâmetros
# =============================================================================
print("\n" + "="*70)
print("TESTE 6: Ver template de parâmetros do CDCMS_CIL_GMean")
print("="*70)

template_cmd = f'java -Xmx2g -cp "{full_classpath}" moa.DoTask "WriteCommandLineTemplate -l meta.CDCMS_CIL_GMean" 2>&1'

result = subprocess.run(template_cmd, shell=True, capture_output=True, text=True, timeout=30)
print("Template de parâmetros:")
if result.stdout:
    for line in result.stdout.split('\n')[:15]:
        if line.strip():
            print(f"  {line}")
if result.stderr:
    print(f"STDERR: {result.stderr[:300]}")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO DOS TESTES")
print("="*70)

tests = [
    ("Teste 1 - HoeffdingTree", output_1),
    ("Teste 2 - CDCMS_CIL base", output_2),
    ("Teste 3 - CDCMS_CIL_GMean curto", output_3),
    ("Teste 5 - CDCMS_CIL_GMean + EM", output_5),
]

for name, path in tests:
    if path.exists() and path.stat().st_size > 0:
        print(f"[OK] {name}: {path.stat().st_size} bytes")
    else:
        print(f"[X] {name}")
