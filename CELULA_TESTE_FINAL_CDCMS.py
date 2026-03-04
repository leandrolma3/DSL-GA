# =============================================================================
# CÉLULA DE TESTE FINAL - CDCMS.CIL
# =============================================================================
# Baseado na análise do projeto ROSE (similar ao CDCMS.CIL):
# - ROSE usa apenas 2 JARs: ROSE-1.0.jar + MOA-dependencies.jar
# - Nós temos 53-182 JARs causando possíveis conflitos
#
# Este script testa múltiplas abordagens em paralelo
# =============================================================================

import subprocess
import time
from pathlib import Path

print("="*70)
print("TESTE FINAL - CDCMS.CIL (Baseado em análise do ROSE)")
print("="*70)

# Caminhos
test_arff_abs = str(test_arff.resolve())
final_dir = TEST_DIR / 'final_tests'
final_dir.mkdir(exist_ok=True)

# Localizar sizeofag no lib/
sizeofag_jar = None
if MOA_LIB_DIR.exists():
    sizeofag_list = list(MOA_LIB_DIR.glob('sizeofag*.jar'))
    if sizeofag_list:
        sizeofag_jar = sizeofag_list[0]
        print(f"sizeofag encontrado: {sizeofag_jar.name}")

# Localizar moa-release no lib/ (JAR principal que contém tudo)
moa_full_jar = None
if MOA_LIB_DIR.exists():
    # Procurar por moa-release ou moa-2018
    for pattern in ['moa-release*.jar', 'moa-2018*.jar', 'moa*.jar']:
        found = list(MOA_LIB_DIR.glob(pattern))
        for jar in found:
            # Verificar se é o JAR grande (com dependências)
            if jar.stat().st_size > 10*1024*1024:  # > 10MB
                moa_full_jar = jar
                break
        if moa_full_jar:
            break

if moa_full_jar:
    print(f"MOA full jar: {moa_full_jar.name} ({moa_full_jar.stat().st_size/(1024*1024):.1f} MB)")
else:
    print("MOA full jar não encontrado, usando moa.jar padrão")

# =============================================================================
# ABORDAGEM 1: Classpath mínimo (apenas cdcms + moa principal)
# =============================================================================
print("\n" + "="*70)
print("ABORDAGEM 1: Classpath MÍNIMO (2 JARs)")
print("="*70)

output_1 = final_dir / 'approach1_output.csv'

# Usar apenas cdcms_cil.jar e o MOA mais completo disponível
if moa_full_jar:
    cp_1 = f"{CDCMS_JAR}:{moa_full_jar}"
else:
    # Fallback: usar todos os JARs do lib/ mas NÃO o deps/
    lib_jars = ':'.join(str(j) for j in MOA_LIB_DIR.glob('*.jar'))
    cp_1 = f"{CDCMS_JAR}:{lib_jars}"

# Comando no formato ROSE (string única com aspas)
cmd_1 = f'''java -Xmx4g -cp "{cp_1}" moa.DoTask "EvaluateInterleavedTestThenTrain -l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2) -s (ArffFileStream -f {test_arff_abs}) -f 500 -d {output_1}"'''

print(f"Classpath: {cp_1.count(':') + 1} JARs")
print("Executando...")

start = time.time()
result = subprocess.run(cmd_1, shell=True, capture_output=True, text=True, timeout=180)
duration = time.time() - start

print(f"Tempo: {duration:.1f}s, Return: {result.returncode}")
if result.stderr.strip():
    # Filtrar apenas erros relevantes
    for line in result.stderr.split('\n')[:5]:
        if 'Exception' in line or 'Error' in line:
            print(f"  {line[:80]}")
print(f"Output: {'[OK] ' + str(output_1.stat().st_size) + ' bytes' if output_1.exists() and output_1.stat().st_size > 0 else '[X]'}")

# =============================================================================
# ABORDAGEM 2: Com sizeofag (como ROSE faz)
# =============================================================================
print("\n" + "="*70)
print("ABORDAGEM 2: Com -javaagent:sizeofag (como ROSE)")
print("="*70)

output_2 = final_dir / 'approach2_output.csv'

if sizeofag_jar:
    cmd_2 = f'''java -javaagent:{sizeofag_jar} -Xmx4g -cp "{cp_1}" moa.DoTask "EvaluateInterleavedTestThenTrain -l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2) -s (ArffFileStream -f {test_arff_abs}) -f 500 -d {output_2}"'''

    print("Com sizeofag agent...")
    start = time.time()
    result = subprocess.run(cmd_2, shell=True, capture_output=True, text=True, timeout=180)
    duration = time.time() - start

    print(f"Tempo: {duration:.1f}s, Return: {result.returncode}")
    if result.stderr.strip():
        for line in result.stderr.split('\n')[:5]:
            if 'Exception' in line or 'Error' in line:
                print(f"  {line[:80]}")
    print(f"Output: {'[OK] ' + str(output_2.stat().st_size) + ' bytes' if output_2.exists() and output_2.stat().st_size > 0 else '[X]'}")
else:
    print("[SKIP] sizeofag não encontrado")

# =============================================================================
# ABORDAGEM 3: Usar CDCMS_CIL base (sem GMean) - mais simples
# =============================================================================
print("\n" + "="*70)
print("ABORDAGEM 3: CDCMS_CIL BASE (sem GMean)")
print("="*70)

output_3 = final_dir / 'approach3_output.csv'

cmd_3 = f'''java -Xmx4g -cp "{cp_1}" moa.DoTask "EvaluateInterleavedTestThenTrain -l (moa.classifiers.meta.CDCMS_CIL -s 10 -t 500) -s (ArffFileStream -f {test_arff_abs}) -f 500 -d {output_3}"'''

print("CDCMS_CIL (versão base, sem GMean)...")
start = time.time()
result = subprocess.run(cmd_3, shell=True, capture_output=True, text=True, timeout=180)
duration = time.time() - start

print(f"Tempo: {duration:.1f}s, Return: {result.returncode}")
if result.stderr.strip():
    for line in result.stderr.split('\n')[:5]:
        if 'Exception' in line or 'Error' in line:
            print(f"  {line[:80]}")
print(f"Output: {'[OK] ' + str(output_3.stat().st_size) + ' bytes' if output_3.exists() and output_3.stat().st_size > 0 else '[X]'}")

# =============================================================================
# ABORDAGEM 4: Verificar se o problema é na instanciação
# =============================================================================
print("\n" + "="*70)
print("ABORDAGEM 4: Testar instanciação direta via Java")
print("="*70)

# Criar um pequeno programa Java para testar
test_java = final_dir / 'TestCDCMS.java'
test_java.write_text('''
import moa.classifiers.meta.CDCMS_CIL_GMean;
import moa.classifiers.meta.CDCMS_CIL;

public class TestCDCMS {
    public static void main(String[] args) {
        System.out.println("Testando instanciacao...");

        try {
            System.out.println("1. Tentando CDCMS_CIL...");
            CDCMS_CIL base = new CDCMS_CIL();
            System.out.println("   [OK] CDCMS_CIL instanciado");
        } catch (Exception e) {
            System.out.println("   [ERRO] " + e.getClass().getName() + ": " + e.getMessage());
        }

        try {
            System.out.println("2. Tentando CDCMS_CIL_GMean...");
            CDCMS_CIL_GMean gmean = new CDCMS_CIL_GMean();
            System.out.println("   [OK] CDCMS_CIL_GMean instanciado");
        } catch (Exception e) {
            System.out.println("   [ERRO] " + e.getClass().getName() + ": " + e.getMessage());
            e.printStackTrace();
        }
    }
}
''')

# Compilar
compile_cmd = f'cd "{final_dir}" && javac -cp "{cp_1}" TestCDCMS.java 2>&1'
compile_result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True, timeout=30)

if (final_dir / 'TestCDCMS.class').exists():
    print("Compilação OK, executando teste...")
    run_cmd = f'cd "{final_dir}" && java -cp ".:{cp_1}" TestCDCMS 2>&1'
    run_result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True, timeout=30)
    print(run_result.stdout)
    if run_result.stderr:
        print(f"STDERR: {run_result.stderr[:300]}")
else:
    print(f"Erro de compilação: {compile_result.stdout[:200]}")

# =============================================================================
# ABORDAGEM 5: Usar GUI do MOA para gerar comando correto
# =============================================================================
print("\n" + "="*70)
print("ABORDAGEM 5: Listar classificadores disponíveis")
print("="*70)

# Listar todos os classificadores meta disponíveis
list_cmd = f'''java -Xmx2g -cp "{cp_1}" moa.DoTask "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL" 2>&1 | head -20'''
result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True, timeout=30)
print("Template CDCMS_CIL:")
if result.stdout:
    for line in result.stdout.split('\n')[:10]:
        print(f"  {line}")

# =============================================================================
# ABORDAGEM 6: Verificar se há problema com o MOA compilado no cdcms_cil.jar
# =============================================================================
print("\n" + "="*70)
print("ABORDAGEM 6: Verificar versão do MOA no cdcms_cil.jar")
print("="*70)

# Verificar se há classes DoTask duplicadas
check_cmd = f'jar tf "{CDCMS_JAR}" 2>/dev/null | grep "moa/DoTask.class"'
result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
if result.stdout.strip():
    print(f"[AVISO] cdcms_cil.jar contém moa/DoTask.class!")
    print("Isso pode causar conflitos. Removendo...")

    # Criar versão limpa do JAR (sem classes MOA conflitantes)
    clean_jar = CDCMS_JARS_DIR / 'cdcms_cil_clean.jar'
    extract_dir = final_dir / 'cdcms_extract'
    extract_dir.mkdir(exist_ok=True)

    # Extrair
    subprocess.run(f'cd "{extract_dir}" && jar xf "{CDCMS_JAR}"', shell=True)

    # Remover pastas MOA duplicadas (manter apenas meta/)
    moa_dirs_to_remove = ['moa/tasks', 'moa/core', 'moa/options', 'moa/streams', 'moa/DoTask.class']
    for dir_name in moa_dirs_to_remove:
        path = extract_dir / dir_name
        if path.exists():
            subprocess.run(f'rm -rf "{path}"', shell=True)

    # Recriar JAR limpo
    subprocess.run(f'cd "{extract_dir}" && jar cf "{clean_jar}" .', shell=True)

    if clean_jar.exists():
        print(f"[OK] JAR limpo criado: {clean_jar.name} ({clean_jar.stat().st_size/(1024*1024):.1f} MB)")

        # Testar com JAR limpo
        print("\nTestando com JAR limpo...")
        output_6 = final_dir / 'approach6_output.csv'

        lib_jars = ':'.join(str(j) for j in MOA_LIB_DIR.glob('*.jar'))
        cp_6 = f"{clean_jar}:{lib_jars}"

        cmd_6 = f'''java -Xmx4g -cp "{cp_6}" moa.DoTask "EvaluateInterleavedTestThenTrain -l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2) -s (ArffFileStream -f {test_arff_abs}) -f 500 -d {output_6}"'''

        result = subprocess.run(cmd_6, shell=True, capture_output=True, text=True, timeout=180)
        print(f"Return: {result.returncode}")
        if result.stderr.strip():
            for line in result.stderr.split('\n')[:3]:
                if 'Exception' in line:
                    print(f"  {line[:80]}")
        print(f"Output: {'[OK]' if output_6.exists() and output_6.stat().st_size > 0 else '[X]'}")
else:
    print("[OK] cdcms_cil.jar NÃO contém moa/DoTask.class")

# =============================================================================
# RESUMO FINAL
# =============================================================================
print("\n" + "="*70)
print("RESUMO FINAL")
print("="*70)

outputs = [
    ("Abordagem 1 - Classpath mínimo", output_1),
    ("Abordagem 2 - Com sizeofag", output_2),
    ("Abordagem 3 - CDCMS_CIL base", output_3),
]

if 'output_6' in dir():
    outputs.append(("Abordagem 6 - JAR limpo", output_6))

sucesso = None
for name, path in outputs:
    if path.exists() and path.stat().st_size > 0:
        print(f"[OK] {name}: {path.stat().st_size} bytes")
        if sucesso is None:
            sucesso = (name, path)
    else:
        print(f"[X] {name}")

if sucesso:
    print(f"\n*** SUCESSO: {sucesso[0]} ***")
    import shutil
    shutil.copy(sucesso[1], output_file)
    print(f"Copiado para: {output_file}")
else:
    print("\n[ERRO] Nenhuma abordagem funcionou!")
    print("\nPróximos passos sugeridos:")
    print("  1. Verificar se há conflito de versões MOA no cdcms_cil.jar")
    print("  2. Recompilar CDCMS.CIL sem incluir classes MOA")
    print("  3. Contatar autores do CDCMS.CIL para JAR funcional")
