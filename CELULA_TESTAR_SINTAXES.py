# =============================================================================
# TESTAR DIFERENTES SINTAXES PARA CDCMS
# =============================================================================
# NaiveBayes funciona, entao o problema e na especificacao do CDCMS
# =============================================================================

import subprocess
from pathlib import Path
import time

print("="*70)
print("TESTAR DIFERENTES SINTAXES PARA CDCMS")
print("="*70)

WORK_DIR = Path('/content')
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

CDCMS_JAR = CDCMS_JARS_DIR / 'cdcms_cil_final.jar'

# Classpath sem MOA externo
all_jars = list(DEPS_DIR.glob("*.jar"))
non_moa_jars = [j for j in all_jars if 'moa' not in j.name.lower()]
classpath = str(CDCMS_JAR) + ":" + ":".join(str(j) for j in non_moa_jars)

test_arff = TEST_DIR / 'test_imbalanced.arff'

print(f"Arquivo de teste: {test_arff}")

# =============================================================================
# PASSO 1: Testar diferentes sintaxes de classificador
# =============================================================================
print("\n--- PASSO 1: Testar sintaxes de classificador ---")

classifier_syntaxes = [
    # Sintaxe 1: Nome completo sem parenteses
    "moa.classifiers.meta.CDCMS_CIL_GMean",

    # Sintaxe 2: Nome completo com parenteses vazios
    "(moa.classifiers.meta.CDCMS_CIL_GMean)",

    # Sintaxe 3: Com parametros
    "(moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2)",

    # Sintaxe 4: Apenas nome da classe
    "CDCMS_CIL_GMean",

    # Sintaxe 5: Outras variantes CDCMS
    "moa.classifiers.meta.CDCMS_CIL",
    "moa.classifiers.meta.CDCMS_tnnls2020",
]

for i, classifier in enumerate(classifier_syntaxes, 1):
    print(f"\n  Teste {i}: {classifier[:50]}...")

    output_file = TEST_DIR / f'test_syntax_{i}.csv'

    # Task string
    task = f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff}) -l {classifier} -f 1000"

    cmd = ["java", "-Xmx2g", "-cp", classpath, "moa.DoTask", task]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"    [OK] Funciona! Output: {output_file.stat().st_size} bytes")
            break
        elif result.returncode == 0 and not result.stderr:
            # Pode ter funcionado mas sem output file
            print(f"    [OK?] Return code 0, verificar stdout")
            if result.stdout:
                print(f"    {result.stdout[:100]}")
        else:
            # Extrair erro
            error = result.stderr.split('\n')[0] if result.stderr else "Unknown"
            print(f"    [X] {error[:60]}")

    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT]")
    except Exception as e:
        print(f"    [ERRO] {e}")

# =============================================================================
# PASSO 2: Verificar se a classe CDCMS esta registrada
# =============================================================================
print("\n--- PASSO 2: Verificar classes disponiveis ---")

# Listar classes de classificadores no JAR
print("\nClassificadores meta no JAR:")
result = subprocess.run(
    f'jar tf "{CDCMS_JAR}" | grep "classifiers/meta/.*\\.class$" | grep -v "\\$"',
    shell=True, capture_output=True, text=True
)

if result.stdout:
    classes = result.stdout.strip().split('\n')
    for cls in classes[:15]:
        # Converter path para nome de classe
        class_name = cls.replace('/', '.').replace('.class', '')
        print(f"  {class_name}")

# =============================================================================
# PASSO 3: Testar instanciacao direta da classe
# =============================================================================
print("\n--- PASSO 3: Testar instanciacao direta ---")

# Criar um pequeno programa Java para testar
test_java = TEST_DIR / 'TestCDCMS.java'
test_java_content = '''
import moa.classifiers.meta.CDCMS_CIL_GMean;

public class TestCDCMS {
    public static void main(String[] args) {
        try {
            CDCMS_CIL_GMean classifier = new CDCMS_CIL_GMean();
            System.out.println("Classe instanciada com sucesso!");
            System.out.println("Nome: " + classifier.getClass().getName());
            System.out.println("CLI: " + classifier.getCLICreationString(classifier.getClass()));
        } catch (Exception e) {
            System.out.println("Erro: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
'''

with open(test_java, 'w') as f:
    f.write(test_java_content)

# Compilar
print("Compilando TestCDCMS.java...")
compile_result = subprocess.run(
    ["javac", "-cp", classpath, str(test_java)],
    capture_output=True, text=True, timeout=30
)

if compile_result.returncode == 0:
    print("[OK] Compilado")

    # Executar
    print("Executando...")
    run_result = subprocess.run(
        ["java", "-cp", f"{classpath}:{TEST_DIR}", "TestCDCMS"],
        capture_output=True, text=True, timeout=30
    )

    print("Saida:")
    print(run_result.stdout)
    if run_result.stderr:
        print("Erros:")
        print(run_result.stderr[:500])
else:
    print("[ERRO] Compilacao falhou")
    print(compile_result.stderr[:500])

# =============================================================================
# PASSO 4: Verificar interface do classificador
# =============================================================================
print("\n--- PASSO 4: Verificar hierarquia de classes ---")

test_java2 = TEST_DIR / 'CheckHierarchy.java'
test_java2_content = '''
import moa.classifiers.meta.CDCMS_CIL_GMean;
import moa.classifiers.Classifier;
import moa.classifiers.AbstractClassifier;

public class CheckHierarchy {
    public static void main(String[] args) {
        CDCMS_CIL_GMean c = new CDCMS_CIL_GMean();

        System.out.println("CDCMS_CIL_GMean hierarchy:");
        System.out.println("  Is Classifier: " + (c instanceof Classifier));
        System.out.println("  Is AbstractClassifier: " + (c instanceof AbstractClassifier));

        Class<?> cls = c.getClass();
        System.out.println("  Superclass: " + cls.getSuperclass().getName());

        System.out.println("  Interfaces:");
        for (Class<?> iface : cls.getInterfaces()) {
            System.out.println("    - " + iface.getName());
        }
    }
}
'''

with open(test_java2, 'w') as f:
    f.write(test_java2_content)

compile_result2 = subprocess.run(
    ["javac", "-cp", classpath, str(test_java2)],
    capture_output=True, text=True, timeout=30
)

if compile_result2.returncode == 0:
    run_result2 = subprocess.run(
        ["java", "-cp", f"{classpath}:{TEST_DIR}", "CheckHierarchy"],
        capture_output=True, text=True, timeout=30
    )
    print(run_result2.stdout)
else:
    print("[ERRO] Falha ao compilar")
    print(compile_result2.stderr[:300])

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO")
print("="*70)

# Verificar se algum teste funcionou
success = False
for i in range(1, 7):
    output_file = TEST_DIR / f'test_syntax_{i}.csv'
    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"\n[SUCESSO] Sintaxe {i} funcionou!")
        success = True
        break

if not success:
    print("\nNenhuma sintaxe funcionou diretamente.")
    print("Verifique os resultados dos testes de instanciacao acima.")
