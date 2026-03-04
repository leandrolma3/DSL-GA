# =============================================================================
# TESTAR CDCMS.CIL - USANDO CODIGO JAVA DIRETO
# =============================================================================
# IMPORTANTE: moa.DoTask NAO funciona com CDCMS (bug de parsing)
# Solucao: usar codigo Java direto que instancia a classe
# =============================================================================

import subprocess
from pathlib import Path
import time
import random

print("="*70)
print("TESTAR CDCMS.CIL")
print("="*70)

WORK_DIR = Path('/content')
CDCMS_JAR = WORK_DIR / 'cdcms_jars' / 'cdcms_cil_final.jar'
MOA_DEPS_JAR = WORK_DIR / 'rose_jars' / 'MOA-dependencies.jar'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

TEST_DIR.mkdir(exist_ok=True)

# =============================================================================
# PASSO 1: Verificar JARs
# =============================================================================
print("\n--- PASSO 1: Verificar JARs ---")

if CDCMS_JAR.exists():
    print(f"[OK] CDCMS JAR: {CDCMS_JAR.stat().st_size/(1024*1024):.1f} MB")
else:
    print("[ERRO] CDCMS JAR nao encontrado!")

if MOA_DEPS_JAR.exists():
    print(f"[OK] MOA-deps JAR: {MOA_DEPS_JAR.stat().st_size/(1024*1024):.1f} MB")
else:
    print("[ERRO] MOA-deps JAR nao encontrado!")

# =============================================================================
# PASSO 2: Criar arquivo ARFF de teste
# =============================================================================
print("\n--- PASSO 2: Criar dados de teste ---")

test_arff = TEST_DIR / 'test_imbalanced.arff'

with open(test_arff, 'w') as f:
    f.write("@relation test_imbalanced\n")
    f.write("@attribute a1 numeric\n")
    f.write("@attribute a2 numeric\n")
    f.write("@attribute a3 numeric\n")
    f.write("@attribute class {0,1}\n")
    f.write("@data\n")

    random.seed(42)
    for i in range(3000):
        # Simular drift
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
        cls = 0 if random.random() < 0.9 else 1  # 90-10 imbalanced
        f.write(f"{a1:.4f},{a2:.4f},{a3:.4f},{cls}\n")

print(f"[OK] Arquivo criado: {test_arff}")
print(f"     Instancias: 3000")

# =============================================================================
# PASSO 3: Criar programa Java para avaliacao
# =============================================================================
print("\n--- PASSO 3: Criar CDCMSEvaluator.java ---")

java_code = f'''
import moa.classifiers.meta.CDCMS_CIL_GMean;
import moa.streams.ArffFileStream;
import com.yahoo.labs.samoa.instances.Instance;
import java.io.*;

public class CDCMSEvaluator {{
    public static void main(String[] args) {{
        System.out.println("=== CDCMS Evaluator ===");

        try {{
            // Criar stream
            ArffFileStream stream = new ArffFileStream();
            stream.arffFileOption.setValue("{test_arff}");
            stream.prepareForUse();

            System.out.println("Stream preparado: " + stream.getHeader().numAttributes() + " atributos");

            // Criar classificador
            CDCMS_CIL_GMean classifier = new CDCMS_CIL_GMean();
            classifier.prepareForUse();

            System.out.println("Classificador: " + classifier.getClass().getSimpleName());

            // Metricas
            int numInstances = 0;
            int correctPredictions = 0;
            long startTime = System.currentTimeMillis();

            // Test-then-train
            while (stream.hasMoreInstances() && numInstances < 3000) {{
                Instance instance = stream.nextInstance().getData();

                // Prever
                double[] prediction = classifier.getVotesForInstance(instance);
                int predictedClass = 0;
                double maxVote = 0;
                for (int i = 0; i < prediction.length; i++) {{
                    if (prediction[i] > maxVote) {{
                        maxVote = prediction[i];
                        predictedClass = i;
                    }}
                }}

                int actualClass = (int) instance.classValue();
                if (predictedClass == actualClass) {{
                    correctPredictions++;
                }}

                // Treinar
                classifier.trainOnInstance(instance);

                numInstances++;

                // Progresso
                if (numInstances % 500 == 0) {{
                    double accuracy = (double) correctPredictions / numInstances * 100;
                    System.out.println("  Processadas: " + numInstances + " | Accuracy: " + String.format("%.2f", accuracy) + "%");
                }}
            }}

            long endTime = System.currentTimeMillis();
            double finalAccuracy = (double) correctPredictions / numInstances * 100;

            System.out.println();
            System.out.println("=== RESULTADOS ===");
            System.out.println("Instancias: " + numInstances);
            System.out.println("Predicoes corretas: " + correctPredictions);
            System.out.println("Accuracy: " + String.format("%.2f", finalAccuracy) + "%");
            System.out.println("Tempo: " + (endTime - startTime) + " ms");

            // Salvar resultado
            PrintWriter writer = new PrintWriter(new FileWriter("{TEST_DIR}/result.txt"));
            writer.println("SUCCESS");
            writer.println("Instances: " + numInstances);
            writer.println("Accuracy: " + finalAccuracy);
            writer.println("Time_ms: " + (endTime - startTime));
            writer.close();

            System.out.println();
            System.out.println("*** CDCMS.CIL FUNCIONANDO! ***");

        }} catch (Exception e) {{
            System.out.println("ERRO: " + e.getMessage());
            e.printStackTrace();
        }}
    }}
}}
'''

java_file = TEST_DIR / 'CDCMSEvaluator.java'
with open(java_file, 'w') as f:
    f.write(java_code)

print(f"[OK] CDCMSEvaluator.java criado")

# =============================================================================
# PASSO 4: Compilar
# =============================================================================
print("\n--- PASSO 4: Compilar CDCMSEvaluator ---")

classpath = f"{CDCMS_JAR}:{MOA_DEPS_JAR}"

compile_cmd = ["javac", "-cp", classpath, str(java_file)]
result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)

if result.returncode == 0:
    print("[OK] Compilado com sucesso")
else:
    print("[ERRO] Compilacao falhou")
    print(result.stderr[:500])

# =============================================================================
# PASSO 5: Executar
# =============================================================================
print("\n--- PASSO 5: Executar teste ---")

if result.returncode == 0:
    run_cmd = [
        "java", "-Xmx4g",
        "-cp", f"{classpath}:{TEST_DIR}",
        "CDCMSEvaluator"
    ]

    print("Executando (pode demorar)...")
    start = time.time()

    try:
        run_result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=600)
        duration = time.time() - start

        print(f"\nTempo total: {duration:.1f}s")
        print()
        print(run_result.stdout)

        if run_result.stderr:
            # Filtrar warnings
            errors = [l for l in run_result.stderr.split('\n') if 'WARNING' not in l and l.strip()]
            if errors:
                print("\nErros:")
                for e in errors[:5]:
                    print(f"  {e}")

    except subprocess.TimeoutExpired:
        print("[TIMEOUT] Execucao demorou mais de 10 minutos")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO")
print("="*70)

result_file = TEST_DIR / 'result.txt'
if result_file.exists():
    with open(result_file) as f:
        content = f.read()

    if 'SUCCESS' in content:
        print("\n*** CDCMS.CIL FUNCIONANDO! ***")
        print()
        print(content)
        print()
        print("Proximos passos:")
        print("  1. Atualizar Setup_CDCMS_CIL.ipynb com essas celulas")
        print("  2. Testar com datasets do unified_chunks")
    else:
        print("[FALHA] Verificar logs acima")
else:
    print("[FALHA] Arquivo de resultado nao criado")
