# =============================================================================
# EXECUTAR CDCMS VIA CODIGO JAVA DIRETO
# =============================================================================
# O DoTask tem problemas de parsing. Vamos executar via codigo Java.
# =============================================================================

import subprocess
from pathlib import Path
import time

print("="*70)
print("EXECUTAR CDCMS VIA CODIGO JAVA DIRETO")
print("="*70)

WORK_DIR = Path('/content')
DEPS_DIR = WORK_DIR / 'cdcms_all_deps'
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'

CDCMS_JAR = CDCMS_JARS_DIR / 'cdcms_cil_final.jar'

# Classpath
all_jars = list(DEPS_DIR.glob("*.jar"))
non_moa_jars = [j for j in all_jars if 'moa' not in j.name.lower()]
classpath = str(CDCMS_JAR) + ":" + ":".join(str(j) for j in non_moa_jars)

test_arff = TEST_DIR / 'test_imbalanced.arff'

# =============================================================================
# PASSO 1: Criar programa Java para executar avaliacao
# =============================================================================
print("\n--- PASSO 1: Criar programa Java ---")

java_code = f'''
import moa.classifiers.meta.CDCMS_CIL_GMean;
import moa.streams.ArffFileStream;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.core.TimingUtils;
import com.yahoo.labs.samoa.instances.Instance;
import java.io.FileWriter;
import java.io.PrintWriter;

public class RunCDCMS {{
    public static void main(String[] args) {{
        System.out.println("Iniciando avaliacao CDCMS_CIL_GMean...");

        try {{
            // Criar stream
            ArffFileStream stream = new ArffFileStream();
            stream.arffFileOption.setValue("{test_arff}");
            stream.prepareForUse();

            System.out.println("Stream preparado: " + stream.getHeader().numAttributes() + " atributos");

            // Criar classificador
            CDCMS_CIL_GMean classifier = new CDCMS_CIL_GMean();
            classifier.prepareForUse();

            System.out.println("Classificador preparado: " + classifier.getClass().getSimpleName());

            // Avaliador
            BasicClassificationPerformanceEvaluator evaluator = new BasicClassificationPerformanceEvaluator();

            // Variaveis para metricas
            int numInstances = 0;
            int correctPredictions = 0;
            long startTime = System.currentTimeMillis();

            // Processar stream (test-then-train)
            while (stream.hasMoreInstances() && numInstances < 3000) {{
                Instance instance = stream.nextInstance().getData();

                // Prever (test)
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

                // Treinar (train)
                classifier.trainOnInstance(instance);

                numInstances++;

                // Progresso a cada 500 instancias
                if (numInstances % 500 == 0) {{
                    double accuracy = (double) correctPredictions / numInstances * 100;
                    System.out.println("  Processadas: " + numInstances + " | Accuracy: " + String.format("%.2f", accuracy) + "%");
                }}
            }}

            long endTime = System.currentTimeMillis();
            double finalAccuracy = (double) correctPredictions / numInstances * 100;

            System.out.println("\\n=== RESULTADOS ===");
            System.out.println("Instancias processadas: " + numInstances);
            System.out.println("Predicoes corretas: " + correctPredictions);
            System.out.println("Accuracy: " + String.format("%.2f", finalAccuracy) + "%");
            System.out.println("Tempo: " + (endTime - startTime) + " ms");

            // Salvar resultado em arquivo
            PrintWriter writer = new PrintWriter(new FileWriter("{TEST_DIR}/cdcms_java_result.txt"));
            writer.println("Instancias: " + numInstances);
            writer.println("Accuracy: " + finalAccuracy);
            writer.println("Tempo_ms: " + (endTime - startTime));
            writer.close();

            System.out.println("\\nResultado salvo em: {TEST_DIR}/cdcms_java_result.txt");
            System.out.println("\\n*** SUCESSO! CDCMS.CIL FUNCIONANDO! ***");

        }} catch (Exception e) {{
            System.out.println("ERRO: " + e.getMessage());
            e.printStackTrace();
        }}
    }}
}}
'''

java_file = TEST_DIR / 'RunCDCMS.java'
with open(java_file, 'w') as f:
    f.write(java_code)

print(f"[OK] Codigo Java criado: {java_file}")

# =============================================================================
# PASSO 2: Compilar
# =============================================================================
print("\n--- PASSO 2: Compilar ---")

compile_cmd = ["javac", "-cp", classpath, str(java_file)]

result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)

if result.returncode == 0:
    print("[OK] Compilado com sucesso")
else:
    print("[ERRO] Compilacao falhou")
    print(result.stderr)

# =============================================================================
# PASSO 3: Executar
# =============================================================================
print("\n--- PASSO 3: Executar ---")

if result.returncode == 0:
    run_cmd = ["java", "-Xmx4g", "-cp", f"{classpath}:{TEST_DIR}", "RunCDCMS"]

    print("Executando (pode demorar)...")
    start = time.time()

    try:
        run_result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=600)
        duration = time.time() - start

        print(f"\nTempo total: {duration:.1f}s")
        print("\n" + "="*50)
        print("SAIDA DO PROGRAMA:")
        print("="*50)
        print(run_result.stdout)

        if run_result.stderr:
            print("\nERROS:")
            print(run_result.stderr[:1000])

        # Verificar resultado
        result_file = TEST_DIR / 'cdcms_java_result.txt'
        if result_file.exists():
            print("\n" + "="*50)
            print("ARQUIVO DE RESULTADO:")
            print("="*50)
            with open(result_file) as f:
                print(f.read())

    except subprocess.TimeoutExpired:
        print("[TIMEOUT] Execucao demorou mais de 10 minutos")
    except Exception as e:
        print(f"[ERRO] {e}")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "="*70)
print("RESUMO")
print("="*70)

result_file = TEST_DIR / 'cdcms_java_result.txt'
if result_file.exists():
    print("\n*** CDCMS.CIL FUNCIONANDO VIA CODIGO JAVA! ***")
    print(f"\nJAR: {CDCMS_JAR}")
    print(f"\nPara usar em seus experimentos:")
    print("  - Use codigo Java similar ao RunCDCMS.java")
    print("  - Ou adapte para seu framework de experimentos")
else:
    print("\n[VERIFICAR] Logs acima para diagnostico")
