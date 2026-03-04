# =============================================================================
# CELULA 5.1 CORRIGIDA: Criar CDCMSEvaluator v2.0
# =============================================================================
# IMPORTANTE: Esta versao recebe argumentos de linha de comando!
# Substitui a celula 5.1 original do notebook.
#
# Uso: java CDCMSEvaluator <arff_file> <output_file> <classifier>
# Saida: CSV com colunas [instance, accuracy, prediction, actual]
# =============================================================================

import subprocess
from pathlib import Path

print("="*70)
print("CELULA 5.1 CORRIGIDA: CDCMSEvaluator v2.0")
print("="*70)

# Verificar que as variaveis necessarias existem
try:
    _ = CDCMS_JAR
    _ = MOA_DEPS_JAR
    _ = TEST_DIR
except NameError:
    print("[ERRO] Variaveis nao definidas!")
    print("Execute a celula 1.2 primeiro para definir:")
    print("  - CDCMS_JAR")
    print("  - MOA_DEPS_JAR")
    print("  - TEST_DIR")
    raise

# =============================================================================
# Codigo Java do CDCMSEvaluator v2.0
# =============================================================================
java_evaluator_v2 = '''
/**
 * CDCMSEvaluator v2.0
 *
 * IMPORTANTE: Esta versao recebe argumentos de linha de comando!
 *
 * Uso: java CDCMSEvaluator <arff_file> <output_file> <classifier>
 *
 * Saida: CSV com colunas [instance, accuracy, prediction, actual]
 */

import moa.classifiers.Classifier;
import moa.classifiers.meta.CDCMS_CIL_GMean;
import moa.classifiers.meta.CDCMS_CIL;
import moa.streams.ArffFileStream;
import com.yahoo.labs.samoa.instances.Instance;
import java.io.*;
import java.util.*;

public class CDCMSEvaluator {

    public static void main(String[] args) {
        // Verificar argumentos
        if (args.length < 3) {
            System.out.println("Uso: java CDCMSEvaluator <arff_file> <output_file> <classifier>");
            System.out.println();
            System.out.println("Classificadores:");
            System.out.println("  - CDCMS_CIL_GMean");
            System.out.println("  - CDCMS_CIL");
            return;
        }

        String arffFile = args[0];
        String outputFile = args[1];
        String classifierName = args[2];

        System.out.println("=== CDCMS Evaluator v2.0 ===");
        System.out.println("Input:      " + arffFile);
        System.out.println("Output:     " + outputFile);
        System.out.println("Classifier: " + classifierName);

        try {
            // 1. Criar stream
            ArffFileStream stream = new ArffFileStream();
            stream.arffFileOption.setValue(arffFile);
            stream.prepareForUse();

            System.out.println("Atributos:  " + stream.getHeader().numAttributes());
            System.out.println("Classes:    " + stream.getHeader().numClasses());

            // 2. Criar classificador
            Classifier classifier;
            if (classifierName.equals("CDCMS_CIL_GMean")) {
                classifier = new CDCMS_CIL_GMean();
            } else if (classifierName.equals("CDCMS_CIL")) {
                classifier = new CDCMS_CIL();
            } else {
                System.out.println("[ERRO] Classificador desconhecido: " + classifierName);
                return;
            }
            classifier.prepareForUse();

            // 3. Processar stream (Test-then-Train)
            int numInstances = 0;
            int correctPredictions = 0;
            long startTime = System.currentTimeMillis();

            List<String> results = new ArrayList<>();
            results.add("instance,accuracy,prediction,actual");

            System.out.println();
            System.out.println("Processando...");

            while (stream.hasMoreInstances()) {
                Instance instance = stream.nextInstance().getData();

                // TESTE
                double[] prediction = classifier.getVotesForInstance(instance);
                int predictedClass = 0;
                double maxVote = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < prediction.length; i++) {
                    if (prediction[i] > maxVote) {
                        maxVote = prediction[i];
                        predictedClass = i;
                    }
                }

                int actualClass = (int) instance.classValue();
                if (predictedClass == actualClass) {
                    correctPredictions++;
                }
                numInstances++;

                double accuracy = (double) correctPredictions / numInstances;
                results.add(numInstances + "," +
                           String.format("%.6f", accuracy) + "," +
                           predictedClass + "," + actualClass);

                // TREINO
                classifier.trainOnInstance(instance);

                if (numInstances % 2000 == 0) {
                    System.out.println("  " + numInstances + " instancias | Acc: " +
                                     String.format("%.4f", accuracy));
                }
            }

            long endTime = System.currentTimeMillis();
            double finalAccuracy = (double) correctPredictions / numInstances;

            // 4. Salvar CSV
            PrintWriter writer = new PrintWriter(new FileWriter(outputFile));
            for (String line : results) {
                writer.println(line);
            }
            writer.close();

            // 5. Resumo
            System.out.println();
            System.out.println("=== RESULTADO ===");
            System.out.println("Instancias: " + numInstances);
            System.out.println("Accuracy:   " + String.format("%.4f", finalAccuracy));
            System.out.println("Tempo:      " + (endTime - startTime) + " ms");
            System.out.println("Salvo em:   " + outputFile);

        } catch (Exception e) {
            System.out.println("[ERRO] " + e.getMessage());
            e.printStackTrace();
        }
    }
}
'''

# =============================================================================
# Salvar e compilar
# =============================================================================
evaluator_file = TEST_DIR / 'CDCMSEvaluator.java'

# Remover versao antiga se existir
old_class = TEST_DIR / 'CDCMSEvaluator.class'
if old_class.exists():
    old_class.unlink()
    print("[INFO] Removida versao antiga do .class")

# Salvar nova versao
with open(evaluator_file, 'w') as f:
    f.write(java_evaluator_v2)

print(f"[OK] CDCMSEvaluator.java v2.0 criado")

# Compilar
print("\nCompilando...")

classpath = f"{CDCMS_JAR}:{MOA_DEPS_JAR}"

compile_result = subprocess.run(
    ["javac", "-cp", classpath, str(evaluator_file)],
    capture_output=True,
    text=True,
    timeout=60
)

if compile_result.returncode == 0:
    class_file = TEST_DIR / 'CDCMSEvaluator.class'
    if class_file.exists():
        print(f"[OK] Compilado com sucesso!")
        print(f"     {class_file} ({class_file.stat().st_size} bytes)")
    else:
        print("[ERRO] .class nao foi criado")
else:
    print("[ERRO] Compilacao falhou:")
    print(compile_result.stderr[:500])

# =============================================================================
# Teste rapido para verificar que funciona
# =============================================================================
print("\n" + "-"*50)
print("TESTE: Verificar que aceita argumentos")
print("-"*50)

test_cmd = [
    "java", "-Xmx1g",
    "-cp", f"{classpath}:{TEST_DIR}",
    "CDCMSEvaluator"
]

test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)

if "Uso: java CDCMSEvaluator" in test_result.stdout:
    print("[OK] CDCMSEvaluator v2.0 funcionando!")
    print()
    print("Mensagem de uso:")
    for line in test_result.stdout.strip().split('\n')[:5]:
        print(f"  {line}")
else:
    print("[AVISO] Saida inesperada:")
    print(test_result.stdout[:300])
    if test_result.stderr:
        print("Stderr:", test_result.stderr[:200])

print()
print("="*70)
print("[OK] CDCMSEvaluator v2.0 pronto para uso!")
print("="*70)
print()
print("Agora execute a CELULA 7.5 (Teste Rapido) novamente.")
