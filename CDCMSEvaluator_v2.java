/**
 * CDCMSEvaluator v2.0
 *
 * Executa CDCMS em modo test-then-train e salva predicoes individuais.
 *
 * Saida: CSV com colunas [instance, accuracy, prediction, actual]
 *
 * Uso:
 *   java CDCMSEvaluator <arff_file> <output_file> <classifier>
 *
 * Classifiers:
 *   - CDCMS_CIL_GMean (recomendado)
 *   - CDCMS_CIL
 *
 * Data: 2026-01-26
 * Versao: 2.0 - Com predicoes individuais para calculo de metricas por chunk
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
            System.out.println("Classificadores disponiveis:");
            System.out.println("  - CDCMS_CIL_GMean (recomendado para class imbalance)");
            System.out.println("  - CDCMS_CIL (versao base)");
            System.out.println();
            System.out.println("Exemplo:");
            System.out.println("  java CDCMSEvaluator dataset.arff output.csv CDCMS_CIL_GMean");
            return;
        }

        String arffFile = args[0];
        String outputFile = args[1];
        String classifierName = args[2];

        System.out.println("=== CDCMS Evaluator v2.0 ===");
        System.out.println("Input:      " + arffFile);
        System.out.println("Output:     " + outputFile);
        System.out.println("Classifier: " + classifierName);
        System.out.println();

        try {
            // ================================================================
            // 1. Criar stream a partir do arquivo ARFF
            // ================================================================
            ArffFileStream stream = new ArffFileStream();
            stream.arffFileOption.setValue(arffFile);
            stream.prepareForUse();

            int numAttributes = stream.getHeader().numAttributes();
            int numClasses = stream.getHeader().numClasses();

            System.out.println("Stream preparado:");
            System.out.println("  Atributos: " + numAttributes);
            System.out.println("  Classes:   " + numClasses);
            System.out.println();

            // ================================================================
            // 2. Criar classificador
            // ================================================================
            Classifier classifier;

            if (classifierName.equals("CDCMS_CIL_GMean")) {
                classifier = new CDCMS_CIL_GMean();
            } else if (classifierName.equals("CDCMS_CIL")) {
                classifier = new CDCMS_CIL();
            } else {
                System.out.println("[ERRO] Classificador desconhecido: " + classifierName);
                System.out.println("Use: CDCMS_CIL_GMean ou CDCMS_CIL");
                return;
            }

            classifier.prepareForUse();
            System.out.println("Classificador iniciado: " + classifier.getClass().getSimpleName());

            // ================================================================
            // 3. Processar stream (Test-then-Train)
            // ================================================================
            int numInstances = 0;
            int correctPredictions = 0;
            long startTime = System.currentTimeMillis();

            // Lista para armazenar todas as predicoes
            List<String> results = new ArrayList<>();
            results.add("instance,accuracy,prediction,actual");  // Header

            System.out.println();
            System.out.println("Processando stream (test-then-train)...");

            while (stream.hasMoreInstances()) {
                Instance instance = stream.nextInstance().getData();

                // -----------------------------------------------------------
                // TESTE: Prever classe da instancia atual
                // -----------------------------------------------------------
                double[] prediction = classifier.getVotesForInstance(instance);

                // Encontrar classe com maior voto
                int predictedClass = 0;
                double maxVote = Double.NEGATIVE_INFINITY;

                for (int i = 0; i < prediction.length; i++) {
                    if (prediction[i] > maxVote) {
                        maxVote = prediction[i];
                        predictedClass = i;
                    }
                }

                // Classe real
                int actualClass = (int) instance.classValue();

                // Atualizar contagem de acertos
                if (predictedClass == actualClass) {
                    correctPredictions++;
                }

                numInstances++;

                // Calcular accuracy acumulada
                double accuracy = (double) correctPredictions / numInstances;

                // Salvar resultado desta instancia
                results.add(numInstances + "," +
                           String.format("%.6f", accuracy) + "," +
                           predictedClass + "," +
                           actualClass);

                // -----------------------------------------------------------
                // TREINO: Atualizar modelo com a instancia atual
                // -----------------------------------------------------------
                classifier.trainOnInstance(instance);

                // Progresso a cada 1000 instancias
                if (numInstances % 1000 == 0) {
                    System.out.println("  Processadas: " + numInstances +
                                     " | Accuracy: " + String.format("%.4f", accuracy));
                }
            }

            long endTime = System.currentTimeMillis();
            double finalAccuracy = (double) correctPredictions / numInstances;
            double timeSeconds = (endTime - startTime) / 1000.0;

            // ================================================================
            // 4. Salvar resultados em CSV
            // ================================================================
            PrintWriter writer = new PrintWriter(new FileWriter(outputFile));
            for (String line : results) {
                writer.println(line);
            }
            writer.close();

            // ================================================================
            // 5. Exibir resumo
            // ================================================================
            System.out.println();
            System.out.println("=== RESULTADOS ===");
            System.out.println("Instancias processadas: " + numInstances);
            System.out.println("Predicoes corretas:     " + correctPredictions);
            System.out.println("Accuracy final:         " + String.format("%.4f", finalAccuracy));
            System.out.println("Tempo de execucao:      " + String.format("%.1f", timeSeconds) + " s");
            System.out.println();
            System.out.println("Resultados salvos em: " + outputFile);
            System.out.println();
            System.out.println("*** EXECUCAO CONCLUIDA COM SUCESSO ***");

        } catch (FileNotFoundException e) {
            System.out.println("[ERRO] Arquivo nao encontrado: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            System.out.println("[ERRO] " + e.getMessage());
            e.printStackTrace();
        }
    }
}
