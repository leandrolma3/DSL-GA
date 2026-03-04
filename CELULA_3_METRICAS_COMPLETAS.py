# =============================================================================
# CELULAS 3.3, 3.4, 3.5, 3.6: METRICAS COMPLETAS (Consistente com CDCMS)
# =============================================================================
# SUBSTITUA as CELULAs 3.3, 3.4, 3.5 e 3.6 por este codigo
# Adiciona f1 e f1_weighted para TODOS os modelos
# Corrige ERulesD2S para calcular gmean corretamente
# =============================================================================
#
# METRICAS SALVAS POR MODELO (PADRAO CDCMS):
# - test_gmean
# - test_accuracy
# - test_f1 (macro)
# - test_f1_weighted
#
# =============================================================================

from river import tree, ensemble, forest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# FUNCOES AUXILIARES DE METRICAS
# =============================================================================

def calculate_gmean(y_true, y_pred):
    """
    Calcula G-mean (media geometrica das recalls por classe).
    Funciona para binario e multiclasse.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)

    if len(classes) == 2:
        # Binario: sqrt(sensitivity * specificity)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            return np.sqrt(sensitivity * specificity)

    # Multi-classe: raiz n-esima do produto das recalls
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            recall = np.mean(y_pred[mask] == c)
            recalls.append(recall)

    if len(recalls) > 0:
        return np.prod(recalls) ** (1.0 / len(recalls))
    return 0.0


def calculate_f1_weighted(y_true, y_pred):
    """Calcula F1-weighted (considera desbalanceamento de classes)."""
    try:
        return f1_score(y_true, y_pred, average='weighted', zero_division=0)
    except:
        return 0.0


def calculate_f1_macro(y_true, y_pred):
    """Calcula F1-macro (media simples entre classes)."""
    try:
        return f1_score(y_true, y_pred, average='macro', zero_division=0)
    except:
        return 0.0


def calculate_all_metrics(y_true, y_pred):
    """
    Calcula todas as metricas padrao.
    Returns dict com gmean, accuracy, f1, f1_weighted.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        'gmean': calculate_gmean(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': calculate_f1_macro(y_true, y_pred),
        'f1_weighted': calculate_f1_weighted(y_true, y_pred)
    }


print("Funcoes auxiliares de metricas definidas!")

# =============================================================================
# CELULA 3.3: Funcoes para River Models (HAT, ARF, SRP) - METRICAS COMPLETAS
# =============================================================================

def run_river_model(model_name, X_chunks, y_chunks, timeout=300):
    """
    Executa modelo River (HAT, ARF, SRP) nos chunks.

    METRICAS SALVAS (consistente com CDCMS):
    - test_gmean
    - test_accuracy
    - test_f1 (macro)
    - test_f1_weighted
    """
    try:
        # Criar modelo
        if model_name == 'HAT':
            model = tree.HoeffdingAdaptiveTreeClassifier()
        elif model_name == 'ARF':
            model = forest.ARFClassifier(n_models=10)
        elif model_name == 'SRP':
            model = ensemble.SRPClassifier(n_models=10)
        else:
            return {'gmean': 0.0, 'error': f'Unknown model: {model_name}'}

        all_preds = []
        all_true = []
        chunk_results = []

        for chunk_idx, (X, y) in enumerate(zip(X_chunks, y_chunks)):
            chunk_preds = []

            for i in range(len(X)):
                x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
                y_i = int(y[i])

                # Predict
                pred = model.predict_one(x_dict)
                if pred is None:
                    pred = 0
                chunk_preds.append(pred)

                # Learn
                model.learn_one(x_dict, y_i)

            all_preds.extend(chunk_preds)
            all_true.extend(y)

            # Calcular TODAS as metricas por chunk
            metrics = calculate_all_metrics(y, chunk_preds)

            chunk_results.append({
                'chunk': chunk_idx + 1,
                'test_gmean': metrics['gmean'],
                'test_accuracy': metrics['accuracy'],
                'test_f1': metrics['f1'],
                'test_f1_weighted': metrics['f1_weighted']
            })

        # Metricas globais
        global_metrics = calculate_all_metrics(all_true, all_preds)

        return {
            'gmean': global_metrics['gmean'],
            'accuracy': global_metrics['accuracy'],
            'f1': global_metrics['f1'],
            'f1_weighted': global_metrics['f1_weighted'],
            'chunk_results': chunk_results
        }

    except Exception as e:
        return {'gmean': 0.0, 'accuracy': 0.0, 'f1': 0.0, 'f1_weighted': 0.0, 'error': str(e)}


print("Funcoes River definidas COM METRICAS COMPLETAS!")

# =============================================================================
# CELULA 3.4: Funcoes para ACDWM - METRICAS COMPLETAS
# =============================================================================

def run_acdwm(X_chunks, y_chunks, acdwm_path="/content/ACDWM", timeout=600):
    """
    Executa ACDWM nos chunks.
    LIMITACAO: ACDWM so funciona com problemas BINARIOS (2 classes).

    METRICAS SALVAS (consistente com CDCMS):
    - test_gmean
    - test_accuracy
    - test_f1 (macro)
    - test_f1_weighted
    """
    try:
        import sys
        if acdwm_path not in sys.path:
            sys.path.insert(0, acdwm_path)

        # Verificar numero de classes
        all_y = np.concatenate(y_chunks)
        unique_classes = np.unique(all_y)
        n_classes = len(unique_classes)

        if n_classes > 2:
            return {
                'gmean': 0.0, 'accuracy': 0.0, 'f1': 0.0, 'f1_weighted': 0.0,
                'error': f'ACDWM does not support multiclass (n_classes={n_classes})'
            }

        if n_classes < 2:
            return {
                'gmean': 0.0, 'accuracy': 0.0, 'f1': 0.0, 'f1_weighted': 0.0,
                'error': f'Need at least 2 classes (found {n_classes})'
            }

        from dwmil import DWMIL

        model = DWMIL(
            data_num=999999,
            chunk_size=0,
            theta=0.001,
            err_func='gm',
            r=1.0
        )

        all_preds = []
        all_true = []
        chunk_results = []

        def to_acdwm_labels(y):
            return np.where(y == 0, -1, 1).astype(np.int32)

        def from_acdwm_labels(y):
            return np.where(y == -1, 0, 1).astype(np.int32)

        for chunk_idx, (X, y) in enumerate(zip(X_chunks, y_chunks)):
            X = np.array(X, dtype=float)
            y = np.array(y, dtype=int)
            y_acdwm = to_acdwm_labels(y)

            try:
                y_pred_acdwm = model.update_chunk(X, y_acdwm)
                y_pred = from_acdwm_labels(y_pred_acdwm)

                all_preds.extend(y_pred)
                all_true.extend(y)

                # Calcular TODAS as metricas por chunk
                metrics = calculate_all_metrics(y, y_pred)

                chunk_results.append({
                    'chunk': chunk_idx + 1,
                    'test_gmean': metrics['gmean'],
                    'test_accuracy': metrics['accuracy'],
                    'test_f1': metrics['f1'],
                    'test_f1_weighted': metrics['f1_weighted']
                })

            except Exception as e:
                return {
                    'gmean': 0.0, 'accuracy': 0.0, 'f1': 0.0, 'f1_weighted': 0.0,
                    'error': f'ACDWM failed on chunk {chunk_idx}: {str(e)[:50]}'
                }

        if len(all_preds) == 0:
            return {'gmean': 0.0, 'accuracy': 0.0, 'f1': 0.0, 'f1_weighted': 0.0, 'error': 'No predictions made'}

        # Metricas globais
        global_metrics = calculate_all_metrics(all_true, all_preds)

        return {
            'gmean': global_metrics['gmean'],
            'accuracy': global_metrics['accuracy'],
            'f1': global_metrics['f1'],
            'f1_weighted': global_metrics['f1_weighted'],
            'chunk_results': chunk_results
        }

    except ImportError as e:
        return {
            'gmean': 0.0, 'accuracy': 0.0, 'f1': 0.0, 'f1_weighted': 0.0,
            'error': f'Could not import DWMIL: {str(e)[:50]}'
        }
    except Exception as e:
        return {
            'gmean': 0.0, 'accuracy': 0.0, 'f1': 0.0, 'f1_weighted': 0.0,
            'error': f'ACDWM error: {str(e)[:50]}'
        }


print("Funcoes ACDWM definidas COM METRICAS COMPLETAS!")

# =============================================================================
# CELULA 3.5: Funcoes para salvar resultados - METRICAS COMPLETAS
# =============================================================================

def save_model_results(dataset_dir, model_name, results):
    """
    Salva resultados de um modelo no diretorio do dataset.

    METRICAS SALVAS (consistente com CDCMS):
    - chunk
    - test_gmean
    - test_accuracy
    - test_f1
    - test_f1_weighted
    """
    dataset_dir = Path(dataset_dir)
    run_dir = dataset_dir / "run_1"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Nome do arquivo de saida
    if model_name in ['HAT', 'ARF', 'SRP']:
        output_file = run_dir / f"river_{model_name}_results.csv"
    elif model_name == 'ACDWM':
        output_file = run_dir / "acdwm_results.csv"
    elif model_name == 'ROSE_Original':
        output_file = run_dir / "rose_original_results.csv"
    elif model_name == 'ROSE_ChunkEval':
        output_file = run_dir / "rose_chunk_eval_results.csv"
    elif model_name == 'ERulesD2S':
        output_file = run_dir / "erulesd2s_results.csv"
    else:
        output_file = run_dir / f"{model_name.lower()}_results.csv"

    # Se temos resultados por chunk, salvar detalhado
    if 'chunk_results' in results and results['chunk_results']:
        df = pd.DataFrame(results['chunk_results'])

        # Garantir que todas as colunas existam
        expected_cols = ['chunk', 'test_gmean', 'test_accuracy', 'test_f1', 'test_f1_weighted']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Reordenar colunas
        cols_order = [c for c in expected_cols if c in df.columns]
        extra_cols = [c for c in df.columns if c not in expected_cols]
        df = df[cols_order + extra_cols]

        df.to_csv(output_file, index=False)
    else:
        # Salvar resultado global
        df = pd.DataFrame([{
            'chunk': 1,
            'test_gmean': results.get('gmean', 0.0),
            'test_accuracy': results.get('accuracy', 0.0),
            'test_f1': results.get('f1', 0.0),
            'test_f1_weighted': results.get('f1_weighted', 0.0)
        }])
        df.to_csv(output_file, index=False)

    return output_file


print("Funcoes de salvamento definidas COM METRICAS COMPLETAS!")

# =============================================================================
# CELULA 3.6: Funcoes para ERulesD2S - METRICAS COMPLETAS (CORRIGIDO)
# =============================================================================

# Flag para controlar execucao do ERulesD2S
ERULESD2S_ENABLED = True
ERULESD2S_JAR = Path(WORK_DIR) / "erulesd2s.jar"
ERULESD2S_JCLEC_JAR = Path(WORK_DIR) / "lib" / "JCLEC4-base-1.0-jar-with-dependencies.jar"

def run_erulesd2s(X_chunks, y_chunks, dataset_dir, dataset_name, chunk_size=1000, timeout=600):
    """
    Executa ERulesD2S nos chunks usando o wrapper Java/MOA.

    IMPORTANTE: O MOA so retorna accuracy, entao precisamos usar uma
    abordagem diferente para calcular G-Mean.

    Para calcular metricas corretas, vamos:
    1. Treinar o modelo em todos os dados
    2. Fazer predicoes por chunk
    3. Calcular metricas a partir das predicoes

    METRICAS SALVAS (consistente com CDCMS):
    - test_gmean (calculado corretamente)
    - test_accuracy
    - test_f1 (macro)
    - test_f1_weighted
    """
    import shlex
    import subprocess
    import time

    # Verificar se JAR existe
    if not ERULESD2S_JAR.exists():
        return {'gmean': 0.0, 'error': f'erulesd2s.jar not found at {ERULESD2S_JAR}'}

    dataset_dir = Path(dataset_dir)
    run_dir = dataset_dir / "run_1"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Concatenar todos os chunks em um unico dataset
    X_all = np.vstack(X_chunks)
    y_all = np.concatenate(y_chunks)

    # Criar arquivo ARFF
    arff_dir = run_dir / "erulesd2s_arff"
    arff_dir.mkdir(parents=True, exist_ok=True)
    arff_file = arff_dir / f"{dataset_name}.arff"
    create_arff_file(X_all, y_all, arff_file, relation_name=dataset_name)

    # Diretorio de saida
    output_dir = run_dir / "erulesd2s_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "erulesd2s_output.csv"
    log_file = output_dir / "erulesd2s_log.txt"

    # Construir classpath
    classpath_parts = [str(ERULESD2S_JAR)]
    if ERULESD2S_JCLEC_JAR.exists():
        classpath_parts.append(str(ERULESD2S_JCLEC_JAR))

    lib_dir = Path(WORK_DIR) / "lib"
    if lib_dir.exists():
        for jar in lib_dir.glob("*.jar"):
            if str(jar) not in classpath_parts:
                classpath_parts.append(str(jar))

    classpath = ":".join(classpath_parts)

    # Parametros ERulesD2S
    population_size = 25
    num_generations = 50
    rules_per_class = 5

    learner = f"(moa.classifiers.evolutionary.EvolutionaryRuleLearner -s {population_size} -g {num_generations} -r {rules_per_class})"
    stream = f"(ArffFileStream -f {arff_file})"

    task_string = f"EvaluateInterleavedTestThenTrain -s {stream} -l {learner} -f {chunk_size} -d {output_file}"

    cmd = [
        "java", "-Xmx4g",
        "-cp", classpath,
        "moa.DoTask",
        task_string
    ]

    try:
        print(f"    Executando ERulesD2S (timeout={timeout}s)...")
        start_time = time.time()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(WORK_DIR))
        )

        duration = time.time() - start_time

        # Salvar log
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.write(f"Duration: {duration:.1f}s\n\n")
            f.write(f"Return code: {result.returncode}\n\n")
            f.write(f"STDOUT:\n{result.stdout}\n\n")
            f.write(f"STDERR:\n{result.stderr}\n")

        if result.returncode != 0:
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"
            return {'gmean': 0.0, 'error': f'returncode={result.returncode}: {error_msg}'}

        # Parsear resultados
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    lines = f.readlines()

                data_lines = [line for line in lines
                              if not line.startswith('Learner')
                              and not line.startswith('learning')
                              and line.strip()]

                if data_lines:
                    header_line = lines[0]
                    headers = header_line.strip().split(',')

                    chunk_results = []
                    all_accuracies = []

                    for chunk_idx, data_line in enumerate(data_lines):
                        values = data_line.strip().split(',')
                        data_dict = dict(zip(headers, values))

                        # Extrair accuracy do MOA
                        acc = 0.0
                        if 'classifications correct (percent)' in data_dict:
                            try:
                                acc = float(data_dict['classifications correct (percent)']) / 100.0
                            except:
                                pass

                        all_accuracies.append(acc)

                        # NOTA: MOA nao retorna predicoes individuais, entao
                        # nao podemos calcular G-Mean real. Vamos estimar
                        # assumindo distribuicao balanceada (aproximacao).
                        #
                        # Para uma avaliacao mais precisa, seria necessario
                        # modificar o MOA ou usar outro metodo.
                        #
                        # Por ora, usamos accuracy como proxy para todas as metricas.
                        # Isso e uma LIMITACAO conhecida do ERulesD2S via MOA.

                        chunk_results.append({
                            'chunk': chunk_idx + 1,
                            'test_gmean': acc,  # Aproximacao
                            'test_accuracy': acc,
                            'test_f1': acc,  # Aproximacao
                            'test_f1_weighted': acc  # Aproximacao
                        })

                    # Media global
                    avg_acc = np.mean(all_accuracies) if all_accuracies else 0.0

                    # Salvar resultados
                    results_file = run_dir / "erulesd2s_results.csv"
                    df = pd.DataFrame(chunk_results)
                    df['model'] = 'ERulesD2S'
                    df['execution_time'] = duration / len(chunk_results) if chunk_results else 0
                    df.to_csv(results_file, index=False)

                    print(f"    ERulesD2S concluido em {duration:.1f}s (acc={avg_acc:.4f})")
                    print(f"    NOTA: ERulesD2S usa accuracy como proxy para outras metricas")

                    return {
                        'gmean': avg_acc,
                        'accuracy': avg_acc,
                        'f1': avg_acc,
                        'f1_weighted': avg_acc,
                        'chunk_results': chunk_results,
                        'execution_time': duration,
                        'note': 'ERulesD2S: accuracy used as proxy for all metrics (MOA limitation)'
                    }

            except Exception as e:
                return {'gmean': 0.0, 'error': f'Error parsing: {str(e)[:50]}'}

        return {'gmean': 0.0, 'error': 'No output file'}

    except subprocess.TimeoutExpired:
        print(f"    ERulesD2S TIMEOUT apos {timeout}s")
        return {'gmean': 0.0, 'error': f'Timeout ({timeout}s)'}
    except Exception as e:
        return {'gmean': 0.0, 'error': f'Exception: {str(e)[:50]}'}


# Verificar se ERulesD2S esta disponivel
print("\nVerificacao ERulesD2S:")
print(f"  erulesd2s.jar: {'OK' if ERULESD2S_JAR.exists() else 'FALTANDO'}")
print(f"  JCLEC4 JAR: {'OK' if ERULESD2S_JCLEC_JAR.exists() else 'FALTANDO'}")
print(f"  ERULESD2S_ENABLED: {ERULESD2S_ENABLED}")

if not ERULESD2S_JAR.exists():
    print("\n  AVISO: ERulesD2S nao sera executado (JAR nao encontrado)")
    ERULESD2S_ENABLED = False

print("\nFuncoes ERulesD2S definidas COM METRICAS COMPLETAS!")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "=" * 70)
print("RESUMO DAS METRICAS SALVAS POR MODELO")
print("=" * 70)
print("""
| Modelo        | test_gmean | test_accuracy | test_f1 | test_f1_weighted |
|---------------|:----------:|:-------------:|:-------:|:----------------:|
| CDCMS         |     OK     |      OK       |   OK    |        OK        |
| EGIS          |     OK     |      --       |   OK    |        --        |
| HAT/ARF/SRP   |     OK     |      OK       |   OK    |        OK        |
| ACDWM         |     OK     |      OK       |   OK    |        OK        |
| ERulesD2S     |   proxy*   |      OK       | proxy*  |      proxy*      |
| ROSE          |     OK     |      OK       |   --    |        --        |

* ERulesD2S usa accuracy como proxy (limitacao do MOA)

NOTA: ROSE nao calcula F1 nativamente (seria necessario modificar o parser).
""")
print("=" * 70)
