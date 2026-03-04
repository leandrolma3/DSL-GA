#!/usr/bin/env python3
"""Script para atualizar a função run_erulesd2s no notebook."""

import json

with open('Execute_Comparative_All_Experiments.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New run_erulesd2s function (corrected)
new_erulesd2s_cell = """# CÉLULA 3.6: Funções para ERulesD2S (EXECUÇÃO REAL)

# Flag para controlar execução do ERulesD2S
ERULESD2S_ENABLED = True  # Mude para False se quiser apenas usar cache
ERULESD2S_JAR = Path(WORK_DIR) / "erulesd2s.jar"
ERULESD2S_JCLEC_JAR = Path(WORK_DIR) / "lib" / "JCLEC4-base-1.0-jar-with-dependencies.jar"

def run_erulesd2s(X_chunks, y_chunks, dataset_dir, dataset_name, chunk_size=1000, timeout=600):
    \"\"\"
    Executa ERulesD2S nos chunks usando o wrapper Java/MOA.

    IMPORTANTE: Baseado no erulesd2s_wrapper.py que funciona corretamente.

    Args:
        X_chunks: Lista de arrays X (features)
        y_chunks: Lista de arrays y (labels)
        dataset_dir: Diretório do dataset
        dataset_name: Nome do dataset
        chunk_size: Tamanho do chunk para avaliação
        timeout: Timeout em segundos (padrão: 10 min - ajustado para velocidade)

    Returns:
        dict com 'gmean', 'accuracy', 'chunk_results' ou 'error'
    \"\"\"
    import shlex

    # Verificar se JAR existe
    if not ERULESD2S_JAR.exists():
        return {'gmean': 0.0, 'error': f'erulesd2s.jar not found at {ERULESD2S_JAR}'}

    dataset_dir = Path(dataset_dir)
    run_dir = dataset_dir / "run_1"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Concatenar todos os chunks em um único dataset
    X_all = np.vstack(X_chunks)
    y_all = np.concatenate(y_chunks)

    # Criar arquivo ARFF
    arff_dir = run_dir / "erulesd2s_arff"
    arff_dir.mkdir(parents=True, exist_ok=True)
    arff_file = arff_dir / f"{dataset_name}.arff"
    create_arff_file(X_all, y_all, arff_file, relation_name=dataset_name)

    # Diretório de saída
    output_dir = run_dir / "erulesd2s_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "erulesd2s_output.csv"
    log_file = output_dir / "erulesd2s_log.txt"

    # Construir classpath (Linux usa :)
    classpath_parts = [str(ERULESD2S_JAR)]
    if ERULESD2S_JCLEC_JAR.exists():
        classpath_parts.append(str(ERULESD2S_JCLEC_JAR))

    # Adicionar outras JARs na pasta lib/
    lib_dir = Path(WORK_DIR) / "lib"
    if lib_dir.exists():
        for jar in lib_dir.glob("*.jar"):
            if str(jar) not in classpath_parts:
                classpath_parts.append(str(jar))

    classpath = ":".join(classpath_parts)

    # Parâmetros ERulesD2S (REDUZIDOS para velocidade)
    population_size = 15   # Reduzido de 25
    num_generations = 25   # Reduzido de 50
    rules_per_class = 3    # Reduzido de 5

    # Construir comando como lista (formato correto para subprocess)
    # IMPORTANTE: A task string precisa ser um único argumento para moa.DoTask
    learner = f"(moa.classifiers.evolutionary.EvolutionaryRuleLearner -s {population_size} -g {num_generations} -r {rules_per_class})"
    stream = f"(ArffFileStream -f {arff_file})"

    task_string = f"EvaluateInterleavedTestThenTrain -s {stream} -l {learner} -f {chunk_size} -d {output_file}"

    cmd = [
        "java",
        "-Xmx4g",
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
            f.write(f"Command: {' '.join(cmd)}\\n\\n")
            f.write(f"Duration: {duration:.1f}s\\n\\n")
            f.write(f"Return code: {result.returncode}\\n\\n")
            f.write(f"STDOUT:\\n{result.stdout}\\n\\n")
            f.write(f"STDERR:\\n{result.stderr}\\n")

        if result.returncode != 0:
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"
            return {
                'gmean': 0.0,
                'error': f'returncode={result.returncode}: {error_msg}'
            }

        # Parsear resultados
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    lines = f.readlines()

                # Filtrar linhas de dados (não headers)
                data_lines = [line for line in lines
                              if not line.startswith('Learner')
                              and not line.startswith('learning')
                              and line.strip()]

                if data_lines:
                    header_line = lines[0]
                    headers = header_line.strip().split(',')

                    accuracies = []
                    chunk_results = []

                    for chunk_idx, data_line in enumerate(data_lines):
                        values = data_line.strip().split(',')
                        data_dict = dict(zip(headers, values))

                        # Extrair accuracy
                        acc = 0.0
                        if 'classifications correct (percent)' in data_dict:
                            try:
                                acc = float(data_dict['classifications correct (percent)']) / 100.0
                            except:
                                pass

                        accuracies.append(acc)
                        chunk_results.append({
                            'chunk': chunk_idx + 1,
                            'test_gmean': acc,
                            'test_accuracy': acc
                        })

                    # Calcular média
                    gmean = np.mean(accuracies) if accuracies else 0.0

                    # Salvar resultados no formato esperado
                    results_file = run_dir / "erulesd2s_results.csv"
                    df = pd.DataFrame(chunk_results)
                    df['model'] = 'ERulesD2S'
                    df['execution_time'] = duration / len(chunk_results) if chunk_results else 0
                    df.to_csv(results_file, index=False)

                    print(f"    ERulesD2S concluído em {duration:.1f}s (gmean={gmean:.4f})")

                    return {
                        'gmean': gmean,
                        'accuracy': gmean,
                        'chunk_results': chunk_results,
                        'execution_time': duration
                    }

            except Exception as e:
                return {'gmean': 0.0, 'error': f'Error parsing: {str(e)[:50]}'}

        return {'gmean': 0.0, 'error': 'No output file'}

    except subprocess.TimeoutExpired:
        print(f"    ERulesD2S TIMEOUT após {timeout}s")
        return {'gmean': 0.0, 'error': f'Timeout ({timeout}s)'}
    except Exception as e:
        return {'gmean': 0.0, 'error': f'Exception: {str(e)[:50]}'}


# Verificar se ERulesD2S está disponível
print("Verificação ERulesD2S:")
print(f"  erulesd2s.jar: {'OK' if ERULESD2S_JAR.exists() else 'FALTANDO'}")
print(f"  JCLEC4 JAR: {'OK' if ERULESD2S_JCLEC_JAR.exists() else 'FALTANDO'}")
print(f"  ERULESD2S_ENABLED: {ERULESD2S_ENABLED}")

if not ERULESD2S_JAR.exists():
    print("\\n  AVISO: ERulesD2S não será executado (JAR não encontrado)")
    print("  Apenas resultados em cache serão usados")
    ERULESD2S_ENABLED = False

print("\\nFunções ERulesD2S definidas!")"""

# Find and update the cell
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'def run_erulesd2s' in src and 'CÉLULA 3.6' in src:
            cell['source'] = new_erulesd2s_cell.split('\n')
            cell['source'] = [line + '\n' if j < len(new_erulesd2s_cell.split('\n')) - 1 else line
                              for j, line in enumerate(new_erulesd2s_cell.split('\n'))]
            print(f"Updated run_erulesd2s function at cell {i}")
            break

with open('Execute_Comparative_All_Experiments.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook saved!")
