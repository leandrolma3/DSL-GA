# =============================================================================
# CELULA 3.2 CORRIGIDA: Executar CDCMS.CIL_GMean
# =============================================================================
# Correções aplicadas:
# 1. Verificação prévia da classe CDCMS_CIL_GMean.class no JAR
# 2. Uso de caminho completo para moa.streams.ArffFileStream
# 3. Captura completa de STDOUT e STDERR
# 4. Verificação do diretório de trabalho
# 5. Busca do arquivo de saída em múltiplos locais
# 6. Teste de validação do classpath antes da execução principal
# =============================================================================

import subprocess
import time
import os
from pathlib import Path

print("="*60)
print("EXECUTANDO CDCMS.CIL_GMean (VERSAO CORRIGIDA)")
print("="*60)

# Output
output_file = TEST_DIR / 'cdcms_output.csv'
log_file = TEST_DIR / 'cdcms_log.txt'

# Limpar arquivos anteriores
if output_file.exists():
    output_file.unlink()

# =============================================================================
# PASSO 1: Verificar classe principal no JAR
# =============================================================================
print("\n--- PASSO 1: Verificando classe CDCMS_CIL_GMean ---")

if CDCMS_JAR.exists():
    # Verificar se a classe principal existe (não apenas classes internas)
    check_class = subprocess.run(
        f'jar tf "{CDCMS_JAR}" | grep "CDCMS_CIL_GMean.class$"',
        shell=True, capture_output=True, text=True
    )

    if check_class.stdout.strip():
        print(f"[OK] Classe principal encontrada: {check_class.stdout.strip()}")
    else:
        print("[AVISO] Classe CDCMS_CIL_GMean.class NAO encontrada diretamente!")
        print("Verificando todas as classes CDCMS...")
        all_cdcms = subprocess.run(
            f'jar tf "{CDCMS_JAR}" | grep -E "CDCMS.*\\.class$" | grep -v "\\$"',
            shell=True, capture_output=True, text=True
        )
        if all_cdcms.stdout.strip():
            print("Classes principais encontradas:")
            for line in all_cdcms.stdout.strip().split('\n'):
                print(f"  - {line}")
        else:
            print("[ERRO] Nenhuma classe principal CDCMS encontrada!")
            print("O JAR pode estar incompleto.")
else:
    print("[ERRO] cdcms_cil.jar nao encontrado!")

# =============================================================================
# PASSO 2: Construir classpath
# =============================================================================
print("\n--- PASSO 2: Construindo classpath ---")

if CDCMS_JAR.exists():
    jar_size = CDCMS_JAR.stat().st_size / (1024*1024)
    print(f"JAR: {CDCMS_JAR.name} ({jar_size:.1f} MB)")

    classpath_parts = [str(CDCMS_JAR)]

    # Se JAR < 10MB, precisa das dependencias
    if jar_size < 10:
        print("JAR pequeno - adicionando dependencias...")

        # Adicionar dependencias do Maven se existirem
        deps_dir = CDCMS_MOA_DIR / 'deps'
        if deps_dir.exists():
            dep_jars = list(deps_dir.glob('*.jar'))
            for jar in dep_jars:
                classpath_parts.append(str(jar))
            print(f"  Dependencias Maven: {len(dep_jars)} JARs")

        # Adicionar MOA local
        if MOA_JAR.exists():
            classpath_parts.append(str(MOA_JAR))
            print(f"  MOA local: adicionado")

        if MOA_LIB_DIR.exists():
            lib_jars = list(MOA_LIB_DIR.glob('*.jar'))
            for jar in lib_jars:
                classpath_parts.append(str(jar))
            print(f"  MOA lib/: {len(lib_jars)} JARs")
    else:
        print("JAR grande (fat jar) - usando apenas ele")

    full_classpath = ':'.join(classpath_parts)
    print(f"\nTotal JARs no classpath: {len(classpath_parts)}")
else:
    print("[ERRO] cdcms_cil.jar nao encontrado!")
    full_classpath = None

# =============================================================================
# PASSO 3: Teste de validação do classpath
# =============================================================================
print("\n--- PASSO 3: Validando classpath ---")

if full_classpath:
    # Testar se consegue carregar a classe
    test_cmd = [
        "java", "-Xmx1g", "-cp", full_classpath,
        "moa.classifiers.meta.CDCMS_CIL_GMean", "--help"
    ]

    print("Testando carregamento da classe CDCMS_CIL_GMean...")
    test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)

    if test_result.returncode == 0 or 'Usage' in test_result.stdout or 'Option' in test_result.stdout:
        print("[OK] Classe carregada com sucesso!")
    else:
        print(f"[AVISO] Return code: {test_result.returncode}")
        if test_result.stderr:
            # Procurar por ClassNotFoundException
            if 'ClassNotFoundException' in test_result.stderr:
                print("[ERRO] ClassNotFoundException detectado!")
                for line in test_result.stderr.split('\n')[:5]:
                    if line.strip():
                        print(f"  {line[:100]}")
            elif 'NoClassDefFoundError' in test_result.stderr:
                print("[ERRO] NoClassDefFoundError detectado!")
                for line in test_result.stderr.split('\n')[:5]:
                    if line.strip():
                        print(f"  {line[:100]}")
            else:
                # Pode ser só porque não tem main(), o que é OK
                print("[INFO] Pode ser esperado (classe não tem main)")

# =============================================================================
# PASSO 4: Executar CDCMS.CIL_GMean
# =============================================================================
print("\n--- PASSO 4: Executando CDCMS.CIL_GMean ---")

if full_classpath:
    # Parametros
    chunk_size = 500
    ensemble_size = 10
    time_steps = 500

    # Usar caminhos absolutos
    test_arff_abs = str(test_arff.resolve())
    output_file_abs = str(output_file.resolve())

    # CORREÇÃO: Usar caminho completo para ArffFileStream
    learner = f"(moa.classifiers.meta.CDCMS_CIL_GMean -s {ensemble_size} -t {time_steps})"
    stream = f"(moa.streams.ArffFileStream -f {test_arff_abs})"

    # Task com caminho absoluto para output
    task = f"EvaluateInterleavedTestThenTrain -s {stream} -l {learner} -f {chunk_size} -d {output_file_abs}"

    cmd = ["java", "-Xmx4g", "-cp", full_classpath, "moa.DoTask", task]

    print(f"\nConfiguração:")
    print(f"  Learner: CDCMS_CIL_GMean")
    print(f"  Ensemble size: {ensemble_size}")
    print(f"  Time steps interval: {time_steps}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Input: {test_arff_abs}")
    print(f"  Output: {output_file_abs}")
    print(f"  Working dir: {os.getcwd()}")

    print(f"\nComando completo:")
    print(f"  java -Xmx4g -cp <{len(classpath_parts)} JARs> moa.DoTask \\")
    print(f"    \"EvaluateInterleavedTestThenTrain \\")
    print(f"     -s (moa.streams.ArffFileStream -f ...) \\")
    print(f"     -l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500) \\")
    print(f"     -f 500 -d ...\"")

    print(f"\nExecutando...")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(TEST_DIR)  # Executar no diretório de teste
        )

        duration = time.time() - start_time

        # Salvar log COMPLETO (STDOUT + STDERR)
        with open(log_file, 'w') as f:
            f.write(f"="*60 + "\n")
            f.write(f"CDCMS.CIL_GMean Execution Log\n")
            f.write(f"="*60 + "\n\n")
            f.write(f"Duration: {duration:.1f}s\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Working dir: {TEST_DIR}\n\n")
            f.write(f"--- STDOUT ---\n{result.stdout}\n\n")
            f.write(f"--- STDERR ---\n{result.stderr}\n")

        print(f"\nTempo: {duration:.1f}s")
        print(f"Return code: {result.returncode}")

        # Mostrar STDOUT se houver
        if result.stdout.strip():
            print(f"\nSTDOUT:")
            for line in result.stdout.strip().split('\n')[:10]:
                print(f"  {line[:100]}")

        # Mostrar STDERR se houver erros
        if result.stderr.strip():
            print(f"\nSTDERR:")
            for line in result.stderr.strip().split('\n')[:10]:
                print(f"  {line[:100]}")

        if result.returncode == 0:
            print("\n[OK] Execucao concluida!")
        else:
            print(f"\n[ERRO] Falhou com codigo {result.returncode}")

        # =============================================================================
        # PASSO 5: Verificar arquivo de saída
        # =============================================================================
        print("\n--- PASSO 5: Verificando arquivo de saída ---")

        # Procurar em múltiplos locais
        possible_locations = [
            output_file,                          # Caminho especificado
            TEST_DIR / 'cdcms_output.csv',        # No diretório de teste
            WORK_DIR / 'cdcms_output.csv',        # No diretório de trabalho
            Path('cdcms_output.csv'),             # Diretório atual
            Path(output_file_abs),                # Caminho absoluto
        ]

        found_output = None
        for loc in possible_locations:
            if loc.exists() and loc.stat().st_size > 0:
                found_output = loc
                print(f"[OK] Arquivo encontrado: {loc}")
                print(f"     Tamanho: {loc.stat().st_size} bytes")
                break

        if not found_output:
            print("[AVISO] Arquivo de saída NÃO encontrado em nenhum local!")
            print("Locais verificados:")
            for loc in possible_locations:
                print(f"  - {loc}: {'existe' if loc.exists() else 'não existe'}")

            # Listar arquivos no diretório de teste
            print(f"\nArquivos em {TEST_DIR}:")
            for f in TEST_DIR.iterdir():
                print(f"  - {f.name}")
        else:
            # Copiar para local esperado se encontrado em outro lugar
            if found_output != output_file:
                import shutil
                shutil.copy(found_output, output_file)
                print(f"[OK] Copiado para: {output_file}")

    except subprocess.TimeoutExpired:
        print("[ERRO] Timeout!")
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# PASSO 6: Diagnóstico adicional se falhou
# =============================================================================
if not output_file.exists() or output_file.stat().st_size == 0:
    print("\n" + "="*60)
    print("DIAGNÓSTICO ADICIONAL")
    print("="*60)

    # Tentar executar com apenas um classificador simples para validar o MOA
    print("\nTestando MOA com classificador simples (NaiveBayes)...")

    simple_learner = "(moa.classifiers.bayes.NaiveBayes)"
    simple_stream = f"(moa.streams.ArffFileStream -f {test_arff_abs})"
    simple_output = str((TEST_DIR / 'simple_test.csv').resolve())
    simple_task = f"EvaluateInterleavedTestThenTrain -s {simple_stream} -l {simple_learner} -f 500 -d {simple_output}"

    simple_cmd = ["java", "-Xmx2g", "-cp", full_classpath, "moa.DoTask", simple_task]

    simple_result = subprocess.run(simple_cmd, capture_output=True, text=True, timeout=60)

    if Path(simple_output).exists():
        print(f"[OK] MOA funcionando! Output simples criado ({Path(simple_output).stat().st_size} bytes)")
        print("     O problema é específico do CDCMS_CIL_GMean")
    else:
        print("[ERRO] MOA não está funcionando corretamente!")
        print(f"STDERR: {simple_result.stderr[:500]}")

print("\n" + "="*60)
print("FIM DA EXECUÇÃO")
print("="*60)
