# =============================================================================
# DIAGNOSTICO DE CONFLITOS DE CLASSES ENTRE JARs
# =============================================================================
# Este script verifica se cdcms_cil.jar contem classes MOA/Weka duplicadas
# que podem conflitar com MOA-dependencies.jar
#
# Se houver duplicatas, vamos criar uma versao "limpa" do cdcms_cil.jar
# contendo APENAS as classes CDCMS (sem MOA/Weka)
# =============================================================================

import subprocess
from pathlib import Path

print("="*70)
print("DIAGNOSTICO DE CONFLITOS DE CLASSES")
print("="*70)

# Caminhos
CDCMS_JAR = CDCMS_JARS_DIR / 'cdcms_cil.jar'
MOA_DEPS_JAR = WORK_DIR / 'rose_jars' / 'MOA-dependencies.jar'

if not CDCMS_JAR.exists():
    print("[ERRO] cdcms_cil.jar nao encontrado!")
else:
    print(f"\nAnalisando: {CDCMS_JAR.name}")
    print(f"Tamanho: {CDCMS_JAR.stat().st_size / (1024*1024):.2f} MB")

    # Listar todas as classes no cdcms_cil.jar
    result = subprocess.run(
        f'jar tf "{CDCMS_JAR}" | grep "\\.class$"',
        shell=True, capture_output=True, text=True
    )
    all_classes = result.stdout.strip().split('\n') if result.stdout.strip() else []
    print(f"Total de classes: {len(all_classes)}")

    # Separar classes por pacote
    moa_classes = [c for c in all_classes if c.startswith('moa/') and 'CDCMS' not in c]
    weka_classes = [c for c in all_classes if c.startswith('weka/')]
    cdcms_classes = [c for c in all_classes if 'CDCMS' in c]
    other_classes = [c for c in all_classes if c not in moa_classes + weka_classes + cdcms_classes]

    print(f"\nClasses por categoria:")
    print(f"  CDCMS (nosso codigo): {len(cdcms_classes)}")
    print(f"  MOA (potencial conflito): {len(moa_classes)}")
    print(f"  Weka (potencial conflito): {len(weka_classes)}")
    print(f"  Outras: {len(other_classes)}")

    # =============================================================================
    # VERIFICAR CLASSES CDCMS
    # =============================================================================
    print("\n--- Classes CDCMS (queremos manter) ---")
    for cls in sorted(cdcms_classes)[:10]:
        print(f"  {cls}")
    if len(cdcms_classes) > 10:
        print(f"  ... e mais {len(cdcms_classes) - 10}")

    # =============================================================================
    # VERIFICAR CONFLITOS MOA
    # =============================================================================
    if moa_classes:
        print("\n--- Classes MOA DUPLICADAS (conflito potencial) ---")
        # Mostrar as mais criticas
        critical_moa = [c for c in moa_classes if any(x in c for x in ['DoTask', 'Classifier', 'Task', 'Stream'])]
        for cls in sorted(critical_moa)[:15]:
            print(f"  [!] {cls}")
        if len(moa_classes) > 15:
            print(f"  ... e mais {len(moa_classes) - 15} classes MOA")

    # =============================================================================
    # VERIFICAR CONFLITOS WEKA
    # =============================================================================
    if weka_classes:
        print("\n--- Classes Weka DUPLICADAS (conflito potencial) ---")
        for cls in sorted(weka_classes)[:10]:
            print(f"  [!] {cls}")
        if len(weka_classes) > 10:
            print(f"  ... e mais {len(weka_classes) - 10} classes Weka")

    # =============================================================================
    # CRIAR JAR LIMPO (SE NECESSARIO)
    # =============================================================================
    if moa_classes or weka_classes:
        print("\n" + "="*70)
        print("CRIANDO cdcms_cil_clean.jar (apenas classes CDCMS)")
        print("="*70)

        clean_dir = TEST_DIR / 'cdcms_clean_extract'
        clean_dir.mkdir(exist_ok=True)

        # Extrair JAR
        print("\n1. Extraindo JAR original...")
        subprocess.run(f'cd "{clean_dir}" && jar xf "{CDCMS_JAR}"', shell=True)

        # Remover classes conflitantes
        print("2. Removendo classes MOA/Weka conflitantes...")

        # Remover pastas MOA (exceto moa/classifiers/meta onde esta CDCMS)
        moa_dirs_to_remove = ['moa/tasks', 'moa/core', 'moa/options', 'moa/streams',
                              'moa/evaluation', 'moa/gui', 'moa/capabilities']
        for dir_name in moa_dirs_to_remove:
            dir_path = clean_dir / dir_name
            if dir_path.exists():
                subprocess.run(f'rm -rf "{dir_path}"', shell=True)
                print(f"   Removido: {dir_name}/")

        # Remover arquivos MOA na raiz (como DoTask.class)
        moa_root = clean_dir / 'moa'
        if moa_root.exists():
            for f in moa_root.glob('*.class'):
                f.unlink()
                print(f"   Removido: moa/{f.name}")

        # Remover pasta weka inteira
        weka_dir = clean_dir / 'weka'
        if weka_dir.exists():
            subprocess.run(f'rm -rf "{weka_dir}"', shell=True)
            print(f"   Removido: weka/")

        # Recriar JAR limpo
        print("3. Criando JAR limpo...")
        clean_jar = CDCMS_JARS_DIR / 'cdcms_cil_clean.jar'
        subprocess.run(f'cd "{clean_dir}" && jar cf "{clean_jar}" .', shell=True)

        if clean_jar.exists():
            print(f"\n[OK] JAR limpo criado: {clean_jar}")
            print(f"    Tamanho original: {CDCMS_JAR.stat().st_size / 1024:.1f} KB")
            print(f"    Tamanho limpo: {clean_jar.stat().st_size / 1024:.1f} KB")

            # Verificar classes no JAR limpo
            result = subprocess.run(
                f'jar tf "{clean_jar}" | grep "CDCMS" | head -5',
                shell=True, capture_output=True, text=True
            )
            print(f"\nClasses CDCMS no JAR limpo:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")

            # Verificar se ainda tem MOA/Weka
            result_moa = subprocess.run(
                f'jar tf "{clean_jar}" | grep -E "^moa/(DoTask|core|task)" | head -5',
                shell=True, capture_output=True, text=True
            )
            if result_moa.stdout.strip():
                print(f"\n[AVISO] Ainda ha classes MOA conflitantes!")
            else:
                print(f"\n[OK] Sem classes MOA conflitantes!")

        # Limpar diretorio temporario
        subprocess.run(f'rm -rf "{clean_dir}"', shell=True)

    else:
        print("\n[OK] Nenhum conflito detectado! cdcms_cil.jar esta limpo.")

# =============================================================================
# TESTE RAPIDO COM JAR LIMPO
# =============================================================================
clean_jar = CDCMS_JARS_DIR / 'cdcms_cil_clean.jar'

if clean_jar.exists() and MOA_DEPS_JAR.exists():
    print("\n" + "="*70)
    print("TESTE RAPIDO COM JAR LIMPO")
    print("="*70)

    # Classpath: JAR limpo + MOA-dependencies
    classpath = f"{clean_jar}:{MOA_DEPS_JAR}"
    print(f"\nClasspath: {clean_jar.name}:{MOA_DEPS_JAR.name}")

    # Testar WriteCommandLineTemplate
    print("\nTestando carregamento da classe CDCMS_CIL_GMean...")
    test_cmd = f'java -Xmx2g -cp "{classpath}" moa.DoTask "WriteCommandLineTemplate -l moa.classifiers.meta.CDCMS_CIL_GMean"'
    result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=30)

    if result.returncode == 0:
        print("[OK] Classe carregada com sucesso!")
        if result.stdout:
            print("Template:")
            for line in result.stdout.strip().split('\n')[:5]:
                print(f"  {line}")
    else:
        print(f"[ERRO] Falha ao carregar classe")
        if result.stderr:
            print(f"STDERR: {result.stderr[:300]}")

    # Testar execucao real
    print("\n--- Teste de execucao real ---")
    test_arff_abs = str(test_arff.resolve())
    output_test = TEST_DIR / 'cdcms_clean_test.csv'

    task_parts = [
        "EvaluateInterleavedTestThenTrain",
        f"-s (ArffFileStream -f {test_arff_abs})",
        "-l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2)",
        "-f 500",
        f"-d {output_test}"
    ]
    task_str = " ".join(task_parts)

    run_cmd = f'java -Xmx4g -cp "{classpath}" moa.DoTask "{task_str}"'

    print("Executando teste...")
    result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True, timeout=120)

    print(f"Return code: {result.returncode}")
    if output_test.exists() and output_test.stat().st_size > 0:
        print(f"[SUCESSO] Arquivo criado: {output_test.stat().st_size} bytes")
    else:
        print(f"[FALHA] Arquivo nao criado")
        if result.stderr:
            for line in result.stderr.strip().split('\n')[:5]:
                print(f"  {line[:100]}")
