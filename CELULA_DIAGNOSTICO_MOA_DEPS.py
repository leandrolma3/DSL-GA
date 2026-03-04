# =============================================================================
# DIAGNÓSTICO: Analisar conteúdo do MOA-dependencies.jar
# =============================================================================
# Este script verifica se o MOA-dependencies.jar do ROSE contém todas as
# classes necessárias para executar CDCMS.CIL
#
# HIPÓTESE: O MOA-dependencies.jar é uma versão customizada que só funciona
# com ROSE, não com outros algoritmos
# =============================================================================

import subprocess
from pathlib import Path

print("="*70)
print("DIAGNÓSTICO: Conteúdo do MOA-dependencies.jar")
print("="*70)

# Caminhos
MOA_DEPS_JAR = WORK_DIR / 'rose_jars' / 'MOA-dependencies.jar'
ROSE_JAR = WORK_DIR / 'rose_jars' / 'ROSE-1.0.jar'

if not MOA_DEPS_JAR.exists():
    print("[ERRO] MOA-dependencies.jar não encontrado!")
else:
    print(f"\nArquivo: {MOA_DEPS_JAR.name}")
    print(f"Tamanho: {MOA_DEPS_JAR.stat().st_size / (1024*1024):.1f} MB")

    # =============================================================================
    # TESTE 1: Verificar se contém classes de Tasks do MOA
    # =============================================================================
    print("\n" + "-"*50)
    print("TESTE 1: Classes de Tasks do MOA")
    print("-"*50)

    # Procurar por tasks específicas
    tasks_to_check = [
        "moa/tasks/EvaluateInterleavedTestThenTrain",
        "moa/tasks/WriteCommandLineTemplate",
        "moa/tasks/MainTask",
        "moa/DoTask"
    ]

    for task in tasks_to_check:
        result = subprocess.run(
            f'jar tf "{MOA_DEPS_JAR}" 2>/dev/null | grep "{task}"',
            shell=True, capture_output=True, text=True
        )
        found = bool(result.stdout.strip())
        status = "[OK]" if found else "[X] FALTANDO"
        print(f"  {task}: {status}")
        if found:
            print(f"      -> {result.stdout.strip().split(chr(10))[0]}")

    # =============================================================================
    # TESTE 2: Contar classes por pacote
    # =============================================================================
    print("\n" + "-"*50)
    print("TESTE 2: Distribuição de classes por pacote")
    print("-"*50)

    result = subprocess.run(
        f'jar tf "{MOA_DEPS_JAR}" 2>/dev/null | grep "\\.class$"',
        shell=True, capture_output=True, text=True
    )
    all_classes = result.stdout.strip().split('\n') if result.stdout.strip() else []

    # Contar por pacote principal
    packages = {}
    for cls in all_classes:
        if '/' in cls:
            pkg = cls.split('/')[0]
            packages[pkg] = packages.get(pkg, 0) + 1

    print(f"  Total de classes: {len(all_classes)}")
    print(f"  Pacotes principais:")
    for pkg in sorted(packages.keys()):
        if packages[pkg] > 100:
            print(f"    {pkg}: {packages[pkg]} classes")

    # =============================================================================
    # TESTE 3: Verificar classes Weka (necessárias para clusterers)
    # =============================================================================
    print("\n" + "-"*50)
    print("TESTE 3: Classes Weka (necessárias para CDCMS.CIL)")
    print("-"*50)

    weka_classes_to_check = [
        "weka/clusterers/EM",
        "weka/clusterers/SimpleKMeans",
        "weka/core/Instances",
        "weka/classifiers/Classifier"
    ]

    for cls in weka_classes_to_check:
        result = subprocess.run(
            f'jar tf "{MOA_DEPS_JAR}" 2>/dev/null | grep "{cls}"',
            shell=True, capture_output=True, text=True
        )
        found = bool(result.stdout.strip())
        status = "[OK]" if found else "[X] FALTANDO"
        print(f"  {cls}: {status}")

    # =============================================================================
    # TESTE 4: Testar moa.DoTask listando tasks disponíveis
    # =============================================================================
    print("\n" + "-"*50)
    print("TESTE 4: Listar tasks disponíveis via moa.DoTask")
    print("-"*50)

    # Classpath simples com MOA-dependencies.jar
    classpath = str(MOA_DEPS_JAR)

    # Testar sem argumentos (deve mostrar help/tasks)
    test_cmd = f'java -Xmx1g -cp "{classpath}" moa.DoTask 2>&1 | head -30'
    result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=30)

    print("Saída de moa.DoTask sem argumentos:")
    output = result.stdout + result.stderr
    for line in output.strip().split('\n')[:15]:
        print(f"  {line[:80]}")

    # =============================================================================
    # TESTE 5: Comparar com como ROSE é executado
    # =============================================================================
    print("\n" + "-"*50)
    print("TESTE 5: Testar execução do ROSE (referência)")
    print("-"*50)

    if ROSE_JAR.exists():
        # Classpath igual ao que funciona
        cp_rose = f"{ROSE_JAR}:{MOA_DEPS_JAR}"

        # Testar se ROSE pode ser listado
        rose_test = f'java -Xmx1g -cp "{cp_rose}" moa.DoTask "WriteCommandLineTemplate -l moa.classifiers.meta.imbalanced.ROSE" 2>&1 | head -10'
        result = subprocess.run(rose_test, shell=True, capture_output=True, text=True, timeout=30)

        output = result.stdout + result.stderr
        if "Exception" in output:
            print("  ROSE WriteCommandLineTemplate: [FALHA]")
            for line in output.strip().split('\n')[:3]:
                print(f"    {line[:70]}")
        else:
            print("  ROSE WriteCommandLineTemplate: [OK]")
            for line in output.strip().split('\n')[:5]:
                print(f"    {line[:70]}")

        # Testar execução real do ROSE com arquivo de teste
        print("\n  Testando execução real do ROSE...")
        output_rose_test = TEST_DIR / 'rose_test_output.csv'

        rose_run = [
            "java", "-Xmx2g",
            "-cp", cp_rose,
            "moa.DoTask",
            f"EvaluateInterleavedTestThenTrain -s (ArffFileStream -f {test_arff}) -l (moa.classifiers.meta.imbalanced.ROSE) -f 500 -d {output_rose_test}"
        ]

        try:
            result = subprocess.run(rose_run, capture_output=True, text=True, timeout=120)
            if output_rose_test.exists() and output_rose_test.stat().st_size > 0:
                print(f"  ROSE Execução: [OK] - {output_rose_test.stat().st_size} bytes")
            else:
                print(f"  ROSE Execução: [FALHA]")
                if result.stderr:
                    for line in result.stderr.strip().split('\n')[:3]:
                        print(f"    {line[:70]}")
        except Exception as e:
            print(f"  ROSE Execução: [ERRO] {e}")
    else:
        print("  [SKIP] ROSE-1.0.jar não encontrado")

    # =============================================================================
    # TESTE 6: Verificar se o problema é a versão do MOA
    # =============================================================================
    print("\n" + "-"*50)
    print("TESTE 6: Verificar versão do MOA no MOA-dependencies.jar")
    print("-"*50)

    # Procurar por arquivos de manifesto ou versão
    version_check = subprocess.run(
        f'unzip -p "{MOA_DEPS_JAR}" META-INF/MANIFEST.MF 2>/dev/null | head -20',
        shell=True, capture_output=True, text=True
    )

    if version_check.stdout.strip():
        print("  MANIFEST.MF:")
        for line in version_check.stdout.strip().split('\n')[:10]:
            print(f"    {line}")

    # Verificar se há moa/gui/GUI.class (indicador de MOA completo)
    gui_check = subprocess.run(
        f'jar tf "{MOA_DEPS_JAR}" 2>/dev/null | grep "moa/gui/GUI.class"',
        shell=True, capture_output=True, text=True
    )

    if gui_check.stdout.strip():
        print("\n  MOA GUI presente: [SIM] - MOA parece completo")
    else:
        print("\n  MOA GUI presente: [NÃO] - MOA pode estar incompleto")

# =============================================================================
# CONCLUSÃO E PRÓXIMOS PASSOS
# =============================================================================
print("\n" + "="*70)
print("CONCLUSÃO")
print("="*70)

print("""
Com base nos testes acima, podemos determinar:

1. Se MOA-dependencies.jar contém todas as tasks necessárias
2. Se o problema é específico do CDCMS ou do MOA em si
3. Se precisamos usar uma versão diferente do MOA

PRÓXIMOS PASSOS POSSÍVEIS:

A) Se MOA-dependencies.jar está incompleto:
   -> Baixar MOA oficial e usar seus JARs

B) Se ROSE funciona mas CDCMS não:
   -> O problema pode estar na forma como CDCMS registra suas classes
   -> Pode ser necessário recompilar CDCMS contra a versão correta do MOA

C) Se nem ROSE funciona:
   -> Problema com Java ou ambiente
""")
