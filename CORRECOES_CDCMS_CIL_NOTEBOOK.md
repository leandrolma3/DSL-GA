# Correções Necessárias no Setup_CDCMS_CIL.ipynb

**Data:** 2026-01-24
**Baseado em:** Análise dos logs da última execução

---

## Resumo do Problema

A compilação foi bem-sucedida (1326 classes, JAR de 2.08 MB), mas a execução não gerou o arquivo de resultados. O tempo de 0.9s para 5000 samples é suspeito.

---

## Problemas Identificados

### 1. ArffFileStream não qualificado
**Atual:**
```python
stream = f"(ArffFileStream -f {test_arff})"
```

**Corrigido:**
```python
stream = f"(moa.streams.ArffFileStream -f {test_arff})"
```

### 2. Falta verificação da classe principal
O log mostrou apenas classes internas (`$1`, `$ClassifierWithInfo`), mas não confirmou se `CDCMS_CIL_GMean.class` existe.

**Adicionar:**
```python
# Verificar classe principal (não apenas internas)
check = subprocess.run(
    f'jar tf "{CDCMS_JAR}" | grep "CDCMS_CIL_GMean.class$"',
    shell=True, capture_output=True, text=True
)
```

### 3. STDOUT não capturado no log
O log salvo mostra apenas STDERR, mas erros podem estar no STDOUT.

**Corrigido:**
```python
with open(log_file, 'w') as f:
    f.write(f"STDOUT:\n{result.stdout}\n\n")
    f.write(f"STDERR:\n{result.stderr}\n")
```

### 4. Caminhos relativos podem falhar
Usar caminhos absolutos para evitar problemas com diretório de trabalho.

**Corrigido:**
```python
test_arff_abs = str(test_arff.resolve())
output_file_abs = str(output_file.resolve())
```

### 5. Falta verificação do arquivo em múltiplos locais
O arquivo pode ter sido criado em local diferente do esperado.

---

## Célula 3.2 Corrigida

```python
# CELULA 3.2: Executar CDCMS.CIL_GMean (CORRIGIDA)
import subprocess
import time
import os
from pathlib import Path

print("="*60)
print("EXECUTANDO CDCMS.CIL_GMean")
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
print("\n--- Verificando classe CDCMS_CIL_GMean ---")

if CDCMS_JAR.exists():
    check_class = subprocess.run(
        f'jar tf "{CDCMS_JAR}" | grep "CDCMS_CIL_GMean.class$"',
        shell=True, capture_output=True, text=True
    )

    if check_class.stdout.strip():
        print(f"[OK] Classe principal encontrada: {check_class.stdout.strip()}")
    else:
        print("[AVISO] Classe CDCMS_CIL_GMean.class NAO encontrada!")
        # Listar todas as classes principais (sem $)
        all_main = subprocess.run(
            f'jar tf "{CDCMS_JAR}" | grep -E "CDCMS.*\\.class$" | grep -v "\\$"',
            shell=True, capture_output=True, text=True
        )
        if all_main.stdout.strip():
            print("Classes principais disponíveis:")
            print(all_main.stdout)

# =============================================================================
# PASSO 2: Construir classpath
# =============================================================================
if CDCMS_JAR.exists():
    jar_size = CDCMS_JAR.stat().st_size / (1024*1024)
    print(f"\nJAR: {CDCMS_JAR.name} ({jar_size:.1f} MB)")

    classpath_parts = [str(CDCMS_JAR)]

    if jar_size < 10:
        print("JAR pequeno - adicionando dependencias...")

        deps_dir = CDCMS_MOA_DIR / 'deps'
        if deps_dir.exists():
            for jar in deps_dir.glob('*.jar'):
                classpath_parts.append(str(jar))
            print(f"  Dependencias Maven: {len(list(deps_dir.glob('*.jar')))} JARs")

        if MOA_JAR.exists():
            classpath_parts.append(str(MOA_JAR))
            print(f"  MOA local: adicionado")

        if MOA_LIB_DIR.exists():
            for jar in MOA_LIB_DIR.glob('*.jar'):
                classpath_parts.append(str(jar))
            print(f"  MOA lib/: {len(list(MOA_LIB_DIR.glob('*.jar')))} JARs")

    full_classpath = ':'.join(classpath_parts)
    print(f"\nTotal JARs no classpath: {len(classpath_parts)}")
else:
    print("[ERRO] cdcms_cil.jar nao encontrado!")
    full_classpath = None

# =============================================================================
# PASSO 3: Executar
# =============================================================================
if full_classpath:
    # Parametros
    chunk_size = 500
    ensemble_size = 10
    time_steps = 500

    # CORREÇÃO 1: Usar caminhos absolutos
    test_arff_abs = str(test_arff.resolve())
    output_file_abs = str(output_file.resolve())

    # CORREÇÃO 2: Usar caminho completo para ArffFileStream
    learner = f"(moa.classifiers.meta.CDCMS_CIL_GMean -s {ensemble_size} -t {time_steps})"
    stream = f"(moa.streams.ArffFileStream -f {test_arff_abs})"
    task = f"EvaluateInterleavedTestThenTrain -s {stream} -l {learner} -f {chunk_size} -d {output_file_abs}"

    cmd = ["java", "-Xmx4g", "-cp", full_classpath, "moa.DoTask", task]

    print(f"\nLearner: CDCMS_CIL_GMean")
    print(f"Ensemble size: {ensemble_size}")
    print(f"Time steps interval: {time_steps}")
    print(f"Chunk size: {chunk_size}")
    print(f"Input (abs): {test_arff_abs}")
    print(f"Output (abs): {output_file_abs}")
    print(f"\nExecutando...")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(TEST_DIR)  # Definir diretório de trabalho
        )

        duration = time.time() - start_time

        # CORREÇÃO 3: Salvar log COMPLETO
        with open(log_file, 'w') as f:
            f.write(f"Duration: {duration:.1f}s\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Working dir: {TEST_DIR}\n\n")
            f.write(f"--- STDOUT ---\n{result.stdout}\n\n")
            f.write(f"--- STDERR ---\n{result.stderr}\n")

        print(f"\nTempo: {duration:.1f}s")
        print(f"Return code: {result.returncode}")

        # Mostrar saídas
        if result.stdout.strip():
            print(f"\nSTDOUT (primeiras linhas):")
            for line in result.stdout.strip().split('\n')[:5]:
                print(f"  {line[:100]}")

        if result.stderr.strip():
            print(f"\nSTDERR (primeiras linhas):")
            for line in result.stderr.strip().split('\n')[:5]:
                print(f"  {line[:100]}")

        if result.returncode == 0:
            print("\n[OK] Execucao concluida!")
        else:
            print(f"\n[ERRO] Falhou")

    except subprocess.TimeoutExpired:
        print("[ERRO] Timeout!")
    except Exception as e:
        print(f"[ERRO] {e}")

# =============================================================================
# PASSO 4: Verificar arquivo de saída
# =============================================================================
print("\n--- Verificando arquivo de saída ---")

# CORREÇÃO 4: Procurar em múltiplos locais
possible_locations = [
    output_file,
    TEST_DIR / 'cdcms_output.csv',
    WORK_DIR / 'cdcms_output.csv',
    Path('cdcms_output.csv'),
]

found = None
for loc in possible_locations:
    if loc.exists() and loc.stat().st_size > 0:
        found = loc
        print(f"[OK] Encontrado: {loc} ({loc.stat().st_size} bytes)")
        break

if not found:
    print("[AVISO] Arquivo NAO encontrado!")
    print(f"\nArquivos em {TEST_DIR}:")
    for f in TEST_DIR.iterdir():
        print(f"  - {f.name}")
```

---

## Diagnóstico Adicional

Se ainda falhar, adicionar após a execução:

```python
# Testar MOA com classificador simples
print("\nTestando MOA com NaiveBayes...")
simple_task = f"EvaluateInterleavedTestThenTrain -s (moa.streams.ArffFileStream -f {test_arff_abs}) -l (moa.classifiers.bayes.NaiveBayes) -f 500"
simple_cmd = ["java", "-Xmx2g", "-cp", full_classpath, "moa.DoTask", simple_task]
simple_result = subprocess.run(simple_cmd, capture_output=True, text=True, timeout=60)
print(f"NaiveBayes return code: {simple_result.returncode}")
```

---

## Checklist de Verificação

Antes de executar, confirmar:

- [ ] `CDCMS_CIL_GMean.class` existe no JAR (não apenas classes internas)
- [ ] Classpath inclui todas as dependências (moa.jar, weka, etc.)
- [ ] Caminhos são absolutos
- [ ] `ArffFileStream` usa pacote completo `moa.streams.ArffFileStream`
- [ ] Diretório de trabalho está definido

---

## Arquivo de Correção

O arquivo `CELULA_3_2_CORRIGIDA.py` contém a versão completa com todas as correções.

Para usar no notebook:
1. Copie o conteúdo do arquivo
2. Substitua a célula 3.2 original
3. Execute novamente

---

## Próximos Passos

1. Atualizar o notebook `Setup_CDCMS_CIL.ipynb` com as correções
2. Re-executar no Colab
3. Verificar se o arquivo de saída é gerado
4. Se funcionar, atualizar a função `run_cdcms_cil()` na Parte 5
