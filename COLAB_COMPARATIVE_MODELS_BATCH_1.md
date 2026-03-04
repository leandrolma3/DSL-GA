# EXECUCAO MODELOS COMPARATIVOS - BATCH 1 (Google Colab)

Execute os modelos River (HAT, ARF, SRP), ACDWM e ERulesD2S nos chunks ja gerados pelo GBML Batch 1.

---

## CELULA 1: Montar Drive e Setup Inicial

```python
from google.colab import drive
import os
import sys
from pathlib import Path

# Montar Drive
drive.mount('/content/drive')

# Definir paths
DRIVE_BASE = "/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid"
WORK_DIR = "/content/DSL-AG-hybrid"

# Verificar se existe
if not os.path.exists(DRIVE_BASE):
    print(f"ERRO: Diretorio nao encontrado: {DRIVE_BASE}")
    raise FileNotFoundError(DRIVE_BASE)

print(f"Drive montado com sucesso!")
print(f"Diretorio base: {DRIVE_BASE}")
```

---

## CELULA 2: Copiar Repositorio para /content

```python
# Remover diretorio antigo se existir
if os.path.exists(WORK_DIR):
    print("Removendo diretorio antigo...")
    !rm -rf {WORK_DIR}

# Copiar do Drive (necessario para executar scripts Python)
print(f"Copiando repositorio...")
!cp -r {DRIVE_BASE} {WORK_DIR}

# Mudar para diretorio de trabalho
os.chdir(WORK_DIR)
print(f"Diretorio de trabalho: {os.getcwd()}")

# Verificar estrutura
print("\nArquivos principais:")
!ls -lh *.py | head -10
```

---

## CELULA 3: Instalar Dependencias

```python
# Instalar dependencias Python
print("Instalando dependencias...")
!pip install -q river scikit-learn pyyaml matplotlib seaborn pandas numpy

# Verificar instalacao
import river
from river import tree, ensemble, drift
print(f"\nRiver version: {river.__version__}")
print("Dependencias instaladas com sucesso!")
```

---

## CELULA 4: Clonar ACDWM Repository

```python
# Clonar repositorio ACDWM se nao existir
ACDWM_DIR = "/content/ACDWM"

if not os.path.exists(ACDWM_DIR):
    print("Clonando repositorio ACDWM...")
    !git clone https://github.com/jasonyanglu/ACDWM.git {ACDWM_DIR}
    print(f"ACDWM clonado em: {ACDWM_DIR}")
else:
    print(f"ACDWM ja existe em: {ACDWM_DIR}")

# Verificar arquivos
print("\nArquivos ACDWM:")
!ls -lh {ACDWM_DIR}/*.py | head -5
```

---

## CELULA 5: Verificar Chunks Disponiveis

```python
import pandas as pd

# Diretorio de resultados GBML
RESULTS_DIR = Path(WORK_DIR) / "experiments_6chunks_phase1_gbml" / "batch_1"

# Datasets do Batch 1
DATASETS = [
    "SEA_Abrupt_Simple",
    "AGRAWAL_Abrupt_Simple_Severe",
    "RBF_Abrupt_Severe",
    "HYPERPLANE_Abrupt_Simple",
    "STAGGER_Abrupt_Chain"
]

print("="*80)
print("VERIFICACAO DE CHUNKS DISPONIVEIS")
print("="*80)

for dataset in DATASETS:
    chunk_dir = RESULTS_DIR / dataset / "run_1" / "chunk_data"

    if chunk_dir.exists():
        train_chunks = list(chunk_dir.glob("chunk_*_train.csv"))
        test_chunks = list(chunk_dir.glob("chunk_*_test.csv"))

        print(f"\n{dataset}:")
        print(f"  Train chunks: {len(train_chunks)}")
        print(f"  Test chunks: {len(test_chunks)}")

        # Verificar tamanho do primeiro chunk de teste
        if test_chunks:
            first_test = sorted(test_chunks)[0]
            df = pd.read_csv(first_test)
            print(f"  Exemplo chunk shape: {df.shape}")
    else:
        print(f"\n{dataset}: CHUNKS NAO ENCONTRADOS!")
        print(f"  Path esperado: {chunk_dir}")

print("\n" + "="*80)
```

---

## CELULA 6: Executar River Models + ACDWM (1 Dataset)

Execute esta celula para processar UM dataset por vez:

```python
import subprocess
import time
from datetime import datetime

# ESCOLHER DATASET (mude conforme necessario)
DATASET_NAME = "SEA_Abrupt_Simple"  # Altere para outros datasets

# Configuracoes
BASE_DIR = Path(WORK_DIR) / "experiments_6chunks_phase1_gbml" / "batch_1"

print("="*80)
print(f"EXECUTANDO MODELOS COMPARATIVOS - {DATASET_NAME}")
print("="*80)
print(f"Inicio: {datetime.now()}")
print()

# Comando para executar run_comparative_on_existing_chunks.py
# Este script CARREGA os chunks salvos pelo GBML (nao regenera!)
cmd = [
    sys.executable,
    'run_comparative_on_existing_chunks.py',
    '--dataset', DATASET_NAME,
    '--base-dir', str(BASE_DIR),
    '--models', 'HAT', 'ARF', 'SRP',
    '--acdwm',
    '--acdwm-path', ACDWM_DIR,
    '--run', '1'
]

print("Comando:")
print(" ".join(cmd))
print()

start_time = time.time()

try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )

    duration = time.time() - start_time

    print("="*80)
    print("EXECUCAO CONCLUIDA!")
    print("="*80)
    print(f"Duracao: {duration/60:.2f} minutos")
    print(f"\nStdout:\n{result.stdout}")

    # Listar resultados
    dataset_dir = BASE_DIR / DATASET_NAME / "run_1"
    print(f"\nDiretorio de resultados: {dataset_dir}")
    print("\nArquivos gerados:")
    !ls -lh {dataset_dir}/*.csv

except subprocess.CalledProcessError as e:
    print("="*80)
    print("ERRO NA EXECUCAO!")
    print("="*80)
    print(f"Codigo de erro: {e.returncode}")
    print(f"\nStderr:\n{e.stderr}")
    print(f"\nStdout:\n{e.stdout}")
```

---

## CELULA 7: Executar TODOS os Datasets em Loop (River + ACDWM + ERulesD2S)

Execute esta celula para processar todos os 5 datasets sequencialmente com TODOS os modelos:

```python
import subprocess
import time
from datetime import datetime
import pandas as pd

DATASETS = [
    "SEA_Abrupt_Simple",
    "AGRAWAL_Abrupt_Simple_Severe",
    "RBF_Abrupt_Severe",
    "HYPERPLANE_Abrupt_Simple",
    "STAGGER_Abrupt_Chain"
]

BASE_DIR = Path(WORK_DIR) / "experiments_6chunks_phase1_gbml" / "batch_1"

print("="*80)
print("EXECUTANDO MODELOS COMPARATIVOS - BATCH 1 COMPLETO")
print("="*80)
print(f"Datasets: {len(DATASETS)}")
print(f"Modelos: HAT, ARF, SRP, ACDWM, ERulesD2S")
print(f"Inicio: {datetime.now()}")
print("="*80)

results_summary = []
total_start = time.time()

for idx, dataset in enumerate(DATASETS, 1):
    print(f"\n{'='*80}")
    print(f"[{idx}/{len(DATASETS)}] Processando: {dataset}")
    print('='*80)

    dataset_start = time.time()

    cmd = [
        sys.executable,
        'run_comparative_on_existing_chunks.py',
        '--dataset', dataset,
        '--base-dir', str(BASE_DIR),
        '--models', 'HAT', 'ARF', 'SRP',
        '--acdwm',
        '--acdwm-path', ACDWM_DIR,
        '--erulesd2s',
        '--erulesd2s-pop', '25',
        '--erulesd2s-gen', '50',
        '--run', '1'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=7200  # 2 horas timeout por dataset (ERulesD2S pode demorar)
        )

        duration = time.time() - dataset_start
        status = "SUCESSO"

        print(f"\n{dataset}: {status} ({duration/60:.2f} min)")

        results_summary.append({
            'dataset': dataset,
            'status': status,
            'duration_min': duration/60
        })

    except subprocess.CalledProcessError as e:
        duration = time.time() - dataset_start
        status = "FALHA"

        print(f"\n{dataset}: {status} ({duration/60:.2f} min)")
        print(f"Erro: {e.stderr[:200]}")

        results_summary.append({
            'dataset': dataset,
            'status': status,
            'duration_min': duration/60,
            'error': str(e)[:100]
        })

    except subprocess.TimeoutExpired:
        print(f"\n{dataset}: TIMEOUT (>2h)")
        results_summary.append({
            'dataset': dataset,
            'status': 'TIMEOUT',
            'duration_min': 120
        })

total_duration = time.time() - total_start

print("\n" + "="*80)
print("RESUMO FINAL")
print("="*80)
print(f"Duracao total: {total_duration/3600:.2f} horas")
print(f"Datasets processados: {len([r for r in results_summary if r['status'] == 'SUCESSO'])}/{len(DATASETS)}")
print()

# Tabela resumo
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))
print("="*80)
```

---

## CELULA 8: Instalar Java e Maven para ERulesD2S

```python
# Instalar Java 11 e Maven
print("Instalando Java 11 e Maven...")

!apt-get update -qq
!apt-get install -y -qq openjdk-11-jdk maven > /dev/null 2>&1

# Configurar JAVA_HOME
import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'

# Verificar instalacao
!java -version
!mvn -version

print("\nJava e Maven instalados com sucesso!")
```

---

## CELULA 9: Setup ERulesD2S

```python
# Executar setup do ERulesD2S
print("Configurando ERulesD2S...")

# Verificar se setup_erulesd2s.py existe
setup_script = Path(WORK_DIR) / "setup_erulesd2s.py"

if setup_script.exists():
    !python setup_erulesd2s.py

    # Verificar se JAR foi criado
    if Path("erulesd2s.jar").exists():
        print("\nERulesD2S configurado com sucesso!")
        print("JAR encontrado: erulesd2s.jar")

        # Verificar JCLEC4 (obrigatorio)
        jclec_jar = Path("lib/JCLEC4-base-1.0-jar-with-dependencies.jar")
        if jclec_jar.exists():
            print(f"JCLEC4 encontrado: {jclec_jar}")
        else:
            print("AVISO: JCLEC4 nao encontrado!")
    else:
        print("ERRO: erulesd2s.jar nao foi criado!")
else:
    print(f"ERRO: {setup_script} nao encontrado!")
    print("Pulando setup ERulesD2S")
```

---

## CELULA 10: Executar Apenas ERulesD2S (Opcional)

Se voce quiser executar APENAS ERulesD2S separadamente (sem River/ACDWM):

```python
import subprocess
import time
from datetime import datetime

# ESCOLHER DATASET
DATASET_NAME = "SEA_Abrupt_Simple"  # Altere conforme necessario

BASE_DIR = Path(WORK_DIR) / "experiments_6chunks_phase1_gbml" / "batch_1"

print("="*80)
print(f"EXECUTANDO APENAS ERULESD2S - {DATASET_NAME}")
print("="*80)
print(f"Inicio: {datetime.now()}")
print()

start_time = time.time()

# Executar apenas ERulesD2S (sem River, sem ACDWM)
cmd = [
    sys.executable,
    'run_comparative_on_existing_chunks.py',
    '--dataset', DATASET_NAME,
    '--base-dir', str(BASE_DIR),
    '--models',  # Sem modelos River
    '--erulesd2s',
    '--erulesd2s-pop', '25',
    '--erulesd2s-gen', '50',
    '--run', '1'
]

try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        timeout=3600
    )

    duration = time.time() - start_time

    print("="*80)
    print("EXECUCAO CONCLUIDA!")
    print("="*80)
    print(f"Duracao: {duration/60:.2f} minutos")
    print(f"\nStdout:\n{result.stdout}")

    # Verificar resultados
    dataset_dir = BASE_DIR / DATASET_NAME / "run_1"
    erulesd2s_file = dataset_dir / "erulesd2s_results.csv"

    if erulesd2s_file.exists():
        import pandas as pd
        df = pd.read_csv(erulesd2s_file)
        print(f"\nResultados ERulesD2S:")
        print(df)

except subprocess.CalledProcessError as e:
    print("="*80)
    print("ERRO NA EXECUCAO!")
    print("="*80)
    print(f"Codigo de erro: {e.returncode}")
    print(f"\nStderr:\n{e.stderr}")
```

---

## CELULA 11: Consolidar Resultados

```python
import pandas as pd
from pathlib import Path

BASE_DIR = Path(WORK_DIR) / "experiments_6chunks_phase1_gbml" / "batch_1"

DATASETS = [
    "SEA_Abrupt_Simple",
    "AGRAWAL_Abrupt_Simple_Severe",
    "RBF_Abrupt_Severe",
    "HYPERPLANE_Abrupt_Simple",
    "STAGGER_Abrupt_Chain"
]

print("="*80)
print("CONSOLIDACAO DE RESULTADOS - BATCH 1")
print("="*80)

all_results = []

for dataset in DATASETS:
    dataset_dir = BASE_DIR / dataset / "run_1"

    print(f"\nDataset: {dataset}")

    # Listar todos os CSVs de resultados (exceto chunks)
    result_files = [
        f for f in dataset_dir.glob("*.csv")
        if "chunk_" not in f.name and "comparative" not in f.name
    ]

    for csv_file in result_files:
        try:
            df = pd.read_csv(csv_file)

            # Adicionar coluna de dataset se nao existir
            if 'dataset' not in df.columns:
                df['dataset'] = dataset

            all_results.append(df)
            print(f"  - {csv_file.name}: {len(df)} linhas")

        except Exception as e:
            print(f"  ! Erro ao ler {csv_file.name}: {e}")

# Consolidar tudo
if all_results:
    consolidated = pd.concat(all_results, ignore_index=True)

    # Salvar consolidado na pasta batch_1
    consolidated_file = BASE_DIR / "batch_1_all_comparative_models.csv"
    consolidated.to_csv(consolidated_file, index=False)

    print(f"\n{'='*80}")
    print("RESULTADOS CONSOLIDADOS")
    print(f"{'='*80}")
    print(f"Total de linhas: {len(consolidated)}")
    print(f"Modelos: {sorted(consolidated['model'].unique())}")
    print(f"Datasets: {sorted(consolidated['dataset'].unique())}")
    print(f"\nArquivo salvo: {consolidated_file}")

    # Estatisticas resumidas por modelo
    print(f"\n{'='*80}")
    print("ESTATISTICAS POR MODELO (Media +/- Desvio Padrao)")
    print(f"{'='*80}")

    summary = consolidated.groupby('model').agg({
        'accuracy': ['mean', 'std', 'count'],
        'gmean': ['mean', 'std'],
        'f1_weighted': ['mean', 'std']
    }).round(4)

    print(summary)

    # Estatisticas por modelo E dataset
    print(f"\n{'='*80}")
    print("ESTATISTICAS POR MODELO E DATASET")
    print(f"{'='*80}")

    summary_by_dataset = consolidated.groupby(['dataset', 'model']).agg({
        'gmean': 'mean',
        'accuracy': 'mean'
    }).round(4)

    print(summary_by_dataset)

else:
    print("\nNenhum resultado encontrado!")
    print("Verifique se os modelos foram executados com sucesso")
```

---

## RESUMO DE USO

### Execucao Completa (Recomendada):

1. **CELULAS 1-3**: Setup inicial (5-10 min)
2. **CELULA 4**: Clonar ACDWM (1 min)
3. **CELULA 5**: Verificar chunks (30 seg)
4. **CELULAS 8-9**: Setup ERulesD2S (5 min)
5. **CELULA 7**: Executar TODOS modelos em TODOS datasets (4-6 horas)
6. **CELULA 11**: Consolidar resultados (1 min)

### Execucao Individual (por dataset):

Use **CELULA 6** para processar um dataset por vez (sem ERulesD2S)
Use **CELULA 10** para executar apenas ERulesD2S em um dataset

### Tempo Total Estimado:

- River + ACDWM + ERulesD2S (5 datasets): 4-6 horas
- Somente River + ACDWM (5 datasets): 2-3 horas
- Somente ERulesD2S (5 datasets): 2-3 horas

### Observacoes:

- Nao feche o navegador durante a execucao
- Use Colab Pro para garantir 24h de runtime
- Os chunks GBML ja existentes serao reutilizados
- Resultados salvos automaticamente no Drive

---

**Criado em**: 2025-11-18
**Para**: Batch 1 - Modelos Comparativos (River, ACDWM, ERulesD2S)
**Tempo estimado**: 3-5 horas
