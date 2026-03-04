# CELULAS PARA COLAB - BATCH 1 EXECUCAO 1

Copie e cole cada celula abaixo no Google Colab, uma de cada vez.

---

## CELULA 1: Montar Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Aguarde**: Autorize o acesso ao Google Drive quando solicitado.

---

## CELULA 2: Configurar Variaveis e Preparar Ambiente

```python
import os
import time
from datetime import datetime

# Configuracoes
EXEC_NAME = "batch_1_exec_1"
DATASETS = ["SEA_Abrupt_Simple", "AGRAWAL_Abrupt_Simple_Severe"]
CONFIG_FILE = "config_batch_1_exec_1.yaml"

# Paths
DRIVE_BASE = "/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid"
WORK_DIR = "/content/DSL-AG-hybrid"
LOG_FILE = f"{DRIVE_BASE}/{EXEC_NAME}.log"

print("="*80)
print(f"  EXECUCAO: {EXEC_NAME}")
print("="*80)
print(f"Datasets: {', '.join(DATASETS)}")
print(f"Config: {CONFIG_FILE}")
print(f"Log: {LOG_FILE}")
print(f"Inicio: {datetime.now()}")
print("="*80)

# Verificar se diretorio do Drive existe
if not os.path.exists(DRIVE_BASE):
    print(f"ERRO: Diretorio {DRIVE_BASE} nao encontrado!")
    raise FileNotFoundError(DRIVE_BASE)

print(f"\n✓ Diretorio do Drive encontrado")
```

---

## CELULA 3: Copiar Repositorio para /content

```python
# Remover diretorio antigo se existir
if os.path.exists(WORK_DIR):
    print("Removendo diretorio antigo...")
    !rm -rf {WORK_DIR}

# Copiar do Drive
print(f"Copiando repositorio de {DRIVE_BASE} para {WORK_DIR}...")
!cp -r {DRIVE_BASE} {WORK_DIR}

# Mudar para diretorio de trabalho
os.chdir(WORK_DIR)
print(f"\n✓ Diretorio de trabalho: {os.getcwd()}")

# Listar arquivos principais
print("\nArquivos principais:")
!ls -lh main.py config.yaml requirements.txt 2>/dev/null || echo "Alguns arquivos podem estar faltando"
```

---

## CELULA 4: Instalar Dependencias

```python
# Verificar e instalar dependencias
if os.path.exists("requirements.txt"):
    print("Instalando dependencias...")
    !pip install -q -r requirements.txt
    print("✓ Dependencias instaladas")
else:
    print("! AVISO: requirements.txt nao encontrado")
    print("Instalando pacotes basicos...")
    !pip install -q river numpy pandas scikit-learn pyyaml
```

---

## CELULA 5: Configurar Execucao

```python
# Copiar config correto
config_source = f"configs/{CONFIG_FILE}"
config_target = "config.yaml"

if not os.path.exists(config_source):
    print(f"ERRO: Config {config_source} nao encontrado!")
    raise FileNotFoundError(config_source)

!cp {config_source} {config_target}
print(f"✓ Config copiado: {CONFIG_FILE} → config.yaml")

# Verificar datasets configurados
print("\nDatasets configurados:")
!grep -A 3 "drift_simulation_experiments" config.yaml

# Verificar diretorio de resultados
print("\nDiretorio de resultados:")
!grep "base_results_dir" config.yaml
```

---

## CELULA 6: Executar Experimento (ESTA CELULA DEMORA ~20 HORAS!)

```python
# Timestamp de inicio
exec_start = datetime.now()
print("="*80)
print(f"INICIANDO EXECUCAO: {exec_start}")
print("="*80)
print(f"\nEste processo levara aproximadamente 18-22 horas...")
print(f"Log sera salvo em: {LOG_FILE}")
print("\nVoce pode:")
print("1. Fechar esta aba (a execucao continuara)")
print("2. Usar a CELULA 7 para monitorar o progresso")
print("3. Aguardar a conclusao\n")

# Executar main.py
!python main.py > {LOG_FILE} 2>&1

# Timestamp de fim
exec_end = datetime.now()
exec_duration = (exec_end - exec_start).total_seconds() / 3600

print("\n" + "="*80)
print(f"EXECUCAO CONCLUIDA!")
print("="*80)
print(f"Inicio: {exec_start}")
print(f"Fim: {exec_end}")
print(f"Duracao: {exec_duration:.2f} horas")
```

**IMPORTANTE**: Esta celula pode demorar 18-22 horas! Nao feche o navegador.
Se precisar monitorar, use a Celula 7 em outra aba.

---

## CELULA 7: Monitorar Progresso (EXECUTAR EM OUTRA ABA)

Execute esta celula periodicamente para ver o progresso:

```python
import os
from datetime import datetime

LOG_FILE = "/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/batch_1_exec_1.log"

print(f"Status em: {datetime.now()}\n")

# Verificar se log existe
if not os.path.exists(LOG_FILE):
    print("Log ainda nao criado. Aguarde...")
else:
    # Tamanho do log
    size_mb = os.path.getsize(LOG_FILE) / 1024 / 1024
    print(f"Tamanho do log: {size_mb:.2f} MB\n")

    print("="*80)
    print("ULTIMAS 30 LINHAS DO LOG:")
    print("="*80)
    !tail -n 30 {LOG_FILE}

    print("\n" + "="*80)
    print("CHUNKS FINALIZADOS:")
    print("="*80)
    !grep -i "chunk.*final" {LOG_FILE} | tail -5

    print("\n" + "="*80)
    print("PROGRESSO GERAL:")
    print("="*80)
    !grep -E "(Dataset|CHUNK.*INICIO|G-mean)" {LOG_FILE} | tail -10
```

---

## CELULA 8: Validar Resultados (APOS CONCLUSAO)

```python
import os
import json

RESULTS_DIR = "/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/experiments_6chunks_phase1_gbml/batch_1"
DATASETS = ["SEA_Abrupt_Simple", "AGRAWAL_Abrupt_Simple_Severe"]

print("="*80)
print("VALIDACAO DE RESULTADOS")
print("="*80)

for dataset in DATASETS:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print('='*60)

    dataset_dir = f"{RESULTS_DIR}/{dataset}/run_1"

    if os.path.exists(dataset_dir):
        print("✓ Diretorio criado")

        # Verificar chunks
        chunk_dir = f"{dataset_dir}/chunk_data"
        if os.path.exists(chunk_dir):
            chunk_count = len([f for f in os.listdir(chunk_dir) if f.endswith('_test.csv')])
            print(f"  - Chunks de teste: {chunk_count}/6")

        # Verificar plots
        plots_dir = f"{dataset_dir}/plots"
        if os.path.exists(plots_dir):
            plot_count = len([f for f in os.listdir(plots_dir) if f.endswith('.png')])
            print(f"  - Plots gerados: {plot_count}")

        # Verificar summary
        summary_file = f"{dataset_dir}/experiment_summary.json"
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            print(f"  - Test G-mean: {summary.get('test_gmean', 'N/A'):.4f}")
            print(f"  - Train G-mean: {summary.get('train_gmean', 'N/A'):.4f}")
        else:
            print("  ! Summary nao encontrado")

    else:
        print("✗ Diretorio NAO criado")

print("\n" + "="*80)
print("VALIDACAO CONCLUIDA")
print("="*80)
```

---

## CELULA 9: Verificar Espaco e Fazer Backup (OPCIONAL)

```python
# Verificar espaco usado
RESULTS_DIR = "/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/experiments_6chunks_phase1_gbml/batch_1"

print("Espaco usado pelos resultados:")
!du -sh {RESULTS_DIR}

# Listar estrutura
print("\nEstrutura de resultados:")
!find {RESULTS_DIR} -type d | head -20

# Contar arquivos
print("\nTotal de arquivos criados:")
!find {RESULTS_DIR} -type f | wc -l
```

---

## RESUMO DE USO

### Ordem de Execucao:

1. **CELULA 1**: Montar Drive (1 minuto)
2. **CELULA 2**: Configurar ambiente (30 segundos)
3. **CELULA 3**: Copiar repositorio (1-2 minutos)
4. **CELULA 4**: Instalar dependencias (2-3 minutos)
5. **CELULA 5**: Configurar execucao (30 segundos)
6. **CELULA 6**: Executar experimento (18-22 HORAS!)
7. **CELULA 7**: Monitorar (executar quando quiser)
8. **CELULA 8**: Validar resultados (apos CELULA 6)
9. **CELULA 9**: Verificar backup (opcional)

### Dicas:

- Execute CELULAS 1-5 em sequencia
- Execute CELULA 6 e deixe rodando
- Abra outra aba do Colab e execute CELULA 7 para monitorar
- Nao feche o navegador enquanto CELULA 6 executa
- Apos conclusao, execute CELULAS 8 e 9

### Em caso de erro:

- Verifique o log: `batch_1_exec_1.log`
- Execute CELULA 7 para ver ultimas linhas
- Se necessario, reinicie do CELULA 3

---

**Criado em**: 2025-11-17
**Para**: Batch 1 - Execucao 1 (SEA + AGRAWAL)
**Tempo estimado**: 18-22 horas
