"""
Script otimizado para executar Batch 1 - Execucao 1 no Google Colab.

Datasets: SEA_Abrupt_Simple + AGRAWAL_Abrupt_Simple_Severe
Tempo estimado: 18-22 horas
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from google.colab import drive

# ==============================================================================
# CONFIGURACOES
# ==============================================================================

# Configuracao da execucao
EXEC_NAME = "batch_1_exec_1"
CONFIG_FILE = "config_batch_1_exec_1.yaml"
DATASETS = ["SEA_Abrupt_Simple", "AGRAWAL_Abrupt_Simple_Severe"]

# Paths
DRIVE_BASE = "/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid"
WORK_DIR = "/content/DSL-AG-hybrid"
LOG_FILE = f"/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/{EXEC_NAME}.log"

# ==============================================================================
# FUNCOES AUXILIARES
# ==============================================================================

def print_header(message):
    """Imprime cabecalho formatado"""
    print("\n" + "="*80)
    print(f"  {message}")
    print("="*80 + "\n")

def print_step(step_num, message):
    """Imprime passo formatado"""
    print(f"\n[PASSO {step_num}] {message}")
    print("-" * 80)

def run_command(cmd, description=""):
    """Executa comando e retorna resultado"""
    if description:
        print(f"\n>>> {description}")
    print(f"$ {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0

def get_timestamp():
    """Retorna timestamp formatado"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================

print_header(f"EXECUCAO: {EXEC_NAME}")
print(f"Data/Hora inicio: {get_timestamp()}")
print(f"Datasets: {', '.join(DATASETS)}")
print(f"Config: {CONFIG_FILE}")
print(f"Log: {LOG_FILE}")

start_time = time.time()

# ------------------------------------------------------------------------------
# PASSO 1: Montar Google Drive
# ------------------------------------------------------------------------------

print_step(1, "Montando Google Drive")

try:
    drive.mount('/content/drive')
    print("✓ Google Drive montado com sucesso")
except Exception as e:
    print(f"✗ ERRO ao montar Google Drive: {e}")
    sys.exit(1)

# Verificar se diretorio existe
if not os.path.exists(DRIVE_BASE):
    print(f"✗ ERRO: Diretorio {DRIVE_BASE} nao encontrado")
    sys.exit(1)

print(f"✓ Diretorio encontrado: {DRIVE_BASE}")

# ------------------------------------------------------------------------------
# PASSO 2: Clonar/Copiar Repositorio para /content
# ------------------------------------------------------------------------------

print_step(2, "Preparando diretorio de trabalho")

# Remover diretorio antigo se existir
if os.path.exists(WORK_DIR):
    run_command(f"rm -rf {WORK_DIR}", "Removendo diretorio antigo")

# Copiar do Drive para /content (mais rapido)
if not run_command(f"cp -r {DRIVE_BASE} {WORK_DIR}", "Copiando repositorio do Drive"):
    print("✗ ERRO ao copiar repositorio")
    sys.exit(1)

print(f"✓ Repositorio copiado para {WORK_DIR}")

# Mudar para diretorio de trabalho
os.chdir(WORK_DIR)
print(f"✓ Diretorio de trabalho: {os.getcwd()}")

# ------------------------------------------------------------------------------
# PASSO 3: Instalar Dependencias
# ------------------------------------------------------------------------------

print_step(3, "Instalando dependencias")

# Verificar se requirements.txt existe
if os.path.exists("requirements.txt"):
    if not run_command("pip install -r requirements.txt", "Instalando pacotes Python"):
        print("✗ ERRO ao instalar dependencias")
        sys.exit(1)
    print("✓ Dependencias instaladas")
else:
    print("! AVISO: requirements.txt nao encontrado")

# ------------------------------------------------------------------------------
# PASSO 4: Copiar Config Correto
# ------------------------------------------------------------------------------

print_step(4, "Configurando execucao")

config_source = f"configs/{CONFIG_FILE}"
config_target = "config.yaml"

if not os.path.exists(config_source):
    print(f"✗ ERRO: Config {config_source} nao encontrado")
    sys.exit(1)

if not run_command(f"cp {config_source} {config_target}", f"Copiando {CONFIG_FILE}"):
    print("✗ ERRO ao copiar config")
    sys.exit(1)

print(f"✓ Config copiado: {CONFIG_FILE} → config.yaml")

# Verificar datasets no config
print("\nDatasets configurados:")
run_command("grep -A 5 'drift_simulation_experiments' config.yaml", "")

# ------------------------------------------------------------------------------
# PASSO 5: Executar Experimento
# ------------------------------------------------------------------------------

print_step(5, "Executando experimento")

print(f"Inicio da execucao: {get_timestamp()}")
print(f"Log sera salvo em: {LOG_FILE}")
print("\nEste processo pode levar 18-22 horas...")
print("Acompanhe o progresso no arquivo de log.\n")

# Executar main.py e redirecionar output para log
exec_cmd = f"python main.py > {LOG_FILE} 2>&1"

exec_start = time.time()

# Executar
if not run_command(exec_cmd, "Executando main.py"):
    print(f"✗ ERRO durante execucao")
    print(f"Verifique o log em: {LOG_FILE}")
    sys.exit(1)

exec_end = time.time()
exec_duration = exec_end - exec_start

print(f"\n✓ Execucao concluida!")
print(f"Tempo de execucao: {exec_duration/3600:.2f} horas")

# ------------------------------------------------------------------------------
# PASSO 6: Validar Resultados
# ------------------------------------------------------------------------------

print_step(6, "Validando resultados")

results_dir = f"{DRIVE_BASE}/experiments_6chunks_phase1_gbml/batch_1"

print(f"Verificando resultados em: {results_dir}")

for dataset in DATASETS:
    dataset_dir = f"{results_dir}/{dataset}"

    if os.path.exists(dataset_dir):
        print(f"  ✓ {dataset}: Diretorio criado")

        # Verificar arquivos importantes
        run_1_dir = f"{dataset_dir}/run_1"
        if os.path.exists(run_1_dir):
            # Contar chunks
            chunk_count = len([f for f in os.listdir(f"{run_1_dir}/chunk_data")
                             if f.endswith('_test.csv')])
            print(f"      - Chunks gerados: {chunk_count}")

            # Verificar plots
            plots_dir = f"{run_1_dir}/plots"
            if os.path.exists(plots_dir):
                plot_count = len([f for f in os.listdir(plots_dir) if f.endswith('.png')])
                print(f"      - Plots gerados: {plot_count}")
        else:
            print(f"  ✗ {dataset}: run_1 nao encontrado")
    else:
        print(f"  ✗ {dataset}: Diretorio NAO criado")

# ------------------------------------------------------------------------------
# PASSO 7: Fazer Backup dos Resultados
# ------------------------------------------------------------------------------

print_step(7, "Backup dos resultados")

backup_dir = f"{DRIVE_BASE}/backups/{EXEC_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if not run_command(f"mkdir -p {backup_dir}", "Criando diretorio de backup"):
    print("! AVISO: Erro ao criar diretorio de backup")
else:
    # Copiar log
    run_command(f"cp {LOG_FILE} {backup_dir}/", "Copiando log")

    # Copiar resultados
    for dataset in DATASETS:
        dataset_dir = f"{results_dir}/{dataset}"
        if os.path.exists(dataset_dir):
            run_command(f"cp -r {dataset_dir} {backup_dir}/", f"Copiando {dataset}")

    print(f"✓ Backup salvo em: {backup_dir}")

# ------------------------------------------------------------------------------
# FINALIZACAO
# ------------------------------------------------------------------------------

end_time = time.time()
total_duration = end_time - start_time

print_header("EXECUCAO CONCLUIDA COM SUCESSO!")

print(f"Data/Hora fim: {get_timestamp()}")
print(f"Tempo total: {total_duration/3600:.2f} horas")
print(f"Datasets processados: {', '.join(DATASETS)}")
print(f"\nResultados salvos em: {results_dir}")
print(f"Log completo em: {LOG_FILE}")
print(f"Backup em: {backup_dir}")

print("\n" + "="*80)
print("PROXIMOS PASSOS:")
print("="*80)
print("1. Validar qualidade dos resultados")
print("2. Verificar G-mean dos datasets")
print("3. Se OK, executar batch_1_exec_2 e batch_1_exec_3")
print("4. Consolidar resultados do Batch 1 completo")
print("="*80 + "\n")
