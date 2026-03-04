# 📘 Guia Completo: Google Colab + Google Drive

## 🎯 Objetivo

Este guia mostra como executar experimentos do **DSL-AG-hybrid** no Google Colab com **salvamento automático no Google Drive**, garantindo que nada seja perdido se a sessão SSH cair.

---

## 🚀 Método 1: Usando o Wrapper Automático (RECOMENDADO)

### Passo 1: Preparar Notebook no Colab

Crie um novo notebook no Google Colab e execute as células abaixo:

#### Célula 1: Instalar Dependências
```python
# Instala dependências necessárias
!pip install -q river scikit-learn pyyaml matplotlib seaborn pandas numpy joblib
```

#### Célula 2: Clonar Repositório
```python
# Clona o repositório (substitua pela URL correta)
!git clone https://github.com/seu-usuario/DSL-AG-hybrid.git
%cd DSL-AG-hybrid

# Ou, se o código já está no Drive:
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/DSL-AG-hybrid
```

#### Célula 3: Executar Experimento
```python
# Executa o experimento com salvamento automático no Drive
!python run_experiment_colab.py \
    --stream RBF_Abrupt_Severe \
    --chunks 3 \
    --chunk-size 6000 \
    --population 120 \
    --max-generations 200 \
    --experiment-name "Teste_HC_Inteligente"
```

**Pronto!** O sistema vai:
1. ✅ Montar o Google Drive automaticamente
2. ✅ Configurar logging duplo (console + arquivo no Drive)
3. ✅ Executar o experimento
4. ✅ Salvar TUDO no Drive em tempo real

---

## 🛠️ Método 2: Controle Manual (Avançado)

Se você quiser mais controle, use o script Python diretamente:

### Script Python no Colab

```python
# === CÉLULA 1: Setup ===
from colab_drive_setup import setup_colab_environment
import logging

# Configura ambiente (monta Drive, cria pastas, configura logging)
drive_results_dir, log_file, paths = setup_colab_environment(
    project_name="DSL-AG-hybrid",
    experiment_name="RBF_Test_Manual",
    backup_code=True,
    log_level=logging.INFO
)

print(f"Resultados serão salvos em: {drive_results_dir}")
print(f"Logs em tempo real: {log_file}")
```

```python
# === CÉLULA 2: Executar Experimento ===
import sys
import yaml

# Atualiza config.yaml para usar diretório do Drive
config_file = "config.yaml"

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Redireciona resultados para o Drive
config['experiment_settings']['base_results_dir'] = drive_results_dir
config['experiment_settings']['standard_experiments'] = ['RBF_Abrupt_Severe']
config['data_params']['num_chunks'] = 3

with open(config_file, 'w') as f:
    yaml.dump(config, f)

# Executa main.py
import main
# main.py vai rodar e salvar tudo no Drive automaticamente
```

```python
# === CÉLULA 3: Verificar Resultados ===
# Lista arquivos salvos no Drive
import os
for root, dirs, files in os.walk(paths['experiment']):
    level = root.replace(paths['experiment'], '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    sub_indent = ' ' * 2 * (level + 1)
    for file in files[:10]:  # Mostra primeiros 10 arquivos
        print(f'{sub_indent}{file}')
```

---

## 📂 Estrutura de Diretórios no Drive

Após a execução, você terá:

```
/content/drive/MyDrive/
└── DSL-AG-hybrid/
    ├── experiments/
    │   └── RBF_Test_20251018_143022/      ← Experimento com timestamp
    │       ├── logs/
    │       │   └── RBF_Test.log           ← Log completo em tempo real
    │       ├── results/
    │       │   ├── RBF_Abrupt_Severe/
    │       │   │   ├── run_1/
    │       │   │   │   ├── chunk_data/
    │       │   │   │   ├── *.csv
    │       │   │   │   ├── *.png
    │       │   │   │   └── *.pkl
    │       ├── checkpoints/
    │       │   └── *.pkl                  ← Checkpoints periódicos
    │       ├── plots/
    │       │   └── *.png
    │       └── EXPERIMENT_SUMMARY.txt     ← Resumo do experimento
    └── code_backup/
        └── 20251018_143022/               ← Backup do código usado
            ├── main.py
            ├── ga.py
            ├── config.yaml
            └── ...
```

---

## 🔄 Método 3: Conexão SSH + Salvamento no Drive

Se você estiver usando SSH (Cloudflare Tunnel), configure assim:

### Passo 1: No Servidor (via SSH)

```bash
# Conecta via SSH
ssh your-tunnel.trycloudflare.com

# Navega para o projeto
cd ~/DSL-AG-hybrid

# Verifica se o Drive está montado
ls /content/drive/MyDrive  # Se estiver no Colab
```

### Passo 2: Modifica Config para Salvar no Drive

```bash
# Edita config.yaml
nano config.yaml

# Altera base_results_dir para o Drive:
# base_results_dir: "/content/drive/MyDrive/DSL-AG-hybrid/experiments"
```

### Passo 3: Executa com Logging para Arquivo

```bash
# Executa redirecionando output para arquivo no Drive
python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --chunks 3 \
    --chunk-size 6000 \
    --seed 42 \
    2>&1 | tee /content/drive/MyDrive/DSL-AG-hybrid/experiments/experiment_$(date +%Y%m%d_%H%M%S).log
```

**Explicação do comando:**
- `2>&1`: Redireciona stderr para stdout
- `| tee <arquivo>`: Mostra no console E salva em arquivo
- `$(date +...)`: Adiciona timestamp ao nome do arquivo

---

## 📊 Monitoramento em Tempo Real

### Opção 1: No Colab (Interface Gráfica)

Se você executou via notebook, pode monitorar em tempo real:

```python
# Célula separada - Execute periodicamente
!tail -n 50 {log_file}  # Mostra últimas 50 linhas do log
```

### Opção 2: Via SSH (Terminal)

```bash
# Monitora log em tempo real
tail -f /content/drive/MyDrive/DSL-AG-hybrid/experiments/experiment.log

# Ou monitora saída do processo
tail -f nohup.out
```

### Opção 3: Google Drive Web (Manual)

1. Abra Google Drive no navegador
2. Navegue para `MyDrive/DSL-AG-hybrid/experiments/`
3. Abra o arquivo `.log` mais recente
4. Clique em "Abrir com" → "Google Docs" ou editor de texto
5. Recarregue a página periodicamente para ver atualizações

---

## 💾 Salvamento Automático de Checkpoints

O sistema salva checkpoints automaticamente durante a execução:

```python
# No seu código (já implementado em run_experiment_colab.py)
from colab_drive_setup import save_checkpoint

# Exemplo: Salvar estado após cada chunk
checkpoint_data = {
    'chunk_idx': chunk_idx,
    'best_individual': best_individual,
    'population': population,
    'fitness_history': fitness_history
}

save_checkpoint(
    data=checkpoint_data,
    checkpoint_name=f"chunk_{chunk_idx}",
    checkpoints_dir=paths['checkpoints']
)
```

**Recuperação de checkpoint:**
```python
import pickle

# Carrega checkpoint
checkpoint_file = "/content/drive/MyDrive/DSL-AG-hybrid/experiments/.../checkpoints/chunk_3_*.pkl"
with open(checkpoint_file, 'rb') as f:
    checkpoint = pickle.load(f)

# Continua execução de onde parou
best_individual = checkpoint['best_individual']
population = checkpoint['population']
```

---

## ⚠️ Solução de Problemas

### Problema 1: Sessão Colab/SSH Caiu

**Solução:** Todos os dados estão no Drive!

1. Abra Google Drive
2. Navegue para `MyDrive/DSL-AG-hybrid/experiments/`
3. Encontre a pasta do seu experimento (tem timestamp)
4. Todos os logs e resultados estão lá

### Problema 2: "Drive não montado"

**Solução:**
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Problema 3: "Permission denied" ao salvar

**Solução:**
```python
# Verifica permissões
import os
test_dir = "/content/drive/MyDrive/DSL-AG-hybrid"
print(f"Existe: {os.path.exists(test_dir)}")
print(f"Pode escrever: {os.access(test_dir, os.W_OK)}")

# Recria diretório com permissões corretas
os.makedirs(test_dir, exist_ok=True)
```

### Problema 4: Log não está sendo atualizado

**Solução:**
```python
# Força flush do log
import logging
for handler in logging.root.handlers:
    handler.flush()
```

### Problema 5: Experimento travou, como retomar?

**Solução:**
```python
# 1. Carrega último checkpoint
import pickle
checkpoint_files = sorted(glob.glob(f"{paths['checkpoints']}/chunk_*.pkl"))
latest = checkpoint_files[-1]

with open(latest, 'rb') as f:
    state = pickle.load(f)

# 2. Modifica config para pular chunks já processados
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Começa do próximo chunk
last_chunk = state['chunk_idx']
config['data_params']['start_chunk'] = last_chunk + 1

with open('config.yaml', 'w') as f:
    yaml.dump(config, f)

# 3. Retoma execução
!python run_experiment_colab.py --stream ... --chunks ...
```

---

## 🎯 Argumentos do run_experiment_colab.py

```bash
python run_experiment_colab.py --help

Argumentos:
  --stream STREAM              Nome do stream (ex: RBF_Abrupt_Severe)
  --chunks N                   Número de chunks (padrão: 3)
  --chunk-size N               Tamanho do chunk (padrão: 6000)
  --population N               População do GA (padrão: 120)
  --max-generations N          Max gerações (padrão: 200)
  --experiment-name NAME       Nome do experimento (padrão: stream name)
  --project-name NAME          Nome do projeto (padrão: DSL-AG-hybrid)
  --no-backup-code             Não fazer backup do código
  --log-level LEVEL            Nível de log (DEBUG, INFO, WARNING, ERROR)
  --config FILE                Arquivo de configuração (padrão: config.yaml)
```

---

## 📝 Checklist de Execução

Antes de executar o experimento:

- [ ] Google Drive está montado? (`/content/drive/MyDrive` existe?)
- [ ] Dependências instaladas? (`!pip install -r requirements.txt`)
- [ ] Código está atualizado? (git pull ou upload manual)
- [ ] `config.yaml` está correto?
- [ ] Diretório de resultados aponta para o Drive?
- [ ] Você tem espaço suficiente no Drive? (>5 GB recomendado)

Durante a execução:

- [ ] Logs estão sendo salvos? (verifique arquivo `.log` no Drive)
- [ ] Resultados aparecem na pasta `results/`?
- [ ] Checkpoints estão sendo criados?

Após a execução:

- [ ] Todos os arquivos estão no Drive?
- [ ] Logs completos foram salvos?
- [ ] Gráficos foram gerados?
- [ ] Arquivo de resumo existe?

---

## 🚀 Exemplo Completo de Execução

```python
# ========== CÉLULA 1: Setup Completo ==========
!pip install -q river scikit-learn pyyaml matplotlib seaborn pandas numpy joblib

# Clone ou copie o código
!git clone https://github.com/seu-repo/DSL-AG-hybrid.git
%cd DSL-AG-hybrid

# ========== CÉLULA 2: Executar Experimento ==========
!python run_experiment_colab.py \
    --stream RBF_Abrupt_Severe \
    --chunks 5 \
    --chunk-size 6000 \
    --population 120 \
    --max-generations 200 \
    --experiment-name "HC_Inteligente_Crossover_Balanceado_Final" \
    --log-level INFO

# ========== CÉLULA 3: Verificar Resultados ==========
# Lista arquivos criados
!ls -lh /content/drive/MyDrive/DSL-AG-hybrid/experiments/

# Mostra últimas linhas do log
!tail -50 /content/drive/MyDrive/DSL-AG-hybrid/experiments/*/logs/*.log

# ========== CÉLULA 4: Download de Resultados (Opcional) ==========
# Compacta resultados para download
!cd /content/drive/MyDrive/DSL-AG-hybrid/experiments && \
 zip -r resultados_$(date +%Y%m%d).zip */results */logs */EXPERIMENT_SUMMARY.txt

# Link de download aparecerá no Colab
from google.colab import files
files.download('/content/drive/MyDrive/DSL-AG-hybrid/experiments/resultados_*.zip')
```

---

## 💡 Dicas Avançadas

### 1. Executar Múltiplos Experimentos em Sequência

```python
# Cria script batch
streams = ['RBF_Abrupt_Severe', 'AGRAWAL_Abrupt_Simple', 'SEA_Gradual_Simple']

for stream in streams:
    !python run_experiment_colab.py \
        --stream {stream} \
        --chunks 3 \
        --experiment-name "Batch_{stream}"
```

### 2. Notificação por Email ao Concluir

```python
# No final do run_experiment_colab.py, adicione:
def send_completion_email(email, experiment_name):
    import smtplib
    from email.mime.text import MIMEText

    msg = MIMEText(f"Experimento {experiment_name} concluído!")
    msg['Subject'] = f'[Colab] Experimento Concluído'
    msg['From'] = 'seu_email@gmail.com'
    msg['To'] = email

    # Configure SMTP (ex: Gmail)
    # ... código de envio ...

# Uso:
send_completion_email('seu_email@gmail.com', args.experiment_name)
```

### 3. Monitoramento com Telegram Bot

```python
# Envie atualizações para Telegram durante execução
import requests

def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    requests.post(url, data=data)

# Uso: Notifica ao fim de cada chunk
send_telegram_message(
    token="YOUR_BOT_TOKEN",
    chat_id="YOUR_CHAT_ID",
    message=f"Chunk {chunk_idx} concluído! G-mean: {gmean:.2%}"
)
```

---

## 📞 Suporte

Se encontrar problemas:

1. Verifique os logs no Drive
2. Confira se o Drive está montado (`!ls /content/drive/MyDrive`)
3. Teste com um experimento pequeno primeiro (1 chunk)
4. Consulte a seção de Troubleshooting acima

---

**Versão:** 1.0
**Data:** 2025-10-18
**Autor:** Claude Code
**Compatibilidade:** Google Colab, Python 3.7+
