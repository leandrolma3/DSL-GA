# 🚀 Workflow Completo: Google Colab via SSH + Google Drive

## 📋 Visão Geral

Este guia mostra como executar experimentos no Google Colab **100% via terminal**, sem abrir navegador, com salvamento automático no Google Drive.

**Arquivos criados:**
- ✅ `sync_to_colab.sh` - Upload de arquivos via SCP (Linux/Mac)
- ✅ `colab_workflow.ps1` - Workflow completo PowerShell (Windows)
- ✅ `setup_colab_remote.py` - Setup do Drive via SSH

---

## 🎯 Pré-requisitos

### 1. Colab Notebook com SSH Configurado

Você já tem o `colab_ssh_setup.ipynb` rodando, gerando comandos como:
```
ssh logged-minerals-axis-infrastructure.trycloudflare.com
```

### 2. Google Drive Montado

**IMPORTANTE**: O Drive precisa ser montado **UMA VEZ** via notebook Colab (navegador):

```python
# Execute esta célula NO NOTEBOOK Colab (apenas 1 vez)
from google.colab import drive
drive.mount('/content/drive')
```

**Por quê?** Via SSH puro não é possível fazer a autorização OAuth do Google Drive. Mas uma vez autorizado no notebook, o Drive fica montado e acessível via SSH.

### 3. Ferramentas Instaladas

- **Windows**: PowerShell 5.1+, OpenSSH Client
- **Linux/Mac**: Bash, SSH, SCP

---

## 🚀 Método 1: Workflow Automatizado (Windows - PowerShell)

### Uso Básico

```powershell
# Workflow completo (sync + setup + run)
.\colab_workflow.ps1 `
    -SSHHost logged-minerals-axis-infrastructure.trycloudflare.com `
    -Action all `
    -Stream RBF_Abrupt_Severe `
    -Chunks 3
```

### Etapas Separadas

```powershell
# 1. Apenas sync (upload de arquivos)
.\colab_workflow.ps1 -SSHHost <seu-host>.trycloudflare.com -Action sync

# 2. Apenas setup (configura Drive)
.\colab_workflow.ps1 -SSHHost <seu-host>.trycloudflare.com -Action setup

# 3. Apenas run (executa experimento)
.\colab_workflow.ps1 `
    -SSHHost <seu-host>.trycloudflare.com `
    -Action run `
    -Stream RBF_Abrupt_Severe `
    -Chunks 3 `
    -Population 120 `
    -MaxGenerations 200

# 4. Download de resultados
.\colab_workflow.ps1 -SSHHost <seu-host>.trycloudflare.com -Action download
```

### Parâmetros Disponíveis

| Parâmetro | Obrigatório | Padrão | Descrição |
|-----------|-------------|--------|-----------|
| `-SSHHost` | ✅ Sim | - | Host SSH do Cloudflare |
| `-Action` | Não | `all` | `sync`, `setup`, `run`, `download`, `all` |
| `-Stream` | Não | `RBF_Abrupt_Severe` | Nome do stream |
| `-Chunks` | Não | `3` | Número de chunks |
| `-Population` | Não | `120` | População do GA |
| `-MaxGenerations` | Não | `200` | Max gerações |

---

## 🐧 Método 2: Workflow Manual (Linux/Mac - Bash)

### Passo 1: Upload de Arquivos

```bash
# Torna script executável
chmod +x sync_to_colab.sh

# Executa sync
./sync_to_colab.sh logged-minerals-axis-infrastructure.trycloudflare.com
```

### Passo 2: Setup do Ambiente

```bash
# Conecta via SSH
ssh logged-minerals-axis-infrastructure.trycloudflare.com

# Dentro do SSH
cd /root/DSL-AG-hybrid
python setup_colab_remote.py
```

### Passo 3: Executar Experimento

**Opção A: Usar wrapper (RECOMENDADO)**

```bash
/root/run_experiment.sh python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --chunks 3 \
    --population 120 \
    --max-generations 200
```

**Opção B: Usar função bash**

```bash
# Carrega ambiente
source ~/.bashrc

# Executa
run-experiment python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --chunks 3
```

**Opção C: Executar diretamente**

```bash
python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --chunks 3 \
    2>&1 | tee /content/drive/MyDrive/DSL-AG-hybrid/experiments/ssh_session_*/logs/experiment.log
```

### Passo 4: Monitorar Execução

```bash
# Em outro terminal, conecte via SSH
ssh logged-minerals-axis-infrastructure.trycloudflare.com

# Monitore log em tempo real
tail -f /content/drive/MyDrive/DSL-AG-hybrid/experiments/ssh_session_*/logs/*.log

# Ou use alias
drive-tail
```

---

## 📁 Estrutura de Arquivos

### No Google Drive

Após setup, você terá:

```
/content/drive/MyDrive/DSL-AG-hybrid/
└── experiments/
    └── ssh_session_20251018_143022/    ← Timestamp único
        ├── logs/
        │   └── experiment_*.log         ← Logs completos
        ├── results/
        │   └── [CSVs, PNGs, PKLs]
        ├── checkpoints/
        │   └── [Checkpoints automáticos]
        └── plots/
            └── [Gráficos]
```

### No Colab (via SSH)

```
/root/DSL-AG-hybrid/
├── main.py
├── ga.py
├── config.yaml
├── setup_colab_remote.py
├── .drive_paths.sh              ← Gerado automaticamente
└── run_experiment.sh            ← Wrapper gerado
```

---

## 🔧 Aliases e Funções Criadas

Após executar `setup_colab_remote.py`, você terá:

### Variáveis de Ambiente

```bash
$DRIVE_LOGS        # /content/drive/.../logs
$DRIVE_RESULTS     # /content/drive/.../results
$DRIVE_CHECKPOINTS # /content/drive/.../checkpoints
```

### Aliases

```bash
drive-logs     # Navega para diretório de logs
drive-results  # Navega para diretório de resultados
drive-tail     # Monitora logs em tempo real
```

### Funções

```bash
# Executa comando com logging automático no Drive
run-experiment <comando>

# Exemplo:
run-experiment python compare_gbml_vs_river.py --stream RBF --chunks 3
```

---

## 📊 Fluxo de Trabalho Recomendado

### Setup Inicial (Uma vez)

1. **No Notebook Colab** (navegador - APENAS 1 VEZ):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Na sua máquina** (Windows):
   ```powershell
   .\colab_workflow.ps1 `
       -SSHHost <seu-host>.trycloudflare.com `
       -Action sync

   .\colab_workflow.ps1 `
       -SSHHost <seu-host>.trycloudflare.com `
       -Action setup
   ```

### Execução de Experimentos (Toda vez)

```powershell
# Workflow completo
.\colab_workflow.ps1 `
    -SSHHost <seu-host>.trycloudflare.com `
    -Action run `
    -Stream RBF_Abrupt_Severe `
    -Chunks 3

# Ou via SSH manual
ssh <seu-host>.trycloudflare.com
cd /root/DSL-AG-hybrid
/root/run_experiment.sh python compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 3
```

### Download de Resultados

```powershell
# Download automático
.\colab_workflow.ps1 `
    -SSHHost <seu-host>.trycloudflare.com `
    -Action download

# Ou download manual
scp -r <seu-host>.trycloudflare.com:/content/drive/MyDrive/DSL-AG-hybrid/experiments/ssh_session_* ./results/
```

---

## ⚠️ Solução de Problemas

### 1. "Drive não está montado"

**Problema**: Ao executar `setup_colab_remote.py`, aparece erro de Drive não montado.

**Solução**:
```python
# No NOTEBOOK Colab (navegador)
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### 2. "Permission denied" no SCP

**Problema**: Erro ao fazer upload via SCP.

**Solução**:
```bash
# Verifique se SSH está acessível
ssh <seu-host>.trycloudflare.com "echo OK"

# Se OK aparecer, o SSH funciona
```

### 3. "Comando não encontrado: run-experiment"

**Problema**: Função bash não disponível.

**Solução**:
```bash
# Carrega .bashrc
source ~/.bashrc

# Ou reconecte via SSH
exit
ssh <seu-host>.trycloudflare.com
```

### 4. "Sessão SSH caiu, perdi tudo?"

**Resposta**: NÃO! Todos os logs e resultados estão no Google Drive.

**Como acessar**:
```powershell
# Download via PowerShell
.\colab_workflow.ps1 -SSHHost <seu-host>.trycloudflare.com -Action download

# Ou acesse pelo navegador
# drive.google.com → MyDrive/DSL-AG-hybrid/experiments/
```

### 5. "Experimento está rodando, mas não vejo log"

**Solução**: Monitore em tempo real via SSH:

```bash
# Em outro terminal
ssh <seu-host>.trycloudflare.com

# Monitora log
tail -f /content/drive/MyDrive/DSL-AG-hybrid/experiments/ssh_session_*/logs/*.log
```

---

## 💡 Dicas Avançadas

### 1. Executar Múltiplos Experimentos em Batch

```bash
# Via SSH
for stream in RBF_Abrupt_Severe SEA_Abrupt_Simple AGRAWAL_Gradual_Simple; do
    /root/run_experiment.sh python compare_gbml_vs_river.py \
        --stream $stream \
        --chunks 3
done
```

### 2. Executar em Background com nohup

```bash
nohup /root/run_experiment.sh python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --chunks 5 > /tmp/nohup.out 2>&1 &

# Monitora
tail -f /tmp/nohup.out
```

### 3. Sincronizar Apenas Arquivos Modificados

```bash
# Use rsync em vez de SCP (se disponível)
rsync -avz --exclude '__pycache__' --exclude '.git' \
    ./*.py ./config.yaml \
    <seu-host>.trycloudflare.com:/root/DSL-AG-hybrid/
```

### 4. Download Seletivo

```bash
# Apenas logs
scp -r <seu-host>.trycloudflare.com:/content/drive/.../logs ./

# Apenas gráficos
scp -r <seu-host>.trycloudflare.com:/content/drive/.../plots ./

# Apenas CSVs
scp <seu-host>.trycloudflare.com:/content/drive/.../results/*.csv ./
```

---

## 🎯 Checklist de Execução

### Antes de Executar

- [ ] Notebook Colab está rodando?
- [ ] SSH tunnel está ativo? (testa: `ssh <host> echo OK`)
- [ ] Google Drive está montado no Colab? (via notebook, uma vez)
- [ ] Arquivos foram sincronizados? (`sync_to_colab.sh` ou `-Action sync`)
- [ ] Setup foi executado? (`setup_colab_remote.py` ou `-Action setup`)

### Durante Execução

- [ ] Logs estão sendo salvos no Drive? (`drive-tail`)
- [ ] Experimento está progredindo? (monitore log)
- [ ] Sem erros de permissão?

### Após Execução

- [ ] Logs completos no Drive?
- [ ] Resultados (CSVs, PNGs) salvos?
- [ ] Checkpoints criados?
- [ ] Download feito? (opcional)

---

## 📞 Resumo de Comandos

### Windows (PowerShell)

```powershell
# Workflow completo
.\colab_workflow.ps1 -SSHHost <host> -Action all -Stream RBF_Abrupt_Severe -Chunks 3

# Sync + Setup + Run separados
.\colab_workflow.ps1 -SSHHost <host> -Action sync
.\colab_workflow.ps1 -SSHHost <host> -Action setup
.\colab_workflow.ps1 -SSHHost <host> -Action run -Stream RBF -Chunks 3

# Download
.\colab_workflow.ps1 -SSHHost <host> -Action download
```

### Linux/Mac (Bash)

```bash
# Sync
./sync_to_colab.sh <host>

# Setup
ssh <host> "cd /root/DSL-AG-hybrid && python setup_colab_remote.py"

# Run
ssh <host> "/root/run_experiment.sh python compare_gbml_vs_river.py --stream RBF --chunks 3"

# Download
scp -r <host>:/content/drive/MyDrive/DSL-AG-hybrid/experiments/ssh_session_* ./results/
```

---

## ✅ Vantagens desta Solução

1. ✅ **100% via terminal** - sem abrir navegador
2. ✅ **Logs persistentes** - salvos no Drive em tempo real
3. ✅ **Recuperação automática** - mesmo se SSH cair
4. ✅ **Workflow automatizado** - um comando para tudo
5. ✅ **Monitoramento em tempo real** - via `tail -f`
6. ✅ **Organização** - estrutura de diretórios com timestamps
7. ✅ **Reusável** - scripts podem ser usados repetidamente

---

**Versão**: 1.0
**Data**: 2025-10-18
**Autor**: Claude Code
**Status**: ✅ Pronto para uso

---

## 🚀 Começando Agora

```powershell
# 1. Certifique-se de que Drive está montado (UMA VEZ no notebook Colab)
# Notebook Colab:
#   from google.colab import drive
#   drive.mount('/content/drive')

# 2. Execute workflow completo
.\colab_workflow.ps1 `
    -SSHHost logged-minerals-axis-infrastructure.trycloudflare.com `
    -Action all `
    -Stream RBF_Abrupt_Severe `
    -Chunks 3

# 3. Aguarde conclusão e veja resultados no Drive!
```

🎉 **Pronto! Agora você pode executar experimentos no Colab 100% via terminal!**
