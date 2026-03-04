# 🚀 Solução Completa: Google Colab SSH + Google Drive

## 📋 Sumário Executivo

Esta solução permite executar experimentos no Google Colab **100% via terminal SSH**, sem necessidade de abrir navegador após configuração inicial, com **salvamento automático de todos os logs e resultados no Google Drive**.

**Problema resolvido:**
- ❌ Antes: Sessão SSH cai → logs perdidos
- ✅ Agora: Logs salvos em tempo real no Drive → dados sempre disponíveis

---

## 🎯 Arquivos Criados

Esta solução consiste em **7 arquivos principais**:

### 1. **`colab_workflow.ps1`** (Windows - Script Principal)
- **Função:** Workflow completo automatizado via PowerShell
- **Uso:** `.\colab_workflow.ps1 -SSHHost <host> -Action all -Stream RBF -Chunks 3`
- **Recursos:**
  - ✅ Upload automático de arquivos via SCP
  - ✅ Setup do ambiente remoto
  - ✅ Execução de experimentos
  - ✅ Download de resultados
  - ✅ Tudo em um único comando

### 2. **`sync_to_colab.sh`** (Linux/Mac - Upload)
- **Função:** Script bash para upload de arquivos via SCP
- **Uso:** `./sync_to_colab.sh <host>.trycloudflare.com`
- **Recursos:**
  - ✅ Upload batch de todos os arquivos .py e .yaml
  - ✅ Validação de arquivos
  - ✅ Resumo de sincronização

### 3. **`setup_colab_remote.py`** (Setup Remoto)
- **Função:** Configura ambiente Colab quando acessado via SSH
- **Uso:** Executado remotamente via SSH
- **Recursos:**
  - ✅ Verifica se Drive está montado
  - ✅ Cria estrutura de diretórios no Drive
  - ✅ Gera variáveis de ambiente
  - ✅ Adiciona aliases ao .bashrc
  - ✅ Cria script wrapper `/root/run_experiment.sh`

### 4. **`test_ssh_workflow.ps1`** (Validação)
- **Função:** Valida que o ambiente está configurado corretamente
- **Uso:** `.\test_ssh_workflow.ps1 -SSHHost <host>`
- **Testa:**
  - ✅ Arquivos locais existem
  - ✅ Conectividade SSH
  - ✅ Google Colab environment
  - ✅ Google Drive montado e acessível
  - ✅ Upload via SCP funcional
  - ✅ Dependências Python instaladas

### 5. **`README_WORKFLOW_SSH.md`** (Documentação Completa)
- **Função:** Tutorial completo e detalhado
- **Conteúdo:**
  - 🚀 Métodos de execução (Windows, Linux, Mac)
  - 📂 Estrutura de diretórios
  - 🔧 Aliases e funções criadas
  - ⚠️ Troubleshooting
  - 💡 Dicas avançadas
  - ✅ Checklist de execução

### 6. **`QUICK_REFERENCE_SSH.txt`** (Referência Rápida)
- **Função:** Guia de consulta rápida em texto simples
- **Formato:** Texto puro, fácil copiar/colar
- **Conteúdo:**
  - Comandos essenciais
  - Parâmetros do experimento
  - Troubleshooting rápido
  - Exemplo completo passo a passo

### 7. **`SOLUCAO_COMPLETA_SSH_DRIVE.md`** (Este Arquivo)
- **Função:** Visão geral da solução completa
- **Propósito:** Documentação executiva e técnica

---

## 🔄 Fluxo de Trabalho

### Configuração Inicial (Uma vez)

```
┌─────────────────────────────────────────────────────────────┐
│ PASSO 1: Montar Google Drive (UMA VEZ)                      │
│                                                              │
│ No Notebook Colab (navegador):                              │
│   from google.colab import drive                            │
│   drive.mount('/content/drive')                             │
│                                                              │
│ Autoriza → Fecha navegador (Drive fica montado)             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PASSO 2: Upload de Arquivos                                 │
│                                                              │
│ Windows:                                                     │
│   .\colab_workflow.ps1 -SSHHost <host> -Action sync         │
│                                                              │
│ Linux/Mac:                                                   │
│   ./sync_to_colab.sh <host>                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PASSO 3: Setup do Ambiente                                  │
│                                                              │
│ Windows:                                                     │
│   .\colab_workflow.ps1 -SSHHost <host> -Action setup        │
│                                                              │
│ Linux/Mac:                                                   │
│   ssh <host>                                                │
│   cd /root/DSL-AG-hybrid                                    │
│   python setup_colab_remote.py                              │
└─────────────────────────────────────────────────────────────┘
```

### Execução de Experimentos (Sempre)

```
┌─────────────────────────────────────────────────────────────┐
│ OPÇÃO 1: Workflow Automático (Windows)                      │
│                                                              │
│ .\colab_workflow.ps1 \                                      │
│     -SSHHost <host> \                                       │
│     -Action run \                                           │
│     -Stream RBF_Abrupt_Severe \                             │
│     -Chunks 3                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ OPÇÃO 2: Via SSH Manual                                     │
│                                                              │
│ ssh <host>                                                  │
│ /root/run_experiment.sh python compare_gbml_vs_river.py \   │
│     --stream RBF_Abrupt_Severe \                            │
│     --chunks 3                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ OPÇÃO 3: Usando Função Bash                                 │
│                                                              │
│ ssh <host>                                                  │
│ source ~/.bashrc                                            │
│ run-experiment python compare_gbml_vs_river.py \            │
│     --stream RBF_Abrupt_Severe \                            │
│     --chunks 3                                              │
└─────────────────────────────────────────────────────────────┘
```

### Monitoramento em Tempo Real

```
┌─────────────────────────────────────────────────────────────┐
│ Em outro terminal:                                           │
│                                                              │
│ ssh <host>                                                  │
│ tail -f /content/drive/.../logs/*.log                       │
│                                                              │
│ Ou use alias:                                               │
│ drive-tail                                                  │
└─────────────────────────────────────────────────────────────┘
```

### Download de Resultados

```
┌─────────────────────────────────────────────────────────────┐
│ OPÇÃO 1: Via PowerShell                                     │
│                                                              │
│ .\colab_workflow.ps1 -SSHHost <host> -Action download       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ OPÇÃO 2: Via SCP Manual                                     │
│                                                              │
│ scp -r <host>:/content/drive/.../experiments/ssh_session_*  │
│        ./results/                                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ OPÇÃO 3: Google Drive Web                                   │
│                                                              │
│ drive.google.com → MyDrive/DSL-AG-hybrid/experiments/       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Estrutura de Diretórios Criada

### No Google Drive

```
/content/drive/MyDrive/DSL-AG-hybrid/
└── experiments/
    └── ssh_session_20251018_143022/      ← Timestamp único
        ├── logs/
        │   └── experiment_TIMESTAMP.log  ← Log completo em tempo real
        ├── results/
        │   ├── RBF_Abrupt_Severe/
        │   │   └── run_1/
        │   │       ├── chunk_data/
        │   │       ├── *.csv             ← Métricas
        │   │       ├── *.png             ← Gráficos
        │   │       └── *.pkl             ← Modelos salvos
        ├── checkpoints/
        │   └── *.pkl                     ← Checkpoints automáticos
        └── plots/
            └── *.png                     ← Visualizações
```

### No Colab (via SSH)

```
/root/DSL-AG-hybrid/
├── main.py                               ← Código sincronizado
├── ga.py
├── config.yaml
├── setup_colab_remote.py
├── .drive_paths.sh                       ← Gerado automaticamente
└── /root/run_experiment.sh               ← Wrapper gerado
```

---

## 🛠️ Recursos Criados Automaticamente

### Variáveis de Ambiente

Após `setup_colab_remote.py`:

```bash
$DRIVE_BASE              # /content/drive/MyDrive/DSL-AG-hybrid
$DRIVE_EXPERIMENTS       # /content/drive/.../experiments
$DRIVE_CURRENT_EXPERIMENT # /content/drive/.../ssh_session_TIMESTAMP
$DRIVE_LOGS              # /content/drive/.../logs
$DRIVE_RESULTS           # /content/drive/.../results
$DRIVE_CHECKPOINTS       # /content/drive/.../checkpoints
$DRIVE_PLOTS             # /content/drive/.../plots
```

### Aliases Bash

```bash
drive-logs               # Navega para diretório de logs
drive-results            # Navega para diretório de resultados
drive-tail               # Monitora logs em tempo real
```

### Funções Bash

```bash
run-experiment <comando>
# Executa comando com logging automático no Drive
# Exemplo:
#   run-experiment python compare_gbml_vs_river.py --stream RBF --chunks 3
```

### Script Wrapper

```bash
/root/run_experiment.sh <comando>
# Wrapper completo com header, logging, e resumo final
# Exemplo:
#   /root/run_experiment.sh python compare_gbml_vs_river.py --stream RBF --chunks 3
```

---

## ✅ Principais Vantagens

### 1. **100% Terminal**
- ✅ Sem necessidade de abrir navegador (após setup inicial)
- ✅ Workflow totalmente automatizado
- ✅ Scripts reutilizáveis

### 2. **Logs Persistentes**
- ✅ Salvamento em tempo real no Google Drive
- ✅ Dual logging (console + arquivo)
- ✅ Timestamps únicos por sessão

### 3. **Recuperação Automática**
- ✅ Mesmo se SSH cair, dados estão no Drive
- ✅ Checkpoints periódicos salvos
- ✅ Estado completo preservado

### 4. **Facilidade de Uso**
- ✅ Um comando para tudo (`-Action all`)
- ✅ Aliases e funções prontas
- ✅ Documentação completa e referência rápida

### 5. **Multiplataforma**
- ✅ Windows (PowerShell)
- ✅ Linux (Bash)
- ✅ Mac (Bash)
- ✅ Git Bash no Windows

### 6. **Validação Automática**
- ✅ Script de teste (`test_ssh_workflow.ps1`)
- ✅ Diagnóstico de problemas
- ✅ Feedback claro

---

## 🎯 Casos de Uso

### Caso 1: Pesquisador Executando Experimentos Longos

**Problema:** Experimento de 8 horas, sessão SSH pode cair.

**Solução:**
```powershell
# Inicia experimento
.\colab_workflow.ps1 -SSHHost <host> -Action run -Stream RBF -Chunks 5

# Sessão SSH cai durante execução (não importa!)
# Logs continuam sendo salvos no Drive

# Depois, download de resultados
.\colab_workflow.ps1 -SSHHost <host> -Action download

# Ou acessa via Drive Web
# drive.google.com → MyDrive/DSL-AG-hybrid/experiments/
```

### Caso 2: Batch de Experimentos

**Problema:** Precisa executar múltiplos streams sequencialmente.

**Solução:**
```powershell
$streams = @('RBF_Abrupt_Severe', 'SEA_Abrupt_Simple', 'AGRAWAL_Gradual_Simple')

foreach ($stream in $streams) {
    .\colab_workflow.ps1 `
        -SSHHost <host> `
        -Action run `
        -Stream $stream `
        -Chunks 3
}
```

### Caso 3: Desenvolvimento Iterativo

**Problema:** Modificando código localmente, testando no Colab.

**Solução:**
```powershell
# 1. Modifica código local (main.py, ga.py, etc.)

# 2. Sync + Run
.\colab_workflow.ps1 -SSHHost <host> -Action sync
.\colab_workflow.ps1 -SSHHost <host> -Action run -Stream RBF -Chunks 1

# 3. Analisa resultados
.\colab_workflow.ps1 -SSHHost <host> -Action download

# Repete ciclo
```

### Caso 4: Monitoramento Remoto

**Problema:** Experimento rodando, quer monitorar de outro lugar.

**Solução:**
```bash
# De qualquer máquina com SSH
ssh <host>.trycloudflare.com

# Monitora em tempo real
tail -f /content/drive/.../logs/*.log

# Ou
drive-tail

# Desconecta (Ctrl+C), experimento continua rodando
```

---

## ⚠️ Pontos de Atenção

### 1. **Montagem do Drive (Obrigatória)**

O Google Drive **DEVE** ser montado via notebook Colab (navegador) pelo menos uma vez:

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Por quê?** Autorização OAuth não funciona via SSH puro.

**Quando?** Apenas na primeira vez (ou quando montar novamente).

### 2. **Duração da Sessão SSH**

Cloudflare Tunnel pode ter timeout. Se a sessão cair:
- ✅ Logs estão no Drive
- ✅ Checkpoints salvos
- ✅ Pode retomar de onde parou

### 3. **Espaço no Drive**

Recomendado: **>5 GB livres** no Google Drive.

Verificar:
```bash
ssh <host> "df -h /content/drive"
```

### 4. **Permissões**

Se houver erro "permission denied":
```bash
ssh <host> "chmod -R 755 /content/drive/MyDrive/DSL-AG-hybrid"
```

---

## 📚 Documentação Completa

Para informações detalhadas, consulte:

- **`README_WORKFLOW_SSH.md`** - Tutorial completo passo a passo
- **`QUICK_REFERENCE_SSH.txt`** - Referência rápida (comandos essenciais)
- **`test_ssh_workflow.ps1`** - Validação do ambiente

---

## 🚀 Início Rápido

### Setup Inicial (UMA VEZ)

```powershell
# 1. Validar ambiente
.\test_ssh_workflow.ps1 -SSHHost logged-minerals-axis-infrastructure.trycloudflare.com

# 2. Upload + Setup
.\colab_workflow.ps1 -SSHHost logged-minerals-axis-infrastructure.trycloudflare.com -Action sync
.\colab_workflow.ps1 -SSHHost logged-minerals-axis-infrastructure.trycloudflare.com -Action setup
```

### Execução de Experimento

```powershell
# Workflow completo
.\colab_workflow.ps1 `
    -SSHHost logged-minerals-axis-infrastructure.trycloudflare.com `
    -Action run `
    -Stream RBF_Abrupt_Severe `
    -Chunks 3
```

### Download de Resultados

```powershell
.\colab_workflow.ps1 -SSHHost logged-minerals-axis-infrastructure.trycloudflare.com -Action download
```

---

## 🎉 Conclusão

Esta solução resolve completamente o problema de perda de logs em sessões SSH do Google Colab, permitindo:

✅ **Workflow 100% via terminal**
✅ **Logs persistentes no Drive**
✅ **Recuperação automática**
✅ **Monitoramento em tempo real**
✅ **Scripts reutilizáveis e documentados**

**Status:** ✅ Pronto para uso
**Versão:** 1.0
**Data:** 2025-10-18
**Autor:** Claude Code

---

**🚀 Pronto para começar!**

Execute o teste de validação e depois o workflow completo:

```powershell
.\test_ssh_workflow.ps1 -SSHHost logged-minerals-axis-infrastructure.trycloudflare.com

.\colab_workflow.ps1 `
    -SSHHost logged-minerals-axis-infrastructure.trycloudflare.com `
    -Action all `
    -Stream RBF_Abrupt_Severe `
    -Chunks 3
```
