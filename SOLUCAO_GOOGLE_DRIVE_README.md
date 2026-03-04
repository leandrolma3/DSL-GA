# 🎯 Solução Completa: Google Colab + Google Drive

## 📦 Arquivos Criados

Esta solução consiste em **4 arquivos** que resolvem completamente o problema de perder logs quando a sessão SSH do Colab termina:

### 1. **`colab_drive_setup.py`** (Principal)
   - **O que faz:** Configura automaticamente o ambiente do Colab
   - **Funções principais:**
     - ✅ Monta o Google Drive
     - ✅ Cria estrutura de diretórios organizada
     - ✅ Configura logging DUPLO (console + arquivo no Drive)
     - ✅ Faz backup automático do código-fonte
     - ✅ Salva metadados do experimento
     - ✅ Gerencia checkpoints

### 2. **`run_experiment_colab.py`** (Wrapper)
   - **O que faz:** Wrapper que orquestra tudo automaticamente
   - **Recursos:**
     - ✅ Parseia argumentos da linha de comando
     - ✅ Chama `colab_drive_setup.py` para configurar ambiente
     - ✅ Atualiza `config.yaml` com parâmetros do Colab
     - ✅ Executa o experimento principal (main.py)
     - ✅ Gera relatório resumido ao final

### 3. **`GUIA_GOOGLE_COLAB.md`** (Documentação Completa)
   - **O que contém:** Tutorial completo e detalhado
   - **Seções:**
     - 🚀 3 métodos de execução (automático, manual, SSH)
     - 📂 Estrutura de diretórios no Drive
     - 🔄 Monitoramento em tempo real
     - 💾 Salvamento e recuperação de checkpoints
     - ⚠️ Solução de problemas
     - 💡 Dicas avançadas

### 4. **`QUICK_START_COLAB.txt`** (Referência Rápida)
   - **O que contém:** Guia de consulta rápida
   - **Formato:** Texto simples, fácil de copiar/colar
   - **Conteúdo:**
     - Início rápido (3 passos)
     - Comandos úteis
     - Parâmetros do experimento
     - Checklist
     - Troubleshooting

---

## 🚀 Como Usar (Início Rápido)

### Opção 1: Método Automático (MAIS FÁCIL)

**Passo 1:** Crie um notebook no Google Colab

**Passo 2:** Execute estas 3 células:

```python
# Célula 1: Preparar ambiente
!pip install -q river scikit-learn pyyaml matplotlib seaborn pandas numpy joblib
!git clone https://github.com/seu-repo/DSL-AG-hybrid.git
%cd DSL-AG-hybrid

# Célula 2: Executar experimento
!python run_experiment_colab.py \
    --stream RBF_Abrupt_Severe \
    --chunks 3 \
    --population 120 \
    --max-generations 200 \
    --experiment-name "Teste_HC_Inteligente"

# Célula 3: Ver resultados
!tail -50 /content/drive/MyDrive/DSL-AG-hybrid/experiments/*/logs/*.log
```

**Pronto!** Todos os logs e resultados serão salvos automaticamente no Google Drive.

---

### Opção 2: Controle Manual (Python)

```python
# Setup
from colab_drive_setup import setup_colab_environment
import logging

drive_results_dir, log_file, paths = setup_colab_environment(
    project_name="DSL-AG-hybrid",
    experiment_name="Meu_Experimento",
    backup_code=True,
    log_level=logging.INFO
)

# Agora execute seu código normalmente
# Tudo será salvo automaticamente no Drive
import main  # ou qualquer outro script
```

---

## 📂 O Que Acontece Automaticamente

### 1. **Montagem do Google Drive**
   - Monta `/content/drive/MyDrive/` automaticamente
   - Solicita autorização apenas na primeira vez
   - Reutiliza montagem existente se já estiver montado

### 2. **Criação de Estrutura de Diretórios**
```
/content/drive/MyDrive/DSL-AG-hybrid/
├── experiments/
│   └── RBF_Test_20251018_143022/      ← Timestamp único
│       ├── logs/
│       │   └── experiment.log         ← Log em tempo real
│       ├── results/
│       │   └── [CSVs, PNGs, PKLs]
│       ├── checkpoints/
│       │   └── [Checkpoints automáticos]
│       ├── plots/
│       └── EXPERIMENT_SUMMARY.txt
└── code_backup/
    └── 20251018_143022/
        └── [Arquivos .py, .yaml]
```

### 3. **Logging Duplo**
   - **Console:** Você vê o progresso em tempo real no Colab
   - **Arquivo:** Tudo é salvo simultaneamente no Drive
   - **Formato:** Timestamps, níveis de log, módulo, mensagem
   - **Persistência:** Mesmo se a sessão cair, o log está salvo

### 4. **Backup Automático de Código**
   - Copia todos os arquivos `.py`, `.yaml`, `.yml` para o Drive
   - Timestamp único garante versões separadas
   - Ignora automaticamente `venv`, `__pycache__`, `.git`

### 5. **Metadados do Experimento**
   - Arquivo JSON com informações completas
   - Data/hora, parâmetros, paths, versão Python
   - Facilita rastreamento e reprodutibilidade

---

## 🎯 Principais Vantagens

### ✅ **Problema Resolvido: Perda de Logs**
**ANTES:**
- ❌ Sessão SSH cai → logs perdidos
- ❌ Precisa reprocessar tudo
- ❌ Sem histórico do que aconteceu

**DEPOIS:**
- ✅ Tudo salvo em tempo real no Drive
- ✅ Logs completos sempre disponíveis
- ✅ Pode retomar de checkpoints

### ✅ **Zero Configuração Manual**
- Não precisa editar código
- Não precisa criar pastas manualmente
- Não precisa configurar logging
- Tudo funciona "out of the box"

### ✅ **Monitoramento em Tempo Real**
- Via Colab: vê no notebook
- Via SSH: `tail -f <arquivo.log>`
- Via Drive Web: abre no navegador
- Múltiplas opções para acompanhar

### ✅ **Recuperação Automática**
- Checkpoints salvos periodicamente
- Pode retomar de qualquer ponto
- Estado completo preservado

---

## 🔧 Parâmetros Configuráveis

### Via Linha de Comando:
```bash
python run_experiment_colab.py \
    --stream RBF_Abrupt_Severe       # Stream a executar
    --chunks 5                        # Número de chunks
    --chunk-size 6000                 # Tamanho do chunk
    --population 120                  # População do GA
    --max-generations 200             # Max gerações
    --experiment-name "Nome"          # Nome do experimento
    --project-name "Projeto"          # Nome do projeto
    --log-level INFO                  # Nível de log
    --no-backup-code                  # Desabilita backup
```

### Via Código Python:
```python
setup_colab_environment(
    project_name="DSL-AG-hybrid",
    experiment_name="Meu_Experimento",
    config_file="config.yaml",
    backup_code=True,
    log_level=logging.INFO
)
```

---

## 💾 Salvamento de Checkpoints

### Automático (já implementado):
```python
# No run_experiment_colab.py e main.py
# Checkpoints são salvos automaticamente:
# - Após cada chunk processado
# - A cada N gerações (configurável)
# - Quando estagnação é detectada
```

### Manual (se precisar):
```python
from colab_drive_setup import save_checkpoint

checkpoint_data = {
    'chunk_idx': 3,
    'best_individual': elite,
    'population': population,
    'fitness_history': history
}

save_checkpoint(
    data=checkpoint_data,
    checkpoint_name="manual_checkpoint",
    checkpoints_dir=paths['checkpoints']
)
```

### Recuperação:
```python
import pickle

# Carrega último checkpoint
with open('/content/drive/.../checkpoints/chunk_3.pkl', 'rb') as f:
    state = pickle.load(f)

# Retoma execução
elite = state['best_individual']
population = state['population']
# ... continua de onde parou
```

---

## ⚠️ Solução de Problemas Comuns

### 1. **Sessão Caiu, Perdi Tudo?**
**NÃO!** Todos os dados estão no Drive:
```
/content/drive/MyDrive/DSL-AG-hybrid/experiments/
```
Navegue até a pasta com o timestamp do seu experimento.

### 2. **Drive Não Está Montado**
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### 3. **Permission Denied**
```bash
!chmod -R 755 /content/drive/MyDrive/DSL-AG-hybrid
```

### 4. **Log Não Atualiza**
```python
import logging
for handler in logging.root.handlers:
    handler.flush()
```

### 5. **Pouco Espaço no Drive**
- Verifique: `!df -h /content/drive`
- Recomendado: >5 GB livres
- Solução: Limpe arquivos antigos ou upgrade do Drive

---

## 📊 Exemplo de Output

### Console (Colab):
```
======================================================================
🚀 SETUP AUTOMÁTICO PARA GOOGLE COLAB + GOOGLE DRIVE
======================================================================
🔄 Montando Google Drive...
✅ Google Drive montado com sucesso em: /content/drive

📁 Criando estrutura de diretórios no Drive...
   ✅ base           : /content/drive/MyDrive/DSL-AG-hybrid
   ✅ experiment     : /content/.../RBF_Test_20251018_143022
   ✅ logs           : /content/.../logs
   ✅ results        : /content/.../results
   ✅ checkpoints    : /content/.../checkpoints
   ✅ plots          : /content/.../plots
   ✅ code_backup    : /content/.../code_backup/20251018_143022

✅ SETUP CONCLUÍDO COM SUCESSO!
======================================================================
📂 Resultados:    /content/drive/.../results
📝 Logs:          /content/drive/.../logs/RBF_Test.log
💾 Checkpoints:   /content/drive/.../checkpoints
📊 Plots:         /content/drive/.../plots
======================================================================

2025-10-18 14:30:22 [INFO    ] main: Iniciando experimento: RBF_Test
2025-10-18 14:30:23 [INFO    ] main: Chunk 0: Gerando dados...
2025-10-18 14:30:25 [INFO    ] ga: Starting GA run: Pop=120, MaxGen=200
...
```

### Arquivo de Log (no Drive):
```
2025-10-18 14:30:22 [INFO    ] main: ======================================================================
2025-10-18 14:30:22 [INFO    ] main: Iniciando experimento: RBF_Test
2025-10-18 14:30:22 [INFO    ] main: Diretório de resultados: /content/drive/.../results
2025-10-18 14:30:22 [INFO    ] main: Arquivo de log: /content/drive/.../logs/RBF_Test.log
2025-10-18 14:30:22 [INFO    ] main: ======================================================================
2025-10-18 14:30:23 [INFO    ] main: Chunk 0: Gerando dados...
...
[TODO O HISTÓRICO PRESERVADO MESMO SE SESSÃO CAIR]
```

---

## 📝 Checklist de Uso

### Antes de Executar:
- [ ] Google Drive montado?
- [ ] Dependências instaladas?
- [ ] Código atualizado (git pull)?
- [ ] config.yaml correto?
- [ ] Espaço suficiente no Drive (>5 GB)?

### Durante Execução:
- [ ] Logs sendo salvos no Drive?
- [ ] Resultados aparecendo em `results/`?
- [ ] Checkpoints sendo criados?

### Após Execução:
- [ ] Todos os arquivos no Drive?
- [ ] Logs completos?
- [ ] Gráficos gerados?
- [ ] EXPERIMENT_SUMMARY.txt criado?

---

## 🎯 Próximos Passos

1. **Upload dos Arquivos:**
   - Faça upload de `colab_drive_setup.py`
   - Faça upload de `run_experiment_colab.py`
   - (Os guias são apenas documentação)

2. **Teste Rápido:**
   ```python
   !python run_experiment_colab.py \
       --stream RBF_Abrupt_Severe \
       --chunks 1 \
       --experiment-name "Teste_Setup"
   ```

3. **Experimento Real:**
   ```python
   !python run_experiment_colab.py \
       --stream RBF_Abrupt_Severe \
       --chunks 3 \
       --population 120 \
       --max-generations 200
   ```

4. **Verifique no Drive:**
   - Navegue até `MyDrive/DSL-AG-hybrid/experiments/`
   - Confira se logs e resultados estão sendo salvos

---

## 📞 Suporte

### Documentação:
- **Tutorial Completo:** `GUIA_GOOGLE_COLAB.md`
- **Referência Rápida:** `QUICK_START_COLAB.txt`
- **Código Fonte:** `colab_drive_setup.py`, `run_experiment_colab.py`

### Solução de Problemas:
1. Verifique logs no Drive primeiro
2. Consulte seção "Troubleshooting" do guia
3. Execute teste simples (1 chunk) para validar setup

---

**Versão:** 1.0
**Data:** 2025-10-18
**Autor:** Claude Code
**Status:** ✅ Pronto para uso

**🎉 Solução Completa Implementada!**

Agora você pode executar experimentos no Colab sem medo de perder logs,
mesmo que a sessão SSH caia. Tudo é salvo automaticamente no Google Drive! 🚀
