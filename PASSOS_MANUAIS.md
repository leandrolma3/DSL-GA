# 🚀 Passos Manuais: Executar Experimento via SSH

**Host SSH:** `logged-minerals-axis-infrastructure.trycloudflare.com`

---

## PASSO 1: Montar Google Drive (UMA VEZ - via Notebook Colab)

Abra o notebook Colab no navegador e execute:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Autorize quando solicitado. **Não feche o notebook!**

---

## PASSO 2: Upload de Arquivos via SCP

No PowerShell (Windows), execute:

```powershell
# Navegar para o diretório do projeto
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"

# Upload de arquivos principais (um por vez, fornecendo senha quando solicitado)
scp main.py logged-minerals-axis-infrastructure.trycloudflare.com:/root/
scp ga.py logged-minerals-axis-infrastructure.trycloudflare.com:/root/
scp ga_operators.py logged-minerals-axis-infrastructure.trycloudflare.com:/root/
scp config.yaml logged-minerals-axis-infrastructure.trycloudflare.com:/root/
scp compare_gbml_vs_river.py logged-minerals-axis-infrastructure.trycloudflare.com:/root/
scp hill_climbing_v2.py logged-minerals-axis-infrastructure.trycloudflare.com:/root/
scp intelligent_hc_strategies.py logged-minerals-axis-infrastructure.trycloudflare.com:/root/
scp dt_rule_extraction.py logged-minerals-axis-infrastructure.trycloudflare.com:/root/
scp early_stopping.py logged-minerals-axis-infrastructure.trycloudflare.com:/root/
scp utils.py logged-minerals-axis-infrastructure.trycloudflare.com:/root/
scp setup_colab_remote.py logged-minerals-axis-infrastructure.trycloudflare.com:/root/
```

**Nota:** Você precisará digitar a senha do root para cada arquivo (padrão: `root`).

**Alternativa:** Upload em lote (pode não funcionar se pedir senha):

```powershell
scp *.py config.yaml logged-minerals-axis-infrastructure.trycloudflare.com:/root/DSL-AG-hybrid/
```

---

## PASSO 3: Conectar via SSH e Preparar Ambiente

```powershell
# Conectar
ssh logged-minerals-axis-infrastructure.trycloudflare.com
```

Digite a senha quando solicitado.

Agora, **dentro do SSH**, execute:

```bash
# Criar diretório do projeto
mkdir -p /root/DSL-AG-hybrid
cd /root/DSL-AG-hybrid

# Mover arquivos para o diretório correto (se não fez upload direto para DSL-AG-hybrid)
mv /root/*.py /root/DSL-AG-hybrid/ 2>/dev/null
mv /root/config.yaml /root/DSL-AG-hybrid/ 2>/dev/null

# Verificar arquivos
ls -lh

# Executar setup do Drive
python3 setup_colab_remote.py
```

**Esperado:** O script deve:
- Verificar se o Drive está montado
- Criar estrutura de diretórios no Drive
- Configurar aliases e variáveis de ambiente

---

## PASSO 4: Executar Experimento

Ainda **dentro do SSH**, execute:

### Opção A: Usar o wrapper (recomendado)

```bash
/root/run_experiment.sh python3 compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --chunks 3
```

### Opção B: Executar diretamente com logging

```bash
python3 compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --chunks 3 \
    2>&1 | tee /content/drive/MyDrive/DSL-AG-hybrid/experiments/ssh_session_*/logs/experiment.log
```

### Opção C: Usando a função bash

```bash
# Carregar aliases
source ~/.bashrc

# Executar
run-experiment python3 compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --chunks 3
```

---

## PASSO 5: Monitorar Execução (Opcional)

Em **outro terminal PowerShell**, conecte novamente via SSH:

```powershell
ssh logged-minerals-axis-infrastructure.trycloudflare.com
```

Monitore o log em tempo real:

```bash
tail -f /content/drive/MyDrive/DSL-AG-hybrid/experiments/ssh_session_*/logs/*.log
```

Ou use o alias:

```bash
drive-tail
```

Para sair do monitoramento: `Ctrl+C` (o experimento continua rodando)

---

## PASSO 6: Acessar Resultados

### Opção 1: Via Google Drive Web

1. Abra: https://drive.google.com
2. Navegue para: `MyDrive/DSL-AG-hybrid/experiments/`
3. Encontre a pasta com timestamp: `ssh_session_YYYYMMDD_HHMMSS/`
4. Arquivos disponíveis:
   - `logs/*.log` - Logs completos
   - `results/*.csv` - Métricas
   - `results/*.png` - Gráficos
   - `checkpoints/*.pkl` - Checkpoints

### Opção 2: Download via SCP

```powershell
# Criar diretório local
mkdir results_download

# Download recursivo (fornecendo senha)
scp -r logged-minerals-axis-infrastructure.trycloudflare.com:/content/drive/MyDrive/DSL-AG-hybrid/experiments/ssh_session_* ./results_download/
```

### Opção 3: Via SSH (visualizar no terminal)

```bash
# Ver últimas linhas do log
tail -50 /content/drive/MyDrive/DSL-AG-hybrid/experiments/ssh_session_*/logs/*.log

# Listar resultados
ls -lh /content/drive/MyDrive/DSL-AG-hybrid/experiments/ssh_session_*/results/

# Ver resumo do experimento
cat /content/drive/MyDrive/DSL-AG-hybrid/experiments/ssh_session_*/EXPERIMENT_SUMMARY.txt
```

---

## 🔧 Troubleshooting

### Problema: "Drive não está montado"

**Solução:** No notebook Colab (navegador):

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

Depois execute `setup_colab_remote.py` novamente via SSH.

---

### Problema: "Arquivo não encontrado"

**Solução:** Verifique se o upload funcionou:

```bash
# Via SSH
ls -lh /root/DSL-AG-hybrid/
```

Se faltam arquivos, repita o upload via SCP.

---

### Problema: "Permission denied"

**Solução:** Dê permissões aos arquivos:

```bash
# Via SSH
chmod +x /root/run_experiment.sh
chmod -R 755 /root/DSL-AG-hybrid/
```

---

### Problema: "ModuleNotFoundError"

**Solução:** Instale dependências:

```bash
# Via SSH
pip install -q river scikit-learn pyyaml matplotlib seaborn pandas numpy joblib
```

---

## 📊 Exemplo Completo

```bash
# ===== DENTRO DO SSH =====

# 1. Navegar para o projeto
cd /root/DSL-AG-hybrid

# 2. Verificar arquivos
ls -lh

# 3. Setup (se ainda não fez)
python3 setup_colab_remote.py

# 4. Executar experimento simples (1 chunk para teste)
/root/run_experiment.sh python3 compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --chunks 1

# 5. Ver se funcionou
tail -50 /content/drive/MyDrive/DSL-AG-hybrid/experiments/ssh_session_*/logs/*.log
```

---

## ✅ Checklist

- [ ] Google Drive montado no Colab (via notebook)
- [ ] Arquivos .py e .yaml enviados via SCP
- [ ] Conectado via SSH
- [ ] Executado `setup_colab_remote.py`
- [ ] Experimento executado
- [ ] Logs salvos no Drive
- [ ] Resultados acessíveis

---

**Data:** 2025-10-18
**Autor:** Claude Code

**🎉 Boa sorte com seus experimentos!**
