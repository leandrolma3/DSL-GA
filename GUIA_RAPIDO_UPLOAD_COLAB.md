# 📤 GUIA RÁPIDO: UPLOAD PARA GOOGLE COLAB

**Data**: 2025-10-28
**Objetivo**: Preparar Google Drive para teste de 6 chunks

---

## 📁 ARQUIVOS OBRIGATÓRIOS PARA UPLOAD

### Diretório no Google Drive: `/DSL-AG-hybrid/`

**Arquivos Python principais** (9 arquivos):
```
✅ main.py
✅ ga.py
✅ data_handling_v8.py
✅ hill_climbing_v2.py
✅ plotting.py
✅ analyze_concept_difference.py
✅ chunk_transition_analyzer.py
✅ rule_diff_analyzer.py
✅ analyze_standard_drift.py
```

**Arquivo de configuração**:
```
✅ config_test_single.yaml  ← NOVO (criado nesta sessão)
```

**Opcional** (mas recomendado):
```
compare_gbml_vs_river.py
```

**Diretório de dados** (se existir):
```
drift_analysis/
  └── concept_differences.json
```

---

## 🚀 MÉTODO DE UPLOAD

### Opção 1: Upload Manual (Recomendado para primeira vez)

1. **Acesse Google Drive**: https://drive.google.com
2. **Crie pasta**: `DSL-AG-hybrid` (se não existir)
3. **Upload arquivos**:
   - Arraste todos os 10 arquivos `.py` + `config_test_single.yaml`
   - Aguarde upload completar (~5-10 MB total)

### Opção 2: Google Drive Desktop (Mais Rápido)

1. **Sincronizar pasta local** com Google Drive
2. **Copiar arquivos** para a pasta sincronizada
3. **Aguardar sincronização** automática

### Opção 3: Comando (se tiver rclone configurado)

```bash
rclone copy . gdrive:DSL-AG-hybrid --include "*.py" --include "config_test_single.yaml"
```

---

## ✅ CHECKLIST DE VALIDAÇÃO

Após upload, validar no Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/DSL-AG-hybrid')

# Verificar arquivos
required = [
    'main.py', 'ga.py', 'data_handling_v8.py', 'config_test_single.yaml',
    'hill_climbing_v2.py', 'plotting.py', 'analyze_concept_difference.py'
]

for f in required:
    status = "✅" if os.path.exists(f) else "❌ FALTANDO"
    print(f"{status} {f}")
```

**Resultado esperado**: 7 ✅ (todos os arquivos encontrados)

---

## 📋 LISTA COMPLETA DE ARQUIVOS PARA COPIAR

### Da pasta local: `C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid\`

Para o Google Drive: `/DSL-AG-hybrid/`

| Arquivo | Tamanho Aprox. | Obrigatório |
|---------|----------------|-------------|
| `main.py` | ~50 KB | ✅ Sim |
| `ga.py` | ~80 KB | ✅ Sim |
| `data_handling_v8.py` | ~30 KB | ✅ Sim |
| `hill_climbing_v2.py` | ~40 KB | ✅ Sim |
| `plotting.py` | ~25 KB | ✅ Sim |
| `analyze_concept_difference.py` | ~20 KB | ✅ Sim |
| `chunk_transition_analyzer.py` | ~25 KB | ⚠️ Recomendado |
| `rule_diff_analyzer.py` | ~30 KB | ⚠️ Recomendado |
| `analyze_standard_drift.py` | ~20 KB | ⚠️ Recomendado |
| `config_test_single.yaml` | ~10 KB | ✅ Sim |
| `compare_gbml_vs_river.py` | ~25 KB | ⏭️ Opcional |

**Total**: ~355 KB

---

## 🔧 COMANDO PARA LISTAR ARQUIVOS (Local)

### Windows PowerShell:
```powershell
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"
dir main.py, ga.py, data_handling_v8.py, hill_climbing_v2.py, plotting.py, analyze_concept_difference.py, chunk_transition_analyzer.py, rule_diff_analyzer.py, analyze_standard_drift.py, config_test_single.yaml
```

### Windows CMD:
```cmd
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"
dir main.py & dir ga.py & dir data_handling_v8.py & dir hill_climbing_v2.py & dir plotting.py & dir config_test_single.yaml
```

---

## 📝 NOTAS IMPORTANTES

1. **config_test_single.yaml** é específico para este teste (1 stream, 6 chunks)
2. **Não fazer upload de**:
   - `venv/` ou `venv_comparison/`
   - `__pycache__/`
   - `.git/`
   - Logs antigos (`*.log`)
   - Resultados antigos (`results/`, `experiments/`)

3. **Se faltar algum arquivo no Colab**:
   - Verificar nome exato (case-sensitive)
   - Verificar diretório correto (`/DSL-AG-hybrid/` não `/DSL-AG-hybrid/src/`)

---

## 🎯 PRÓXIMO PASSO

Após upload completo:

1. ✅ Abrir Google Colab
2. ✅ Criar novo notebook ou usar existente
3. ✅ Copiar células do arquivo `TESTE_COLAB_SINGLE_STREAM.md`
4. ✅ Executar teste!

---

**Tempo estimado para upload**: 2-5 minutos (depende da conexão)
**Tempo estimado para teste no Colab**: 6-8 horas

**BOA SORTE!** 🚀
