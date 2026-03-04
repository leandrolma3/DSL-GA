# 📤 ARQUIVOS PARA SINCRONIZAR - RE-TESTE 6 CHUNKS

**Data**: 2025-10-29
**Objetivo**: Listar arquivos que precisam ser atualizados no Google Drive/Colab
**Destino**: `/DSL-AG-hybrid/` no Google Drive

---

## 🎯 RESUMO EXECUTIVO

### Arquivos que MUDARAM (precisam ser substituídos):
1. ✅ **main.py** - 2 ajustes críticos (linhas 344-346 e 1474)

### Arquivos que já existem e NÃO mudaram:
- `config_test_single.yaml` - Já estava correto
- `test_real_results_heatmaps/concept_heatmapsS/concept_differences.json` - Já existe

### Total a sincronizar: **1 arquivo** (main.py)

---

## 📋 CHECKLIST DE SINCRONIZAÇÃO

### OPÇÃO 1: Sincronizar APENAS o Arquivo Modificado (RECOMENDADO)

**Vantagem**: Rápido (apenas 1 arquivo ~50KB)

**Passos**:
1. ✅ Acessar Google Drive: https://drive.google.com
2. ✅ Navegar para: `/DSL-AG-hybrid/`
3. ✅ **Substituir** arquivo existente:
   ```
   main.py  (~50 KB)
   ```
4. ✅ Aguardar upload completar

**Tempo estimado**: 10-30 segundos

---

### OPÇÃO 2: Sincronizar Pasta Completa (Seguro)

**Vantagem**: Garante que tudo está atualizado

**Arquivos essenciais** (já devem existir no Drive, mas vale verificar):

```
DSL-AG-hybrid/
├── main.py                                    ⬆️ SUBSTITUIR (MODIFICADO)
├── config_test_single.yaml                    ✅ Já existe (não mudou)
├── ga.py                                      ✅ Já existe (não mudou)
├── data_handling_v8.py                        ✅ Já existe (não mudou)
├── hill_climbing_v2.py                        ✅ Já existe (não mudou)
├── plotting.py                                ✅ Já existe (não mudou)
├── analyze_concept_difference.py              ✅ Já existe (não mudou)
└── test_real_results_heatmaps/
    └── concept_heatmapsS/
        └── concept_differences.json           ✅ Já existe (não mudou)
```

**Tempo estimado**: 1-2 minutos (se usar Google Drive Desktop)

---

## 🔍 VALIDAÇÃO PRÉ-UPLOAD

### Verificar Modificações em main.py:

**Ajuste 1 - Linha 1474** (deve ter):
```python
config_file_path = os.path.join(script_dir, "config_test_single.yaml")
```

**Ajuste 2 - Linhas 344-346** (deve ter):
```python
diff_file_path = os.path.join(diff_file_path, "concept_differences.json")
# Make path absolute relative to script directory to handle execution from different locations
if not os.path.isabs(diff_file_path):
    diff_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), diff_file_path)
if os.path.exists(diff_file_path):
```

**Comando para validar localmente**:
```bash
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"

# Verificar linha 1474
grep -n "config_test_single.yaml" main.py

# Verificar linhas 344-346
grep -n -A2 "Make path absolute" main.py
```

**Resultado esperado**:
```
1474:    config_file_path = os.path.join(script_dir, "config_test_single.yaml")
345:    if not os.path.isabs(diff_file_path):
346:        diff_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), diff_file_path)
```

---

## 📤 MÉTODOS DE SINCRONIZAÇÃO

### Método 1: Upload Manual via Web (Mais Simples)

1. **Acessar**: https://drive.google.com
2. **Navegar**: Pasta `/DSL-AG-hybrid/`
3. **Deletar** arquivo antigo: `main.py`
4. **Upload** arquivo novo:
   - Arrastar `main.py` da pasta local
   - OU clicar "Novo" → "Upload de arquivo"
5. **Confirmar**: Aguardar upload completar (barra de progresso)

**Tempo**: ~30 segundos

---

### Método 2: Google Drive Desktop (Mais Rápido)

**Se já tem Google Drive instalado**:

1. **Abrir** pasta sincronizada no Windows:
   ```
   C:\Users\Leandro Almeida\Google Drive\DSL-AG-hybrid\
   ```
2. **Substituir** arquivo:
   - Copiar `main.py` da pasta local para pasta sincronizada
   - Confirmar substituição
3. **Aguardar** sincronização automática (ícone de nuvem verde)

**Tempo**: ~10 segundos (sincronização automática)

---

### Método 3: Linha de Comando (Para Usuários Avançados)

**Se tem `rclone` configurado**:

```bash
rclone copy "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid\main.py" \
    "gdrive:DSL-AG-hybrid/" --verbose
```

**Se tem `gcloud` configurado**:

```bash
gsutil cp "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid\main.py" \
    "gs://seu-bucket/DSL-AG-hybrid/"
```

---

## ✅ CHECKLIST PÓS-UPLOAD

### No Google Colab, validar arquivos:

**Célula 1 - Validar Upload**:
```python
from google.colab import drive
import os

drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/DSL-AG-hybrid')

# Verificar arquivo modificado
print("Validando main.py...")
with open('main.py', 'r') as f:
    content = f.read()

# Verificar ajuste 1
if 'config_test_single.yaml' in content:
    print("✅ Ajuste 1 OK: config_test_single.yaml encontrado")
else:
    print("❌ ERRO: Ajuste 1 não encontrado!")

# Verificar ajuste 2
if 'Make path absolute' in content:
    print("✅ Ajuste 2 OK: Caminho absoluto implementado")
else:
    print("❌ ERRO: Ajuste 2 não encontrado!")

# Verificar concept_differences.json
diff_path = 'test_real_results_heatmaps/concept_heatmapsS/concept_differences.json'
if os.path.exists(diff_path):
    print(f"✅ concept_differences.json existe")
    import json
    with open(diff_path) as f:
        data = json.load(f)
    severity = data.get('RBF', {}).get('c1_vs_c2_severe', 0)
    print(f"   Severidade c1_vs_c2_severe: {severity:.2f}%")
else:
    print(f"❌ ERRO: {diff_path} não encontrado!")

print("\n✅ Todos os arquivos validados com sucesso!" if all([
    'config_test_single.yaml' in content,
    'Make path absolute' in content,
    os.path.exists(diff_path)
]) else "\n❌ Há erros! Verificar arquivos.")
```

**Resultado esperado**:
```
✅ Ajuste 1 OK: config_test_single.yaml encontrado
✅ Ajuste 2 OK: Caminho absoluto implementado
✅ concept_differences.json existe
   Severidade c1_vs_c2_severe: 60.45%

✅ Todos os arquivos validados com sucesso!
```

---

## 🚨 TROUBLESHOOTING

### Problema 1: Upload Lento

**Sintomas**: Upload demora mais de 2 minutos

**Soluções**:
1. Verificar conexão de internet
2. Tentar upload via Google Drive Desktop (mais estável)
3. Dividir upload em lotes menores

---

### Problema 2: Arquivo Antigo Ainda Aparece

**Sintomas**: No Colab, main.py não tem os ajustes

**Soluções**:
1. **No Colab**, fazer reload:
   ```python
   # Desmontar e remontar Drive
   from google.colab import drive
   drive.flush_and_unmount()
   drive.mount('/content/drive', force_remount=True)
   ```

2. **No Google Drive**, verificar:
   - Se upload completou (sem ícone de sincronização)
   - Se não há versões antigas em cache

3. **Confirmar tamanho**: main.py deve ter ~51-52 KB (com os ajustes)

---

### Problema 3: concept_differences.json Não Encontrado

**Sintomas**: Arquivo não existe no Drive

**Solução**:
1. **Copiar da máquina local**:
   ```
   Local: C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid\test_real_results_heatmaps\concept_heatmapsS\concept_differences.json
   Drive: /DSL-AG-hybrid/test_real_results_heatmaps/concept_heatmapsS/concept_differences.json
   ```

2. **Criar diretórios se necessário**:
   ```python
   # No Colab
   import os
   os.makedirs('test_real_results_heatmaps/concept_heatmapsS', exist_ok=True)
   ```

3. **Upload do arquivo**

---

## 📊 COMPARAÇÃO DE MÉTODOS

| Método | Tempo | Dificuldade | Confiabilidade |
|--------|-------|-------------|----------------|
| **Upload Web Manual** | ~30s | Fácil | Alta ✅ |
| **Google Drive Desktop** | ~10s | Fácil | Muito Alta ✅✅ |
| **rclone** | ~5s | Difícil | Alta ✅ |

**Recomendação**: Upload Web Manual (mais simples e confiável)

---

## 🎯 RESUMO PARA AÇÃO RÁPIDA

### Passo a Passo Simplificado:

1. ✅ Acessar: https://drive.google.com
2. ✅ Ir para pasta: `/DSL-AG-hybrid/`
3. ✅ Deletar arquivo antigo: `main.py`
4. ✅ Upload arquivo novo: Arrastar `main.py` da pasta:
   ```
   C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid\main.py
   ```
5. ✅ Aguardar upload completar
6. ✅ No Colab, executar célula de validação (código acima)

**Tempo total**: 1-2 minutos

---

## 📝 NOTAS IMPORTANTES

### 1. NÃO Precisa Sincronizar:

- ❌ `venv/` ou `venv_comparison/`
- ❌ `__pycache__/`
- ❌ `experiments_test/` (resultados antigos)
- ❌ Logs antigos (`*.log`)
- ❌ Documentos markdown (`*.md`) - apenas para referência local

### 2. Backup Recomendado:

Antes de substituir `main.py` no Drive, considere:
- Renomear o antigo para `main_old.py` (backup)
- Ou fazer download do antigo antes de substituir

### 3. Após Re-Teste:

Quando o re-teste terminar, considere:
- Baixar resultados novos de `experiments_test/` para local
- Comparar logs (teste anterior vs re-teste)
- Gerar plots de ambos para comparação lado a lado

---

**Documento criado por**: Claude Code
**Data**: 2025-10-29
**Status**: ✅ **PRONTO PARA SINCRONIZAR - APENAS 1 ARQUIVO**
