# Guia Completo: Execução no Google Colab

**Experimento:** 11 Datasets + 6 Modelos (GBML, ACDWM, HAT, ARF, SRP, ERulesD2S)

**Tempo Total:** ~14 horas

**Data:** 2025-11-12

---

## 📋 Pré-requisitos

1. ✅ **Google Colab Pro** (recomendado para 24h de runtime)
2. ✅ **Google Drive** com ~5GB de espaço livre
3. ✅ **Arquivos do projeto** preparados

---

## 🎯 Passo a Passo

### 1. Preparar Arquivos Localmente (10 min)

#### 1.1 Compactar Projeto

No Windows:
```powershell
cd "C:\Users\Leandro Almeida\Downloads"
Compress-Archive -Path "DSL-AG-hybrid" -DestinationPath "DSL-AG-hybrid.zip"
```

Ou usar WinRAR/7-Zip para criar `DSL-AG-hybrid.zip`

#### 1.2 Upload para Google Drive

1. Abrir Google Drive no navegador
2. Criar pasta: `Meus experimentos`
3. Upload do `DSL-AG-hybrid.zip` (~50MB)
4. Aguardar conclusão do upload

**Arquivos essenciais no ZIP:**
- `config_experiment_expanded.yaml` ✅
- `compare_gbml_vs_river.py` ✅
- `arff_converter.py` ✅
- `erulesd2s_wrapper.py` ✅
- `setup_erulesd2s.py` ✅
- Todos os módulos Python (ga.py, river_evaluator.py, etc.)

---

### 2. Abrir Notebook no Colab (2 min)

#### 2.1 Upload do Notebook

1. Acessar: https://colab.research.google.com/
2. File → Upload notebook
3. Selecionar: `Experimento_Completo_11_Datasets_6_Modelos.ipynb`

#### 2.2 Configurar Runtime

1. Runtime → Change runtime type
2. Selecionar:
   - **Hardware accelerator:** GPU
   - **GPU type:** T4 (ou superior se disponível)
3. Clicar **Save**

#### 2.3 Verificar GPU

Executar célula:
```python
!nvidia-smi
```

Deve mostrar GPU Tesla T4 (ou V100/A100).

---

### 3. Execução Passo a Passo (~14h total)

#### 3.1 Setup Inicial (5-10 min)

**Células 1.1 a 1.4:**
- ✅ Verificar GPU
- ✅ Montar Google Drive
- ✅ Descompactar projeto
- ✅ Instalar dependências Python

**Verificar:**
```python
import river
print(f"River: {river.__version__}")  # Deve mostrar versão
```

---

#### 3.2 Setup ERulesD2S (15-20 min)

**Células 2.1 a 2.3:**
- ✅ Instalar Java 11 e Maven (~5 min)
- ✅ Clonar e compilar ERulesD2S (~10 min)
- ✅ Teste rápido (~2 min)

**Verificar:**
```bash
ls -lh erulesd2s.jar  # Deve existir (30-50 MB)
```

---

#### 3.3 Configuração e Preparação (2 min)

**Células 3.1 e 3.2:**
- ✅ Verificar configuração (11 datasets)
- ✅ Ativar keep-alive script
- ✅ Ativar backup automático (a cada hora)

---

#### 3.4 EXECUÇÃO PRINCIPAL (~14 horas)

**Célula 4.2 - NÃO INTERROMPER!**

Esta célula executa todo o experimento usando `run_experiment_complete.py`:

**O script executa automaticamente:**

**FASE 1:** Modelos Python (~11h)
- Itera sobre 11 datasets
- Para cada dataset: GBML, ACDWM, HAT, ARF, SRP
- 4 chunks × 15 min/chunk (GBML) + overhead (outros modelos)
- Total: ~11 horas

**FASE 2:** ERulesD2S (~3h)
- Itera sobre 11 datasets
- Para cada dataset: converte ARFF + executa ERulesD2S
- 4 chunks × 4 min/chunk
- Total: ~3 horas

**Monitoramento:**
- Backup automático salva resultados a cada 1 hora
- Use célula 4.3 para ver progresso sem interromper
- Progresso é exibido automaticamente durante execução

---

#### 3.5 Consolidação e Backup (~5 min)

**Células 5.1 a 6.2:**
- ✅ Consolidar resultados
- ✅ Copiar para Google Drive
- ✅ Criar backup compactado
- ✅ Verificar integridade

---

### 4. Durante a Execução

#### 4.1 Monitorar Progresso

**Executar célula 4.3 periodicamente:**
```python
# Ver progresso
!ls experiment_expanded_results/*/comparison_table.csv | wc -l  # Datasets completados
```

#### 4.2 Verificar Logs

```python
# Ver últimas linhas do log
!tail -n 50 experiment_expanded_results/latest.log
```

#### 4.3 Verificar Backups

```python
# Listar backups no Drive
!ls -lh /content/drive/MyDrive/experiment_backup/
```

---

### 5. Após Conclusão

#### 5.1 Baixar Resultados

1. Acessar Google Drive
2. Pasta: `/Meus experimentos/`
3. Baixar:
   - `experiment_expanded_results.tar.gz` (~500 MB)
   - `experiment_expanded_results_complete_6_models.csv` (~200 KB)

#### 5.2 Extrair Localmente

```powershell
cd "C:\Users\Leandro Almeida\Downloads"
tar -xzf experiment_expanded_results.tar.gz
```

#### 5.3 Análise Estatística

```bash
cd DSL-AG-hybrid
python statistical_analysis_expanded.py
```

---

## ⏱️ Cronograma Detalhado

| Fase | Duração | Células | Pode Interromper? |
|------|---------|---------|-------------------|
| Setup inicial | 5-10 min | 1.1-1.4 | ✅ Sim |
| Setup ERulesD2S | 15-20 min | 2.1-2.3 | ✅ Sim |
| Configuração | 2 min | 3.1-3.2 | ✅ Sim |
| **EXECUÇÃO** | **14 horas** | **4.2** | **❌ NÃO!!!** |
| Consolidação | 5 min | 5.1-6.2 | ✅ Sim |
| **Total** | **~14h 30min** | | |

---

## 🔍 Troubleshooting

### Problema 1: Sessão desconecta

**Sintoma:** Colab desconecta após algumas horas

**Solução:**
- Keep-alive já está ativado (célula 3.2)
- Verificar se navegador não hiberna
- Usar Colab Pro (24h vs 12h)

### Problema 2: GPU out of memory

**Sintoma:** CUDA out of memory

**Solução:**
- Restart runtime
- Reduzir `population_size` no config.yaml

### Problema 3: Java/Maven falha

**Sintoma:** ERulesD2S não compila

**Solução:**
```bash
# Re-executar setup
!apt-get install --reinstall -y openjdk-11-jdk-headless maven
!python setup_erulesd2s.py
```

### Problema 4: Backup não salva

**Sintoma:** Erro ao copiar para Drive

**Solução:**
```python
# Remount Drive
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive')
```

### Problema 5: Runtime restart inesperado

**Sintoma:** Execução interrompida

**Solução:**
1. Verificar últimos resultados salvos:
```bash
ls -lt experiment_expanded_results/*/comparison_table.csv | head
```

2. Identificar último dataset completado

3. Modificar `config_experiment_expanded.yaml`:
```yaml
drift_simulation_experiments:
  # Remover datasets já completados
  - Dataset_Faltante_1
  - Dataset_Faltante_2
  # ...
```

4. Re-executar célula 4.2

---

## 📊 Resultados Esperados

### Arquivos Gerados

```
experiment_expanded_results/
├── RBF_Abrupt_Severe_seed42_*/
│   ├── comparison_table.csv
│   ├── gbml_results.csv
│   ├── acdwm_results.csv
│   ├── river_HAT_results.csv
│   ├── river_ARF_results.csv
│   ├── river_SRP_results.csv
│   ├── erulesd2s_results.csv  ← Novo!
│   └── *.png
├── (+ 10 datasets)
└── ...

experiment_expanded_results_complete_6_models.csv  ← Consolidado
```

### Métricas Coletadas

Por cada avaliação (33 por modelo × 6 modelos = 198 total):
- Accuracy
- G-mean (métrica principal)
- F1-score weighted
- Precision/Recall por classe
- Tempo de execução

### Ranking Esperado

```
RANKING POR G-MEAN MÉDIO (6 MODELOS)
────────────────────────────────────
1. SRP           : 0.7200
2. ARF           : 0.7050
3. ERulesD2S     : 0.6950  ← Novo!
4. ACDWM         : 0.6900
5. GBML          : 0.6800
6. HAT           : 0.6700
```

---

## ✅ Checklist Final

### Antes de Executar

- [ ] Google Colab Pro ativado
- [ ] GPU configurada (T4+)
- [ ] ZIP do projeto no Google Drive
- [ ] Notebook carregado no Colab
- [ ] Keep-alive ativado
- [ ] Backup automático ativado

### Durante Execução

- [ ] Monitorar progresso (célula 4.3)
- [ ] Verificar backups horários
- [ ] Não fechar navegador
- [ ] Não interromper célula 4.2

### Após Conclusão

- [ ] Baixar `experiment_expanded_results.tar.gz`
- [ ] Baixar CSV consolidado
- [ ] Verificar 11 datasets completados
- [ ] Executar análise estatística local
- [ ] Gerar relatório final

---

## 📞 Suporte

### Se algo der errado:

1. **Verificar logs:**
```bash
tail -n 100 experiment_expanded_results/latest.log
```

2. **Verificar backups:**
```bash
ls -lh /content/drive/MyDrive/experiment_backup/
```

3. **Salvar estado atual:**
```bash
!tar -czf emergency_backup.tar.gz experiment_expanded_results/
!cp emergency_backup.tar.gz /content/drive/MyDrive/
```

4. **Restart runtime** e retomar de onde parou

---

## 🎓 Após os Resultados

### Análise Estatística

1. **Carregar dados:**
```python
import pandas as pd
df = pd.read_csv('experiment_expanded_results_complete_6_models.csv')
```

2. **Testes estatísticos:**
   - Kruskal-Wallis (6 modelos)
   - Mann-Whitney U (comparações pareadas)
   - Cohen's d (tamanho de efeito)
   - Intervalos de confiança (95%)

3. **Análises específicas:**
   - Performance por tipo de drift (abrupt vs gradual)
   - Performance por gerador (RBF, SEA, etc.)
   - Tempo de adaptação após drift
   - Consistência entre datasets

### Relatório Final

Gerar documento com:
- Resumo executivo
- Tabelas consolidadas
- Gráficos comparativos
- Análise estatística
- Conclusões e próximos passos

---

**Última atualização:** 2025-11-12

**Status:** ✅ Pronto para Execução no Colab

**Tempo total:** ~14h 30min (setup + execução + backup)

**Boa sorte com o experimento!** 🚀
