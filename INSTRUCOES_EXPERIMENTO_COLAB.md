# INSTRUCOES - EXPERIMENTO COMPARATIVO NO GOOGLE COLAB

**Experimento:** GBML vs River (HAT, ARF, SRP)
**Datasets:** 3 (RBF_Abrupt_Severe, RBF_Abrupt_Moderate, RBF_Gradual_Moderate)
**Chunks:** 3 por dataset
**Tempo estimado:** 15.8 horas
**Status:** PRONTO PARA EXECUTAR

---

## ARQUIVOS CRIADOS

### Novos Arquivos:
1. `config_comparison.yaml` - Configuracao do experimento
2. `run_comparison_colab.py` - Script de execucao automatizado
3. `PLANO_EXPERIMENTO_COMPARATIVO_RIVER.md` - Documentacao completa
4. `INSTRUCOES_EXPERIMENTO_COLAB.md` - Este arquivo

### Arquivos Existentes (JA PRONTOS):
1. `compare_gbml_vs_river.py` - Comparacao principal
2. `baseline_river.py` - Modelos River
3. `shared_evaluation.py` - Avaliacao compartilhada
4. `gbml_evaluator.py` - Avaliador GBML
5. `main.py` - GBML principal
6. `ga.py` - Algoritmo genetico (com Layer 1)

---

## COMANDO UNICO PARA EXECUTAR NO COLAB

### Passo 1: Montar Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Passo 2: Navegar para o Diretorio

```python
import os
os.chdir('/content/drive/MyDrive/DSL-AG-hybrid')
```

### Passo 3: Instalar Dependencias (se necessario)

```python
!pip install river pandas matplotlib seaborn pyyaml scikit-learn imbalanced-learn
```

### Passo 4: Executar Experimento

```python
!python run_comparison_colab.py 2>&1 | tee experiment_comparison_full.log
```

**PRONTO!** O experimento vai rodar por ~15.8h e salvar todos os dados automaticamente.

---

## ESTRUTURA DE SAIDA

Apos o experimento, voce tera:

```
comparison_results/
└── experiment_20251105_HHMMSS/
    ├── experiment_config.json
    ├── experiment_summary.json
    ├── consolidated_results.csv       ← TODOS os resultados juntos
    ├── summary_statistics.txt         ← Estatisticas resumidas
    ├── RBF_Abrupt_Severe_seed42_.../
    │   ├── gbml_results.csv
    │   ├── river_HAT_results.csv
    │   ├── river_ARF_results.csv
    │   ├── river_SRP_results.csv
    │   ├── comparison_table.csv
    │   ├── summary.txt
    │   └── *.png (graficos)
    ├── RBF_Abrupt_Moderate_seed42_.../
    │   └── (mesma estrutura)
    └── RBF_Gradual_Moderate_seed42_.../
        └── (mesma estrutura)
```

---

## MONITORAMENTO DURANTE EXECUCAO

### Ver Progresso

```python
!tail -f experiment_comparison_full.log
```

### Ver Ultimo Dataset Executado

```python
!ls -lt comparison_results/experiment_*/
```

### Ver Tempo Decorrido

```python
!cat comparison_results/experiment_*/experiment_summary.json
```

---

## SE ALGO DER ERRADO

### Erro: "River nao instalado"

```python
!pip install river --upgrade
!python -c "import river; print(river.__version__)"
```

### Erro: "compare_gbml_vs_river.py nao encontrado"

Verifique se esta no diretorio correto:

```python
!pwd
!ls -la *.py
```

### Erro: "CUDA out of memory"

Reinicie o runtime:
- Runtime → Restart Runtime
- Execute novamente do Passo 1

### Parar Experimento

```python
# Ctrl+C no notebook ou
# Runtime → Interrupt execution
```

---

## COMANDOS ALTERNATIVOS

### Executar Apenas 1 Dataset (teste)

```python
!python compare_gbml_vs_river.py \
  --stream RBF_Abrupt_Severe \
  --config config_comparison.yaml \
  --models HAT ARF SRP \
  --chunks 3 \
  --chunk-size 6000 \
  --seed 42 \
  --output comparison_results
```

Tempo: ~5.5h (1 dataset)

---

### Executar Sem GBML (apenas River - rapido)

```python
!python compare_gbml_vs_river.py \
  --stream RBF_Abrupt_Severe \
  --config config_comparison.yaml \
  --models HAT ARF SRP \
  --chunks 3 \
  --no-gbml \
  --output comparison_results
```

Tempo: ~15min (1 dataset, 3 modelos River)

---

### Executar Sem River (apenas GBML)

```python
!python compare_gbml_vs_river.py \
  --stream RBF_Abrupt_Severe \
  --config config_comparison.yaml \
  --chunks 3 \
  --no-river \
  --output comparison_results
```

Tempo: ~4.5h (1 dataset, apenas GBML)

---

## ANALISE DOS RESULTADOS

### Carregar Resultados Consolidados

```python
import pandas as pd

# Carregar resultados
df = pd.read_csv('comparison_results/experiment_XXXXXX/consolidated_results.csv')

# Ver primeiras linhas
print(df.head())

# Estatisticas por modelo
summary = df.groupby('model').agg({
    'accuracy': ['mean', 'std'],
    'gmean': ['mean', 'std'],
    'f1_weighted': ['mean', 'std']
}).round(4)

print(summary)
```

### Ver Graficos Gerados

```python
from IPython.display import Image, display

# Lista todos os graficos
import glob
plots = glob.glob('comparison_results/experiment_*/**/*.png', recursive=True)

# Mostra todos
for plot in plots:
    print(f"\n{plot}")
    display(Image(filename=plot))
```

### Comparar GBML vs Melhor River

```python
# Calcular medias por modelo
model_means = df.groupby('model')['gmean'].mean().sort_values(ascending=False)

print("Ranking por G-mean medio:")
print(model_means)

# Melhor River vs GBML
best_river = model_means[model_means.index != 'GBML'].iloc[0]
gbml_score = model_means.loc['GBML']

print(f"\nMelhor River: {model_means.index[0]} = {best_river:.4f}")
print(f"GBML: {gbml_score:.4f}")
print(f"Diferenca: {gbml_score - best_river:+.4f}")
```

---

## CHECKLIST PRE-EXECUCAO

Antes de executar, verifique:

- [ ] Google Drive montado
- [ ] Diretorio correto (`/content/drive/MyDrive/DSL-AG-hybrid`)
- [ ] Arquivos presentes:
  - [ ] `run_comparison_colab.py`
  - [ ] `config_comparison.yaml`
  - [ ] `compare_gbml_vs_river.py`
  - [ ] `baseline_river.py`
  - [ ] `shared_evaluation.py`
  - [ ] `gbml_evaluator.py`
  - [ ] `main.py`
  - [ ] `ga.py`
- [ ] Dependencias instaladas (river, pandas, etc)
- [ ] Espaco em disco: ~1GB livre
- [ ] Colab Pro/Pro+ recomendado (tempo longo)

---

## TIMELINE ESTIMADA

| Hora | Atividade | Status |
|------|-----------|--------|
| 0:00 | Inicio - RBF_Abrupt_Severe | Em andamento |
| 5:30 | RBF_Abrupt_Severe concluido | |
| 5:30 | Inicio - RBF_Abrupt_Moderate | |
| 11:00 | RBF_Abrupt_Moderate concluido | |
| 11:00 | Inicio - RBF_Gradual_Moderate | |
| 16:30 | RBF_Gradual_Moderate concluido | |
| 16:30 | Consolidacao de resultados | |
| 17:00 | EXPERIMENTO COMPLETO | |

**Margem:** 1h (ate 18:00h max)

---

## TROUBLESHOOTING COMUM

### "Colab desconectou"

**Solucao:** Use Colab Pro/Pro+ ou configure keep-alive:

```javascript
// Execute no console do navegador (F12)
function KeepClicking(){
   console.log("Keeping session alive");
   document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(KeepClicking,60000)
```

### "Disco cheio"

```python
# Limpar cache
!rm -rf /content/drive/MyDrive/DSL-AG-hybrid/__pycache__
!rm -rf /content/drive/MyDrive/DSL-AG-hybrid/comparison_results/old_*

# Ver espaco
!df -h /content/drive
```

### "Muito lento"

Verifique GPU:

```python
!nvidia-smi  # Se GPU disponivel
```

Se nao tiver GPU, considere:
- Reduzir para 2 datasets
- Ou reduzir chunks para 2
- Ou usar Colab Pro

---

## COMANDOS QUICK REFERENCE

```bash
# Executar experimento completo
python run_comparison_colab.py

# Ver progresso
tail -f experiment_comparison_full.log

# Listar resultados
ls -lh comparison_results/experiment_*/

# Ver summary
cat comparison_results/experiment_*/summary_statistics.txt

# Consolidar manualmente (se script falhar)
python -c "
import pandas as pd
import glob
files = glob.glob('comparison_results/**/comparison_table.csv', recursive=True)
dfs = [pd.read_csv(f) for f in files]
consolidated = pd.concat(dfs)
consolidated.to_csv('consolidated_all.csv', index=False)
print(f'Consolidado: {len(consolidated)} linhas')
"
```

---

## PROXIMOS PASSOS APOS EXPERIMENTO

1. Baixar resultados:
   - `consolidated_results.csv`
   - `summary_statistics.txt`
   - Graficos PNG

2. Analisar:
   - Qual modelo teve melhor G-mean?
   - GBML competitivo vs River?
   - Performance em drift abrupto vs gradual?

3. Documentar:
   - Criar tabelas comparativas
   - Graficos adicionais
   - Interpretacao dos resultados

4. Publicar:
   - Paper/relatorio
   - Apresentacao
   - GitHub

---

**PRONTO PARA EXECUTAR!**

Execute o comando no Passo 4 e aguarde ~16h.

Todos os dados serao salvos automaticamente.

BOA SORTE!
