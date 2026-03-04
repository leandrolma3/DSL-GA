# Situacao dos Plots - Batch 1

**Data**: 2025-11-19
**Diagnostico**: Plots ja foram gerados, mas em formato diferente do esperado

---

## DESCOBERTA IMPORTANTE

O experimento GBML **JA GEROU OS PLOTS** durante a execucao, mas:
- Os plots estao no diretorio `run_1/` (raiz)
- O script `generate_plots.py` espera criar plots em `run_1/plots/` (subpasta)
- O formato de salvamento mudou de **pickle** para **JSON**

---

## ARQUIVOS QUE EXISTEM

Para cada experimento em `experiments_6chunks_phase2_gbml/batch_1/<DATASET>/run_1/`:

### Plots (PNG) - JA GERADOS:
```
✅ GA_Evolution_Chunk0_<DATASET>_Run1.png
✅ GA_Evolution_Chunk1_<DATASET>_Run1.png
✅ GA_Evolution_Chunk2_<DATASET>_Run1.png
✅ GA_Evolution_Chunk3_<DATASET>_Run1.png
✅ GA_Evolution_Chunk4_<DATASET>_Run1.png
✅ G-meanPlot_Periodic_<DATASET>_Run1.png          <- GRAFICO PRINCIPAL
✅ RuleComponents_Heatmap_<DATASET>_Run1.png
✅ RuleComponents_Radar_<DATASET>_Run1.png
```

### Dados (JSON) - FORMATO NOVO:
```
✅ chunk_metrics.json                    (metricas por chunk)
✅ periodic_gmean.json                   (G-mean periodico)
✅ ga_history_per_chunk.json             (historia do GA)
✅ rule_details_per_chunk.json           (detalhes das regras)
✅ attribute_usage_per_chunk.json        (uso de atributos)
```

### Dados (PKL):
```
✅ best_individuals.pkl                  (melhores individuos)
```

### Outros:
```
✅ RulesHistory_<DATASET>_Run1.txt       (historia das regras)
✅ run_config.json                       (configuracao da execucao)
```

---

## ARQUIVOS QUE NAO EXISTEM

### Formato Antigo (Pickle):
```
❌ ResultsDictionary_<DATASET>_Run1_ChunkSize1000.pkl
```

Este arquivo era gerado pela versao antiga do codigo. A versao atual usa JSON.

### Predicoes (CSV):
```
❌ test_predictions_chunk*.csv
❌ train_predictions_chunk*.csv
```

Estes arquivos nao sao mais gerados (ou foram movidos para outro local).

---

## O QUE SIGNIFICA?

**BOA NOTICIA**: Todos os plots que voce precisa **JA EXISTEM**!

**Plots Disponiveis:**
1. **G-meanPlot_Periodic** - Grafico principal mostrando:
   - G-mean ao longo do tempo
   - Periodico (train-then-test)
   - **IMPORTANTE**: Verificar se mostra marcadores de drift

2. **GA_Evolution_Chunk** - Evolucao do GA em cada chunk:
   - Fitness, acuracia, F1
   - Tamanho da populacao
   - Convergencia

3. **RuleComponents_Heatmap** - Heatmap de componentes de regras
4. **RuleComponents_Radar** - Radar de componentes de regras

---

## DIFERENCA DOS PLOTS

### Plots JA GERADOS (experimento):
- Localizacao: `run_1/` (raiz)
- Gerados durante o experimento
- Usam dados da execucao em tempo real

### Plots que generate_plots.py GERARIA:
- Localizacao: `run_1/plots/` (subpasta)
- Gerados apos o experimento
- Leem dados do ResultsDictionary.pkl
- **INCLUEM**: Marcadores de drift com severity (se concept_differences.json existir)

**DIFERENCA PRINCIPAL**: Os plots gerados por `generate_plots.py` mostrariam as **linhas verticais de drift** com os **percentuais de severity**.

---

## VERIFICACAO NECESSARIA

Os plots existentes **TEM marcadores de drift?**

Para verificar, abra o arquivo:
```
experiments_6chunks_phase2_gbml/batch_1/SEA_Abrupt_Simple/run_1/G-meanPlot_Periodic_SEA_Abrupt_Simple_Run1.png
```

Procure por:
- ✅ Linhas verticais vermelhas em 3000 instancias
- ✅ Texto "Drift at 3000 (XX% severity)"
- ✅ Queda/mudanca na acuracia nessas posicoes

Se **SIM**: Os plots ja estao completos!
Se **NAO**: Precisamos gerar novos plots com marcadores de drift.

---

## SOLUCOES DISPONIVEIS

### Solucao 1: Organizar Plots Existentes (RAPIDO)

**Execute**:
```python
!cd /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid && python organizar_plots_batch_1.py
```

**O que faz**:
- Copia plots existentes para subpasta `plots/`
- Organiza estrutura de diretorios
- Mantem originais intactos

**Resultado**:
```
experiments_6chunks_phase2_gbml/batch_1/<DATASET>/run_1/
├── GA_Evolution_Chunk0_<DATASET>_Run1.png
├── G-meanPlot_Periodic_<DATASET>_Run1.png
├── ... (originais)
└── plots/
    ├── GA_Evolution_Chunk0_<DATASET>_Run1.png
    ├── G-meanPlot_Periodic_<DATASET>_Run1.png
    └── ... (copias organizadas)
```

**Vantagem**: Rapido, usa plots que ja existem
**Desvantagem**: Plots podem nao ter marcadores de drift severity

---

### Solucao 2: Atualizar generate_plots.py (COMPLETO mas COMPLEXO)

Modificar `generate_plots.py` para:
- Ler dados de `chunk_metrics.json` ao inves de `ResultsDictionary.pkl`
- Gerar plots com marcadores de drift
- Incluir severity de `concept_differences.json`

**Vantagem**: Plots com informacao completa de drift
**Desvantagem**: Requer modificacao do codigo

---

### Solucao 3: Usar Plots Existentes Diretamente (IMEDIATO)

Usar os plots que ja existem sem mover:
```python
# Para cada base:
plots_dir = "experiments_6chunks_phase2_gbml/batch_1/<DATASET>/run_1/"
# Abrir G-meanPlot_Periodic_*.png diretamente
```

**Vantagem**: Imediato, sem modificacoes
**Desvantagem**: Plots nao estao na subpasta esperada

---

## RECOMENDACAO

### PASSO 1: Verificar plots existentes

Abra alguns plots existentes e verifique se tem marcadores de drift:
```python
from IPython.display import Image, display
display(Image('/content/drive/.../SEA_Abrupt_Simple/run_1/G-meanPlot_Periodic_SEA_Abrupt_Simple_Run1.png'))
```

### PASSO 2A: Se plots TEM marcadores de drift

Execute o script de organizacao:
```python
!python organizar_plots_batch_1.py
```

Pronto! Plots organizados em subpastas.

### PASSO 2B: Se plots NAO TEM marcadores de drift

Precisamos criar uma versao adaptada do `generate_plots.py` que:
1. Leia dados de JSON ao inves de pickle
2. Adicione marcadores de drift com severity
3. Gere plots na subpasta `plots/`

---

## PROXIMOS PASSOS

Apos verificar/organizar os plots:

1. **Analise Visual**:
   - Abrir Plot_AccuracyPeriodic de cada base
   - Verificar se drifts aparecem nas posicoes corretas
   - Confirmar que severities sao diferentes de 0%

2. **Rule Diff Analysis**:
   - Ja foi executado (arquivos rule_diff_analysis_* existem)
   - Verificar relatorios para ver mudancas de regras

3. **Comparativos**:
   - Executar modelos River (HAT, ARF, SRP)
   - Executar ACDWM e ERulesD2S
   - Gerar comparacoes estatisticas

4. **Consolidacao**:
   - CELULA 11: Statistical comparison
   - CELULA 12: Final plots
   - Gerar tabelas para paper

---

## SCRIPTS DISPONIVEIS

| Script | Funcao | Quando Usar |
|--------|--------|-------------|
| `diagnostico_batch_1.py` | Verificar arquivos | Antes de pos-processar |
| `organizar_plots_batch_1.py` | Copiar plots para plots/ | Se plots ja tem drift markers |
| `post_process_batch_1.py` | Pos-processamento completo | Se ResultsDictionary.pkl existir |

---

## RESUMO

**Status Atual**:
- ✅ Experimentos executados (12/12)
- ✅ Plots gerados (na raiz de run_1/)
- ✅ RulesHistory criado (12/12)
- ✅ Rule diff analysis executado
- ❌ ResultsDictionary.pkl (formato antigo nao usado)
- ❌ Plots na subpasta plots/ (ainda nao organizados)

**Proximo Passo**:
1. Verificar se plots existentes tem marcadores de drift
2. Se SIM: executar `organizar_plots_batch_1.py`
3. Se NAO: criar versao adaptada do generate_plots.py

---

**Status**: PLOTS JA EXISTEM - VERIFICACAO NECESSARIA
**Acao Recomendada**: Abrir plots e verificar marcadores de drift antes de decidir proxima acao
