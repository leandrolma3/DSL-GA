# PLANO DE ACAO - ARTIGO IEEE

**Data**: 2025-12-16
**Objetivo**: Integrar todos os experimentos e preparar artigo IEEE com foco em explicabilidade

---

## 1. MAPEAMENTO DOS EXPERIMENTOS EXISTENTES

### 1.1 Diretorios de Experimentos

| Diretorio | Descricao | chunk_size | PENALTY_WEIGHT | Batches |
|-----------|-----------|------------|----------------|---------|
| `experiments_6chunks_phase1_gbml` | Experimento original (drift) | 1000 | 0.0 | 1-4 |
| `experiments_6chunks_phase2_gbml` | Experimento original (drift) | 1000 | 0.0 | 1-4 |
| `experiments_6chunks_phase3_real` | Experimento original (real+stat) | 1000 | 0.0 | 5-7 |
| `experiments_chunk2000_phase1` | Chunk dobrado (drift) | 2000 | 0.0 | 1-4 |
| `experiments_chunk2000_phase2` | Chunk dobrado (real+stat) | 2000 | 0.0 | 5-7 |
| `experiments_balanced_phase1` | Balanceado (drift) | 2000 | 0.1 | 1-4 |
| `experiments_balanced_phase2` | Balanceado (real+stat) | 2000 | 0.1 | 5-7 |

### 1.2 Configuracao dos Batches (52 datasets total)

| Batch | Tipo | # Datasets | Exemplos |
|-------|------|------------|----------|
| 1 | Abrupt Drift | 12 | SEA, AGRAWAL, STAGGER, RBF |
| 2 | Gradual Drift | 9 | SEA_Gradual, AGRAWAL_Gradual |
| 3 | Drift + Ruido | 8 | SEA_Noise, AGRAWAL_Noise |
| 4 | SINE/LED/WAVE | 6 | SINE_Abrupt, LED_Abrupt, WAVEFORM |
| 5 | Real | 5 | Electricity, Shuttle, CovType, Poker, IntelLab |
| 6 | Sintetico Estac. | 6 | SEA_Stationary, AGRAWAL_Stationary |
| 7 | Sintetico Estac. | 6 | STAGGER_Stationary, WAVEFORM_Stationary |

### 1.3 Modelos Comparativos (8 modelos, incluindo 2 versoes ROSE)

| # | Modelo | Tipo | Explicavel? | Limitacoes |
|---|--------|------|-------------|------------|
| 1 | **GBML (proposto)** | Regras evolucionarias | SIM | - |
| 2 | **ROSE_Original** | Ensemble (MOA) | NAO | Avaliacao prequential |
| 3 | **ROSE_ChunkEval** | Ensemble (MOA) | NAO | Avaliacao por chunk |
| 4 | HAT | Arvore adaptativa | Parcial | - |
| 5 | ARF | Ensemble floresta | NAO | - |
| 6 | SRP | Ensemble patches | NAO | - |
| 7 | ACDWM | Ensemble DWM | NAO | **So binario** |
| 8 | ERulesD2S | Regras evolucionarias | SIM* | Sem metricas transicao |

**NOTA IMPORTANTE sobre ROSE:**
- **ROSE_Original**: Avaliacao prequential instance-by-instance (como no paper original)
- **ROSE_ChunkEval**: Avaliacao por chunk (media das metricas, **comparavel com GBML**)

*ERulesD2S exporta apenas agregados (NumberRules, NumberConditions), nao as regras em si.

---

## 2. PROPOSTA DE NOME PARA O MODELO

### 2.1 Criterios para o Nome

1. Enfatizar **explicabilidade/interpretabilidade** como diferencial
2. Mencionar **data streams** ou processamento temporal
3. Ser memoravel e adequado para IEEE
4. Refletir a natureza baseada em **regras**

### 2.2 Opcoes Propostas

| Nome | Significado | Justificativa |
|------|-------------|---------------|
| **XStream-Rules** | eXplainable Stream Rules | Claro, direto, enfatiza explicabilidade |
| **EGIS** | Evolutionary Grammar Interpretable Streams | Curto, memoravel, academico |
| **SIRL** | Stream Interpretable Rule Learner | Descritivo, focado em interpretabilidade |
| **GE-XRL** | Grammar Evolution eXplainable Rule Learning | Tecnico, detalhado |
| **DSL-AG-X** | DSL with Adaptive Grammar - eXplainable | Extensao do nome atual |

**Recomendacao**: XStream-Rules ou EGIS

---

## 3. ESTRUTURA DO NOTEBOOK ATUAL

### 3.1 Execute_All_Comparative_Models.ipynb

**Celulas principais:**
- 1.1-1.7: Setup (Drive, Java, ACDWM, ERulesD2S)
- 2.0: Verificacao de dependencias
- 2.1: Configuracao de batches e datasets
- 3.1: Funcoes ROSE (run_rose_original, run_rose_chunk_eval)
- 3.2: Funcoes River (run_river_model - HAT, ARF, SRP)
- 3.3: Funcoes ACDWM (run_acdwm)
- 4.x: Execucao por batch
- 5.x: Consolidacao e analise

**Diretorios atualmente configurados:**
```python
EXPERIMENT_DIRS = [
    'experiments_6chunks_phase2_gbml',
    'experiments_6chunks_phase3_real'
]
```

**Timeouts:**
- ROSE_Original: 600s
- ROSE_ChunkEval: 600s
- HAT: 300s
- ARF: 600s
- SRP: 600s
- ACDWM: 600s
- ERulesD2S: 1800s

**Variavel critica:**
```python
CHUNK_SIZE = 1000  # <<< DEVE SER ALTERADO PARA 2000 nos novos experimentos
```

### 3.2 analyze_explainability.py

**Funcionalidades:**
- `extract_gbml_rule_details()`: Le rule_details_per_chunk.json
- `parse_rules_history()`: Parseia RulesHistory_*.txt
- `calculate_transition_metrics()`: Calcula TCS, RIR, AMS
- `extract_erulesd2s_metrics()`: Le CSVs do ERulesD2S
- `generate_statistical_analysis()`: Gera relatorio

**Metricas de Transicao (exclusivas GBML):**
- **TCS** (Transition Change Score): Score composto de mudanca
- **RIR** (Rule Instability Rate): Taxa de instabilidade
- **AMS** (Average Modification Severity): Severidade das modificacoes

**Pesos TCS:**
```python
W_INSTABILITY = 0.6
W_MODIFICATION_IMPACT = 0.4
```

---

## 4. PLANO DE ACAO EM FASES

### FASE 1: Preparacao dos Dados (Prioridade Alta)

**1.1 Verificar completude dos experimentos**
- [ ] Listar todos os datasets em cada diretorio de experimento
- [ ] Identificar datasets faltantes em experiments_chunk2000_*
- [ ] Verificar se reruns foram executados (STAGGER, RANDOMTREE, IntelLabSensors)
- [ ] Documentar status de cada batch

**1.2 Executar modelos comparativos para chunk_size=2000**
- [ ] Adaptar Execute_All_Comparative_Models.ipynb para experiments_chunk2000_*
- [ ] Alterar CHUNK_SIZE de 1000 para 2000
- [ ] Alterar base_dir em BATCH_CONFIG para experiments_chunk2000_*
- [ ] Executar ROSE_Original e ROSE_ChunkEval em todos os 52 datasets
- [ ] Executar HAT, ARF, SRP em todos os 52 datasets
- [ ] Executar ACDWM (somente datasets binarios - ~30 datasets)
- [ ] Executar ERulesD2S em todos os 52 datasets
- [ ] Salvar resultados em estrutura organizada

**1.3 Aguardar experimentos balanced**
- [ ] Monitorar execucao de experiments_balanced_*
- [ ] Apos conclusao, executar modelos comparativos

### FASE 2: Analise de Explicabilidade (Prioridade Alta)

**2.1 Adaptar analyze_explainability.py**
- [ ] Adicionar suporte a todos os diretorios de experimento
- [ ] Parametrizar EXPERIMENT_DIRS via argumento
- [ ] Adicionar geracao de CSVs separados por experimento

**2.2 Executar analise para cada conjunto**
- [ ] Chunk_size=1000 (original)
- [ ] Chunk_size=2000
- [ ] Balanced (PENALTY_WEIGHT=0.1)

**2.3 Consolidar metricas de transicao**
- [ ] Gerar tabela TCS/RIR/AMS por dataset e experimento
- [ ] Calcular estatisticas agregadas
- [ ] Identificar padroes relacionados a drift

### FASE 3: Comparacao Estatistica (Prioridade Alta)

**3.1 Tabelas de performance**
- [ ] G-Mean medio por modelo e dataset
- [ ] Desvio padrao e intervalos de confianca
- [ ] Ranking de modelos por metrica

**3.2 Testes estatisticos**
- [ ] Friedman test para comparacao multipla
- [ ] Nemenyi post-hoc test
- [ ] Wilcoxon signed-rank para pares

**3.3 Tabelas de complexidade (explicabilidade)**
- [ ] Numero medio de regras
- [ ] Condicoes por regra
- [ ] Comparacao GBML vs ERulesD2S

### FASE 4: Visualizacoes (Prioridade Media)

**4.1 Graficos de performance**
- [ ] Boxplots comparativos de G-Mean
- [ ] Heatmaps de performance por drift type
- [ ] Critical Difference Diagram

**4.2 Graficos de explicabilidade**
- [ ] Evolucao de TCS ao longo dos chunks
- [ ] Scatter plot: Performance vs Complexidade
- [ ] Comparacao de numero de regras GBML vs ERulesD2S

**4.3 Graficos de estabilidade**
- [ ] RIR ao longo do stream
- [ ] Correlacao drift detection vs RIR

### FASE 5: Preparacao do Artigo (Prioridade Media)

**5.1 Estrutura LaTeX**
```
paper/
  main.tex
  sections/
    abstract.tex
    introduction.tex
    related_work.tex
    methodology.tex
    experiments.tex
    results.tex
    conclusion.tex
  figures/
    performance_boxplot.pdf
    critical_difference.pdf
    explainability_comparison.pdf
    tcs_evolution.pdf
  tables/
    performance_comparison.tex
    complexity_comparison.tex
    statistical_tests.tex
  bibliography.bib
```

**5.2 Tabelas para artigo**
- [ ] Table 1: Dataset characteristics
- [ ] Table 2: Performance comparison (G-Mean)
- [ ] Table 3: Statistical significance tests
- [ ] Table 4: Interpretability metrics (rules, conditions)
- [ ] Table 5: Transition metrics (TCS, RIR, AMS)

**5.3 Figuras para artigo**
- [ ] Figure 1: System architecture
- [ ] Figure 2: Performance comparison boxplot
- [ ] Figure 3: Critical difference diagram
- [ ] Figure 4: Explainability comparison
- [ ] Figure 5: Rule stability over stream

### FASE 6: Validacao e Revisao (Prioridade Baixa)

**6.1 Validacao de resultados**
- [ ] Verificar consistencia entre experimentos
- [ ] Cruzar resultados com logs de execucao
- [ ] Identificar outliers e investigar

**6.2 Revisao do manuscrito**
- [ ] Revisar claims principais
- [ ] Verificar reproducibilidade
- [ ] Preparar material suplementar

---

## 5. MODIFICACOES NECESSARIAS NO NOTEBOOK

### 5.1 Estrutura Proposta

```python
# Configuracao de experimentos
EXPERIMENTS = {
    'chunk1000': {
        'dirs': [
            'experiments_6chunks_phase2_gbml/batch_1',
            'experiments_6chunks_phase2_gbml/batch_2',
            # ... batches 3-4
            'experiments_6chunks_phase3_real/batch_5',
            # ... batches 6-7
        ],
        'chunk_size': 1000,
        'description': 'Original (chunk_size=1000, PENALTY_WEIGHT=0.0)'
    },
    'chunk2000': {
        'dirs': [
            'experiments_chunk2000_phase1/batch_1',
            # ... batches 2-4
            'experiments_chunk2000_phase2/batch_5',
            # ... batches 6-7
        ],
        'chunk_size': 2000,
        'description': 'Doubled chunk (chunk_size=2000, PENALTY_WEIGHT=0.0)'
    },
    'balanced': {
        'dirs': [
            'experiments_balanced_phase1/batch_1',
            # ... batches 2-4
            'experiments_balanced_phase2/batch_5',
            # ... batches 6-7
        ],
        'chunk_size': 2000,
        'description': 'Balanced (chunk_size=2000, PENALTY_WEIGHT=0.1)'
    }
}

# Selecionar experimento
CURRENT_EXPERIMENT = 'chunk2000'  # Alterar conforme necessidade
```

### 5.2 Output Organizado

```
comparison_results/
  chunk1000/
    all_models_consolidated_results.csv
    pivot_gmean_all_models.csv
    critical_difference_diagram.png
    by_batch/
      batch_1_results.csv
      ...
    by_model/
      ROSE_Original_results.csv
      ROSE_ChunkEval_results.csv
      HAT_results.csv
      ARF_results.csv
      SRP_results.csv
      ACDWM_results.csv
      ERulesD2S_results.csv
  chunk2000/
    (mesma estrutura)
  balanced/
    (mesma estrutura)
  cross_experiment/
    all_experiments_comparison.csv
    statistical_analysis.csv
```

---

## 6. METRICAS PARA O ARTIGO

### 6.1 Performance

| Metrica | Descricao |
|---------|-----------|
| G-Mean | Media geometrica de sensitividades por classe |
| F1-Score | Media harmonica de precisao e recall |
| Accuracy | Acuracia (referencia) |

### 6.2 Explicabilidade (Diferencial)

| Metrica | Descricao | Disponivel em |
|---------|-----------|---------------|
| # Regras | Numero de regras por chunk | GBML, ERulesD2S |
| # Condicoes | Condicoes por regra | GBML, ERulesD2S |
| **TCS** | Transition Change Score | **GBML apenas** |
| **RIR** | Rule Instability Rate | **GBML apenas** |
| **AMS** | Average Modification Severity | **GBML apenas** |

### 6.3 Eficiencia

| Metrica | Descricao |
|---------|-----------|
| Tempo de treinamento | Segundos por chunk |
| Memoria | Uso de memoria (se disponivel) |

---

## 7. CRONOGRAMA SUGERIDO

| Fase | Tarefas | Dependencias |
|------|---------|--------------|
| 1 | Preparacao dados | Nenhuma |
| 2 | Analise explicabilidade | Fase 1 |
| 3 | Comparacao estatistica | Fases 1, 2 |
| 4 | Visualizacoes | Fase 3 |
| 5 | Preparacao artigo | Fases 3, 4 |
| 6 | Validacao e revisao | Fase 5 |

---

## 8. PROXIMOS PASSOS IMEDIATOS

1. **Verificar status dos experimentos chunk_size=2000**
   - Quantos datasets completaram?
   - Quais faltam?

2. **Definir nome final do modelo**
   - Escolher entre: XStream-Rules, EGIS, SIRL, GE-XRL, DSL-AG-X
   - Renomear referencias no codigo e documentacao

3. **Adaptar notebook para chunk_size=2000**
   - Modificar EXPERIMENT_DIRS
   - Ajustar chunk_size nos parametros
   - Testar com 1 dataset antes de execucao completa

4. **Criar estrutura de pastas para artigo**
   - paper/
   - figures/
   - tables/

---

## 9. RESUMO DAS CORRECOES APLICADAS

### Correcoes da Revisao (2025-12-16)
1. **ROSE adicionado**: Incluidas duas versoes (ROSE_Original e ROSE_ChunkEval)
2. **Total de modelos corrigido**: 8 modelos (nao 6)
3. **CHUNK_SIZE documentado**: Variavel critica que deve ser alterada
4. **Timeouts atualizados**: Incluidos ROSE_Original e ROSE_ChunkEval
5. **Output estruturado**: Adicionados arquivos especificos do ROSE

### Documentacao Complementar
- `REVISAO_COMPLETA_NOTEBOOK_COMPARATIVO.md`: Analise detalhada do notebook

---

**Autor**: Claude Code
**Status**: PLANO REVISADO E CORRIGIDO
**Ultima atualizacao**: 2025-12-16
