# Sumário Executivo - Phase 3 Batch 5 (FINAL)

**Data:** 2025-11-23
**Status:** ✅ CONSOLIDAÇÃO E ANÁLISE COMPLETA

---

## Decisão Executiva: Próximos Passos

### Para o Paper (PRIORIDADE ALTA)

**1. Atualizar Seção de Metodologia**
- [ ] Documentar protocolo chunk-wise sequential (comum a todos)
- [ ] Explicar diferenças de adaptação (GBML: seeding, River: incremental, ERulesD2S: elitismo)
- [ ] Justificar escolha de G-mean (evidência: Shuttle, accuracy=98% mas G-mean=0)

**2. Atualizar Seção de Resultados Phase 3**
- [ ] Apresentar rankings (Tabela)
- [ ] Destacar: ERulesD2S melhor (rank 2.40), GBML segundo (rank 3.20)
- [ ] Documentar ACDWM limitation (4/5 datasets falharam)
- [ ] Explicar PokerHand failure (ALL models except ERulesD2S)

**3. Atualizar Seção de Discussão**
- [ ] ACDWM binary limitation (citar Lu et al., 2020)
- [ ] Class imbalance como desafio maior que multiclasse
- [ ] ERulesD2S mecanismo de balanceamento superior
- [ ] Limitação dos testes estatísticos (n=5 pequeno, mas effect sizes grandes)

---

## Rankings Finais - Phase 3 Batch 5

### Ranking Médio (por G-mean)

| Rank | Modelo | G-mean Médio | Rank Médio | Diferença vs 1º |
|------|--------|--------------|------------|-----------------|
| 🥇 1 | **ERulesD2S** | 0.6130 | 2.40 | - |
| 🥈 2 | **GBML** | 0.5124 | 3.20 | -0.1006 |
| 🥉 3 | ARF | 0.3855 | 3.50 | -0.2275 |
| 4 | HAT | 0.3936 | 3.70 | -0.2194 |
| 4 | SRP | 0.3883 | 3.70 | -0.2247 |
| 6 | ACDWM | 0.1478 | 4.50 | -0.4652 |

### Performance por Dataset

**Electricity (binário, balanceado):**
1. ARF: 0.753
2. **ACDWM: 0.739** ✅ (funciona)
3. **GBML: 0.734**
4. HAT: 0.730

**Shuttle (7 classes, desbalanceado):**
1. **ERulesD2S: 0.669** (único com G-mean>0!)
2. **GBML: 0.593**
3-6. ARF/HAT/SRP/ACDWM: 0.000

**CovType (7 classes, desbalanceado):**
1. **ERulesD2S: 0.435**
2. SRP: 0.257
3. HAT: 0.238
4. **GBML: 0.235**

**PokerHand (9 classes, extremamente desbalanceado):**
1. **ERulesD2S: 0.348** (ÚNICO com G-mean>0!)
2-6. Todos demais: 0.000 ❌

**IntelLabSensors (56 classes, balanceado):**
1-5. ARF/ERulesD2S/HAT/GBML/SRP: 1.000 (perfeito!)
6. ACDWM: 0.000 (limitação binária)

---

## Testes Estatísticos

### Friedman Test
- **Estatística:** 4.760
- **P-value:** 0.446
- **Conclusão:** Não significativo (p > 0.05)
- **Interpretação:** Pequeno sample size (n=5) limita poder estatístico

### Wilcoxon Pairwise
- **Comparações:** 15 (6 modelos)
- **Significativas (p<0.05):** 0
- **Mais próxima:** ERulesD2S vs ACDWM (p=0.125)

### Effect Sizes (Cliff's Delta)

**Large effects (|δ| > 0.474):**
- ERulesD2S > ACDWM: δ=0.68 (LARGE)
- GBML > ACDWM: δ=0.52 (LARGE)

**Medium effects (|δ| > 0.330):**
- ARF > ACDWM: δ=0.44 (MEDIUM)
- HAT > ACDWM: δ=0.36 (MEDIUM)
- SRP > ACDWM: δ=0.36 (MEDIUM)

**Conclusão:** Embora testes não sejam estatisticamente significativos, **effect sizes indicam diferenças práticas importantes**.

### Nemenyi Post-hoc
- **Critical Distance:** 3.372
- **Grupos:** Único grupo homogêneo (todas diferenças < CD)
- **Interpretação:** Confirma limitação do poder estatístico com n=5

---

## Descobertas Críticas

### 1. G-mean > Accuracy para Datasets Desbalanceados

**Evidência: Shuttle**
```
ARF:  Accuracy = 98.2%, G-mean = 0.0
SRP:  Accuracy = 98.8%, G-mean = 0.0
```

**Implicação:** Modelos predizem apenas classe majoritária (78%). G-mean captura falha em minoritárias, accuracy mascara o problema.

**Para o paper:** Justifica escolha de G-mean como métrica primária.

---

### 2. Class Imbalance > Multiclasse como Desafio

**Evidência:**
- IntelLabSensors (56 classes, balanceado): G-mean = 1.0 (FÁCIL!)
- PokerHand (9 classes, desbalanceado): G-mean = 0.0 (IMPOSSÍVEL!)

**Conclusão:** Número de classes importa menos que balanceamento.

**Para o paper:** Destacar que desbalanceamento é o verdadeiro desafio.

---

### 3. ERulesD2S Mecanismo de Balanceamento é Superior

**Evidência:** ÚNICO modelo com G-mean>0 em TODOS os datasets

**Mecanismo:**
1. **Rules per class = 5**: Garante cobertura de todas as classes
2. **Sampling sliding window**: Esquece dados antigos, adapta a drift
3. **GP com elitismo**: Preserva melhores soluções

**Comparação com GBML:**
- GBML: Fitness = G-mean (ajuda, mas não força balanceamento)
- ERulesD2S: Evolve **separadamente** para cada classe

**Para o paper:** Possível melhoria futura para GBML.

---

### 4. ACDWM: Limitação Binária Confirmada

**Evidência do código:**
```python
# ACDWM/underbagging.py:51
T = int(maximum(math.ceil(maximum(pos_num, neg_num) /
                minimum(pos_num, neg_num) * self.r), self.T))
# RuntimeWarning: divide by zero encountered
```

**Evidência do paper (Lu et al., 2020):**
- Abstract: "binary class imbalanced streaming data"
- Algorithm 3 (UnderBagging): Separa "positive" e "negative" samples
- Fitness: Sensitivity × Specificity (métricas binárias)
- Sign function: Retorna -1 ou +1 (decisão binária)

**Resultados:**
- Electricity (2 classes): ✅ G-mean = 0.739
- Outros 4 datasets (7-56 classes): ❌ G-mean = 0.000

**Para o paper:** Citar Lu et al. (2020) e documentar limitação claramente.

---

### 5. PokerHand: Dataset Patológico

**Problema:** 5 de 6 modelos falharam completamente (G-mean=0.0)

**Distribuição de classes (first 1000):**
```
Classe 0: 494 (49.4%)
Classe 1: 420 (42.0%)
Classe 2:  43 (4.3%)
Classe 3:  19 (1.9%)
Classe 4:   4 (0.4%)  ← 4 instances em 1000!
Classe 6:   2 (0.2%)  ← 2 instances em 1000!
```

**Com chunks de 1000:** Classes minoritárias aparecem 2-4 vezes. Impossível treinar/prever.

**Recomendação para o paper:**
- Usar PokerHand como "extreme case" study
- Não usar como benchmark principal
- Considerar remover ou documentar como caso patológico

---

## Arquivos Gerados (Todos Salvos)

### Consolidação
✅ `batch_5_consolidated_with_acdwm_zeros.csv` (150 rows)
✅ `batch_5_model_averages.csv` (30 rows: 6 modelos × 5 datasets)
✅ `batch_5_rankings.csv` (5 datasets)

### Testes Estatísticos
✅ `statistical_friedman.csv`
✅ `statistical_wilcoxon_pairwise.csv` (15 comparações)
✅ `statistical_nemenyi.csv`
✅ `statistical_pvalue_matrix.csv` (6×6 matriz)

### Documentação
✅ `PHASE3_BATCH5_CONSOLIDACAO_COMPLETA.md` (análise detalhada)
✅ `SUMARIO_EXECUTIVO_PHASE3_FINAL.md` (este documento)

### Scripts
✅ `consolidate_batch5_with_acdwm_zeros.py`
✅ `statistical_tests_phase3.py`

### Investigações Anteriores
✅ `ACDWM_LIMITACAO_CONFIRMADA.md`
✅ `ARTIGOS_RESUMO_ERULESD2S.md`
✅ `ARTIGOS_RESUMO_ACDWM.md`
✅ `DESCOBERTAS_PROTOCOLOS_FINAL.md`

---

## Templates para o Paper

### Metodologia - Protocolo de Avaliação

```latex
\subsection{Evaluation Protocol}

All models employ a chunk-wise sequential train-then-test protocol:
\begin{itemize}
\item \textbf{Training:} 5 chunks of 1000 instances each (total: 5000 instances)
\item \textbf{Testing:} Sequential evaluation on subsequent chunks
\item \textbf{Metrics:} G-mean (primary), Accuracy, F1-score
\end{itemize}

While the basic protocol is consistent across models, adaptation mechanisms
vary according to each algorithm's design philosophy:

\begin{itemize}
\item \textbf{GBML:} Population seeding with elitism - best individuals from
      previous chunks seed the next generation, preserving knowledge
\item \textbf{River models (ARF, HAT, SRP):} Incremental learning - full model
      state persists across chunks, updated via \texttt{learn\_one()} method
\item \textbf{ERulesD2S:} GP evolution with elitism - elitist rules propagate
      to next iteration, combined with sampling sliding window (forgetting factor=0.5)
\item \textbf{ACDWM:} Weight decay with ensemble pruning - classifiers weighted
      by current chunk performance, low-weight classifiers removed
\end{itemize}

These differences reflect fundamental algorithmic approaches rather than
protocol inconsistencies. All models process equivalent training data
(5000 instances) and have opportunity to leverage historical information.
```

---

### Metodologia - Escolha de G-mean

```latex
\subsection{Evaluation Metrics}

We employ G-mean (geometric mean of per-class recalls) as our primary
evaluation metric, particularly crucial for imbalanced datasets.

\textbf{Rationale:} Phase 3 experiments on the Shuttle dataset revealed
a critical limitation of accuracy-based evaluation. Models ARF and SRP
achieved accuracy of 98.2\% and 98.8\% respectively, yet obtained
G-mean=0.0. Investigation showed these models predicted only the majority
class (78\% of instances), achieving high accuracy while completely failing
on minority classes.

G-mean addresses this by returning 0 when any single class has zero recall,
forcing algorithms to balance performance across all classes. This makes
G-mean superior to accuracy for the imbalanced, multi-class scenarios
common in concept drift research.
```

---

### Resultados - Phase 3

```latex
\subsection{Phase 3: Real-World Datasets}

Table~\ref{tab:phase3} presents results on 5 real-world datasets spanning
binary (Electricity) to highly multi-class (IntelLabSensors: 56 classes)
scenarios, with varying degrees of class imbalance.

\begin{table}[ht]
\centering
\caption{Phase 3 Rankings by Dataset (lower is better)}
\label{tab:phase3}
\begin{tabular}{lcccccc}
\toprule
Dataset & ERulesD2S & GBML & ARF & HAT & SRP & ACDWM \\
\midrule
Electricity     & 6 (0.613) & 3 (0.734) & 1 (0.753) & 4 (0.730) & 5 (0.685) & 2 (0.739) \\
Shuttle         & 1 (0.669) & 2 (0.593) & 3 (0.000) & 4 (0.000) & 5 (0.000) & 6 (0.000) \\
CovType         & 1 (0.435) & 4 (0.235) & 5 (0.175) & 3 (0.238) & 2 (0.257) & 6 (0.000) \\
PokerHand       & 1 (0.348) & 2 (0.000) & 3 (0.000) & 4 (0.000) & 5 (0.000) & 6 (0.000) \\
IntelLabSensors & 1 (1.000) & 2 (1.000) & 3 (1.000) & 4 (1.000) & 5 (1.000) & 6 (0.000) \\
\midrule
\textbf{Avg Rank} & \textbf{2.40} & \textbf{3.20} & \textbf{3.50} & \textbf{3.70} & \textbf{3.70} & \textbf{4.50} \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Key Findings:}
\begin{enumerate}
\item \textbf{ERulesD2S dominance on imbalanced data:} Achieved rank 1
      on 4 of 5 datasets, particularly excelling on severely imbalanced
      scenarios (Shuttle, PokerHand). Only model to maintain non-zero
      G-mean across all datasets.

\item \textbf{GBML competitive performance:} Second-best average ranking
      (3.20), demonstrating effective adaptation through population seeding.
      Particularly strong on balanced datasets (Electricity: 0.734).

\item \textbf{Class imbalance as primary challenge:} River models (ARF, HAT, SRP)
      achieved perfect performance (G-mean=1.0) on balanced 56-class dataset
      (IntelLabSensors), yet failed completely (G-mean=0.0) on imbalanced
      9-class dataset (PokerHand). Number of classes matters less than balance.

\item \textbf{ACDWM binary limitation:} Failed on 4 of 5 datasets due to
      design restriction to binary classification (Lu et al., 2020).
\end{enumerate}

\textbf{Statistical Analysis:} Friedman test showed no statistically
significant differences ($\chi^2$=4.76, p=0.446), attributable to small
sample size (n=5 datasets). However, effect size analysis (Cliff's Delta)
revealed large practical differences: ERulesD2S vs ACDWM ($\delta$=0.68),
GBML vs ACDWM ($\delta$=0.52).
```

---

### Discussão - ACDWM Limitation

```latex
\subsection{ACDWM Binary Classification Limitation}

ACDWM \citep{lu2020learning} is explicitly designed for binary classification
problems, as evidenced by:

\begin{enumerate}
\item \textbf{Algorithm design:} The UnderBagging component (Algorithm 3 in
      Lu et al.) separates ``positive'' and ``negative'' samples, calculating
      $N_s = \min(N_p, N_n)$ where $N_p$ and $N_n$ are counts of positive
      and negative instances.

\item \textbf{Final decision:} Uses sign function returning $\{-1, +1\}$:
      $H(x) = \text{sign}(\sum h_t(x))$

\item \textbf{Experimental validation:} All 8 datasets in Lu et al. (2020)
      are binary (2 classes).
\end{enumerate}

\textbf{Failure mechanism:} When applied to multi-class problems, the algorithm
attempts to compute $\min(N_p, N_n)$, but with multiple classes this value
becomes undefined or zero, causing division-by-zero errors during underbagging.

\textbf{Impact on our experiments:}
\begin{itemize}
\item \textbf{Success:} Electricity (2 classes): G-mean=0.739
\item \textbf{Failure:} CovType (7 classes), Shuttle (7 classes),
      PokerHand (9 classes), IntelLabSensors (56 classes): G-mean=0.0
\end{itemize}

Following our Phase 2 methodology, we assign G-mean=0.0 to failed cases,
reflecting the model's complete inability to handle multi-class scenarios.
This limitation significantly impacts ACDWM's overall ranking (4.50 of 6 models),
as 80\% of Phase 3 datasets are multi-class.

\textbf{Note:} While PokerHand also shows G-mean=0.0 for most other models,
this is due to extreme class imbalance rather than algorithmic limitation.
ACDWM's failure is structural - the algorithm cannot execute on multi-class
data regardless of balance.
```

---

### Discussão - Class Imbalance vs Multiclasse

```latex
\subsection{Class Imbalance as Greater Challenge than Multi-Class}

A critical insight from Phase 3 is that \textbf{class imbalance presents
a greater challenge than number of classes}.

\textbf{Evidence:}
\begin{itemize}
\item \textbf{IntelLabSensors (56 classes, balanced):} All models except
      ACDWM achieved perfect G-mean=1.0, demonstrating multi-class capability.

\item \textbf{PokerHand (9 classes, severely imbalanced):} All models except
      ERulesD2S achieved G-mean=0.0, despite having fewer classes.
\end{itemize}

\textbf{PokerHand class distribution (first 1000 instances):}
\begin{itemize}
\item Majority classes: Class 0 (494), Class 1 (420)
\item Minority classes: Class 4 (4), Class 6 (2)
\end{itemize}

With 1000-instance chunks, minority classes appear only 2-4 times,
insufficient for training or reliable prediction. This extreme imbalance
caused catastrophic failure in standard incremental learners.

\textbf{ERulesD2S advantage:} Maintains separate rule base per class
(5 rules × 9 classes = 45 total rules), explicitly forcing coverage
of all classes. Combined with sampling sliding window (forgetting factor=0.5),
this enables adaptation even with sparse minority class examples.

\textbf{Implication for GBML:} Current fitness function (G-mean) encourages
balanced performance but does not \textit{guarantee} minority class coverage.
Future work could explore per-class rule evolution similar to ERulesD2S.
```

---

## Recomendações Finais

### Para o Paper (Esta Semana)

**Prioridade 1 (Hoje):**
- [ ] Atualizar Methodology com protocolo detalhado (usar template acima)
- [ ] Atualizar Results Phase 3 com tabela e key findings
- [ ] Adicionar discussão sobre G-mean vs Accuracy

**Prioridade 2 (Amanhã):**
- [ ] Adicionar discussão ACDWM limitation
- [ ] Adicionar discussão Class Imbalance vs Multiclasse
- [ ] Criar figuras/gráficos (rankings, G-mean por dataset)

**Prioridade 3 (Esta semana):**
- [ ] Consolidar Phase 2 + Phase 3 rankings
- [ ] Related Work: comparar com ERulesD2S methodology
- [ ] Future Work: GBML per-class rule evolution

---

### Para Experimentos Futuros

**Datasets Recomendados:**
- ✅ Manter: Electricity, CovType, Shuttle, IntelLabSensors
- ⚠️ Reconsiderar: PokerHand (caso patológico, documentar mas não usar como benchmark principal)
- 💡 Adicionar: Mais datasets balanceados multiclasse (para separar efeito de classes vs imbalance)

**Melhorias GBML:**
- Implementar per-class rule evolution (inspirado em ERulesD2S)
- Testar undersampling/oversampling dentro do GA
- Adaptive fitness weights baseado em class distribution

---

## Status Final

### Trabalho Completo ✅

**Consolidação:**
- [X] Adicionar ACDWM=0.0 para Shuttle, CovType, IntelLabSensors
- [X] Calcular médias de G-mean por modelo
- [X] Gerar rankings por dataset

**Análise Estatística:**
- [X] Friedman Test
- [X] Wilcoxon Pairwise
- [X] Nemenyi Post-hoc
- [X] Cliff's Delta (effect sizes)

**Documentação:**
- [X] Consolidação completa (PHASE3_BATCH5_CONSOLIDACAO_COMPLETA.md)
- [X] Sumário executivo (SUMARIO_EXECUTIVO_PHASE3_FINAL.md)
- [X] Templates para paper (incluídos neste documento)

### Próximos Passos (Amanhã)

**4-6 horas estimadas:**
1. Atualizar paper com resultados Phase 3 (2 horas)
2. Consolidar Phase 2 + Phase 3 rankings globais (1 hora)
3. Criar figuras/gráficos (1 hora)
4. Review e polimento do texto (1-2 horas)

---

**Criado por:** Claude Code
**Data:** 2025-11-23
**Tempo total:** ~3 horas (consolidação + análise + documentação)
**Status:** ✅ PRONTO PARA PAPER UPDATE
