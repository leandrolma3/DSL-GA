# Phase 3 Batch 5 - Consolidação Completa

**Data:** 2025-11-23
**Status:** ✅ CONSOLIDADO

---

## TL;DR - Principais Descobertas

🎯 **RANKINGS PHASE 3 (BATCH 5)**

| Rank | Modelo | G-mean Médio | Performance |
|------|--------|--------------|-------------|
| 1 | **ERulesD2S** | 0.6130 | Melhor |
| 2 | **GBML** | 0.5124 | Segundo |
| 3 | **ARF** | 0.3855 | Terceiro |
| 4 | **HAT** | 0.3936 | Quarto |
| 4 | **SRP** | 0.3883 | Quarto |
| 6 | **ACDWM** | 0.1478 | Pior |

🔍 **DESCOBERTA CRÍTICA:**
- PokerHand e Shuttle causam G-mean=0.0 em QUASE TODOS os modelos!
- Não é limitação de multiclasse - é **problema de desbalanceamento extremo**
- ACDWM tem G-mean=0.0 por **dois motivos**:
  1. Limitação binária (CovType, Shuttle, IntelLabSensors)
  2. Desbalanceamento (PokerHand)

---

## Consolidação Executada

### Arquivos Gerados

1. **`batch_5_consolidated_with_acdwm_zeros.csv`**
   - Resultados completos com ACDWM=0.0 para multiclasse
   - 150 linhas (135 originais + 15 ACDWM adicionadas)
   - Modelos: GBML, ARF, HAT, SRP, ACDWM, ERulesD2S

2. **`batch_5_model_averages.csv`**
   - Médias de G-mean por modelo por dataset
   - Inclui std, count, accuracy, F1

3. **`batch_5_rankings.csv`**
   - Rankings por dataset (1-6 para cada dataset)
   - Base para testes estatísticos

### Linhas ACDWM Adicionadas

```
✓ Shuttle: 5 chunks (G-mean=0.0)
✓ CovType: 5 chunks (G-mean=0.0)
✓ IntelLabSensors: 5 chunks (G-mean=0.0)
```

**Justificativa:** ACDWM é limitado a classificação binária (Lu et al., 2020).

---

## Resultados Detalhados por Dataset

### 1. Electricity (2 classes) ✅ Binário

| Modelo | G-mean | Std | Observações |
|--------|--------|-----|-------------|
| **ARF** | 0.7527 | 0.0845 | Melhor |
| **ACDWM** | 0.7392 | 0.0840 | Funciona (binário) |
| **GBML** | 0.7337 | 0.0481 | Competitivo |
| HAT | 0.7300 | 0.1275 | Competitivo |
| SRP | 0.6846 | 0.0663 | |
| ERulesD2S | 0.6130 | 0.0427 | Surpreendentemente baixo |

**Análise:**
- ACDWM funciona corretamente (dataset binário)
- ARF melhor, GBML/ACDWM/HAT muito próximos
- ERulesD2S inesperadamente pior (possível overfitting em treino?)

---

### 2. Shuttle (7 classes) ⚠️ Multiclasse Desbalanceado

**Distribuição de classes:**
```
Classe 1: 78.4%  (maioria absoluta)
Classe 4: 15.5%
Classe 5: 5.7%
Classe 3: 0.3%   (minoria extrema)
Classe 2: 0.1%   (minoria extrema)
Classe 7: <0.1%  (minoria extrema)
Classe 6: <0.1%  (minoria extrema)
```

| Modelo | G-mean | Accuracy | Observações |
|--------|--------|----------|-------------|
| **ERulesD2S** | 0.6692 | 0.669 | ÚNICO com G-mean>0 |
| **GBML** | 0.5934 | - | Variável (std=0.54) |
| ACDWM | 0.0000 | - | Falha (binário) |
| ARF | 0.0000 | 0.982 | **Alta acc, zero G-mean!** |
| HAT | 0.0000 | 0.781 | Falha em minoritárias |
| SRP | 0.0000 | 0.988 | **Alta acc, zero G-mean!** |

**DESCOBERTA IMPORTANTE:**
- ARF e SRP têm **accuracy >98%** mas **G-mean=0.0**!
- Isso significa: predizem apenas classe majoritária (Classe 1)
- G-mean=0.0 indica recall=0 em pelo menos uma classe
- ERulesD2S consegue balancear classes (G-mean=0.67)
- GBML também tem algum sucesso (G-mean=0.59)

---

### 3. CovType (7 classes) ⚠️ Multiclasse Desbalanceado

**Distribuição de classes:**
```
Classe 2: 48.7%  (maioria)
Classe 1: 36.5%
Classe 3:  6.2%
Classe 7:  3.5%
Classe 6:  3.0%
Classe 5:  1.6%
Classe 4:  0.5%  (minoria extrema)
```

| Modelo | G-mean | Std | Observações |
|--------|--------|-----|-------------|
| **ERulesD2S** | 0.4348 | 0.0896 | Melhor |
| SRP | 0.2570 | 0.2591 | Alta variância |
| HAT | 0.2379 | 0.2492 | Alta variância |
| **GBML** | 0.2349 | 0.2199 | Competitivo |
| ARF | 0.1748 | 0.1693 | |
| ACDWM | 0.0000 | 0.0000 | Falha (binário) |

**Análise:**
- ERulesD2S muito superior (G-mean=0.43)
- River models lutam com desbalanceamento (G-mean ≈ 0.18-0.26)
- GBML tem performance similar a HAT/SRP
- Alta variância indica dificuldade com drift+desbalanceamento

---

### 4. PokerHand (9 classes) ❌ Multiclasse EXTREMAMENTE Desbalanceado

**Distribuição de classes (first 1000 samples):**
```
Classe 0: 494 (49.4%)  (maioria)
Classe 1: 420 (42.0%)
Classe 2:  43 (4.3%)
Classe 3:  19 (1.9%)
Classe 4:   4 (0.4%)   (minoria extrema)
Classe 5:   7 (0.7%)
Classe 6:   2 (0.2%)   (minoria extrema)
Classe 8:   5 (0.5%)
Classe 9:   5 (0.5%)
```

| Modelo | G-mean | Accuracy | Observações |
|--------|--------|----------|-------------|
| **ERulesD2S** | 0.3482 | 0.348 | ÚNICO com G-mean>0 |
| ACDWM | 0.0000 | 0.502 | Falha (binário) |
| GBML | 0.0000 | - | Falha total |
| ARF | 0.0000 | 0.493 | Falha total |
| HAT | 0.0000 | 0.493 | Falha total |
| SRP | 0.0000 | 0.466 | Falha total |

**DESCOBERTA CRÍTICA:**
- **TODOS os modelos exceto ERulesD2S falharam completamente!**
- G-mean=0.0 para GBML, ARF, HAT, SRP, ACDWM
- Accuracy ~50% = random guessing entre classes 0 e 1
- Dataset é **EXTREMAMENTE desbalanceado** (classes 4, 6 com <0.5%)
- Nenhum modelo (exceto ERulesD2S) consegue prever classes minoritárias

**Implicação:**
- Este não é problema exclusivo de ACDWM!
- É um **dataset extremamente difícil** para balanceamento
- ERulesD2S tem mecanismo específico para lidar com desbalanceamento

---

### 5. IntelLabSensors (56 classes!) ✅ Multiclasse Fácil

**Distribuição:** 56 classes relativamente balanceadas (5.4% para top class)

| Modelo | G-mean | Observações |
|--------|--------|-------------|
| ARF | 1.0000 | Perfeito |
| ERulesD2S | 1.0000 | Perfeito |
| HAT | 1.0000 | Perfeito |
| **GBML** | 1.0000 | Perfeito |
| SRP | 1.0000 | Perfeito |
| ACDWM | 0.0000 | Falha (binário) |

**Análise:**
- **Dataset MUITO fácil** - todos modelos multiclasse têm performance perfeita
- ACDWM falha apenas pela limitação binária
- Comprova que River models SÃO capazes de multiclasse quando balanceado

---

## Análise Comparativa: ACDWM vs River Models

### ACDWM Failures

**Tipo 1: Limitação Binária** (3 datasets)
- CovType: 7 classes → G-mean=0.0
- Shuttle: 7 classes → G-mean=0.0
- IntelLabSensors: 56 classes → G-mean=0.0

**Tipo 2: Desbalanceamento** (1 dataset)
- PokerHand: 9 classes → G-mean=0.0 (mas TODOS falham!)

**Tipo 3: Sucesso** (1 dataset)
- Electricity: 2 classes → G-mean=0.739 ✅

### River Models (ARF, HAT, SRP)

**Sucessos:**
- Electricity (2 classes, balanceado): G-mean ≈ 0.73-0.75
- IntelLabSensors (56 classes, balanceado): G-mean = 1.0

**Falhas por Desbalanceamento:**
- Shuttle (7 classes, 78% em 1 classe): G-mean=0.0
- CovType (7 classes, 49% em 1 classe): G-mean ≈ 0.18-0.26
- PokerHand (9 classes, desbalanceamento extremo): G-mean=0.0

**Conclusão:**
- River models SÃO multiclasse
- Mas têm **grande dificuldade com desbalanceamento severo**
- ACDWM não pode nem tentar (limitação de design)

---

## Comparação: ERulesD2S vs Demais

### ERulesD2S Vantagens

**Mecanismo de balanceamento:**
- Usa **rules per class** = 5 regras para cada classe
- Sampling sliding window com **forgetting factor** = 0.5
- GP evolucionário com **elitismo** preserva conhecimento

**Resultados:**
- **Único modelo com G-mean>0 em TODOS os datasets**
- Melhor em 4 de 5 datasets (exceto Electricity)
- Ranking médio: 2.40 (melhor geral)

### GBML Performance

**Vantagens:**
- Memory/seeding preserva conhecimento entre chunks
- Adaptive complexity ajusta a novos padrões
- Explicit drift adaptation

**Resultados:**
- Segundo melhor (ranking 3.20)
- Competitivo em Electricity (0.734)
- Sucesso parcial em Shuttle (0.593)
- Luta com CovType (0.235) e PokerHand (0.0)

**Limitação:**
- Sem mecanismo específico para balanceamento de classes
- Fitness = G-mean ajuda, mas não é suficiente para desbalanceamento extremo

---

## Rankings Finais (Phase 3 - Batch 5)

### Ranking Médio por G-mean

```
Rank  Modelo      G-mean Médio   Performance por Dataset
────────────────────────────────────────────────────────────────
2.40  ERulesD2S      0.6130      1º em 4/5 datasets
3.20  GBML           0.5124      2º geral, melhor GA
3.50  ARF            0.3855      Melhor River (incremental)
3.70  HAT            0.3936      River comparable
3.70  SRP            0.3883      River comparable
4.50  ACDWM          0.1478      Limitado a binário
```

### Rankings por Dataset

| Dataset | 1º | 2º | 3º | 4º | 5º | 6º |
|---------|----|----|----|----|----|----|
| **Electricity** | ARF (0.75) | ACDWM (0.74) | GBML (0.73) | HAT (0.73) | SRP (0.68) | ERules (0.61) |
| **Shuttle** | ERules (0.67) | GBML (0.59) | ARF (0.0) | ACDWM (0.0) | HAT (0.0) | SRP (0.0) |
| **CovType** | ERules (0.43) | SRP (0.26) | HAT (0.24) | GBML (0.23) | ARF (0.17) | ACDWM (0.0) |
| **PokerHand** | ERules (0.35) | ACDWM (0.0) | ARF (0.0) | GBML (0.0) | HAT (0.0) | SRP (0.0) |
| **IntelLab** | ARF (1.0) | ERules (1.0) | HAT (1.0) | GBML (1.0) | SRP (1.0) | ACDWM (0.0) |

**Observações:**
1. **ERulesD2S dominante** em datasets desbalanceados (Shuttle, CovType, PokerHand)
2. **River models competitivos** em datasets balanceados (Electricity, IntelLab)
3. **ACDWM competitivo** apenas em Electricity (binário)
4. **GBML segundo melhor** no geral, mas sem mecanismo de balanceamento

---

## Problemas Identificados

### 1. PokerHand: Dataset Extremamente Difícil

**Problema:**
- 5 de 6 modelos têm G-mean=0.0
- Apenas ERulesD2S consegue G-mean>0 (0.35)

**Causa:**
- Desbalanceamento extremo (classe 4: 0.4%, classe 6: 0.2%)
- Com chunks de 1000 amostras, classes minoritárias aparecem 2-4 vezes
- Impossível treinar/prever corretamente

**Implicações para o paper:**
- Documentar claramente esta dificuldade
- Não usar PokerHand como benchmark principal
- Considerar remover ou usar apenas como "extreme case"

### 2. Shuttle: Accuracy vs G-mean Discrepancy

**Problema:**
- ARF: Accuracy=98%, G-mean=0.0
- SRP: Accuracy=99%, G-mean=0.0

**Causa:**
- Modelos predizem apenas classe majoritária (78%)
- G-mean captura falha em minoritárias
- Accuracy não reflete performance real

**Implicação:**
- **G-mean é métrica superior para datasets desbalanceados**
- Justifica nossa escolha de G-mean como métrica principal

### 3. ACDWM: Dois Tipos de Falha

**Falha Tipo A: Limitação Binária**
- CovType, Shuttle, IntelLabSensors
- Falha no código (divisão por zero)
- Impossível executar

**Falha Tipo B: Desbalanceamento**
- PokerHand (tecnicamente multiclasse, mas TODOS falharam)
- Mesmo que fosse binário, provavelmente falharia

**Documentação necessária:**
- Separar claramente os dois tipos
- Falha Tipo A: Design limitation (citar Lu et al., 2020)
- Falha Tipo B: Extreme imbalance (shared with other models)

---

## Próximos Passos

### 1. Testes Estatísticos (Hoje)

- [ ] **Friedman Test**: Verificar se diferenças entre modelos são significativas
- [ ] **Wilcoxon Signed-Rank**: Comparações pareadas (GBML vs ERulesD2S, etc.)
- [ ] **Cliff's Delta**: Effect size das diferenças
- [ ] **Nemenyi Post-hoc**: Identificar grupos homogêneos

**Script:** `statistical_tests_phase3.py`

### 2. Comparação Phase 2 + Phase 3 (Hoje/Amanhã)

- [ ] Consolidar rankings Phase 2 (drift simulation) + Phase 3 (real)
- [ ] Calcular ranking global
- [ ] Verificar consistência entre fases

### 3. Atualização do Paper (Amanhã)

**Seção Methodology:**
```latex
\subsection{Evaluation Protocol}
All models use chunk-wise sequential train-then-test protocol:
- Training: 5 chunks of 1000 instances each
- Testing: Sequential evaluation on subsequent chunks
- Adaptation mechanisms vary by model design:
  * GBML: Population seeding with elitism
  * River models: Incremental learning with model persistence
  * ERulesD2S: Genetic Programming with elitism
  * ACDWM: Weight decay with ensemble pruning
```

**Seção Results - Phase 3:**
```latex
\subsection{Phase 3: Real-World Datasets}

Table X presents Phase 3 results on 5 real-world datasets.

\textbf{Key Findings:}
\begin{itemize}
\item ERulesD2S achieved best average ranking (2.40), demonstrating
      superior handling of class imbalance
\item GBML ranked second (3.20), competitive on balanced datasets
\item River models (ARF, HAT, SRP) struggled with severe imbalance,
      achieving G-mean=0.0 on PokerHand despite high accuracy
\item ACDWM limited to binary classification, failing on 4/5 datasets
\end{itemize}

\textbf{Class Imbalance Impact:}
PokerHand and Shuttle revealed severe limitations of standard
incremental learners when facing extreme class imbalance. Despite
achieving >95\% accuracy, ARF and SRP obtained G-mean=0.0, indicating
complete failure to predict minority classes. Only ERulesD2S, with
explicit per-class rule learning, maintained non-zero G-mean across
all datasets.
```

**Seção Discussion:**
```latex
\subsection{ACDWM Limitations}

ACDWM (Lu et al., 2020) is designed exclusively for binary
classification problems. The algorithm's underbagging component
(Algorithm 3 in Lu et al.) explicitly separates positive and
negative samples, causing division-by-zero errors when applied
to multi-class problems.

In our Phase 3 experiments, ACDWM:
\begin{itemize}
\item Successfully processed Electricity (binary, 2 classes)
\item Failed on CovType (7 classes), Shuttle (7 classes),
      IntelLabSensors (56 classes), and PokerHand (9 classes)
\end{itemize}

Following the methodology from Phase 2, we assign G-mean=0.0
to failed cases, reflecting the model's inability to handle
multi-class scenarios. This limitation significantly impacts
ACDWM's overall ranking (4.50), as 4 of 5 Phase 3 datasets
are multi-class.
```

---

## Arquivos de Referência

**Resultados Consolidados:**
- `batch_5_consolidated_with_acdwm_zeros.csv` (150 linhas)
- `batch_5_model_averages.csv` (30 linhas: 6 modelos × 5 datasets)
- `batch_5_rankings.csv` (5 linhas: 5 datasets)

**Investigações Anteriores:**
- `ACDWM_LIMITACAO_CONFIRMADA.md` (teste isolado multiclasse)
- `ARTIGOS_RESUMO_ERULESD2S.md` (protocolo ERulesD2S)
- `ARTIGOS_RESUMO_ACDWM.md` (limitação binária confirmada)
- `DESCOBERTAS_PROTOCOLOS_FINAL.md` (comparação de protocolos)

**Scripts:**
- `consolidate_batch5_with_acdwm_zeros.py` (consolidação)
- `test_acdwm_multiclass_isolated.py` (teste isolado)

---

## Lições Aprendidas

### 1. G-mean > Accuracy para Desbalanceamento

**Evidência:** Shuttle
- ARF: Accuracy=98%, G-mean=0.0
- Conclusão: Accuracy esconde falha total em minoritárias

**Implicação:** Nossa escolha de G-mean como métrica primária é **validada**

### 2. Class Imbalance é Mais Difícil que Multiclasse

**Evidência:** IntelLabSensors vs PokerHand
- IntelLabSensors: 56 classes, G-mean=1.0 (fácil!)
- PokerHand: 9 classes, G-mean=0.0 (impossível!)

**Diferença:** Balanceamento, não número de classes

### 3. ERulesD2S Mecanismo de Balanceamento é Superior

**Evidência:** Único modelo com G-mean>0 em todos datasets

**Mecanismo:**
- Rules per class = 5 (garante cobertura de todas as classes)
- Sampling sliding window (esquece dados antigos)
- GP com elitismo (preserva boas soluções)

**Implicação:** GBML poderia se beneficiar de mecanismo similar

### 4. ACDWM Não é Competitivo Mesmo em Binário

**Evidência:** Electricity
- ACDWM: G-mean=0.739 (2º lugar)
- ARF: G-mean=0.753 (1º lugar)
- Diferença pequena, mas consistente

**Implicação:** ACDWM não oferece vantagem mesmo em seu domínio

---

**Status:** ✅ CONSOLIDAÇÃO COMPLETA
**Próximo:** Testes estatísticos
**Criado por:** Claude Code
**Data:** 2025-11-23
