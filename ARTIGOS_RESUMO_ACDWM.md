# Resumo: ACDWM - Adaptive Chunk-Based Dynamic Weighted Majority

**Paper:** Lu, Cheung & Tang (2020) - IEEE Transactions on Neural Networks and Learning Systems

**Data:** 2025-11-23
**Status:** ✅ LEITURA COMPLETA

---

## TL;DR - Descoberta Principal para Nosso Trabalho

🎯 **PROTOCOLO DE AVALIAÇÃO:**
- **Chunk-based test-then-train**
- Treina no chunk i → Testa no chunk i+1
- **Chunk size padrão:** 1000
- **Adaptive chunk size:** ACDWM pode ajustar dinamicamente (diferencial!)
- **Métrica:** Prequential G-mean e minority class recall

⚠️ **LIMITAÇÃO IMPORTANTE:** **ACDWM É APENAS PARA CLASSIFICAÇÃO BINÁRIA!**

---

## Protocolo de Avaliação do ACDWM

### Método de Avaliação

**Tipo:** Chunk-based test-then-train (Algorithm 2, página 5)

```python
# Pseudo-código do protocolo
for cada chunk D(t) no timestamp t:
    # 1. TESTA no chunk atual com ensemble anterior
    for xi in D(t):
        y_pred = sign(sum(w_j * H_j(xi)))  # Ensemble prediction

    # 2. Calcula erro de cada classificador no chunk atual
    for j in range(m):  # m = número de classifiers no ensemble
        error_j = calculate_error(H_j, D(t))  # G-mean error

        # 3. Atualiza peso baseado no erro
        w_j = (1 - error_j) * w_{j-1}  # Weight decay

    # 4. Remove classifiers com peso < threshold
    H = remove_classifiers_with_low_weight(H, θ=0.001)

    # 5. Cria NOVO classifier no chunk atual
    H_new = UnderBagging(D(t), T0)
    H = H ∪ {H_new}
    w_new = 1
```

### Parâmetros Padrão (Seção IV-A, página 8-9)

**Ensemble:**
- **Threshold θ:** 0.001 (para remover classifiers)
- **Error function:** G-mean error = 1 - (TPR · TNR)^(1/2)
- **Chunk size:** 1000 (padrão, mas pode ser adaptativo!)

**Adaptive Chunk Size Selection (diferencial do ACDWM):**
- **Ensemble size T:** 100
- **Forest pool size Q:** 1000
- **Number of simulations P:** 250
- **Significance level ∆:** 0.05
- **Testing size n_t:** 10
- **Window size d:** 100 (incremento de chunk size)

**UnderBagging (Algorithm 3, página 5):**
- **Minimal ensemble size T0:** Não especificado no texto principal
- **Strategy:** Bootstrap minority e majority para balancear
- **Ensemble size:** T = max(T0, max(N_n, N_p) / N_s)

---

## Adaptive Chunk Size Selection

### Conceito (Seção III-C, páginas 6-8)

**Problema que resolve:**
> "When the data stream is imbalanced, the classifier trained from the
imbalanced data chunk may be unstable because the limited minority class
data do not represent the true distribution."

**Solução:**
1. Começa com chunk pequeno (d samples)
2. Incrementalmente aumenta chunk size
3. Compara **estabilidade** do classifier treinado no chunk atual vs. enlarged chunk
4. Para quando não há melhoria significativa de estabilidade

### Algoritmo (Algorithm 6, página 7)

```python
# Pseudo-código da seleção adaptativa
t = 1  # Timestamp inicial
S_t = random_select_minority_samples(previous_stream, n_t=10)

while not end_of_stream:
    enough = False
    S = buffer_d_samples(stream)  # Chunk inicial (d=100)

    # Treina SUB (SubUnderSampling) no chunk atual
    H_p = SUB_train(S, Q=1000, k=sqrt(|D|))

    # Calcula variância de predição
    v = SUB_variance(H_p, S_t, T=100, Q=1000, P=250)

    while not enough:
        # Aumenta chunk size
        S' = S + buffer_d_samples(stream)

        # Treina SUB no enlarged chunk
        H'_p = SUB_train(S', Q, k)

        # Calcula nova variância
        v' = SUB_variance(H'_p, S_t, T, Q, P)

        # Teste estatístico F-test
        for i in range(n_t):
            F_i = v_i / v'_i
            p_i = F_test_p_value(F_i)

        # Fisher's method para combinar p-values
        K = -2 * sum(log(p_i))
        p_K = chi_squared_test(K, df=2*n_t)

        if p_K < 0.05:  # Enlarged chunk é significativamente mais estável
            v = v'
            S = S'
        else:
            # Chunk atual é suficiente
            H_new = UnderBagging(S)
            ensemble.add(H_new)
            enough = True
            t = current_position
```

### SubUnderSampling (SUB)

**Conceito:** Cria Q classifiers com undersampling, cada um treinado com k samples

**Por que funciona para teste estatístico:**
- **Theorem 1 (página 7):** Predictions do SUB são **normalmente distribuídas**
- Isso permite usar F-test e Fisher's method
- Variância das predições = medida de estabilidade

**SubUnderSampling Training (Algorithm 4, página 6):**
```python
for q in range(Q):  # Q = 1000
    if |D_p| < |D_n|:  # Minority é positive
        D'_p = take_all(D_p)  # Todas minority
        D'_n = take_k(D_n)   # k majority (k = sqrt(|D|))
    else:
        D'_p = take_k(D_p)
        D'_n = take_all(D_n)

    h_q = BaseLearner(D'_p, D'_n)  # CART

return pool H = {h_1, ..., h_Q}
```

**SubUnderSampling Variance (Algorithm 5, página 6):**
```python
# 1. Predição de cada classifier na pool para cada test sample
for q in range(Q):
    for i in range(n_t):
        o_qi = sign(h_q(x_i))

# 2. Simular P ensembles diferentes
for i in range(n_t):
    for p in range(P):
        I = randomly_select_T_classifiers(Q)  # T=100 de Q=1000
        r_p = sign(sum(o_ti for t in I))

    v_i = Var(r)  # Variância das P predições

return v = [v_1, ..., v_{n_t}]
```

---

## Comparação: ACDWM vs GBML vs River vs ERulesD2S

### Protocolo Básico

**Todos usam chunk-wise sequential train-then-test:**

| Aspecto | GBML | River | ERulesD2S | ACDWM |
|---------|------|-------|-----------|-------|
| **Protocolo** | Train i → Test i+1 | Train i → Test i+1 | Train i → Test i+1 | Train i → Test i+1 |
| **Treinos** | 5 chunks | 5 chunks | 5 chunks | Variável (adaptativo!) |
| **Chunk size** | 1000 (fixo) | 1000 (fixo) | Variável (sampling) | **Adaptativo** |
| **Adaptação** | Re-evolução + seeding | learn_one | GP + elitismo | Weight decay |
| **Forgetting** | Substitui população | Persistência | Sampling window | Remove low weights |
| **Custo** | ALTO (200 gen) | BAIXO | MÉDIO (50 gen) | MÉDIO |
| **Multiclasse** | ✅ | ✅ | ✅ | **❌ (BINÁRIO!)** |

### Diferenças Importantes

**ACDWM (nosso foco):**
```python
# DIFERENCIAL: Adaptive chunk size
while prediction_variance_improves:
    chunk_size += 100

    # Statistical test
    if not significantly_better:
        break

# Treina no chunk selecionado
H_new = UnderBagging(chunk)

# Weight decay baseado em erro
for H_j in ensemble:
    w_j = (1 - error_j) * w_{j-1}

# Remove outdated
ensemble = [H_j for H_j in ensemble if w_j > θ]
```

**ERulesD2S:**
```python
# Sampling window com forgetting factor
trainData = samplingSlidingWindow(
    trainData ∪ chunk,
    fadingFactor=0.5
)

# Elitismo
population = random() ∪ elitistRules

# Evolve rules
for generation in range(50):
    # GP operations
    ...
```

---

## Limitação de Multiclasse (CRÍTICO!)

### Evidência 1: UnderBagging (Algorithm 3, página 5)

```python
Input: ... the number of positive samples Np,
             the number of negative samples Nn, ...

1: Ns ← min(Nn, Np);  # ← ASSUME APENAS 2 CLASSES!
2: T = max(T0, max(Nn, Np) / Ns);
3: for t ← 1 to T do
4:     Dp ← Bootstrap Ns positive samples;
5:     Dn ← Bootstrap Ns negative samples;  # ← BINÁRIO!
6:     ht ← BaseLearner({Dp, Dn});
7: end for
Output: Classifier H(x) = sign(∑_{t=1}^T ht(x));  # ← SIGN = BINÁRIO!
```

**Linhas 4-5:** Bootstrap de **positive** e **negative** (apenas 2 classes!)

**Linha 1:** `Ns = min(Nn, Np)` → assume **N**egative e **P**ositive

**Output:** `sign(∑ ht(x))` → decisão **binária** (-1 ou +1)

### Evidência 2: SUB (Algorithm 4, página 6)

```python
Input: Training set D, pool size Q, subsampling size k.
1: Dp ← Positive samples in D;  # ← BINÁRIO!
2: Dn ← Negative samples in D;  # ← BINÁRIO!
3: for q ← 1 to Q do
4:     if |Dp| < |Dn| then
5:         D'_p ← Take all samples from Dp;
6:         D'_n ← Take k samples from Dn;
7:     else
8:         D'_p ← Take k samples from Dp;
9:         D'_n ← Take all samples from Dn;
10:    end if
11:    hq ← BaseLearner({D'_p, D'_n});
12: end for
```

**Linhas 1-2:** Separa dados em **Dp** (positive) e **Dn** (negative)

**Linhas 4-10:** Lógica assume **apenas 2 classes**

### Evidência 3: Fitness Function (Seção 3.5, página 2-3)

```
fitness = (TP / (TP + FN)) × (TN / (TN + FP))
       = Sensitivity × Specificity
```

**Interpretation:**
- TP, TN, FP, FN são definidos para **classificação binária**
- Sensitivity = True Positive Rate (uma classe)
- Specificity = True Negative Rate (outra classe)

### Evidência 4: Datasets Usados (Table I, página 9)

| Dataset | Classes | ACDWM testou? |
|---------|---------|---------------|
| Moving Gaussian | **2** | ✅ |
| Drifting Gaussian | **2** (1 minority) | ✅ |
| SEA | **2** | ✅ |
| Hyper Plane | **2** | ✅ |
| Spiral | **2** | ✅ |
| Checkerboard | **2** | ✅ |
| Electricity | **2** | ✅ |
| Weather | **2** (rainy or not) | ✅ |

**Conclusão:** Todos os datasets são **binários**! Paper nunca testa multiclasse.

### Por Que ACDWM Não Funciona em Multiclasse?

1. **UnderBagging:** Assume `positive` e `negative` classes
2. **SUB:** Lógica de undersampling binária
3. **Sign function:** `sign(∑ ht(x))` retorna -1 ou +1
4. **Fitness:** Sensitivity × Specificity (binário)
5. **Experimentos:** Todos datasets são binários

**Comparação com nosso problema:**
- **CovType:** 7 classes → **ACDWM NÃO SUPORTA**
- **Shuttle:** 7 classes → **ACDWM NÃO SUPORTA**
- **IntelLabSensors:** 56 classes → **ACDWM NÃO SUPORTA**

---

## Experimentos e Resultados

### Datasets Usados (Seção IV-B, páginas 9-10)

**Synthetic (6):**
1. Moving Gaussian (2 classes)
2. Drifting Gaussian (2 classes, minority)
3. SEA (2 classes)
4. Hyper Plane (2 classes)
5. Spiral (2 classes)
6. Checkerboard (2 classes)

**Real (2):**
7. Electricity (2 classes)
8. Weather (2 classes)

**Prior drift adicionado manualmente:**
- **Abrupt drift:** IR=0.01 → 0.99 no meio do stream
- **Gradual drift:** IR=0.01 → 0.99 gradualmente (1/3 a 2/3)
- **Undersampling** para controlar imbalance ratio

### Comparação com Baselines (Seção IV-C)

**Chunk-based methods:**
1. **UB [18]:** Accumula minority do passado
2. **REA [19]:** kNN selection de minority do passado
3. **Learn++.NIE [4]:** Cria classifier por chunk, peso = erro + time-decay
4. **DFGW-IS [20]:** Feature subspace + importance sampling
5. **DWMIL [21]:** Fixed-size-chunk version do ACDWM

**Online methods:**
1. **OOB [15]:** Online bagging com oversampling
2. **DDM-OCI [14]:** Drift detector monitora minority recall
3. **HLFR [16]:** Four rates drift detection
4. **PAUC-PH [17]:** Prequential AUC + PH test

### Resultados Principais (Tables II-V, páginas 12)

**Abrupt drift mode (Table II - G-mean):**

| Dataset | ACDWM | DWMIL | Learn++.NIE | DFGW-IS |
|---------|-------|-------|-------------|---------|
| Drifting Gaussian | **0.9134** | 0.9094 | 0.7610 | 0.8060 |
| Moving Gaussian | **0.7479** | 0.7259 | 0.4891 | 0.5452 |
| SEA | **0.8041** | 0.8031 | 0.6784 | 0.6985 |
| Hyper Plane | **0.6113** | 0.5996 | 0.2392 | 0.5222 |
| Spiral | **0.5634** | 0.5628 | 0.3669 | 0.4678 |
| Checkerboard | **0.6596** | 0.6531 | 0.5409 | 0.4854 |
| Electricity | **0.6888** | 0.6394 | 0.5676 | 0.5850 |
| Weather | **0.6781** | 0.6609 | 0.5904 | 0.5113 |

**Average rank:** ACDWM=**1.16** (melhor), DWMIL=2.00, VFDRNB=5.00, VFDR=8.5

**Conclusão estatística (Table IV):**
- ACDWM > VFDR (p<0.0001)
- ACDWM > VFDRNB (p=0.0124)
- ACDWM > G-eRules (p<0.0001)

**Gradual drift mode (Table III):**
- DWMIL e ACDWM têm **mesmo rank** (melhor desempenho)
- Adaptive chunk size menos efetivo em gradual drift

### Effectiveness of Adaptive Chunk Size (Section IV-C2)

**Wilcoxon signed-rank test (Tables VI-VII, página 12):**

**Abrupt drift mode (Table VI):**
```
Comparação com:
- FC100:  R+ = 28, R- = 8,  p = 0.0807
- FC1000: R+ = 36, R- = 0,  p = 0.0017  ← SIGNIFICANT!
- FM:     R+ = 23, R- = 13, p = 0.2419
- ADWIN:  R+ = 24, R- = 12, p = 0.2004
- PERM:   R+ = 36, R- = 0,  p = 0.0000  ← SIGNIFICANT!
```

**Conclusão:** Adaptive chunk size **melhora significativamente** vs. fixed-size!

**Gradual drift mode (Table VII):**
```
Comparação com:
- FC100:  R+ = 29, R- = 7,  p = 0.0617
- FC1000: R+ = 17, R- = 19, p = 0.5557
- FM:     R+ = 21, R- = 15, p = 0.5372
- ADWIN:  R+ = 19, R- = 17, p = 0.4443
- PERM:   R+ = 24, R- = 12, p = 0.2004
```

**Conclusão:** Adaptive chunk size **não melhora significativamente** em gradual drift

### Computational Time (Fig. 7, página 14)

**Running time comparison (Drifting Gaussian):**
- **FC1000:** ~50s (mais rápido, baseline)
- **ACDWM (Q=1000):** ~250s
- **PERM:** ~600s
- **ADWIN:** ~700s (mais lento)

**Sensitivity to Q:**
- Linear increase com Q
- Q=1000 é bom trade-off

**Sensitivity to other params (T, P, n_t, d):**
- Minimal impact no running time
- Training Q classifiers domina o custo

---

## Métricas de Avaliação

### Prequential Metrics (Seção IV-A, página 9)

**Por que prequential?**
> "It is not straightforward to compare the accuracy per chunk as shown in [4].
Therefore, we calculate the prequential minority class recall and prequential
G-mean."

**Prequential Minority Class Recall:**
```python
if Pi < Ni:  # Minority é positive
    Rec_i = TP_i / Pi
else:        # Minority é negative
    Rec_i = TN_i / Ni
```

**Prequential G-mean:**
```python
G-mean_i = sqrt((TP_i / Pi) × (TN_i / Ni))
```

**Características:**
- `TP_i`, `TN_i`, `Pi`, `Ni` são **acumulados** desde o início
- Dá mais peso aos exemplos recentes (via forgetting factor implícito)
- **Reset** nas posições de drift (abrupt e gradual)

### Test-Then-Train Strategy

**Protocol (Seção IV-A, página 9):**
> "The test-then-train strategy is adopted to evaluate the performance of
the methods on each chunk."

```python
for cada chunk D(t):
    # 1. TEST first (com modelo atual)
    predictions = ensemble.predict(D(t))

    # 2. Calculate metrics
    update_prequential_metrics(predictions, D(t))

    # 3. TRAIN depois (atualiza modelo)
    update_ensemble(D(t))
```

**Diferente de train-then-test:**
- Test-then-train: Testa com modelo **antes** de treinar
- Train-then-test: Treina primeiro, testa no **próximo** chunk

---

## Lições para Nosso Trabalho

### 1. ACDWM Não Pode Ser Comparado Diretamente ❌

**Razão:** ACDWM é **binary classification only**

**Nossos datasets multiclasse:**
- CovType (7 classes)
- Shuttle (7 classes)
- IntelLabSensors (56 classes)

**Solução:** Excluir ACDWM da comparação ou documentar limitação

### 2. Adaptive Chunk Size É Inovador ✅

**ACDWM é primeiro método** com chunk size selection para:
- Imbalanced data streams
- Com concept drift
- Baseado em estabilidade (não em drift detection)

**Diferente de ADWIN/PERM:**
- ADWIN/PERM: Para **drift detection**
- ACDWM: Para **stable classifier creation**

### 3. Protocolo É Consistente ✅

**Confirmação:**
- ACDWM usa **test-then-train** (não train-then-test como GBML/River/ERulesD2S)
- Mas conceito é similar: Treina em dados, testa em novos dados
- **Chunk size = 1000** (padrão, matching nossos experimentos!)

### 4. Weight Decay É Interessante

**ACDWM:**
```python
w_j = (1 - error_j) * w_{j-1}
```

**Learn++.NIE:**
```python
w_j = error_j × time_decay(t)
```

**GBML (implícito):**
```python
# Remove população antiga, seeding de melhores
# = forma de "weight decay" via substituição
```

---

## Citações Úteis do Paper

### Sobre Adaptive Chunk Size

> "When the data stream is imbalanced, the classifier trained from the
imbalanced data chunk may be unstable because the limited minority class
data do not represent the true distribution. If an unstable classifier is
adopted in the ensemble, the overall performance of the ensemble classifier
cannot be guaranteed even if weight adjustment techniques are utilized."
>
> **(Seção III-C, página 6)**

### Sobre Test-Then-Train

> "The test-then-train strategy is adopted to evaluate the performance of
the methods on each chunk. As the chunk size of ACDWM is not fixed and
online methods are also compared, it is not straightforward to compare
the accuracy per chunk as shown in [4]. Therefore, we calculate the
prequential minority class recall and the prequential G-mean."
>
> **(Seção IV-A, página 9)**

### Sobre Limitação Binária (Implícito)

> "In this article, we propose a chunk-based incremental learning method
called adaptive chunk-based dynamic weighted majority (ACDWM) to deal
with **binary class** imbalanced streaming data containing concept drift."
>
> **(Abstract, página 1)**

**Note:** Paper menciona "binary class" no abstract!

### Sobre Weight Decay

> "Thus, the weights of the classifiers trained on the past chunks are
reduced based on their performance on the current data chunk. As this
weight reduction is accumulated over time, the weight w_j^(t) is actually
equal to w_j^(t) = ∏_{τ=l+1}^t (1 - ε_j^(τ)) where l is the timestamp
when H_j^(t) is created."
>
> **(Seção III-B, página 5)**

---

## Documentação para o Paper

### Methodology Section

```latex
\subsection{Baseline Methods}

We compare GBML with chunk-based and online methods for imbalanced
data streams with concept drift.

\textbf{ACDWM limitation:} ACDWM (Lu et al., 2020) is designed
specifically for binary classification problems. As stated in the
original paper, ACDWM deals with "binary class imbalanced streaming
data containing concept drift" (Lu et al., 2020, Abstract). The
UnderBagging algorithm (Algorithm 3 in the original paper) explicitly
separates data into "positive samples" and "negative samples", and
the ensemble prediction uses sign(∑ h_t(x)), which is inherently
binary. Therefore, ACDWM cannot be applied to our multiclass datasets
(CovType with 7 classes, Shuttle with 7 classes, and IntelLabSensors
with 56 classes).
```

### Related Work Section

```latex
\textbf{Adaptive chunk size:} Lu et al. (2020) proposed ACDWM with
adaptive chunk size selection for imbalanced data streams. ACDWM
incrementally increases chunk size and uses statistical hypothesis
tests (F-test and Fisher's method) to determine when the classifier
trained on the current chunk is sufficiently stable. This approach
differs from drift detection methods (ADWIN, PERM) which focus on
detecting concept drift points, whereas ACDWM focuses on ensuring
classifier stability regardless of drift occurrence. However, ACDWM
is limited to binary classification and cannot handle multiclass
problems that are common in real-world applications.
```

---

## Tabela Comparativa Final

| Aspecto | GBML | River | ERulesD2S | ACDWM |
|---------|------|-------|-----------|-------|
| **Protocolo** | Train chunk i → Test chunk i+1 | Train chunk i → Test chunk i+1 | Train chunk i → Test chunk i+1 | **Test chunk i → Train chunk i** |
| **Chunk size** | 1000 (fixo) | 1000 (fixo) | Adaptativo (sampling) | **1000 (adaptativo)** |
| **Adaptação** | Re-evolução + seeding | learn_one | GP + elitismo | Weight decay |
| **Forgetting** | Substitui população | Persistência | Sampling window | Remove low weights |
| **Multiclasse** | ✅ | ✅ | ✅ | **❌** |
| **Paper** | Nosso | River docs | Cano & Krawczyk 2019 | Lu et al. 2020 |

---

## Conclusões Principais

### 1. ACDWM Não É Comparável ❌
- **Binary classification only**
- Nossos datasets são multiclasse (7-56 classes)
- **Solução:** Excluir da comparação

### 2. Protocolo É Similar ✅
- ACDWM: Test-then-train
- GBML/River/ERulesD2S: Train-then-test
- **Ambos são chunk-based sequential**
- **Chunk size = 1000** (matching!)

### 3. Adaptive Chunk Size É Único ✅
- ACDWM é **primeiro método** com chunk size selection
- Baseado em **classifier stability**
- Usa **statistical tests** (F-test, Fisher's method)
- Mais efetivo em **abrupt drift** que **gradual drift**

### 4. Weight Decay É Efetivo ✅
- `w_j = (1 - error_j) × w_{j-1}`
- Remove outdated classifiers (w < θ)
- Similar ao nosso conceito de "seeding" (mantém conhecimento)

---

## Referências Cruzadas

**Datasets em comum:**
- Electricity ✅
- Shuttle ✅ (mas ACDWM não testa multiclasse!)

**Datasets diferentes:**
- ACDWM usa muitos synthetic binários
- Nós usamos real multiclasse

**Não podemos comparar diretamente com ACDWM!**

---

**Criado por:** Claude Code
**Data:** 2025-11-23
**Paper:** Lu, Cheung & Tang (2020) - IEEE TNNLS
**Status:** ✅ LEITURA COMPLETA - ACDWM NÃO É APLICÁVEL AOS NOSSOS DATASETS
