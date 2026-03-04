# Análise Completa: MCMO Paper

**Paper:** Reduced-space Multistream Classification based on Multi-objective Evolutionary Optimization
**Autores:** Botao Jiao, Yinan Guo, Shengxiang Yang, Jiayang Pu, Dunwei Gong
**Publicação:** IEEE TEVC (Transactions on Evolutionary Computation)
**GitHub:** https://github.com/Jesen-BT/MCMO
**Data da Análise:** 2025-11-23

---

## TL;DR - Resumo Executivo

🎯 **O que é MCMO?**
- **M**ultistream **C**lassification based on **M**ulti-**O**bjective Optimization
- Classifica stream **target sem labels** usando múltiplos **source streams com labels**
- Usa seleção de features via NSGA-II + ponderação GMM + ensemble de classificadores

🔑 **Principais Inovações:**
1. **Feature selection** em espaço reduzido (primeira abordagem para multistream)
2. **GMM-based weighting** para corrigir covariate shift
3. **Asynchronous drift adaptation** (2 estratégias diferentes)
4. **Multi-source streams** (não apenas single source)

⚠️ **Diferença Crítica do Nosso Setup:**
- MCMO: N source streams (labeled) → 1 target stream (unlabeled)
- Nosso: 1 stream → classificação incremental com labels disponíveis

---

## 1. Problema Que MCMO Resolve

### Definição do Problema

**Multistream Classification:**
- **N source streams:** $S^1, S^2, ..., S^N$ (todas com labels)
- **1 target stream:** $T$ (sem labels)
- **Mesma feature space:** Todas têm mesmas D features
- **Distribuições diferentes:** $P^{S^i}_t(X) \neq P^{S^j}_t(X) \neq P^T_t(X)$

**Exemplo prático (do paper):**
- **Recomendação:** Alguns usuários reportam preferências (source), outros não (target)
- **Fault detection:** Múltiplos sensores, caro rotular todos → rotular apenas alguns

### Três Desafios Principais

**1. Asynchronous Concept Drift**
```
Drift pode ocorrer em diferentes tempos:
- Source drift: Drift só em source stream
- Target drift: Drift só em target stream
- Source+Target drift: Drift em ambos
```

**2. Covariate Shift**
```
P^{S^n}_t(X) ≠ P^T_t(X)
→ Classificador treinado em source não funciona bem em target
```

**3. Label Scarcity**
```
Apenas source streams têm labels
Target stream: completamente sem labels
```

---

## 2. Solução MCMO - Arquitetura

### Framework Geral

```
Data Streams → Feature Selection → GMM Weighting → Ensemble → Prediction
     ↓              (NSGA-II)         (Target)      (Source)       ↓
  [S¹,S²,S³,T]  →  Reduced Space  →  Sample Weights → Base Clfs → Target Labels
```

### Componente 1: Feature Selection (NSGA-II)

**Multi-objective Optimization:**

```python
Objective 1: Minimize discriminative loss on source streams
f1 = Σ(n=1 to N) Tr(Cw(X^{S^n}) / Cb(X^{S^n}))

Objective 2: Minimize distribution discrepancy
f2 = Σ(n=1 to N) D(X^{S^n}, X^T)
```

**Onde:**
- `Cw`: Within-class scatter matrix (quão próximos samples da mesma classe)
- `Cb`: Between-class scatter matrix (quão distantes classes diferentes)
- `D(X^{S^n}, X^T)`: Mean discrepancy (MMD) entre source n e target

**Fluxo:**
1. Coletar m samples iniciais de cada stream
2. NSGA-II evolui feature subsets (binary encoding)
3. Pareto front com tradeoffs entre f1 e f2
4. Fuzzy decision making seleciona subset final

**Exemplo de encoding:**
```
Original features: [f1, f2, f3, f4, f5, f6, f7, f8]
Individual:        [ 1,  0,  1,  0,  1,  1,  0,  1]
                    ↓
Selected features: [f1, f3, f5, f6, f8]
```

---

### Componente 2: GMM-based Sample Weighting

**Ideia:** Samples de source stream mais próximos da distribuição de target devem ter maior peso

**Algoritmo:**
```python
# 1. Construir GMM no target stream
GMM = fit_gaussian_mixture(X^T, k_components)

# 2. Para cada sample de source stream
for x_t^{S^n} in S^n:
    # Calcular conditional probability para cada Gaussian
    probs = [P(x_t^{S^n} | G_i) for i in range(k)]

    # Weight = max probability
    w_t^{S^n} = max(probs)

# 3. Treinar classificador com weights
classifier.fit(X^{S^n}, y^{S^n}, sample_weight=w^{S^n})
```

**Intuição:**
- GMM captura distribuição de target em k Gaussians
- Source samples com alta prob. em algum Gaussian → relevantes para target
- Source samples com baixa prob. em todos → irrelevantes (peso baixo)

**Parâmetros:**
- `k`: Número de Gaussians (grid search em [1,10])
- `m`: Tamanho de subset inicial (paper usa 200)

---

### Componente 3: Ensemble de Base Classifiers

**Estrutura:**
```python
# Um base classifier por source stream
Base Classifiers:
  f^{S^1} ← trained on weighted S^1
  f^{S^2} ← trained on weighted S^2
  f^{S^3} ← trained on weighted S^3

# Weight de cada classifier = avg sample weight
cw^{S^n} = (1/m) * Σ w_t^{S^n}

# Ensemble prediction
f^E(x) = Σ(n=1 to N) (cw^{S^n} / Σcw^{S^j}) * f^{S^n}(x)
```

**Base learner:** Hoeffding Tree (incremental decision tree)

---

### Componente 4: Drift Adaptation

**Classificação de Drifts:**

**Categoria 1: Source-only Drift**
```
∃i: P^{S^i}_t ≠ P^{S^i}_{t+1}  AND  P^T_t = P^T_{t+1}
→ Target não mudou, apenas source
```

**Ação:**
- Criar novo base classifier para source stream que driftou
- Manter base classifiers históricos em pool P
- Tamanho máximo do pool: Pmax (paper usa 5)

**Categoria 2: Target-inclusive Drift**
```
P^T_t ≠ P^T_{t+1}
→ Target mudou (pode ou não ter source drift junto)
```

**Ação:**
- **Re-executar feature selection** (features podem ter mudado importância)
- **Reconstruir GMM** (distribuição target mudou)
- **Reset todos base classifiers**

**Drift Detection:**

**Source streams** (supervised):
- DDM (Drift Detection Method)
- Monitora erro de classificação

**Target stream** (unsupervised):
- Monitora conditional probability no GMM
- Usa duas janelas: reference (m samples iniciais) e detection (m recentes)

```python
μ_ref = (1/m) * Σ max_i P(x_t^T | G_i)  # Reference window
μ_det = (1/m) * Σ max_i P(x_t^T | G_i)  # Detection window

# Drift detectado se:
if μ_det < μ_ref - z*(σ/√m):  # z=3 (99% confidence)
    trigger_target_drift_adaptation()
```

---

## 3. Protocolo Experimental (Paper)

### Datasets

**Synthetic (4):**
- AGRAWAL (9 features, 2 classes, 20k samples)
- Tree (20 features, 2 classes, 20k samples)
- RBF (10 features, 2 classes, 20k samples)
- Hyp (30 features, 2 classes, 20k samples)

**Real-world (4):**
- Weather (8 features, 2 classes, 18k samples)
- Electricity (8 features, 2 classes, 45k samples)
- Covtype (54 features, 7 classes, 581k samples)
- Sensor (128 features, 6 classes, 13k samples)

**Multistream (2):**
- CNNIBN (124 features, 2 classes, 100k samples)
- BBC (124 features, 2 classes, 100k samples)

### Criação de Multistream (Single → Multi)

**Processo:**
1. Dividir dataset em 4 batches
2. Calcular mean μ e std σ de cada batch
3. Ordenar samples por P(x) = exp(-(x-μ)²/2σ²)
4. Atribuir samples a streams conforme tabela:

```
          Batch 1   Batch 2   Batch 3   Batch 4
Target T:  0-10%    0-30%     0-50%     0-10%
Source S1: 10-40%   30-80%    50-60%    10-20%
Source S2: 40-90%   80-90%    60-70%    20-50%
Source S3: 90-100%  90-100%   70-100%   50-100%
```

**Efeito:** Cria covariate shift entre streams

### Métricas

**Accuracy:**
```python
Accuracy = Σ(c=1 to K) Σ(i=1 to |Tc|) (ŷ_c,i == y_c,i) / Σ(c=1 to K) |Tc|
```

**G-mean:**
```python
G-mean = (∏(c=1 to K) (Σ(i=1 to |Tc|) (ŷ_c,i == y_c,i) / |Tc|))^(1/K)
```

### Baselines Comparados

1. **MulStream:** Merge todas streams e treina um modelo
2. **FUSION:** Ensemble com single source stream
3. **AOMSDA:** Autoencoder-based multistream (deep learning)
4. **Melanie:** Supervised (usa labels de target também)

### Resultados Principais

**Rankings médios (10 datasets):**
```
1. Melanie:  1.5  (supervised, advantage unfair)
2. MCMO:     1.8  ← Melhor unsupervised!
3. AOMSDA:   3.9
4. FUSION:   4.4-4.5 (varia por source)
5. MulStream: 5.8
```

**MCMO vs Melanie:**
- Melanie usa labels de target (supervised)
- MCMO completamente unsupervised em target
- Performance muito próxima!

---

## 4. Ablation Studies (Componentes)

**Variants testados:**

```
MCMOv1:  Sem feature selection
MCMOv2:  Sem sample weighting
MCMOv3:  Sem drift adaptation
MCMOv4:  Feature selection só inicial (não re-executa em drift)
MCMOv5:  Baseline (sem nada)
```

**Resultados (accuracy drop vs MCMO):**
```
Dataset        v1      v2      v3      v4      v5
AGRAWAL      -1.10   -2.08   -7.67   +0.33   -5.79
Electricity  -3.04   -9.08   -8.65   +0.69   -8.21
Covtype      -7.74  -13.10  -13.75   -7.54   -8.76
Sensor       -3.99  -19.53  -17.92   -1.07  -18.17
```

**Conclusões:**
- **Sample weighting** muito importante (v2: até -19.53%)
- **Drift adaptation** crítico (v3: até -17.92%)
- **Feature selection** essencial em high-dim (Covtype: -7.74%)
- **Re-executar FS** em drift ajuda, mas nem sempre necessário (v4)

---

## 5. Comparação: MCMO vs Nosso Setup

### Setup MCMO (Paper)

```
Problem: Multistream classification
  - Input: N source streams (labeled) + 1 target stream (unlabeled)
  - Output: Labels para target stream

Assumptions:
  ✓ Same feature space across all streams
  ✓ Different distributions (covariate shift)
  ✓ Asynchronous concept drift
  ✓ Labels only in source streams

Protocol:
  - Chunk-wise learning (m=200 samples)
  - Prequential evaluation (test-then-train)
  - Base learner: Hoeffding Tree
```

### Nosso Setup (DSL-AG-hybrid)

```
Problem: Single stream classification with drift
  - Input: 1 stream (labeled)
  - Output: Classification accuracy on stream

Assumptions:
  ✓ Labels available (possivelmente com delay)
  ✓ Concept drift occurs
  ✓ Class imbalance may exist

Protocol:
  - Chunk-wise sequential train-then-test
  - 5 training chunks (1000 samples each)
  - Metrics: G-mean (primary), Accuracy, F1
  - Base learner: Varies (GBML: GP, River: HT/ARF/HAT/SRP, etc)
```

### Diferenças Críticas

| Aspecto | MCMO | Nosso Setup |
|---------|------|-------------|
| **Streams** | Multi-source + 1 target | Single stream |
| **Labels** | Source: yes, Target: no | All: yes (or delayed) |
| **Feature selection** | Yes (NSGA-II) | No |
| **Sample weighting** | Yes (GMM-based) | No |
| **Ensemble** | Multiple sources | Varies by model |
| **Drift detection** | DDM (source) + GMM monitoring (target) | Varies by model |
| **Evaluation** | Prequential | Train-then-test |

---

## 6. Possíveis Adaptações para Nosso Setup

### Opção A: Adaptar MCMO para Single Stream

**Ideia:** Criar "artificial multistream" de um single stream

```python
# Dividir stream em múltiplas fontes
Historical data (older chunks) → Source streams
Current data (recent chunk)    → Target stream

Example:
  Chunks 0-2 → Source stream S1
  Chunks 1-3 → Source stream S2
  Chunks 2-4 → Source stream S3
  Chunk 5    → Target stream T (unlabeled temporariamente)
```

**Challenges:**
- Não há verdadeiro covariate shift (mesma stream)
- Pode não fazer sentido semântico
- Labels estão disponíveis (não precisa unsupervised target)

### Opção B: Extrair Componentes Úteis

**Componentes que podemos aproveitar:**

1. **Feature Selection via NSGA-II**
   - Útil se tivermos datasets high-dimensional
   - Pode melhorar GBML e outros modelos
   - **Implementar como preprocessing step**

2. **GMM-based Weighting**
   - Pode ser usado para pesar samples antigos vs recentes
   - Relevante para lidar com concept drift
   - **Integrar em GBML fitness function**

3. **Drift Detection via GMM Monitoring**
   - Unsupervised drift detection
   - Complementa DDM (que é supervised)
   - **Adicionar como opção de drift detector**

### Opção C: Comparação Direta (Criar Multistream Scenarios)

**Criar datasets multistream artificiais:**

```python
# Para cada dataset real (Electricity, CovType, etc):
1. Split em 4 batches
2. Aplicar covariate shift (como no paper MCMO)
3. Criar 3 source streams + 1 target
4. Rodar MCMO vs nossos modelos

Comparison:
  - MCMO: Usa only source labels
  - GBML/River: Pode usar target labels?
    → Testar com e sem
```

---

## 7. Plano de Integração

### Fase 1: Exploração do Código (HOJE)

**Objetivos:**
- [ ] Clonar repositório GitHub
- [ ] Entender estrutura de código
- [ ] Identificar dependências
- [ ] Rodar exemplo básico

**Arquivos esperados:**
```
MCMO/
  ├── MCMO.py              # Classe principal
  ├── feature_selection.py # NSGA-II
  ├── gmm_weighting.py     # GMM
  ├── drift_detection.py   # DDM + GMM monitoring
  ├── datasets/            # Data loaders
  └── experiments/         # Scripts de experimento
```

### Fase 2: Teste Isolado (HOJE/AMANHÃ)

**Objetivo:** Rodar MCMO em 1 dataset para entender comportamento

**Dataset escolhido:** Electricity
- Binário (2 classes)
- Mesma feature space que usamos
- Usado no paper MCMO

**Protocolo:**
```python
# 1. Criar multistream do Electricity
electricity_data = load_electricity()
S1, S2, S3, T = split_into_multistream(electricity_data)

# 2. Rodar MCMO
mcmo = MCMO(
    n_sources=3,
    k_gaussians=7,
    pmax=5,
    m=200,
    nsga_pop=50,
    nsga_gen=50
)

results = mcmo.fit_predict(S1, S2, S3, T)

# 3. Analisar
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"G-mean: {results['gmean']:.4f}")
print(f"Features selected: {results['n_features']}/{D}")
```

### Fase 3: Integração no Pipeline (AMANHÃ)

**Objetivo:** Adicionar MCMO como baseline ao nosso framework

**Adaptação necessária:**
```python
# baseline_mcmo.py
class MCMOEvaluator:
    def __init__(self, multistream_config):
        self.config = multistream_config
        self.mcmo = MCMO(...)

    def evaluate_on_stream(self, stream_data):
        # Converter single stream → multistream
        sources, target = self._create_multistream(stream_data)

        # Avaliar MCMO
        predictions = self.mcmo.fit_predict(sources, target)

        # Calcular métricas (G-mean, accuracy)
        return metrics

    def _create_multistream(self, stream):
        # Implementar estratégia de split
        # Opção 1: Temporal (chunks antigos = source)
        # Opção 2: Sampling-based (como paper)
        ...
```

**Integração em main.py:**
```python
# Adicionar MCMO aos baselines
baseline_models = {
    'GBML': GBMLEvaluator(...),
    'ARF': RiverEvaluator(model='ARF'),
    'HAT': RiverEvaluator(model='HAT'),
    'SRP': RiverEvaluator(model='SRP'),
    'ACDWM': ACDWMEvaluator(...),
    'ERulesD2S': ERulesD2SEvaluator(...),
    'MCMO': MCMOEvaluator(...)  # ← NEW
}
```

### Fase 4: Comparação e Análise (AMANHÃ)

**Experimentos:**
```
1. Comparar MCMO vs baselines em Phase 3 datasets
2. Analisar:
   - Accuracy e G-mean
   - Features selecionados
   - Tempo de execução
   - Comportamento em concept drift
3. Testes estatísticos (Friedman, Wilcoxon)
```

**Questões de pesquisa:**
- MCMO melhora performance vs single-stream models?
- Feature selection do MCMO é efetiva?
- GMM weighting ajuda com class imbalance?
- MCMO adapta bem a concept drift?

---

## 8. Desafios Técnicos Antecipados

### Desafio 1: Multistream vs Single Stream

**Problema:** MCMO espera múltiplas source streams, nós temos single stream

**Soluções possíveis:**
1. **Temporal splitting:** Chunks antigos = sources, chunk recente = target
2. **Bootstrap splitting:** Criar múltiplas "views" do mesmo chunk via sampling
3. **Class-based splitting:** Separar por classes (se balanceado)

**Minha recomendação:** Temporal splitting (mais natural para drift)

### Desafio 2: Labeled vs Unlabeled Target

**Problema:** MCMO assume target sem labels, mas nós temos labels

**Soluções:**
1. **Ignorar labels de target** durante treinamento (fair comparison)
2. **Usar labels** só para avaliação (calcular G-mean)
3. **Experimento duplo:** Com e sem labels de target

**Minha recomendação:** Ignorar labels em treinamento, usar só em avaliação

### Desafio 3: Feature Selection Overhead

**Problema:** NSGA-II pode ser computacionalmente caro

**Paper reporta:**
- Sensor dataset: Mais tempo que Electricity (apesar de menos samples)
- Razão: Mais drifts → mais re-execuções de feature selection

**Monitorar:**
- Tempo de execução por drift detection
- Número de feature selections executadas
- Tradeoff: performance vs tempo

### Desafio 4: Parâmetros do MCMO

**Muitos hyperparameters:**
```python
m:      Tamanho subset inicial (paper: 200)
k:      Gaussians no GMM (paper: grid search 1-10)
Pmax:   Pool size (paper: 5)
Npop:   População NSGA-II (paper: 50)
Mgen:   Gerações NSGA-II (paper: 50)
pc:     Crossover prob (paper: 0.9)
pm:     Mutation prob (paper: 0.2)
z:      Significance level drift (paper: 3)
```

**Estratégia:**
- Usar valores default do paper inicialmente
- Grid search em k (mais importante)
- Se tempo permitir, tuning dos demais

---

## 9. Cronograma Detalhado

### HOJE (4-6 horas)

**09:00-10:00: Setup**
- [X] Ler paper MCMO (COMPLETO)
- [ ] Clonar repositório GitHub
- [ ] Instalar dependências
- [ ] Entender estrutura de código

**10:00-12:00: Teste Isolado**
- [ ] Implementar data loader para Electricity
- [ ] Criar função de split multistream
- [ ] Rodar MCMO em Electricity
- [ ] Debugar e resolver issues

**12:00-13:00: Almoço**

**13:00-15:00: Análise de Resultados**
- [ ] Comparar MCMO vs results anteriores
- [ ] Analisar features selecionados
- [ ] Documentar comportamento
- [ ] Criar visualizações

### AMANHÃ (4-6 horas)

**09:00-11:00: Integração**
- [ ] Criar baseline_mcmo.py
- [ ] Integrar em main.py
- [ ] Testar em 2-3 datasets

**11:00-13:00: Experimentos**
- [ ] Rodar Phase 3 completa com MCMO
- [ ] Consolidar resultados
- [ ] Calcular rankings

**13:00-14:00: Almoço**

**14:00-15:00: Análise Estatística**
- [ ] Friedman test
- [ ] Wilcoxon pairwise
- [ ] Cliff's Delta

**15:00-16:00: Documentação**
- [ ] Atualizar paper draft
- [ ] Criar tabelas de comparação
- [ ] Escrever discussão sobre MCMO

---

## 10. Critérios de Sucesso

### Mínimo Viável (Must Have)

✅ **Código funciona:**
- MCMO executa sem erros em 1 dataset
- Produz accuracy e G-mean

✅ **Comparação básica:**
- MCMO vs GBML/River em Electricity
- Rankings calculados

✅ **Documentação:**
- Análise de resultados
- Identificação de strengths/weaknesses

### Desejável (Should Have)

🎯 **Experimentos completos:**
- MCMO em todos Phase 3 datasets
- Testes estatísticos
- Ablation study (com/sem components)

🎯 **Integração robusta:**
- Adapter genérico single→multi stream
- Handling de edge cases
- Logging e debugging

### Bônus (Nice to Have)

💡 **Adaptações customizadas:**
- MCMO feature selection + GBML
- GMM weighting em outros modelos
- Hybrid approaches

💡 **Análise profunda:**
- Feature importance analysis
- Drift detection comparison
- Computational cost analysis

---

## 11. Referências e Links

**Paper:**
- Jiao et al. (2023). "Reduced-space Multistream Classification based on Multi-objective Evolutionary Optimization". IEEE TEVC.

**GitHub:**
- https://github.com/Jesen-BT/MCMO

**Papers relacionados (citados):**
- FUSION (Haque et al., 2017)
- AOMSDA (Xie et al., 2022)
- DDM (Gama et al., 2004)
- NSGA-II (Deb et al., 2002)

**Nossos papers de referência:**
- ACDWM (Lu et al., 2020)
- ERulesD2S (Cano & Krawczyk, 2019)

---

## 12. Próximos Passos Imediatos

1. **Explorar GitHub:**
   ```bash
   git clone https://github.com/Jesen-BT/MCMO
   cd MCMO
   python --version  # Check compatibility
   pip install -r requirements.txt
   ```

2. **Ler código principal:**
   - Entender API do MCMO
   - Identificar funções key
   - Verificar compatibilidade com nosso setup

3. **Teste hello-world:**
   ```python
   # Rodar exemplo mais simples possível
   from MCMO import MCMO

   # Synthetic data
   X_s1, y_s1 = generate_source(1000, seed=1)
   X_s2, y_s2 = generate_source(1000, seed=2)
   X_t = generate_target(1000)

   mcmo = MCMO()
   predictions = mcmo.fit_predict([X_s1, X_s2], [y_s1, y_s2], X_t)
   ```

4. **Documentar findings:**
   - Issues encontrados
   - Workarounds necessários
   - Performance observada

---

**Status:** ✅ Paper lido e analisado
**Próximo:** Explorar repositório GitHub
**Criado por:** Claude Code
**Data:** 2025-11-23
