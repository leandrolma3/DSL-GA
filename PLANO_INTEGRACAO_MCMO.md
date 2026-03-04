# Plano de Integração MCMO - Sumário Executivo

**Data:** 2025-11-23
**Objetivo:** Integrar MCMO como baseline no pipeline de comparação

---

## 📋 Resumo do MCMO

**O que é:**
- **M**ultistream **C**lassification via **M**ulti-**O**bjective Optimization
- Classifica target stream (sem labels) usando múltiplos source streams (com labels)

**Componentes principais:**
1. **Feature Selection** via NSGA-II (reduz dimensionalidade)
2. **GMM-based Weighting** (corrige covariate shift)
3. **Ensemble Classifiers** (um por source stream)
4. **Drift Adaptation** (2 estratégias: source-only e target-inclusive)

**Performance (paper):**
- Rank médio: 1.8 (segundo melhor, perdendo só para método supervised)
- Supera FUSION, AOMSDA, MulStream

---

## ⚠️ Desafio Principal: Incompatibilidade de Setup

### MCMO (Paper)
```
Input:  3 source streams (labeled) → 1 target stream (unlabeled)
Output: Predictions para target
Método: Unsupervised multistream classification
```

### Nosso Pipeline
```
Input:  1 stream (labeled)
Output: Classification metrics
Método: Single stream with concept drift
```

**Conclusão:** Precisamos **adaptar** MCMO para funcionar com single stream!

---

## 🎯 Estratégia de Integração (Recomendada)

### Opção Escolhida: Temporal Splitting

**Ideia:** Criar "multistream artificial" usando chunks temporais

```python
# Exemplo com 6 chunks
Chunks 0-2 (past)    → Source Stream 1
Chunks 1-3 (mid)     → Source Stream 2
Chunks 2-4 (recent)  → Source Stream 3
Chunk 5 (current)    → Target Stream (sem usar labels em treino)

# Avaliar MCMO predicting Chunk 5
# Comparar com ground truth para calcular G-mean
```

**Vantagens:**
- ✅ Mantém ordem temporal natural
- ✅ Simula covariate shift real (distribuição muda ao longo do tempo)
- ✅ Faz sentido semântico com concept drift

**Desvantagens:**
- ❌ Precisa adaptar código MCMO
- ❌ Pode ter overhead computacional (feature selection por drift)

---

## 📊 Comparação: MCMO vs Outros Baselines

| Modelo | Setup Original | Labels Target | Feature Selection | Sample Weighting |
|--------|----------------|---------------|-------------------|------------------|
| **GBML** | Single stream | Yes | No | No (fitness-based) |
| **River (ARF/HAT/SRP)** | Single stream | Yes | No | No |
| **ACDWM** | Single stream | Yes | No | Yes (underbagging) |
| **ERulesD2S** | Single stream | Yes | No | Yes (per-class rules) |
| **MCMO** | **Multi-stream** | **No** | **Yes (NSGA-II)** | **Yes (GMM)** |

**Diferencial do MCMO:**
1. **Única abordagem com feature selection explícito**
2. **Único método truly unsupervised no target**
3. **GMM-based weighting** (diferente de undersampling do ACDWM)

---

## 🛠️ Implementação: Passo a Passo

### Passo 1: Clonar e Explorar Repositório

```bash
# No terminal
cd "C:\Users\Leandro Almeida\Downloads"
git clone https://github.com/Jesen-BT/MCMO
cd MCMO

# Explorar estrutura
ls -la
cat README.md  # Se existir
python --version  # Verificar compatibilidade
```

**Arquivos esperados:**
- `MCMO.py` ou `mcmo/` - Código principal
- `requirements.txt` - Dependências
- `datasets/` - Data loaders
- `experiments/` - Scripts de exemplo

### Passo 2: Entender API do MCMO

**Objetivo:** Descobrir como usar MCMO

```python
# Provável API (baseado no paper)
from MCMO import MCMO

# Criar modelo
mcmo = MCMO(
    n_sources=3,           # Número de source streams
    k_gaussians=7,         # Componentes GMM
    pool_size=5,           # Max historical classifiers
    m_samples=200,         # Initial subset size
    nsga_pop=50,           # NSGA-II population
    nsga_gen=50            # NSGA-II generations
)

# Treinar e predizer
predictions = mcmo.fit_predict(
    source_data=[S1, S2, S3],  # List of (X, y) tuples
    source_labels=[y1, y2, y3],
    target_data=T               # Unlabeled X only
)
```

### Passo 3: Criar Adapter para Single Stream

**Arquivo:** `baseline_mcmo.py`

```python
"""
Adapter MCMO para single stream classification.

Converte single stream → multistream via temporal splitting.
"""
import numpy as np
from MCMO import MCMO  # Assumindo que existe

class MCMOEvaluator:
    """
    Wrapper para MCMO que adapta single stream → multistream.
    """
    def __init__(self, n_sources=3, window_size=1000, **mcmo_params):
        """
        Args:
            n_sources: Número de source streams a criar
            window_size: Tamanho de cada chunk/window
            **mcmo_params: Parâmetros para MCMO (k_gaussians, etc.)
        """
        self.n_sources = n_sources
        self.window_size = window_size
        self.mcmo = MCMO(n_sources=n_sources, **mcmo_params)

        # Buffers para manter chunks
        self.chunk_buffer = []
        self.current_chunk = 0

    def _create_multistream(self, stream_chunks):
        """
        Converte lista de chunks → multistream.

        Args:
            stream_chunks: List of (X, y) tuples

        Returns:
            sources: List of N source (X, y) tuples
            target: Target X (sem y)
        """
        if len(stream_chunks) < self.n_sources + 1:
            raise ValueError(f"Need at least {self.n_sources + 1} chunks")

        # Temporal splitting
        # Example: chunks [0,1,2] → S1, chunks [1,2,3] → S2, ..., chunk [N] → T
        sources = []
        for i in range(self.n_sources):
            # Merge consecutive chunks for each source
            source_chunks = stream_chunks[i:i+self.n_sources]
            X_source = np.vstack([chunk[0] for chunk in source_chunks])
            y_source = np.hstack([chunk[1] for chunk in source_chunks])
            sources.append((X_source, y_source))

        # Target = last chunk (only X, ignore y for training)
        target_X, target_y_true = stream_chunks[-1]

        return sources, target_X, target_y_true

    def train_on_chunk(self, X, y):
        """
        Acumula chunk para posterior multistream conversion.
        """
        self.chunk_buffer.append((X, y))
        self.current_chunk += 1

        # Não treina até ter chunks suficientes
        if len(self.chunk_buffer) < self.n_sources + 1:
            return {'status': 'buffering', 'chunk': self.current_chunk}

        return {'status': 'ready', 'chunk': self.current_chunk}

    def test_on_chunk(self, X, y_true):
        """
        Testa MCMO no chunk atual.
        """
        if len(self.chunk_buffer) < self.n_sources + 1:
            # Ainda bufferizando, retornar dummy metrics
            return {
                'gmean': 0.0,
                'accuracy': 0.0,
                'f1': 0.0,
                'status': 'buffering'
            }

        # Criar multistream
        sources, target_X, target_y = self._create_multistream(self.chunk_buffer)

        # Treinar MCMO (usando sources) e predizer target
        try:
            predictions = self.mcmo.fit_predict(sources, target_X)

            # Calcular métricas
            from sklearn.metrics import accuracy_score, f1_score
            from scipy.stats.mstats import gmean

            accuracy = accuracy_score(target_y, predictions)
            f1 = f1_score(target_y, predictions, average='weighted')

            # G-mean
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(target_y, predictions)
            recalls = cm.diagonal() / cm.sum(axis=1)
            gmean_val = gmean(recalls) if all(recalls > 0) else 0.0

            # Slide window (remove oldest chunk)
            self.chunk_buffer.pop(0)

            return {
                'gmean': gmean_val,
                'accuracy': accuracy,
                'f1': f1,
                'n_features_selected': self.mcmo.n_features_selected_,
                'status': 'ok'
            }

        except Exception as e:
            print(f"MCMO Error: {e}")
            return {
                'gmean': 0.0,
                'accuracy': 0.0,
                'f1': 0.0,
                'status': f'error: {e}'
            }
```

### Passo 4: Integrar em main.py

```python
# Em main.py, adicionar MCMO aos baselines

from baseline_mcmo import MCMOEvaluator

# Configurar baselines
baseline_models = {
    'GBML': GBMLEvaluator(...),
    'ARF': RiverEvaluator(model='ARF'),
    'HAT': RiverEvaluator(model='HAT'),
    'SRP': RiverEvaluator(model='SRP'),
    'ACDWM': ACDWMEvaluator(...),
    'ERulesD2S': ERulesD2SEvaluator(...),
    'MCMO': MCMOEvaluator(
        n_sources=3,
        window_size=1000,
        k_gaussians=7,
        pool_size=5,
        m_samples=200
    )
}

# Loop de avaliação (mesmo código)
for chunk_idx in range(num_chunks):
    X_train, y_train = get_chunk(chunk_idx)

    for model_name, evaluator in baseline_models.items():
        # Train
        evaluator.train_on_chunk(X_train, y_train)

        # Test (MCMO vai usar buffer interno)
        X_test, y_test = get_chunk(chunk_idx + 1)
        metrics = evaluator.test_on_chunk(X_test, y_test)

        # Log
        results[model_name][chunk_idx] = metrics
```

### Passo 5: Teste Isolado

**Dataset:** Electricity (mesmo do paper MCMO)

```python
# Script: test_mcmo_isolated.py

from baseline_mcmo import MCMOEvaluator
import pandas as pd

# Load Electricity
df = pd.read_csv('datasets/processed/electricity_processed.csv')

# Create chunks
chunk_size = 1000
chunks = []
for i in range(0, len(df), chunk_size):
    chunk_df = df.iloc[i:i+chunk_size]
    X = chunk_df.drop('class', axis=1).values
    y = chunk_df['class'].values
    chunks.append((X, y))

# Test MCMO
evaluator = MCMOEvaluator(n_sources=3)

# Buffer initial chunks
for i in range(3):
    evaluator.train_on_chunk(*chunks[i])

# Test on subsequent chunks
for i in range(3, min(10, len(chunks))):
    evaluator.train_on_chunk(*chunks[i])
    metrics = evaluator.test_on_chunk(*chunks[i+1])

    print(f"Chunk {i+1}: G-mean={metrics['gmean']:.4f}, "
          f"Acc={metrics['accuracy']:.4f}, "
          f"Features={metrics.get('n_features_selected', '?')}")
```

---

## 📈 Experimentos Planejados

### Experimento 1: Teste Isolado (HOJE)

**Objetivo:** Verificar que MCMO funciona

**Dataset:** Electricity (binário, 45k samples)

**Protocolo:**
- Dividir em chunks de 1000
- Usar chunks 0-2 como buffer inicial
- Avaliar em chunks 3-9
- Comparar com resultados GBML/River

**Métricas:**
- G-mean (primary)
- Accuracy
- Number of features selected
- Execution time

**Sucesso se:**
- ✅ Código executa sem erros
- ✅ G-mean > 0.0
- ✅ Performance comparável a outros modelos

### Experimento 2: Phase 3 Completa (AMANHÃ)

**Objetivo:** Avaliar MCMO em todos datasets

**Datasets:**
- Electricity (2 classes)
- CovType (7 classes)
- Shuttle (7 classes)
- PokerHand (9 classes)
- IntelLabSensors (56 classes)

**Protocolo:** Mesmo que Phase 3 atual (5 training chunks)

**Comparação:**
- MCMO vs GBML/ARF/HAT/SRP/ACDWM/ERulesD2S
- Rankings por dataset
- Testes estatísticos (Friedman, Wilcoxon)

### Experimento 3: Ablation Study (Se Tempo Permitir)

**Variantes:**
```
MCMO_full:     Feature selection + GMM weighting + Drift adaptation
MCMO_noFS:     Sem feature selection
MCMO_noGMM:    Sem GMM weighting
MCMO_noDrift:  Sem drift adaptation
```

**Objetivo:** Entender contribuição de cada componente

---

## ⏱️ Cronograma

### HOJE (Restante - ~4 horas)

**16:00-17:00: GitHub + Setup**
- [ ] Clonar repositório MCMO
- [ ] Instalar dependências
- [ ] Entender estrutura de código
- [ ] Identificar API principal

**17:00-18:30: Implementação Adapter**
- [ ] Criar baseline_mcmo.py
- [ ] Implementar MCMOEvaluator
- [ ] Criar test_mcmo_isolated.py

**18:30-19:30: Teste Isolado**
- [ ] Rodar em Electricity
- [ ] Debug e correções
- [ ] Documentar resultados

**19:30-20:00: Wrap-up**
- [ ] Commit código
- [ ] Atualizar documentação
- [ ] Planejar amanhã

### AMANHÃ (~4-6 horas)

**09:00-11:00: Integração Pipeline**
- [ ] Integrar MCMO em main.py
- [ ] Testar em 2-3 datasets
- [ ] Ajustes e bugfixes

**11:00-13:00: Experimentos Phase 3**
- [ ] Rodar todos 5 datasets
- [ ] Consolidar resultados
- [ ] Calcular rankings

**13:00-14:00: Almoço**

**14:00-15:30: Análise**
- [ ] Testes estatísticos
- [ ] Comparação com baselines
- [ ] Identificar strengths/weaknesses

**15:30-16:30: Documentação**
- [ ] Atualizar paper
- [ ] Criar tabelas/figuras
- [ ] Escrever discussão MCMO

---

## ✅ Critérios de Sucesso

### Mínimo Viável

✓ **MCMO executa** em pelo menos 1 dataset sem erros
✓ **Produz métricas** válidas (G-mean, accuracy)
✓ **Documentação básica** de como funciona

### Desejável

🎯 **Experimentos completos** em 5 datasets Phase 3
🎯 **Comparação estatística** com outros baselines
🎯 **Análise de componentes** (ablation)

### Bônus

💡 **Feature selection** integrado em GBML
💡 **GMM weighting** como opção em outros modelos
💡 **Hybrid approaches** (melhor de cada modelo)

---

## 🚧 Riscos e Mitigações

### Risco 1: Código MCMO não disponível/incompleto

**Probabilidade:** Média
**Impacto:** Alto

**Mitigação:**
- Se GitHub vazio/incompleto: Implementar do zero baseado no paper
- Priorizar componentes core (feature selection, GMM)
- Usar implementações alternativas (scikit-learn GMM, DEAP NSGA-II)

### Risco 2: Performance muito ruim

**Probabilidade:** Baixa (paper mostra bons resultados)
**Impacto:** Médio

**Mitigação:**
- Verificar hyperparameters (usar valores do paper)
- Debug temporal splitting (pode estar criando covariate shift artificial)
- Considerar que MCMO foi desenhado para outro setup

### Risco 3: Tempo de execução muito alto

**Probabilidade:** Média-Alta (NSGA-II é custoso)
**Impacto:** Médio

**Mitigação:**
- Reduzir gerações NSGA-II (paper usa 50, testar 20-30)
- Limitar feature selection re-executions
- Rodar em subset menor de datasets inicialmente

---

## 📚 Recursos e Links

**Paper:**
- IEEE TEVC 2023: "Reduced-space Multistream Classification based on Multi-objective Evolutionary Optimization"

**GitHub:**
- https://github.com/Jesen-BT/MCMO

**Bibliotecas Úteis:**
- `scikit-learn`: GMM, metrics
- `DEAP`: NSGA-II implementation
- `River`: Data stream utilities

**Nossa documentação:**
- `ANALISE_MCMO_PAPER.md`: Análise completa do paper
- `PHASE3_BATCH5_CONSOLIDACAO_COMPLETA.md`: Resultados Phase 3 atuais

---

## 🎯 Próximo Passo Imediato

```bash
# 1. Acessar GitHub
cd "C:\Users\Leandro Almeida\Downloads"
git clone https://github.com/Jesen-BT/MCMO
cd MCMO
ls -la

# 2. Se repositório tem código:
#    → Explorar e entender API
# 3. Se repositório vazio:
#    → Implementar do zero (mais trabalhoso)

# 4. Reportar findings ao usuário
```

---

**Status:** ✅ Paper analisado, plano criado
**Próximo:** Clonar e explorar GitHub
**Estimativa:** 4-6 horas para integração básica
**Criado por:** Claude Code
**Data:** 2025-11-23
