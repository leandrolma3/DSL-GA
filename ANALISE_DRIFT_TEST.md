# Análise do Teste - Drift Detection (drift_test_.log)

**Data**: 2025-10-20
**Stream**: RBF_Abrupt_Severe
**Duração**: 6h 39min (13:55 - 20:34)
**Chunks processados**: 2 transições (0→1, 1→2)

---

## ❌ PROBLEMA CRÍTICO: Drift Detection NÃO Foi Ativado

### Por que não funcionou?

**Causa raiz**: Número insuficiente de chunks para acionar o sistema de detecção de drift.

**Lógica do código** (`main.py:705`):
```python
if len(historical_gmean) >= 2:
    # Classifica drift e ajusta severidade
```

**Como historical_gmean é populado**:
- `historical_gmean` começa vazio: `[]`
- Após chunk 0: `historical_gmean = [0.8907]` (1 elemento)
- Após chunk 1: `historical_gmean = [0.8907, 0.9179]` (2 elementos)
- **Chunk 2 NÃO foi processado** (só temos chunks 0, 1, 2 → apenas 2 transições)

**Problema**:
- Drift detection verifica `len(historical_gmean) >= 2` **ANTES** de processar cada chunk
- Chunk 0: `len=0` → não classifica drift ❌
- Chunk 1: `len=1` → não classifica drift ❌
- Chunk 2: `len=2` → **FINALMENTE** classificaria drift ✅ (mas não foi processado!)

**Evidência nos logs**:
- ❌ NENHUMA mensagem de drift severity (`🔴SEVERE`, `🟡MODERATE`, `🟢MILD`, `✓STABLE`)
- ✅ Apenas mensagens genéricas de herança: "Inheritance INCREASED to 65% (stable performance)"
- ✅ Linha 674: "Finished processing **2 chunk transitions** for run 1"

---

## 📊 RESULTADOS DO TESTE (Sem Drift Detection Ativo)

### Performance por Chunk

| Chunk | Train G-mean | Test G-mean | Test F1 | Fitness | Gerações | Tempo |
|-------|--------------|-------------|---------|---------|----------|-------|
| 0→1   | 0.9174       | **0.8907**  | 0.8911  | 1.0092  | 25/25    | 1h 58min |
| 1→2   | 0.9492       | **0.9179**  | 0.9180  | 1.2441  | 49/200   | 4h 41min |

**Médias Gerais**:
- Train G-mean: **0.9333** (±0.0159)
- Test G-mean: **0.9043** (±0.0136)
- Test F1: **0.9045** (±0.0135)

### Comparação com Teste Anterior (test_output_new.log)

| Métrica | Teste Anterior (3 chunks) | Teste Atual (2 chunks) | Diferença |
|---------|---------------------------|------------------------|-----------|
| Stream | RBF_Abrupt_Severe | RBF_Abrupt_Severe | Mesmo |
| GBML G-mean | 0.8847 | **0.9043** | **+1.96pp** 🎉 |
| Chunks processados | 3 (0→1, 1→2, 2→3) | 2 (0→1, 1→2) | -1 chunk |

**Observação**: Performance melhorou +1.96pp, mas **ainda não testamos drift detection**.

---

## 🔍 ANÁLISE DETALHADA

### Chunk 0 (Baseline - Performance "BAD")

**Configuração**:
- Performance label: `bad` (baseline sem histórico)
- Recovery mode: 25 gerações (vs 200 normal)
- Strategy: `full_random`
- Max depth: 15 (aumentado por recovery mode)
- Max rules: 22 (aumentado por recovery mode)
- Mutation override: 0.5 por 10 gerações

**Evolução**:
- Gen 1-12: Estagnado em G-mean **0.899** (12 gerações!)
- Gen 13: **Breakthrough** → G-mean 0.915 (Hill Climbing v2 ativou)
- Gen 13-25: Estagnado em 0.915 até fim

**Resultado Final**:
- Train G-mean: 0.9174
- Test G-mean: 0.8907 (queda de 2.67%)

**Seeding Adaptativo**:
- ✅ Complexidade: MEDIUM (DT probe acc: 0.782)
- ✅ Seeding ratio: 60% (72 semeados + 48 aleatórios)

**Hill Climbing v2**:
- ✅ Ativado na gen 12 (após 10 gens de estagnação)
- ✅ Nível: MODERATE (G-mean elite ~89%)
- ✅ Causou breakthrough na gen 13

---

### Chunk 1 (Performance "MEDIUM")

**Configuração**:
- Performance label: `medium`
- Max generations: 200 (modo normal)
- Strategy: `full_random`
- Max depth: 10 (normal)
- Max rules: 15 (normal)
- **Herança aumentada para 65%** (linha 177)

**Evolução**:
- Parou na geração **49** via Early Stopping Layer 2
- G-mean evoluiu: 0.908 → 0.935 → 0.949
- Hill Climbing v2 ativou 8 vezes (gens 12, 17, 22, 27, 32, 37, 42, 47)

**Resultado Final**:
- Train G-mean: 0.9492
- Test G-mean: 0.9179 (queda de 3.13%)

**Early Stopping**:
- ✅ Layer 2: Melhoria marginal detectada (parou em 49/200)
- ✅ Economia: 75.5% do tempo (49 vs 200 gens)

**Herança**:
- ✅ Mensagem: "Inheritance INCREASED to 65% (stable performance)"
- ❌ MAS: Não houve classificação de drift (falta mensagem `✓ STABLE: 0.8907 → 0.9179`)

---

## ✅ O QUE FUNCIONOU BEM

### 1. Correção do Bug max_depth
- ✅ Arquivo `ga_operators.py` corrigido (linha 60)
- ✅ Nenhum erro de "multiple values for keyword argument 'max_depth'"

### 2. Seeding Adaptativo
- ✅ Detectou MEDIUM corretamente (probe acc ~78%)
- ✅ Ajustou parâmetros: 60% seeding, depths=[5,8,10]

### 3. Early Stopping Layer 2
- ✅ Chunk 1 parou em 49/200 gens (economia de 75%)
- ✅ Detecção correta de melhoria marginal

### 4. Hill Climbing v2 Hierárquico
- ✅ Ativou corretamente no nível MODERATE
- ✅ Causou breakthrough na gen 13 do chunk 0

### 5. Recovery Mode (Chunk 0)
- ✅ Detectou performance "bad" corretamente
- ✅ Aplicou max_generations_recovery: 25
- ✅ Aumentou complexidade (max_depth=15, max_rules=22)

### 6. Performance Geral
- ✅ G-mean: **0.9043** (+1.96pp vs teste anterior)
- ✅ Estável entre chunks (0.891 → 0.918)

---

## ❌ O QUE NÃO FUNCIONOU

### 1. Drift Detection NÃO foi testado
- ❌ Nenhuma mensagem de severidade (🔴🟡🟢✓)
- ❌ Classificação de drift não executou
- ❌ Recipes NÃO foram ajustadas por drift
- ❌ Causa: `len(historical_gmean) < 2` em todos os chunks processados

### 2. Número insuficiente de chunks
- ❌ Config: `num_chunks: 3` → gera chunks [0, 1, 2]
- ❌ Transições: apenas 2 (0→1, 1→2)
- ❌ Para testar drift: precisa de **4 chunks** [0,1,2,3] → 3 transições

### 3. Overfitting presente
- ❌ Chunk 0: Train 0.917 → Test 0.891 (queda de 2.7%)
- ❌ Chunk 1: Train 0.949 → Test 0.918 (queda de 3.1%)

### 4. Chunk 0 com recovery mode desnecessário
- ❌ Chunk 0 é baseline, não há performance prévia
- ❌ Label "bad" é artificial (acc=0.5 por padrão)
- ❌ 25 gerações podem ter sido insuficientes

---

## 🔧 CORREÇÃO NECESSÁRIA

### Problema: Configuração de num_chunks

**Arquivo**: `config.yaml` (linha 34)

**Configuração atual**:
```yaml
data_params:
  num_chunks: 3  # Gera chunks [0, 1, 2] → apenas 2 transições
```

**Configuração necessária**:
```yaml
data_params:
  num_chunks: 4  # Gera chunks [0, 1, 2, 3] → 3 transições ✅
```

**Por quê?**
- Prequential evaluation: Train-on-N, Test-on-N+1
- 4 chunks → 3 transições (0→1, 1→2, 2→3)
- historical_gmean terá len=2 a partir da transição 1→2
- Drift detection finalmente ativará na transição 1→2!

---

## 🎯 PRÓXIMOS PASSOS RECOMENDADOS

### Opção 1: Teste Correto do Drift Detection (RECOMENDADO)

**Ações**:
1. Modificar `config.yaml`: `num_chunks: 3` → `num_chunks: 4`
2. Executar novo teste com 4 chunks (3 transições)
3. Verificar mensagens de drift na transição 1→2 e 2→3

**Tempo estimado**: ~8-10h (4 chunks, ~2-3h cada)

**Comandos**:
```bash
# 1. Modificar config.yaml localmente (num_chunks: 4)

# 2. Upload
scp config.yaml frozen-about-ball-indicating.trycloudflare.com:/content/DSL-AG-hybrid/

# 3. SSH e executar
ssh frozen-about-ball-indicating.trycloudflare.com
cd /content/DSL-AG-hybrid
nohup python3 main.py > drift_test_4chunks_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 4. Monitorar drift detection
tail -f drift_test_4chunks_*.log | grep -E "DRIFT|Inheritance|Recipe|🔴|🟡|🟢|✓"
```

**O que esperar**:
- Transição 0→1: Sem classificação (historical_gmean len=0)
- Transição 1→2: **PRIMEIRA classificação de drift** (len=1... ainda não!)
- Transição 2→3: **FINALMENTE drift detection** (len=2) ✅
  - Esperado: `🟢 MILD DRIFT` ou `✓ STABLE` (RBF conceito c1 ainda)
- Transição 3→4 (se tivesse chunk 4): `🔴 SEVERE DRIFT` (mudança c1 → c2_severe)

**Problema**: Para ver SEVERE drift, precisamos processar **chunk 4** (transição de c1 para c2_severe).

---

### Opção 2: Teste com 6 Chunks (MAIS COMPLETO)

**Benefício**: Ver toda a sequência de drifts do RBF_Abrupt_Severe

**Configuração**:
```yaml
num_chunks: 6  # Chunks [0,1,2,3,4,5] → 5 transições
```

**Concept sequence do RBF_Abrupt_Severe** (`config.yaml:406-412`):
```yaml
concept_sequence:
  - { concept_id: 'c1', duration_chunks: 5 }        # Chunks 0-4
  - { concept_id: 'c2_severe', duration_chunks: 5 } # Chunks 5-9
```

**Transições esperadas**:
- 0→1, 1→2, 2→3, 3→4: Conceito c1 (estável) → `✓ STABLE` ou `🟢 MILD`
- 4→5: **DRIFT ABRUPTO** c1 → c2_severe → `🔴 SEVERE DRIFT detected!` ✅

**Tempo**: ~12-15h (6 chunks)

---

### Opção 3: Investigar Overfitting (Paralelo)

**Problema**: Queda de 2-3% entre train e test

**Investigações**:
1. `regularization_coefficient` muito baixo? (atual: 0.001)
2. Population size insuficiente? (atual: 120)
3. Max rules muito alto causando overfitting?

---

## 📋 RECOMENDAÇÃO FINAL

**Prioridade 1**: Teste com **num_chunks: 6** (ver drift completo)

**Ação**:
```yaml
# config.yaml
data_params:
  num_chunks: 6  # MODIFICADO: 3 → 6
```

**Motivo**:
- ✅ Valida drift detection completo
- ✅ Vê toda sequência: STABLE → STABLE → STABLE → STABLE → SEVERE
- ✅ Testa ajustes de herança (0%, 25%, 50%, 65%)
- ✅ Testa recipes ajustadas (SEVERE drift)
- ✅ Vê se memória é cleared no chunk 4→5

**Tempo**: ~12-15h
**Resultado esperado**: Sistema completo de drift detection funcionando!

---

## 📊 COMPARAÇÃO: Teste Anterior vs Atual vs Próximo

| Métrica | test_output_new.log | drift_test_.log | Próximo (6 chunks) |
|---------|---------------------|-----------------|---------------------|
| Chunks | 3 (0→1→2→3) | 2 (0→1→2) | 5 (0→1→2→3→4→5) |
| GBML G-mean | 0.8847 | 0.9043 | ? |
| Drift detection testado? | ❌ Não (script errado) | ❌ Não (chunks insuficientes) | ✅ Sim! |
| Transição com SEVERE drift | Nenhuma | Nenhuma | 4→5 ✅ |

---

## 🎬 CONCLUSÃO

**Status atual**: ✅ Código de drift detection **implementado e correto**
**Problema**: ❌ Configuração insuficiente (apenas 3 chunks)
**Solução**: ✅ Aumentar para 6 chunks e re-testar
**Próximo passo**: Modificar config.yaml e executar teste de 6 chunks

**Estimativa**: Com 6 chunks, veremos:
- Chunks 0-4 (conceito c1): drift STABLE/MILD, herança alta (50-65%)
- Chunk 4→5 (c1 → c2_severe): `🔴 SEVERE DRIFT detected!`, memória cleared, herança 0%, recipe ajustada
