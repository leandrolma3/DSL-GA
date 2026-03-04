# 🔄 RETHINKING: ESTRATEGIA DE EXPERIMENTOS E ADAPTACAO A DRIFT

**Data**: 2025-10-30
**Objetivo**: Repensar estrategia completa baseado em evidencias empiricas
**Foco**: Tempo de execucao, multiplos drifts, seeding agressivo

---

## 🎯 PROBLEMAS IDENTIFICADOS (Visao Holistica)

### Problema 1: Restricao de Tempo 24h (Google Colab)

**Evidencia empirica**:
```
Teste anterior: 10h 11min (6 chunks, pop 80, max_gen 200)
Re-teste: 9h 19min (6 chunks, pop 80, max_gen 400 mas parou em 42)
```

**Calculo de impacto**:
- Pop 80 → 100: Aumento de **25%** no tempo por geracao
- Max_gen 200 → 400: Potencial **dobrar** o tempo (se nao houver early stop)
- Tempo estimado worst case: 10h * 1.25 * 2 = **25h** ❌ EXCEDE 24h!

**Conclusao**: Dobrar max_generations para 400 e INVIAVEL para 6 chunks.

---

### Problema 2: Apenas 1 Drift no Stream Atual

**Config atual** (RBF_Abrupt_Severe):
```
Chunk 0-2: c1 (3 chunks)        ✅ Estavel ~91%
Chunk 3-5: c2_severe (3 chunks) ❌ Drift UNICO (queda para 43-45%)
```

**Problema**:
- Apenas UMA transicao de drift (c1 → c2_severe)
- NAO testa recovery (performance continua baixa ate o final)
- NAO testa multiplos drifts consecutivos
- NAO valida se sistema aprende com drifts anteriores

**Evidencia nos logs**:
```
Chunks 0-3: Performance excelente (89-91%)
Chunk 4: Queda catastrofica (43-45%)
Chunk 5: Nao processado (falta chunk 6 para teste)
```

**Conclusao**: Precisamos de stream com **multiplos drifts** e **recovery visivel**.

---

### Problema 3: Seeding Adaptativo vs Seeding Agressivo

**Seeding adaptativo atual**:
```python
DT probe acc: 0.785 (MEDIUM)
→ seeding_ratio = 0.6 (moderado)
```

**Seeding SEVERE esperado**:
```python
drift_severity = 'SEVERE' (60.68%)
→ seeding_ratio = 0.85 (agressivo)
```

**Problema**: Seeding adaptativo SOBRESCREVE seeding por drift severity!

**Conclusao**: Seeding por drift severity deve ter PRIORIDADE MAXIMA.

---

## 📊 ANALISE DE DADOS PASSADOS

### Fase 2 (Experimentos Anteriores):

**Resultados** (do documento ANALISE_FASE2_FINAL.md):
```
Avg Test G-mean: 85.56%
HC Taxa: 28-30%
Chunks: 8 (variavel)
Populacao: 120
```

**Observacao**: Performance MELHOR que testes atuais (85.56% vs 80-81%)

**Diferenca-chave**: Fase 2 tinha populacao MAIOR (120 vs 80)

---

### Teste Atual (6 chunks):

**Configuracao**:
```
Chunks: 6
Populacao: 80
Max gen: 200
Tempo: 9-10h
```

**Performance**:
```
Avg Test G-mean: 80-81% (MENOR que Fase 2)
Chunk 4 (pos-drift): 43-45% (CATASTROFICO)
```

**Hipotese**: Populacao 80 pode ser INSUFICIENTE para drift SEVERE.

---

## 💡 PROPOSTA: POPULACAO 100 vs MAX_GEN 400

### Analise de Trade-offs:

**Opcao A: Dobrar Max Generations (200 → 400)**

Prós:
- ✅ Mais tempo para evoluir no chunk de drift
- ✅ Pode melhorar adaptacao (se early stop nao parar cedo)

Contras:
- ❌ Tempo: 10h * 2 = 20h (proximo do limite 24h)
- ❌ Early stop pode parar cedo (desperdicar geracoes)
- ❌ Se houver problemas, pode EXCEDER 24h

**Risco**: ALTO (pode perder experimento por timeout)

---

**Opcao B: Aumentar Populacao (80 → 100)**

Prós:
- ✅ Mais diversidade DESDE O INICIO
- ✅ Mais individuos para Hill Climbing explorar
- ✅ Aumento linear e previsivel: +25% tempo
- ✅ Tempo estimado: 10h * 1.25 = 12.5h (SEGURO!)

Contras:
- ⚠️ Aumento moderado (nao dramatico)
- ⚠️ Pode nao ser suficiente para drift SEVERE

**Risco**: BAIXO (seguro dentro de 24h)

---

**Opcao C: Populacao 100 + Max Gen 300 (Hibrido)**

Prós:
- ✅ Balanco entre exploracao (pop) e explotacao (gen)
- ✅ Aumento de 50% em max_gen (menos arriscado que 100%)
- ✅ Tempo estimado: 10h * 1.25 * 1.5 = 18.75h (LIMITE!)

Contras:
- ⚠️ Proximo ao limite de 24h
- ❌ Se houver atraso, pode exceder

**Risco**: MEDIO

---

### Recomendacao: OPCAO B (Pop 100, Max Gen 200)

**Justificativa**:
1. ✅ Tempo SEGURO (12.5h << 24h)
2. ✅ Melhora diversidade (critico para drift)
3. ✅ Permite testar MULTIPLOS drifts (ver secao seguinte)
4. ✅ Aumento previsivel e controlado

**Trade-off aceito**: Sacrificar geracoes extras por seguranca de tempo

---

## 🔄 PROPOSTA: STREAM COM MULTIPLOS DRIFTS

### Problema do Stream Atual:

```
RBF_Abrupt_Severe (6 chunks):
├── c1 (3 chunks)      → Estavel 91%
└── c2_severe (3 chunks) → Queda 43%, SEM recovery
```

**Limitacoes**:
- Apenas 1 drift
- Sem recovery
- Sem teste de memoria/aprendizado

---

### Proposta: Stream com 3 Drifts

**Config**: RBF_Multi_Drift (8 chunks)

```yaml
experimental_streams:
  RBF_Multi_Drift:
    dataset_type: RBF
    drift_type: abrupt
    concept_sequence:
      - concept_id: c1
        duration_chunks: 2          # Chunks 0-1: Baseline
      - concept_id: c2_moderate     # DRIFT 1 (MODERATE)
        duration_chunks: 2          # Chunks 2-3: Adaptar
      - concept_id: c1             # DRIFT 2 (Volta ao original)
        duration_chunks: 2          # Chunks 4-5: Recovery
      - concept_id: c2_severe      # DRIFT 3 (SEVERE)
        duration_chunks: 2          # Chunks 6-7: Teste severo
```

**Expectativa de Performance**:

| Chunk | Conceito | Drift | Performance Esperada | Observacao |
|-------|----------|-------|---------------------|------------|
| **0-1** | c1 | - | 88-92% | Baseline (estavel) |
| **2** | c2_moderate | MODERATE | 70-80% (queda) | Drift 1: Adaptar |
| **3** | c2_moderate | - | 82-88% (recovery) | Recovery visivel |
| **4** | c1 | MODERATE | 75-85% (queda) | Drift 2: Volta ao conhecido |
| **5** | c1 | - | 88-92% (recovery) | Recovery ao baseline |
| **6** | c2_severe | SEVERE | 50-60% (queda forte) | Drift 3: Teste severo |
| **7** | c2_severe | - | 65-75% (recovery) | Recovery parcial |

**Vantagens**:
1. ✅ Testa 3 tipos de drift: MODERATE, volta ao original, SEVERE
2. ✅ Valida recovery (chunks 3, 5, 7)
3. ✅ Testa memoria (volta a c1 no chunk 4)
4. ✅ Curva de performance VARIAVEL (nao monotonica)

**Tempo estimado**: 12.5h * (8/6) = **16.7h** (SEGURO dentro de 24h!)

---

### Alternativa: Stream com 2 Drifts + Recovery

**Config**: RBF_Drift_Recovery (6 chunks)

```yaml
experimental_streams:
  RBF_Drift_Recovery:
    dataset_type: RBF
    drift_type: abrupt
    concept_sequence:
      - concept_id: c1
        duration_chunks: 2          # Chunks 0-1: Baseline
      - concept_id: c2_moderate     # DRIFT 1 (MODERATE)
        duration_chunks: 2          # Chunks 2-3: Adaptar
      - concept_id: c1             # DRIFT 2 (Recovery)
        duration_chunks: 2          # Chunks 4-5: Volta ao baseline
```

**Expectativa**:

| Chunk | Conceito | Performance Esperada |
|-------|----------|---------------------|
| **0-1** | c1 | 88-92% (baseline) |
| **2** | c2_moderate | 70-80% (queda) |
| **3** | c2_moderate | 82-88% (recovery) |
| **4** | c1 | 75-85% (queda menor - memoria?) |
| **5** | c1 | 88-92% (recovery completo) |

**Vantagens**:
- ✅ 2 drifts testados
- ✅ Recovery testado 2 vezes
- ✅ Mesmo tempo que config atual (~12.5h)
- ✅ Testa memoria (volta a c1)

**Recomendacao**: Esta config para VALIDACAO inicial!

---

## 🚀 RETHINKING: SEEDING AGRESSIVO PARA SEVERE

### Problema Atual:

**Seeding adaptativo**:
```python
if DT_probe_acc between 0.75-0.90:
    seeding_ratio = 0.6  # MODERADO
```

**Seeding por severity**:
```python
if drift_severity == 'SEVERE':
    seeding_ratio = 0.85  # AGRESSIVO
```

**Conflito**: Adaptativo SOBRESCREVE severity!

---

### Solucao Proposta: Sistema de Prioridades

**Hierarquia**:
1. **PRIORIDADE MAXIMA**: Drift severity (SEVERE → 85%)
2. **PRIORIDADE MEDIA**: Performance ruim (bad → recovery mode)
3. **PRIORIDADE BAIXA**: Seeding adaptativo (baseado em complexidade)

**Pseudocodigo**:
```python
def determinar_seeding(drift_severity, performance_label, enable_adaptive):
    # NIVEL 1: Drift severity (prioridade MAXIMA)
    if drift_severity == 'SEVERE':
        return {
            'seeding_ratio': 0.85,
            'injection_ratio': 0.90,
            'reason': 'SEVERE DRIFT detected'
        }

    # NIVEL 2: Performance ruim (prioridade MEDIA)
    if performance_label == 'bad':
        return {
            'seeding_ratio': 0.60,
            'injection_ratio': 0.60,
            'reason': 'Recovery mode (bad performance)'
        }

    # NIVEL 3: Seeding adaptativo (prioridade BAIXA)
    if enable_adaptive:
        complexity = estimar_complexidade_dt_probe()
        if complexity == 'HARD':
            return {'seeding_ratio': 0.80, 'reason': 'Adaptive - HARD'}
        elif complexity == 'MEDIUM':
            return {'seeding_ratio': 0.60, 'reason': 'Adaptive - MEDIUM'}
        else:  # SIMPLE
            return {'seeding_ratio': 0.40, 'reason': 'Adaptive - SIMPLE'}

    # DEFAULT: Seeding padrao
    return {
        'seeding_ratio': 0.80,  # Valor do config
        'reason': 'Default seeding'
    }
```

**Implementacao**: Modificar `ga.py` linha ~42-48 (funcao de inicializacao)

---

### Seeding AINDA MAIS Agressivo?

**Proposta**: SEVERE drift = 90% (nao 85%)

**Justificativa**:
- Drift SEVERE = 60%+ diferenca entre conceitos
- Sistema precisa REAPRENDER quase do zero
- 85% pode ainda ser insuficiente

**Config**:
```python
if drift_severity == 'SEVERE':
    seeding_ratio = 0.90  # 90% da populacao semeada!
    injection_ratio = 0.95  # 95% das regras de cada individuo
```

**Trade-off**: Menos diversidade aleatoria, mas populacao inicial MUITO melhor

**Recomendacao**: Testar 90% (mais agressivo que 85%)

---

## 📋 PLANO DE ACAO ROBUSTO

### Fase 1: VALIDACAO com Config Simples (1 experimento)

**Objetivo**: Validar correcoes basicas

**Config**: RBF_Drift_Recovery (6 chunks)
- 2 drifts (c1 → c2_moderate → c1)
- Testa recovery e memoria

**Parametros**:
```yaml
data_params:
  num_chunks: 6
  chunk_size: 6000

ga_params:
  population_size: 100        # ERA: 80 (+25%)
  max_generations: 200        # Manter
  early_stopping_patience: 20 # Manter

  # Seeding agressivo para SEVERE
  dt_seeding_ratio_severe: 0.90  # NOVO (era 0.85 implicito)
  enable_adaptive_seeding: false # DESABILITAR para este teste
```

**Mudancas em main.py**:
```python
# Ao detectar SEVERE drift:
if drift_severity == 'SEVERE':
    # NAO dobrar max_gen (manter 200)
    # Passar flag para ga.py usar seeding 90%
    init_strategy_config['force_seeding_ratio'] = 0.90
    init_strategy_config['disable_adaptive'] = True
```

**Mudancas em ga.py**:
```python
# Em funcao de inicializacao:
if 'force_seeding_ratio' in init_strategy_config:
    seeding_ratio = init_strategy_config['force_seeding_ratio']
    # NAO aplicar seeding adaptativo
else:
    # Logica normal (com adaptativo se habilitado)
```

**Tempo estimado**: 12.5h (seguro)

**Metricas de sucesso**:
- ✅ Seeding 90% ativado no chunk 2 (drift 1)
- ✅ Recovery visivel no chunk 3 (80%+)
- ✅ Seeding 90% ativado no chunk 4 (drift 2)
- ✅ Recovery visivel no chunk 5 (88%+)
- ✅ Avg Test G-mean ≥ 83%

---

### Fase 2: TESTE com Multiplos Drifts (1 experimento)

**Objetivo**: Validar adaptacao a multiplos drifts consecutivos

**Config**: RBF_Multi_Drift (8 chunks)
- 3 drifts (c1 → c2_moderate → c1 → c2_severe)

**Parametros**: Mesmos da Fase 1

**Tempo estimado**: 16.7h (seguro)

**Metricas de sucesso**:
- ✅ Seeding ativado nos 3 drifts
- ✅ Recovery visivel apos cada drift
- ✅ Performance no chunk 7 (pos-SEVERE) ≥ 65%
- ✅ Avg Test G-mean ≥ 78%

---

### Fase 3: BATCH de Streams (Opcional, se Fase 1-2 OK)

**Objetivo**: Validar em multiplos tipos de drift

**Streams sugeridos** (5 streams, ~3 dias total):
1. RBF_Drift_Recovery (6 chunks, MODERATE)
2. RBF_Multi_Drift (8 chunks, MODERATE + SEVERE)
3. SEA_Abrupt (6 chunks, 2 drifts)
4. STAGGER_Recurring (8 chunks, recorrente)
5. HYPERPLANE_Gradual (8 chunks, gradual)

**Tempo estimado**: 13h * 5 = 65h (~3 dias em multiplas maquinas)

---

## 🔧 CORRECOES TECNICAS NECESSARIAS

### Correcao 1: Sistema de Prioridades de Seeding (CRITICO)

**Arquivo**: `ga.py` (funcao de inicializacao, linha ~42-100)

**Mudanca**:
```python
def initialize_population_with_seeding(..., force_seeding_ratio=None):
    # PRIORIDADE 1: Seeding forcado (para SEVERE drift)
    if force_seeding_ratio is not None:
        seeding_ratio = force_seeding_ratio
        logging.info(f"  -> FORCED SEEDING: {seeding_ratio*100:.0f}%")

    # PRIORIDADE 2: Seeding adaptativo (se nao forcado)
    elif enable_adaptive_seeding and drift_severity != 'SEVERE':
        # ... logica adaptativa atual ...
        seeding_ratio = calcular_baseado_complexidade()

    # PRIORIDADE 3: Default
    else:
        seeding_ratio = dt_seeding_ratio_on_init_config
```

---

### Correcao 2: Passar force_seeding_ratio de main.py (CRITICO)

**Arquivo**: `main.py` (linha ~1289-1296)

**Mudanca**:
```python
if drift_severity_for_chunk_i == 'SEVERE':
    logging.warning(f"SEVERE DRIFT detected for chunk {i}")

    # Configurar seeding agressivo
    ga_p_adjusted = ga_p.copy()
    ga_p_adjusted['force_seeding_ratio'] = 0.90  # SEEDING 90%!
    ga_p_adjusted['force_injection_ratio'] = 0.95

    logging.info(f"  -> AGGRESSIVE SEEDING activated: 90% seeding, 95% injection")
```

---

### Correcao 3: Criar Configs de Streams (MEDIA)

**Arquivo**: `config_test_drift_recovery.yaml` (NOVO)

**Conteudo**: Config da Fase 1 (RBF_Drift_Recovery, 6 chunks)

**Arquivo**: `config_test_multi_drift.yaml` (NOVO)

**Conteudo**: Config da Fase 2 (RBF_Multi_Drift, 8 chunks)

---

### Correcao 4: Atualizar Populacao para 100 (FACIL)

**Arquivo**: Configs criados acima

**Mudanca**:
```yaml
ga_params:
  population_size: 100  # ERA: 80
```

---

## 📊 TABELA DE DECISAO

| Aspecto | Opcao A (Max Gen 400) | Opcao B (Pop 100) | **ESCOLHIDA** |
|---------|---------------------|------------------|--------------|
| **Tempo** | 20h (arriscado) | 12.5h (seguro) | **Opcao B** ✅ |
| **Diversidade** | Mesma (80 ind) | Maior (+25%) | **Opcao B** ✅ |
| **Exploracao** | Mais geracoes | Mais individuos | **Opcao B** ✅ |
| **Risco timeout** | ALTO | BAIXO | **Opcao B** ✅ |
| **Permite multiplos drifts** | Nao (tempo) | Sim (8 chunks OK) | **Opcao B** ✅ |

---

## 🎯 EXPECTATIVAS DO NOVO PLANO

### Fase 1 (RBF_Drift_Recovery):

**Esperado**:
```
Chunk 0-1 (c1): 88-92% (baseline)
Chunk 2 (c2_mod, drift): 70-80% (queda, seeding 90% ativado)
Chunk 3 (c2_mod): 82-88% (recovery com pop 100)
Chunk 4 (c1, drift): 75-85% (queda menor - memoria)
Chunk 5 (c1): 88-92% (recovery completo)
Avg Test G-mean: ≥ 83% (meta)
```

**Se falhar**: Problema nao e quantidade de geracao, e outra coisa (regras DT ruins, fitness inadequado, etc)

---

### Fase 2 (RBF_Multi_Drift):

**Esperado**:
```
Chunk 0-1 (c1): 88-92%
Chunk 2-3 (c2_mod): 70-80% → 82-88%
Chunk 4-5 (c1): 75-85% → 88-92%
Chunk 6-7 (c2_sev): 50-60% → 65-75%
Avg Test G-mean: ≥ 78% (meta)
```

**Se OK**: Sistema e robusto a multiplos drifts consecutivos!

---

## ✅ CHECKLIST DE IMPLEMENTACAO

### Codigo:
- [ ] Correcao 1: Sistema prioridades seeding em ga.py
- [ ] Correcao 2: Passar force_seeding_ratio em main.py
- [ ] Correcao 3: Criar config_test_drift_recovery.yaml
- [ ] Correcao 4: Criar config_test_multi_drift.yaml

### Validacao Local:
- [ ] Testar que seeding 90% e ativado quando SEVERE
- [ ] Testar que seeding adaptativo NAO sobrescreve forced
- [ ] Validar configs YAML (syntax, soma de chunks)

### Execucao:
- [ ] Fase 1: RBF_Drift_Recovery (12.5h)
- [ ] Analisar resultados Fase 1
- [ ] Fase 2: RBF_Multi_Drift (16.7h) - se Fase 1 OK
- [ ] Analisar resultados Fase 2

---

## 📖 CONCLUSAO

### Decisoes-Chave:

1. ✅ **Populacao 100** (nao dobrar max_gen para 400)
2. ✅ **Seeding 90%** para SEVERE (nao 85%)
3. ✅ **Sistema de prioridades** de seeding (severity > adaptativo)
4. ✅ **Multiplos drifts** nos streams (nao apenas 1)
5. ✅ **6-8 chunks** (seguro dentro de 24h)

### Hipoteses a Testar:

**H1**: Populacao 100 > Max_gen 400 para drift adaptation
**H2**: Seeding 90% e suficiente para drift SEVERE
**H3**: Sistema pode recovery apos multiplos drifts
**H4**: Memoria ajuda quando drift volta ao conceito anterior

### Proxima Acao:

**IMPLEMENTAR CORRECOES 1-4 e EXECUTAR FASE 1**

---

**Documento criado por**: Claude Code
**Data**: 2025-10-30
**Status**: 🔄 **ESTRATEGIA REPENSADA - PRONTO PARA IMPLEMENTAR**
