# Plano de Re-execução Batch 1 - Configurações Corrigidas

**Data**: 2025-01-18
**Status**: PRONTO PARA EXECUÇÃO

---

## ANÁLISE DE TEMPOS DE EXECUÇÃO (Execução Original)

### Tempos Individuais por Dataset:

| Dataset | Tempo (segundos) | Tempo (minutos) | Tempo (horas) | Complexidade |
|---------|------------------|-----------------|---------------|--------------|
| SEA_Abrupt_Simple | 1,582 | 26.4 | 0.44 | ⚡ RÁPIDO |
| STAGGER_Abrupt_Chain | 1,690 | 28.2 | 0.47 | ⚡ RÁPIDO |
| AGRAWAL_Abrupt_Simple_Severe | 4,471 | 74.5 | 1.25 | 🔶 MÉDIO |
| RBF_Abrupt_Severe | 8,116 | 135.3 | 2.25 | 🔴 LENTO |
| HYPERPLANE_Abrupt_Simple | 10,941 | 182.3 | 3.05 | 🔴 MUITO LENTO |

**Total**: ~7.46 horas para todos os 5 datasets

### Agrupamento Original (Ineficiente):

| Grupo | Datasets | Tempo Total | Eficiência |
|-------|----------|-------------|------------|
| exec_1 | SEA + AGRAWAL | ~1.68 horas | ✅ BOM |
| exec_2 | RBF + HYPERPLANE | ~5.29 horas | ⚠️ MUITO LONGO |
| exec_3 | STAGGER (sozinho) | ~0.47 horas | ❌ DESPERDÍCIO |

**Problema**: exec_2 muito longo (risco de timeout Colab), exec_3 subutilizado

---

## PROPOSTA OTIMIZADA: 2 Instâncias Colab

### Agrupamento Balanceado:

| Grupo | Datasets | Tempo Estimado | % Colab (6h) | Status |
|-------|----------|----------------|--------------|--------|
| **Colab 1** | HYPERPLANE + SEA + STAGGER | ~3.95 horas | 66% | ✅ ÓTIMO |
| **Colab 2** | RBF + AGRAWAL | ~3.50 horas | 58% | ✅ ÓTIMO |

**Vantagens**:
- ✅ Apenas 2 instâncias Colab necessárias
- ✅ Ambas abaixo de 4 horas (margem de segurança)
- ✅ Uso balanceado dos recursos
- ✅ Pode executar em paralelo (termina em ~4 horas total)

**Margem de segurança**: ~2 horas em cada instância para overhead/variações

---

## CONFIGURAÇÕES CORRIGIDAS

### Arquivo: `configs/config_batch_1.yaml`

#### Mudanças Necessárias (Apenas seção experimental_streams):

```yaml
experimental_streams:
  # ... (manter todas as outras configs intactas)

  # ========== CORREÇÃO 1: SEA_Abrupt_Simple ==========
  SEA_Abrupt_Simple:
    dataset_type: SEA
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: f1
      duration_chunks: 3          # ✏️ MUDOU: era 5
    - concept_id: f3
      duration_chunks: 3          # ✏️ MUDOU: era 5
    # ✅ Drift agora em 3000 instâncias (chunk 3) - DENTRO do range de treinamento

  # ========== CORREÇÃO 2: AGRAWAL_Abrupt_Simple_Severe ==========
  AGRAWAL_Abrupt_Simple_Severe:
    dataset_type: AGRAWAL
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: f1
      duration_chunks: 3          # ✏️ MUDOU: era 5
    - concept_id: f6
      duration_chunks: 3          # ✏️ MUDOU: era 5
    # ✅ Drift agora em 3000 instâncias (chunk 3) - DENTRO do range de treinamento

  # ========== CORREÇÃO 3: RBF_Abrupt_Severe ==========
  RBF_Abrupt_Severe:
    dataset_type: RBF
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: c1
      duration_chunks: 3          # ✏️ MUDOU: era 5
    - concept_id: c2_severe
      duration_chunks: 3          # ✏️ MUDOU: era 5
    # ✅ Drift agora em 3000 instâncias (chunk 3) - DENTRO do range de treinamento

  # ========== CORREÇÃO 4: STAGGER_Abrupt_Chain ==========
  STAGGER_Abrupt_Chain:
    dataset_type: STAGGER
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: f1
      duration_chunks: 2          # ✏️ MUDOU: era 4
    - concept_id: f2
      duration_chunks: 2          # ✏️ MUDOU: era 4
    - concept_id: f3
      duration_chunks: 2          # ✏️ MUDOU: era 4
    # ✅ Drift 1 em 2000 instâncias (chunk 2) - DENTRO do range
    # ✅ Drift 2 em 4000 instâncias (chunk 4) - DENTRO do range

  # ========== CORREÇÃO 5: HYPERPLANE_Abrupt_Simple ==========
  HYPERPLANE_Abrupt_Simple:
    dataset_type: HYPERPLANE
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: plane1
      duration_chunks: 3          # ✏️ MUDOU: era 6
    - concept_id: plane2
      duration_chunks: 3          # ✏️ MUDOU: era 6
    # ✅ Drift agora em 3000 instâncias (chunk 3) - DENTRO do range de treinamento
```

### Resumo das Mudanças:

| Dataset | Campo | Antes | Depois | Drift Position |
|---------|-------|-------|--------|----------------|
| SEA | duration_chunks (ambos) | 5 → 5 | 3 → 3 | 5000 → 3000 ✅ |
| AGRAWAL | duration_chunks (ambos) | 5 → 5 | 3 → 3 | 5000 → 3000 ✅ |
| RBF | duration_chunks (ambos) | 5 → 5 | 3 → 3 | 5000 → 3000 ✅ |
| STAGGER | duration_chunks (3 conceitos) | 4 → 4 → 4 | 2 → 2 → 2 | 4000,8000 → 2000,4000 ✅ |
| HYPERPLANE | duration_chunks (ambos) | 6 → 6 | 3 → 3 | 6000 → 3000 ✅ |

**Total de linhas modificadas**: 10 linhas (5 datasets × 2 valores médios)

---

## VALORES ESPERADOS DE DRIFT SEVERITY (Após Correção)

Com base nas definições de conceitos e na metodologia do ChiuTNNLS2020.pdf:

| Dataset | Conceitos | Severity Esperada | Justificativa |
|---------|-----------|-------------------|---------------|
| **SEA** | f1 vs f3 | **35-45%** | Threshold muda de 8.0 → 7.0 (mudança moderada na fronteira de decisão) |
| **AGRAWAL** | f1 vs f6 | **55-70%** | Funções de classificação completamente diferentes (1 vs 6), mudança severa |
| **RBF** | c1 vs c2_severe | **60-75%** | Seeds diferentes (42 vs 84) criam centróides RBF muito diferentes, mudança severa |
| **STAGGER** | f1 vs f2 | **25-35%** | Regras diferentes mas com alguma sobreposição |
| **STAGGER** | f2 vs f3 | **30-40%** | Regras diferentes, mudança moderada |
| **HYPERPLANE** | plane1 vs plane2 | **15-25%** | mag_change=0.01 (mudança pequena no hiperplano) |

**IMPORTANTE**: Esses valores NÃO devem ser 0.0%! Se aparecer 0.0%, indica problema na geração/cálculo.

---

## PLANO DE EXECUÇÃO PASSO A PASSO

### FASE 1: Preparação (10 minutos)

1. **Backup da configuração atual** (opcional mas recomendado):
   ```bash
   cp configs/config_batch_1.yaml configs/config_batch_1_OLD.yaml
   ```

2. **Atualizar config_batch_1.yaml**:
   - Editar seção `experimental_streams` (linhas 411-477)
   - Aplicar as 10 mudanças de duration_chunks conforme tabela acima
   - Salvar arquivo

3. **Validar sintaxe YAML**:
   ```python
   import yaml
   with open('configs/config_batch_1.yaml', 'r') as f:
       config = yaml.safe_load(f)
   print("✅ YAML válido!" if config else "❌ Erro de sintaxe!")
   ```

4. **Criar script de execução para Colab**:
   - `batch_1_corrected_group1.yaml` (HYPERPLANE + SEA + STAGGER)
   - `batch_1_corrected_group2.yaml` (RBF + AGRAWAL)

---

### FASE 2: Re-execução GBML (Colab)

#### Grupo 1: Colab Instance #1 (~3.95 horas)

**Datasets**: HYPERPLANE_Abrupt_Simple + SEA_Abrupt_Simple + STAGGER_Abrupt_Chain

**Config temporário** (`config_batch_1_group1.yaml`):
```yaml
experiment_settings:
  run_mode: drift_simulation
  drift_simulation_experiments:
  - HYPERPLANE_Abrupt_Simple
  - SEA_Abrupt_Simple
  - STAGGER_Abrupt_Chain
  num_runs: 1
  base_results_dir: /content/drive/.../experiments_6chunks_phase1_gbml/batch_1_CORRECTED
  # ... resto igual ao config_batch_1.yaml
```

**Comando**:
```python
!python main.py --config configs/config_batch_1_group1.yaml
```

**Monitoramento**:
- SEA deve completar em ~26 min ✓
- STAGGER deve completar em ~28 min ✓
- HYPERPLANE deve completar em ~182 min ✓
- **Total estimado**: ~236 min (~3.95 horas)

---

#### Grupo 2: Colab Instance #2 (~3.50 horas)

**Datasets**: RBF_Abrupt_Severe + AGRAWAL_Abrupt_Simple_Severe

**Config temporário** (`config_batch_1_group2.yaml`):
```yaml
experiment_settings:
  run_mode: drift_simulation
  drift_simulation_experiments:
  - RBF_Abrupt_Severe
  - AGRAWAL_Abrupt_Simple_Severe
  num_runs: 1
  base_results_dir: /content/drive/.../experiments_6chunks_phase1_gbml/batch_1_CORRECTED
  # ... resto igual ao config_batch_1.yaml
```

**Comando**:
```python
!python main.py --config configs/config_batch_1_group2.yaml
```

**Monitoramento**:
- RBF deve completar em ~135 min ✓
- AGRAWAL deve completar em ~74 min ✓
- **Total estimado**: ~209 min (~3.50 horas)

---

### FASE 3: Validação dos Resultados GBML (30 minutos)

**Para cada dataset, verificar**:

1. **Drift positions corretas**:
   ```python
   # Ler Plot_AccuracyPeriodic_<DATASET>_Run1.png
   # Verificar que os drift markers estão em:
   # - SEA, AGRAWAL, RBF, HYPERPLANE: 3000 instâncias
   # - STAGGER: 2000 e 4000 instâncias
   ```

2. **Drift severity não-zero**:
   ```python
   # Verificar que os plots mostram severidades como:
   # - SEA: ~35-45%
   # - AGRAWAL: ~55-70%
   # - RBF: ~60-75%
   # - STAGGER drift1: ~25-35%, drift2: ~30-40%
   # - HYPERPLANE: ~15-25%
   ```

3. **Modelos treinam em AMBOS conceitos**:
   ```python
   # Verificar chunk_metrics.json
   # - Chunks 0-2: conceito 1
   # - Chunks 3-4: conceito 2 (ou chunk 4 para conceito 3 no STAGGER)
   ```

4. **Performance drops visíveis nos drifts**:
   - Accuracy/G-mean deve cair nos chunks de drift
   - Deve recuperar nos chunks seguintes (se o modelo adaptar)

**Critérios de aceitação**:
- ✅ Todos os drifts dentro do range 0-5000 instâncias
- ✅ Todas as severidades > 0%
- ✅ Modelo treina em todos os conceitos
- ✅ Drops de performance visíveis

---

### FASE 4: Re-execução Modelos Comparativos (Colab)

**Pré-requisito**: FASE 3 completa e validada

**Tempo estimado**: ~2-3 horas (todos os datasets, todos os modelos)

**Script**: `run_comparative_on_existing_chunks.py` (já validado, sem mudanças necessárias)

**Modelos**: River (HAT, ARF, SRP) + ACDWM + ERulesD2S

**Comando** (Colab):
```python
# Célula 6 (já existente, sem mudanças)
# Aponta para: batch_1_CORRECTED em vez de batch_1
```

**Output esperado**:
- river_HAT_results.csv, river_ARF_results.csv, river_SRP_results.csv
- acdwm_results.csv
- erulesd2s_results.csv

Para cada dataset.

---

### FASE 5: Post-processing (15 minutos)

**Para cada dataset, executar**:

1. **generate_plots.py**:
   ```python
   !python generate_plots.py --dataset_dir batch_1_CORRECTED/<DATASET>/run_1
   ```
   - Output: plots/ com Plot_AccuracyPeriodic, GA_Evolution, etc.

2. **rule_diff_analyzer.py**:
   ```python
   !python rule_diff_analyzer.py --dataset_dir batch_1_CORRECTED/<DATASET>/run_1
   ```
   - Output: rule_transitions/ com report.txt, matrix.csv, matrix.png

3. **analyze_concept_difference.py**:
   ```python
   !python analyze_concept_difference.py --config config_batch_1.yaml
   ```
   - Output: concept_differences.json com severidades calculadas

**Validação**:
- ✅ Plots gerados sem erros
- ✅ Rule transitions mostram mudanças entre chunks de drift
- ✅ Concept differences match valores esperados (±5%)

---

### FASE 6: Consolidação e Análise Estatística (20 minutos)

**Executar no Colab**:

1. **CELULA 11** (já validada):
   - Consolidar resultados GBML + River + ACDWM + ERulesD2S
   - Testes estatísticos (Friedman, Wilcoxon)
   - Ranking por test_gmean

2. **CELULA 12** (já validada):
   - Gerar 6 plots comparativos
   - lineplot_chunks_evolution.png
   - boxplots por métrica

**Output esperado**:
- consolidated_results.csv
- statistical_tests.txt
- model_ranking.csv
- 6 plots comparativos

**Validação**:
- ✅ Todos os modelos aparecem nos plots (sem sobreposição invisível)
- ✅ Testes estatísticos executam sem erros
- ✅ Ranking faz sentido (modelos com melhor test_gmean no topo)

---

## CHECKLIST DE VALIDAÇÃO FINAL

### Pré-execução:
- [ ] Config backup criado (config_batch_1_OLD.yaml)
- [ ] Config atualizado com 10 mudanças de duration_chunks
- [ ] Sintaxe YAML validada
- [ ] Configs de grupo criados (group1.yaml, group2.yaml)

### Pós GBML:
- [ ] 5 datasets executados sem erros
- [ ] Drift markers em posições corretas (2000-4000, não 5000-8000)
- [ ] Drift severities > 0% e dentro dos ranges esperados
- [ ] chunk_metrics.json mostra treinamento em todos os conceitos
- [ ] Performance drops visíveis nos plots

### Pós Comparativos:
- [ ] 5 datasets × 6 modelos = 30 arquivos CSV gerados
- [ ] ERulesD2S chunks 0-4 (não 0-5)
- [ ] Todos os modelos têm train e test metrics

### Pós Post-processing:
- [ ] 5 datasets × plots organizados
- [ ] 5 datasets × rule transitions
- [ ] concept_differences.json com valores corretos

### Pós Consolidação:
- [ ] consolidated_results.csv completo
- [ ] Testes estatísticos executados
- [ ] 6 plots comparativos gerados
- [ ] Todos os modelos visíveis (sem overlap)

---

## CRONOGRAMA ESTIMADO

| Fase | Atividade | Tempo | Responsável |
|------|-----------|-------|-------------|
| 1 | Preparação e ajuste config | 10 min | Manual |
| 2a | Re-execução GBML Grupo 1 (Colab) | 3.95 h | Automatizado |
| 2b | Re-execução GBML Grupo 2 (Colab) | 3.50 h | Automatizado |
| 3 | Validação GBML | 30 min | Manual |
| 4 | Re-execução Comparativos (Colab) | 2-3 h | Automatizado |
| 5 | Post-processing | 15 min | Automatizado |
| 6 | Consolidação | 20 min | Automatizado |

**Total tempo de máquina**: ~7-8 horas (Colab)
**Total tempo de trabalho manual**: ~1 hora 10 min
**Total tempo wall-clock** (se executar grupos em paralelo): ~4-5 horas

---

## MUDANÇAS NO CÓDIGO

**Resposta curta**: ❌ **NENHUMA**

**Arquivos que precisam ser modificados**:
1. ✅ `configs/config_batch_1.yaml` - APENAS seção experimental_streams (10 linhas)

**Arquivos que permanecem intactos**:
- ✅ Todo o código GBML (main.py, data_handling_final.py, etc.)
- ✅ run_comparative_on_existing_chunks.py
- ✅ generate_plots.py
- ✅ rule_diff_analyzer.py
- ✅ analyze_concept_difference.py
- ✅ CELULA 11 e CELULA 12 (scripts de consolidação)

**Risco**: 🟢 **MÍNIMO** - Mudança puramente configuracional

---

## RESULTADOS ESPERADOS

### Antes da Correção:
- ❌ Drifts em 4000-8000 instâncias (fora do range)
- ❌ Severity 0.0% (não calculada corretamente)
- ❌ Modelos não treinam em todos os conceitos
- ❌ Resultados INVÁLIDOS para publicação

### Depois da Correção:
- ✅ Drifts em 2000-4000 instâncias (dentro do range)
- ✅ Severity 15-75% (valores reais e significativos)
- ✅ Modelos treinam em todos os conceitos
- ✅ Resultados VÁLIDOS para publicação científica

---

## PRÓXIMOS PASSOS IMEDIATOS

1. **AGORA**: Revisar este plano e aprovar as mudanças
2. **HOJE**: Atualizar config_batch_1.yaml com as correções
3. **HOJE/AMANHÃ**: Executar Grupo 1 no Colab (iniciar noite, validar manhã)
4. **AMANHÃ**: Executar Grupo 2 no Colab (em paralelo com Grupo 1 se possível)
5. **AMANHÃ**: Validar resultados GBML
6. **DEPOIS**: Executar comparativos e consolidação

**Tempo total até resultados finais**: 1-2 dias (depende de disponibilidade Colab)

---

## OBSERVAÇÕES IMPORTANTES

1. **Não deletar resultados antigos**: Manter `batch_1/` e criar `batch_1_CORRECTED/` separadamente
2. **Logs detalhados**: Salvar logs de cada execução para debugging
3. **Validação incremental**: Validar cada fase antes de prosseguir
4. **Backup antes de consolidar**: Backup de consolidated_results.csv antes de sobrescrever

---

**Status**: ✅ PRONTO PARA EXECUÇÃO
**Aprovação necessária**: ⏳ AGUARDANDO REVISÃO DO USUÁRIO
**Risco**: 🟢 BAIXO
**Impacto**: 🔴 ALTO (resultados válidos vs inválidos)
