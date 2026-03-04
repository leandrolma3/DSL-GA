# Fase 3 - Batch 5 Pronto para Execução

**Data:** 2025-11-21
**Status:** CONFIG CRIADO - PRONTO PARA TESTAR

---

## O Que Foi Feito

### 1. Planejamento Completo ✅
- Documento `PLANO_FASE3_DATASETS_REAIS.md` criado
- 17 datasets identificados (5 reais + 12 sintéticos estacionários)
- Organização em 3 batches definida
- Estratégia de execução planejada

### 2. Verificação de Datasets ✅
- Datasets CSV verificados e disponíveis:
  - ✅ covertype_processed.csv (75.7 MB)
  - ✅ intellabsensors_processed.csv (47.8 MB)
  - ✅ poker_processed.csv (0.6 MB)
  - ✅ shuttle_processed.csv (1.2 MB)
- Dataset River disponível:
  - ✅ Electricity (river.datasets.Elec2)

### 3. Config Batch 5 Criado ✅
**Arquivo:** `configs/config_batch_5.yaml` (3.4 KB)

**Principais Características:**
- `run_mode: standard` (modo estacionário/real)
- 5 datasets reais listados:
  1. Electricity
  2. Shuttle
  3. CovType
  4. PokerHand
  5. IntelLabSensors
- Diretório de resultados: `experiments_6chunks_phase3_real/batch_5`
- Parâmetros GA, memory, fitness mantidos da Fase 2
- 6 chunks, 1000 pontos cada = train-then-test em 5 chunks

---

## Estrutura do Config Batch 5

```yaml
experiment_settings:
  run_mode: standard                    # <-- MUDANÇA PRINCIPAL
  standard_experiments:                  # <-- LISTA DE DATASETS REAIS
    - Electricity                        # River dataset
    - Shuttle                            # CSV local
    - CovType                            # CSV local
    - PokerHand                          # CSV local
    - IntelLabSensors                    # CSV local
  base_results_dir: .../batch_5         # <-- Fase 3

data_params:
  chunk_size: 1000                       # 1000 pontos por chunk
  num_chunks: 6                          # 6 chunks total
  max_instances: 24000                   # Máximo

# Todos os parâmetros GA, memory, fitness mantidos da Fase 2
ga_params:
  population_size: 120
  max_generations: 200
  # ... (todos os outros parâmetros)
```

---

## Diferenças: Fase 2 vs Fase 3

| Aspecto | Fase 2 (Drift Simulation) | Fase 3 (Standard/Real) |
|---------|---------------------------|------------------------|
| **run_mode** | drift_simulation | standard |
| **Datasets** | 32 sintéticos com drift artificial | 17 reais/estacionários |
| **Drift** | Injetado artificialmente | Natural ou ausente |
| **Objetivo** | Avaliar adaptação a drift | Avaliar aprendizado puro |
| **Diretório** | experiments_6chunks_phase2_gbml | experiments_6chunks_phase3_real |

---

## Como o Modo Standard Funciona

### Train-Then-Test com 6 Chunks

```
Chunk 0 [0-999]:     TREINO INICIAL
  ↓
Chunk 1 [1000-1999]: TESTE 1 → Avaliação
  ↓
Chunk 2 [2000-2999]: TESTE 2 → Avaliação
  ↓
Chunk 3 [3000-3999]: TESTE 3 → Avaliação
  ↓
Chunk 4 [4000-4999]: TESTE 4 → Avaliação
  ↓
Chunk 5 [5000-5999]: TESTE 5 → Avaliação
```

**Total:** 5 avaliações por dataset

### Diferença do Drift Simulation
- **Drift Simulation:** Injeta drifts em posições específicas, ativa mecanismos de recuperação
- **Standard:** Usa dados como estão, sem injeção de drift, sem recuperação ativa
- **Vantagem Standard:** Comparação mais justa com baselines sem adaptação específica

---

## Próximos Passos

### Passo 1: Testar com 1 Dataset (Electricity)
**Objetivo:** Validar que o modo standard funciona corretamente

**Comando:**
```bash
cd C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid
python main.py --config configs/config_batch_5.yaml
```

**O que observar:**
- ✅ Carregamento do dataset Electricity
- ✅ Criação de 6 chunks
- ✅ Treino no chunk 0
- ✅ 5 avaliações (chunks 1-5)
- ✅ Métricas salvas corretamente
- ⚠️ Verificar se não há erros de "drift detection" (não deve ter)

**Tempo estimado:** 30-60 minutos

### Passo 2: Executar Batch 5 Completo (GBML Apenas)
**Objetivo:** Executar GBML nos 5 datasets reais

**Opção A - Sequencial:**
```bash
python main.py --config configs/config_batch_5.yaml
```
Executa os 5 datasets um após o outro.

**Opção B - Modificar config para 1 dataset por vez:**
```yaml
standard_experiments:
  - Electricity  # Só Electricity
```
Repetir para cada dataset.

**Tempo estimado:** 2-4 horas (dependendo da complexidade)

### Passo 3: Executar Todos os Modelos (Batch 5)
**Objetivo:** Executar GBML + 5 baselines

**Opções:**

**A) Usar notebook comparativo (se disponível):**
```python
# Similar ao usado na Fase 2
# Executar GBML + ACDWM + ARF + SRP + HAT + ERulesD2S
```

**B) Executar script run_comparative_on_existing_chunks.py:**
```bash
python run_comparative_on_existing_chunks.py --config configs/config_batch_5.yaml
```

**C) Executar manualmente cada modelo:**
```bash
# GBML (já executado)
python main.py --config configs/config_batch_5.yaml

# Baselines (se tiver scripts individuais)
python baseline_acdwm.py --config configs/config_batch_5.yaml
python baseline_river.py --config configs/config_batch_5.yaml --model ARF
python baseline_river.py --config configs/config_batch_5.yaml --model SRP
python baseline_river.py --config configs/config_batch_5.yaml --model HAT
python run_erulesd2s_only.py --config configs/config_batch_5.yaml
```

**Tempo estimado:** 8-12 horas total

### Passo 4: Verificar Resultados
**Objetivo:** Validar que os resultados foram salvos corretamente

**Diretório esperado:**
```
experiments_6chunks_phase3_real/batch_5/
├── Electricity/
│   └── run_1/
│       ├── chunk_metrics.json
│       ├── final_metrics.json
│       └── ...
├── Shuttle/
├── CovType/
├── PokerHand/
└── IntelLabSensors/
```

**Verificação:**
```bash
cd experiments_6chunks_phase3_real/batch_5
find . -name "chunk_metrics.json" | wc -l  # Deve mostrar 5
find . -name "final_metrics.json" | wc -l  # Deve mostrar 5
```

### Passo 5: Pós-Processamento
**Objetivo:** Consolidar resultados e gerar análises

**Scripts a executar:**
```bash
# Consolidar resultados de todos os modelos
python post_process_batch_5.py

# Gerar plots
python generate_plots.py --batch 5

# Análise comparativa
python analyze_complete_results.py --batch 5
```

**Nota:** Pode ser necessário criar/adaptar scripts de pós-processamento

---

## Checklist de Execução

### Preparação
- [X] Plano criado (PLANO_FASE3_DATASETS_REAIS.md)
- [X] Datasets CSV verificados
- [X] config_batch_5.yaml criado
- [ ] Testar com Electricity apenas

### Execução GBML
- [ ] Executar Batch 5 com GBML
- [ ] Verificar logs
- [ ] Confirmar 5 datasets processados
- [ ] Verificar chunk_metrics.json salvos

### Execução Baselines
- [ ] ACDWM executado
- [ ] ARF executado
- [ ] SRP executado
- [ ] HAT executado
- [ ] ERulesD2S executado

### Pós-Processamento
- [ ] Resultados consolidados
- [ ] Plots gerados
- [ ] Análise estatística
- [ ] Comparação com Fase 2

---

## Problemas Potenciais e Soluções

### Problema 1: Datasets CSV Não Encontrados
**Sintoma:** Erro "File not found: datasets/processed/..."

**Solução:**
1. Verificar se os arquivos existem:
   ```bash
   ls -lh datasets/processed/*.csv
   ```
2. Se não existirem, executar scripts de pré-processamento
3. Ou ajustar paths no config

### Problema 2: Modo Standard Não Reconhecido
**Sintoma:** Erro "Invalid run_mode: standard"

**Solução:**
1. Verificar versão do código
2. Pode ser que modo standard não esteja implementado
3. Verificar em `main.py` se há suporte para `run_mode: standard`

### Problema 3: Adaptação a Drift Ativa no Modo Standard
**Sintoma:** Logs mostram "Drift detected" em modo standard

**Solução:**
1. Isso é esperado se o dataset real tiver drift natural
2. Não é um problema, apenas uma observação
3. GBML pode adaptar se necessário

### Problema 4: Performance Muito Alta
**Sintoma:** G-mean > 0.95 em datasets estacionários

**Solução:**
1. Isso é esperado! Datasets estacionários são mais fáceis
2. Sem drift para se adaptar, modelos podem atingir alta performance
3. Documentar e comparar com Fase 2

---

## Expectativas de Resultados

### Performance Esperada

**Fase 2 (Com Drift):**
- GBML: 0.7775
- ARF: 0.7240
- ACDWM: 0.6998

**Fase 3 (Sem Drift - Expectativa):**
- Todos os modelos: Performance MAIOR
- Razão: Sem drift para se adaptar
- GBML: Pode ser ~0.85-0.90 (estimativa)
- Baselines: Também devem melhorar

### Questões de Pesquisa

1. **GBML mantém vantagem sem drift?**
   - Fase 2: GBML melhor com drift
   - Fase 3: GBML melhor sem drift?

2. **Qual modelo é melhor em cenários estacionários?**
   - Modelos ensemble (ARF, SRP) podem se destacar
   - GBML pode ter overhead de GA desnecessário

3. **Datasets reais têm drift natural?**
   - Electricity: Provavelmente sim (variação temporal)
   - Shuttle, CovType, PokerHand: Provavelmente não
   - Performance vai revelar

---

## Estrutura Final Esperada

```
DSL-AG-hybrid/
├── configs/
│   ├── config_batch_1.yaml  # Fase 2 - Batch 1
│   ├── config_batch_2.yaml  # Fase 2 - Batch 2
│   ├── config_batch_3.yaml  # Fase 2 - Batch 3
│   ├── config_batch_4.yaml  # Fase 2 - Batch 4
│   ├── config_batch_5.yaml  # Fase 3 - Batch 5 (REAIS) ✅ CRIADO
│   ├── config_batch_6.yaml  # Fase 3 - Batch 6 (Estacionários 1) - A CRIAR
│   └── config_batch_7.yaml  # Fase 3 - Batch 7 (Estacionários 2) - A CRIAR
│
├── experiments_6chunks_phase2_gbml/  # Fase 2 completa
│   ├── batch_1/ (12 datasets)
│   ├── batch_2/ (9 datasets)
│   ├── batch_3/ (7 datasets)
│   └── batch_4/ (4 datasets)
│
└── experiments_6chunks_phase3_real/  # Fase 3 (em progresso)
    ├── batch_5/  # 5 reais (em execução)
    ├── batch_6/  # 6 estacionários (pendente)
    └── batch_7/  # 6 estacionários (pendente)
```

---

## Resumo Executivo

### O Que Temos Agora
✅ **Config criado:** `configs/config_batch_5.yaml`
✅ **Datasets verificados:** 5 datasets reais disponíveis
✅ **Plano completo:** PLANO_FASE3_DATASETS_REAIS.md
✅ **Pronto para executar:** Apenas rodar `python main.py --config configs/config_batch_5.yaml`

### Próxima Ação Imediata
**Testar com Electricity:**
```bash
# Modificar config para apenas Electricity
# Executar
python main.py --config configs/config_batch_5.yaml
# Verificar resultados
# Se OK, executar com todos os 5 datasets
```

### Timeline Estimado
- **Teste Electricity:** 30-60 min
- **Batch 5 GBML:** 2-4 horas
- **Batch 5 Todos modelos:** 8-12 horas
- **Pós-processamento:** 1-2 horas
- **Total Batch 5:** ~1 dia de trabalho

---

**Status Atual:** PRONTO PARA INICIAR TESTES
**Próximo Passo:** Executar teste com Electricity
