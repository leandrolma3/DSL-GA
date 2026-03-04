# Índice de Documentação - Fase 3

**Data:** 2025-11-21
**Status:** PRONTO PARA EXECUÇÃO

---

## Documentação Principal da Fase 3

### 1. **DOCUMENTACAO_FASE3_COMPLETA.md** ⭐ DOCUMENTO PRINCIPAL
**O que contém:**
- Visão geral completa da Fase 3
- Diferenças entre Fase 2 e Fase 3
- Estrutura dos 3 batches (5, 6, 7)
- Detalhes dos configs criados
- Guia completo de execução
- Estrutura de resultados esperados
- Expectativas e hipóteses de pesquisa
- Checklist completo de execução
- Troubleshooting detalhado
- Guia de pós-processamento

**Quando usar:** Principal referência para toda a Fase 3

---

### 2. **PLANO_FASE3_DATASETS_REAIS.md**
**O que contém:**
- Planejamento inicial da Fase 3
- Lista de 17 datasets (5 reais + 12 sintéticos)
- Organização em 3 batches
- Estratégia de execução
- Justificativa para cada batch

**Quando usar:** Entender o planejamento e organização da Fase 3

---

### 3. **FASE3_BATCH5_PRONTO.md**
**O que contém:**
- Foco específico no Batch 5 (datasets reais)
- Verificação de datasets CSV disponíveis
- Detalhes do config_batch_5.yaml
- Guia passo-a-passo para Batch 5
- Checklist específico do Batch 5

**Quando usar:** Executar especificamente o Batch 5

---

## Configs Criados

### 4. **configs/config_batch_5.yaml** (3.4 KB)
**Datasets:** 5 reais
- Electricity
- Shuttle
- CovType
- PokerHand
- IntelLabSensors

**run_mode:** standard
**Diretório:** experiments_6chunks_phase3_real/batch_5

---

### 5. **configs/config_batch_6.yaml** (3.5 KB)
**Datasets:** 6 sintéticos estacionários (parte 1)
- SEA_Stationary
- AGRAWAL_Stationary
- RBF_Stationary
- LED_Stationary
- HYPERPLANE_Stationary
- RANDOMTREE_Stationary

**run_mode:** standard
**Diretório:** experiments_6chunks_phase3_real/batch_6

---

### 6. **configs/config_batch_7.yaml** (3.7 KB)
**Datasets:** 6 sintéticos estacionários (parte 2)
- STAGGER_Stationary
- WAVEFORM_Stationary
- SINE_Stationary
- AssetNegotiation_F2
- AssetNegotiation_F3
- AssetNegotiation_F4

**run_mode:** standard
**Diretório:** experiments_6chunks_phase3_real/batch_7

---

## Organização por Propósito

### Para Entender a Fase 3 Completa
1. **DOCUMENTACAO_FASE3_COMPLETA.md** ⭐ (leia primeiro)
2. **PLANO_FASE3_DATASETS_REAIS.md**
3. **FASE3_BATCH5_PRONTO.md**

### Para Executar os Experimentos
1. **DOCUMENTACAO_FASE3_COMPLETA.md** (seção "Como Executar")
2. **configs/config_batch_5.yaml** (Batch 5)
3. **configs/config_batch_6.yaml** (Batch 6)
4. **configs/config_batch_7.yaml** (Batch 7)

### Para Entender as Diferenças da Fase 2
1. **DOCUMENTACAO_FASE3_COMPLETA.md** (seção "Diferenças")
2. Comparar configs Fase 2 (config_batch_1.yaml) vs Fase 3 (config_batch_5.yaml)

---

## Estrutura Completa de Documentação

```
DSL-AG-hybrid/
├── DOCUMENTACAO_FASE3_COMPLETA.md      ⭐ Documento principal Fase 3
├── INDICE_DOCUMENTACAO_FASE3.md        Este arquivo
├── PLANO_FASE3_DATASETS_REAIS.md       Planejamento da Fase 3
├── FASE3_BATCH5_PRONTO.md              Guia Batch 5
│
├── configs/
│   ├── config_batch_1.yaml             Fase 2 - Abrupt (12 datasets)
│   ├── config_batch_2.yaml             Fase 2 - Gradual (9 datasets)
│   ├── config_batch_3.yaml             Fase 2 - Noise (7 datasets)
│   ├── config_batch_4.yaml             Fase 2 - Multiclass (4 datasets)
│   ├── config_batch_5.yaml             ⭐ Fase 3 - Reais (5 datasets)
│   ├── config_batch_6.yaml             ⭐ Fase 3 - Estacionários 1 (6 datasets)
│   └── config_batch_7.yaml             ⭐ Fase 3 - Estacionários 2 (6 datasets)
│
├── experiments_6chunks_phase2_gbml/    Resultados Fase 2 (completa)
│   ├── batch_1/ (12 datasets)
│   ├── batch_2/ (9 datasets)
│   ├── batch_3/ (7 datasets)
│   └── batch_4/ (4 datasets)
│
└── experiments_6chunks_phase3_real/    Resultados Fase 3 (a executar)
    ├── batch_5/ (5 datasets reais)
    ├── batch_6/ (6 estacionários)
    └── batch_7/ (6 estacionários)
```

---

## Conexão com paperLu

### Documentação no paperLu
O diretório `paperLu` contém:
- Paper LaTeX completo (Fase 2)
- Dados consolidados da Fase 2 (32 datasets)
- Tabelas e figuras geradas
- Documentação de correções aplicadas

**Arquivo de conexão:** `C:\Users\Leandro Almeida\Downloads\paperLu\CONEXAO_COM_DSL_AG_HYBRID.md`

### Fluxo Fase 2 → Fase 3 → Paper Final

```
Fase 2 (DSL-AG-hybrid)          Fase 3 (DSL-AG-hybrid)          Paper Final (paperLu)
├── 32 datasets drift     →     ├── 17 datasets reais      →    ├── Análise comparativa
├── 4 batches completos         ├── 3 batches a executar         ├── Tabelas consolidadas
├── Resultados em               ├── configs criados              ├── Figuras comparativas
│   experiments_phase2/         │   batch_5, 6, 7                ├── Discussão Fase 2 vs 3
└── Logs completos              └── Pronto para executar         └── Paper final atualizado
```

---

## Quick Start - Fase 3

### 1. Verificar Pré-requisitos
```bash
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"

# Verificar datasets CSV
ls -lh datasets/processed/*.csv

# Verificar configs
ls -lh configs/config_batch_*.yaml
```

### 2. Teste Rápido (Electricity)
```bash
# Modificar config_batch_5.yaml para apenas Electricity
# Executar
python main.py --config configs/config_batch_5.yaml
```

### 3. Executar Batch 5 Completo
```bash
# Restaurar lista completa de datasets no config
python main.py --config configs/config_batch_5.yaml
```

### 4. Executar Batches 6 e 7
```bash
python main.py --config configs/config_batch_6.yaml
python main.py --config configs/config_batch_7.yaml
```

### 5. Executar Baselines
```bash
# Para cada batch
python run_comparative_on_existing_chunks.py --config configs/config_batch_5.yaml
python run_comparative_on_existing_chunks.py --config configs/config_batch_6.yaml
python run_comparative_on_existing_chunks.py --config configs/config_batch_7.yaml
```

---

## Checklist Rápido

### Preparação
- [X] Configs criados (batch_5, 6, 7)
- [X] Documentação completa criada
- [X] Índice de documentação criado
- [ ] Datasets CSV verificados
- [ ] Teste com Electricity executado

### Execução
- [ ] Batch 5 GBML completo
- [ ] Batch 6 GBML completo
- [ ] Batch 7 GBML completo
- [ ] Batch 5 baselines
- [ ] Batch 6 baselines
- [ ] Batch 7 baselines

### Análise
- [ ] Resultados consolidados
- [ ] Testes estatísticos
- [ ] Comparação Fase 2 vs Fase 3
- [ ] Paper atualizado com Fase 3

---

## Perguntas Frequentes

### Q1: Por que criar Fase 3 se Fase 2 já está completa?
**A:** A Fase 2 avaliou adaptação a drift. A Fase 3 avalia desempenho puro sem drift, para entender se GBML mantém vantagem em cenários mais simples.

### Q2: Quantos datasets serão testados na Fase 3?
**A:** 17 datasets (5 reais + 12 sintéticos estacionários)

### Q3: Os parâmetros são os mesmos da Fase 2?
**A:** Sim, todos os parâmetros GA, memory e fitness são idênticos. Apenas run_mode mudou.

### Q4: Quanto tempo vai levar?
**A:** Estimado 2-3 dias de trabalho total (incluindo baselines e pós-processamento)

### Q5: ACDWM vai falhar novamente?
**A:** Possivelmente sim, em datasets multiclass como LED_Stationary e WAVEFORM_Stationary.

### Q6: Como comparar com Fase 2?
**A:** Após execução, consolidar resultados e gerar tabela comparativa com médias de ambas as fases.

---

## Resumo Executivo

### Status Atual
✅ **Fase 2:** COMPLETA (32 datasets, 4 batches)
✅ **Fase 3 - Configs:** CRIADOS (17 datasets, 3 batches)
✅ **Fase 3 - Documentação:** COMPLETA
⏳ **Fase 3 - Execução:** PENDENTE

### Próxima Ação
**Imediata:** Testar com Electricity
**Sequência:** Batch 5 → Batch 6 → Batch 7 → Baselines → Análise

### Timeline
- **Teste:** 30-60 min
- **GBML completo:** 6-10 horas
- **Todos modelos:** 24-36 horas
- **Análise:** 2-4 horas
- **Total:** 2-3 dias

---

**Última Atualização:** 2025-11-21
**Status:** PRONTO PARA EXECUÇÃO
