# Experimento: Penalidade de Complexidade Aumentada (gamma=0.3)

**Data de Início**: 2026-02-06
**Status**: Em Execução

---

## 1. Objetivo do Experimento

Avaliar o impacto do aumento da penalidade de complexidade (gamma) de 0.1 para 0.3 (3x maior) na evolução de regras do algoritmo DSL-AG.

### Hipótese Científica

Com gamma=0.3, espera-se:
- Regras mais simples (menos condições por regra)
- Menor número total de regras
- Possível leve redução no G-Mean (trade-off complexidade vs. performance)

### Pergunta de Pesquisa

> Qual é o trade-off entre complexidade e performance quando gamma aumenta de 0.1 para 0.3?

---

## 2. Configuração dos Experimentos

### 2.1 Arquivos YAML Criados

| Arquivo | Datasets | Localização |
|---------|----------|-------------|
| `config_unified_chunk500_penalty03_batch_1.yaml` | 18 | `configs/` |
| `config_unified_chunk500_penalty03_batch_2.yaml` | 17 | `configs/` |
| `config_unified_chunk500_penalty03_batch_3.yaml` | 17 | `configs/` |

### 2.2 Parâmetros Principais

| Parâmetro | Valor |
|-----------|-------|
| **gamma** | **0.3** (era 0.1) |
| chunk_size | 500 |
| num_chunks | 24 |
| max_instances | 12000 |
| population_size | 120 |
| max_generations | 200 |
| resume_from_checkpoint | true |

### 2.3 Diretório de Resultados

```
experiments_unified/
└── chunk_500_penalty_03/
    ├── batch_1/
    ├── batch_2/
    └── batch_3/
```

### 2.4 Diretório de Chunks Pré-gerados

```
/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/unified_chunks/chunk_500
```

---

## 3. Datasets por Batch

### Batch 1 (18 datasets) - Abrupt + Gradual Drift

**Abrupt (12):**
- SEA_Abrupt_Simple
- SEA_Abrupt_Chain
- SEA_Abrupt_Recurring
- AGRAWAL_Abrupt_Simple_Mild
- AGRAWAL_Abrupt_Simple_Severe
- AGRAWAL_Abrupt_Chain_Long
- RBF_Abrupt_Severe
- RBF_Abrupt_Blip
- STAGGER_Abrupt_Chain
- STAGGER_Abrupt_Recurring
- HYPERPLANE_Abrupt_Simple
- RANDOMTREE_Abrupt_Simple

**Gradual (6):**
- SEA_Gradual_Simple_Fast
- SEA_Gradual_Simple_Slow
- SEA_Gradual_Recurring
- STAGGER_Gradual_Chain
- RBF_Gradual_Moderate
- RBF_Gradual_Severe

### Batch 2 (17 datasets) - Gradual + Noise + Mixed

**Gradual (3):**
- HYPERPLANE_Gradual_Simple
- RANDOMTREE_Gradual_Simple
- LED_Gradual_Simple

**Noise (8):**
- SEA_Abrupt_Chain_Noise
- STAGGER_Abrupt_Chain_Noise
- AGRAWAL_Abrupt_Simple_Severe_Noise
- SINE_Abrupt_Recurring_Noise
- RBF_Abrupt_Blip_Noise
- RBF_Gradual_Severe_Noise
- HYPERPLANE_Gradual_Noise
- RANDOMTREE_Gradual_Noise

**Mixed (6):**
- SINE_Abrupt_Simple
- SINE_Gradual_Recurring
- LED_Abrupt_Simple
- WAVEFORM_Abrupt_Simple
- WAVEFORM_Gradual_Simple
- RANDOMTREE_Abrupt_Recurring

### Batch 3 (17 datasets) - Real-world + Stationary

**Real-world (5):**
- Electricity
- Shuttle
- CovType
- PokerHand
- IntelLabSensors

**Stationary (9):**
- SEA_Stationary
- AGRAWAL_Stationary
- RBF_Stationary
- LED_Stationary
- HYPERPLANE_Stationary
- RANDOMTREE_Stationary
- STAGGER_Stationary
- WAVEFORM_Stationary
- SINE_Stationary

**AssetNegotiation (3):**
- AssetNegotiation_F2
- AssetNegotiation_F3
- AssetNegotiation_F4

---

## 4. Comparação com Experimentos Base (gamma=0.1)

### 4.1 Arquivos de Referência

| gamma=0.1 (base) | gamma=0.3 (novo) |
|------------------|------------------|
| `config_unified_chunk500_penalty_batch_1.yaml` | `config_unified_chunk500_penalty03_batch_1.yaml` |
| `config_unified_chunk500_penalty_batch_2.yaml` | `config_unified_chunk500_penalty03_batch_2.yaml` |
| `config_unified_chunk500_penalty_batch_3.yaml` | `config_unified_chunk500_penalty03_batch_3.yaml` |

### 4.2 Diretórios de Resultados

| gamma | Diretório |
|-------|-----------|
| 0.1 | `experiments_unified/chunk_500_penalty/` |
| 0.3 | `experiments_unified/chunk_500_penalty_03/` |

### 4.3 Métricas Esperadas

| Métrica | gamma=0.1 (estimado) | gamma=0.3 (esperado) |
|---------|----------------------|----------------------|
| Avg Rules | ~15-24 | ~8-15 (↓ 30-50%) |
| Conditions/Rule | ~4.8-5.8 | ~3-4 (↓ 25-40%) |
| G-Mean | ~0.80-0.86 | ~0.75-0.82 (↓ leve) |

---

## 5. Comandos para Execução

### Executar Batch 1
```bash
python main.py --config configs/config_unified_chunk500_penalty03_batch_1.yaml
```

### Executar Batch 2
```bash
python main.py --config configs/config_unified_chunk500_penalty03_batch_2.yaml
```

### Executar Batch 3
```bash
python main.py --config configs/config_unified_chunk500_penalty03_batch_3.yaml
```

### Verificar Progresso
```bash
# Listar experimentos concluídos
ls -la experiments_unified/chunk_500_penalty_03/batch_1/

# Contar datasets processados
find experiments_unified/chunk_500_penalty_03/ -name "metrics.json" | wc -l
```

---

## 6. Retomada de Experimentos

Os experimentos podem ser retomados automaticamente graças ao parâmetro:
```yaml
resume_from_checkpoint: true
```

Para retomar, basta executar o mesmo comando. O sistema detectará checkpoints existentes e continuará de onde parou.

---

## 7. Análise Pós-Execução

### 7.1 Scripts de Análise Disponíveis

| Script | Função |
|--------|--------|
| `collect_results_for_paper.py` | Coleta métricas para tabelas do paper |
| `analyze_experiment_detailed.py` | Análise detalhada por dataset |
| `generate_comparison_tables.py` | Gera tabelas comparativas |
| `statistical_analysis.py` | Testes estatísticos |

### 7.2 Comparação gamma=0.1 vs gamma=0.3

Após conclusão, executar:
```bash
# Coletar resultados gamma=0.3
python collect_results_for_paper.py --results_dir experiments_unified/chunk_500_penalty_03

# Comparar com gamma=0.1
python generate_comparison_tables.py \
    --baseline experiments_unified/chunk_500_penalty \
    --experiment experiments_unified/chunk_500_penalty_03 \
    --output paper/tables/
```

---

## 8. Checklist de Progresso

### Criação dos YAMLs
- [x] `config_unified_chunk500_penalty03_batch_1.yaml` criado
- [x] `config_unified_chunk500_penalty03_batch_2.yaml` criado
- [x] `config_unified_chunk500_penalty03_batch_3.yaml` criado
- [x] Todos com gamma=0.3
- [x] Todos com base_results_dir correto
- [x] 52 datasets no total (18+17+17)

### Execução
- [ ] Batch 1 concluído (0/18 datasets)
- [ ] Batch 2 concluído (0/17 datasets)
- [ ] Batch 3 concluído (0/17 datasets)

### Análise
- [ ] Métricas coletadas
- [ ] Tabelas comparativas geradas
- [ ] Testes estatísticos executados
- [ ] Conclusões documentadas

---

## 9. Notas e Observações

### 2026-02-06
- Criados os 3 arquivos YAML para experimentos com gamma=0.3
- Todos os 52 datasets estão ativos (nenhum comentado)
- Experimentos iniciados em execução no Colab

---

## 10. Referências

- Plano original: `PLANO_EXPERIMENTO_PENALTY_03.md` (se existir)
- Configurações base: `configs/config_unified_chunk500_penalty_batch_*.yaml`
- Documentação do GA: `MCMO_API_DOCUMENTATION.md`
