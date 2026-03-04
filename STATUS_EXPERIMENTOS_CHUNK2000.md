# STATUS DOS EXPERIMENTOS - CHUNK SIZE 2000

**Data de Verificacao**: 2025-12-15
**Objetivo**: Identificar datasets que nao foram executados

---

## RESUMO EXECUTIVO

| Batch | Esperados | Executados | Faltando | Status |
|-------|-----------|------------|----------|--------|
| Batch 1 | 12 | 10 | 2 | INCOMPLETO |
| Batch 2 | 9 | 9 | 0 | COMPLETO |
| Batch 3 | 8 | 8 | 0 | COMPLETO |
| Batch 4 | 6 | 5 | 1 | INCOMPLETO |
| Batch 5 | 5 | 4 | 1 | INCOMPLETO |
| Batch 6 | 6 | 6 | 0 | COMPLETO |
| Batch 7 | 6 | 6 | 0 | COMPLETO |
| **TOTAL** | **52** | **48** | **4** | **92% completo** |

---

## DATASETS FALTANTES (4 no total)

### Batch 1 - Phase 1 (drift_simulation)
| Dataset | Status |
|---------|--------|
| STAGGER_Abrupt_Chain | NAO EXECUTADO |
| STAGGER_Abrupt_Recurring | NAO EXECUTADO |

### Batch 4 - Phase 1 (drift_simulation)
| Dataset | Status |
|---------|--------|
| RANDOMTREE_Abrupt_Recurring | NAO EXECUTADO |

### Batch 5 - Phase 2 (standard - datasets reais)
| Dataset | Status |
|---------|--------|
| IntelLabSensors | NAO EXECUTADO |

---

## DETALHAMENTO POR BATCH

### BATCH 1 (experiments_chunk2000_phase1/batch_1)
**Tipo**: drift_simulation (drifts abruptos)

| Dataset | Executado |
|---------|-----------|
| SEA_Abrupt_Simple | SIM |
| SEA_Abrupt_Chain | SIM |
| SEA_Abrupt_Recurring | SIM |
| AGRAWAL_Abrupt_Simple_Mild | SIM |
| AGRAWAL_Abrupt_Simple_Severe | SIM |
| AGRAWAL_Abrupt_Chain_Long | SIM |
| RBF_Abrupt_Severe | SIM |
| RBF_Abrupt_Blip | SIM |
| STAGGER_Abrupt_Chain | **NAO** |
| STAGGER_Abrupt_Recurring | **NAO** |
| HYPERPLANE_Abrupt_Simple | SIM |
| RANDOMTREE_Abrupt_Simple | SIM |

---

### BATCH 2 (experiments_chunk2000_phase1/batch_2)
**Tipo**: drift_simulation (drifts graduais)
**Status**: COMPLETO (9/9)

| Dataset | Executado |
|---------|-----------|
| SEA_Gradual_Simple_Fast | SIM |
| SEA_Gradual_Simple_Slow | SIM |
| SEA_Gradual_Recurring | SIM |
| STAGGER_Gradual_Chain | SIM |
| RBF_Gradual_Moderate | SIM |
| RBF_Gradual_Severe | SIM |
| HYPERPLANE_Gradual_Simple | SIM |
| RANDOMTREE_Gradual_Simple | SIM |
| LED_Gradual_Simple | SIM |

---

### BATCH 3 (experiments_chunk2000_phase1/batch_3)
**Tipo**: drift_simulation (com ruido)
**Status**: COMPLETO (8/8)

| Dataset | Executado |
|---------|-----------|
| SEA_Abrupt_Chain_Noise | SIM |
| STAGGER_Abrupt_Chain_Noise | SIM |
| AGRAWAL_Abrupt_Simple_Severe_Noise | SIM |
| SINE_Abrupt_Recurring_Noise | SIM |
| RBF_Abrupt_Blip_Noise | SIM |
| RBF_Gradual_Severe_Noise | SIM |
| HYPERPLANE_Gradual_Noise | SIM |
| RANDOMTREE_Gradual_Noise | SIM |

---

### BATCH 4 (experiments_chunk2000_phase1/batch_4)
**Tipo**: drift_simulation (SINE, LED, WAVEFORM)

| Dataset | Executado |
|---------|-----------|
| SINE_Abrupt_Simple | SIM |
| SINE_Gradual_Recurring | SIM |
| LED_Abrupt_Simple | SIM |
| WAVEFORM_Abrupt_Simple | SIM |
| WAVEFORM_Gradual_Simple | SIM |
| RANDOMTREE_Abrupt_Recurring | **NAO** |

---

### BATCH 5 (experiments_chunk2000_phase2/batch_5)
**Tipo**: standard (datasets reais)

| Dataset | Executado |
|---------|-----------|
| Electricity | SIM |
| Shuttle | SIM |
| CovType | SIM |
| PokerHand | SIM |
| IntelLabSensors | **NAO** |

---

### BATCH 6 (experiments_chunk2000_phase2/batch_6)
**Tipo**: standard (sinteticos estacionarios - parte 1)
**Status**: COMPLETO (6/6)

| Dataset | Executado |
|---------|-----------|
| SEA_Stationary | SIM |
| AGRAWAL_Stationary | SIM |
| RBF_Stationary | SIM |
| LED_Stationary | SIM |
| HYPERPLANE_Stationary | SIM |
| RANDOMTREE_Stationary | SIM |

---

### BATCH 7 (experiments_chunk2000_phase2/batch_7)
**Tipo**: standard (sinteticos estacionarios - parte 2)
**Status**: COMPLETO (6/6)

| Dataset | Executado |
|---------|-----------|
| STAGGER_Stationary | SIM |
| WAVEFORM_Stationary | SIM |
| SINE_Stationary | SIM |
| AssetNegotiation_F2 | SIM |
| AssetNegotiation_F3 | SIM |
| AssetNegotiation_F4 | SIM |

---

## ACAO REQUERIDA

Para completar os experimentos, e necessario criar YAMLs especificos para re-executar os 4 datasets faltantes:

1. **config_chunk2000_batch_1_rerun.yaml** - Para STAGGER_Abrupt_Chain e STAGGER_Abrupt_Recurring
2. **config_chunk2000_batch_4_rerun.yaml** - Para RANDOMTREE_Abrupt_Recurring
3. **config_chunk2000_batch_5_rerun.yaml** - Para IntelLabSensors

---

**Autor**: Claude Code
**Data**: 2025-12-15
