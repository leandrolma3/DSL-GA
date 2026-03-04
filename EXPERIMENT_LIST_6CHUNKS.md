# LISTA DE EXPERIMENTOS - 6 CHUNKS

**Total de streams**: 59
**Configuracao**: 6 chunks x 6000 instances = 36000 instances
**Populacao**: 80 individuos

---

## Categoria: Abrupt
**Total**: 12 streams

| # | Stream Name | Dataset Type | Drift Type | Concepts | Sequence |
|---|-------------|--------------|------------|----------|----------|
| 1 | `AGRAWAL_Abrupt_Chain_Long` | AGRAWAL | abrupt | 5 | f1(1) → f2(1) → f3(1) → f4(1) → f5(2) |
| 2 | `AGRAWAL_Abrupt_Simple_Mild` | AGRAWAL | abrupt | 2 | f1(3) → f2(3) |
| 3 | `AGRAWAL_Abrupt_Simple_Severe` | AGRAWAL | abrupt | 2 | f1(3) → f6(3) |
| 4 | `HYPERPLANE_Abrupt_Simple` | HYPERPLANE | abrupt | 2 | plane1(3) → plane2(3) |
| 5 | `LED_Abrupt_Simple` | LED | abrupt | 2 | c1_no_noise(3) → c2_with_noise(3) |
| 6 | `RANDOMTREE_Abrupt_Simple` | RANDOMTREE | abrupt | 2 | tree1(3) → tree2(3) |
| 7 | `RBF_Abrupt_Severe` | RBF | abrupt | 2 | c1(3) → c2_severe(3) |
| 8 | `SEA_Abrupt_Chain` | SEA | abrupt | 3 | f1(2) → f2(2) → f4(2) |
| 9 | `SEA_Abrupt_Simple` | SEA | abrupt | 2 | f1(3) → f3(3) |
| 10 | `SINE_Abrupt_Simple` | SINE | abrupt | 2 | f1_sum(3) → f2_prod(3) |
| 11 | `STAGGER_Abrupt_Chain` | STAGGER | abrupt | 3 | f1(2) → f2(2) → f3(2) |
| 12 | `WAVEFORM_Abrupt_Simple` | WAVEFORM | abrupt | 2 | wave1(3) → wave2(3) |

---

## Categoria: Gradual
**Total**: 11 streams

| # | Stream Name | Dataset Type | Drift Type | Concepts | Sequence |
|---|-------------|--------------|------------|----------|----------|
| 1 | `AGRAWAL_Gradual_Mild_to_Severe` | AGRAWAL | gradual | 3 | f1(2) → f2(2) → f6(2) |
| 2 | `Bartosz_RandomTree_drift` | RANDOMTREE | gradual | 2 | tree1(3) → tree2(3) |
| 3 | `HYPERPLANE_Gradual_Simple` | HYPERPLANE | gradual | 2 | plane1(3) → plane2(3) |
| 4 | `LED_Gradual_Simple` | LED | gradual | 2 | c1_no_noise(3) → c2_with_noise(3) |
| 5 | `RANDOMTREE_Gradual_Simple` | RANDOMTREE | gradual | 2 | tree1(3) → tree2(3) |
| 6 | `RBF_Gradual_Moderate` | RBF | gradual | 2 | c1(3) → c3_moderate(3) |
| 7 | `RBF_Gradual_Severe` | RBF | gradual | 2 | c1(3) → c2_severe(3) |
| 8 | `SEA_Gradual_Simple_Fast` | SEA | gradual | 2 | f1(3) → f3(3) |
| 9 | `SEA_Gradual_Simple_Slow` | SEA | gradual | 2 | f1(3) → f3(3) |
| 10 | `STAGGER_Gradual_Chain` | STAGGER | gradual | 3 | f1(2) → f2(2) → f3(2) |
| 11 | `WAVEFORM_Gradual_Simple` | WAVEFORM | gradual | 2 | wave1(3) → wave2(3) |

---

## Categoria: Recurring
**Total**: 9 streams

| # | Stream Name | Dataset Type | Drift Type | Concepts | Sequence |
|---|-------------|--------------|------------|----------|----------|
| 1 | `AGRAWAL_Gradual_Recurring` | AGRAWAL | gradual | 2 | f2(2) → f7(2) → f2(2) |
| 2 | `Bartosz_Agrawal_recurring_drift` | AGRAWAL | abrupt | 2 | f1(2) → f6(2) → f1(2) |
| 3 | `RANDOMTREE_Abrupt_Recurring` | RANDOMTREE | abrupt | 2 | tree1(2) → tree2(2) → tree1(2) |
| 4 | `RBF_Abrupt_Blip` | RBF | abrupt | 2 | c1(2) → c3_moderate(1) → c1(3) |
| 5 | `RBF_Severe_Gradual_Recurrent` | RBF | gradual | 2 | c1(2) → c2_severe(2) → c1(2) |
| 6 | `SEA_Abrupt_Recurring` | SEA | abrupt | 2 | f1(2) → f3(2) → f1(2) |
| 7 | `SEA_Gradual_Recurring` | SEA | gradual | 2 | f1(2) → f4(2) → f1(2) |
| 8 | `SINE_Gradual_Recurring` | SINE | gradual | 2 | f1_sum(2) → f3_sum_alt(2) → f1_sum(2) |
| 9 | `STAGGER_Abrupt_Recurring` | STAGGER | abrupt | 2 | f1(2) → f3(2) → f1(2) |

---

## Categoria: Noise
**Total**: 10 streams

| # | Stream Name | Dataset Type | Drift Type | Concepts | Sequence |
|---|-------------|--------------|------------|----------|----------|
| 1 | `AGRAWAL_Abrupt_Simple_Severe_Noise` | AGRAWAL | abrupt | 2 | f1(3) → f6(3) |
| 2 | `AGRAWAL_Gradual_Recurring_Noise` | AGRAWAL | gradual | 2 | f2(2) → f7(2) → f2(2) |
| 3 | `Bartosz_SEA_drift_noise` | SEA | abrupt | 4 | f1(1) → f2(1) → f3(2) → f4(2) |
| 4 | `HYPERPLANE_Gradual_Noise` | HYPERPLANE | gradual | 2 | plane1(3) → plane2(3) |
| 5 | `RANDOMTREE_Gradual_Noise` | RANDOMTREE | gradual | 2 | tree1(3) → tree2(3) |
| 6 | `RBF_Abrupt_Blip_Noise` | RBF | abrupt | 2 | c1(2) → c3_moderate(1) → c1(3) |
| 7 | `RBF_Gradual_Severe_Noise` | RBF | gradual | 2 | c1(3) → c2_severe(3) |
| 8 | `SEA_Abrupt_Chain_Noise` | SEA | abrupt | 3 | f1(2) → f2(2) → f4(2) |
| 9 | `SINE_Abrupt_Recurring_Noise` | SINE | abrupt | 2 | f1_sum(2) → f2_prod(2) → f1_sum(2) |
| 10 | `STAGGER_Abrupt_Chain_Noise` | STAGGER | abrupt | 3 | f1(2) → f2(2) → f3(2) |

---

## Categoria: Stationary
**Total**: 12 streams

| # | Stream Name | Dataset Type | Drift Type | Concepts | Sequence |
|---|-------------|--------------|------------|----------|----------|
| 1 | `AGRAWAL_Stationary` | AGRAWAL | N/A | 1 | Stationary/Real |
| 2 | `AssetNegotiation_F2` | ASSETNEGOTIATION | N/A | 1 | Stationary/Real |
| 3 | `AssetNegotiation_F3` | ASSETNEGOTIATION | N/A | 1 | Stationary/Real |
| 4 | `AssetNegotiation_F4` | ASSETNEGOTIATION | N/A | 1 | Stationary/Real |
| 5 | `HYPERPLANE_Stationary` | HYPERPLANE | N/A | 1 | Stationary/Real |
| 6 | `LED_Stationary` | LED | N/A | 1 | Stationary/Real |
| 7 | `RANDOMTREE_Stationary` | RANDOMTREE | N/A | 1 | Stationary/Real |
| 8 | `RBF_Stationary` | RBF | N/A | 1 | Stationary/Real |
| 9 | `SEA_Stationary` | SEA | N/A | 1 | Stationary/Real |
| 10 | `SINE_Stationary` | SINE | N/A | 1 | Stationary/Real |
| 11 | `STAGGER_Stationary` | STAGGER | N/A | 1 | Stationary/Real |
| 12 | `WAVEFORM_Stationary` | WAVEFORM | N/A | 1 | Stationary/Real |

---

## Categoria: Real Datasets
**Total**: 5 streams

| # | Stream Name | Dataset Type | Drift Type | Concepts | Sequence |
|---|-------------|--------------|------------|----------|----------|
| 1 | `CovType` | COVERTYPE | N/A | 1 | Stationary/Real |
| 2 | `Electricity` | ELECTRICITY | N/A | 1 | Stationary/Real |
| 3 | `IntelLabSensors` | INTELLABSENSORS | N/A | 1 | Stationary/Real |
| 4 | `PokerHand` | POKER | N/A | 1 | Stationary/Real |
| 5 | `Shuttle` | SHUTTLE | N/A | 1 | Stationary/Real |

---

## Estatisticas

- **Total de streams**: 59
- **Drift simulations**: 42
- **Real/Stationary**: 17

- **Abrupt**: 12 (20.3%)
- **Gradual**: 11 (18.6%)
- **Recurring**: 9 (15.3%)
- **Noise**: 10 (16.9%)
- **Stationary**: 12 (20.3%)
- **Real Datasets**: 5 (8.5%)

---

**Gerado por**: adjust_config_for_mass_experiments.py
**Data**: 2025-10-28