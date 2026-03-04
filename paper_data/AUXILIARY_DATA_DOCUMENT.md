# Auxiliary Data Document for IEEE TKDE Paper

Generated: 2026-02-27 15:15:41

**Source data**:
- consolidated_results.csv: 1302 rows
- egis_rules_per_chunk.csv: 4848 rows
- egis_transition_metrics.csv: 4512 rows

**Dimensions**: 48 datasets (41 binary + 7 multiclass), 8 models, 7 configs

---

## Section 1 -- Experiment Dimensions

- **Total datasets**: 48
- **Binary datasets**: 41
- **Multiclass datasets**: 7
- **Multiclass list**: LED_Abrupt_Simple, LED_Gradual_Simple, LED_Stationary, RBF_Stationary, WAVEFORM_Abrupt_Simple, WAVEFORM_Gradual_Simple, WAVEFORM_Stationary
- **EGIS configs**: 7
- **Baseline configs (with comparative models)**: 3
- **Total models**: 8
- **Total config_labels in CSV**: 7
- **Total rows in consolidated_results.csv**: 1302
- **EGIS result rows**: 336
- **Baseline result rows**: 966

### Datasets per Drift Type

| Drift Type | Count |
|---|---|
| abrupt | 16 |
| gradual | 11 |
| noisy | 8 |
| stationary | 9 |
| real | 4 |

### Binary Datasets per Drift Type

| Drift Type | Count |
|---|---|
| abrupt | 14 |
| gradual | 9 |
| noisy | 8 |
| stationary | 6 |
| real | 4 |

### Data for tab:dataset_sizes (Dataset Dimensions and Chunk Structure)

| Config | Chunk Size | Penalty | Num Chunks | Evals per Run | Total Datasets |
|---|---|---|---|---|---|
| EXP-500-NP | 500 | 0.0 | 24 | 24 | 48 |
| EXP-500-P | 500 | 0.1 | 24 | 24 | 48 |
| EXP-500-P03 | 500 | 0.3 | 24 | 24 | 48 |
| EXP-1000-NP | 1000 | 0.0 | 12 | 12 | 48 |
| EXP-1000-P | 1000 | 0.1 | 12 | 12 | 48 |
| EXP-2000-NP | 2000 | 0.0 | 6 | 6 | 48 |
| EXP-2000-P | 2000 | 0.1 | 6 | 6 | 48 |

## Section 2 -- Experimental Configurations (tab:exp_config)

| Label | Chunk Size | Penalty (gamma) | Num Chunks | Instances/Chunk | Total Instances | Datasets |
|---|---|---|---|---|---|---|
| EXP-500-NP | 500 | 0.0 | 24 | 500 | 12000 | 48 |
| EXP-500-P | 500 | 0.1 | 24 | 500 | 12000 | 48 |
| EXP-500-P03 | 500 | 0.3 | 24 | 500 | 12000 | 48 |
| EXP-1000-NP | 1000 | 0.0 | 12 | 1000 | 12000 | 48 |
| EXP-1000-P | 1000 | 0.1 | 12 | 1000 | 12000 | 48 |
| EXP-2000-NP | 2000 | 0.0 | 6 | 2000 | 12000 | 48 |
| EXP-2000-P | 2000 | 0.1 | 6 | 2000 | 12000 | 48 |

### EGIS Hyperparameters (tab:egis_params)

| Parameter | Value |
|---|---|
| Population size | 120 |
| Max generations | 200 |
| Elitism rate | 0.1 |
| Intelligent mutation rate | 0.8 |
| Tournament size (initial/final) | 2 / 5 |
| Max rules per class | 15 |
| Initial max depth | 10 |
| Stagnation threshold | 10 |
| Early stopping patience | 20 |
| DT seeding on init | True (ratio=0.8) |
| DT seeding depths | 4, 7, 10, 13 |
| Balanced crossover | True |
| Penalty gamma values | 0.0, 0.1, 0.3 |
| Memory max size | 20 |
| Active memory pruning | True |

## Section 3 -- G-Mean Performance

### Block A -- Binary Datasets (41) -- EGIS Variants + Baselines per Chunk Size

For tab:summary_all (binary block). Each chunk_size shows all EGIS variants + baselines.


#### Chunk Size: 500

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8591 | 0.1247 | 41 |
| EGIS (P) | 0.8596 | 0.1218 | 41 |
| EGIS (P03) | 0.8464 | 0.1309 | 41 |
| ARF | 0.8776 | 0.1405 | 41 |
| SRP | 0.8691 | 0.1498 | 41 |
| HAT | 0.8149 | 0.1502 | 41 |
| ROSE | 0.8923 | 0.1109 | 41 |
| ACDWM | 0.8603 | 0.0929 | 41 |
| ERulesD2S | 0.6050 | 0.0866 | 41 |
| CDCMS | 0.8460 | 0.1518 | 41 |

#### Chunk Size: 1000

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8692 | 0.1133 | 41 |
| EGIS (P) | 0.8688 | 0.1132 | 41 |
| ARF | 0.8786 | 0.1395 | 41 |
| SRP | 0.8618 | 0.1524 | 41 |
| HAT | 0.8191 | 0.1462 | 41 |
| ROSE | 0.8923 | 0.1109 | 41 |
| ACDWM | 0.8182 | 0.0781 | 41 |
| ERulesD2S | 0.5994 | 0.0841 | 41 |
| CDCMS | 0.8414 | 0.1520 | 41 |

#### Chunk Size: 2000

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8550 | 0.1085 | 41 |
| EGIS (P) | 0.8513 | 0.1084 | 41 |
| ARF | 0.8777 | 0.1402 | 41 |
| SRP | 0.8621 | 0.1528 | 41 |
| HAT | 0.8200 | 0.1484 | 41 |
| ROSE | 0.8923 | 0.1109 | 41 |
| ACDWM | 0.7399 | 0.0686 | 41 |
| ERulesD2S | 0.5984 | 0.0824 | 41 |
| CDCMS | 0.8385 | 0.1534 | 41 |

### Block A2 -- Multiclass Datasets (7) -- EGIS Variants + Baselines per Chunk Size

ACDWM/CDCMS excluded (binary-only models).


#### Chunk Size: 500

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8383 | 0.1263 | 7 |
| EGIS (P) | 0.8271 | 0.1321 | 7 |
| EGIS (P03) | 0.7786 | 0.1770 | 7 |
| ARF | 0.7737 | 0.1077 | 7 |
| SRP | 0.7136 | 0.1972 | 7 |
| HAT | 0.4848 | 0.3990 | 7 |
| ROSE | 0.6372 | 0.2829 | 7 |
| ERulesD2S | 0.3072 | 0.0939 | 7 |

#### Chunk Size: 1000

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8448 | 0.1229 | 7 |
| EGIS (P) | 0.8512 | 0.1237 | 7 |
| ARF | 0.8373 | 0.0536 | 7 |
| SRP | 0.7656 | 0.1305 | 7 |
| HAT | 0.4810 | 0.3962 | 7 |
| ROSE | 0.6372 | 0.2829 | 7 |
| ERulesD2S | 0.3004 | 0.0912 | 7 |

#### Chunk Size: 2000

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8560 | 0.1261 | 7 |
| EGIS (P) | 0.8232 | 0.1497 | 7 |
| ARF | 0.8212 | 0.0695 | 7 |
| SRP | 0.7784 | 0.1206 | 7 |
| HAT | 0.4914 | 0.3949 | 7 |
| ROSE | 0.6372 | 0.2829 | 7 |
| ERulesD2S | 0.3045 | 0.0913 | 7 |

### Block B -- All Datasets (48) -- EGIS Variants + Baselines

ACDWM/CDCMS only have 41 datasets (binary only).


#### Chunk Size: 500

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8561 | 0.1238 | 48 |
| EGIS (P) | 0.8549 | 0.1224 | 48 |
| EGIS (P03) | 0.8365 | 0.1384 | 48 |
| ARF | 0.8624 | 0.1402 | 48 |
| SRP | 0.8464 | 0.1648 | 48 |
| HAT | 0.7667 | 0.2310 | 48 |
| ROSE | 0.8551 | 0.1702 | 48 |
| ACDWM | 0.8603 | 0.0929 | 41 |
| ERulesD2S | 0.5616 | 0.1371 | 48 |
| CDCMS | 0.8460 | 0.1518 | 41 |

#### Chunk Size: 1000

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8657 | 0.1137 | 48 |
| EGIS (P) | 0.8663 | 0.1136 | 48 |
| ARF | 0.8726 | 0.1309 | 48 |
| SRP | 0.8478 | 0.1521 | 48 |
| HAT | 0.7698 | 0.2297 | 48 |
| ROSE | 0.8551 | 0.1702 | 48 |
| ACDWM | 0.8182 | 0.0781 | 41 |
| ERulesD2S | 0.5558 | 0.1358 | 48 |
| CDCMS | 0.8414 | 0.1520 | 41 |

#### Chunk Size: 2000

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8551 | 0.1097 | 48 |
| EGIS (P) | 0.8472 | 0.1139 | 48 |
| ARF | 0.8695 | 0.1333 | 48 |
| SRP | 0.8499 | 0.1504 | 48 |
| HAT | 0.7721 | 0.2289 | 48 |
| ROSE | 0.8551 | 0.1702 | 48 |
| ACDWM | 0.7399 | 0.0686 | 41 |
| ERulesD2S | 0.5555 | 0.1335 | 48 |
| CDCMS | 0.8385 | 0.1534 | 41 |

### Block C -- Performance by Drift Type (Binary)

For tab:drift_performance. Shows EGIS variants + baselines per drift type.


#### Chunk 500 -- Drift Type: abrupt (14 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8595 | 0.1214 | 14 |
| EGIS (P) | 0.8624 | 0.1160 | 14 |
| EGIS (P03) | 0.8446 | 0.1285 | 14 |
| ARF | 0.8629 | 0.1537 | 14 |
| SRP | 0.8572 | 0.1599 | 14 |
| HAT | 0.7762 | 0.1542 | 14 |
| ROSE | 0.8877 | 0.1103 | 14 |
| ACDWM | 0.8615 | 0.0852 | 14 |
| ERulesD2S | 0.5996 | 0.0927 | 14 |
| CDCMS | 0.8126 | 0.1706 | 14 |

#### Chunk 500 -- Drift Type: gradual (9 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8548 | 0.1158 | 9 |
| EGIS (P) | 0.8554 | 0.1116 | 9 |
| EGIS (P03) | 0.8399 | 0.1272 | 9 |
| ARF | 0.8884 | 0.1360 | 9 |
| SRP | 0.8755 | 0.1499 | 9 |
| HAT | 0.8315 | 0.1430 | 9 |
| ROSE | 0.8999 | 0.0959 | 9 |
| ACDWM | 0.8588 | 0.0774 | 9 |
| ERulesD2S | 0.6071 | 0.0764 | 9 |
| CDCMS | 0.8588 | 0.1337 | 9 |

#### Chunk 500 -- Drift Type: noisy (8 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8456 | 0.1160 | 8 |
| EGIS (P) | 0.8417 | 0.1145 | 8 |
| EGIS (P03) | 0.8288 | 0.1219 | 8 |
| ARF | 0.8547 | 0.1591 | 8 |
| SRP | 0.8472 | 0.1672 | 8 |
| HAT | 0.7928 | 0.1341 | 8 |
| ROSE | 0.8884 | 0.0917 | 8 |
| ACDWM | 0.8490 | 0.0767 | 8 |
| ERulesD2S | 0.5928 | 0.0912 | 8 |
| CDCMS | 0.8181 | 0.1519 | 8 |

#### Chunk 500 -- Drift Type: stationary (6 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8621 | 0.1878 | 6 |
| EGIS (P) | 0.8634 | 0.1864 | 6 |
| EGIS (P03) | 0.8571 | 0.1906 | 6 |
| ARF | 0.8835 | 0.1604 | 6 |
| SRP | 0.8769 | 0.1819 | 6 |
| HAT | 0.8346 | 0.2086 | 6 |
| ROSE | 0.9014 | 0.1605 | 6 |
| ACDWM | 0.8600 | 0.1554 | 6 |
| ERulesD2S | 0.6020 | 0.1028 | 6 |
| CDCMS | 0.8888 | 0.1860 | 6 |

#### Chunk 500 -- Drift Type: real (4 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8900 | 0.1257 | 4 |
| EGIS (P) | 0.8894 | 0.1246 | 4 |
| EGIS (P03) | 0.8865 | 0.1225 | 4 |
| ARF | 0.9414 | 0.0378 | 4 |
| SRP | 0.9286 | 0.0359 | 4 |
| HAT | 0.9274 | 0.0449 | 4 |
| ROSE | 0.8853 | 0.1540 | 4 |
| ACDWM | 0.8825 | 0.1124 | 4 |
| ERulesD2S | 0.6486 | 0.0844 | 4 |
| CDCMS | 0.9259 | 0.0459 | 4 |

#### Chunk 1000 -- Drift Type: abrupt (14 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8676 | 0.1082 | 14 |
| EGIS (P) | 0.8672 | 0.1075 | 14 |
| ARF | 0.8636 | 0.1502 | 14 |
| SRP | 0.8519 | 0.1576 | 14 |
| HAT | 0.7795 | 0.1562 | 14 |
| ROSE | 0.8877 | 0.1103 | 14 |
| ACDWM | 0.8075 | 0.0766 | 14 |
| ERulesD2S | 0.5933 | 0.0848 | 14 |
| CDCMS | 0.8038 | 0.1743 | 14 |

#### Chunk 1000 -- Drift Type: gradual (9 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8668 | 0.1008 | 9 |
| EGIS (P) | 0.8660 | 0.1015 | 9 |
| ARF | 0.8851 | 0.1468 | 9 |
| SRP | 0.8663 | 0.1534 | 9 |
| HAT | 0.8349 | 0.1373 | 9 |
| ROSE | 0.8999 | 0.0959 | 9 |
| ACDWM | 0.8186 | 0.0732 | 9 |
| ERulesD2S | 0.6016 | 0.0859 | 9 |
| CDCMS | 0.8606 | 0.1219 | 9 |

#### Chunk 1000 -- Drift Type: noisy (8 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8507 | 0.1025 | 8 |
| EGIS (P) | 0.8506 | 0.1024 | 8 |
| ARF | 0.8631 | 0.1508 | 8 |
| SRP | 0.8354 | 0.1749 | 8 |
| HAT | 0.7940 | 0.1343 | 8 |
| ROSE | 0.8884 | 0.0917 | 8 |
| ACDWM | 0.8038 | 0.0530 | 8 |
| ERulesD2S | 0.5831 | 0.0869 | 8 |
| CDCMS | 0.8103 | 0.1434 | 8 |

#### Chunk 1000 -- Drift Type: stationary (6 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8726 | 0.1826 | 6 |
| EGIS (P) | 0.8726 | 0.1826 | 6 |
| ARF | 0.8847 | 0.1613 | 6 |
| SRP | 0.8726 | 0.1863 | 6 |
| HAT | 0.8484 | 0.1853 | 6 |
| ROSE | 0.9014 | 0.1605 | 6 |
| ACDWM | 0.8376 | 0.1174 | 6 |
| ERulesD2S | 0.5985 | 0.0954 | 6 |
| CDCMS | 0.8900 | 0.1930 | 6 |

#### Chunk 1000 -- Drift Type: real (4 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.9125 | 0.0984 | 4 |
| EGIS (P) | 0.9118 | 0.0984 | 4 |
| ARF | 0.9385 | 0.0319 | 4 |
| SRP | 0.9230 | 0.0496 | 4 |
| HAT | 0.9286 | 0.0494 | 4 |
| ROSE | 0.8853 | 0.1540 | 4 |
| ACDWM | 0.8543 | 0.0919 | 4 |
| ERulesD2S | 0.6496 | 0.0794 | 4 |
| CDCMS | 0.9186 | 0.0624 | 4 |

#### Chunk 2000 -- Drift Type: abrupt (14 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8374 | 0.1023 | 14 |
| EGIS (P) | 0.8358 | 0.1006 | 14 |
| ARF | 0.8598 | 0.1542 | 14 |
| SRP | 0.8431 | 0.1653 | 14 |
| HAT | 0.7886 | 0.1433 | 14 |
| ROSE | 0.8877 | 0.1103 | 14 |
| ACDWM | 0.7226 | 0.0702 | 14 |
| ERulesD2S | 0.5934 | 0.0854 | 14 |
| CDCMS | 0.7965 | 0.1770 | 14 |

#### Chunk 2000 -- Drift Type: gradual (9 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8582 | 0.1018 | 9 |
| EGIS (P) | 0.8524 | 0.0998 | 9 |
| ARF | 0.8867 | 0.1366 | 9 |
| SRP | 0.8657 | 0.1537 | 9 |
| HAT | 0.8394 | 0.1282 | 9 |
| ROSE | 0.8999 | 0.0959 | 9 |
| ACDWM | 0.7454 | 0.0642 | 9 |
| ERulesD2S | 0.5975 | 0.0773 | 9 |
| CDCMS | 0.8608 | 0.1198 | 9 |

#### Chunk 2000 -- Drift Type: noisy (8 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8358 | 0.0829 | 8 |
| EGIS (P) | 0.8264 | 0.0894 | 8 |
| ARF | 0.8637 | 0.1541 | 8 |
| SRP | 0.8509 | 0.1634 | 8 |
| HAT | 0.7929 | 0.1358 | 8 |
| ROSE | 0.8884 | 0.0917 | 8 |
| ACDWM | 0.7199 | 0.0453 | 8 |
| ERulesD2S | 0.5851 | 0.0833 | 8 |
| CDCMS | 0.7949 | 0.1379 | 8 |

#### Chunk 2000 -- Drift Type: stationary (6 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.8811 | 0.1681 | 6 |
| EGIS (P) | 0.8793 | 0.1688 | 6 |
| ARF | 0.8828 | 0.1643 | 6 |
| SRP | 0.8729 | 0.1871 | 6 |
| HAT | 0.8268 | 0.2347 | 6 |
| ROSE | 0.9014 | 0.1605 | 6 |
| ACDWM | 0.7697 | 0.0907 | 6 |
| ERulesD2S | 0.6026 | 0.0990 | 6 |
| CDCMS | 0.8895 | 0.1891 | 6 |

#### Chunk 2000 -- Drift Type: real (4 binary datasets)

| Model | Mean G-Mean | Std | N |
|---|---|---|---|
| EGIS (NP) | 0.9085 | 0.1147 | 4 |
| EGIS (P) | 0.9112 | 0.1056 | 4 |
| ARF | 0.9410 | 0.0364 | 4 |
| SRP | 0.9266 | 0.0386 | 4 |
| HAT | 0.9308 | 0.0443 | 4 |
| ROSE | 0.8853 | 0.1540 | 4 |
| ACDWM | 0.7826 | 0.0712 | 4 |
| ERulesD2S | 0.6379 | 0.0877 | 4 |
| CDCMS | 0.9461 | 0.0219 | 4 |

### Block D -- Per-Dataset G-Mean (EXP-500)

Columns: EGIS variants (EGIS (NP), EGIS (P), EGIS (P03)) + baselines.
Grouped by drift type. Best value per row in **bold**.
ACDWM/CDCMS shown as '--' for multiclass datasets.

| Dataset | EGIS (NP) | EGIS (P) | EGIS (P03) | ARF | SRP | HAT | ROSE | ACDWM | ERulesD2S | CDCMS |
|---|---|---|---|---|---|---|---|---|---|---|
| **Abrupt (14)** |||||||||||
| AGRAWAL_Abrupt_Chain_Long | 0.8881 | **0.8959** | 0.8691 | 0.7878 | 0.7846 | 0.6645 | 0.8301 | 0.8558 | 0.5285 | 0.6552 |
| AGRAWAL_Abrupt_Simple_Mild | **0.9055** | 0.8881 | 0.8496 | 0.8249 | 0.8662 | 0.5850 | 0.9005 | 0.8882 | 0.5420 | 0.6280 |
| AGRAWAL_Abrupt_Simple_Severe | 0.8997 | 0.8980 | 0.8756 | 0.8398 | 0.8894 | 0.6800 | **0.9250** | 0.8905 | 0.5432 | 0.6941 |
| HYPERPLANE_Abrupt_Simple | 0.7555 | 0.7523 | 0.7133 | 0.8354 | 0.7713 | 0.8822 | 0.8634 | 0.7970 | 0.5252 | **0.9039** |
| RANDOMTREE_Abrupt_Recurring | 0.6149 | 0.6290 | 0.5997 | 0.5333 | 0.5242 | 0.5334 | 0.6540 | **0.6851** | 0.5519 | 0.5386 |
| RANDOMTREE_Abrupt_Simple | 0.6139 | 0.6300 | 0.5897 | 0.5499 | 0.5113 | 0.5423 | 0.6595 | **0.6840** | 0.5416 | 0.5281 |
| RBF_Abrupt_Blip | 0.8318 | 0.8433 | 0.8221 | 0.9102 | **0.9162** | 0.8371 | 0.8960 | 0.8700 | 0.5212 | 0.8540 |
| RBF_Abrupt_Severe | 0.8069 | 0.8215 | 0.7986 | 0.8938 | **0.9015** | 0.7893 | 0.8697 | 0.8417 | 0.5217 | 0.7948 |
| SEA_Abrupt_Chain | 0.9472 | 0.9465 | 0.9440 | 0.9718 | 0.9557 | 0.9303 | **0.9774** | 0.9263 | 0.6232 | 0.9361 |
| SEA_Abrupt_Recurring | 0.9452 | 0.9443 | 0.9425 | **0.9772** | 0.9392 | 0.9459 | 0.9755 | 0.9313 | 0.6460 | 0.9540 |
| SEA_Abrupt_Simple | 0.9496 | 0.9504 | 0.9468 | 0.9772 | 0.9731 | 0.9524 | **0.9785** | 0.9362 | 0.6204 | 0.9412 |
| SINE_Abrupt_Simple | 0.9510 | 0.9498 | 0.9488 | **0.9881** | 0.9868 | 0.9708 | 0.9796 | 0.9427 | 0.6547 | 0.9766 |
| STAGGER_Abrupt_Chain | 0.9676 | 0.9676 | 0.9676 | **0.9943** | 0.9929 | 0.7630 | 0.9215 | 0.8952 | 0.7776 | 0.9825 |
| STAGGER_Abrupt_Recurring | 0.9565 | 0.9565 | 0.9565 | 0.9969 | 0.9889 | 0.7902 | **0.9973** | 0.9167 | 0.7969 | 0.9895 |
| **Gradual (9)** |||||||||||
| HYPERPLANE_Gradual_Simple | 0.7555 | 0.7523 | 0.7100 | 0.8316 | 0.7744 | 0.8792 | 0.8634 | 0.7986 | 0.5361 | **0.9039** |
| RANDOMTREE_Gradual_Simple | 0.6099 | 0.6265 | 0.5852 | 0.5485 | 0.5113 | 0.5229 | 0.6688 | **0.6974** | 0.5612 | 0.5257 |
| RBF_Gradual_Moderate | 0.8191 | 0.8158 | 0.7969 | 0.8945 | **0.9044** | 0.8197 | 0.8893 | 0.8501 | 0.5347 | 0.8234 |
| RBF_Gradual_Severe | 0.8232 | 0.8222 | 0.8020 | **0.9025** | 0.8966 | 0.8298 | 0.8944 | 0.8513 | 0.5259 | 0.8196 |
| SEA_Gradual_Recurring | 0.9388 | 0.9369 | 0.9320 | **0.9613** | 0.9539 | 0.9202 | 0.9596 | 0.9156 | 0.6350 | 0.9221 |
| SEA_Gradual_Simple_Fast | 0.9529 | 0.9518 | 0.9461 | 0.9771 | 0.9765 | 0.9501 | **0.9795** | 0.9385 | 0.6384 | 0.9558 |
| SEA_Gradual_Simple_Slow | 0.9566 | 0.9561 | 0.9527 | 0.9775 | 0.9761 | 0.9477 | **0.9811** | 0.9385 | 0.6326 | 0.9358 |
| SINE_Gradual_Recurring | 0.9173 | 0.9163 | 0.9138 | **0.9522** | 0.9501 | 0.9285 | 0.9428 | 0.9001 | 0.6359 | 0.9237 |
| STAGGER_Gradual_Chain | 0.9195 | 0.9203 | 0.9200 | **0.9500** | 0.9360 | 0.6859 | 0.9203 | 0.8385 | 0.7637 | 0.9188 |
| **Noisy (8)** |||||||||||
| AGRAWAL_Abrupt_Simple_Severe_Noise | 0.9173 | 0.8977 | 0.8765 | 0.7790 | 0.8356 | 0.6868 | **0.9401** | 0.8867 | 0.5474 | 0.6622 |
| HYPERPLANE_Gradual_Noise | 0.7527 | 0.7472 | 0.7134 | 0.8115 | 0.7591 | 0.8556 | 0.8478 | 0.7837 | 0.5295 | **0.8734** |
| RANDOMTREE_Gradual_Noise | 0.6202 | 0.6225 | 0.6081 | 0.5112 | 0.4766 | 0.5322 | 0.6902 | **0.6903** | 0.5502 | 0.5196 |
| RBF_Abrupt_Blip_Noise | 0.8267 | 0.8261 | 0.8056 | 0.9040 | **0.9046** | 0.8254 | 0.8859 | 0.8592 | 0.5228 | 0.8500 |
| RBF_Gradual_Severe_Noise | 0.8242 | 0.8182 | 0.8060 | 0.8903 | **0.9012** | 0.8177 | 0.8771 | 0.8527 | 0.5184 | 0.8226 |
| SEA_Abrupt_Chain_Noise | 0.9458 | 0.9451 | 0.9442 | 0.9655 | 0.9374 | 0.9172 | **0.9712** | 0.9215 | 0.6324 | 0.9281 |
| SINE_Abrupt_Recurring_Noise | 0.9133 | 0.9126 | 0.9124 | 0.9807 | **0.9813** | 0.9491 | 0.9731 | 0.8994 | 0.6678 | 0.9464 |
| STAGGER_Abrupt_Chain_Noise | 0.9647 | 0.9645 | 0.9645 | **0.9953** | 0.9817 | 0.7584 | 0.9216 | 0.8981 | 0.7737 | 0.9428 |
| **Stationary (6)** |||||||||||
| AGRAWAL_Stationary | **1.0000** | 1.0000 | 1.0000 | 0.9335 | 0.9983 | 0.9813 | 0.9967 | 0.9583 | 0.5239 | 0.9868 |
| HYPERPLANE_Stationary | 0.7229 | 0.7318 | 0.7001 | 0.8185 | 0.7621 | 0.8745 | 0.8523 | 0.7818 | 0.5259 | **0.9044** |
| RANDOMTREE_Stationary | 0.5425 | 0.5425 | 0.5425 | 0.5852 | 0.5550 | 0.4321 | **0.5941** | 0.5748 | 0.5278 | 0.5157 |
| SEA_Stationary | 0.9544 | 0.9539 | 0.9501 | 0.9798 | 0.9620 | 0.9505 | **0.9836** | 0.9433 | 0.6123 | 0.9460 |
| SINE_Stationary | 0.9531 | 0.9522 | 0.9497 | 0.9858 | **0.9867** | 0.9684 | 0.9824 | 0.9436 | 0.6348 | 0.9798 |
| STAGGER_Stationary | **1.0000** | 1.0000 | 1.0000 | 0.9980 | 0.9975 | 0.8009 | 0.9994 | 0.9583 | 0.7872 | 1.0000 |
| **Real (4)** |||||||||||
| AssetNegotiation_F2 | 0.9494 | 0.9486 | 0.9460 | 0.9672 | 0.9717 | 0.9472 | **0.9820** | 0.9425 | 0.6512 | 0.9495 |
| AssetNegotiation_F3 | **0.9952** | 0.9952 | 0.9952 | 0.9656 | 0.9215 | 0.9614 | 0.9922 | 0.9577 | 0.7623 | 0.9468 |
| AssetNegotiation_F4 | 0.9055 | 0.9025 | 0.8903 | 0.9464 | 0.9362 | 0.9395 | 0.6618 | 0.9134 | 0.5616 | **0.9502** |
| Electricity | 0.7096 | 0.7113 | 0.7143 | 0.8865 | 0.8851 | 0.8614 | **0.9053** | 0.7162 | 0.6193 | 0.8570 |
| **Multiclass (7)** |||||||||||
| LED_Abrupt_Simple | **0.9077** | 0.8950 | 0.9063 | 0.8298 | 0.5787 | 0.1185 | 0.4049 | -- | 0.2208 | -- |
| LED_Gradual_Simple | 0.9869 | **0.9871** | 0.9866 | 0.6807 | 0.5889 | 0.0699 | 0.3472 | -- | 0.2265 | -- |
| LED_Stationary | **1.0000** | 1.0000 | 1.0000 | 0.5900 | 0.3970 | 0.0000 | 0.2743 | -- | 0.2204 | -- |
| RBF_Stationary | 0.8137 | 0.7842 | 0.6196 | 0.9212 | **0.9687** | 0.9002 | 0.9627 | -- | 0.2676 | -- |
| WAVEFORM_Abrupt_Simple | 0.7224 | 0.7036 | 0.6305 | 0.7958 | 0.8194 | 0.7712 | **0.8243** | -- | 0.4048 | -- |
| WAVEFORM_Gradual_Simple | 0.7152 | 0.7166 | 0.6344 | 0.7843 | 0.8149 | 0.7679 | **0.8225** | -- | 0.3815 | -- |
| WAVEFORM_Stationary | 0.7224 | 0.7035 | 0.6725 | 0.8141 | **0.8274** | 0.7662 | 0.8248 | -- | 0.4289 | -- |
|---|---|---|---|---|---|---|---|---|---|---|
| **Binary Summary (n=41)** |||||||||||
| Mean | 0.8591 | 0.8596 | 0.8464 | 0.8776 | 0.8691 | 0.8149 | 0.8923 | 0.8603 | 0.6050 | 0.8460 |
| Std | 0.1232 | 0.1203 | 0.1293 | 0.1388 | 0.1480 | 0.1484 | 0.1096 | 0.0917 | 0.0856 | 0.1499 |
| EGIS(NP) W/L/D | -- | 13/9/19 | 31/1/9 | 11/30/0 | 13/28/0 | 26/13/2 | 7/32/2 | 24/17/0 | 41/0/0 | 21/17/3 |
| Avg Rank | 4.91 | 5.27 | 6.54 | 3.39 | 4.27 | 6.66 | 2.76 | 6.02 | 9.51 | 5.67 |
| **All Summary (n=48)** |||||||||||
| Mean | 0.8561 | 0.8549 | 0.8365 | 0.8624 | 0.8464 | 0.7667 | 0.8551 | 0.8603 | 0.5616 | 0.8460 |
| Std | 0.1225 | 0.1211 | 0.1370 | 0.1387 | 0.1630 | 0.2286 | 0.1684 | 0.0917 | 0.1357 | 0.1499 |

### Block E -- Per-Dataset G-Mean (EXP-1000)

Columns: EGIS variants (EGIS (NP), EGIS (P)) + baselines.
Grouped by drift type. Best value per row in **bold**.
ACDWM/CDCMS shown as '--' for multiclass datasets.

| Dataset | EGIS (NP) | EGIS (P) | ARF | SRP | HAT | ROSE | ACDWM | ERulesD2S | CDCMS |
|---|---|---|---|---|---|---|---|---|---|
| **Abrupt (14)** ||||||||||
| AGRAWAL_Abrupt_Chain_Long | **0.8723** | 0.8723 | 0.7897 | 0.8606 | 0.6756 | 0.8301 | 0.7417 | 0.5260 | 0.6234 |
| AGRAWAL_Abrupt_Simple_Mild | **0.9117** | 0.9117 | 0.7989 | 0.8028 | 0.6456 | 0.9005 | 0.8278 | 0.5373 | 0.6127 |
| AGRAWAL_Abrupt_Simple_Severe | 0.9020 | 0.9020 | 0.8471 | 0.8566 | 0.6902 | **0.9250** | 0.8341 | 0.5374 | 0.6908 |
| HYPERPLANE_Abrupt_Simple | 0.7812 | 0.7769 | 0.8306 | 0.7728 | 0.8771 | 0.8634 | 0.7670 | 0.5242 | **0.8993** |
| RANDOMTREE_Abrupt_Recurring | 0.6448 | 0.6448 | 0.5611 | 0.5298 | 0.5198 | 0.6540 | **0.6695** | 0.5479 | 0.5168 |
| RANDOMTREE_Abrupt_Simple | 0.6489 | 0.6564 | 0.5479 | 0.5012 | 0.4922 | 0.6595 | **0.6697** | 0.5497 | 0.5330 |
| RBF_Abrupt_Blip | 0.8578 | 0.8578 | 0.9140 | **0.9154** | 0.8360 | 0.8960 | 0.8300 | 0.5273 | 0.8413 |
| RBF_Abrupt_Severe | 0.8322 | 0.8238 | **0.8978** | 0.8958 | 0.8117 | 0.8697 | 0.7758 | 0.5080 | 0.7810 |
| SEA_Abrupt_Chain | 0.9616 | 0.9616 | 0.9721 | 0.9308 | 0.9295 | **0.9774** | 0.8787 | 0.6181 | 0.9343 |
| SEA_Abrupt_Recurring | 0.9659 | 0.9659 | **0.9765** | 0.9309 | 0.9452 | 0.9755 | 0.8900 | 0.6441 | 0.9582 |
| SEA_Abrupt_Simple | 0.9635 | 0.9635 | 0.9752 | 0.9608 | 0.9487 | **0.9785** | 0.8962 | 0.6166 | 0.9415 |
| SINE_Abrupt_Simple | 0.9639 | 0.9639 | **0.9873** | 0.9864 | 0.9747 | 0.9796 | 0.9016 | 0.6497 | 0.9747 |
| STAGGER_Abrupt_Chain | 0.9311 | 0.9311 | **0.9951** | 0.9925 | 0.7811 | 0.9215 | 0.7899 | 0.7563 | 0.9680 |
| STAGGER_Abrupt_Recurring | 0.9091 | 0.9091 | **0.9975** | 0.9905 | 0.7858 | 0.9973 | 0.8333 | 0.7630 | 0.9780 |
| **Gradual (9)** ||||||||||
| HYPERPLANE_Gradual_Simple | 0.7810 | 0.7743 | 0.8369 | 0.7712 | 0.8787 | 0.8634 | 0.7719 | 0.5399 | **0.8993** |
| RANDOMTREE_Gradual_Simple | 0.6539 | 0.6539 | 0.5131 | 0.4874 | 0.5271 | 0.6688 | **0.7005** | 0.5368 | 0.5523 |
| RBF_Gradual_Moderate | 0.8474 | 0.8474 | 0.8961 | **0.9024** | 0.8193 | 0.8893 | 0.8126 | 0.5144 | 0.8307 |
| RBF_Gradual_Severe | 0.8476 | 0.8476 | 0.9051 | **0.9051** | 0.8251 | 0.8944 | 0.8172 | 0.5203 | 0.8470 |
| SEA_Gradual_Recurring | 0.9424 | 0.9424 | 0.9594 | 0.9334 | 0.9143 | **0.9596** | 0.8754 | 0.6480 | 0.9073 |
| SEA_Gradual_Simple_Fast | 0.9654 | 0.9654 | 0.9777 | 0.9323 | 0.9470 | **0.9795** | 0.8991 | 0.6004 | 0.9513 |
| SEA_Gradual_Simple_Slow | 0.9625 | 0.9625 | 0.9774 | 0.9716 | 0.9446 | **0.9811** | 0.9011 | 0.6285 | 0.9366 |
| SINE_Gradual_Recurring | 0.9227 | 0.9227 | **0.9519** | 0.9504 | 0.9356 | 0.9428 | 0.8620 | 0.6455 | 0.9168 |
| STAGGER_Gradual_Chain | 0.8783 | 0.8783 | **0.9486** | 0.9428 | 0.7221 | 0.9203 | 0.7276 | 0.7805 | 0.9043 |
| **Noisy (8)** ||||||||||
| AGRAWAL_Abrupt_Simple_Severe_Noise | 0.9281 | 0.9271 | 0.8094 | 0.7486 | 0.6597 | **0.9401** | 0.8355 | 0.5454 | 0.6366 |
| HYPERPLANE_Gradual_Noise | 0.7582 | 0.7582 | 0.8224 | 0.7634 | 0.8550 | 0.8478 | 0.7560 | 0.5371 | **0.8757** |
| RANDOMTREE_Gradual_Noise | 0.6509 | 0.6509 | 0.5315 | 0.4635 | 0.5518 | 0.6902 | **0.7049** | 0.5359 | 0.5456 |
| RBF_Abrupt_Blip_Noise | 0.8530 | 0.8530 | 0.9021 | **0.9042** | 0.8238 | 0.8859 | 0.8310 | 0.5192 | 0.8337 |
| RBF_Gradual_Severe_Noise | 0.8421 | 0.8421 | 0.8961 | **0.8999** | 0.8157 | 0.8771 | 0.8180 | 0.5256 | 0.8192 |
| SEA_Abrupt_Chain_Noise | 0.9568 | 0.9568 | 0.9681 | 0.9360 | 0.9287 | **0.9712** | 0.8762 | 0.5975 | 0.9252 |
| SINE_Abrupt_Recurring_Noise | 0.8840 | 0.8840 | 0.9799 | **0.9811** | 0.9533 | 0.9731 | 0.8184 | 0.6275 | 0.9255 |
| STAGGER_Abrupt_Chain_Noise | 0.9325 | 0.9325 | **0.9951** | 0.9868 | 0.7640 | 0.9216 | 0.7902 | 0.7769 | 0.9211 |
| **Stationary (6)** ||||||||||
| AGRAWAL_Stationary | **1.0000** | 1.0000 | 0.9409 | 0.9984 | 0.9796 | 0.9967 | 0.9167 | 0.5308 | 0.9958 |
| HYPERPLANE_Stationary | 0.7371 | 0.7371 | 0.8199 | 0.7624 | 0.8806 | 0.8523 | 0.7452 | 0.5248 | **0.9104** |
| RANDOMTREE_Stationary | 0.5606 | 0.5606 | 0.5838 | 0.5398 | 0.4957 | 0.5941 | **0.6398** | 0.5141 | 0.5020 |
| SEA_Stationary | 0.9714 | 0.9714 | 0.9795 | 0.9499 | 0.9581 | **0.9836** | 0.9034 | 0.6251 | 0.9519 |
| SINE_Stationary | 0.9665 | 0.9665 | 0.9868 | **0.9883** | 0.9711 | 0.9824 | 0.9040 | 0.6350 | 0.9799 |
| STAGGER_Stationary | **1.0000** | 1.0000 | 0.9974 | 0.9970 | 0.8053 | 0.9994 | 0.9167 | 0.7609 | 1.0000 |
| **Real (4)** ||||||||||
| AssetNegotiation_F2 | 0.9635 | 0.9638 | 0.9583 | 0.9726 | 0.9514 | **0.9820** | 0.9050 | 0.6272 | 0.9397 |
| AssetNegotiation_F3 | **0.9976** | 0.9976 | 0.9573 | 0.9292 | 0.9673 | 0.9922 | 0.9161 | 0.7605 | 0.9589 |
| AssetNegotiation_F4 | 0.9149 | 0.9120 | 0.9469 | 0.9358 | 0.9394 | 0.6618 | 0.8776 | 0.5724 | **0.9500** |
| Electricity | 0.7739 | 0.7739 | 0.8913 | 0.8543 | 0.8565 | **0.9053** | 0.7185 | 0.6385 | 0.8258 |
| **Multiclass (7)** ||||||||||
| LED_Abrupt_Simple | **0.9101** | 0.9101 | 0.8872 | 0.6035 | 0.1169 | 0.4049 | -- | 0.2119 | -- |
| LED_Gradual_Simple | **0.9882** | 0.9882 | 0.8020 | 0.6754 | 0.0701 | 0.3472 | -- | 0.2220 | -- |
| LED_Stationary | **1.0000** | 1.0000 | 0.8541 | 0.6425 | 0.0000 | 0.2743 | -- | 0.2058 | -- |
| RBF_Stationary | 0.8347 | 0.8804 | 0.9268 | **0.9737** | 0.9000 | 0.9627 | -- | 0.2834 | -- |
| WAVEFORM_Abrupt_Simple | 0.7265 | 0.7297 | 0.7916 | 0.8211 | 0.7621 | **0.8243** | -- | 0.3843 | -- |
| WAVEFORM_Gradual_Simple | 0.7273 | 0.7229 | 0.7883 | 0.8209 | 0.7576 | **0.8225** | -- | 0.3789 | -- |
| WAVEFORM_Stationary | 0.7272 | 0.7272 | 0.8113 | 0.8224 | 0.7600 | **0.8248** | -- | 0.4163 | -- |
|---|---|---|---|---|---|---|---|---|---|
| **Binary Summary (n=41)** ||||||||||
| Mean | 0.8692 | 0.8688 | 0.8786 | 0.8618 | 0.8191 | 0.8923 | 0.8182 | 0.5994 | 0.8414 |
| Std | 0.1119 | 0.1118 | 0.1378 | 0.1505 | 0.1444 | 0.1096 | 0.0771 | 0.0830 | 0.1502 |
| EGIS(NP) W/L/D | -- | 5/1/35 | 12/29/0 | 21/20/0 | 31/10/0 | 7/33/1 | 35/6/0 | 41/0/0 | 27/12/2 |
| Avg Rank | 4.16 | 4.23 | 3.17 | 4.27 | 6.05 | 2.61 | 6.61 | 8.59 | 5.32 |
| **All Summary (n=48)** ||||||||||
| Mean | 0.8657 | 0.8663 | 0.8726 | 0.8478 | 0.7698 | 0.8551 | 0.8182 | 0.5558 | 0.8414 |
| Std | 0.1125 | 0.1124 | 0.1295 | 0.1505 | 0.2273 | 0.1684 | 0.0771 | 0.1344 | 0.1502 |

### Block F -- Per-Dataset G-Mean (EXP-2000)

Columns: EGIS variants (EGIS (NP), EGIS (P)) + baselines.
Grouped by drift type. Best value per row in **bold**.
ACDWM/CDCMS shown as '--' for multiclass datasets.

| Dataset | EGIS (NP) | EGIS (P) | ARF | SRP | HAT | ROSE | ACDWM | ERulesD2S | CDCMS |
|---|---|---|---|---|---|---|---|---|---|
| **Abrupt (14)** ||||||||||
| AGRAWAL_Abrupt_Chain_Long | 0.7610 | 0.7646 | 0.7453 | 0.7460 | 0.6784 | **0.8301** | 0.6475 | 0.5248 | 0.5961 |
| AGRAWAL_Abrupt_Simple_Mild | 0.8331 | 0.8696 | 0.8350 | 0.7006 | 0.6532 | **0.9005** | 0.7467 | 0.5366 | 0.5950 |
| AGRAWAL_Abrupt_Simple_Severe | 0.8524 | 0.8358 | 0.8254 | 0.9187 | 0.6985 | **0.9250** | 0.7475 | 0.5433 | 0.6797 |
| HYPERPLANE_Abrupt_Simple | 0.7889 | 0.7771 | 0.8343 | 0.7733 | 0.8779 | 0.8634 | 0.6900 | 0.5335 | **0.9128** |
| RANDOMTREE_Abrupt_Recurring | 0.6739 | **0.6751** | 0.5405 | 0.5146 | 0.5474 | 0.6540 | 0.6016 | 0.5424 | 0.5032 |
| RANDOMTREE_Abrupt_Simple | 0.6613 | **0.6647** | 0.5520 | 0.5292 | 0.5509 | 0.6595 | 0.6446 | 0.5474 | 0.5301 |
| RBF_Abrupt_Blip | 0.8642 | 0.8487 | 0.9035 | **0.9169** | 0.8370 | 0.8960 | 0.7516 | 0.5126 | 0.8461 |
| RBF_Abrupt_Severe | 0.7836 | 0.7782 | **0.8936** | 0.8929 | 0.8130 | 0.8697 | 0.6835 | 0.5120 | 0.7986 |
| SEA_Abrupt_Chain | 0.9589 | 0.9571 | 0.9758 | 0.9638 | 0.9263 | **0.9774** | 0.7861 | 0.6346 | 0.9240 |
| SEA_Abrupt_Recurring | 0.9652 | 0.9564 | **0.9760** | 0.9000 | 0.9441 | 0.9755 | 0.8084 | 0.6237 | 0.9409 |
| SEA_Abrupt_Simple | 0.9587 | 0.9595 | 0.9773 | 0.9782 | 0.9478 | **0.9785** | 0.8142 | 0.6313 | 0.9455 |
| SINE_Abrupt_Simple | 0.9749 | 0.9671 | 0.9858 | **0.9872** | 0.9732 | 0.9796 | 0.8222 | 0.6422 | 0.9837 |
| STAGGER_Abrupt_Chain | 0.8479 | 0.8479 | **0.9957** | 0.9920 | 0.7905 | 0.9215 | 0.7066 | 0.7553 | 0.9438 |
| STAGGER_Abrupt_Recurring | 0.8000 | 0.8000 | 0.9970 | 0.9900 | 0.8014 | **0.9973** | 0.6667 | 0.7674 | 0.9516 |
| **Gradual (9)** ||||||||||
| HYPERPLANE_Gradual_Simple | 0.7704 | 0.7674 | 0.8295 | 0.7679 | 0.8780 | 0.8634 | 0.6908 | 0.5302 | **0.9128** |
| RANDOMTREE_Gradual_Simple | 0.6872 | 0.6786 | 0.5454 | 0.4890 | 0.5557 | 0.6688 | **0.6887** | 0.5372 | 0.5565 |
| RBF_Gradual_Moderate | 0.8488 | 0.8418 | 0.8920 | **0.9041** | 0.8155 | 0.8893 | 0.7384 | 0.5260 | 0.8364 |
| RBF_Gradual_Severe | 0.8510 | 0.8545 | **0.9019** | 0.9004 | 0.8211 | 0.8944 | 0.7370 | 0.5265 | 0.8621 |
| SEA_Gradual_Recurring | 0.9424 | 0.9343 | 0.9567 | 0.9055 | 0.9198 | **0.9596** | 0.7947 | 0.6167 | 0.9078 |
| SEA_Gradual_Simple_Fast | 0.9737 | 0.9645 | 0.9764 | 0.9558 | 0.9456 | **0.9795** | 0.8191 | 0.6326 | 0.9524 |
| SEA_Gradual_Simple_Slow | 0.9764 | 0.9655 | 0.9762 | 0.9765 | 0.9497 | **0.9811** | 0.8216 | 0.6235 | 0.9425 |
| SINE_Gradual_Recurring | 0.9092 | 0.9006 | **0.9524** | 0.9518 | 0.9312 | 0.9428 | 0.7813 | 0.6252 | 0.8946 |
| STAGGER_Gradual_Chain | 0.7646 | 0.7646 | **0.9495** | 0.9401 | 0.7383 | 0.9203 | 0.6372 | 0.7600 | 0.8823 |
| **Noisy (8)** ||||||||||
| AGRAWAL_Abrupt_Simple_Severe_Noise | 0.8975 | 0.8956 | 0.8518 | 0.8629 | 0.6676 | **0.9401** | 0.7542 | 0.5390 | 0.5918 |
| HYPERPLANE_Gradual_Noise | 0.7640 | 0.7491 | 0.8180 | 0.7593 | 0.8505 | 0.8478 | 0.6808 | 0.5295 | **0.8709** |
| RANDOMTREE_Gradual_Noise | **0.6975** | 0.6697 | 0.5151 | 0.4861 | 0.5338 | 0.6902 | 0.6762 | 0.5529 | 0.5611 |
| RBF_Abrupt_Blip_Noise | 0.8738 | 0.8569 | 0.8965 | **0.9045** | 0.8322 | 0.8859 | 0.7521 | 0.5155 | 0.8408 |
| RBF_Gradual_Severe_Noise | 0.8615 | 0.8502 | 0.8879 | **0.9019** | 0.8164 | 0.8771 | 0.7472 | 0.5296 | 0.8232 |
| SEA_Abrupt_Chain_Noise | 0.9595 | 0.9567 | 0.9680 | 0.9343 | 0.9234 | **0.9712** | 0.7835 | 0.6195 | 0.9150 |
| SINE_Abrupt_Recurring_Noise | 0.7858 | 0.7856 | 0.9811 | **0.9820** | 0.9458 | 0.9731 | 0.6584 | 0.6349 | 0.8851 |
| STAGGER_Abrupt_Chain_Noise | 0.8472 | 0.8472 | **0.9907** | 0.9764 | 0.7733 | 0.9216 | 0.7072 | 0.7603 | 0.8716 |
| **Stationary (6)** ||||||||||
| AGRAWAL_Stationary | **1.0000** | 1.0000 | 0.9413 | 0.9985 | 0.9842 | 0.9967 | 0.8333 | 0.5259 | 1.0000 |
| HYPERPLANE_Stationary | 0.7130 | 0.7397 | 0.8140 | 0.7597 | 0.8792 | 0.8523 | 0.6657 | 0.5243 | **0.8986** |
| RANDOMTREE_Stationary | 0.6225 | 0.5998 | 0.5772 | 0.5393 | 0.3680 | 0.5941 | **0.6409** | 0.5165 | 0.5112 |
| SEA_Stationary | 0.9787 | 0.9718 | 0.9795 | 0.9563 | 0.9519 | **0.9836** | 0.8241 | 0.6423 | 0.9527 |
| SINE_Stationary | 0.9722 | 0.9645 | 0.9868 | **0.9876** | 0.9724 | 0.9824 | 0.8209 | 0.6411 | 0.9743 |
| STAGGER_Stationary | **1.0000** | 1.0000 | 0.9980 | 0.9960 | 0.8049 | 0.9994 | 0.8333 | 0.7656 | 1.0000 |
| **Real (4)** ||||||||||
| AssetNegotiation_F2 | 0.9752 | 0.9653 | 0.9604 | 0.9698 | 0.9491 | **0.9820** | 0.8237 | 0.6213 | 0.9387 |
| AssetNegotiation_F3 | **0.9981** | 0.9981 | 0.9679 | 0.9225 | 0.9694 | 0.9922 | 0.8322 | 0.7638 | 0.9652 |
| AssetNegotiation_F4 | 0.9161 | 0.9212 | 0.9478 | 0.9372 | 0.9375 | 0.6618 | 0.7963 | 0.5611 | **0.9620** |
| Electricity | 0.7443 | 0.7601 | 0.8879 | 0.8769 | 0.8673 | 0.9053 | 0.6783 | 0.6056 | **0.9185** |
| **Multiclass (7)** ||||||||||
| LED_Abrupt_Simple | **0.9363** | 0.9359 | 0.7179 | 0.6078 | 0.1175 | 0.4049 | -- | 0.2195 | -- |
| LED_Gradual_Simple | 0.9835 | **0.9840** | 0.8056 | 0.6894 | 0.1042 | 0.3472 | -- | 0.2187 | -- |
| LED_Stationary | **1.0000** | 1.0000 | 0.8681 | 0.7069 | 0.0000 | 0.2743 | -- | 0.2148 | -- |
| RBF_Stationary | 0.8927 | 0.7934 | 0.9433 | **0.9760** | 0.9002 | 0.9627 | -- | 0.2858 | -- |
| WAVEFORM_Abrupt_Simple | 0.7247 | 0.6564 | 0.8002 | 0.8215 | 0.7733 | **0.8243** | -- | 0.3825 | -- |
| WAVEFORM_Gradual_Simple | 0.7158 | 0.6532 | 0.8013 | 0.8174 | 0.7712 | **0.8225** | -- | 0.3865 | -- |
| WAVEFORM_Stationary | 0.7387 | 0.7395 | 0.8123 | **0.8301** | 0.7732 | 0.8248 | -- | 0.4239 | -- |
|---|---|---|---|---|---|---|---|---|---|
| **Binary Summary (n=41)** ||||||||||
| Mean | 0.8550 | 0.8513 | 0.8777 | 0.8621 | 0.8200 | 0.8923 | 0.7399 | 0.5984 | 0.8385 |
| Std | 0.1071 | 0.1071 | 0.1385 | 0.1510 | 0.1466 | 0.1096 | 0.0677 | 0.0814 | 0.1516 |
| EGIS(NP) W/L/D | -- | 24/8/9 | 12/27/2 | 20/20/1 | 30/10/1 | 8/32/1 | 39/2/0 | 41/0/0 | 24/15/2 |
| Avg Rank | 3.99 | 4.52 | 3.24 | 4.07 | 5.68 | 2.59 | 7.20 | 8.61 | 5.10 |
| **All Summary (n=48)** ||||||||||
| Mean | 0.8551 | 0.8472 | 0.8695 | 0.8499 | 0.7721 | 0.8551 | 0.7399 | 0.5555 | 0.8385 |
| Std | 0.1086 | 0.1127 | 0.1319 | 0.1488 | 0.2265 | 0.1684 | 0.0677 | 0.1321 | 0.1516 |

## Section 4 -- Transition Metrics (TCS, RIR, AMS)

For tab:transitions. Values from egis_transition_metrics.csv.

**CRITICAL**: Paper tab:transitions currently shows TCS ~0.957-1.000.
Correct values from CSV are ~0.22-0.32.

### Overall Summary per Config

| Config | TCS mean | TCS std | RIR mean | RIR std | AMS mean | AMS std | N transitions |
|---|---|---|---|---|---|---|---|
| EXP-500-NP | 0.2225 | 0.0871 | 0.2229 | 0.1553 | 0.2837 | 0.0849 | 1056 |
| EXP-500-P | 0.2218 | 0.0923 | 0.2222 | 0.1561 | 0.2848 | 0.0904 | 1056 |
| EXP-500-P03 | 0.2680 | 0.1273 | 0.3105 | 0.2269 | 0.2817 | 0.0892 | 1056 |
| EXP-1000-NP | 0.2242 | 0.1077 | 0.2294 | 0.1886 | 0.2805 | 0.0845 | 480 |
| EXP-1000-P | 0.2251 | 0.1077 | 0.2312 | 0.1898 | 0.2801 | 0.0841 | 480 |
| EXP-2000-NP | 0.2403 | 0.1265 | 0.2645 | 0.2258 | 0.2641 | 0.0898 | 192 |
| EXP-2000-P | 0.2852 | 0.1251 | 0.3424 | 0.2139 | 0.2722 | 0.0908 | 192 |

### By Drift Type per Config


#### Config: EXP-500-NP

| Drift Type | TCS mean+std | RIR mean+std | AMS mean+std | N |
|---|---|---|---|---|
| abrupt | 0.2336 +/- 0.0946 | 0.2448 +/- 0.1687 | 0.2866 +/- 0.0943 | 352 |
| gradual | 0.2120 +/- 0.0686 | 0.1997 +/- 0.1253 | 0.2857 +/- 0.0805 | 242 |
| noisy | 0.2294 +/- 0.0836 | 0.2233 +/- 0.1543 | 0.3034 +/- 0.0660 | 176 |
| stationary | 0.2128 +/- 0.0943 | 0.2134 +/- 0.1587 | 0.2688 +/- 0.0927 | 198 |
| real | 0.2153 +/- 0.0866 | 0.2196 +/- 0.1597 | 0.2609 +/- 0.0583 | 88 |

#### Config: EXP-500-P

| Drift Type | TCS mean+std | RIR mean+std | AMS mean+std | N |
|---|---|---|---|---|
| abrupt | 0.2303 +/- 0.1003 | 0.2394 +/- 0.1657 | 0.2884 +/- 0.0975 | 352 |
| gradual | 0.2124 +/- 0.0806 | 0.2006 +/- 0.1352 | 0.2930 +/- 0.0879 | 242 |
| noisy | 0.2288 +/- 0.0773 | 0.2195 +/- 0.1446 | 0.3042 +/- 0.0608 | 176 |
| stationary | 0.2176 +/- 0.1091 | 0.2290 +/- 0.1811 | 0.2601 +/- 0.1070 | 198 |
| real | 0.2088 +/- 0.0698 | 0.2030 +/- 0.1218 | 0.2652 +/- 0.0573 | 88 |

#### Config: EXP-500-P03

| Drift Type | TCS mean+std | RIR mean+std | AMS mean+std | N |
|---|---|---|---|---|
| abrupt | 0.2756 +/- 0.1338 | 0.3268 +/- 0.2349 | 0.2818 +/- 0.0988 | 352 |
| gradual | 0.2649 +/- 0.1304 | 0.3012 +/- 0.2339 | 0.2864 +/- 0.0822 | 242 |
| noisy | 0.2823 +/- 0.1110 | 0.3232 +/- 0.2130 | 0.3039 +/- 0.0675 | 176 |
| stationary | 0.2668 +/- 0.1374 | 0.3207 +/- 0.2362 | 0.2628 +/- 0.1019 | 198 |
| real | 0.2202 +/- 0.0818 | 0.2222 +/- 0.1503 | 0.2667 +/- 0.0612 | 88 |

#### Config: EXP-1000-NP

| Drift Type | TCS mean+std | RIR mean+std | AMS mean+std | N |
|---|---|---|---|---|
| abrupt | 0.2364 +/- 0.1132 | 0.2515 +/- 0.2022 | 0.2850 +/- 0.0915 | 160 |
| gradual | 0.2128 +/- 0.0952 | 0.2106 +/- 0.1607 | 0.2732 +/- 0.0818 | 110 |
| noisy | 0.2257 +/- 0.0924 | 0.2182 +/- 0.1674 | 0.3005 +/- 0.0627 | 80 |
| stationary | 0.2167 +/- 0.1290 | 0.2244 +/- 0.2219 | 0.2701 +/- 0.0971 | 90 |
| real | 0.2211 +/- 0.0925 | 0.2260 +/- 0.1601 | 0.2655 +/- 0.0616 | 40 |

#### Config: EXP-1000-P

| Drift Type | TCS mean+std | RIR mean+std | AMS mean+std | N |
|---|---|---|---|---|
| abrupt | 0.2356 +/- 0.1121 | 0.2510 +/- 0.2037 | 0.2835 +/- 0.0912 | 160 |
| gradual | 0.2172 +/- 0.0970 | 0.2184 +/- 0.1641 | 0.2751 +/- 0.0829 | 110 |
| noisy | 0.2268 +/- 0.0901 | 0.2199 +/- 0.1610 | 0.3005 +/- 0.0617 | 80 |
| stationary | 0.2194 +/- 0.1310 | 0.2309 +/- 0.2261 | 0.2676 +/- 0.0954 | 90 |
| real | 0.2139 +/- 0.0921 | 0.2105 +/- 0.1617 | 0.2676 +/- 0.0617 | 40 |

#### Config: EXP-2000-NP

| Drift Type | TCS mean+std | RIR mean+std | AMS mean+std | N |
|---|---|---|---|---|
| abrupt | 0.2836 +/- 0.1275 | 0.3462 +/- 0.2290 | 0.2660 +/- 0.1014 | 64 |
| gradual | 0.2333 +/- 0.1282 | 0.2433 +/- 0.2280 | 0.2746 +/- 0.0734 | 44 |
| noisy | 0.2651 +/- 0.1297 | 0.3015 +/- 0.2548 | 0.2674 +/- 0.0927 | 32 |
| stationary | 0.1710 +/- 0.0964 | 0.1467 +/- 0.1445 | 0.2548 +/- 0.0992 | 36 |
| real | 0.1927 +/- 0.0920 | 0.1868 +/- 0.1568 | 0.2427 +/- 0.0484 | 16 |

#### Config: EXP-2000-P

| Drift Type | TCS mean+std | RIR mean+std | AMS mean+std | N |
|---|---|---|---|---|
| abrupt | 0.2897 +/- 0.1278 | 0.3573 +/- 0.2204 | 0.2759 +/- 0.1060 | 64 |
| gradual | 0.2678 +/- 0.1311 | 0.3106 +/- 0.2283 | 0.2652 +/- 0.0951 | 44 |
| noisy | 0.3204 +/- 0.1223 | 0.3910 +/- 0.1999 | 0.2873 +/- 0.0654 | 32 |
| stationary | 0.2843 +/- 0.1256 | 0.3521 +/- 0.2138 | 0.2566 +/- 0.0912 | 36 |
| real | 0.2466 +/- 0.0915 | 0.2512 +/- 0.1486 | 0.2811 +/- 0.0469 | 16 |

## Section 5 -- Model Parameters (tab:baseline_params)

| Model | Framework | Key Parameters | Evaluation Protocol |
|---|---|---|---|
| EGIS | Python (custom) | pop=120, gen=200, elitism=0.1, mut=0.8, DT seeding=0.8, gamma={0.0,0.1,0.3} | Train-then-test (chunk-based) |
| ARF | River | n_models=10, defaults | Prequential (per-instance) |
| SRP | River | n_models=10, defaults | Prequential (per-instance) |
| HAT | River | defaults (grace_period=200, split_confidence=1e-7) | Prequential (per-instance) |
| ROSE | MOA/ROSE | WindowAUC evaluator, window=500 | Prequential (windowed) |
| ACDWM | Python/DWMIL | theta=0.001, err_func=gm, r=1.0, binary only | Prequential (per-instance) |
| ERulesD2S | MOA/JCLEC | pop=25, gen=50, rules_per_class=5 | Prequential (windowed) |
| CDCMS.CIL | MOA/CIL | f=chunk_size, holdout per-chunk eval | Holdout (chunk-based) |

## Section 6 -- Statistical Rankings and Tests

For tab:stat_tests, table_ix_ranking, fig:cd_diagram

Rankings use NP configs for EGIS vs baselines comparison.


### Config: EXP-500-NP

#### Binary (41 datasets, 8 models)

| Model | Mean Rank |
|---|---|
| ROSE | 2.41 |
| ARF | 2.85 |
| SRP | 3.68 |
| EGIS | 4.45 |
| CDCMS | 4.65 |
| ACDWM | 4.95 |
| HAT | 5.49 |
| ERulesD2S | 7.51 |

**Friedman test**: chi2=123.06, p=1.77e-23
**Nemenyi CD** (alpha=0.05, approx): 1.39
**k**=8, **n**=41

#### Pairwise Wilcoxon (EGIS vs others, EXP-500-NP)

| Comparison | p-value | Bonferroni p | Cliff's delta | Effect Size |
|---|---|---|---|---|
| EGIS vs ACDWM | 9.53e-01 | 1.00e+00 | 0.133 | negligible |
| EGIS vs ARF | 7.57e-02 | 5.30e-01 | -0.172 | small |
| EGIS vs CDCMS | 5.91e-01 | 1.00e+00 | 0.030 | negligible |
| EGIS vs ERulesD2S | 9.09e-13 | 6.37e-12 | 0.843 | large |
| EGIS vs HAT | 2.74e-02 | 1.92e-01 | 0.180 | small |
| EGIS vs ROSE | 8.39e-05 | 5.87e-04 | -0.184 | small |
| EGIS vs SRP | 8.25e-02 | 5.77e-01 | -0.116 | negligible |

#### Multiclass (7 datasets, 6 models)

| Model | Mean Rank |
|---|---|
| SRP | 2.14 |
| ARF | 2.57 |
| ROSE | 2.57 |
| EGIS | 3.29 |
| HAT | 4.86 |
| ERulesD2S | 5.57 |

### Config: EXP-1000-NP

#### Binary (41 datasets, 8 models)

| Model | Mean Rank |
|---|---|
| ROSE | 2.41 |
| ARF | 2.88 |
| EGIS | 3.70 |
| SRP | 3.76 |
| CDCMS | 4.62 |
| HAT | 5.29 |
| ACDWM | 5.76 |
| ERulesD2S | 7.59 |

**Friedman test**: chi2=136.17, p=3.23e-26
**Nemenyi CD** (alpha=0.05, approx): 1.39
**k**=8, **n**=41

#### Pairwise Wilcoxon (EGIS vs others, EXP-1000-NP)

| Comparison | p-value | Bonferroni p | Cliff's delta | Effect Size |
|---|---|---|---|---|
| EGIS vs ACDWM | 3.39e-06 | 2.37e-05 | 0.402 | medium |
| EGIS vs ARF | 1.63e-01 | 1.00e+00 | -0.133 | negligible |
| EGIS vs CDCMS | 9.29e-02 | 6.50e-01 | 0.084 | negligible |
| EGIS vs ERulesD2S | 9.09e-13 | 6.37e-12 | 0.926 | large |
| EGIS vs HAT | 7.45e-04 | 5.22e-03 | 0.197 | small |
| EGIS vs ROSE | 4.90e-06 | 3.43e-05 | -0.179 | small |
| EGIS vs SRP | 8.78e-01 | 1.00e+00 | -0.055 | negligible |

#### Multiclass (7 datasets, 6 models)

| Model | Mean Rank |
|---|---|
| SRP | 2.29 |
| ROSE | 2.43 |
| ARF | 2.57 |
| EGIS | 3.29 |
| HAT | 4.86 |
| ERulesD2S | 5.57 |

### Config: EXP-2000-NP

#### Binary (41 datasets, 8 models)

| Model | Mean Rank |
|---|---|
| ROSE | 2.39 |
| ARF | 2.93 |
| SRP | 3.66 |
| EGIS | 3.68 |
| CDCMS | 4.49 |
| HAT | 4.98 |
| ACDWM | 6.27 |
| ERulesD2S | 7.61 |

**Friedman test**: chi2=145.81, p=3.08e-28
**Nemenyi CD** (alpha=0.05, approx): 1.39
**k**=8, **n**=41

#### Pairwise Wilcoxon (EGIS vs others, EXP-2000-NP)

| Comparison | p-value | Bonferroni p | Cliff's delta | Effect Size |
|---|---|---|---|---|
| EGIS vs ACDWM | 3.26e-08 | 2.28e-07 | 0.611 | large |
| EGIS vs ARF | 4.37e-02 | 3.06e-01 | -0.200 | small |
| EGIS vs CDCMS | 2.25e-01 | 1.00e+00 | 0.002 | negligible |
| EGIS vs ERulesD2S | 9.09e-13 | 6.37e-12 | 0.930 | large |
| EGIS vs HAT | 6.11e-03 | 4.28e-02 | 0.115 | negligible |
| EGIS vs ROSE | 2.45e-05 | 1.71e-04 | -0.248 | small |
| EGIS vs SRP | 5.81e-01 | 1.00e+00 | -0.135 | negligible |

#### Multiclass (7 datasets, 6 models)

| Model | Mean Rank |
|---|---|
| SRP | 2.14 |
| ARF | 2.57 |
| ROSE | 2.57 |
| EGIS | 3.29 |
| HAT | 4.86 |
| ERulesD2S | 5.57 |

### EGIS Variants Comparison (Binary)

Comparing EGIS penalty variants against each other.


#### Chunk Size: 500 (3 variants)

| EGIS Variant | Mean G-Mean | Mean Rank |
|---|---|---|
| EGIS (NP) | 0.8591 | 1.46 |
| EGIS (P) | 0.8596 | 1.77 |
| EGIS (P03) | 0.8464 | 2.77 |

Wilcoxon EGIS (NP) vs EGIS (P): p=4.1280e-01

#### Chunk Size: 1000 (2 variants)

| EGIS Variant | Mean G-Mean | Mean Rank |
|---|---|---|
| EGIS (NP) | 0.8692 | 1.46 |
| EGIS (P) | 0.8688 | 1.54 |

Wilcoxon EGIS (NP) vs EGIS (P): p=2.3672e-01

#### Chunk Size: 2000 (2 variants)

| EGIS Variant | Mean G-Mean | Mean Rank |
|---|---|---|
| EGIS (NP) | 0.8550 | 1.30 |
| EGIS (P) | 0.8513 | 1.70 |

Wilcoxon EGIS (NP) vs EGIS (P): p=6.0697e-03

## Section 7 -- EGIS Rule Complexity (tab:complexity_detailed)

### Overall Complexity per Config

**Note**: 'N chunk records' counts the number of per-chunk RulesHistory entries available.
For 4 datasets (AGRAWAL_Stationary, RBF_Gradual_Severe_Noise, SEA_Gradual_Simple_Fast,
WAVEFORM_Abrupt_Simple), some configs have fewer records than expected due to incomplete
RulesHistory logging (I/O interruption). Avg values remain correct for recorded chunks.

| Config | Avg Rules | Avg Cond/Rule | Avg AND/chunk | Avg OR/chunk | Avg Total Cond/chunk | N chunk records |
|---|---|---|---|---|---|---|
| EXP-500-NP | 14.99 | 5.30 | 68.37 | 3.59 | 86.94 | 1104 |
| EXP-500-P | 13.47 | 4.62 | 54.37 | 1.73 | 69.58 | 1104 |
| EXP-500-P03 | 12.11 | 4.65 | 51.27 | 2.47 | 65.85 | 1104 |
| EXP-1000-NP | 19.71 | 5.67 | 98.64 | 3.33 | 121.68 | 528 |
| EXP-1000-P | 19.64 | 5.66 | 98.18 | 3.32 | 121.14 | 528 |
| EXP-2000-NP | 23.62 | 5.86 | 126.58 | 3.38 | 153.59 | 240 |
| EXP-2000-P | 17.89 | 5.34 | 92.52 | 2.77 | 113.17 | 240 |

### Complexity by Drift Type (EXP-500-NP)

| Drift Type | Avg Rules | Avg Cond/Rule | N chunks |
|---|---|---|---|
| abrupt | 14.83 | 5.55 | 368 |
| gradual | 15.61 | 5.54 | 253 |
| noisy | 15.41 | 5.35 | 184 |
| stationary | 14.45 | 4.95 | 207 |
| real | 14.27 | 4.40 | 92 |

## Section 8 -- Per-Dataset Analysis Data

Data for generating figures (fig:chunk_size_effect, fig:config_comparison, etc.)

### Chunk Size Effect on EGIS (fig:chunk_size_effect)

| Drift Type | EXP-500-NP | EXP-1000-NP | EXP-2000-NP |
|---|---|---|---|
| abrupt | 0.8540 +/- 0.1190 | 0.8614 +/- 0.1075 | 0.8366 +/- 0.1028 |
| gradual | 0.8541 +/- 0.1201 | 0.8651 +/- 0.1075 | 0.8566 +/- 0.1090 |
| noisy | 0.8456 +/- 0.1160 | 0.8507 +/- 0.1025 | 0.8358 +/- 0.0829 |
| stationary | 0.8565 +/- 0.1647 | 0.8664 +/- 0.1602 | 0.8797 +/- 0.1482 |
| real | 0.8900 +/- 0.1257 | 0.9125 +/- 0.0984 | 0.9085 +/- 0.1147 |

### Penalty Effect on EGIS (tab:penalty_effect)

| Chunk Size | No Penalty | Penalty 0.1 | Penalty 0.3 | Delta (P-NP) |
|---|---|---|---|---|
| 500 | 0.8561 +/- 0.1238 | 0.8549 +/- 0.1224 | 0.8365 +/- 0.1384 | -0.0012 |
| 1000 | 0.8657 +/- 0.1137 | 0.8663 +/- 0.1136 | N/A | 0.0006 |
| 2000 | 0.8551 +/- 0.1097 | 0.8472 +/- 0.1139 | N/A | -0.0079 |

### All EGIS Configs by Drift Type (fig:config_comparison)

| Drift Type | EXP-500-NP | EXP-500-P | EXP-500-P03 | EXP-1000-NP | EXP-1000-P | EXP-2000-NP | EXP-2000-P |
|---| ---|---|---|---|---|---|---|
| abrupt | 0.8540 | 0.8545 | 0.8351 | 0.8614 | 0.8613 | 0.8366 | 0.8309 |
| gradual | 0.8541 | 0.8547 | 0.8345 | 0.8651 | 0.8641 | 0.8566 | 0.8463 |
| noisy | 0.8456 | 0.8417 | 0.8288 | 0.8507 | 0.8506 | 0.8358 | 0.8264 |
| stationary | 0.8565 | 0.8520 | 0.8261 | 0.8664 | 0.8715 | 0.8797 | 0.8676 |
| real | 0.8900 | 0.8894 | 0.8865 | 0.9125 | 0.9118 | 0.9085 | 0.9112 |

### Performance Distribution (fig:boxplots, EXP-500-NP, binary)

| Model | Min | Q1 | Median | Q3 | Max | Mean | Std |
|---|---|---|---|---|---|---|---|
| EGIS | 0.5425 | 0.8069 | 0.9133 | 0.9510 | 1.0000 | 0.8591 | 0.1232 |
| ARF | 0.5112 | 0.8316 | 0.9335 | 0.9772 | 0.9980 | 0.8776 | 0.1388 |
| SRP | 0.4766 | 0.8356 | 0.9215 | 0.9731 | 0.9983 | 0.8691 | 0.1480 |
| HAT | 0.4321 | 0.7584 | 0.8556 | 0.9459 | 0.9813 | 0.8149 | 0.1484 |
| ROSE | 0.5941 | 0.8634 | 0.9215 | 0.9785 | 0.9994 | 0.8923 | 0.1096 |
| ACDWM | 0.5748 | 0.8385 | 0.8905 | 0.9313 | 0.9583 | 0.8603 | 0.0917 |
| ERulesD2S | 0.5184 | 0.5295 | 0.5616 | 0.6384 | 0.7969 | 0.6050 | 0.0856 |
| CDCMS | 0.5157 | 0.8196 | 0.9188 | 0.9468 | 1.0000 | 0.8460 | 0.1499 |

### Drift Detection Summary

Total records: 336

#### Average Drifts Detected per Config

| Config | Avg Drifts | Max Drifts |
|---|---|---|
| EXP-500-NP | 1.19 | 7 |
| EXP-500-P | 1.10 | 7 |
| EXP-500-P03 | 2.42 | 11 |
| EXP-1000-NP | 0.69 | 3 |
| EXP-1000-P | 0.67 | 3 |
| EXP-2000-NP | 0.42 | 2 |
| EXP-2000-P | 0.54 | 2 |

## Section 9 -- Rule Evolution Data

Data for fig:rule_evolution, fig:transition_matrix, fig:evolution_heatmaps


### Rule Evolution Summary (336 records)

#### Rule Change Counts per Config

| Config | Avg Unchanged | Avg Modified | Avg New | Avg Deleted | Avg Similarity | Avg Rules/Chunk | N |
|---|---|---|---|---|---|---|---|
| EXP-500-NP | 0.15 | 8.22 | 6.62 | 6.58 | 0.7384 | 15.0 | 48 |
| EXP-500-P | 0.38 | 6.98 | 6.06 | 6.10 | 0.7406 | 13.5 | 48 |
| EXP-500-P03 | 0.22 | 5.67 | 6.09 | 6.19 | 0.7418 | 12.1 | 48 |
| EXP-1000-NP | 0.17 | 11.44 | 8.09 | 7.96 | 0.7420 | 19.7 | 48 |
| EXP-1000-P | 0.17 | 11.36 | 8.10 | 7.80 | 0.7414 | 19.6 | 48 |
| EXP-2000-NP | 0.16 | 14.77 | 8.71 | 8.85 | 0.7470 | 23.6 | 48 |
| EXP-2000-P | 0.24 | 7.30 | 8.92 | 9.64 | 0.7444 | 17.9 | 48 |

#### Rule Changes by Drift Type (EXP-500-NP)

| Drift Type | Avg Modified | Avg New | Avg Deleted | Avg Similarity | N |
|---|---|---|---|---|---|
| abrupt | 7.64 | 7.05 | 6.95 | 0.7336 | 16 |
| gradual | 8.79 | 6.64 | 6.52 | 0.7354 | 11 |
| noisy | 8.44 | 6.80 | 6.91 | 0.7297 | 8 |
| stationary | 7.93 | 6.37 | 6.37 | 0.7511 | 9 |
| real | 9.17 | 5.09 | 5.10 | 0.7553 | 4 |

### AST-based Detailed Evolution Analysis (4512 records)

#### AST-based Transition Metrics per Config

| Config | TCS mean | RIR mean | AMS mean | N |
|---|---|---|---|---|
| EXP-500-NP | 0.4041 | 0.5319 | 0.4944 | 1056 |
| EXP-500-P | 0.3953 | 0.5201 | 0.4935 | 1056 |
| EXP-500-P03 | 0.4170 | 0.5664 | 0.4534 | 1056 |
| EXP-1000-NP | 0.3937 | 0.5107 | 0.4895 | 480 |
| EXP-1000-P | 0.3935 | 0.5100 | 0.4887 | 480 |
| EXP-2000-NP | 0.3848 | 0.4827 | 0.4982 | 192 |
| EXP-2000-P | 0.4225 | 0.5718 | 0.4700 | 192 |
