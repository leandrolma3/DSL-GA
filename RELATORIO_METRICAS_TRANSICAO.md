# Transition Metrics Report (RIR, AMS, TCS)

Generated automatically from `experiments_unified/` directories.

**Formulas:**
- **RIR** (Rule Instability Rate) = (new + deleted) / (rules_from + rules_to)
- **AMS** (Average Modification Severity) = mean(1 - similarity) for modified rules
- **TCS** (Transition Change Score) = 0.6 * RIR + 0.4 * prop_modified * AMS

**Excluded datasets:** CovType, IntelLabSensors, PokerHand, Shuttle

## 1. Summary by Drift Type

### EXP-1000

| Drift Type | N datasets | RIR | AMS | TCS |
|---|---|---|---|---|
| unknown | 4 | 0.2735 +/- 0.0488 | 0.2637 +/- 0.0246 | 0.2454 +/- 0.0245 |

### EXP-1000-P

| Drift Type | N datasets | RIR | AMS | TCS |
|---|---|---|---|---|
| unknown | 4 | 0.2717 +/- 0.0412 | 0.2647 +/- 0.0229 | 0.2443 +/- 0.0205 |

### EXP-2000

| Drift Type | N datasets | RIR | AMS | TCS |
|---|---|---|---|---|
| unknown | 7 | 0.2193 +/- 0.1362 | 0.2498 +/- 0.0429 | 0.2125 +/- 0.0765 |

### EXP-500

| Drift Type | N datasets | RIR | AMS | TCS |
|---|---|---|---|---|
| unknown | 3 | 0.2544 +/- 0.0452 | 0.2728 +/- 0.0246 | 0.2372 +/- 0.0197 |

### EXP-500-P

| Drift Type | N datasets | RIR | AMS | TCS |
|---|---|---|---|---|
| unknown | 3 | 0.2526 +/- 0.0435 | 0.2745 +/- 0.0279 | 0.2360 +/- 0.0182 |

## 2. Results by Individual Dataset

### EXP-1000

| Dataset | Drift Type | Runs | RIR | AMS | TCS |
|---|---|---|---|---|---|
| batch_1 | unknown | 13 | 0.2577 +/- 0.1357 | 0.2981 +/- 0.0470 | 0.2435 +/- 0.0684 |
| batch_2 | unknown | 13 | 0.2116 +/- 0.1060 | 0.2755 +/- 0.0624 | 0.2131 +/- 0.0630 |
| batch_3 | unknown | 13 | 0.3472 +/- 0.2711 | 0.2363 +/- 0.1086 | 0.2821 +/- 0.1379 |
| batch_4 | unknown | 13 | 0.2777 +/- 0.2426 | 0.2449 +/- 0.0882 | 0.2428 +/- 0.1283 |

### EXP-1000-P

| Dataset | Drift Type | Runs | RIR | AMS | TCS |
|---|---|---|---|---|---|
| batch_1 | unknown | 13 | 0.2542 +/- 0.1352 | 0.2965 +/- 0.0478 | 0.2414 +/- 0.0679 |
| batch_2 | unknown | 13 | 0.2211 +/- 0.1146 | 0.2765 +/- 0.0626 | 0.2180 +/- 0.0676 |
| batch_3 | unknown | 13 | 0.3339 +/- 0.2454 | 0.2420 +/- 0.1014 | 0.2756 +/- 0.1258 |
| batch_4 | unknown | 13 | 0.2774 +/- 0.2441 | 0.2438 +/- 0.0869 | 0.2425 +/- 0.1293 |

### EXP-2000

| Dataset | Drift Type | Runs | RIR | AMS | TCS |
|---|---|---|---|---|---|
| batch_1 | unknown | 7 | 0.3546 +/- 0.0754 | 0.3204 +/- 0.0210 | 0.3039 +/- 0.0397 |
| batch_2 | unknown | 5 | 0.1902 +/- 0.1232 | 0.2702 +/- 0.0349 | 0.2032 +/- 0.0482 |
| batch_3 | unknown | 5 | 0.2103 +/- 0.1364 | 0.2804 +/- 0.0602 | 0.2185 +/- 0.0797 |
| batch_4 | unknown | 3 | 0.0781 +/- 0.0155 | 0.2129 +/- 0.0397 | 0.1241 +/- 0.0083 |
| batch_5 | unknown | 3 | 0.4747 +/- 0.3151 | 0.1787 +/- 0.0967 | 0.3382 +/- 0.1554 |
| batch_6 | unknown | 4 | 0.0740 +/- 0.0402 | 0.2469 +/- 0.0661 | 0.1271 +/- 0.0531 |
| batch_7 | unknown | 6 | 0.1533 +/- 0.0913 | 0.2392 +/- 0.0612 | 0.1724 +/- 0.0610 |

### EXP-500

| Dataset | Drift Type | Runs | RIR | AMS | TCS |
|---|---|---|---|---|---|
| batch_1 | unknown | 18 | 0.2486 +/- 0.0908 | 0.2955 +/- 0.0590 | 0.2389 +/- 0.0457 |
| batch_2 | unknown | 17 | 0.2021 +/- 0.0797 | 0.2843 +/- 0.0653 | 0.2123 +/- 0.0556 |
| batch_3 | unknown | 17 | 0.3124 +/- 0.2350 | 0.2386 +/- 0.0815 | 0.2605 +/- 0.1201 |

### EXP-500-P

| Dataset | Drift Type | Runs | RIR | AMS | TCS |
|---|---|---|---|---|---|
| batch_1 | unknown | 18 | 0.2365 +/- 0.0837 | 0.3042 +/- 0.0473 | 0.2342 +/- 0.0510 |
| batch_2 | unknown | 17 | 0.2093 +/- 0.0662 | 0.2819 +/- 0.0756 | 0.2147 +/- 0.0565 |
| batch_3 | unknown | 17 | 0.3121 +/- 0.2199 | 0.2372 +/- 0.0895 | 0.2592 +/- 0.1131 |

## 3. Comparison EXP-500 vs EXP-1000 vs EXP-2000

Aggregated across all datasets (without penalty variants).

| Experiment | N | RIR | AMS | TCS |
|---|---|---|---|---|
| EXP-500 | 3 | 0.2544 +/- 0.0452 | 0.2728 +/- 0.0246 | 0.2372 +/- 0.0197 |
| EXP-1000 | 4 | 0.2735 +/- 0.0488 | 0.2637 +/- 0.0246 | 0.2454 +/- 0.0245 |
| EXP-2000 | 7 | 0.2193 +/- 0.1362 | 0.2498 +/- 0.0429 | 0.2125 +/- 0.0765 |

### With vs Without Penalty

| Experiment | N | RIR | AMS | TCS |
|---|---|---|---|---|
| EXP-500 | 3 | 0.2544 +/- 0.0452 | 0.2728 +/- 0.0246 | 0.2372 +/- 0.0197 |
| EXP-500-P | 3 | 0.2526 +/- 0.0435 | 0.2745 +/- 0.0279 | 0.2360 +/- 0.0182 |
| EXP-1000 | 4 | 0.2735 +/- 0.0488 | 0.2637 +/- 0.0246 | 0.2454 +/- 0.0245 |
| EXP-1000-P | 4 | 0.2717 +/- 0.0412 | 0.2647 +/- 0.0229 | 0.2443 +/- 0.0205 |
| EXP-2000 | 7 | 0.2193 +/- 0.1362 | 0.2498 +/- 0.0429 | 0.2125 +/- 0.0765 |

## 4. Drift Type x Chunk Size (main experiments only)

| Drift Type | EXP-500 TCS | EXP-1000 TCS | EXP-2000 TCS |
|---|---|---|---|
| unknown | 0.2372 +/- 0.0197 | 0.2454 +/- 0.0245 | 0.2125 +/- 0.0765 |

## 5. Pattern Analysis

### EXP-500

**Top 5 most stable (lowest TCS):**

- batch_2 (unknown): TCS=0.2123
- batch_1 (unknown): TCS=0.2389
- batch_3 (unknown): TCS=0.2605

**Top 5 most unstable (highest TCS):**

- batch_2 (unknown): TCS=0.2123
- batch_1 (unknown): TCS=0.2389
- batch_3 (unknown): TCS=0.2605

### EXP-1000

**Top 5 most stable (lowest TCS):**

- batch_2 (unknown): TCS=0.2131
- batch_4 (unknown): TCS=0.2428
- batch_1 (unknown): TCS=0.2435
- batch_3 (unknown): TCS=0.2821

**Top 5 most unstable (highest TCS):**

- batch_2 (unknown): TCS=0.2131
- batch_4 (unknown): TCS=0.2428
- batch_1 (unknown): TCS=0.2435
- batch_3 (unknown): TCS=0.2821

### EXP-2000

**Top 5 most stable (lowest TCS):**

- batch_4 (unknown): TCS=0.1241
- batch_6 (unknown): TCS=0.1271
- batch_7 (unknown): TCS=0.1724
- batch_2 (unknown): TCS=0.2032
- batch_3 (unknown): TCS=0.2185

**Top 5 most unstable (highest TCS):**

- batch_7 (unknown): TCS=0.1724
- batch_2 (unknown): TCS=0.2032
- batch_3 (unknown): TCS=0.2185
- batch_1 (unknown): TCS=0.3039
- batch_5 (unknown): TCS=0.3382

### Stationary vs Drift Datasets

