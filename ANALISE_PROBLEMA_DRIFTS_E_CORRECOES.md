# ANALYSIS: Drift Configuration Problem in Batch 1 Datasets

**Date**: 2025-01-18
**Issue**: Drifts occurring OUTSIDE the 6-chunk experimental window
**Impact**: CRITICAL - Invalidates current Batch 1 results

---

## EXECUTIVE SUMMARY

All 5 Batch 1 datasets have **CRITICAL configuration errors** where concept drifts occur outside the experimental window of 6 chunks (5 processed). This invalidates the comparative analysis results.

**Root Cause**: Configuration designed for `chunk_size=3000` but executed with `chunk_size=1000` without proportional adjustment of `duration_chunks` values.

**Solution**: Reconfigure all datasets to fit drifts within 6 chunks (5000 training instances).

---

## PROBLEM IDENTIFICATION

### Actual Execution Parameters (confirmed from run_config.json):
- **chunk_size**: 1000 instances
- **num_chunks_processed**: 5 chunks
- **Total instances**: 6000 (chunks 0-5)
- **Training range**: chunks 0-4 (5000 instances)
- **Testing range**: chunks 1-5 (used for evaluation only)

### Expected vs Reality:
The config_batch_1.yaml was designed for:
- **chunk_size**: 3000 instances
- **num_chunks**: 8
- **Total**: 24,000 instances

But the system executed with **MUCH SMALLER** chunks (1000 instead of 3000), causing drifts to be positioned at instances 4000, 5000, 6000, 8000 - most of which are **OUTSIDE** the 5000-instance training range!

---

## DETAILED ANALYSIS BY DATASET

### 1. SEA_Abrupt_Simple

**Current Configuration** (config_batch_1.yaml lines 411-419):
```yaml
concept_sequence:
  - concept_id: f1
    duration_chunks: 5
  - concept_id: f3
    duration_chunks: 5
```

**Actual Execution** (run_config.json):
- f1: chunks 0-4 (instances 0-5000) ✓
- f3: chunks 5-9 (instances 5000-10000) ✗

**PROBLEM**:
- Total execution: only 6000 instances (6 chunks)
- Drift positioned at instance 5000 (chunk 5)
- **Model trains ONLY on f1** (chunks 0-4)
- **Model NEVER sees f3 during training** (f3 only appears in final test chunk)
- Plot confirms: drift marker at 5000 instances with 0.0% severity

**Visual Evidence**: Plot_AccuracyPeriodic_SEA_Abrupt_Simple_Run1.png shows drift at 5000 (end of range)

**PROPOSED CORRECTION**:
```yaml
concept_sequence:
  - concept_id: f1
    duration_chunks: 3    # chunks 0-2 (0-3000 instances)
  - concept_id: f3
    duration_chunks: 3    # chunks 3-5 (3000-6000 instances)
```
- **Drift position**: chunk 3 (3000 instances) - WITHIN training range
- **Model will train on**: f1 in chunks 0-2, f3 in chunks 3-4

---

### 2. AGRAWAL_Abrupt_Simple_Severe

**Current Configuration** (config_batch_1.yaml lines 440-448):
```yaml
concept_sequence:
  - concept_id: f1
    duration_chunks: 5
  - concept_id: f6
    duration_chunks: 5
```

**Actual Execution**:
- f1: chunks 0-4 (instances 0-5000) ✓
- f6: chunks 5-9 (instances 5000-10000) ✗

**PROBLEM**:
- **IDENTICAL issue to SEA**
- Model trains only on f1, never sees f6 during training
- Drift at 5000 instances (outside training range)

**PROPOSED CORRECTION**:
```yaml
concept_sequence:
  - concept_id: f1
    duration_chunks: 3    # chunks 0-2 (0-3000 instances)
  - concept_id: f6
    duration_chunks: 3    # chunks 3-5 (3000-6000 instances)
```
- **Drift position**: chunk 3 (3000 instances)
- **Model will train on**: f1 in chunks 0-2, f6 in chunks 3-4

---

### 3. RBF_Abrupt_Severe

**Current Configuration** (config_batch_1.yaml lines 449-457):
```yaml
concept_sequence:
  - concept_id: c1
    duration_chunks: 5
  - concept_id: c2_severe
    duration_chunks: 5
```

**Actual Execution**:
- c1: chunks 0-4 (instances 0-5000) ✓
- c2_severe: chunks 5-9 (instances 5000-10000) ✗

**PROBLEM**:
- **IDENTICAL issue to SEA and AGRAWAL**
- Model trains only on c1, never sees c2_severe during training
- Drift at 5000 instances (outside training range)

**PROPOSED CORRECTION**:
```yaml
concept_sequence:
  - concept_id: c1
    duration_chunks: 3    # chunks 0-2 (0-3000 instances)
  - concept_id: c2_severe
    duration_chunks: 3    # chunks 3-5 (3000-6000 instances)
```
- **Drift position**: chunk 3 (3000 instances)
- **Model will train on**: c1 in chunks 0-2, c2_severe in chunks 3-4

---

### 4. STAGGER_Abrupt_Chain ⚠️ MOST CRITICAL

**Current Configuration** (config_batch_1.yaml lines 458-468):
```yaml
concept_sequence:
  - concept_id: f1
    duration_chunks: 4
  - concept_id: f2
    duration_chunks: 4
  - concept_id: f3
    duration_chunks: 4
```

**Actual Execution**:
- f1: chunks 0-3 (instances 0-4000) ✓
- f2: chunks 4-7 (instances 4000-8000) ⚠️ PARTIALLY OUTSIDE
- f3: chunks 8-11 (instances 8000-12000) ✗ COMPLETELY OUTSIDE

**PROBLEM**:
- **SEVERE MISMATCH**: Configured for 12 chunks, executed with only 6 chunks!
- Drift 1 at instance 4000 (chunk 4) - OUTSIDE training range (we train chunks 0-4, but chunk 4 already has new concept)
- Drift 2 at instance 8000 (chunk 8) - COMPLETELY OUTSIDE execution range!
- **Model trains on**: f1 (chunks 0-3), then sees f2 start in chunk 4 (PROBLEM: chunk 4 is last training chunk!)
- **Model NEVER sees f3**

**Visual Evidence**: Plot_AccuracyPeriodic_STAGGER_Abrupt_Chain_Run1.png shows:
- Drift markers at 4000 and 8000 instances
- Severe accuracy drop at 4000 instances (from 100% to ~41%)
- X-axis extends to 8000+ instances but we only have 6000 total
- Both drifts labeled with 0.0% severity

**PROPOSED CORRECTION**:
```yaml
concept_sequence:
  - concept_id: f1
    duration_chunks: 2    # chunks 0-1 (0-2000 instances)
  - concept_id: f2
    duration_chunks: 2    # chunks 2-3 (2000-4000 instances)
  - concept_id: f3
    duration_chunks: 2    # chunks 4-5 (4000-6000 instances)
```
- **Drift 1 position**: chunk 2 (2000 instances) - WITHIN training range
- **Drift 2 position**: chunk 4 (4000 instances) - WITHIN training range
- **Model will train on**: f1 (chunks 0-1), f2 (chunks 2-3), f3 (chunk 4)

---

### 5. HYPERPLANE_Abrupt_Simple ⚠️ NO DRIFT IN RANGE

**Current Configuration** (config_batch_1.yaml lines 469-477):
```yaml
concept_sequence:
  - concept_id: plane1
    duration_chunks: 6
  - concept_id: plane2
    duration_chunks: 6
```

**Actual Execution**:
- plane1: chunks 0-5 (instances 0-6000) ✓ COVERS ENTIRE RANGE
- plane2: chunks 6-11 (instances 6000-12000) ✗ COMPLETELY OUTSIDE

**PROBLEM**:
- **WORST CASE**: NO drift occurs within the experimental window!
- Drift positioned at instance 6000 (chunk 6)
- **Model trains EXCLUSIVELY on plane1** for all 5 training chunks
- **Model NEVER sees plane2** at all
- This explains why the plot shows NO performance drop - there was no drift!

**Visual Evidence**: Plot_AccuracyPeriodic_HYPERPLANE_Abrupt_Simple_Run1.png shows:
- Drift marker at 6000 instances (at the very end)
- Drift labeled "plane1 → plane2 (0.0%)"
- Flat performance line (~74-80%) across all chunks - NO drift response
- Model never encountered a concept change

**PROPOSED CORRECTION**:
```yaml
concept_sequence:
  - concept_id: plane1
    duration_chunks: 3    # chunks 0-2 (0-3000 instances)
  - concept_id: plane2
    duration_chunks: 3    # chunks 3-5 (3000-6000 instances)
```
- **Drift position**: chunk 3 (3000 instances) - WITHIN training range
- **Model will train on**: plane1 in chunks 0-2, plane2 in chunks 3-4

---

## CONCEPT DIFFERENCE CALCULATION ANALYSIS

### Implementation Verification (analyze_concept_difference.py)

The `calculate_difference()` function (lines 230-243) correctly implements **Equation 6 from ChiuTNNLS2020.pdf**:

```python
def calculate_difference(concept_func, params_a, params_b, n_samples, feature_bounds, n_features):
    samples = uniform_sample_space(n_samples, feature_bounds, n_features)
    total_diff = 0
    for i in range(n_samples):
        instance = samples[i, :]
        label_a = concept_func(instance, params_a)
        label_b = concept_func(instance, params_b)
        total_diff += abs(label_a - label_b)
    return (total_diff / n_samples) * 100.0
```

**This matches the paper**: `diff(fa, fb) = Σ|y_fa - y_fb| / n`

### Why We See 0.0% Severity

The 0.0% severity values shown in the plots are **NOT from analyze_concept_difference.py**. They are calculated during data generation and appear to be placeholder values or incorrectly calculated.

**Evidence**:
1. The `analyze_concept_difference.py` script uses `severity_samples: 20000` (config line 85)
2. It correctly implements sampling from the feature space
3. The concept functions (SEA, AGRAWAL, RBF, STAGGER, HYPERPLANE) are properly implemented

**The 0.0% likely indicates**:
- The drift severity calculation during stream generation uses a different (possibly incorrect) method
- OR the concepts are being compared at the wrong time/position
- OR it's a placeholder value that wasn't properly calculated

### Action Required:
After fixing the drift positions, we should:
1. Re-run `analyze_concept_difference.py` with corrected configurations
2. Verify the severity values match what appears in the plots
3. Ensure the drift severity is calculated at the correct positions (new drift points at 2000, 3000, 4000)

---

## IMPACT ASSESSMENT

### Scientific Validity: ❌ INVALID
- Current results **CANNOT be published** or used for scientific comparison
- Models were evaluated on streams where:
  - Some drifts never occurred (HYPERPLANE)
  - Some drifts occurred only in final test chunk (SEA, AGRAWAL, RBF)
  - Some drifts were partially outside range (STAGGER)

### Comparative Models Results: ❌ INVALID
- River models (HAT, ARF, SRP), ACDWM, ERulesD2S all compared against GBML using **incorrect drift scenarios**
- Statistical tests (Friedman, Wilcoxon) were performed on **invalid data**

### What IS Valid: ✅
- The methodology framework (train-then-test, non-incremental learning)
- The comparative model execution scripts (run_comparative_on_existing_chunks.py)
- The consolidation and plotting scripts (CELULA 11, CELULA 12)
- The post-processing scripts (generate_plots.py, rule_diff_analyzer.py, analyze_concept_difference.py)

---

## PROPOSED SOLUTION

### Step 1: Update config_batch_1.yaml

Create corrected configuration with proper `duration_chunks` for 6-chunk execution:

```yaml
experimental_streams:
  # ... (keep all other configs)

  SEA_Abrupt_Simple:
    dataset_type: SEA
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: f1
      duration_chunks: 3    # CHANGED: was 5
    - concept_id: f3
      duration_chunks: 3    # CHANGED: was 5

  AGRAWAL_Abrupt_Simple_Severe:
    dataset_type: AGRAWAL
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: f1
      duration_chunks: 3    # CHANGED: was 5
    - concept_id: f6
      duration_chunks: 3    # CHANGED: was 5

  RBF_Abrupt_Severe:
    dataset_type: RBF
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: c1
      duration_chunks: 3    # CHANGED: was 5
    - concept_id: c2_severe
      duration_chunks: 3    # CHANGED: was 5

  STAGGER_Abrupt_Chain:
    dataset_type: STAGGER
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: f1
      duration_chunks: 2    # CHANGED: was 4
    - concept_id: f2
      duration_chunks: 2    # CHANGED: was 4
    - concept_id: f3
      duration_chunks: 2    # CHANGED: was 4

  HYPERPLANE_Abrupt_Simple:
    dataset_type: HYPERPLANE
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: plane1
      duration_chunks: 3    # CHANGED: was 6
    - concept_id: plane2
      duration_chunks: 3    # CHANGED: was 6
```

### Step 2: Validation Before Re-execution

Before re-running experiments, verify:
1. ✓ Drift positions will be: 2000 (STAGGER), 3000 (all others), 4000 (STAGGER)
2. ✓ All drifts occur WITHIN chunks 0-4 (training range)
3. ✓ Models will train on BOTH concepts for simple drifts, ALL THREE for chain drifts

### Step 3: Re-execution Plan

1. **Re-run GBML experiments** for Batch 1 with corrected config
2. **Re-run comparative models** (River, ACDWM, ERulesD2S) on new chunks
3. **Re-run post-processing**:
   - generate_plots.py
   - rule_diff_analyzer.py
   - analyze_concept_difference.py
4. **Re-run consolidation** (CELULA 11 and 12)

### Step 4: Validation of Results

After re-execution, verify:
- Plot_AccuracyPeriodic shows drift markers at 2000 or 3000 (not 4000+)
- Drift severity values are non-zero (e.g., 15-65% depending on concepts)
- Models show performance drops at drift points
- Concept difference calculations show meaningful values

---

## ESTIMATED IMPACT

### Time Cost:
- Re-running 5 datasets × 1 run each in Colab: ~2-3 hours
- Re-running comparative models: ~1-2 hours
- Post-processing and validation: ~30 minutes
- **Total**: ~4-6 hours of computation time

### Code Changes Required:
- ✅ **MINIMAL**: Only config_batch_1.yaml needs modification
- ✅ All scripts (GBML, comparative, consolidation) remain UNCHANGED
- ✅ No risk of breaking validated code

### Scientific Benefits:
- ✅ Valid experimental results suitable for publication
- ✅ Meaningful drift response analysis
- ✅ Accurate comparative model evaluation
- ✅ Correct drift severity calculations

---

## RECOMMENDED NEXT STEPS

1. **IMMEDIATE**: Review this analysis document and approve proposed corrections
2. **SHORT-TERM**: Update config_batch_1.yaml with corrected duration_chunks
3. **MEDIUM-TERM**: Re-execute Batch 1 experiments with corrected configuration
4. **VALIDATION**: Verify drift positions and severity values in new results
5. **LONG-TERM**: Apply same analysis to other batches if they exist

---

## CONCLUSION

This analysis identified a **CRITICAL but FIXABLE** configuration error affecting all 5 Batch 1 datasets. The root cause is a mismatch between configuration design (chunk_size=3000) and actual execution (chunk_size=1000), causing drifts to occur outside the experimental window.

**The good news**:
- The problem is purely configurational
- The fix requires only 10 lines of YAML changes
- All validated code remains intact
- Re-execution is straightforward

**The solution is clear and low-risk**: Update duration_chunks values to fit the 6-chunk execution window, then re-run experiments.

---

**Author**: Claude Code
**Reference**: ChiuTNNLS2020.pdf (Equation 6 for concept difference)
**Status**: READY FOR IMPLEMENTATION
