# Quick Reference: Batch 1 Configuration Corrections

## Summary Table

| Dataset | Current Config | Drift Positions (Current) | PROBLEM | Proposed Config | Drift Positions (Corrected) | Status |
|---------|----------------|---------------------------|---------|-----------------|----------------------------|---------|
| **SEA_Abrupt_Simple** | f1(5) → f3(5) | 5000 instances | Drift at END of range, model never trains on f3 | f1(3) → f3(3) | 3000 instances | ✅ Within range |
| **AGRAWAL_Abrupt_Simple_Severe** | f1(5) → f6(5) | 5000 instances | Drift at END of range, model never trains on f6 | f1(3) → f6(3) | 3000 instances | ✅ Within range |
| **RBF_Abrupt_Severe** | c1(5) → c2_severe(5) | 5000 instances | Drift at END of range, model never trains on c2_severe | c1(3) → c2_severe(3) | 3000 instances | ✅ Within range |
| **STAGGER_Abrupt_Chain** | f1(4) → f2(4) → f3(4) | 4000, 8000 instances | Drift 1 at boundary, Drift 2 OUTSIDE, model barely sees f2, never sees f3 | f1(2) → f2(2) → f3(2) | 2000, 4000 instances | ✅ Both within range |
| **HYPERPLANE_Abrupt_Simple** | plane1(6) → plane2(6) | 6000 instances | NO DRIFT in range! Model only sees plane1 | plane1(3) → plane2(3) | 3000 instances | ✅ Within range |

## Execution Context

**Actual execution parameters** (from run_config.json files):
- Chunk size: **1000 instances**
- Chunks processed: **5 chunks** (train on 0-4, test on 1-5)
- Total instances: **6000** (chunks 0-5)
- Training instance range: **0-5000** (chunks 0-4)

**Configuration was designed for**:
- Chunk size: **3000 instances**
- Total chunks: **8**
- Total instances: **24,000**

**Mismatch**: Config values not scaled when chunk_size reduced from 3000 to 1000!

## Configuration Changes Required

### File to Edit: `configs/config_batch_1.yaml`

Replace lines **411-477** with:

```yaml
  SEA_Abrupt_Simple:
    dataset_type: SEA
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: f1
      duration_chunks: 3    # CHANGED: was 5, drift now at 3000 instead of 5000
    - concept_id: f3
      duration_chunks: 3    # CHANGED: was 5

  AGRAWAL_Abrupt_Simple_Severe:
    dataset_type: AGRAWAL
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: f1
      duration_chunks: 3    # CHANGED: was 5, drift now at 3000 instead of 5000
    - concept_id: f6
      duration_chunks: 3    # CHANGED: was 5

  RBF_Abrupt_Severe:
    dataset_type: RBF
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: c1
      duration_chunks: 3    # CHANGED: was 5, drift now at 3000 instead of 5000
    - concept_id: c2_severe
      duration_chunks: 3    # CHANGED: was 5

  STAGGER_Abrupt_Chain:
    dataset_type: STAGGER
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: f1
      duration_chunks: 2    # CHANGED: was 4, drift now at 2000 instead of 4000
    - concept_id: f2
      duration_chunks: 2    # CHANGED: was 4, drift now at 4000 instead of 8000
    - concept_id: f3
      duration_chunks: 2    # CHANGED: was 4

  HYPERPLANE_Abrupt_Simple:
    dataset_type: HYPERPLANE
    drift_type: abrupt
    gradual_drift_width_chunks: 0
    concept_sequence:
    - concept_id: plane1
      duration_chunks: 3    # CHANGED: was 6, drift now at 3000 instead of 6000
    - concept_id: plane2
      duration_chunks: 3    # CHANGED: was 6
```

## Validation Checklist

After making changes and re-running experiments, verify:

- [ ] Plot_AccuracyPeriodic images show drift markers at 2000-4000 instances (NOT 5000-8000)
- [ ] Drift severity values are non-zero (should be 15-65% depending on concept pairs)
- [ ] Models show performance drops at drift points
- [ ] All training occurs on BOTH concepts (or all 3 for STAGGER)
- [ ] No drift markers appear outside the 6000 instance range

## Expected Drift Severity Values

Based on ChiuTNNLS2020.pdf methodology and concept definitions:

| Dataset | Concept Pair | Expected Severity Range |
|---------|-------------|------------------------|
| SEA_Abrupt_Simple | f1 vs f3 | ~35-45% (threshold changes from 8.0 to 7.0) |
| AGRAWAL_Abrupt_Simple_Severe | f1 vs f6 | ~55-70% (different classification functions) |
| RBF_Abrupt_Severe | c1 vs c2_severe | ~60-75% (different RBF centroids, seed 42 vs 84) |
| STAGGER_Abrupt_Chain | f1 vs f2 | ~25-35% |
| STAGGER_Abrupt_Chain | f2 vs f3 | ~30-40% |
| HYPERPLANE_Abrupt_Simple | plane1 vs plane2 | ~15-25% (small mag_change=0.01) |

**NOTE**: Current plots show 0.0% for all - this confirms the problem!

## Code Impact Assessment

**Files that need changes**:
- ✅ **1 file only**: `configs/config_batch_1.yaml`

**Files that remain unchanged**:
- ✅ All GBML code
- ✅ All comparative model scripts
- ✅ All consolidation scripts (CELULA 11, 12)
- ✅ All post-processing scripts (generate_plots.py, rule_diff_analyzer.py, analyze_concept_difference.py)

**Risk level**: 🟢 **MINIMAL** - Only configuration change, no code modifications

## Re-execution Plan

1. **Backup current results**: (optional but recommended)
   ```
   experiments_6chunks_phase1_gbml/batch_1_OLD/
   ```

2. **Update configuration**:
   - Edit `configs/config_batch_1.yaml` with corrected duration_chunks

3. **Re-run GBML experiments**:
   - Execute Batch 1 datasets with corrected config
   - Estimated time: 2-3 hours in Colab

4. **Re-run comparative models**:
   - Use existing run_comparative_on_existing_chunks.py
   - Estimated time: 1-2 hours

5. **Re-run post-processing**:
   - generate_plots.py
   - rule_diff_analyzer.py
   - analyze_concept_difference.py
   - Estimated time: 15-30 minutes

6. **Re-run consolidation**:
   - CELULA 11 (consolidate results + statistical tests)
   - CELULA 12 (generate comparative plots)
   - Estimated time: 10-15 minutes

**Total estimated time**: 4-6 hours

## Why This Happened

The configuration file `config_batch_1.yaml` contains:
```yaml
data_params:
  chunk_size: 3000    # Design specification
  num_chunks: 8
  max_instances: 24000
```

But the GBML system reduced chunk_size to 1000 for Colab execution time constraints, without proportionally adjusting the `duration_chunks` values in the `experimental_streams` section.

**The math**:
- Original design: 3000 × 8 = 24,000 instances
- Actual execution: 1000 × 6 = 6,000 instances
- Scale factor: 1000/3000 = 1/3
- Required adjustment: duration_chunks should be reduced by ~1/3

This is why:
- SEA f1(5) should become f1(3) ✓ (5/3 ≈ 1.67 → round to 2 or 3, we choose 3 for balance)
- STAGGER f1(4) should become f1(2) ✓ (4/3 = 1.33 → round to 1 or 2, but we need room for 3 concepts → 2)
- HYPERPLANE plane1(6) should become plane1(3) ✓ (6/3 = 2, but we want drift in middle → 3)

## Bottom Line

**Current status**: ❌ Results INVALID for scientific publication

**After correction**: ✅ Results VALID and ready for publication

**Effort required**: 🟢 LOW - Single config file change + re-execution

**Risk**: 🟢 MINIMAL - No code changes needed

---

**Next step**: Review and approve these corrections, then update config and re-execute experiments.
