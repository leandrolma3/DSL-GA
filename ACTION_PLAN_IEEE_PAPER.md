# ACTION PLAN - IEEE PAPER ON EXPLAINABLE DATA STREAM CLASSIFICATION

**Date**: 2025-12-16
**Last Updated**: 2025-12-18
**Objective**: Consolidate all experiments and prepare comprehensive analysis for IEEE journal submission
**Status**: PHASE 5 COMPLETE - Paper draft generated

---

## 1. EXPERIMENTAL FRAMEWORK OVERVIEW

### 1.1 Three Experimental Configurations

The study comprises three distinct experimental configurations, each addressing specific research questions:

| Configuration | Identifier | chunk_size | PENALTY_WEIGHT | Purpose |
|---------------|------------|------------|----------------|---------|
| **EXP-A** | chunk1000 | 1000 | 0.0 | Baseline performance-focused |
| **EXP-B** | chunk2000 | 2000 | 0.0 | Impact of larger evaluation windows |
| **EXP-C** | balanced | 2000 | 0.1 | Trade-off: performance vs interpretability |

### 1.2 Directory Structure

```
EXP-A (chunk_size=1000):
  experiments_6chunks_phase2_gbml/batch_1-4   (drift simulation)
  experiments_6chunks_phase3_real/batch_5-7   (real + stationary)

EXP-B (chunk_size=2000):
  experiments_chunk2000_phase1/batch_1-4      (drift simulation)
  experiments_chunk2000_phase2/batch_5-7      (real + stationary)

EXP-C (balanced):
  experiments_balanced_phase1/batch_1-4       (drift simulation)
  experiments_balanced_phase2/batch_5-7       (real + stationary)
```

### 1.3 Dataset Distribution (52 datasets total)

| Batch | Category | Count | Examples |
|-------|----------|-------|----------|
| 1 | Abrupt Drift | 12 | SEA_Abrupt_*, AGRAWAL_Abrupt_*, RBF_Abrupt_*, STAGGER_Abrupt_* |
| 2 | Gradual Drift | 9 | SEA_Gradual_*, RBF_Gradual_*, HYPERPLANE_Gradual_* |
| 3 | Drift with Noise | 8 | SEA_*_Noise, AGRAWAL_*_Noise, RBF_*_Noise |
| 4 | Mixed Generators | 6 | SINE_*, LED_*, WAVEFORM_*, RANDOMTREE_* |
| 5 | Real-world | 5 | Electricity, Shuttle, CovType, PokerHand, IntelLabSensors |
| 6 | Synthetic Stationary | 6 | SEA_Stationary, AGRAWAL_Stationary, RBF_Stationary |
| 7 | Synthetic Stationary | 6 | STAGGER_Stationary, WAVEFORM_Stationary, AssetNegotiation_* |

---

## 2. COMPARATIVE MODELS

### 2.1 Models Under Evaluation (8 models)

| ID | Model | Type | Interpretable | Constraints |
|----|-------|------|---------------|-------------|
| 1 | **Proposed Method** | Evolutionary Rule-based | Yes | - |
| 2 | ROSE_Original | Ensemble (MOA) | No | Prequential evaluation |
| 3 | ROSE_ChunkEval | Ensemble (MOA) | No | Chunk-based evaluation |
| 4 | HAT | Adaptive Tree | Partial | - |
| 5 | ARF | Ensemble Forest | No | - |
| 6 | SRP | Ensemble Patches | No | - |
| 7 | ACDWM | Ensemble DWM | No | Binary classification only |
| 8 | ERulesD2S | Evolutionary Rules | Yes* | No transition metrics |

*Note: ERulesD2S provides aggregate complexity metrics only (NumberRules, NumberConditions, NumberNodes).

### 2.2 Key Distinction: ROSE Variants

- **ROSE_Original**: Instance-by-instance prequential evaluation (as per original paper)
- **ROSE_ChunkEval**: Chunk-based evaluation (comparable methodology with proposed method)

---

## 3. PROPOSED MODEL NAMING

### 3.1 Naming Criteria

The model name should:
- Emphasize explainability/interpretability as the key contribution
- Reference data stream processing
- Be memorable and appropriate for academic publication
- Reflect the rule-based nature of the approach

### 3.2 Candidate Names

| Name | Full Form | Rationale |
|------|-----------|-----------|
| XStream-Rules | eXplainable Stream Rules | Clear emphasis on explainability |
| **EGIS** | Evolutionary Grammar for Interpretable Streams | Concise, academic |
| SIRL | Stream Interpretable Rule Learner | Descriptive |
| GE-XRL | Grammar Evolution eXplainable Rule Learning | Technical detail |

**SELECTED: EGIS (Evolutionary Grammar for Interpretable Streams)** - Chosen 2025-12-18

---

## 4. CURRENT EXPERIMENT STATUS

### 4.1 EXP-A (chunk_size=1000) - COMPLETE

| Batch | Datasets | GBML | Comparative Models |
|-------|----------|------|-------------------|
| 1 | 12/12 | Complete | Complete |
| 2 | 9/9 | Complete | Complete |
| 3 | 8/8 | Complete | Complete |
| 4 | 6/6 | Complete | Complete |
| 5 | 5/5 | Complete | Complete |
| 6 | 6/6 | Complete | Complete |
| 7 | 6/6 | Complete | Complete |
| **Total** | **52/52** | **Complete** | **Complete** |

### 4.2 EXP-B (chunk_size=2000) - COMPLETE

| Batch | Datasets | GBML | Comparative Models |
|-------|----------|------|-------------------|
| 1-4 | 35/35 | Complete | **Complete** |
| 5-7 | 17/17 | Complete | **Complete** |
| **Total** | **52/52** | **Complete** | **Complete** |

**Updated 2025-12-18**: All 8 comparative models executed (ROSE, ARF, HAT, SRP, ACDWM, ERulesD2S)

### 4.3 EXP-C (balanced) - PARTIAL (29/52 datasets)

| Batch | Datasets | GBML | Comparative Models |
|-------|----------|------|-------------------|
| 1 | 11/12 | Complete | Complete |
| 2 | 9/9 | Complete | Complete |
| 3 | 0/8 | **Pending** | Pending |
| 4 | 5/6 | Complete | Complete |
| 5 | 4/5 | Complete | Complete |
| 6-7 | 0/12 | **Pending** | Pending |

**Updated 2025-12-18**: 32 datasets with results, 23 datasets pending execution

---

## 5. UNIFIED ANALYSIS SCRIPT SPECIFICATION

### 5.1 Script Requirements

**Input**: All experiment directories (complete and partial)
**Output**: Structured data for tables, plots, and statistical tests

### 5.2 Output Directory Structure

```
analysis_output/
  exp_a_chunk1000/
    data/
      gbml_results.csv
      gbml_rules_per_chunk.csv
      gbml_transition_metrics.csv
      comparative_results.csv
      consolidated_all_models.csv
    tables/
      performance_comparison.tex
      statistical_tests.tex
      complexity_comparison.tex
    figures/
      performance_boxplot.pdf
      critical_difference.pdf
      rules_comparison.pdf
  exp_b_chunk2000/
    (same structure)
  exp_c_balanced/
    (same structure)
  cross_experiment/
    chunk_size_impact.csv
    balance_impact.csv
    summary_statistics.csv
```

### 5.3 Data Extraction Requirements

**From GBML experiments:**
- chunk_metrics.json: Performance metrics per chunk
- RulesHistory_*.txt: Rules for transition metrics calculation
- rule_details_per_chunk.json: Detailed rule structure

**From Comparative models:**
- rose_original_results.csv / rose_chunk_eval_results.csv
- river_HAT_results.csv / river_ARF_results.csv / river_SRP_results.csv
- acdwm_results.csv
- erulesd2s_results.csv

### 5.4 Metrics to Extract

**Performance Metrics:**
- G-Mean (primary metric)
- F1-Score
- Accuracy

**Interpretability Metrics (GBML and ERulesD2S):**
- Number of rules per chunk
- Number of conditions per rule
- Average rule complexity

**Transition Metrics (GBML only):**
- TCS (Transition Change Score)
- RIR (Rule Instability Rate)
- AMS (Average Modification Severity)

---

## 6. STATISTICAL ANALYSIS PLAN

### 6.1 Within-Experiment Comparisons

For each experiment (EXP-A, EXP-B, EXP-C):

1. **Friedman Test**: Non-parametric test for comparing multiple classifiers
2. **Nemenyi Post-hoc Test**: Pairwise comparisons with Bonferroni correction
3. **Wilcoxon Signed-Rank Test**: Paired comparison of proposed method vs each baseline
4. **Critical Difference Diagram**: Visual representation of statistical differences

### 6.2 Cross-Experiment Comparisons

1. **Chunk Size Impact**: Compare EXP-A vs EXP-B (same PENALTY_WEIGHT, different chunk_size)
2. **Balance Impact**: Compare EXP-B vs EXP-C (same chunk_size, different PENALTY_WEIGHT)
3. **Interpretability Trade-off**: Analyze performance vs complexity across configurations

---

## 7. TABLES FOR IEEE PAPER

### 7.1 Required Tables

| Table | Description | Content |
|-------|-------------|---------|
| 1 | Dataset Characteristics | Name, instances, features, classes, drift type |
| 2 | Performance Comparison (EXP-A) | G-Mean for all models, chunk_size=1000 |
| 3 | Performance Comparison (EXP-B) | G-Mean for all models, chunk_size=2000 |
| 4 | Performance Comparison (EXP-C) | G-Mean for all models, balanced |
| 5 | Statistical Significance | Friedman, Nemenyi, Wilcoxon results |
| 6 | Interpretability Metrics | Rules, conditions, GBML vs ERulesD2S |
| 7 | Transition Metrics | TCS, RIR, AMS by drift type |
| 8 | Chunk Size Impact | EXP-A vs EXP-B comparison |
| 9 | Balance Trade-off | Performance vs complexity analysis |

### 7.2 Table Format (LaTeX)

All tables should be generated in LaTeX format with:
- Proper column alignment
- Bold for best results
- Significance markers (* p < 0.05, ** p < 0.01)
- Caption and label for referencing

---

## 8. FIGURES FOR IEEE PAPER

### 8.1 Required Figures

| Figure | Description | Type |
|--------|-------------|------|
| 1 | System Architecture | Diagram |
| 2 | Performance Comparison Boxplot | Box plot |
| 3 | Critical Difference Diagram | CD diagram |
| 4 | Rules per Chunk Evolution | Line plot |
| 5 | Transition Metrics over Stream | Multi-line plot |
| 6 | Performance vs Complexity Trade-off | Scatter plot |
| 7 | Chunk Size Impact | Bar chart |

### 8.2 Figure Format

- PDF format for vector graphics
- Minimum 300 DPI for raster elements
- Consistent color scheme across all figures
- Accessible color palette (colorblind-friendly)

---

## 9. PHASED ACTION PLAN

### PHASE 1: Data Consolidation (Priority: Critical) - COMPLETE

**1.1 Verify Experiment Completeness**
- [x] Audit all directories for missing datasets
- [x] Document partial results status
- [x] Identify and resolve data inconsistencies

**1.2 Execute Comparative Models for EXP-B**
- [x] Modify Execute_All_Comparative_Models.ipynb
  - Change CHUNK_SIZE from 1000 to 2000
  - Update BATCH_CONFIG base_dir to experiments_chunk2000_*
- [x] Execute all 7 models on 52 datasets
- [x] Validate results completeness

**1.3 Monitor EXP-C (Balanced) Execution**
- [x] Track GBML experiment progress
- [ ] Plan comparative model execution after GBML completion (23 datasets pending)

### PHASE 2: Unified Analysis Script Development (Priority: High) - COMPLETE

**2.1 Script Architecture**
- [x] Design modular script structure
- [x] Implement directory traversal for all experiment types
- [x] Create data extraction functions for each model type

**2.2 Core Functions**
- [x] extract_gbml_performance(): Read chunk_metrics.json
- [x] extract_gbml_rules(): Parse RulesHistory_*.txt
- [x] calculate_transition_metrics(): Compute TCS, RIR, AMS
- [x] extract_comparative_results(): Read model CSVs
- [x] consolidate_results(): Merge all data sources

**2.3 Output Generation**
- [x] Generate CSV files for all metrics
- [x] Create LaTeX table templates
- [x] Produce figure-ready data files

**Script location**: `unified_analysis.py`
**Output directory**: `analysis_output/`

### PHASE 3: Statistical Analysis (Priority: High) - COMPLETE

**3.1 Within-Experiment Analysis**
- [x] Implement Friedman test
- [x] Implement Nemenyi post-hoc test
- [x] Implement Wilcoxon signed-rank test
- [ ] Generate Critical Difference diagrams (optional)

**3.2 Cross-Experiment Analysis**
- [x] Analyze chunk size impact (EXP-A vs EXP-B)
- [x] Analyze balance trade-off (EXP-B vs EXP-C)
- [x] Document findings with statistical support

### PHASE 4: Visualization (Priority: Medium) - PARTIAL

**4.1 Performance Visualizations**
- [x] Generate boxplots comparing all models
- [ ] Create CD diagrams for each experiment
- [x] Produce heatmaps by drift type

**4.2 Interpretability Visualizations**
- [x] Plot rules evolution over stream
- [ ] Create scatter plot: G-Mean vs Number of Rules
- [x] Generate transition metrics temporal plots

### PHASE 5: Paper Preparation (Priority: Medium) - COMPLETE

**5.1 LaTeX Structure**
- [x] Create paper directory structure (`paper/`)
- [x] Set up main.tex with IEEE template
- [x] Prepare sections skeleton

**5.2 Content Integration**
- [x] Insert generated tables (9 tables total)
- [ ] Insert generated figures (pending integration)
- [x] Write results interpretation

**Paper location**: `paper/main.tex`
**PDF generated**: `paper/main.pdf` (205 KB)
**Model name**: EGIS (Evolutionary Grammar for Interpretable Streams)

### PHASE 6: Validation and Review (Priority: Standard) - IN PROGRESS

**6.1 Results Validation**
- [x] Cross-check statistical results
- [x] Verify table accuracy
- [ ] Review figure clarity

**6.2 Manuscript Review**
- [ ] Technical accuracy review
- [ ] Language and formatting check
- [ ] Prepare supplementary materials

---

## 10. SCRIPT PSEUDOCODE

```python
# unified_analysis.py - Main analysis script

EXPERIMENT_CONFIGS = {
    'exp_a_chunk1000': {
        'dirs': {
            'batch_1': 'experiments_6chunks_phase2_gbml/batch_1',
            'batch_2': 'experiments_6chunks_phase2_gbml/batch_2',
            'batch_3': 'experiments_6chunks_phase2_gbml/batch_3',
            'batch_4': 'experiments_6chunks_phase2_gbml/batch_4',
            'batch_5': 'experiments_6chunks_phase3_real/batch_5',
            'batch_6': 'experiments_6chunks_phase3_real/batch_6',
            'batch_7': 'experiments_6chunks_phase3_real/batch_7',
        },
        'chunk_size': 1000,
        'penalty_weight': 0.0,
        'description': 'Baseline configuration'
    },
    'exp_b_chunk2000': {
        'dirs': {
            'batch_1': 'experiments_chunk2000_phase1/batch_1',
            'batch_2': 'experiments_chunk2000_phase1/batch_2',
            'batch_3': 'experiments_chunk2000_phase1/batch_3',
            'batch_4': 'experiments_chunk2000_phase1/batch_4',
            'batch_5': 'experiments_chunk2000_phase2/batch_5',
            'batch_6': 'experiments_chunk2000_phase2/batch_6',
            'batch_7': 'experiments_chunk2000_phase2/batch_7',
        },
        'chunk_size': 2000,
        'penalty_weight': 0.0,
        'description': 'Larger evaluation window'
    },
    'exp_c_balanced': {
        'dirs': {
            'batch_1': 'experiments_balanced_phase1/batch_1',
            'batch_2': 'experiments_balanced_phase1/batch_2',
            'batch_3': 'experiments_balanced_phase1/batch_3',
            'batch_4': 'experiments_balanced_phase1/batch_4',
            'batch_5': 'experiments_balanced_phase2/batch_5',
            'batch_6': 'experiments_balanced_phase2/batch_6',
            'batch_7': 'experiments_balanced_phase2/batch_7',
        },
        'chunk_size': 2000,
        'penalty_weight': 0.1,
        'description': 'Performance-complexity balance'
    }
}

OUTPUT_DIR = 'analysis_output'

def main():
    for exp_name, config in EXPERIMENT_CONFIGS.items():
        print(f"Processing {exp_name}...")

        # Extract GBML results
        gbml_perf = extract_gbml_performance(config)
        gbml_rules = extract_gbml_rules(config)
        transitions = calculate_transition_metrics(gbml_rules)

        # Extract comparative model results
        comparative = extract_comparative_results(config)

        # Consolidate
        consolidated = consolidate_results(gbml_perf, comparative)

        # Generate outputs
        save_csv_files(exp_name, consolidated, transitions)
        generate_latex_tables(exp_name, consolidated)
        generate_figures(exp_name, consolidated, transitions)

    # Cross-experiment analysis
    perform_cross_experiment_analysis()
```

---

## 11. IMMEDIATE NEXT STEPS

~~1. **Validate this action plan** with stakeholder review~~ - DONE
~~2. **Finalize model name** (XStream-Rules or EGIS recommended)~~ - DONE: EGIS selected
~~3. **Begin Phase 1.2**: Execute comparative models for EXP-B (chunk_size=2000)~~ - DONE
~~4. **Develop unified analysis script** based on Section 10 specification~~ - DONE

### Current Next Steps (2025-12-18):

1. **Complete EXP-C experiments** - 23 datasets still pending
2. **Add figures to paper** - Critical Difference diagrams, performance plots
3. **Review paper content** - Technical accuracy and language
4. **Prepare for submission** - Check IEEE formatting requirements

---

## 12. DEPENDENCIES AND REQUIREMENTS

### Python Packages
- pandas, numpy: Data manipulation
- scipy: Statistical tests
- scikit-learn: Metrics calculation
- matplotlib, seaborn: Visualization
- Orange3: Critical Difference diagrams (optional)

### External Tools
- Java 11: Required for ROSE and ERulesD2S
- Maven: Build tool for Java components
- LaTeX: Document preparation

---

**Author**: Claude Code
**Status**: PHASE 5 COMPLETE - Paper draft ready for review
**Last Updated**: 2025-12-18

---

## 13. GENERATED ARTIFACTS

| Artifact | Location | Description |
|----------|----------|-------------|
| Paper (LaTeX) | `paper/main.tex` | IEEE format, EGIS paper |
| Paper (PDF) | `paper/main.pdf` | Compiled PDF (205 KB) |
| Analysis Script | `unified_analysis.py` | Main analysis pipeline |
| Analysis Output | `analysis_output/` | Tables, data, figures |
| Session Summary | `ARTIGO_IEEE_EGIS_RESUMO.md` | Summary of paper actions |
