# ACDWM Code Structure Analysis

**Date**: 2025-01-07
**Purpose**: Document ACDWM implementation for integration into comparison framework

---

## 1. CORE CLASSES

### 1.1 ChunkBase (chunk_based_methods.py)

**Purpose**: Abstract base class for chunk-based ensemble methods

**Key Attributes**:
```python
self.ensemble = list()          # List of trained models
self.chunk_count = 0            # Number of chunks processed
self.w = array([])              # Weights for ensemble members
self.buf_data = array([])       # Buffer for accumulating data
self.buf_label = array([])      # Buffer for accumulating labels
```

**Key Methods**:
```python
def update_chunk(self, data, label):
    """
    Main interface for prequential evaluation (test-then-train)
    1. Predicts on data with current ensemble
    2. Calls _update_chunk() to train on data
    3. Returns predictions
    """
    pred = self.predict(data)
    self._update_chunk(data, label)
    return pred

def predict(self, test_data):
    """Weighted ensemble prediction"""
    all_pred = sign(self._predict_base(test_data))
    if len(self.w) != 0:
        pred = sign(dot(all_pred, self.w))
    else:
        pred = all_pred
    return pred

def _predict_base(self, test_data):
    """Get predictions from all ensemble members"""
    pred = zeros([test_data.shape[0], len(self.ensemble)])
    for i in range(len(self.ensemble)):
        pred[:, i] = self.ensemble[i].predict(test_data)
    return pred

@abstractmethod
def _update_chunk(self, data, label):
    """Subclasses implement this to train on chunk"""
    pass
```

---

### 1.2 DWMIL Class (dwmil.py)

**Purpose**: Dynamic Weighted Majority for Imbalanced Learning

**Constructor**:
```python
def __init__(self, data_num, chunk_size=1000, theta=0.1, err_func='gm', r=1):
    """
    Args:
        data_num: Total number of samples in stream
        chunk_size: Fixed chunk size (0 for adaptive)
        theta: Weight threshold for removing classifiers
        err_func: Error function ('gm' for G-mean)
        r: Imbalance ratio for UnderBagging
    """
```

**Key Method**:
```python
def _update_chunk(self, data, label):
    """
    1. Creates UnderBagging model and trains on chunk
    2. Adds model to ensemble with weight=1
    3. Gets predictions from all ensemble members
    4. Calculates weighted ensemble prediction (excluding new model)
    5. Calculates error for each ensemble member
    6. Updates weights: w = (1 - err) * w
    7. Removes models with weight < theta
    """
    # Create and train new model
    model = UnderBagging(r=self.r, auto_T=True)
    model.train(data, label)
    self.ensemble.append(model)
    self.w = append(self.w, 1)

    # Get predictions from all models
    all_pred = sign(self._predict_base(data))

    # Weighted prediction (excluding new model)
    if self.chunk_count > 1:
        pred = dot(all_pred[:, :-1], self.w[:-1])
    else:
        pred = zeros_like(label)

    # Update weights
    err = self.calculate_err(all_pred, label)
    self.w = (1 - err) * self.w

    # Remove weak models
    remove_idx = nonzero(self.w < self.theta)[0]
    if len(remove_idx) != 0:
        for index in sorted(remove_idx, reverse=True):
            del self.ensemble[index]
        self.w = delete(self.w, remove_idx)
        self.chunk_count -= remove_idx.size

    return pred
```

---

### 1.3 UnderBagging Class (underbagging.py)

**Purpose**: Undersampling-based bagging for imbalanced data

**Constructor**:
```python
def __init__(self, T=11, r=1.0, auto_T=False, auto_r=False, ...):
    """
    Args:
        T: Number of base learners in ensemble
        r: Imbalance ratio (minority/majority)
        auto_T: Automatically adjust T based on imbalance
        auto_r: Automatically adjust r
    """
```

**Methods**:
```python
def train(self, data, label):
    """
    Creates T decision trees by:
    1. Undersampling majority class
    2. Keeping all minority class samples
    3. Training DecisionTreeClassifier on balanced subset
    """

def predict(self, test_data):
    """Returns mean of predictions from all trees"""
    temp_result = zeros([len(self.model), test_num])
    for i in range(len(self.model)):
        temp_result[i, :] = self.model[i].predict(test_data)
    pred_result = mean(temp_result, 0)
    return pred_result
```

---

### 1.4 SubUnderBagging Class (subunderbagging.py)

**Purpose**: Creates normally distributed predictions for variance testing

**Constructor**:
```python
def __init__(self, Q=1000, T=100, k_mode=2):
    """
    Args:
        Q: Number of decision stumps to create
        T: Number of stumps to sample per prediction
        k_mode: Sampling mode (1 or 2)
    """
```

**Methods**:
```python
def train(self, data, label):
    """Creates Q decision stumps (max_depth=1)"""

def predict(self, test_data, P):
    """
    Returns P sets of predictions by randomly sampling T stumps
    Shape: (P, T, n_samples)
    Used for variance calculation in chunk size selection
    """
```

---

### 1.5 ChunkSizeSelect Class (chunk_size_select.py)

**Purpose**: Adaptive chunk size selection using statistical tests

**Constructor**:
```python
def __init__(self, chunk_min=100, min_min=5, P=250, T=100, Q=1000,
             nt=10, delta=0.05, init_num=100, k_mode=2, mute=1):
    """
    Args:
        chunk_min: Minimum chunk size
        min_min: Minimum samples from minority class
        P: Number of prediction sets for variance
        T: Number of stumps per prediction
        Q: Total number of stumps
        nt: Number of test samples
        delta: Significance level (alpha)
        init_num: Initial chunk size
    """
```

**Key Method**:
```python
def check_significance(self):
    """
    Fisher's method combining F-tests:
    1. For each test sample, compute F-test: var_0[i] / var_1[i]
    2. Get p-values from F-distribution
    3. Combine using Fisher's method: K = -2 * sum(log(p_values))
    4. K follows chi-squared distribution with 2*nt degrees of freedom
    5. Return chi-squared p-value

    If p < alpha: variances differ significantly, add more samples
    If p >= alpha: no significant difference, chunk is ready
    """
```

---

## 2. LABEL CONVENTION

**CRITICAL**: ACDWM uses **-1 and +1** labels, NOT 0 and 1

```python
# In check_measure.py:
def gm_measure(pred, label):
    tp = sum(bitwise_and(label == 1, pred == 1))      # Positive = 1
    fn = sum(bitwise_and(label == 1, pred == -1))
    tn = sum(bitwise_and(label == -1, pred == -1))    # Negative = -1
    fp = sum(bitwise_and(label == -1, pred == 1))
    gm = sqrt(tp / (tp + fn) * tn / (tn + fp))
    return gm
```

**Conversion Required**:
- Input: 0/1 labels (from our framework)
- Convert: 0 -> -1, 1 -> 1
- Output: Convert predictions back if needed

---

## 3. USAGE PATTERN (from main.py)

```python
# Load data
load_data = load('data/sea_abrupt.npz')
data = load_data['data']           # Shape: (N, features)
label = load_data['label']         # Shape: (N,) with -1/+1

# Create chunk size selector
acss = ChunkSizeSelect()

# Create DWMIL model
model_acdwm = DWMIL(data_num=data_num, chunk_size=0)

# Prequential evaluation (test-then-train)
pred_acdwm = array([])

for i in range(data_num):
    # Accumulate samples in chunk selector
    acss.update(data[i], label[i])

    # When chunk is ready
    if acss.get_enough() == 1:
        chunk_data, chunk_label = acss.get_chunk()

        # Test-then-train: predict first, then train
        chunk_pred = model_acdwm.update_chunk(chunk_data, chunk_label)

        pred_acdwm = append(pred_acdwm, chunk_pred)

# Calculate prequential metrics
pq_result = prequential_measure(pred_acdwm, label, reset_pos)
```

---

## 4. INTEGRATION STRATEGY

### For baseline_acdwm.py:

#### 4.1 Label Conversion
```python
def convert_labels_to_acdwm(y):
    """Convert 0/1 to -1/+1"""
    y_acdwm = np.where(y == 0, -1, 1)
    return y_acdwm

def convert_labels_from_acdwm(y_pred):
    """Convert -1/+1 to 0/1"""
    y_standard = np.where(y_pred == -1, 0, 1)
    return y_standard
```

#### 4.2 Test-Then-Train Mode (Prequential)
```python
def evaluate_test_then_train(self, chunks):
    """Original ACDWM methodology"""
    for chunk_data in chunks:
        X_train_river, y_train_river, X_test_river, y_test_river = chunk_data

        # Convert to NumPy
        X_chunk, _ = river_to_numpy(X_train_river, self.feature_names)
        y_chunk = convert_labels_to_acdwm(
            river_labels_to_numpy(y_train_river)
        )

        # Test-then-train using update_chunk()
        y_pred = self.model.update_chunk(X_chunk, y_chunk)

        # Convert back and calculate metrics
        y_pred_standard = convert_labels_from_acdwm(y_pred)
        metrics = calculate_shared_metrics(y_train_river, y_pred_standard, self.classes)
```

#### 4.3 Train-Then-Test Mode (For GBML Compatibility)
```python
def evaluate_train_then_test(self, chunks):
    """Compatible with GBML/River methodology"""
    for i in range(len(chunks) - 1):
        X_train_river, y_train_river, _, _ = chunks[i]
        _, _, X_test_river, y_test_river = chunks[i + 1]

        # Train on chunk i
        X_train, _ = river_to_numpy(X_train_river, self.feature_names)
        y_train = convert_labels_to_acdwm(
            river_labels_to_numpy(y_train_river)
        )
        self.model._update_chunk(X_train, y_train)  # Only train

        # Test on chunk i+1
        X_test, _ = river_to_numpy(X_test_river, self.feature_names)
        y_pred = self.model.predict(X_test)

        # Convert and calculate metrics
        y_pred_standard = convert_labels_from_acdwm(y_pred)
        metrics = calculate_shared_metrics(y_test_river, y_pred_standard, self.classes)
```

#### 4.4 Fixed Chunk Size (Simplified)
For comparison framework, we'll use **fixed chunks** (not adaptive):
- Simpler integration
- Fair comparison (same chunk boundaries for all models)
- Adaptive chunk size can be added later

---

## 5. DEPENDENCIES

### Required Imports:
```python
import sys
sys.path.append('ACDWM')  # Add ACDWM directory to path

from ACDWM.dwmil import DWMIL
from ACDWM.underbagging import UnderBagging
from ACDWM.chunk_size_select import ChunkSizeSelect
```

### ACDWM Dependencies:
- numpy
- sklearn (DecisionTreeClassifier, metrics)
- scipy (for F-test and chi-squared in adaptive chunk size)

---

## 6. KEY PARAMETERS

### DWMIL:
- `theta=0.001` (default in paper, line 119 of baseline_acdwm.py)
- `err_func='gm'` (G-mean)
- `r=1` (imbalance ratio for UnderBagging)

### UnderBagging:
- `auto_T=True` (automatically adjust number of trees)
- Default T=11 base learners

### ChunkSizeSelect (if used):
- `chunk_min=100`
- `min_min=5`
- `P=250, T=100, Q=1000`
- `delta=0.05`

---

## 7. NEXT STEPS

1. [X] Analyze ACDWM code structure
2. [ ] Update baseline_acdwm.py with real integration
3. [ ] Test with small synthetic dataset
4. [ ] Extend compare_gbml_vs_river.py
5. [ ] Run full comparison experiments
