# constants.py (Modificado para tipos de features)

"""
Global constants used in the Genetic Algorithm for Rule Learning.
Includes differentiation between numeric and categorical operators.
"""

# --- Operators ---

# Available operators for combining conditions (internal nodes)
LOGICAL_OPERATORS = ["AND", "OR"]

# Operators for NUMERICAL features (leaf nodes)
NUMERIC_COMPARISON_OPERATORS = ["<", ">", "<=", ">="]

# Operators for CATEGORICAL features (leaf nodes)
# Note: Using '==' and '!=' for categories.
CATEGORICAL_COMPARISON_OPERATORS = ["==", "!="]

# Combined list - useful where distinction isn't made (or for initial validation)
# Maintain the original '!=' and add '=='
ALL_COMPARISON_OPERATORS = ["==", "!=", "<", ">", "<=", ">="]

# --- Reproducibility ---

# Seed for reproducibility
RANDOM_SEED = 42

# --- Feature Identification (Example) ---
# Prefix used to identify categorical features (can be adjusted)
# This might be better placed in main.py or config.yaml if it varies often
CATEGORICAL_PREFIX = "x_cat_"