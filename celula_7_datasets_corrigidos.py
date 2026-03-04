# ============================================================================
# LISTA CORRIGIDA DE DATASETS PARA CÉLULA 7
# ============================================================================
# Use esta lista para executar modelos comparativos em TODOS os 12 datasets
# do Batch 1 (ao invés de apenas 5)
# ============================================================================

DATASETS = [
    # GRUPO 1: SEA (3 datasets)
    "SEA_Abrupt_Simple",
    "SEA_Abrupt_Chain",
    "SEA_Abrupt_Recurring",

    # GRUPO 2: AGRAWAL (3 datasets)
    "AGRAWAL_Abrupt_Simple_Mild",
    "AGRAWAL_Abrupt_Simple_Severe",
    "AGRAWAL_Abrupt_Chain_Long",

    # GRUPO 3: RBF (2 datasets)
    "RBF_Abrupt_Severe",
    "RBF_Abrupt_Blip",

    # GRUPO 4: STAGGER (2 datasets)
    "STAGGER_Abrupt_Chain",
    "STAGGER_Abrupt_Recurring",

    # GRUPO 5: OUTROS (2 datasets)
    "HYPERPLANE_Abrupt_Simple",
    "RANDOMTREE_Abrupt_Simple"
]

# Total: 12 datasets
# Tempo estimado: ~25-35 minutos (todos modelos: HAT, ARF, SRP, ACDWM, ERulesD2S)
