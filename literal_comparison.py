# literal_comparison.py

import math
# Se precisarmos de Levenshtein para nomes de atributos ou valores categóricos no futuro:
# import Levenshtein

# --- 1. Definição da Matriz/Lógica de Severidade de Mudança de Operador ---
# Esta é uma parte crucial e pode ser ajustada.
# Pontuações de 0 (sem mudança) a 1 (mudança máxima/oposta).
# Considera-se que op1 é o operador antigo e op2 é o novo.
OPERATOR_SIMILARITY_SCORES = {
    # Mudanças Leves (mantendo a "direção" ou proximidade)
    (">", ">="): 0.7,  # Um pouco menos restritivo
    (">=", ">"): 0.7,  # Um pouco mais restritivo
    ("<", "<="): 0.7,
    ("<=", "<"): 0.7,
    ("==", "!="): 0.2, # Inversão direta, mas ainda sobre igualdade
    ("!=", "=="): 0.2,
    # Mudanças Moderadas (mudando tipo de comparação, ex: igualdade para desigualdade direcional)
    ("==", "<"): 0.4,
    ("==", "<="): 0.4,
    ("==", ">"): 0.4,
    ("==", ">="): 0.4,
    ("!=", "<"): 0.6, # De não-igual para uma direção específica
    ("!=", "<="): 0.6,
    ("!=", ">"): 0.6,
    ("!=", ">="): 0.6,
    # Inversos para mudanças moderadas
    ("<", "=="): 0.4,
    ("<=", "=="): 0.4,
    (">", "=="): 0.4,
    (">=", "=="): 0.4,
    ("<", "!="): 0.6,
    ("<=", "!="): 0.6,
    (">", "!="): 0.6,
    (">=", "!="): 0.6,
    # Mudanças Drásticas (inversão de direção)
    ("<", ">"): 0.0,
    ("<", ">="): 0.0,
    ("<=", ">"): 0.0,
    ("<=", ">="): 0.0,
    # Outras combinações que podem precisar de ajuste fino
    # Por exemplo, de < para != pode ser menos severo que < para >
}

def get_operator_dissimilarity(op1, op2):
    """
    Calcula a dissimilaridade entre dois operadores.
    Retorna um valor entre 0 (idêntico/muito similar) e 1 (completamente diferente/oposto).
    Se a similaridade é S, dissimilaridade é 1-S.
    """
    if op1 == op2:
        return 0.0  # Nenhuma dissimilaridade

    # Usa o score de similaridade, ou o inverso se a ordem for diferente
    similarity = OPERATOR_SIMILARITY_SCORES.get((op1, op2), OPERATOR_SIMILARITY_SCORES.get((op2, op1)))

    if similarity is not None:
        return 1.0 - similarity # Dissimilaridade é 1 - similaridade

    # Se o par não está na matriz, assume dissimilaridade máxima (ou uma penalidade alta)
    # Isso pode acontecer se um operador for ex: '==' e outro for '>' e não listamos ('==', '>')
    # mas listamos ('>', '=='). A lógica acima com .get() duplo deve cobrir isso.
    # Se ainda assim não cobrir, é uma combinação não prevista.
    # print(f"Aviso: Combinação de operadores não prevista na matriz de similaridade: {op1} vs {op2}")
    return 1.0 # Dissimilaridade máxima para pares não definidos explicitamente


# --- 2. Definição da Estratégia de Normalização para Mudança de Valor Numérico ---
def calculate_value_normalized_difference(val1, val2, normalization_method="max_abs"):
    """
    Calcula a diferença normalizada entre dois valores numéricos.
    Retorna um valor idealmente entre 0 e 1.
    normalization_method: "max_abs", "average_abs", "range" (requer min/max do atributo)
    """
    if val1 == val2:
        return 0.0

    diff = abs(val1 - val2)

    if normalization_method == "max_abs":
        denominator = max(abs(val1), abs(val2))
        if denominator == 0: # Ambos são 0 (já tratado) ou um é 0 e outro não.
             # Se um é 0 e outro não, ex: 0 e 5. diff = 5. max(0,5)=5. diff/den = 1.
             # Se ambos 0, diff = 0.
            return 1.0 if diff > 0 else 0.0 # Máxima diferença se um for zero e o outro não
    elif normalization_method == "average_abs":
        denominator = (abs(val1) + abs(val2)) / 2.0
        if denominator == 0:
            return 1.0 if diff > 0 else 0.0
    # Adicionar "range" se tivermos essa informação no futuro
    # elif normalization_method == "range" and attr_range is not None:
    #     denominator = attr_range
    #     if denominator == 0: return 1.0 if diff > 0 else 0.0
    else: # Fallback para max_abs se o método for desconhecido ou incompleto
        denominator = max(abs(val1), abs(val2))
        if denominator == 0:
            return 1.0 if diff > 0 else 0.0
            
    normalized_diff = diff / denominator
    return min(normalized_diff, 1.0) # Garante que a dissimilaridade não passe de 1.0


# --- 3. Função Principal para Comparar Literais Atômicos ---
def calculate_atomic_condition_change_severity(literal1, literal2, attribute_weights=None):
    """
    Calcula a severidade da mudança entre dois literais atômicos.
    literal1, literal2: no formato ['atributo', 'operador', valor_numérico]
    attribute_weights: um dicionário opcional com pesos de importância para atributos.

    Retorna um dicionário com severidades parciais e uma severidade agregada.
    Todas as severidades são normalizadas entre 0 (idêntico) e 1 (completamente diferente/mudança máxima).
    """

    # Pesos para agregar as diferentes fontes de severidade (podem ser ajustados)
    WEIGHT_ATTRIBUTE_CHANGE = 1.0 # Se o atributo mudar, é a mudança máxima para o literal
    WEIGHT_OPERATOR_CHANGE = 0.4
    WEIGHT_VALUE_CHANGE = 0.6

    attr1, op1, val1 = literal1
    attr2, op2, val2 = literal2

    severity_scores = {
        "attribute_changed_flag": 0.0, # 0 ou 1
        "operator_dissimilarity": 0.0, # 0 a 1
        "value_normalized_difference": 0.0, # 0 a 1
        "weighted_literal_severity": 0.0
    }

    # 1. Severidade da Mudança de Atributo
    if attr1 != attr2:
        severity_scores["attribute_changed_flag"] = 1.0
        # Se o atributo é diferente, consideramos a severidade do literal como máxima.
        # A comparação de operador e valor para atributos diferentes pode não ser significativa
        # ou deveria ser tratada como deleção de um literal e adição de outro.
        # Para a severidade da *modificação* deste par, se os atributos são diferentes, é uma grande mudança.
        severity_scores["weighted_literal_severity"] = WEIGHT_ATTRIBUTE_CHANGE * 1.0
        return severity_scores

    # Atributos são iguais, continuar a comparação

    # 2. Severidade da Mudança de Operador
    severity_scores["operator_dissimilarity"] = get_operator_dissimilarity(op1, op2)

    # 3. Severidade da Mudança de Valor Numérico
    # Só calcula se os atributos são os mesmos.
    # Poderíamos adicionar uma condição para só calcular se os operadores são "comparáveis"
    # (ex: ambos são de desigualdade, ou ambos de igualdade), mas por ora vamos calcular.
    # A normalização é importante aqui.
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        severity_scores["value_normalized_difference"] = calculate_value_normalized_difference(val1, val2)
    elif val1 != val2: # Se não forem numéricos mas diferentes (preparação para categóricos)
        severity_scores["value_normalized_difference"] = 1.0 # Mudança total para valores não numéricos diferentes

    # 4. Calcular a Severidade Agregada Ponderada do Literal
    # (Apenas se o atributo não mudou)
    severity_scores["weighted_literal_severity"] = (
        WEIGHT_OPERATOR_CHANGE * severity_scores["operator_dissimilarity"] +
        WEIGHT_VALUE_CHANGE * severity_scores["value_normalized_difference"]
    )
    # Garante que está no intervalo [0,1]
    severity_scores["weighted_literal_severity"] = min(max(severity_scores["weighted_literal_severity"], 0.0), 1.0)

    return severity_scores


# --- Testes da Parte 2 ---
if __name__ == "__main__":
    print("Iniciando testes da Parte 2: Comparação Detalhada de Literais Atômicos...\n")

    test_literals = [
        # Caso 1: Idênticos
        ( (['idade', '<', 30.0], ['idade', '<', 30.0]), "Idênticos" ),
        # Caso 2: Mudança leve de operador
        ( (['idade', '<', 30.0], ['idade', '<=', 30.0]), "Mudança leve de operador (< para <=)" ),
        # Caso 3: Mudança drástica de operador
        ( (['idade', '<', 30.0], ['idade', '>', 30.0]), "Mudança drástica de operador (< para >)" ),
        # Caso 4: Mudança de valor (pequena)
        ( (['salario', '>', 5000.0], ['salario', '>', 5500.0]), "Mudança pequena de valor" ),
        # Caso 5: Mudança de valor (grande)
        ( (['salario', '>', 5000.0], ['salario', '>', 10000.0]), "Mudança grande de valor" ),
        # Caso 6: Mudança de valor (para zero)
        ( (['risco', '<', 0.5], ['risco', '<', 0.0]), "Mudança de valor para zero" ),
        # Caso 7: Mudança de valor (de zero)
        ( (['risco', '<', 0.0], ['risco', '<', 0.5]), "Mudança de valor de zero" ),
        # Caso 8: Mudança de atributo
        ( (['cidade', '==', 1.0], ['estado', '==', 1.0]), "Mudança de atributo (valores categóricos como float por enquanto)" ), # Simula valor categórico como float
        # Caso 9: Mudança de operador e valor
        ( (['temp', '<=', 20.0], ['temp', '>', 25.0]), "Mudança de operador e valor" ),
        # Caso 10: Operadores não listados diretamente, mas simétricos
        ( (['contagem', '>', 5.0], ['contagem', '==', 5.0]), "Mudança moderada de operador (> para ==)" ),
        # Caso 11: Valores muito próximos
        ( (['precisao', '>=', 0.991], ['precisao', '>=', 0.990]), "Mudança de valor muito pequena" ),
        # Caso 12: Mudança de valor com um sendo negativo
        ( (['saldo', '>', 100.0], ['saldo', '>', -50.0]), "Mudança de valor com negativo" ),
        # Caso 13: Comparação com zero
        ( (['items', '==', 0.0], ['items', '==', 0.0]), "Comparação com zero, idênticos" ),
    ]

    for i, (literals, description) in enumerate(test_literals):
        lit1, lit2 = literals
        print(f"\n--- Teste Literal {i+1}: {description} ---")
        print(f"  Literal 1: {lit1}")
        print(f"  Literal 2: {lit2}")
        
        severity = calculate_atomic_condition_change_severity(lit1, lit2)
        print(f"  Resultados da Severidade:")
        for key, value in severity.items():
            print(f"    {key}: {value:.4f}")