# rule_similarity_analyzer.py

import math # Pode ser necessário para futuras métricas, não usado diretamente agora

# --- Importações das Partes 1 e 2 ---
# Certifique-se de que os nomes dos arquivos correspondem aos seus.
# Se estiverem em subpastas, ajuste o caminho do import.
try:
    from rule_condition_parser import parse_rule_condition # PARTE 1
    from literal_comparison import calculate_atomic_condition_change_severity # PARTE 2
except ImportError as e:
    print(f"Erro de importação: {e}")
    print("Certifique-se de que 'rule_condition_parser_v12.py' e 'literal_comparison.py' estão acessíveis.")
    print("Você pode precisar ajustar os nomes dos arquivos importados neste script.")
    exit()

# --- Pesos para a Função de Severidade (podem ser ajustados) ---
WEIGHT_STRUCTURAL_MAJOR = 1.0  # Para mudança de atômica para lógica, ou de AND para OR
# WEIGHT_STRUCTURAL_MINOR = 0.3  # (Não usado na versão atual, mas pode ser para Jaccard)
# WEIGHT_LITERAL_COMPONENT = 0.7 # (Usado implicitamente na média das severidades dos operandos)

def get_all_atomic_literals(ast_node):
    """
    Extrai todas as condições atômicas (literais) de uma AST.
    Retorna um conjunto de tuplas (atributo, operador, valor) para facilitar a comparação de conjuntos.
    """
    literals = set()
    if not isinstance(ast_node, list):
        return literals
    
    # Verifica se é uma condição atômica ['atributo', 'operador', valor]
    if len(ast_node) == 3 and ast_node[1] not in ("AND", "OR"):
        try:
            # Garante que o valor seja hasheável (float e str são)
            # A AST já deve ter o valor convertido para float
            literals.add((ast_node[0], ast_node[1], ast_node[2]))
        except TypeError:
            # Fallback se o valor não for hasheável por algum motivo (ex: uma lista inesperada)
            literals.add((ast_node[0], ast_node[1], str(ast_node[2])))
    # Verifica se é uma expressão lógica ['OPERADOR', esquerdo, direito]
    elif len(ast_node) == 3 and ast_node[0] in ("AND", "OR"): # Operador é o primeiro elemento
        literals.update(get_all_atomic_literals(ast_node[1])) # Processa operando esquerdo
        literals.update(get_all_atomic_literals(ast_node[2])) # Processa operando direito
    # else: # Pode ser uma lista aninhada de um Group que não foi totalmente desembrulhada
          # ou um formato inesperado.
    #    print(f"DEBUG get_all_atomic_literals: Estrutura não reconhecida ou já processada: {ast_node}")
    return literals


def calculate_rule_modification_severity(ast_old, ast_new):
    """
    Calcula a severidade da modificação entre duas ASTs de regras canonizadas.
    Retorna um float entre 0.0 (idêntica) e 1.0 (completamente diferente/mudança máxima).
    """
    # print(f"DEBUG: Comparing ASTs:\n  Old: {ast_old}\n  New: {ast_new}")

    # 1. Se as ASTs canonizadas são idênticas (comparação estrutural e de conteúdo)
    if ast_old == ast_new:
        return 0.0

    # Verifica se as ASTs são válidas (listas)
    if not isinstance(ast_old, list) or not isinstance(ast_new, list):
        # print(f"DEBUG: Uma ou ambas as ASTs não são listas. Old: {type(ast_old)}, New: {type(ast_new)}")
        return WEIGHT_STRUCTURAL_MAJOR * 1.0 # Considera mudança máxima se a estrutura for inválida

    # Dentro de calculate_rule_modification_severity
    # Correção da identificação de condição atômica:
    # Uma condição atômica é ['attr_str', 'op_comparacao_str', val_num]
    # O primeiro elemento NÃO é 'AND' ou 'OR'. O segundo elemento NÃO é 'AND' ou 'OR'.
    is_atomic_old = (isinstance(ast_old, list) and len(ast_old) == 3 and
                    isinstance(ast_old[0], str) and ast_old[0] not in ("AND", "OR") and
                    isinstance(ast_old[1], str) and ast_old[1] not in ("AND", "OR"))

    is_atomic_new = (isinstance(ast_new, list) and len(ast_new) == 3 and
                    isinstance(ast_new[0], str) and ast_new[0] not in ("AND", "OR") and
                    isinstance(ast_new[1], str) and ast_new[1] not in ("AND", "OR"))

    # 2. Caso: Ambas são condições atômicas
    if is_atomic_old and is_atomic_new:
        #print(f"DEBUG: Comparing atomic conditions:") # NOVA LINHA
        #print(f"DEBUG:   Literal Old: {ast_old} (type: {type(ast_old[0])}, {type(ast_old[1])}, {type(ast_old[2])})") # NOVA LINHA
        #print(f"DEBUG:   Literal New: {ast_new} (type: {type(ast_new[0])}, {type(ast_new[1])}, {type(ast_new[2])})") # NOVA LINHA
        severity_details = calculate_atomic_condition_change_severity(ast_old, ast_new)
        return severity_details["weighted_literal_severity"]

    # 3. Caso: Uma é atômica e a outra é uma expressão lógica
    if is_atomic_old != is_atomic_new:
        # print("DEBUG: Mudança estrutural maior - Atômica vs. Lógica")
        return WEIGHT_STRUCTURAL_MAJOR * 1.0

    # 4. Caso: Ambas são expressões lógicas (devem ter o formato ['OPERADOR', esq, dir])
    # Verifica se ambas têm o formato de expressão lógica esperado
    if not (len(ast_old) == 3 and ast_old[0] in ("AND", "OR") and \
            len(ast_new) == 3 and ast_new[0] in ("AND", "OR")):
        # print(f"DEBUG: Estrutura de expressão lógica inesperada. Old: {ast_old}, New: {ast_new}")
        # Isso pode acontecer se uma AST for uma lista, mas não uma condição atômica nem uma expressão lógica válida.
        # Exemplo: se uma das ASTs é uma lista de literais (não deveria acontecer com o parser atual).
        # Ou se a canonização/parsing produziu algo inesperado.
        # Para ser seguro, considera mudança máxima.
        return WEIGHT_STRUCTURAL_MAJOR * 1.0


    op_old = ast_old[0]
    op_new = ast_new[0]

    # 4a. Operadores raiz diferentes (ex: AND vs OR)
    if op_old != op_new:
        # print(f"DEBUG: Mudança estrutural maior - Operador raiz diferente: {op_old} vs {op_new}")
        return WEIGHT_STRUCTURAL_MAJOR * 1.0

    # 4b. Mesmo operador raiz (ex: ambas são AND, ou ambas são OR)
    # Comparamos os operandos recursivamente.
    # A canonização já ordenou os operandos ast_old[1] com ast_old[2], e ast_new[1] com ast_new[2]
    # se o operador for comutativo.
    
    left_operand_old = ast_old[1]
    right_operand_old = ast_old[2]
    left_operand_new = ast_new[1]
    right_operand_new = ast_new[2]

    # Severidade dos operandos esquerdos
    severity_left = calculate_rule_modification_severity(left_operand_old, left_operand_new)
    # Severidade dos operandos direitos
    severity_right = calculate_rule_modification_severity(right_operand_old, right_operand_new)
    
    # Média simples das severidades dos operandos por enquanto.
    # No futuro, poderíamos usar Jaccard sobre os conjuntos de literais atômicos
    # para uma medida de dissimilaridade estrutural mais fina neste nível.
    avg_operand_severity = (severity_left + severity_right) / 2.0
    
    # print(f"DEBUG: Mesmo operador raiz '{op_old}'. Sev_Esq: {severity_left:.4f}, Sev_Dir: {severity_right:.4f}, Média: {avg_operand_severity:.4f}")
    return avg_operand_severity


# --- Testes da Parte 3 ---
if __name__ == "__main__":
    print("Iniciando testes da Parte 3: Cálculo da Severidade de Modificação entre Regras...\n")
    import pprint
    pp = pprint.PrettyPrinter(indent=2, width=160)

    # Reutilizar a função de parsing para conveniência nos testes
    # Supondo que rule_condition_parser_v12.py está no mesmo diretório ou no PYTHONPATH
    # e literal_comparison.py também.

    test_rule_pairs = [
        (   "Idênticas (após canonização)",
            "(A > 1.0 AND B < 2.0)",
            "(B < 2.0 AND A > 1.0)"
        ),
        (   "Atômicas Modificadas (operador e valor)",
            "(idade < 30.0)",
            "(idade >= 35.0)"
        ),
        (   "Atômica vs. Lógica (Adição de complexidade)",
            "(idade < 30.0)",
            "((idade < 30.0) AND (salario > 5000.0))"
        ),
        (   "Lógica vs. Atômica (Redução de complexidade)",
            "((idade < 30.0) AND (salario > 5000.0))",
            "(idade < 30.0)"
        ),
        (   "Mudança de Operador Lógico Principal",
            "((A > 1.0) AND (B < 2.0))",
            "((A > 1.0) OR (B < 2.0))"
        ),
        (   "Mesma Estrutura Lógica, Literais Modificados",
            "((idade < 30.0) AND (salario > 5000.0))",
            "((idade <= 35.0) AND (salario > 7000.0))"
        ),
        (   "Estruturas Lógicas Complexas e Diferentes (deveria ser alta severidade)",
            "(((X > 1) AND (Y < 2)) OR (Z == 3))",
            "((X > 1) AND ((Y < 5) AND (W == 0)))"
        ),
        (   "Canonização Aninhada Complexa (devem ser idênticas)",
            "(C > 3.0 AND (A > 1.0 AND B < 2.0))",
            "((B < 2.0 AND A > 1.0) AND C > 3.0)"
        ),
         (  "Regra de AGRAWAL (Chunk 2 vs Chunk 3 - Deletada/Nova)", # Esperamos alta severidade
            "(hyears > 1.4615)", # Do Chunk 2
            "(elevel <= 3.9378)"  # Do Chunk 3 (uma das novas)
        ),
        (   "Regra de SEA (Complexa, pequena variação de valor)",
            "(((1 <= 1.3240) AND (0 <= 7.1542)) AND (((1 <= 6.4300) AND (0 <= 7.1542)) AND (1 <= 2.6249)))",
            "(((1 <= 1.3240) AND (0 <= 7.1500)) AND (((1 <= 6.4300) AND (0 <= 7.1500)) AND (1 <= 2.6249)))" # 7.1542 -> 7.1500
        ),

    ]

    for i, (description, rule_str_old, rule_str_new) in enumerate(test_rule_pairs):
        print(f"\n--- Teste Regra {i+1}: {description} ---")
        print(f"  Regra Antiga (str): {rule_str_old}")
        print(f"  Regra Nova (str):   {rule_str_new}")

        ast_old = parse_rule_condition(rule_str_old)
        ast_new = parse_rule_condition(rule_str_new)

        if ast_old is None:
            print(f"  Falha ao parsear Regra Antiga: {rule_str_old}")
            continue
        if ast_new is None:
            print(f"  Falha ao parsear Regra Nova: {rule_str_new}")
            continue
        
        # print("  AST Antiga (canonizada):")
        # pp.pprint(ast_old)
        # print("  AST Nova (canonizada):")
        # pp.pprint(ast_new)

        severity = calculate_rule_modification_severity(ast_old, ast_new)
        print(f"  Severidade da Modificação da Regra: {severity:.4f}")