# rule_condition_parser_canonical_v1.py (baseado no v12)
import sys
sys.setrecursionlimit(5000)

from pyparsing import (
    Word, nums, alphas, alphanums, oneOf, opAssoc, infixNotation, Forward,
    Literal, Group, Suppress, ParseException, ParserElement, srange, ParseResults,
    pyparsing_common
)


ParserElement.enablePackrat()
LPAR, RPAR = map(Suppress, "()")

# ... (definições de attribute, numeric_value, comparison_operator, atomic_condition,
#      logical_expression, term, unwrap_paren_action, condition_parser
#      permanecem as mesmas da v12) ...

attribute_named_part = Word(alphas + "_", alphanums + "_")
attribute_numeric_part = Word(nums)
attribute = (attribute_named_part | attribute_numeric_part).setName("attribute")

raw_number = pyparsing_common.number
def ensure_float_action(tokens):
    try: return float(tokens[0])
    except Exception as e: raise ParseException(f"Invalid numeric value for ensure_float: {tokens[0]} - {e}")
numeric_value = raw_number.copy().setParseAction(ensure_float_action).setName("numeric_value")

comparison_operator = oneOf("<= >= < > == !=").setName("comparison_operator")
atomic_condition = Group(attribute + comparison_operator + numeric_value).setName("atomic_condition")

logical_expression = Forward()

def unwrap_paren_action(s, loc, tokens):
    return tokens[0]

term = atomic_condition | Group(LPAR + logical_expression + RPAR).setParseAction(unwrap_paren_action)

# Parse action para operadores binários (NÃO MODIFICADA DA V12)
# Deixa o infixNotation construir a árvore aninhada padrão
# A AND B AND C -> [[A, "AND", B], "AND", C]
# logical_expression << infixNotation(...) (COMO NA V12, SEM PARSE ACTION NOS OPERADORES)
logical_expression << infixNotation(
    term,
    [
        (Literal("AND"), 2, opAssoc.LEFT),
        (Literal("OR"), 2, opAssoc.LEFT),
    ],
) # type: ignore

condition_parser = logical_expression.copy()


def to_pure_python_list_recursive(item):
    if isinstance(item, ParseResults):
        # Se o ParseResults contém um único item e esse item NÃO é uma lista nem ParseResults,
        # então provavelmente é um token simples que queremos desembrulhar diretamente.
        # Ex: ParseResults("AND") ou ParseResults("<=")
        if len(item) == 1 and not isinstance(item[0], (list, ParseResults)):
            return to_pure_python_list_recursive(item[0]) # Retorna o item desembrulhado e processado
        
        # Caso contrário, trata como uma lista de sub-itens
        return [to_pure_python_list_recursive(sub_item) for sub_item in item.asList()]
    elif isinstance(item, list):
        return [to_pure_python_list_recursive(sub_item) for sub_item in item]
    else: # Elementos base (strings, floats, ints)
        return item

# NOVA FUNÇÃO DE PÓS-PROCESSAMENTO (SUBSTITUI build_ast_from_infix_notation_result da v12)
def build_and_canonicalize_ast(parsed_list_from_infix):
    """
    Constrói a AST no formato ['OPERATOR', left, right] e,
    para operadores comutativos (AND, OR), ordena os operandos.
    """
    # print(f"build_and_canonicalize_ast IN: {parsed_list_from_infix}")

    # Caso base: condição atômica ou operando já processado
    if not isinstance(parsed_list_from_infix, list) or \
       (len(parsed_list_from_infix) == 3 and parsed_list_from_infix[1] not in ("AND", "OR")):
        # print(f"  -> atomic or already processed: {parsed_list_from_infix}")
        return parsed_list_from_infix

    # Estrutura de infixNotation para A op B: [A, "op", B]
    # Estrutura de infixNotation para A op B op C (left assoc): [[A, "op", B], "op", C]
    if len(parsed_list_from_infix) == 3 and isinstance(parsed_list_from_infix[1], str) \
       and parsed_list_from_infix[1] in ("AND", "OR"):
        
        op = parsed_list_from_infix[1]
        left_operand_raw = parsed_list_from_infix[0]
        right_operand_raw = parsed_list_from_infix[2]

        # Canoniza recursivamente os operandos
        left_canonical = build_and_canonicalize_ast(left_operand_raw)
        right_canonical = build_and_canonicalize_ast(right_operand_raw)

        # Para operadores comutativos (AND, OR), ordena os operandos canonizados
        # para garantir uma representação única.
        # A ordenação pode ser baseada na representação string dos operandos canonizados.
        if op in ("AND", "OR"):
            # Converta para string para uma ordenação consistente e simples
            # Poderia ser uma ordenação mais complexa se necessário (ex: por profundidade, nro de nós)
            str_left = str(left_canonical)
            str_right = str(right_canonical)
            if str_left > str_right: # Define uma ordem arbitrária mas consistente
                # print(f"  -> SWAPPING for {op}: {str_right} < {str_left}")
                return [op, right_canonical, left_canonical]
            else:
                return [op, left_canonical, right_canonical]
        else: # Para operadores não comutativos (se adicionados no futuro)
            return [op, left_canonical, right_canonical]
            
    # Se for uma lista com um único elemento (resultado de Group(LPAR...RPAR) que não foi totalmente desembrulhado antes)
    if len(parsed_list_from_infix) == 1:
        # print(f"  -> single element list, processing content: {parsed_list_from_infix[0]}")
        return build_and_canonicalize_ast(parsed_list_from_infix[0])

    # print(f"  -> unexpected structure in build_and_canonicalize_ast: {parsed_list_from_infix}")
    return parsed_list_from_infix # Fallback


def parse_rule_condition(condition_string):
    try:
        clean_condition_string = condition_string.strip()
        if not clean_condition_string: return None
        
        parsed_results_obj = condition_parser.parseString(clean_condition_string, parseAll=True)
        # print(f"DEBUG raw parsed_results_obj from pyparsing: {parsed_results_obj.dump()}")
        
        # 1. Converter para listas Python puras, mantendo a estrutura aninhada do infixNotation
        initial_parsed_list_structure = to_pure_python_list_recursive(parsed_results_obj[0])
        # print(f"DEBUG initial_parsed_list_structure: {initial_parsed_list_structure}")

        # 2. Construir e Canonizar a AST
        final_ast = build_and_canonicalize_ast(initial_parsed_list_structure)
        return final_ast

    except ParseException as pe:
        print(f"Erro de parsing na condição: '{condition_string}'")
        print(f"  Original line: {pe.line}")
        print(f"  Marker:      {' ' * (pe.column - 1)}^")
        print(f"  Error:       {pe}")
        return None
    except Exception as e:
        print(f"Erro inesperado durante o parsing da condição '{condition_string}': {e}")
        # import traceback
        # traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Iniciando testes do parser de condição de regra (com canonização)...\n")
    import pprint
    pp = pprint.PrettyPrinter(indent=2, width=160)

    test_conditions = [
        "(car > 1.4408)",                                                                                           # 1
        "(((1 <= 1.3240) AND (0 <= 7.1542)) AND (((1 <= 6.4300) AND (0 <= 7.1542)) AND (1 <= 2.6249)))",          # 2
        "((1 <= 0.9298) AND (0 <= 5.9740))",                                                                       # 3
        # Testes de comutatividade
        "((0 <= 5.9740) AND (1 <= 0.9298))", # Invertido do Teste 3                                                 # 3b
        "(A > 1.0) OR (B == 2.0)",                                                                                 # Comutatividade 1
        "(B == 2.0) OR (A > 1.0)",                                                                                 # Comutatividade 2
        "(C > 3.0 AND (A > 1.0 AND B < 2.0))", # A e B devem ser ordenados, depois o resultado com C
        "((B < 2.0 AND A > 1.0) AND C > 3.0)", # Mesma lógica, estrutura diferente
        # Testes originais
        "((((2 < 4.6128) AND ((2 <= 4.3334) OR (2 > 5.4195))) AND ((2 < 1.2969) AND (0 >= 3.1477))))",             # ...
        "((2 < 4.9744) OR (1 > 3.1631))",
        "(1 <= 1.0 OR 2 > 2.0)",
        "(1 <= 1.0) OR (2 > 2.0)",
        "(A > 1.0) AND ((B < 2.0) OR (C == 3.0))",
        "(x_attr == 1.0)",
        "((y_attr != 2.0))",
        "((attrA > 1) AND (attrB < 2))",
        "(((1 <= 1.3240) AND (0 <= 9.6754)) AND (((((0 >= 3.4788) AND (1 < 9.3053)) OR (((((0 > 6.7311) OR (0 < 3.1027)) OR ((2 <= 6.0448) OR (0 <= 3.3004))) AND ((1 < 6.8433) OR ((0 <= 2.6562) OR (2 <= 6.7832)))) AND (((0 > 0.3796) OR (0 < 1.0288)) OR ((2 < 4.3652) OR (1 <= 7.4110))))) AND (0 <= 7.1542)) AND (1 <= 2.6249)))",
        "((attrA > 1 AND attrB < 2) OR (attrC == 3.0))",
        "(attrA > 1 AND (attrB < 2 OR attrC == 3.0))"
    ]
    # Adicionar erros no final
    test_conditions.extend([
        "(car > 1.A408)",
        "((1 <= 1.0) AND )",
    ])


    for i, condition_str in enumerate(test_conditions):
        current_test_str = condition_str
        print(f"\n--- Teste {i+1} ---")
        print(f"Input String: {current_test_str}")
        
        parsed = parse_rule_condition(current_test_str)
        if parsed:
            print("Estrutura Parseada (Canonizada):")
            pp.pprint(parsed)
        else:
            print("Falha no parsing.")