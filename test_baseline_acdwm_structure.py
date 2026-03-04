"""
Teste de Estrutura: baseline_acdwm.py

PROPÓSITO:
    Validar que o baseline_acdwm.py está estruturalmente correto
    sem precisar das dependências completas.

TESTES:
    1. Importação do módulo
    2. Verificação de classes e funções
    3. Verificação de interfaces
    4. Validação de parâmetros

AUTOR: Claude Code
DATA: 2025-01-07
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s'
)
logger = logging.getLogger("test")


def print_test_header(test_name):
    """Imprime cabeçalho de teste"""
    print("\n" + "="*70)
    print(f"TEST: {test_name}")
    print("="*70)


def test_module_structure():
    """Testa estrutura do módulo baseline_acdwm"""
    print_test_header("Estrutura do Modulo baseline_acdwm.py")

    try:
        # Tenta importar apenas o que não depende de shared_evaluation
        import ast
        import inspect

        # Lê o arquivo
        filepath = Path(__file__).parent / 'baseline_acdwm.py'
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        # Parse AST
        tree = ast.parse(source)

        # Extrai funções e classes
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

        print("\n[OK] Arquivo parseado com sucesso")
        print(f"\nClasses encontradas ({len(set(classes))}):")
        for cls in sorted(set(classes)):
            print(f"  - {cls}")

        print(f"\nFuncoes encontradas ({len(set(functions))}):")
        for func in sorted(set(functions))[:10]:  # Primeiras 10
            print(f"  - {func}")

        # Valida que as principais estruturas existem
        required_classes = ['ACDWMEvaluator']
        required_functions = [
            'import_acdwm_modules',
            'create_acdwm_model',
            'run_acdwm_baseline'
        ]

        print("\nValidando estruturas requeridas:")

        for cls in required_classes:
            if cls in classes:
                print(f"  [OK] Classe {cls} encontrada")
            else:
                print(f"  [X] Classe {cls} NAO encontrada")
                return False

        for func in required_functions:
            if func in functions:
                print(f"  [OK] Funcao {func} encontrada")
            else:
                print(f"  [X] Funcao {func} NAO encontrada")
                return False

        return True

    except SyntaxError as e:
        print(f"[X] Erro de sintaxe: {e}")
        return False

    except Exception as e:
        print(f"[X] Erro: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_interface_compatibility():
    """Testa compatibilidade de interface"""
    print_test_header("Compatibilidade de Interface")

    try:
        import ast

        filepath = Path(__file__).parent / 'baseline_acdwm.py'
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)

        # Encontra classe ACDWMEvaluator
        acdwm_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'ACDWMEvaluator':
                acdwm_class = node
                break

        if not acdwm_class:
            print("[X] Classe ACDWMEvaluator nao encontrada")
            return False

        # Extrai métodos
        methods = [m.name for m in acdwm_class.body if isinstance(m, ast.FunctionDef)]

        print(f"\nMetodos da classe ACDWMEvaluator ({len(methods)}):")
        for method in methods:
            print(f"  - {method}")

        # Valida métodos requeridos (interface ChunkEvaluator)
        required_methods = [
            '__init__',
            'train_on_chunk',
            'test_on_chunk',
            'get_model_info'
        ]

        print("\nValidando interface ChunkEvaluator:")
        for method in required_methods:
            if method in methods:
                print(f"  [OK] Metodo {method} implementado")
            else:
                print(f"  [X] Metodo {method} NAO implementado")
                return False

        # Valida métodos específicos do ACDWM
        acdwm_methods = [
            'evaluate_train_then_test',
            'evaluate_test_then_train'
        ]

        print("\nValidando metodos especificos do ACDWM:")
        for method in acdwm_methods:
            if method in methods:
                print(f"  [OK] Metodo {method} implementado")
            else:
                print(f"  [X] Metodo {method} NAO implementado")
                return False

        return True

    except Exception as e:
        print(f"[X] Erro: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_validation():
    """Testa validação de parâmetros"""
    print_test_header("Validacao de Parametros")

    try:
        import ast

        filepath = Path(__file__).parent / 'baseline_acdwm.py'
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)

        # Encontra __init__ da classe ACDWMEvaluator
        init_method = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'ACDWMEvaluator':
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        init_method = item
                        break

        if not init_method:
            print("[X] Metodo __init__ nao encontrado")
            return False

        # Extrai argumentos
        args = [arg.arg for arg in init_method.args.args if arg.arg != 'self']

        print(f"\nParametros do __init__ ({len(args)}):")
        for arg in args:
            print(f"  - {arg}")

        # Valida parâmetros importantes
        required_params = ['acdwm_path', 'classes']
        optional_params = ['evaluation_mode', 'adaptive_chunk_size', 'theta']

        print("\nValidando parametros requeridos:")
        for param in required_params:
            if param in args:
                print(f"  [OK] Parametro {param} presente")
            else:
                print(f"  [X] Parametro {param} AUSENTE")
                return False

        print("\nValidando parametros opcionais:")
        for param in optional_params:
            if param in args:
                print(f"  [OK] Parametro {param} presente")

        return True

    except Exception as e:
        print(f"[X] Erro: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_documentation():
    """Testa documentação do módulo"""
    print_test_header("Documentacao")

    try:
        filepath = Path(__file__).parent / 'baseline_acdwm.py'
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Conta docstrings
        docstring_count = 0
        in_docstring = False
        docstring_lines = 0

        for line in lines:
            stripped = line.strip()
            if '"""' in stripped or "'''" in stripped:
                in_docstring = not in_docstring
                if in_docstring:
                    docstring_count += 1
            if in_docstring:
                docstring_lines += 1

        print(f"\nEstatisticas de documentacao:")
        print(f"  Total de linhas: {len(lines)}")
        print(f"  Docstrings: {docstring_count}")
        print(f"  Linhas de docstring: {docstring_lines}")
        print(f"  Ratio: {docstring_lines / len(lines) * 100:.1f}%")

        if docstring_count >= 5:
            print("\n  [OK] Bem documentado")
            return True
        else:
            print("\n  [!] Poderia ter mais documentacao")
            return True  # Não falha por isso

    except Exception as e:
        print(f"[X] Erro: {e}")
        return False


def run_all_tests():
    """Executa todos os testes"""

    print("\n" + "="*70)
    print("SUITE DE TESTES: baseline_acdwm.py (Estrutura)")
    print("="*70)

    tests = [
        ("Estrutura do Modulo", test_module_structure),
        ("Compatibilidade de Interface", test_interface_compatibility),
        ("Validacao de Parametros", test_parameter_validation),
        ("Documentacao", test_documentation)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n[OK] {test_name} PASSOU")
            else:
                failed += 1
                print(f"\n[X] {test_name} FALHOU")
        except Exception as e:
            failed += 1
            logger.error(f"[X] {test_name} FALHOU: {e}", exc_info=True)

    # Resumo
    print("\n" + "="*70)
    print("RESUMO DOS TESTES")
    print("="*70)
    print(f"Total: {len(tests)}")
    print(f"[OK] Passou: {passed}")
    print(f"[X] Falhou: {failed}")

    if failed == 0:
        print("\n[SUCCESS] TODOS OS TESTES ESTRUTURAIS PASSARAM!")
        print("\nProximo passo: Clonar repositorio ACDWM e testar integracao completa")
        return 0
    else:
        print(f"\n[WARNING] {failed} teste(s) falharam")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
