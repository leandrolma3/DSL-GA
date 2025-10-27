#!/usr/bin/env python
"""
Teste Rápido do Sistema - Verifica dependências e funcionalidade básica
"""

import sys
import os

print("="*70)
print("TESTE RÁPIDO - Sistema de Comparação GBML vs River")
print("="*70)

# Teste 1: Dependências básicas
print("\n[Teste 1] Verificando dependências básicas...")
basic_deps = ['numpy', 'yaml']
for dep in basic_deps:
    try:
        __import__(dep)
        print(f"  [OK] {dep}")
    except ImportError:
        print(f"  [ERRO] {dep} FALTANDO!")
        sys.exit(1)

# Teste 2: Módulos do GBML
print("\n[Teste 2] Verificando módulos do GBML...")
gbml_modules = ['constants', 'individual', 'rule_tree', 'fitness', 'utils']
for mod in gbml_modules:
    try:
        __import__(mod)
        print(f"  [OK] {mod}.py")
    except Exception as e:
        print(f"  [ERRO] {mod}.py - Erro: {e}")

# Teste 3: Novos módulos de comparação (sem imports pesados)
print("\n[Teste 3] Verificando novos módulos...")
if os.path.exists('shared_evaluation.py'):
    print("  [OK] shared_evaluation.py existe")
else:
    print("  [ERRO] shared_evaluation.py NAO ENCONTRADO!")

if os.path.exists('baseline_river.py'):
    print("  [OK] baseline_river.py existe")
else:
    print("  [ERRO] baseline_river.py NAO ENCONTRADO!")

if os.path.exists('gbml_evaluator.py'):
    print("  [OK] gbml_evaluator.py existe")
else:
    print("  [ERRO] gbml_evaluator.py NAO ENCONTRADO!")

if os.path.exists('compare_gbml_vs_river.py'):
    print("  [OK] compare_gbml_vs_river.py existe")
else:
    print("  [ERRO] compare_gbml_vs_river.py NAO ENCONTRADO!")

# Teste 4: Config
print("\n[Teste 4] Verificando configuracao...")
if os.path.exists('config.yaml'):
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"  [OK] config.yaml carregado ({len(config)} secoes)")
    except Exception as e:
        print(f"  [ERRO] Erro ao carregar config.yaml: {e}")
else:
    print("  [ERRO] config.yaml NAO ENCONTRADO!")

print("\n" + "="*70)
print("[OK] TESTE BASICO CONCLUIDO")
print("="*70)
print("\nPróximo passo: Instale as dependências restantes:")
print("  pip install pandas scikit-learn river matplotlib seaborn")
print("\nDepois execute:")
print("  python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --chunks 2")
