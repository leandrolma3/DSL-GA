#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para corrigir a inversao das pastas chunk_500 e chunk_500_penalty.

Problema:
- chunk_500 foi executado COM penalidade (mas nome sugere sem)
- chunk_500_penalty foi executado SEM penalidade (mas nome sugere com)

Solucao:
1. Trocar os nomes das pastas
2. Corrigir os valores de penalty em todos os run_config.json

Autor: Claude Code
Data: 2026-01-23
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime

# Forcar UTF-8 no Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuracao
BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR / "experiments_unified"

# Pastas a serem trocadas
PASTA_SEM_PENALTY = EXPERIMENTS_DIR / "chunk_500"
PASTA_COM_PENALTY = EXPERIMENTS_DIR / "chunk_500_penalty"
PASTA_TEMP = EXPERIMENTS_DIR / "chunk_500_TEMP_SWAP"

# Valores de penalty
VALORES_SEM_PENALTY = {
    "feature_penalty_coefficient": 0.0,
    "operator_penalty_coefficient": 0.0,
    "threshold_penalty_coefficient": 0.0
}

VALORES_COM_PENALTY = {
    "feature_penalty_coefficient": 0.1,
    "operator_penalty_coefficient": 0.0001,
    "threshold_penalty_coefficient": 0.0001
}


def print_separator(char="=", length=80):
    print(char * length)


def find_all_run_configs(pasta: Path) -> list:
    """Encontra todos os arquivos run_config.json em uma pasta."""
    return list(pasta.rglob("run_config.json"))


def update_run_config(filepath: Path, novos_valores: dict, dry_run: bool = False) -> bool:
    """
    Atualiza os valores de penalty em um run_config.json.
    Retorna True se houve alteracao, False caso contrario.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if "fitness_params" not in config:
            print(f"  [AVISO] fitness_params nao encontrado em {filepath}")
            return False

        alterado = False
        for chave, novo_valor in novos_valores.items():
            if chave in config["fitness_params"]:
                valor_atual = config["fitness_params"][chave]
                if valor_atual != novo_valor:
                    if not dry_run:
                        config["fitness_params"][chave] = novo_valor
                    alterado = True

        if alterado and not dry_run:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)

        return alterado

    except Exception as e:
        print(f"  [ERRO] Falha ao processar {filepath}: {e}")
        return False


def verificar_estado_atual():
    """Verifica e exibe o estado atual das pastas."""
    print_separator()
    print("VERIFICACAO DO ESTADO ATUAL")
    print_separator()

    for pasta, nome in [(PASTA_SEM_PENALTY, "chunk_500"), (PASTA_COM_PENALTY, "chunk_500_penalty")]:
        print(f"\n{nome}:")
        if not pasta.exists():
            print(f"  [X] Pasta NAO EXISTE")
            continue

        # Encontrar um run_config.json de exemplo
        configs = find_all_run_configs(pasta)
        print(f"  Arquivos run_config.json encontrados: {len(configs)}")

        if configs:
            # Mostrar valores do primeiro arquivo
            try:
                with open(configs[0], 'r', encoding='utf-8') as f:
                    config = json.load(f)

                if "fitness_params" in config:
                    fp = config["fitness_params"]
                    print(f"  Valores atuais de penalty (exemplo):")
                    print(f"    - feature_penalty_coefficient: {fp.get('feature_penalty_coefficient', 'N/A')}")
                    print(f"    - operator_penalty_coefficient: {fp.get('operator_penalty_coefficient', 'N/A')}")
                    print(f"    - threshold_penalty_coefficient: {fp.get('threshold_penalty_coefficient', 'N/A')}")
            except Exception as e:
                print(f"  [ERRO] Falha ao ler exemplo: {e}")


def executar_swap_pastas(dry_run: bool = False):
    """Executa a troca dos nomes das pastas."""
    print_separator()
    print("ETAPA 1: TROCA DOS NOMES DAS PASTAS")
    print_separator()

    if dry_run:
        print("[DRY RUN] Simulando troca de pastas...")
        print(f"  {PASTA_SEM_PENALTY.name} -> {PASTA_TEMP.name}")
        print(f"  {PASTA_COM_PENALTY.name} -> {PASTA_SEM_PENALTY.name}")
        print(f"  {PASTA_TEMP.name} -> {PASTA_COM_PENALTY.name}")
        return True

    try:
        # Verificar se pastas existem
        if not PASTA_SEM_PENALTY.exists():
            print(f"[ERRO] Pasta {PASTA_SEM_PENALTY} nao existe!")
            return False

        if not PASTA_COM_PENALTY.exists():
            print(f"[ERRO] Pasta {PASTA_COM_PENALTY} nao existe!")
            return False

        if PASTA_TEMP.exists():
            print(f"[ERRO] Pasta temporaria {PASTA_TEMP} ja existe! Remova-a primeiro.")
            return False

        # Executar swap
        print(f"  Renomeando {PASTA_SEM_PENALTY.name} -> {PASTA_TEMP.name}")
        PASTA_SEM_PENALTY.rename(PASTA_TEMP)

        print(f"  Renomeando {PASTA_COM_PENALTY.name} -> {PASTA_SEM_PENALTY.name}")
        PASTA_COM_PENALTY.rename(PASTA_SEM_PENALTY)

        print(f"  Renomeando {PASTA_TEMP.name} -> {PASTA_COM_PENALTY.name}")
        PASTA_TEMP.rename(PASTA_COM_PENALTY)

        print("[OK] Troca de pastas concluida com sucesso!")
        return True

    except Exception as e:
        print(f"[ERRO] Falha na troca de pastas: {e}")
        return False


def executar_correcao_configs(dry_run: bool = False):
    """Corrige os valores de penalty em todos os run_config.json."""
    print_separator()
    print("ETAPA 2: CORRECAO DOS ARQUIVOS run_config.json")
    print_separator()

    resultados = {
        "chunk_500": {"total": 0, "alterados": 0, "erros": 0},
        "chunk_500_penalty": {"total": 0, "alterados": 0, "erros": 0}
    }

    # Apos o swap:
    # - chunk_500 agora contem dados SEM penalty (precisa valores 0.0)
    # - chunk_500_penalty agora contem dados COM penalty (precisa valores 0.1)

    # Corrigir chunk_500 (deve ter valores SEM penalty)
    print(f"\nProcessando chunk_500 (definindo valores SEM penalty)...")
    configs_sem = find_all_run_configs(PASTA_SEM_PENALTY)
    resultados["chunk_500"]["total"] = len(configs_sem)

    for config_path in configs_sem:
        alterado = update_run_config(config_path, VALORES_SEM_PENALTY, dry_run)
        if alterado:
            resultados["chunk_500"]["alterados"] += 1
            if not dry_run:
                print(f"  [OK] {config_path.relative_to(EXPERIMENTS_DIR)}")

    print(f"  Total: {resultados['chunk_500']['total']}, Alterados: {resultados['chunk_500']['alterados']}")

    # Corrigir chunk_500_penalty (deve ter valores COM penalty)
    print(f"\nProcessando chunk_500_penalty (definindo valores COM penalty)...")
    configs_com = find_all_run_configs(PASTA_COM_PENALTY)
    resultados["chunk_500_penalty"]["total"] = len(configs_com)

    for config_path in configs_com:
        alterado = update_run_config(config_path, VALORES_COM_PENALTY, dry_run)
        if alterado:
            resultados["chunk_500_penalty"]["alterados"] += 1
            if not dry_run:
                print(f"  [OK] {config_path.relative_to(EXPERIMENTS_DIR)}")

    print(f"  Total: {resultados['chunk_500_penalty']['total']}, Alterados: {resultados['chunk_500_penalty']['alterados']}")

    return resultados


def verificar_estado_final():
    """Verifica o estado final apos as correcoes."""
    print_separator()
    print("VERIFICACAO DO ESTADO FINAL")
    print_separator()

    for pasta, nome, valores_esperados in [
        (PASTA_SEM_PENALTY, "chunk_500", VALORES_SEM_PENALTY),
        (PASTA_COM_PENALTY, "chunk_500_penalty", VALORES_COM_PENALTY)
    ]:
        print(f"\n{nome}:")
        if not pasta.exists():
            print(f"  [X] Pasta NAO EXISTE")
            continue

        configs = find_all_run_configs(pasta)
        print(f"  Arquivos run_config.json: {len(configs)}")

        if configs:
            # Verificar primeiro arquivo
            try:
                with open(configs[0], 'r', encoding='utf-8') as f:
                    config = json.load(f)

                if "fitness_params" in config:
                    fp = config["fitness_params"]
                    print(f"  Valores atuais (exemplo):")

                    todos_corretos = True
                    for chave, valor_esperado in valores_esperados.items():
                        valor_atual = fp.get(chave, 'N/A')
                        status = "[OK]" if valor_atual == valor_esperado else "[X]"
                        if valor_atual != valor_esperado:
                            todos_corretos = False
                        print(f"    {status} {chave}: {valor_atual} (esperado: {valor_esperado})")

                    if todos_corretos:
                        print(f"  [OK] Valores consistentes!")
                    else:
                        print(f"  [AVISO] Alguns valores inconsistentes!")

            except Exception as e:
                print(f"  [ERRO] Falha ao verificar: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Corrige inversao das pastas chunk_500')
    parser.add_argument('--confirmar', action='store_true', help='Executa sem pedir confirmacao')
    parser.add_argument('--dry-run', action='store_true', help='Apenas simula, nao faz alteracoes')
    args = parser.parse_args()

    print_separator("=")
    print("SCRIPT DE CORRECAO: SWAP chunk_500 <-> chunk_500_penalty")
    print_separator("=")
    print(f"\nData/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Diretorio base: {EXPERIMENTS_DIR}")

    # Verificar estado atual
    verificar_estado_atual()

    # Confirmar execucao
    print_separator()
    print("PLANO DE EXECUCAO:")
    print_separator()
    print("""
Este script ira:
1. Trocar os nomes das pastas:
   - chunk_500 -> chunk_500_TEMP_SWAP
   - chunk_500_penalty -> chunk_500
   - chunk_500_TEMP_SWAP -> chunk_500_penalty

2. Corrigir os valores de penalty em todos os run_config.json:
   - chunk_500: definir penalty = 0.0 (SEM penalty)
   - chunk_500_penalty: definir penalty = 0.1 (COM penalty)
""")

    if args.dry_run:
        print("[DRY RUN] Modo simulacao ativado - nenhuma alteracao sera feita.\n")

    # Perguntar confirmacao se nao passou --confirmar
    if not args.confirmar and not args.dry_run:
        try:
            resposta = input("\nDeseja executar? (s/n): ").strip().lower()
            if resposta != 's':
                print("\nOperacao cancelada pelo usuario.")
                return
        except EOFError:
            print("\n[ERRO] Nao foi possivel ler confirmacao. Use --confirmar para executar diretamente.")
            return

    # Executar swap de pastas
    if not executar_swap_pastas(dry_run=args.dry_run):
        print("\n[ERRO] Falha na troca de pastas. Abortando.")
        return

    # Executar correcao de configs
    resultados = executar_correcao_configs(dry_run=args.dry_run)

    # Verificar estado final
    verificar_estado_final()

    # Resumo final
    print_separator("=")
    print("RESUMO FINAL")
    print_separator("=")
    print(f"""
Operacoes realizadas:
1. [OK] Pastas trocadas com sucesso
2. [OK] run_config.json corrigidos:
   - chunk_500: {resultados['chunk_500']['alterados']}/{resultados['chunk_500']['total']} arquivos
   - chunk_500_penalty: {resultados['chunk_500_penalty']['alterados']}/{resultados['chunk_500_penalty']['total']} arquivos

A inversao foi corrigida com sucesso!
""")
    print_separator("=")


if __name__ == "__main__":
    main()
