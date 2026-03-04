#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Verificação das Correções - Batch 1
Valida que todas as mudanças de duration_chunks foram aplicadas corretamente
"""

import yaml
import sys
from pathlib import Path

def verificar_correcoes():
    """Verifica se as correções foram aplicadas corretamente no config_batch_1.yaml"""

    config_path = Path(__file__).parent / 'configs' / 'config_batch_1.yaml'

    print("=" * 80)
    print("VERIFICACAO DAS CORRECOES - BATCH 1")
    print("=" * 80)
    print(f"\nArquivo: {config_path}")
    print()

    # Carregar configuração
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("[OK] Arquivo YAML carregado com sucesso")
    except FileNotFoundError:
        print(f"[ERRO] Arquivo nao encontrado: {config_path}")
        return False
    except yaml.YAMLError as e:
        print(f"[ERRO] Sintaxe YAML invalida: {e}")
        return False

    streams = config.get('experimental_streams', {})

    if not streams:
        print("[ERRO] Secao 'experimental_streams' nao encontrada")
        return False

    print("[OK] Secao 'experimental_streams' encontrada")
    print()

    # Definir valores esperados
    verificacoes = {
        'SEA_Abrupt_Simple': {
            'conceitos': ['f1', 'f3'],
            'duration_esperado': 3,
            'duration_antigo': 5,
            'drift_position': 3000
        },
        'AGRAWAL_Abrupt_Simple_Severe': {
            'conceitos': ['f1', 'f6'],
            'duration_esperado': 3,
            'duration_antigo': 5,
            'drift_position': 3000
        },
        'RBF_Abrupt_Severe': {
            'conceitos': ['c1', 'c2_severe'],
            'duration_esperado': 3,
            'duration_antigo': 5,
            'drift_position': 3000
        },
        'STAGGER_Abrupt_Chain': {
            'conceitos': ['f1', 'f2', 'f3'],
            'duration_esperado': 2,
            'duration_antigo': 4,
            'drift_positions': [2000, 4000]
        },
        'HYPERPLANE_Abrupt_Simple': {
            'conceitos': ['plane1', 'plane2'],
            'duration_esperado': 3,
            'duration_antigo': 6,
            'drift_position': 3000
        }
    }

    todas_corretas = True
    resultados = []

    # Verificar cada dataset
    for dataset_name, specs in verificacoes.items():
        print(f"[>] Verificando: {dataset_name}")
        print("-" * 80)

        if dataset_name not in streams:
            print(f"   [ERRO] Dataset nao encontrado na configuracao")
            todas_corretas = False
            resultados.append((dataset_name, False, "Dataset nao encontrado"))
            print()
            continue

        dataset = streams[dataset_name]
        concept_seq = dataset.get('concept_sequence', [])

        if not concept_seq:
            print(f"   [ERRO] concept_sequence vazio ou nao encontrado")
            todas_corretas = False
            resultados.append((dataset_name, False, "concept_sequence vazio"))
            print()
            continue

        # Verificar cada conceito
        erros = []
        for i, concept_id in enumerate(specs['conceitos']):
            if i >= len(concept_seq):
                erros.append(f"Conceito {concept_id} nao encontrado (indice {i})")
                continue

            concept = concept_seq[i]
            concept_id_atual = concept.get('concept_id', '')
            duration = concept.get('duration_chunks', None)

            if concept_id_atual != concept_id:
                erros.append(f"Conceito esperado '{concept_id}', encontrado '{concept_id_atual}'")

            if duration != specs['duration_esperado']:
                if duration == specs['duration_antigo']:
                    erros.append(
                        f"Conceito {concept_id}: duration_chunks={duration} "
                        f"(AINDA TEM O VALOR ANTIGO! Esperado: {specs['duration_esperado']})"
                    )
                else:
                    erros.append(
                        f"Conceito {concept_id}: duration_chunks={duration} "
                        f"(Esperado: {specs['duration_esperado']})"
                    )
            else:
                print(f"   [OK] {concept_id}: duration_chunks = {duration}")

        if erros:
            print("   [ERRO] ERROS ENCONTRADOS:")
            for erro in erros:
                print(f"      - {erro}")
            todas_corretas = False
            resultados.append((dataset_name, False, "; ".join(erros)))
        else:
            # Calcular posição do drift
            if 'drift_positions' in specs:
                positions = specs['drift_positions']
                print(f"   [OK] Drifts em: {', '.join(str(p) for p in positions)} instancias")
            else:
                position = specs['drift_position']
                print(f"   [OK] Drift em: {position} instancias")
            resultados.append((dataset_name, True, "OK"))

        print()

    # Resumo final
    print("=" * 80)
    print("RESUMO DA VERIFICACAO")
    print("=" * 80)
    print()

    print(f"{'Dataset':<35} {'Status':<10} {'Detalhes'}")
    print("-" * 80)
    for dataset, ok, detalhes in resultados:
        status = "[OK]    " if ok else "[ERRO]  "
        print(f"{dataset:<35} {status:<10} {detalhes}")

    print()
    print("=" * 80)

    if todas_corretas:
        print("*** SUCESSO! Todas as 10 correcoes foram aplicadas corretamente! ***")
        print()
        print("Drifts corrigidos:")
        print("  - SEA, AGRAWAL, RBF, HYPERPLANE: 5000/6000 -> 3000 [OK]")
        print("  - STAGGER: 4000,8000 -> 2000,4000 [OK]")
        print()
        print("[OK] Configuracao PRONTA para re-execucao!")
        return True
    else:
        print("[ERRO] FALHA! Algumas correcoes nao foram aplicadas corretamente.")
        print()
        print("Por favor, revise o arquivo config_batch_1.yaml manualmente.")
        return False

if __name__ == '__main__':
    sucesso = verificar_correcoes()
    sys.exit(0 if sucesso else 1)
