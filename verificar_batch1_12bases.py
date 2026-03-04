# -*- coding: utf-8 -*-
"""
Script de Verificação - Batch 1 Expandido (12 bases)
Valida que todas as 12 bases foram corrigidas para 6 chunks
"""

import yaml
import sys
from pathlib import Path

def verificar_batch1_completo():
    """Verifica se as 12 bases do Batch 1 foram corrigidas corretamente"""

    config_path = Path(__file__).parent / 'configs' / 'config_batch_1.yaml'

    print("=" * 80)
    print("VERIFICACAO BATCH 1 EXPANDIDO - 12 BASES ABRUPT")
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
    experiment_list = config.get('experiment_settings', {}).get('drift_simulation_experiments', [])

    print(f"[OK] Bases listadas em experiment_settings: {len(experiment_list)}")
    print()

    # Definir valores esperados para as 12 bases
    verificacoes = {
        # 5 bases originais (já corrigidas)
        'SEA_Abrupt_Simple': {
            'conceitos': ['f1', 'f3'],
            'duration_esperado': [3, 3],
            'drift_positions': [3000]
        },
        'AGRAWAL_Abrupt_Simple_Severe': {
            'conceitos': ['f1', 'f6'],
            'duration_esperado': [3, 3],
            'drift_positions': [3000]
        },
        'RBF_Abrupt_Severe': {
            'conceitos': ['c1', 'c2_severe'],
            'duration_esperado': [3, 3],
            'drift_positions': [3000]
        },
        'STAGGER_Abrupt_Chain': {
            'conceitos': ['f1', 'f2', 'f3'],
            'duration_esperado': [2, 2, 2],
            'drift_positions': [2000, 4000]
        },
        'HYPERPLANE_Abrupt_Simple': {
            'conceitos': ['plane1', 'plane2'],
            'duration_esperado': [3, 3],
            'drift_positions': [3000]
        },
        # 7 bases novas (recém-corrigidas)
        'SEA_Abrupt_Chain': {
            'conceitos': ['f1', 'f2', 'f4'],
            'duration_esperado': [2, 2, 2],
            'drift_positions': [2000, 4000]
        },
        'SEA_Abrupt_Recurring': {
            'conceitos': ['f1', 'f3', 'f1'],
            'duration_esperado': [2, 2, 2],
            'drift_positions': [2000, 4000],
            'recurring': True
        },
        'AGRAWAL_Abrupt_Simple_Mild': {
            'conceitos': ['f1', 'f2'],
            'duration_esperado': [3, 3],
            'drift_positions': [3000]
        },
        'AGRAWAL_Abrupt_Chain_Long': {
            'conceitos': ['f1', 'f2', 'f3', 'f4'],
            'duration_esperado': [2, 2, 1, 1],
            'drift_positions': [2000, 4000, 5000]
        },
        'RBF_Abrupt_Blip': {
            'conceitos': ['c1', 'c3_moderate', 'c1'],
            'duration_esperado': [2, 2, 2],
            'drift_positions': [2000, 4000],
            'blip': True
        },
        'STAGGER_Abrupt_Recurring': {
            'conceitos': ['f1', 'f3', 'f1'],
            'duration_esperado': [2, 2, 2],
            'drift_positions': [2000, 4000],
            'recurring': True
        },
        'RANDOMTREE_Abrupt_Simple': {
            'conceitos': ['tree1', 'tree2'],
            'duration_esperado': [3, 3],
            'drift_positions': [3000]
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
            duration_esperado = specs['duration_esperado'][i]

            if concept_id_atual != concept_id:
                erros.append(f"Conceito esperado '{concept_id}', encontrado '{concept_id_atual}'")

            if duration != duration_esperado:
                erros.append(
                    f"Conceito {concept_id}: duration_chunks={duration} "
                    f"(Esperado: {duration_esperado})"
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
            # Mostrar posições de drift
            positions = specs['drift_positions']
            print(f"   [OK] Drifts em: {', '.join(str(p) for p in positions)} instancias")

            # Indicadores especiais
            if specs.get('recurring'):
                print(f"   [INFO] Conceito RECORRENTE detectado")
            if specs.get('blip'):
                print(f"   [INFO] Padrao BLIP detectado")

            resultados.append((dataset_name, True, "OK"))

        print()

    # Resumo final
    print("=" * 80)
    print("RESUMO DA VERIFICACAO - BATCH 1 (12 BASES)")
    print("=" * 80)
    print()

    print(f"{'Dataset':<40} {'Status':<10} {'Detalhes'}")
    print("-" * 80)
    for dataset, ok, detalhes in resultados:
        status = "[OK]    " if ok else "[ERRO]  "
        print(f"{dataset:<40} {status:<10} {detalhes}")

    print()
    print("=" * 80)

    if todas_corretas:
        print("*** SUCESSO! Todas as 12 bases foram corrigidas corretamente! ***")
        print()
        print("Batch 1 - Distribuicao de Drifts:")
        print("  - 1 drift  em 3000: 5 bases (Simple)")
        print("  - 2 drifts em 2000,4000: 6 bases (Chain, Recurring, Blip)")
        print("  - 3 drifts em 2000,4000,5000: 1 base (Chain Long)")
        print()
        print("Padroes cobertos:")
        print("  - Simple: 5 bases")
        print("  - Chain: 3 bases")
        print("  - Recurring: 2 bases")
        print("  - Blip: 1 base")
        print("  - Chain Long: 1 base")
        print()
        print("Datasets cobertos:")
        print("  - SEA: 3 bases")
        print("  - AGRAWAL: 3 bases")
        print("  - RBF: 2 bases")
        print("  - STAGGER: 2 bases")
        print("  - HYPERPLANE: 1 base")
        print("  - RANDOMTREE: 1 base")
        print()
        print("[OK] Batch 1 PRONTO para re-execucao!")
        return True
    else:
        print("[ERRO] FALHA! Algumas bases nao foram corrigidas corretamente.")
        print()
        print("Por favor, revise o arquivo config_batch_1.yaml manualmente.")
        return False

if __name__ == '__main__':
    sucesso = verificar_batch1_completo()
    sys.exit(0 if sucesso else 1)
