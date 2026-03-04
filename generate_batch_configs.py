"""
generate_batch_configs.py

Script para gerar automaticamente os 12 arquivos config_batch_X.yaml
para o experimento de 6 chunks.

Uso:
    python generate_batch_configs.py --input config.yaml --output-dir configs/

Gera:
    configs/config_batch_1.yaml
    configs/config_batch_2.yaml
    ...
    configs/config_batch_12.yaml
"""

import yaml
import argparse
from pathlib import Path

# Distribuição dos datasets por batch (conforme DISTRIBUICAO_DATASETS_POR_BATCH.yaml)
BATCH_DATASETS = {
    1: [
        "SEA_Abrupt_Simple",
        "AGRAWAL_Abrupt_Simple_Severe",
        "RBF_Abrupt_Severe",
        "HYPERPLANE_Abrupt_Simple",
        "STAGGER_Abrupt_Chain"
    ],
    2: [
        "SEA_Gradual_Simple_Fast",
        "AGRAWAL_Gradual_Chain",
        "RBF_Gradual_Moderate",
        "HYPERPLANE_Gradual_Simple",
        "WAVEFORM_Gradual_Simple"
    ],
    3: [
        "SEA_Abrupt_Recurring",
        "AGRAWAL_Gradual_Recurring",
        "RBF_Severe_Gradual_Recurrent",
        "STAGGER_Abrupt_Recurring",
        "RANDOMTREE_Abrupt_Recurring"
    ],
    4: [
        "SEA_Abrupt_Chain_Noise",
        "AGRAWAL_Abrupt_Simple_Severe_Noise",
        "RBF_Abrupt_Blip_Noise",
        "HYPERPLANE_Gradual_Noise",
        "STAGGER_Abrupt_Chain_Noise"
    ],
    5: [
        "SINE_Abrupt_Simple",
        "SINE_Gradual_Recurring",
        "SINE_Abrupt_Recurring_Noise",
        "SEA_Abrupt_Chain",
        "SEA_Gradual_Simple_Slow"
    ],
    6: [
        "LED_Abrupt_Simple",
        "LED_Gradual_Simple",
        "RANDOMTREE_Abrupt_Simple",
        "RANDOMTREE_Gradual_Simple",
        "WAVEFORM_Abrupt_Simple"
    ],
    7: [
        "AGRAWAL_Gradual_Mild_to_Severe",
        "RBF_Gradual_Severe",
        "RANDOMTREE_Gradual_Noise",
        "STAGGER_Gradual_Chain",
        "STAGGER_Mixed_Recurring"
    ],
    8: [
        "RBF_Abrupt_Blip",
        "AGRAWAL_Gradual_Blip",
        "AGRAWAL_Abrupt_Chain_Long",
        "SEA_Gradual_Recurring",
        "RBF_Gradual_Severe_Noise"
    ],
    9: [
        "Bartosz_RandomTree_drift",
        "Bartosz_Agrawal_recurring_drift",
        "Bartosz_SEA_drift_noise",
        "AGRAWAL_Gradual_Recurring_Noise",
        "AGRAWAL_Abrupt_Simple_Mild"
    ],
    10: [
        "SEA_Stationary",
        "AGRAWAL_Stationary",
        "RBF_Stationary",
        "LED_Stationary",
        "HYPERPLANE_Stationary"
    ],
    11: [
        "RANDOMTREE_Stationary",
        "STAGGER_Stationary",
        "WAVEFORM_Stationary",
        "SINE_Stationary",
        "AssetNegotiation_F2"
    ],
    12: [
        "AssetNegotiation_F3",
        "AssetNegotiation_F4"
    ]
}


def load_base_config(config_path):
    """Carrega o config.yaml base"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_batch_config(base_config, batch_num, datasets, output_dir):
    """Gera um config YAML específico para um batch"""

    # Fazer uma cópia profunda do config base
    import copy
    batch_config = copy.deepcopy(base_config)

    # Modificar parâmetros específicos do batch
    batch_config['experiment_settings']['drift_simulation_experiments'] = datasets
    batch_config['experiment_settings']['base_results_dir'] = \
        f"/content/drive/MyDrive/DSL-AG-hybrid/experiments_6chunks_phase1_gbml/batch_{batch_num}"

    # Garantir parâmetros corretos para 6 chunks
    batch_config['data_params']['chunk_size'] = 3000
    batch_config['data_params']['num_chunks'] = 8  # Gera 6 úteis + 2 buffer
    batch_config['data_params']['max_instances'] = 24000  # 8 × 3000

    # Salvar o arquivo
    output_file = Path(output_dir) / f"config_batch_{batch_num}.yaml"
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(batch_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"[OK] Gerado: {output_file} ({len(datasets)} datasets)")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Gera configs YAML para os 12 batches do experimento 6 chunks'
    )
    parser.add_argument(
        '--input',
        default='config.yaml',
        help='Arquivo config.yaml base (default: config.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        default='configs',
        help='Diretório de saída (default: configs/)'
    )

    args = parser.parse_args()

    # Criar diretório de saída se não existir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Carregar config base
    print(f"\n[*] Carregando config base: {args.input}")
    base_config = load_base_config(args.input)

    # Gerar configs para cada batch
    print(f"\n[*] Gerando configs para 12 batches...\n")

    generated_files = []
    for batch_num in range(1, 13):
        datasets = BATCH_DATASETS[batch_num]
        output_file = generate_batch_config(
            base_config,
            batch_num,
            datasets,
            output_dir
        )
        generated_files.append(output_file)

    # Resumo
    print(f"\n{'=' * 60}")
    print(f"[OK] TODOS OS CONFIGS GERADOS COM SUCESSO!")
    print(f"{'=' * 60}")
    print(f"\nTotal de arquivos gerados: {len(generated_files)}")
    print(f"Diretorio de saida: {output_dir.absolute()}\n")

    print("[*] Arquivos gerados:")
    for i, file in enumerate(generated_files, 1):
        num_datasets = len(BATCH_DATASETS[i])
        print(f"  {i:2d}. {file.name:25s} ({num_datasets} datasets)")

    print(f"\n{'=' * 60}")
    print("[*] PROXIMOS PASSOS:")
    print("=" * 60)
    print("1. Revisar os arquivos gerados em:", output_dir)
    print("2. Fazer upload para o Colab/Google Drive")
    print("3. Executar batch_1 primeiro para validação")
    print("4. Após sucesso, executar os demais batches")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
