"""
Script para gerar YAMLs de re-execucao para datasets faltantes

Datasets faltantes:
- Batch 1: STAGGER_Abrupt_Chain, STAGGER_Abrupt_Recurring
- Batch 4: RANDOMTREE_Abrupt_Recurring
- Batch 5: IntelLabSensors

Autor: Claude Code
Data: 2025-12-15
"""

import yaml
from pathlib import Path


CONFIGS_DIR = Path("configs")


def load_yaml(filepath: Path):
    """Carrega arquivo YAML."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data, filepath: Path):
    """Salva YAML com formatacao adequada."""
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120
        )
    print(f"Salvo: {filepath}")


def create_batch1_rerun():
    """Cria YAML para re-executar STAGGER datasets do batch 1."""
    config = load_yaml(CONFIGS_DIR / "config_chunk2000_batch_1.yaml")

    # Modificar apenas os datasets a executar
    config['experiment_settings']['drift_simulation_experiments'] = [
        'STAGGER_Abrupt_Chain',
        'STAGGER_Abrupt_Recurring'
    ]

    # Remover standard_experiments se existir (nao precisamos)
    if 'standard_experiments' in config['experiment_settings']:
        del config['experiment_settings']['standard_experiments']

    save_yaml(config, CONFIGS_DIR / "config_chunk2000_batch_1_rerun.yaml")
    print("  -> STAGGER_Abrupt_Chain, STAGGER_Abrupt_Recurring")


def create_batch4_rerun():
    """Cria YAML para re-executar RANDOMTREE_Abrupt_Recurring do batch 4."""
    config = load_yaml(CONFIGS_DIR / "config_chunk2000_batch_4.yaml")

    # Modificar apenas os datasets a executar
    config['experiment_settings']['drift_simulation_experiments'] = [
        'RANDOMTREE_Abrupt_Recurring'
    ]

    save_yaml(config, CONFIGS_DIR / "config_chunk2000_batch_4_rerun.yaml")
    print("  -> RANDOMTREE_Abrupt_Recurring")


def create_batch5_rerun():
    """Cria YAML para re-executar IntelLabSensors do batch 5."""
    config = load_yaml(CONFIGS_DIR / "config_chunk2000_batch_5.yaml")

    # Modificar apenas os datasets a executar
    config['experiment_settings']['standard_experiments'] = [
        'IntelLabSensors'
    ]

    save_yaml(config, CONFIGS_DIR / "config_chunk2000_batch_5_rerun.yaml")
    print("  -> IntelLabSensors")


def main():
    print("=" * 60)
    print("GERANDO YAMLS DE RE-EXECUCAO")
    print("=" * 60)
    print()

    print("[Batch 1 - STAGGER datasets]")
    create_batch1_rerun()
    print()

    print("[Batch 4 - RANDOMTREE dataset]")
    create_batch4_rerun()
    print()

    print("[Batch 5 - IntelLabSensors dataset]")
    create_batch5_rerun()
    print()

    print("=" * 60)
    print("RESUMO")
    print("=" * 60)
    print("""
Arquivos criados:
  1. configs/config_chunk2000_batch_1_rerun.yaml
     - STAGGER_Abrupt_Chain
     - STAGGER_Abrupt_Recurring

  2. configs/config_chunk2000_batch_4_rerun.yaml
     - RANDOMTREE_Abrupt_Recurring

  3. configs/config_chunk2000_batch_5_rerun.yaml
     - IntelLabSensors

Comandos para execucao no Colab:

  # Batch 1 rerun
  !python main.py --config configs/config_chunk2000_batch_1_rerun.yaml

  # Batch 4 rerun
  !python main.py --config configs/config_chunk2000_batch_4_rerun.yaml

  # Batch 5 rerun
  !python main.py --config configs/config_chunk2000_batch_5_rerun.yaml
""")


if __name__ == "__main__":
    main()
