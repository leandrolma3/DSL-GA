"""
Script para gerar configuracoes YAML com chunk_size=2000

Este script le os YAMLs existentes (batch 1-7) e cria novos YAMLs
com as seguintes modificacoes:
- chunk_size: 1000 -> 2000
- evaluation_period: 1000 -> 2000
- base_results_dir: nova pasta experiments_chunk2000
- heatmap_save_directory: nova pasta

Autor: Claude Code
Data: 2025-12-12
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any


# Configuracoes
CONFIGS_DIR = Path("configs")
OUTPUT_DIR = Path("configs")  # Mesmo diretorio, novos nomes
BATCHES_TO_PROCESS = [1, 2, 3, 4, 5, 6, 7]

# Novos valores
NEW_CHUNK_SIZE = 2000
NEW_EVALUATION_PERIOD = 2000

# Mapeamento de diretorios antigos para novos
DIR_MAPPINGS = {
    # Phase 2 GBML (batches 1-4)
    "experiments_6chunks_phase2_gbml": "experiments_chunk2000_phase1",
    # Phase 3 Real (batches 5-7)
    "experiments_6chunks_phase3_real": "experiments_chunk2000_phase2",
}


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """Carrega arquivo YAML preservando estrutura."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], filepath: Path):
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
    print(f"  Salvo: {filepath}")


def update_path(path: str, old_base: str, new_base: str) -> str:
    """Atualiza caminho substituindo base antiga pela nova."""
    if old_base in path:
        return path.replace(old_base, new_base)
    return path


def process_config(config: Dict[str, Any], batch_num: int) -> Dict[str, Any]:
    """
    Processa configuracao aplicando modificacoes para chunk_size=2000.

    Modificacoes:
    1. data_params.chunk_size: 1000 -> 2000
    2. experiment_settings.evaluation_period: 1000 -> 2000
    3. experiment_settings.base_results_dir: atualiza pasta
    4. drift_analysis.heatmap_save_directory: atualiza pasta
    """
    # Criar copia para nao modificar original
    new_config = config.copy()

    # 1. Atualizar chunk_size
    if 'data_params' in new_config:
        new_config['data_params'] = new_config['data_params'].copy()
        new_config['data_params']['chunk_size'] = NEW_CHUNK_SIZE
        print(f"    chunk_size: 1000 -> {NEW_CHUNK_SIZE}")

    # 2. Atualizar evaluation_period
    if 'experiment_settings' in new_config:
        new_config['experiment_settings'] = new_config['experiment_settings'].copy()
        new_config['experiment_settings']['evaluation_period'] = NEW_EVALUATION_PERIOD
        print(f"    evaluation_period: 1000 -> {NEW_EVALUATION_PERIOD}")

        # 3. Atualizar base_results_dir
        if 'base_results_dir' in new_config['experiment_settings']:
            old_dir = new_config['experiment_settings']['base_results_dir']
            new_dir = old_dir
            for old_base, new_base in DIR_MAPPINGS.items():
                new_dir = update_path(new_dir, old_base, new_base)
            new_config['experiment_settings']['base_results_dir'] = new_dir
            print(f"    base_results_dir: ...{old_base}... -> ...{new_base}...")

    # 4. Atualizar heatmap_save_directory
    if 'drift_analysis' in new_config:
        new_config['drift_analysis'] = new_config['drift_analysis'].copy()
        if 'heatmap_save_directory' in new_config['drift_analysis']:
            old_heatmap = new_config['drift_analysis']['heatmap_save_directory']
            new_heatmap = old_heatmap
            for old_base, new_base in DIR_MAPPINGS.items():
                new_heatmap = update_path(new_heatmap, old_base, new_base)
            new_config['drift_analysis']['heatmap_save_directory'] = new_heatmap
            print(f"    heatmap_save_directory: atualizado")

    return new_config


def main():
    """Funcao principal."""
    print("=" * 70)
    print("GERACAO DE CONFIGS PARA CHUNK_SIZE=2000")
    print("=" * 70)
    print()

    # Verificar diretorio de configs
    if not CONFIGS_DIR.exists():
        print(f"ERRO: Diretorio {CONFIGS_DIR} nao encontrado!")
        return

    configs_gerados = []
    configs_falharam = []

    for batch_num in BATCHES_TO_PROCESS:
        input_file = CONFIGS_DIR / f"config_batch_{batch_num}.yaml"
        output_file = CONFIGS_DIR / f"config_chunk2000_batch_{batch_num}.yaml"

        print(f"\n[Batch {batch_num}]")
        print(f"  Entrada: {input_file}")

        if not input_file.exists():
            print(f"  AVISO: Arquivo nao encontrado, pulando...")
            configs_falharam.append(batch_num)
            continue

        try:
            # Carregar config original
            config = load_yaml(input_file)

            # Processar config
            new_config = process_config(config, batch_num)

            # Salvar novo config
            save_yaml(new_config, output_file)
            configs_gerados.append(batch_num)

        except Exception as e:
            print(f"  ERRO: {e}")
            configs_falharam.append(batch_num)

    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)
    print(f"\nConfigs gerados com sucesso: {len(configs_gerados)}")
    for batch in configs_gerados:
        print(f"  - config_chunk2000_batch_{batch}.yaml")

    if configs_falharam:
        print(f"\nConfigs que falharam: {len(configs_falharam)}")
        for batch in configs_falharam:
            print(f"  - batch_{batch}")

    print("\n" + "=" * 70)
    print("PROXIMOS PASSOS")
    print("=" * 70)
    print("""
1. Revisar os YAMLs gerados para garantir correcao
2. Criar os diretorios de resultados:
   - experiments_chunk2000_phase1/
   - experiments_chunk2000_phase2/
3. Testar com 1 dataset antes de executar batch completo
4. Executar experimentos no Colab
""")


if __name__ == "__main__":
    main()
