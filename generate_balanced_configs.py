"""
Script para gerar YAMLs do experimento BALANCEADO (Estrategia A - Suave)

Este script le os YAMLs existentes (config_chunk2000_batch_*.yaml) e cria novos YAMLs
modificando APENAS os diretorios de resultados:
- base_results_dir: experiments_balanced_phase1 ou experiments_balanced_phase2
- heatmap_save_directory: experiments_balanced_*/heatmaps

TODOS os outros parametros sao mantidos IDENTICOS para garantir comparacao justa.

Modificacao no codigo (fitness.py linha 389):
- PENALTY_WEIGHT = 0.1 (era 0.0)

Autor: Claude Code
Data: 2025-12-15
"""

import yaml
from pathlib import Path


CONFIGS_DIR = Path("configs")
BATCHES_TO_PROCESS = [1, 2, 3, 4, 5, 6, 7]

# Mapeamento de diretorios
DIR_MAPPINGS = {
    # Phase 1 (batches 1-4): drift_simulation
    "experiments_chunk2000_phase1": "experiments_balanced_phase1",
    # Phase 2 (batches 5-7): standard/real
    "experiments_chunk2000_phase2": "experiments_balanced_phase2",
}


def load_yaml(filepath: Path):
    """Carrega arquivo YAML preservando estrutura."""
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
    print(f"  Salvo: {filepath}")


def update_path(path: str) -> str:
    """Atualiza caminho substituindo base antiga pela nova."""
    for old_base, new_base in DIR_MAPPINGS.items():
        if old_base in path:
            return path.replace(old_base, new_base)
    return path


def process_config(config: dict, batch_num: int) -> dict:
    """
    Processa configuracao modificando APENAS os diretorios.

    Modificacoes:
    1. experiment_settings.base_results_dir
    2. drift_analysis.heatmap_save_directory

    TODOS os outros parametros sao PRESERVADOS.
    """
    # Modificar base_results_dir
    if 'experiment_settings' in config:
        if 'base_results_dir' in config['experiment_settings']:
            old_dir = config['experiment_settings']['base_results_dir']
            new_dir = update_path(old_dir)
            config['experiment_settings']['base_results_dir'] = new_dir
            print(f"    base_results_dir: {old_dir.split('/')[-2]}/{old_dir.split('/')[-1]}")
            print(f"                   -> {new_dir.split('/')[-2]}/{new_dir.split('/')[-1]}")

    # Modificar heatmap_save_directory
    if 'drift_analysis' in config:
        if 'heatmap_save_directory' in config['drift_analysis']:
            old_heatmap = config['drift_analysis']['heatmap_save_directory']
            new_heatmap = update_path(old_heatmap)
            config['drift_analysis']['heatmap_save_directory'] = new_heatmap
            print(f"    heatmap_save_directory: atualizado")

    return config


def validate_consistency(original: dict, modified: dict) -> list:
    """
    Valida que apenas os diretorios foram modificados.
    Retorna lista de diferencas encontradas (alem dos diretorios esperados).
    """
    differences = []

    # Parametros que devem permanecer IDENTICOS
    params_to_check = [
        ('data_params', 'chunk_size'),
        ('data_params', 'num_chunks'),
        ('data_params', 'max_instances'),
        ('experiment_settings', 'evaluation_period'),
        ('experiment_settings', 'num_runs'),
        ('ga_params', 'population_size'),
        ('ga_params', 'max_generations'),
        ('ga_params', 'elitism_rate'),
        ('fitness_params', 'initial_regularization_coefficient'),
        ('fitness_params', 'feature_penalty_coefficient'),
        ('fitness_params', 'gmean_bonus_coefficient'),
    ]

    for section, param in params_to_check:
        orig_val = original.get(section, {}).get(param)
        mod_val = modified.get(section, {}).get(param)
        if orig_val != mod_val:
            differences.append(f"{section}.{param}: {orig_val} -> {mod_val}")

    return differences


def main():
    print("=" * 70)
    print("GERACAO DE YAMLS BALANCEADOS (Estrategia A - PENALTY_WEIGHT=0.1)")
    print("=" * 70)
    print()
    print("IMPORTANTE: Apenas os diretorios de resultados serao modificados.")
    print("Todos os outros parametros permanecem IDENTICOS aos experimentos anteriores.")
    print()

    if not CONFIGS_DIR.exists():
        print(f"ERRO: Diretorio {CONFIGS_DIR} nao encontrado!")
        return

    configs_gerados = []
    configs_falharam = []
    validation_issues = []

    for batch_num in BATCHES_TO_PROCESS:
        input_file = CONFIGS_DIR / f"config_chunk2000_batch_{batch_num}.yaml"
        output_file = CONFIGS_DIR / f"config_balanced_batch_{batch_num}.yaml"

        print(f"\n[Batch {batch_num}]")
        print(f"  Entrada: {input_file.name}")
        print(f"  Saida:   {output_file.name}")

        if not input_file.exists():
            print(f"  AVISO: Arquivo nao encontrado, pulando...")
            configs_falharam.append(batch_num)
            continue

        try:
            # Carregar config original
            original_config = load_yaml(input_file)

            # Fazer copia profunda para modificacao
            import copy
            modified_config = copy.deepcopy(original_config)

            # Processar config (modifica apenas diretorios)
            modified_config = process_config(modified_config, batch_num)

            # Validar consistencia
            issues = validate_consistency(original_config, modified_config)
            if issues:
                print(f"  AVISO: Diferencas inesperadas encontradas:")
                for issue in issues:
                    print(f"    - {issue}")
                validation_issues.extend([(batch_num, issue) for issue in issues])

            # Salvar novo config
            save_yaml(modified_config, output_file)
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
        print(f"  - config_balanced_batch_{batch}.yaml")

    if configs_falharam:
        print(f"\nConfigs que falharam: {len(configs_falharam)}")
        for batch in configs_falharam:
            print(f"  - batch_{batch}")

    if validation_issues:
        print(f"\nAVISO: {len(validation_issues)} diferencas inesperadas encontradas!")
        for batch, issue in validation_issues:
            print(f"  - Batch {batch}: {issue}")
    else:
        print("\nValidacao: OK - Apenas diretorios foram modificados")

    print("\n" + "=" * 70)
    print("ESTRUTURA DE DIRETORIOS DE RESULTADOS")
    print("=" * 70)
    print("""
Experimentos Anteriores (PENALTY_WEIGHT=0.0):
  - experiments_chunk2000_phase1/batch_1-4  (drift_simulation)
  - experiments_chunk2000_phase2/batch_5-7  (standard/real)

Experimentos Balanceados (PENALTY_WEIGHT=0.1):
  - experiments_balanced_phase1/batch_1-4   (drift_simulation)
  - experiments_balanced_phase2/batch_5-7   (standard/real)
""")

    print("=" * 70)
    print("COMANDOS PARA EXECUCAO NO COLAB")
    print("=" * 70)
    print("""
# Batch 1 (12 datasets abrupt drift)
!python main.py --config configs/config_balanced_batch_1.yaml

# Batch 2 (9 datasets gradual drift)
!python main.py --config configs/config_balanced_batch_2.yaml

# Batch 3 (8 datasets com ruido)
!python main.py --config configs/config_balanced_batch_3.yaml

# Batch 4 (6 datasets SINE/LED/WAVEFORM)
!python main.py --config configs/config_balanced_batch_4.yaml

# Batch 5 (5 datasets reais)
!python main.py --config configs/config_balanced_batch_5.yaml

# Batch 6 (6 sinteticos estacionarios)
!python main.py --config configs/config_balanced_batch_6.yaml

# Batch 7 (6 sinteticos estacionarios)
!python main.py --config configs/config_balanced_batch_7.yaml
""")


if __name__ == "__main__":
    main()
