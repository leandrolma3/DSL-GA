import os
import yaml

# Verificar arquivos principais
required_files = [
    'main.py',
    'ga.py',
    'data_handling_v8.py',
    'config_test_single.yaml',
    'hill_climbing_v2.py',
    'plotting.py'
]

print("Verificando arquivos...")
for file in required_files:
    exists = "OK" if os.path.exists(file) else "FALTANDO"
    print(f"{exists} {file}")

# Validar config
print("\nValidando config_test_single.yaml...")
with open('config_test_single.yaml') as f:
    config = yaml.safe_load(f)

print(f"  - Stream: {config['experiment_settings']['drift_simulation_experiments']}")
print(f"  - num_chunks: {config['data_params']['num_chunks']}")
print(f"  - population_size: {config['ga_params']['population_size']}")
print(f"  - num_runs: {config['experiment_settings']['num_runs']}")

# Validar concept_sequence do stream
stream = config['experimental_streams']['RBF_Abrupt_Severe']
seq = stream['concept_sequence']
total = sum(s['duration_chunks'] for s in seq)

# Criar string de sequência sem caracteres especiais
seq_parts = [f"{s['concept_id']}({s['duration_chunks']})" for s in seq]
seq_str = ' -> '.join(seq_parts)
print(f"  - concept_sequence: {seq_str}")
print(f"  - Total chunks: {total}")

assert total == 6, f"ERRO: Total de chunks é {total}, esperado 6!"
print("\nConfig validado com sucesso!")
