#!/usr/bin/env python3
"""
Script para corrigir caminhos base_results_dir em todos os configs de batch.

Substitui:
  /content/drive/MyDrive/DSL-AG-hybrid/...
Por:
  /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/...
"""

import os
import glob

def fix_config_paths(configs_dir='configs'):
    """Corrige caminhos em todos os arquivos config_batch_*.yaml"""

    # Caminhos
    old_path = '/content/drive/MyDrive/DSL-AG-hybrid/'
    new_path = '/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/'

    # Encontrar todos os configs
    config_files = glob.glob(os.path.join(configs_dir, 'config_batch_*.yaml'))

    if not config_files:
        print(f"Nenhum arquivo config_batch_*.yaml encontrado em {configs_dir}/")
        return

    print(f"Encontrados {len(config_files)} arquivos de configuracao\n")

    # Processar cada config
    for config_file in sorted(config_files):
        print(f"Processando: {config_file}")

        # Ler arquivo
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Verificar se precisa correcao
        if old_path not in content:
            print(f"  -> Ja esta correto ou nao contem o caminho antigo")
            continue

        # Substituir caminho
        new_content = content.replace(old_path, new_path)

        # Contar substituicoes
        count = content.count(old_path)

        # Salvar arquivo corrigido
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"  -> Corrigido! ({count} substituicoes)")

    print("\n" + "="*70)
    print("Correcao concluida!")
    print("="*70)

def verify_config_paths(configs_dir='configs'):
    """Verifica os caminhos base_results_dir em todos os configs"""

    config_files = glob.glob(os.path.join(configs_dir, 'config_batch_*.yaml'))

    print("\n" + "="*70)
    print("VERIFICACAO DE CAMINHOS")
    print("="*70 + "\n")

    for config_file in sorted(config_files):
        batch_num = os.path.basename(config_file).replace('config_batch_', '').replace('.yaml', '')

        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                if 'base_results_dir:' in line:
                    path = line.split('base_results_dir:')[1].strip()
                    print(f"Batch {batch_num:2s}: {path}")
                    break

if __name__ == '__main__':
    print("="*70)
    print("CORRECAO DE CAMINHOS NOS CONFIGS")
    print("="*70 + "\n")

    # Corrigir caminhos
    fix_config_paths()

    # Verificar resultado
    verify_config_paths()

    print("\nPRONTO! Todos os configs foram corrigidos.")
    print("Voce pode verificar manualmente com:")
    print("  grep 'base_results_dir' configs/*.yaml")
