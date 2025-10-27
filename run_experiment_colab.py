"""
Wrapper para Executar Experimentos no Google Colab
===================================================

Este script é um wrapper que:
1. Configura o ambiente do Colab (Drive, logging, etc.)
2. Executa o experimento principal (main.py)
3. Salva TUDO automaticamente no Google Drive
4. Garante que nada seja perdido se a sessão cair

Uso no Colab:
-------------
# Célula 1: Instalar dependências
!pip install -q river scikit-learn pyyaml matplotlib seaborn pandas numpy

# Célula 2: Clonar repositório e executar
!git clone https://github.com/seu-repo/DSL-AG-hybrid.git
%cd DSL-AG-hybrid

# Célula 3: Executar experimento
!python run_experiment_colab.py --stream RBF_Abrupt_Severe --chunks 3

Autor: Claude Code
Data: 2025-10-18
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Importa o setup do Colab
from colab_drive_setup import (
    setup_colab_environment,
    save_checkpoint,
    periodic_sync_to_drive
)


def parse_arguments():
    """
    Parseia argumentos da linha de comando.
    """
    parser = argparse.ArgumentParser(
        description='Executa experimentos DSL-AG-hybrid no Google Colab',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Argumentos do experimento
    parser.add_argument(
        '--stream',
        type=str,
        default='RBF_Abrupt_Severe',
        help='Nome do stream a ser executado (deve estar em config.yaml)'
    )

    parser.add_argument(
        '--chunks',
        type=int,
        default=3,
        help='Número de chunks a processar'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=6000,
        help='Tamanho de cada chunk'
    )

    parser.add_argument(
        '--population',
        type=int,
        default=120,
        help='Tamanho da população do GA'
    )

    parser.add_argument(
        '--max-generations',
        type=int,
        default=200,
        help='Número máximo de gerações'
    )

    # Argumentos do Colab/Drive
    parser.add_argument(
        '--project-name',
        type=str,
        default='DSL-AG-hybrid',
        help='Nome do projeto (pasta no Drive)'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Nome do experimento (se None, usa stream name)'
    )

    parser.add_argument(
        '--no-backup-code',
        action='store_true',
        help='Não fazer backup do código para o Drive'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Nível de logging'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Caminho para arquivo de configuração'
    )

    return parser.parse_args()


def update_config_for_colab(config_file, args, drive_results_dir):
    """
    Atualiza o config.yaml com parâmetros do Colab e diretório do Drive.

    Args:
        config_file (str): Caminho para config.yaml
        args: Argumentos parseados
        drive_results_dir (str): Diretório de resultados no Drive
    """
    import yaml

    try:
        # Carrega config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Atualiza parâmetros
        if 'data_params' not in config:
            config['data_params'] = {}

        config['data_params']['chunk_size'] = args.chunk_size
        config['data_params']['num_chunks'] = args.chunks

        if 'ga_params' not in config:
            config['ga_params'] = {}

        config['ga_params']['population_size'] = args.population
        config['ga_params']['max_generations'] = args.max_generations

        # Define stream a executar
        if 'experiment_settings' not in config:
            config['experiment_settings'] = {}

        config['experiment_settings']['standard_experiments'] = [args.stream]
        config['experiment_settings']['num_runs'] = 1
        config['experiment_settings']['logging_level'] = args.log_level

        # IMPORTANTE: Direciona resultados para o Drive
        config['experiment_settings']['base_results_dir'] = drive_results_dir

        # Salva config atualizado
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logging.info(f"✅ Configuração atualizada:")
        logging.info(f"   - Stream: {args.stream}")
        logging.info(f"   - Chunks: {args.chunks} (tamanho {args.chunk_size})")
        logging.info(f"   - População: {args.population}")
        logging.info(f"   - Max Gerações: {args.max_generations}")
        logging.info(f"   - Resultados: {drive_results_dir}")

    except Exception as e:
        logging.error(f"❌ Erro ao atualizar config: {e}")
        raise


def run_main_experiment():
    """
    Executa o experimento principal (main.py).

    Importa e executa a função main() de main.py,
    capturando quaisquer erros.
    """
    try:
        logging.info("="*70)
        logging.info("🚀 INICIANDO EXPERIMENTO PRINCIPAL")
        logging.info("="*70)

        # Importa e executa main
        import main

        # main.py usa sys.argv, então não precisamos passar argumentos
        # Ele vai ler do config.yaml que já atualizamos

        logging.info("✅ Experimento concluído com sucesso!")

    except KeyboardInterrupt:
        logging.warning("⚠️  Experimento interrompido pelo usuário")
        raise

    except Exception as e:
        logging.error(f"❌ ERRO durante execução do experimento: {e}", exc_info=True)
        raise


def create_summary_report(all_paths, args):
    """
    Cria um relatório resumido do experimento.

    Args:
        all_paths (dict): Dicionário com todos os caminhos
        args: Argumentos do experimento
    """
    summary_file = os.path.join(all_paths['experiment'], 'EXPERIMENT_SUMMARY.txt')

    try:
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RELATÓRIO DO EXPERIMENTO\n")
            f.write("="*70 + "\n\n")

            f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Projeto: {args.project_name}\n")
            f.write(f"Experimento: {args.experiment_name}\n\n")

            f.write("PARÂMETROS:\n")
            f.write(f"  - Stream: {args.stream}\n")
            f.write(f"  - Chunks: {args.chunks}\n")
            f.write(f"  - Chunk Size: {args.chunk_size}\n")
            f.write(f"  - População: {args.population}\n")
            f.write(f"  - Max Gerações: {args.max_generations}\n\n")

            f.write("DIRETÓRIOS:\n")
            for key, path in all_paths.items():
                f.write(f"  - {key:15s}: {path}\n")

            f.write("\n" + "="*70 + "\n")
            f.write("INSTRUÇÕES DE ACESSO:\n")
            f.write("="*70 + "\n")
            f.write("1. Abra o Google Drive\n")
            f.write(f"2. Navegue para: MyDrive/{args.project_name}/experiments/\n")
            f.write(f"3. Encontre a pasta: {os.path.basename(all_paths['experiment'])}\n")
            f.write("4. Arquivos principais:\n")
            f.write("   - logs/: Logs completos da execução\n")
            f.write("   - results/: Resultados, gráficos, CSVs\n")
            f.write("   - checkpoints/: Checkpoints salvos durante execução\n")
            f.write("\n")

        logging.info(f"📄 Relatório salvo: {summary_file}")

    except Exception as e:
        logging.warning(f"⚠️  Erro ao criar relatório: {e}")


def main():
    """
    Função principal do wrapper.
    """
    # Parseia argumentos
    args = parse_arguments()

    # Define nome do experimento se não fornecido
    if args.experiment_name is None:
        args.experiment_name = f"{args.stream}_P{args.population}_G{args.max_generations}"

    print("\n" + "="*70)
    print("🚀 WRAPPER PARA GOOGLE COLAB - DSL-AG-HYBRID")
    print("="*70)
    print(f"Experimento: {args.experiment_name}")
    print(f"Stream: {args.stream}")
    print(f"Chunks: {args.chunks} × {args.chunk_size} instâncias")
    print("="*70 + "\n")

    try:
        # 1. Setup do ambiente Colab + Drive
        drive_results_dir, log_file, all_paths = setup_colab_environment(
            project_name=args.project_name,
            experiment_name=args.experiment_name,
            config_file=args.config,
            backup_code=not args.no_backup_code,
            log_level=getattr(logging, args.log_level)
        )

        # 2. Atualiza config.yaml com parâmetros do Colab
        update_config_for_colab(args.config, args, drive_results_dir)

        # 3. Executa experimento principal
        run_main_experiment()

        # 4. Cria relatório resumido
        create_summary_report(all_paths, args)

        # 5. Mensagem final
        print("\n" + "="*70)
        print("✅ EXPERIMENTO CONCLUÍDO COM SUCESSO!")
        print("="*70)
        print(f"📂 Resultados salvos no Google Drive:")
        print(f"   {all_paths['experiment']}")
        print("\n💡 Para acessar:")
        print(f"   1. Abra seu Google Drive")
        print(f"   2. Navegue para: MyDrive/{args.project_name}/experiments/")
        print(f"   3. Abra a pasta: {os.path.basename(all_paths['experiment'])}")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n⚠️  Experimento interrompido pelo usuário")
        print("   Todos os dados até o momento foram salvos no Drive.")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ ERRO FATAL: {e}")
        print("   Verifique os logs no Google Drive para mais detalhes.")
        logging.error("Experimento falhou", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
