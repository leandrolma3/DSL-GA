#!/usr/bin/env python3
"""
Script de Execucao do Experimento Expandido - 11 Datasets

Executa experimento completo com 11 datasets de drift simulation.

Configuracao:
- 11 datasets (5 geradores diferentes)
- 5 modelos Python: GBML, ACDWM, HAT, ARF, SRP
- chunk_size=3000 (validado localmente)
- Tempo estimado: 11 horas
"""

import subprocess
import sys
import time
from datetime import datetime, timedelta
import os

def print_header(message):
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80 + "\n")

def print_section(message):
    print("\n" + "-" * 80)
    print(message)
    print("-" * 80 + "\n")

def format_duration(seconds):
    """Formata duracao em horas, minutos, segundos"""
    td = timedelta(seconds=int(seconds))
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60

    if td.days > 0:
        return f"{td.days}d {hours}h {minutes}min {secs}s"
    elif hours > 0:
        return f"{hours}h {minutes}min {secs}s"
    elif minutes > 0:
        return f"{minutes}min {secs}s"
    else:
        return f"{secs}s"

def main():
    print_header("EXPERIMENTO EXPANDIDO - 11 DATASETS")

    print(f"Data/hora inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuracao:")
    print("  Datasets:   11 (RBF, SEA, AGRAWAL, HYPERPLANE, STAGGER, SINE)")
    print("  Chunks:     4 por dataset = 44 chunks totais")
    print("  Chunk size: 3000 instancias")
    print("  Modelos:    GBML, ACDWM, HAT, ARF, SRP (5 modelos)")
    print()
    print("Tempo estimado: 11 horas")
    print("  - GBML: ~15 min/chunk x 44 chunks = 660 min (11h)")
    print("  - ACDWM: ~1 min/chunk x 44 chunks = 44 min")
    print("  - River (HAT, ARF, SRP): ~1 min/chunk x 44 x 3 = 132 min")
    print("  - Total estimado: 11 horas")
    print()
    print("Margem de seguranca no Colab Pro (24h): 13 horas (54%)")
    print()

    # Verificar se arquivo de config existe
    config_file = 'config_experiment_expanded.yaml'
    if not os.path.exists(config_file):
        print(f"[ERRO] Arquivo de configuracao nao encontrado: {config_file}")
        print()
        print("Certifique-se de que o arquivo config_experiment_expanded.yaml esta no diretorio atual.")
        return 1

    print(f"Configuracao encontrada: {config_file}")
    print()

    # Confirmar execucao (pode ser comentado no Colab)
    # input("Pressione ENTER para iniciar o experimento...")

    print("Iniciando experimento automaticamente...")
    print()

    # Comando de execucao
    cmd = [
        sys.executable,
        'compare_gbml_vs_river.py',
        '--config', config_file,
        '--models', 'HAT', 'ARF', 'SRP',
        '--acdwm',
        '--seed', '42',
        '--output', 'experiment_expanded_results'
    ]

    print_section("EXECUTANDO EXPERIMENTO")
    print("Comando:", " ".join(cmd))
    print()

    # Marca tempo inicial
    start_time = time.time()
    start_datetime = datetime.now()

    try:
        # Executa comando
        result = subprocess.run(
            cmd,
            capture_output=False,  # Mostra output em tempo real
            text=True,
            check=True
        )

        # Marca tempo final
        end_time = time.time()
        end_datetime = datetime.now()
        duration = end_time - start_time

        print_section("EXPERIMENTO CONCLUIDO COM SUCESSO")
        print(f"Inicio:   {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Fim:      {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duracao:  {format_duration(duration)} ({duration/3600:.1f} horas)")
        print()

        # Calcula tempo por chunk
        num_chunks = 44
        time_per_chunk = duration / num_chunks
        print(f"Tempo por chunk: {format_duration(time_per_chunk)} ({time_per_chunk/60:.1f} min)")
        print()

        # Compara com estimativa
        estimated_time = 15 * 60 * num_chunks  # 15 min/chunk em segundos

        print("Comparacao com estimativa:")
        print(f"  Estimado: {format_duration(estimated_time)} ({estimated_time/3600:.1f} horas)")
        print(f"  Obtido:   {format_duration(duration)} ({duration/3600:.1f} horas)")

        if duration < estimated_time:
            diff_pct = ((estimated_time - duration) / estimated_time) * 100
            print(f"  Resultado: MAIS RAPIDO que o estimado (-{diff_pct:.1f}%)")
        else:
            diff_pct = ((duration - estimated_time) / estimated_time) * 100
            print(f"  Resultado: MAIS LENTO que o estimado (+{diff_pct:.1f}%)")

        print()

        # Verifica se resultados foram gerados
        print_section("VERIFICACAO DE RESULTADOS")

        import glob

        result_dir = 'experiment_expanded_results'

        if os.path.exists(result_dir):
            # Conta datasets processados
            dataset_dirs = [d for d in os.listdir(result_dir)
                           if os.path.isdir(os.path.join(result_dir, d))]

            print(f"[OK] Diretorio de resultados encontrado: {result_dir}")
            print(f"[OK] {len(dataset_dirs)} dataset(s) processado(s)")
            print()

            # Lista resultados por dataset
            for dataset_dir in sorted(dataset_dirs):
                dataset_path = os.path.join(result_dir, dataset_dir)
                csv_files = glob.glob(f'{dataset_path}/*.csv')
                pkl_files = glob.glob(f'{dataset_path}/*.pkl')
                png_files = glob.glob(f'{dataset_path}/*.png')

                print(f"  {dataset_dir}:")
                print(f"    CSVs: {len(csv_files)}, PKLs: {len(pkl_files)}, PNGs: {len(png_files)}")

            print()

            # Verifica arquivo consolidado
            comparison_files = glob.glob(f'{result_dir}/**/comparison_table.csv', recursive=True)

            if len(comparison_files) == 11:
                print(f"[OK] Todos os 11 datasets geraram comparison_table.csv")
                print()
                print("Validacao: SUCESSO - Todos os resultados gerados corretamente")
            else:
                print(f"[AVISO] Apenas {len(comparison_files)}/11 datasets geraram comparison_table.csv")
                print("Validacao: PARCIAL - Alguns datasets podem ter falhado")

        else:
            print(f"[ERRO] Diretorio de resultados nao encontrado: {result_dir}")
            print("Validacao: FALHA - Resultados nao foram salvos")

        print()

        # Proximos passos
        print_section("PROXIMOS PASSOS")
        print("1. Fazer backup dos resultados:")
        print("   cp -r experiment_expanded_results /content/drive/MyDrive/")
        print()
        print("2. Analisar resultados:")
        print("   python analyze_experiment_expanded.py")
        print()
        print("3. Gerar relatorio estatistico:")
        print("   python statistical_analysis_expanded.py")
        print()

        return 0

    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time

        print_section("EXPERIMENTO FALHOU")
        print(f"Return code: {e.returncode}")
        print(f"Duracao ate falha: {format_duration(duration)}")
        print()
        print("Verifique os logs acima para identificar o problema.")
        print()

        # Verifica resultados parciais
        print("Verificando resultados parciais...")
        result_dir = 'experiment_expanded_results'
        if os.path.exists(result_dir):
            dataset_dirs = [d for d in os.listdir(result_dir)
                           if os.path.isdir(os.path.join(result_dir, d))]
            print(f"Datasets completados ate a falha: {len(dataset_dirs)}")
            for d in sorted(dataset_dirs):
                print(f"  - {d}")

        print()
        return 1

    except KeyboardInterrupt:
        end_time = time.time()
        duration = end_time - start_time

        print()
        print_section("EXPERIMENTO INTERROMPIDO PELO USUARIO")
        print(f"Duracao ate interrupcao: {format_duration(duration)}")

        # Verifica resultados parciais
        print()
        print("Verificando resultados parciais...")
        result_dir = 'experiment_expanded_results'
        if os.path.exists(result_dir):
            dataset_dirs = [d for d in os.listdir(result_dir)
                           if os.path.isdir(os.path.join(result_dir, d))]
            print(f"Datasets completados ate interrupcao: {len(dataset_dirs)}")
            for d in sorted(dataset_dirs):
                print(f"  - {d}")

        print()
        return 130

    except Exception as e:
        print_section("ERRO INESPERADO")
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensagem: {e}")

        import traceback
        traceback.print_exc()

        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
