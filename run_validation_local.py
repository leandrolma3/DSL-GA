#!/usr/bin/env python3
"""
Script de Validacao Local

Executa teste rapido com chunk_size=3000 para validar estimativas de tempo.

Testa:
- 1 dataset (RBF_Abrupt_Severe)
- 2 chunks (3000 instancias cada)
- 5 modelos (GBML, ACDWM, HAT, ARF, SRP)

Tempo esperado: 70-80 minutos (2 chunks × 35-40 min)
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

    if hours > 0:
        return f"{hours}h {minutes}min {secs}s"
    elif minutes > 0:
        return f"{minutes}min {secs}s"
    else:
        return f"{secs}s"

def main():
    print_header("VALIDACAO LOCAL - chunk_size=3000")

    print(f"Data/hora inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuracao:")
    print("  Dataset:    RBF_Abrupt_Severe")
    print("  Chunks:     3 (2 avaliacoes)")
    print("  Chunk size: 3000 instancias")
    print("  Modelos:    GBML, ACDWM, HAT, ARF, SRP (5 modelos)")
    print()
    print("Tempo estimado: 70-80 minutos")
    print("  - GBML: ~35 min/chunk × 3 chunks = 105 min")
    print("  - ACDWM: ~2 min/chunk × 3 chunks = 6 min")
    print("  - River (HAT, ARF, SRP): ~1 min/chunk × 3 × 3 = 9 min")
    print("  - Total estimado: 120 min (2 horas)")
    print()
    print("Iniciando teste automaticamente...")
    print()

    # Comando de execucao
    cmd = [
        sys.executable,
        'compare_gbml_vs_river.py',
        '--stream', 'RBF_Abrupt_Severe',
        '--config', 'config_validation_local.yaml',
        '--models', 'HAT', 'ARF', 'SRP',
        '--chunks', '3',
        '--chunk-size', '3000',
        '--acdwm',
        '--seed', '42',
        '--output', 'validation_local_results'
    ]

    print_section("EXECUTANDO TESTE")
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

        print_section("TESTE CONCLUIDO COM SUCESSO")
        print(f"Inicio:   {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Fim:      {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duracao:  {format_duration(duration)} ({duration/60:.1f} minutos)")
        print()

        # Calcula tempo por chunk
        num_chunks = 3
        time_per_chunk = duration / num_chunks
        print(f"Tempo por chunk: {format_duration(time_per_chunk)} ({time_per_chunk/60:.1f} min)")
        print()

        # Compara com estimativa
        estimated_time = 40 * 60  # 40 min/chunk em segundos
        estimated_total = estimated_time * num_chunks

        print("Comparacao com estimativa:")
        print(f"  Estimado: {format_duration(estimated_total)} ({estimated_total/60:.1f} min)")
        print(f"  Obtido:   {format_duration(duration)} ({duration/60:.1f} min)")

        if duration < estimated_total:
            diff_pct = ((estimated_total - duration) / estimated_total) * 100
            print(f"  Resultado: MAIS RAPIDO que o estimado (-{diff_pct:.1f}%)")
        else:
            diff_pct = ((duration - estimated_total) / estimated_total) * 100
            print(f"  Resultado: MAIS LENTO que o estimado (+{diff_pct:.1f}%)")

        print()

        # Projeta tempo para experimento completo (6 datasets, 24 chunks)
        print_section("PROJECAO PARA EXPERIMENTO COMPLETO")

        total_chunks_full = 24
        projected_time = time_per_chunk * total_chunks_full
        projected_hours = projected_time / 3600

        print(f"Configuracao planejada:")
        print(f"  Datasets: 6")
        print(f"  Chunks totais: {total_chunks_full}")
        print(f"  Tempo por chunk (medido): {format_duration(time_per_chunk)}")
        print()
        print(f"Tempo total projetado: {format_duration(projected_time)} ({projected_hours:.1f} horas)")
        print()

        if projected_hours < 24:
            margin = 24 - projected_hours
            print(f"VIAVEL no Colab Pro (24h):")
            print(f"  Margem de seguranca: {margin:.1f} horas ({margin/24*100:.1f}%)")
            print()
            print("Recomendacao: PROSSEGUIR com experimento completo")
        else:
            excess = projected_hours - 24
            print(f"NAO VIAVEL no Colab Pro (24h):")
            print(f"  Excesso: {excess:.1f} horas")
            print()
            print("Recomendacao: REDUZIR numero de datasets ou chunk_size")

        print()

        # Verifica se resultados foram gerados
        print_section("VERIFICACAO DE RESULTADOS")

        import os
        import glob

        result_dir = 'validation_local_results'

        if os.path.exists(result_dir):
            result_files = glob.glob(f'{result_dir}/**/*.csv', recursive=True)

            if result_files:
                print(f"[OK] {len(result_files)} arquivo(s) CSV gerado(s)")
                for f in result_files:
                    size_kb = os.path.getsize(f) / 1024
                    print(f"  - {f} ({size_kb:.1f} KB)")
                print()
                print("Validacao: SUCESSO - Resultados gerados corretamente")
            else:
                print("[AVISO] Nenhum arquivo CSV encontrado")
                print("Validacao: PARCIAL - Verifique logs para erros")
        else:
            print(f"[ERRO] Diretorio de resultados nao encontrado: {result_dir}")
            print("Validacao: FALHA - Resultados nao foram salvos")

        print()

        return 0

    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time

        print_section("TESTE FALHOU")
        print(f"Return code: {e.returncode}")
        print(f"Duracao ate falha: {format_duration(duration)}")
        print()
        print("Verifique os logs acima para identificar o problema.")
        return 1

    except KeyboardInterrupt:
        end_time = time.time()
        duration = end_time - start_time

        print()
        print_section("TESTE INTERROMPIDO PELO USUARIO")
        print(f"Duracao ate interrupcao: {format_duration(duration)}")
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
