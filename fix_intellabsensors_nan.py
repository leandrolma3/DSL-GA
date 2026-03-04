#!/usr/bin/env python3
"""
fix_intellabsensors_nan.py

Remove instancias com NaN no target do dataset IntelLabSensors.
Afeta os chunks com problemas identificados: 6, 12, 15, 22

Analise previa identificou:
- Chunk 6: 2 NaN
- Chunk 12: 2 NaN
- Chunk 15: 1 NaN
- Chunk 22: 2 NaN
- Total: 7 instancias

IMPORTANTE: Executar APOS as sessoes Colab atuais terminarem!

Uso:
    python fix_intellabsensors_nan.py --dry-run    # Apenas mostra o que seria feito
    python fix_intellabsensors_nan.py              # Executa a correcao
    python fix_intellabsensors_nan.py --verify     # Apenas verifica estado atual
"""

import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime
import argparse
import sys

# Configuracao
UNIFIED_CHUNKS_BASE = Path("unified_chunks")
DATASET_NAME = "IntelLabSensors"
CHUNK_SIZES = [2000, 1000, 500]


def analyze_nan_status(chunk_dir: Path) -> dict:
    """Analisa o status de NaN em um diretorio de chunks."""
    results = {
        'total_nan': 0,
        'chunks_with_nan': [],
        'total_instances': 0,
        'total_chunks': 0
    }

    if not chunk_dir.exists():
        return results

    for csv_file in sorted(chunk_dir.glob("chunk_*.csv")):
        df = pd.read_csv(csv_file)
        nan_count = df['target'].isna().sum()
        results['total_instances'] += len(df)
        results['total_chunks'] += 1

        if nan_count > 0:
            results['total_nan'] += nan_count
            results['chunks_with_nan'].append({
                'file': csv_file.name,
                'nan_count': nan_count,
                'total_rows': len(df)
            })

    return results


def verify_all_sizes():
    """Verifica o estado atual de NaN em todos os tamanhos de chunk."""
    print("\n" + "="*70)
    print("VERIFICACAO DE STATUS - IntelLabSensors")
    print("="*70)

    for chunk_size in CHUNK_SIZES:
        chunk_dir = UNIFIED_CHUNKS_BASE / f"chunk_{chunk_size}" / DATASET_NAME

        print(f"\n--- chunk_{chunk_size}/{DATASET_NAME} ---")

        if not chunk_dir.exists():
            print(f"  [SKIP] Diretorio nao encontrado: {chunk_dir}")
            continue

        results = analyze_nan_status(chunk_dir)

        if results['total_nan'] == 0:
            print(f"  [OK] Nenhum NaN encontrado")
            print(f"       {results['total_chunks']} chunks, {results['total_instances']} instancias")
        else:
            print(f"  [WARN] {results['total_nan']} NaN encontrados em {len(results['chunks_with_nan'])} chunks:")
            for chunk_info in results['chunks_with_nan']:
                print(f"         - {chunk_info['file']}: {chunk_info['nan_count']} NaN de {chunk_info['total_rows']} linhas")


def dry_run():
    """Mostra o que seria feito sem executar."""
    print("\n" + "="*70)
    print("DRY RUN - Simulacao (nenhuma alteracao sera feita)")
    print("="*70)

    total_would_remove = 0

    for chunk_size in CHUNK_SIZES:
        chunk_dir = UNIFIED_CHUNKS_BASE / f"chunk_{chunk_size}" / DATASET_NAME

        print(f"\n--- chunk_{chunk_size}/{DATASET_NAME} ---")

        if not chunk_dir.exists():
            print(f"  [SKIP] Diretorio nao encontrado")
            continue

        results = analyze_nan_status(chunk_dir)

        if results['total_nan'] == 0:
            print(f"  [OK] Nenhuma acao necessaria")
        else:
            print(f"  [ACAO] Seriam removidas {results['total_nan']} instancias:")
            for chunk_info in results['chunks_with_nan']:
                new_size = chunk_info['total_rows'] - chunk_info['nan_count']
                print(f"         - {chunk_info['file']}: {chunk_info['total_rows']} -> {new_size} linhas")
            total_would_remove += results['total_nan']

    print(f"\n" + "="*70)
    print(f"RESUMO: {total_would_remove} instancias seriam removidas no total")
    print("="*70)


def backup_and_fix():
    """Cria backup e corrige os chunks com NaN."""
    print("\n" + "="*70)
    print("EXECUCAO DE CORRECAO - IntelLabSensors")
    print("="*70)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    total_removed = 0
    backups_created = []

    for chunk_size in CHUNK_SIZES:
        chunk_dir = UNIFIED_CHUNKS_BASE / f"chunk_{chunk_size}" / DATASET_NAME

        print(f"\n--- Processando chunk_{chunk_size}/{DATASET_NAME} ---")

        if not chunk_dir.exists():
            print(f"  [SKIP] Diretorio nao encontrado: {chunk_dir}")
            continue

        # Verificar se ha NaN antes de criar backup
        results = analyze_nan_status(chunk_dir)

        if results['total_nan'] == 0:
            print(f"  [OK] Nenhum NaN encontrado, pulando...")
            continue

        # Criar backup
        backup_dir = chunk_dir.parent / f"{DATASET_NAME}_backup_{timestamp}"
        try:
            shutil.copytree(chunk_dir, backup_dir)
            print(f"  [BACKUP] Criado: {backup_dir.name}")
            backups_created.append(backup_dir)
        except Exception as e:
            print(f"  [ERRO] Falha ao criar backup: {e}")
            print(f"  [ABORT] Abortando para evitar perda de dados")
            return False

        # Processar cada chunk com NaN
        size_removed = 0
        for csv_file in sorted(chunk_dir.glob("chunk_*.csv")):
            df = pd.read_csv(csv_file)
            original_len = len(df)

            # Remover linhas com NaN no target
            df_clean = df.dropna(subset=['target'])
            removed = original_len - len(df_clean)

            if removed > 0:
                df_clean.to_csv(csv_file, index=False)
                print(f"  [FIX] {csv_file.name}: {removed} NaN removidos ({original_len} -> {len(df_clean)})")
                size_removed += removed

        total_removed += size_removed
        print(f"  [DONE] {size_removed} instancias removidas neste tamanho")

    print(f"\n" + "="*70)
    print(f"CORRECAO CONCLUIDA")
    print(f"="*70)
    print(f"Total removido: {total_removed} instancias")
    print(f"Backups criados: {len(backups_created)}")
    for backup in backups_created:
        print(f"  - {backup}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Corrige NaN no dataset IntelLabSensors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
    python fix_intellabsensors_nan.py --verify     # Apenas verifica
    python fix_intellabsensors_nan.py --dry-run    # Simula sem alterar
    python fix_intellabsensors_nan.py              # Executa correcao
        """
    )

    parser.add_argument('--dry-run', action='store_true',
                        help='Simula a correcao sem fazer alteracoes')
    parser.add_argument('--verify', action='store_true',
                        help='Apenas verifica o estado atual')
    parser.add_argument('--no-confirm', action='store_true',
                        help='Pula confirmacao (para automacao)')

    args = parser.parse_args()

    print("="*70)
    print("FIX: IntelLabSensors - Remocao de NaN no Target")
    print("="*70)
    print(f"Diretorio base: {UNIFIED_CHUNKS_BASE.absolute()}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Tamanhos de chunk: {CHUNK_SIZES}")

    if args.verify:
        verify_all_sizes()
        return 0

    if args.dry_run:
        dry_run()
        return 0

    # Modo de execucao real
    print("\n" + "!"*70)
    print("ATENCAO: Este script modifica os dados em unified_chunks/")
    print("Um backup sera criado automaticamente antes de qualquer alteracao.")
    print("!"*70)

    # Mostrar o que sera feito
    dry_run()

    if not args.no_confirm:
        print()
        response = input("Deseja prosseguir com a correcao? (s/N): ")
        if response.lower() != 's':
            print("[ABORT] Operacao cancelada pelo usuario.")
            return 1

    # Executar correcao
    success = backup_and_fix()

    if success:
        # Verificar resultado
        print("\n--- Verificacao pos-correcao ---")
        verify_all_sizes()
        print("\n[SUCCESS] Correcao concluida com sucesso!")
        return 0
    else:
        print("\n[FAILED] Correcao falhou. Verifique os backups.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
