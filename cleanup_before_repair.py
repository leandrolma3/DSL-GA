#!/usr/bin/env python3
"""
cleanup_before_repair.py

Prepara os 4 datasets com arquivos EGIS incompletos para re-execucao.

FASE 1 (pre-execucao): Renomeia a pasta do dataset inteira.
  Ex: SEA_Gradual_Simple_Fast/ -> SEA_Gradual_Simple_Fast_temp/
  O EGIS criara a pasta do zero ao executar.

FASE 2 (pos-execucao): Copia baselines de volta da pasta _temp.
  Ex: SEA_Gradual_Simple_Fast_temp/run_1/baselines -> SEA_Gradual_Simple_Fast/run_1/

Executar LOCALMENTE (nao no Colab) pois os dados estao no Google Drive sincronizado.

Uso:
  python cleanup_before_repair.py prepare   # FASE 1: antes da re-execucao
  python cleanup_before_repair.py restore   # FASE 2: apos a re-execucao
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path

# ============================================================================
# CONFIGURACAO: Base path local (ajustar se necessario)
# ============================================================================
BASE_PATH = Path(r"C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid\experiments_unified")

# Os 4 datasets afetados e seus caminhos relativos dentro de experiments_unified
DATASETS = [
    {
        "name": "SEA_Gradual_Simple_Fast",
        "config_type": "NP",
        "batch_dir": "chunk_500/batch_1",
        "has_baselines": True,
    },
    {
        "name": "RBF_Gradual_Severe_Noise",
        "config_type": "NP",
        "batch_dir": "chunk_500/batch_2",
        "has_baselines": True,
    },
    {
        "name": "WAVEFORM_Abrupt_Simple",
        "config_type": "P",
        "batch_dir": "chunk_500_penalty/batch_2",
        "has_baselines": False,
    },
    {
        "name": "AGRAWAL_Stationary",
        "config_type": "P",
        "batch_dir": "chunk_500_penalty/batch_3",
        "has_baselines": False,
    },
]

# Arquivos/pastas de baselines a copiar de volta (apenas para NP datasets)
BASELINE_FILES = [
    "acdwm_results.csv",
    "erulesd2s_results.csv",
    "river_ARF_results.csv",
    "river_HAT_results.csv",
    "river_SRP_results.csv",
    "rose_chunk_eval_results.csv",
    "rose_original_results.csv",
    "desktop.ini",
]

BASELINE_DIRS = [
    "erulesd2s_arff",
    "erulesd2s_run",
    "rose_arff",
    "rose_chunk_eval_output",
    "rose_original_output",
]

# Diretorios de baselines no NIVEL DO DATASET (fora de run_1/)
DATASET_LEVEL_DIRS = [
    "cdcms_results",
]


def md5_file(filepath: Path) -> str:
    """Calcula MD5 de um arquivo."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_baselines(src_run1: Path, dst_run1: Path) -> list:
    """Copia arquivos e diretorios de baselines de src_run1 para dst_run1.
    Retorna lista de (item, info) para relatorio."""
    report = []

    for fname in BASELINE_FILES:
        src_file = src_run1 / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_run1 / fname)
            md5 = md5_file(dst_run1 / fname)
            report.append((fname, f"MD5={md5}"))

    for dname in BASELINE_DIRS:
        src_dir = src_run1 / dname
        if src_dir.exists():
            dst_dir = dst_run1 / dname
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            file_count = sum(1 for _ in dst_dir.rglob("*") if _.is_file())
            report.append((dname + "/", f"{file_count} files"))

    return report


# ============================================================================
# FASE 1: PREPARE (antes da re-execucao)
# ============================================================================
def phase_prepare():
    """Renomeia DATASET/ -> DATASET_temp/ para cada dataset afetado."""
    print("=" * 70)
    print("  FASE 1: PREPARE - Renomear pastas de datasets")
    print("=" * 70)
    print(f"\n  Base path: {BASE_PATH}")

    if not BASE_PATH.exists():
        print(f"\n  [ERRO FATAL] Base path nao encontrado: {BASE_PATH}")
        sys.exit(1)

    # Verificacao previa
    print(f"\n  Verificando {len(DATASETS)} datasets...")
    all_ok = True
    for ds in DATASETS:
        ds_dir = BASE_PATH / ds["batch_dir"] / ds["name"]
        temp_dir = BASE_PATH / ds["batch_dir"] / (ds["name"] + "_temp")
        exists = ds_dir.exists()
        temp_exists = temp_dir.exists()

        if temp_exists:
            print(f"    {ds['name']:40s} [ERRO: _temp ja existe!]")
            all_ok = False
        elif not exists:
            print(f"    {ds['name']:40s} [ERRO: pasta nao encontrada]")
            all_ok = False
        else:
            print(f"    {ds['name']:40s} [OK]")

    if not all_ok:
        print(f"\n  [ERRO] Problemas detectados. Abortando.")
        sys.exit(1)

    # Confirmacao
    print(f"\n  Operacao: DATASET/ -> DATASET_temp/")
    print(f"  O EGIS criara DATASET/run_1/ do zero na re-execucao.")
    resp = input("\n  Continuar? (s/N): ").strip().lower()
    if resp != "s":
        print("  Abortado pelo usuario.")
        sys.exit(0)

    # Executar renomeacoes
    results = []
    for ds in DATASETS:
        name = ds["name"]
        ds_dir = BASE_PATH / ds["batch_dir"] / name
        temp_dir = BASE_PATH / ds["batch_dir"] / (name + "_temp")

        print(f"\n  Renomeando {name}/ -> {name}_temp/ ...", end=" ")
        os.rename(ds_dir, temp_dir)
        print("OK")
        results.append((name, True))

    # Relatorio
    print(f"\n\n{'='*70}")
    print(f"  FASE 1 CONCLUIDA")
    print(f"{'='*70}")
    for name, ok in results:
        print(f"    {name:40s} -> {name}_temp/")
    print(f"\n  Proximo passo: executar os 4 YAMLs de reparo no Colab.")
    print(f"  Apos as 4 execucoes, rodar: python cleanup_before_repair.py restore")


# ============================================================================
# FASE 2: RESTORE (apos a re-execucao)
# ============================================================================
def phase_restore():
    """Copia baselines de DATASET_temp/run_1/ para DATASET/run_1/."""
    print("=" * 70)
    print("  FASE 2: RESTORE - Copiar baselines de volta")
    print("=" * 70)
    print(f"\n  Base path: {BASE_PATH}")

    if not BASE_PATH.exists():
        print(f"\n  [ERRO FATAL] Base path nao encontrado: {BASE_PATH}")
        sys.exit(1)

    # Verificacao previa
    print(f"\n  Verificando {len(DATASETS)} datasets...")
    all_ok = True
    for ds in DATASETS:
        name = ds["name"]
        ds_dir = BASE_PATH / ds["batch_dir"] / name
        temp_dir = BASE_PATH / ds["batch_dir"] / (name + "_temp")
        run1_new = ds_dir / "run_1"

        issues = []
        if not temp_dir.exists():
            issues.append("_temp nao existe")
        if not ds_dir.exists():
            issues.append("pasta EGIS nao criada")
        elif not run1_new.exists():
            issues.append("run_1/ nao criado pelo EGIS")

        if issues:
            print(f"    {name:40s} [ERRO: {', '.join(issues)}]")
            all_ok = False
        else:
            # Verificar se EGIS gerou chunk_metrics.json
            cm = run1_new / "chunk_metrics.json"
            cm_status = "chunk_metrics OK" if cm.exists() else "SEM chunk_metrics!"
            print(f"    {name:40s} [OK - {cm_status}]")

    if not all_ok:
        print(f"\n  [ERRO] Problemas detectados. Verifique se o EGIS completou.")
        print(f"  Corrija os problemas e re-execute este comando.")
        sys.exit(1)

    # Confirmacao
    print(f"\n  Operacao: copiar baselines de _temp/run_1/ -> DATASET/run_1/")
    resp = input("\n  Continuar? (s/N): ").strip().lower()
    if resp != "s":
        print("  Abortado pelo usuario.")
        sys.exit(0)

    # Copiar baselines
    for ds in DATASETS:
        name = ds["name"]
        temp_dir = BASE_PATH / ds["batch_dir"] / (name + "_temp")
        ds_dir = BASE_PATH / ds["batch_dir"] / name
        run1_new = ds_dir / "run_1"
        run1_temp = temp_dir / "run_1"

        print(f"\n{'='*70}")
        print(f"  Dataset: {name} ({ds['config_type']})")
        print(f"{'='*70}")

        # 1. Copiar baselines de run_1/ (river, rose, acdwm, erulesd2s)
        if ds["has_baselines"]:
            print(f"  [run_1] Copiando baselines de {name}_temp/run_1/ -> {name}/run_1/ ...")
            report = copy_baselines(run1_temp, run1_new)
            if report:
                for item, info in report:
                    print(f"    {item:40s} {info}")
            else:
                print(f"    (nenhum arquivo de baseline encontrado em _temp/run_1/)")
        else:
            print(f"  [run_1] Sem baselines (config penalty-only). Nada a copiar.")

        # 2. Copiar diretorios de baselines no nivel do dataset (cdcms_results/)
        print(f"  [dataset] Copiando dados de nivel dataset de {name}_temp/ -> {name}/ ...")
        ds_level_copied = 0
        for dname in DATASET_LEVEL_DIRS:
            src_dir = temp_dir / dname
            if src_dir.exists():
                dst_dir = ds_dir / dname
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
                file_count = sum(1 for _ in dst_dir.rglob("*") if _.is_file())
                print(f"    {dname + '/':40s} {file_count} files")
                ds_level_copied += 1
        if ds_level_copied == 0:
            print(f"    (nenhum diretorio de nivel dataset encontrado)")

    # Relatorio final
    print(f"\n\n{'='*70}")
    print(f"  FASE 2 CONCLUIDA")
    print(f"{'='*70}")
    print(f"  Baselines restaurados com sucesso.")
    print(f"\n  Pastas _temp preservadas para verificacao manual.")
    print(f"  Apos confirmar que tudo esta correto, delete manualmente:")
    for ds in DATASETS:
        temp_path = BASE_PATH / ds["batch_dir"] / (ds["name"] + "_temp")
        print(f"    {temp_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "restore"):
        print("Uso:")
        print("  python cleanup_before_repair.py prepare   # FASE 1: antes da re-execucao")
        print("  python cleanup_before_repair.py restore   # FASE 2: apos a re-execucao")
        sys.exit(1)

    if sys.argv[1] == "prepare":
        phase_prepare()
    elif sys.argv[1] == "restore":
        phase_restore()


if __name__ == "__main__":
    main()
