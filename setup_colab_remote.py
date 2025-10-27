#!/usr/bin/env python3
"""
Setup Remoto para Google Colab (via SSH)
=========================================

Este script configura o ambiente Colab quando acessado via SSH:
1. Monta Google Drive (se ainda não montado)
2. Cria estrutura de diretórios
3. Configura logging para Drive
4. Prepara ambiente para execução

IMPORTANTE: Este script só funciona dentro do Google Colab!

Uso via SSH:
-----------
ssh your-host.trycloudflare.com
cd ~/DSL-AG-hybrid
python setup_colab_remote.py

Depois:
-------
python compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 3

Autor: Claude Code
Data: 2025-10-18
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path


def check_if_colab():
    """Verifica se está rodando no Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def mount_google_drive_silent():
    """
    Monta Google Drive silenciosamente.

    Nota: Se Drive já estiver montado (via notebook), apenas verifica.
    Se não estiver montado, tenta montar (mas pode falhar via SSH puro).
    """
    drive_mount = "/content/drive"

    # Verifica se já está montado
    if os.path.ismount(drive_mount):
        print(f"✅ Google Drive já está montado em: {drive_mount}")
        return drive_mount

    # Tenta montar (funciona se foi pré-autorizado no notebook)
    try:
        from google.colab import drive
        print("🔄 Tentando montar Google Drive...")
        print("   (Isso só funciona se já foi autorizado no notebook)")

        # Tenta montar sem forçar autorização
        drive.mount(drive_mount, force_remount=False)

        if os.path.ismount(drive_mount):
            print(f"✅ Google Drive montado com sucesso!")
            return drive_mount
        else:
            print("⚠️  Drive não montou, mas isso não é crítico")
            print("   Você pode montar manualmente no notebook Colab")
            return None

    except Exception as e:
        print(f"⚠️  Não foi possível montar Drive via SSH: {e}")
        print("   SOLUÇÃO: Execute no notebook Colab:")
        print("   ```")
        print("   from google.colab import drive")
        print("   drive.mount('/content/drive')")
        print("   ```")
        return None


def create_drive_structure(drive_mount, project_name="DSL-AG-hybrid"):
    """
    Cria estrutura de diretórios no Drive.

    Args:
        drive_mount (str): Path do Drive montado
        project_name (str): Nome do projeto

    Returns:
        dict: Dicionário com paths criados
    """
    if not drive_mount:
        print("❌ Drive não está montado, não é possível criar estrutura")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.join(drive_mount, "MyDrive", project_name)
    experiment_dir = os.path.join(base_dir, "experiments", f"ssh_session_{timestamp}")

    paths = {
        'base': base_dir,
        'experiments': os.path.join(base_dir, "experiments"),
        'current_experiment': experiment_dir,
        'logs': os.path.join(experiment_dir, "logs"),
        'results': os.path.join(experiment_dir, "results"),
        'checkpoints': os.path.join(experiment_dir, "checkpoints"),
        'plots': os.path.join(experiment_dir, "plots"),
    }

    print("\n📁 Criando estrutura no Drive...")
    for name, path in paths.items():
        try:
            os.makedirs(path, exist_ok=True)
            print(f"   ✅ {name:20s}: {path}")
        except Exception as e:
            print(f"   ❌ {name:20s}: ERRO - {e}")

    return paths


def create_env_file(paths):
    """
    Cria arquivo .env com paths do Drive.

    Args:
        paths (dict): Dicionário com paths
    """
    if not paths:
        return

    env_file = "/root/DSL-AG-hybrid/.drive_paths.sh"

    try:
        with open(env_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Paths do Google Drive (gerado automaticamente)\n\n")

            for key, path in paths.items():
                env_var = f"DRIVE_{key.upper()}"
                f.write(f'export {env_var}="{path}"\n')

        os.chmod(env_file, 0o755)
        print(f"\n📄 Arquivo de ambiente criado: {env_file}")
        print(f"   Use: source {env_file}")

    except Exception as e:
        print(f"⚠️  Erro ao criar .env: {e}")


def update_bashrc(paths):
    """
    Atualiza .bashrc com aliases úteis.

    Args:
        paths (dict): Dicionário com paths
    """
    if not paths:
        return

    bashrc = os.path.expanduser("~/.bashrc")

    marker = "# === DSL-AG-HYBRID AUTO-CONFIG ==="

    # Verifica se já foi configurado
    try:
        with open(bashrc, 'r') as f:
            if marker in f.read():
                print("\n⚠️  .bashrc já configurado (pulando)")
                return
    except FileNotFoundError:
        pass

    # Adiciona configuração
    config = f"""

{marker}
# Configuração automática do DSL-AG-hybrid
export DRIVE_LOGS="{paths['logs']}"
export DRIVE_RESULTS="{paths['results']}"
export DRIVE_CHECKPOINTS="{paths['checkpoints']}"

alias drive-logs='cd "$DRIVE_LOGS"'
alias drive-results='cd "$DRIVE_RESULTS"'
alias drive-tail='tail -f "$DRIVE_LOGS"/*.log 2>/dev/null || echo "Nenhum log encontrado"'

# Função para executar com log no Drive
run-experiment() {{
    if [ -z "$1" ]; then
        echo "Uso: run-experiment <comando>"
        return 1
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$DRIVE_LOGS/experiment_$TIMESTAMP.log"

    # Garante que o diretório existe antes de usar tee
    mkdir -p "$DRIVE_LOGS" 2>/dev/null || true

    echo "📝 Log será salvo em: $LOG_FILE"
    echo ""

    "$@" 2>&1 | tee "$LOG_FILE"

    echo ""
    echo "✅ Log completo: $LOG_FILE"
}}

echo "✅ DSL-AG-hybrid environment loaded"
echo "   DRIVE_LOGS: $DRIVE_LOGS"
# {marker}
"""

    try:
        with open(bashrc, 'a') as f:
            f.write(config)
        print("\n✅ .bashrc atualizado com aliases")
        print("   Execute: source ~/.bashrc")
        print("   Ou reconecte via SSH")
    except Exception as e:
        print(f"⚠️  Erro ao atualizar .bashrc: {e}")


def create_config_override(paths):
    """
    Cria arquivo de override para config.yaml que redireciona resultados para Drive.

    Args:
        paths (dict): Dicionário com paths
    """
    if not paths:
        return None

    override_file = "/root/DSL-AG-hybrid/config_drive.yaml"

    try:
        import yaml

        # Configuração mínima que será merged com config.yaml
        override = {
            'experiment_settings': {
                'base_results_dir': paths['results'],
                'save_checkpoints': True,
                'checkpoint_dir': paths['checkpoints']
            }
        }

        with open(override_file, 'w') as f:
            yaml.dump(override, f, default_flow_style=False)

        print(f"\n✅ Config override criado: {override_file}")
        print(f"   Resultados serão salvos em: {paths['results']}")

        return override_file

    except Exception as e:
        print(f"⚠️  Erro ao criar config override: {e}")
        return None


def create_run_script(paths):
    """
    Cria script wrapper para executar experimentos com logging automático.

    Args:
        paths (dict): Dicionário com paths
    """
    if not paths:
        return

    script_path = "/root/run_experiment.sh"

    content = f"""#!/bin/bash
################################################################################
# Script Wrapper para Executar Experimentos
################################################################################
# Gerado automaticamente em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#
# Uso:
#   ./run_experiment.sh python compare_gbml_vs_river.py --stream RBF --chunks 3
################################################################################

DRIVE_LOGS="{paths['logs']}"
DRIVE_RESULTS="{paths['results']}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$DRIVE_LOGS/experiment_$TIMESTAMP.log"

# Garante que o diretório existe antes de usar tee
mkdir -p "$DRIVE_LOGS" || {{
    echo "❌ ERRO: Não foi possível criar diretório de logs: $DRIVE_LOGS"
    exit 1
}}

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         EXECUTANDO EXPERIMENTO COM LOGGING NO GOOGLE DRIVE           ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📝 Log: $LOG_FILE"
echo "📊 Resultados: $DRIVE_RESULTS"
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo ""

# Executa comando e salva no Drive
cd /root/DSL-AG-hybrid
"$@" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                      EXPERIMENTO CONCLUÍDO                           ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "Código de saída: $EXIT_CODE"
echo "Log completo: $LOG_FILE"
echo ""

exit $EXIT_CODE
"""

    try:
        with open(script_path, 'w') as f:
            f.write(content)

        os.chmod(script_path, 0o755)
        print(f"\n✅ Script wrapper criado: {script_path}")
        print(f"   Uso: {script_path} python compare_gbml_vs_river.py --stream RBF --chunks 3")

    except Exception as e:
        print(f"⚠️  Erro ao criar script wrapper: {e}")


def main():
    """Função principal."""
    print("=" * 70)
    print("  SETUP REMOTO: GOOGLE COLAB via SSH")
    print("=" * 70)
    print()

    # 1. Verifica se está no Colab
    if not check_if_colab():
        print("❌ ERRO: Este script deve ser executado no Google Colab!")
        print("   Use sync_to_colab.sh para fazer upload primeiro.")
        sys.exit(1)

    print("✅ Detectado: Google Colab environment")
    print()

    # 2. Monta Drive
    drive_mount = mount_google_drive_silent()

    if not drive_mount:
        print()
        print("=" * 70)
        print("⚠️  ATENÇÃO: Google Drive NÃO está montado")
        print("=" * 70)
        print()
        print("Para montar o Drive, execute NO NOTEBOOK Colab (no navegador):")
        print()
        print("    from google.colab import drive")
        print("    drive.mount('/content/drive')")
        print()
        print("Depois execute este script novamente via SSH.")
        print()
        print("ℹ️  Você ainda pode executar experimentos, mas os resultados")
        print("   NÃO serão salvos no Drive (ficarão apenas no /root/)")
        print()
        sys.exit(1)

    # 3. Cria estrutura
    paths = create_drive_structure(drive_mount)

    if not paths:
        print("❌ Não foi possível criar estrutura no Drive")
        sys.exit(1)

    # 4. Cria arquivos auxiliares
    create_env_file(paths)
    create_config_override(paths)
    update_bashrc(paths)
    create_run_script(paths)

    # 5. Resumo final
    print()
    print("=" * 70)
    print("  ✅ SETUP CONCLUÍDO COM SUCESSO!")
    print("=" * 70)
    print()
    print(f"📂 Logs:        {paths['logs']}")
    print(f"📊 Resultados:  {paths['results']}")
    print(f"💾 Checkpoints: {paths['checkpoints']}")
    print()
    print("=" * 70)
    print("  COMO EXECUTAR EXPERIMENTOS")
    print("=" * 70)
    print()
    print("Opção 1: Usar wrapper (RECOMENDADO)")
    print("-" * 70)
    print("  /root/run_experiment.sh python compare_gbml_vs_river.py \\")
    print("      --stream RBF_Abrupt_Severe \\")
    print("      --chunks 3")
    print()
    print("Opção 2: Usar função bash")
    print("-" * 70)
    print("  source ~/.bashrc")
    print("  run-experiment python compare_gbml_vs_river.py \\")
    print("      --stream RBF_Abrupt_Severe \\")
    print("      --chunks 3")
    print()
    print("Opção 3: Executar diretamente")
    print("-" * 70)
    print("  python compare_gbml_vs_river.py \\")
    print("      --stream RBF_Abrupt_Severe \\")
    print("      --chunks 3 \\")
    print(f"      2>&1 | tee {paths['logs']}/experiment.log")
    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
