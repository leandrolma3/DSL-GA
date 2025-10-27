"""
Setup Automático: SSH + Google Drive
======================================

Este script:
1. Monta o Google Drive
2. Configura logging para salvar no Drive
3. Inicia Cloudflare Tunnel para SSH
4. Mantém sessão ativa

Uso no Colab (Notebook):
------------------------
# Célula 1:
!wget https://raw.githubusercontent.com/seu-repo/DSL-AG-hybrid/main/setup_ssh_with_drive.py
!python setup_ssh_with_drive.py

# Agora conecte via SSH normalmente
# Todos os logs serão salvos automaticamente no Drive

Autor: Claude Code
Data: 2025-10-18
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path


def mount_google_drive():
    """Monta Google Drive."""
    try:
        from google.colab import drive

        drive_mount = "/content/drive"

        if os.path.ismount(drive_mount):
            print("✅ Google Drive já está montado")
            return drive_mount

        print("🔄 Montando Google Drive...")
        drive.mount(drive_mount)
        print(f"✅ Drive montado em: {drive_mount}")

        return drive_mount

    except ImportError:
        print("❌ ERRO: Este script deve ser executado no Google Colab")
        sys.exit(1)


def setup_drive_directories(drive_mount):
    """Cria estrutura de diretórios no Drive."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.join(drive_mount, "MyDrive", "DSL-AG-hybrid")

    dirs = {
        'base': base_dir,
        'experiments': os.path.join(base_dir, "experiments"),
        'logs': os.path.join(base_dir, "logs"),
        'current_session': os.path.join(base_dir, "experiments", f"ssh_session_{timestamp}"),
        'current_logs': os.path.join(base_dir, "logs", timestamp)
    }

    print("\n📁 Criando estrutura no Drive...")
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"   ✅ {name:20s}: {path}")

    return dirs


def configure_logging_to_drive(log_dir):
    """Configura logging para salvar no Drive."""
    log_file = os.path.join(log_dir, "ssh_session.log")

    # Remove handlers existentes
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # File handler (Drive)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Configura root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(console)
    logging.root.addHandler(file_handler)

    print(f"\n📝 Logging configurado:")
    print(f"   - Console: Ativo")
    print(f"   - Arquivo: {log_file}")

    return log_file


def create_ssh_config_file(drive_dirs):
    """Cria arquivo de configuração para SSH."""
    config_file = os.path.join(drive_dirs['current_session'], "ssh_config.txt")

    with open(config_file, 'w') as f:
        f.write("CONFIGURAÇÃO SSH + GOOGLE DRIVE\n")
        f.write("="*70 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("DIRETÓRIOS DO DRIVE:\n")
        for name, path in drive_dirs.items():
            f.write(f"  {name:20s}: {path}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("COMO USAR VIA SSH:\n")
        f.write("="*70 + "\n")
        f.write("1. Conecte via SSH:\n")
        f.write("   ssh your-tunnel.trycloudflare.com\n\n")
        f.write("2. Navegue para o projeto:\n")
        f.write("   cd ~/DSL-AG-hybrid\n\n")
        f.write("3. Execute com logging para o Drive:\n")
        f.write("   python compare_gbml_vs_river.py \\\n")
        f.write("       --stream RBF_Abrupt_Severe \\\n")
        f.write(f"       2>&1 | tee {drive_dirs['current_logs']}/experiment.log\n\n")
        f.write("4. Ou use o wrapper:\n")
        f.write(f"   export DRIVE_LOG_DIR={drive_dirs['current_logs']}\n")
        f.write("   python run_experiment_ssh.py --stream RBF_Abrupt_Severe\n\n")
        f.write("="*70 + "\n")

    print(f"📄 Configuração salva: {config_file}")
    return config_file


def setup_cloudflare_tunnel():
    """Configura e inicia Cloudflare Tunnel."""
    print("\n🔧 Configurando Cloudflare Tunnel...")

    # Download cloudflared
    cloudflared_path = "/tmp/cloudflared"

    if not os.path.exists(cloudflared_path):
        print("   📥 Baixando cloudflared...")
        subprocess.run([
            "wget", "-q", "-O", cloudflared_path,
            "https://github.com/cloudflare/cloudflare-release/releases/latest/download/cloudflared-linux-amd64"
        ])
        subprocess.run(["chmod", "+x", cloudflared_path])

    print("   ✅ Cloudflared pronto")

    return cloudflared_path


def start_ssh_server():
    """Inicia servidor SSH."""
    print("\n🔐 Configurando SSH...")

    # Instala openssh-server se necessário
    subprocess.run(["apt-get", "update", "-qq"], stdout=subprocess.DEVNULL)
    subprocess.run(["apt-get", "install", "-y", "-qq", "openssh-server"], stdout=subprocess.DEVNULL)

    # Configura SSH
    subprocess.run(["mkdir", "-p", "/var/run/sshd"])
    subprocess.run(["echo", "root:root"], stdout=subprocess.PIPE,
                   input=b'root:root\n', check=True)

    # Configura senha (se ainda não estiver configurada)
    subprocess.run(["sh", "-c", "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config"])
    subprocess.run(["sh", "-c", "echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config"])

    # Inicia SSH
    subprocess.run(["/usr/sbin/sshd"])

    print("   ✅ Servidor SSH iniciado")


def create_ssh_wrapper_script(drive_dirs):
    """Cria script wrapper para executar experimentos via SSH."""
    wrapper_path = "/root/run_experiment_ssh.sh"

    content = f"""#!/bin/bash
# Wrapper para executar experimentos com logging no Drive
# Gerado automaticamente em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DRIVE_LOG_DIR="{drive_dirs['current_logs']}"
DRIVE_RESULTS_DIR="{drive_dirs['current_session']}"

# Cria timestamp para o log
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$DRIVE_LOG_DIR/experiment_$TIMESTAMP.log"

# Garante que o diretório de logs existe antes de usar tee
mkdir -p "$(dirname "$LOG_FILE")" || {{
    echo "❌ ERRO: Não foi possível criar diretório de logs"
    exit 1
}}

echo "================================================================================"
echo "  EXECUTANDO EXPERIMENTO COM LOGGING NO GOOGLE DRIVE"
echo "================================================================================"
echo "Log será salvo em: $LOG_FILE"
echo "Resultados em: $DRIVE_RESULTS_DIR"
echo "================================================================================"
echo ""

# Executa o comando passado como argumento e salva no Drive
"$@" 2>&1 | tee "$LOG_FILE"

# Salva código de retorno
EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "  EXPERIMENTO CONCLUÍDO"
echo "================================================================================"
echo "Código de saída: $EXIT_CODE"
echo "Log completo: $LOG_FILE"
echo "================================================================================"

exit $EXIT_CODE
"""

    with open(wrapper_path, 'w') as f:
        f.write(content)

    os.chmod(wrapper_path, 0o755)

    print(f"\n🔧 Wrapper criado: {wrapper_path}")
    print(f"   Uso: /root/run_experiment_ssh.sh python main.py")

    return wrapper_path


def create_bashrc_config(drive_dirs):
    """Adiciona configuração ao .bashrc para facilitar uso."""
    bashrc_additions = f"""

# ========== CONFIGURAÇÃO AUTOMÁTICA: Google Drive ==========
export DRIVE_BASE="{drive_dirs['base']}"
export DRIVE_LOGS="{drive_dirs['current_logs']}"
export DRIVE_RESULTS="{drive_dirs['current_session']}"

# Alias úteis
alias drive-logs='cd "$DRIVE_LOGS"'
alias drive-results='cd "$DRIVE_RESULTS"'
alias drive-tail='tail -f "$DRIVE_LOGS"/*.log'

# Função para executar com log no Drive
run-with-drive-log() {{
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$DRIVE_LOGS/cmd_$TIMESTAMP.log"
    # Garante que o diretório existe antes de usar tee
    mkdir -p "$DRIVE_LOGS" 2>/dev/null || true
    echo "Salvando log em: $LOG_FILE"
    "$@" 2>&1 | tee "$LOG_FILE"
}}

echo "✅ Configuração do Drive carregada!"
echo "   DRIVE_LOGS: $DRIVE_LOGS"
echo "   DRIVE_RESULTS: $DRIVE_RESULTS"
# =========================================================
"""

    bashrc_path = os.path.expanduser("~/.bashrc")

    with open(bashrc_path, 'a') as f:
        f.write(bashrc_additions)

    print("\n⚙️  Configurações adicionadas ao .bashrc")
    print("   Execute: source ~/.bashrc")
    print("   Ou reconecte via SSH")


def main():
    """Função principal."""
    print("="*70)
    print("  SETUP AUTOMÁTICO: SSH + GOOGLE DRIVE")
    print("="*70)

    # 1. Monta Drive
    drive_mount = mount_google_drive()

    # 2. Cria diretórios
    drive_dirs = setup_drive_directories(drive_mount)

    # 3. Configura logging
    log_file = configure_logging_to_drive(drive_dirs['current_logs'])

    logging.info("="*70)
    logging.info("Inicializando setup SSH + Google Drive")
    logging.info("="*70)

    # 4. Cria arquivo de configuração
    config_file = create_ssh_config_file(drive_dirs)

    # 5. Cria wrapper script
    wrapper_script = create_ssh_wrapper_script(drive_dirs)

    # 6. Configura .bashrc
    create_bashrc_config(drive_dirs)

    # 7. Setup Cloudflare
    cloudflared_path = setup_cloudflare_tunnel()

    # 8. Inicia SSH
    start_ssh_server()

    # 9. Resumo final
    print("\n" + "="*70)
    print("  ✅ SETUP CONCLUÍDO COM SUCESSO!")
    print("="*70)
    print(f"📂 Logs do Drive:     {drive_dirs['current_logs']}")
    print(f"📊 Resultados:        {drive_dirs['current_session']}")
    print(f"📝 Log da sessão:     {log_file}")
    print(f"📄 Configuração:      {config_file}")
    print(f"🔧 Wrapper:           {wrapper_script}")
    print("="*70)

    # 10. Inicia Cloudflare Tunnel
    print("\n🚀 Iniciando Cloudflare Tunnel...")
    print("   IMPORTANTE: Copie a URL do SSH que aparecerá abaixo!")
    print("="*70 + "\n")

    logging.info("Iniciando Cloudflare Tunnel")

    try:
        # Inicia tunnel (bloqueante)
        subprocess.run([
            cloudflared_path,
            "tunnel",
            "--url", "ssh://localhost:22"
        ])
    except KeyboardInterrupt:
        logging.info("Tunnel interrompido pelo usuário")
        print("\n\n⚠️  Tunnel encerrado")
    except Exception as e:
        logging.error(f"Erro no tunnel: {e}")
        print(f"\n❌ ERRO: {e}")


if __name__ == "__main__":
    main()
