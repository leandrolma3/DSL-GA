"""
Configuração Automática para Google Colab + Google Drive
==========================================================

Este script configura o ambiente do Google Colab para:
1. Montar o Google Drive automaticamente
2. Configurar logging duplo (console + arquivo no Drive)
3. Criar diretórios de resultados no Drive
4. Salvar checkpoints automáticos

Uso:
    # No Colab, na primeira célula:
    from colab_drive_setup import setup_colab_environment

    # Configura tudo automaticamente
    drive_results_dir, log_file_path = setup_colab_environment(
        project_name="DSL-AG-hybrid",
        experiment_name="RBF_Abrupt_Severe_Test"
    )

    # Agora execute o experimento normalmente
    # Todos os logs e resultados serão salvos no Drive

Autor: Claude Code
Data: 2025-10-18
"""

import os
import sys
import logging
import shutil
from datetime import datetime
from pathlib import Path


def mount_google_drive(force_remount=False):
    """
    Monta o Google Drive no Colab.

    Args:
        force_remount (bool): Se True, força remontagem mesmo se já montado

    Returns:
        str: Caminho para o Google Drive montado
    """
    try:
        from google.colab import drive

        drive_mount_point = "/content/drive"

        # Verifica se já está montado
        if os.path.ismount(drive_mount_point) and not force_remount:
            print(f"✅ Google Drive já montado em: {drive_mount_point}")
            return drive_mount_point

        # Monta o drive
        print("🔄 Montando Google Drive...")
        drive.mount(drive_mount_point, force_remount=force_remount)
        print(f"✅ Google Drive montado com sucesso em: {drive_mount_point}")

        return drive_mount_point

    except ImportError:
        print("⚠️  AVISO: google.colab não está disponível. Você está rodando localmente?")
        print("   Retornando diretório local como fallback.")
        return os.path.expanduser("~")
    except Exception as e:
        print(f"❌ ERRO ao montar Google Drive: {e}")
        raise


def create_drive_directory_structure(drive_mount_point, project_name, experiment_name):
    """
    Cria estrutura de diretórios no Google Drive para o projeto.

    Estrutura criada:
    /content/drive/MyDrive/
    └── DSL-AG-hybrid/
        ├── experiments/
        │   └── RBF_Abrupt_Severe_Test_20251018_143022/
        │       ├── logs/
        │       ├── results/
        │       ├── checkpoints/
        │       └── plots/
        └── code_backup/

    Args:
        drive_mount_point (str): Caminho onde o Drive foi montado
        project_name (str): Nome do projeto (ex: "DSL-AG-hybrid")
        experiment_name (str): Nome do experimento (ex: "RBF_Abrupt_Severe_Test")

    Returns:
        dict: Dicionário com caminhos criados
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir_name = f"{experiment_name}_{timestamp}"

    # Estrutura de diretórios
    base_dir = os.path.join(drive_mount_point, "MyDrive", project_name)
    experiment_dir = os.path.join(base_dir, "experiments", experiment_dir_name)

    paths = {
        'base': base_dir,
        'experiment': experiment_dir,
        'logs': os.path.join(experiment_dir, "logs"),
        'results': os.path.join(experiment_dir, "results"),
        'checkpoints': os.path.join(experiment_dir, "checkpoints"),
        'plots': os.path.join(experiment_dir, "plots"),
        'code_backup': os.path.join(base_dir, "code_backup", timestamp)
    }

    # Cria todos os diretórios
    print(f"\n📁 Criando estrutura de diretórios no Drive...")
    for key, path in paths.items():
        try:
            os.makedirs(path, exist_ok=True)
            print(f"   ✅ {key:15s}: {path}")
        except Exception as e:
            print(f"   ❌ Erro ao criar {key}: {e}")
            raise

    print(f"\n✅ Estrutura de diretórios criada com sucesso!")
    print(f"   📂 Diretório principal: {experiment_dir}")

    return paths


def setup_dual_logging(log_file_path, log_level=logging.INFO):
    """
    Configura logging DUPLO: console + arquivo no Google Drive.

    Todos os logs serão salvos SIMULTANEAMENTE em:
    - Console do Colab (para acompanhamento em tempo real)
    - Arquivo de log no Google Drive (backup persistente)

    Args:
        log_file_path (str): Caminho completo para o arquivo de log no Drive
        log_level (int): Nível de log (logging.DEBUG, INFO, WARNING, etc.)

    Returns:
        logging.Logger: Logger configurado
    """
    # Remove handlers existentes
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configura formato
    log_format = '%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # 1. Handler para CONSOLE (Colab)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. Handler para ARQUIVO (Google Drive)
    try:
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        print(f"   ✅ Logging configurado:")
        print(f"      - Console: Ativo")
        print(f"      - Arquivo: {log_file_path}")
    except Exception as e:
        print(f"   ⚠️  Não foi possível criar arquivo de log: {e}")
        print(f"      Continuando apenas com logging no console.")

    root_logger.setLevel(log_level)

    return root_logger


def backup_code_to_drive(source_dir, backup_dir, extensions=['.py', '.yaml', '.yml']):
    """
    Faz backup do código-fonte para o Google Drive.

    Args:
        source_dir (str): Diretório com código-fonte (ex: /content/DSL-AG-hybrid)
        backup_dir (str): Diretório de destino no Drive
        extensions (list): Lista de extensões de arquivo para copiar

    Returns:
        int: Número de arquivos copiados
    """
    print(f"\n💾 Fazendo backup do código para o Drive...")
    print(f"   Origem: {source_dir}")
    print(f"   Destino: {backup_dir}")

    files_copied = 0

    try:
        for ext in extensions:
            for file_path in Path(source_dir).rglob(f'*{ext}'):
                # Ignora diretórios de ambiente virtual e cache
                if any(ignore in str(file_path) for ignore in ['venv', '__pycache__', '.git']):
                    continue

                # Calcula caminho relativo
                rel_path = file_path.relative_to(source_dir)
                dest_path = os.path.join(backup_dir, rel_path)

                # Cria diretório de destino se necessário
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Copia arquivo
                shutil.copy2(file_path, dest_path)
                files_copied += 1

        print(f"   ✅ {files_copied} arquivos copiados com sucesso!")

    except Exception as e:
        print(f"   ⚠️  Erro durante backup: {e}")

    return files_copied


def create_experiment_metadata(paths, project_name, experiment_name, config_file="config.yaml"):
    """
    Cria arquivo de metadados do experimento.

    Args:
        paths (dict): Dicionário com caminhos criados
        project_name (str): Nome do projeto
        experiment_name (str): Nome do experimento
        config_file (str): Nome do arquivo de configuração

    Returns:
        str: Caminho para o arquivo de metadados
    """
    metadata = {
        'project': project_name,
        'experiment': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'drive_paths': paths,
        'config_file': config_file,
        'python_version': sys.version,
        'working_directory': os.getcwd()
    }

    metadata_file = os.path.join(paths['experiment'], 'experiment_metadata.json')

    try:
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Metadados salvos: {metadata_file}")
    except Exception as e:
        print(f"   ⚠️  Erro ao salvar metadados: {e}")

    return metadata_file


def setup_colab_environment(
    project_name="DSL-AG-hybrid",
    experiment_name="experiment",
    config_file="config.yaml",
    backup_code=True,
    log_level=logging.INFO
):
    """
    Configura TUDO automaticamente para rodar no Google Colab.

    Esta função:
    1. Monta o Google Drive
    2. Cria estrutura de diretórios
    3. Configura logging duplo (console + arquivo)
    4. Faz backup do código (opcional)
    5. Cria arquivo de metadados

    Args:
        project_name (str): Nome do projeto
        experiment_name (str): Nome do experimento
        config_file (str): Nome do arquivo de configuração
        backup_code (bool): Se True, faz backup do código
        log_level (int): Nível de logging

    Returns:
        tuple: (drive_results_dir, log_file_path)
            - drive_results_dir: Diretório de resultados no Drive
            - log_file_path: Caminho para arquivo de log
    """
    print("="*70)
    print("🚀 SETUP AUTOMÁTICO PARA GOOGLE COLAB + GOOGLE DRIVE")
    print("="*70)

    # 1. Monta Google Drive
    drive_mount = mount_google_drive()

    # 2. Cria estrutura de diretórios
    paths = create_drive_directory_structure(drive_mount, project_name, experiment_name)

    # 3. Configura logging duplo
    log_file_path = os.path.join(paths['logs'], f"{experiment_name}.log")
    setup_dual_logging(log_file_path, log_level)

    # 4. Backup do código (se solicitado)
    if backup_code:
        source_dir = os.getcwd()
        backup_code_to_drive(source_dir, paths['code_backup'])

    # 5. Cria metadados
    create_experiment_metadata(paths, project_name, experiment_name, config_file)

    # 6. Cria symlink para facilitar acesso aos resultados
    try:
        results_link = os.path.join(os.getcwd(), "drive_results")
        if os.path.exists(results_link):
            os.remove(results_link)
        os.symlink(paths['results'], results_link)
        print(f"\n🔗 Symlink criado: {results_link} -> {paths['results']}")
    except Exception:
        pass  # Symlinks podem não funcionar em todos os ambientes

    # 7. Resumo final
    print("\n" + "="*70)
    print("✅ SETUP CONCLUÍDO COM SUCESSO!")
    print("="*70)
    print(f"📂 Resultados:    {paths['results']}")
    print(f"📝 Logs:          {log_file_path}")
    print(f"💾 Checkpoints:   {paths['checkpoints']}")
    print(f"📊 Plots:         {paths['plots']}")
    print("="*70)
    print("\n💡 Próximos passos:")
    print("   1. Execute seu experimento normalmente")
    print("   2. Todos os logs e resultados serão salvos automaticamente no Drive")
    print("   3. Mesmo que a sessão do Colab caia, tudo estará salvo!")
    print("\n")

    # Log inicial
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info(f"Iniciando experimento: {experiment_name}")
    logger.info(f"Diretório de resultados: {paths['results']}")
    logger.info(f"Arquivo de log: {log_file_path}")
    logger.info("="*70)

    return paths['results'], log_file_path, paths


# ============================================================================
# FUNÇÕES AUXILIARES PARA USO DURANTE O EXPERIMENTO
# ============================================================================

def save_checkpoint(data, checkpoint_name, checkpoints_dir):
    """
    Salva um checkpoint no Google Drive.

    Args:
        data: Dados a serem salvos (será serializado com pickle)
        checkpoint_name (str): Nome do checkpoint (ex: "chunk_3_elite")
        checkpoints_dir (str): Diretório de checkpoints

    Returns:
        str: Caminho do checkpoint salvo
    """
    import pickle

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = os.path.join(checkpoints_dir, f"{checkpoint_name}_{timestamp}.pkl")

    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Checkpoint salvo: {checkpoint_file}")
        return checkpoint_file
    except Exception as e:
        print(f"❌ Erro ao salvar checkpoint: {e}")
        return None


def periodic_sync_to_drive(source_file, drive_backup_dir):
    """
    Copia um arquivo para o Drive (útil para logs grandes).

    Args:
        source_file (str): Arquivo local a ser copiado
        drive_backup_dir (str): Diretório de destino no Drive
    """
    try:
        if os.path.exists(source_file):
            dest_file = os.path.join(drive_backup_dir, os.path.basename(source_file))
            shutil.copy2(source_file, dest_file)
            logging.debug(f"Sincronizado: {source_file} -> {dest_file}")
    except Exception as e:
        logging.warning(f"Erro ao sincronizar {source_file}: {e}")


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Exemplo de uso do script.
    """
    # Setup completo
    drive_results_dir, log_file, all_paths = setup_colab_environment(
        project_name="DSL-AG-hybrid",
        experiment_name="RBF_Abrupt_Severe_Test",
        backup_code=True,
        log_level=logging.INFO
    )

    # Agora você pode usar logging normalmente
    logging.info("Teste de log - vai para console E arquivo!")

    # Exemplo de checkpoint
    checkpoint_data = {'chunk': 0, 'best_fitness': 0.95}
    save_checkpoint(checkpoint_data, "test_checkpoint", all_paths['checkpoints'])

    print("\n✅ Teste concluído! Verifique o Drive para ver os arquivos salvos.")
