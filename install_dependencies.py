"""
Script de Instalação de Dependências para Sistema de Comparação GBML vs River

Execute este script antes de usar o sistema de comparação.
"""

import subprocess
import sys

DEPENDENCIES = [
    'pandas',
    'matplotlib',
    'seaborn',
    'river',
    'scikit-learn',
    'pyyaml',
    'numpy',
    'python-Levenshtein'  # Para cálculo de diversidade
]

def install_package(package):
    """Instala um pacote usando pip"""
    print(f"Instalando {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} instalado com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Erro ao instalar {package}: {e}")
        return False

def check_package(package):
    """Verifica se um pacote está instalado"""
    try:
        __import__(package.replace('-', '_'))
        return True
    except ImportError:
        return False

def main():
    print("="*70)
    print("INSTALAÇÃO DE DEPENDÊNCIAS - GBML vs River Comparison")
    print("="*70)

    print("\n[1/3] Verificando dependências já instaladas...")
    already_installed = []
    to_install = []

    for package in DEPENDENCIES:
        check_name = package.split('[')[0]  # Remove extras como [all]
        check_name = check_name.replace('python-Levenshtein', 'Levenshtein')

        if check_package(check_name):
            already_installed.append(package)
            print(f"  ✓ {package}")
        else:
            to_install.append(package)
            print(f"  ✗ {package} (faltando)")

    if not to_install:
        print("\n✓ Todas as dependências já estão instaladas!")
        return True

    print(f"\n[2/3] Instalando {len(to_install)} dependência(s) faltante(s)...")
    print(f"  Pacotes: {', '.join(to_install)}")

    failed = []
    for package in to_install:
        if not install_package(package):
            failed.append(package)

    print("\n[3/3] Resumo da instalação:")
    print(f"  ✓ Já instalados: {len(already_installed)}")
    print(f"  ✓ Instalados agora: {len(to_install) - len(failed)}")
    if failed:
        print(f"  ✗ Falhas: {len(failed)} ({', '.join(failed)})")
        return False

    print("\n" + "="*70)
    print("✓ INSTALAÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*70)
    print("\nVocê pode agora executar:")
    print("  python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --chunks 3")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
