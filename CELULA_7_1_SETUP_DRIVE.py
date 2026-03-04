# =============================================================================
# CELULA 7.1: Setup do Google Drive e Caminhos
# =============================================================================
# Cole este codigo no Colab para configurar os caminhos corretos
# =============================================================================

from pathlib import Path

print("="*70)
print("SETUP DO GOOGLE DRIVE")
print("="*70)

# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Caminho base no Drive (ajuste conforme necessario)
DRIVE_BASE = Path('/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid')

# Verificar se existe
if DRIVE_BASE.exists():
    print(f"[OK] Caminho base encontrado: {DRIVE_BASE}")
else:
    print(f"[ERRO] Caminho nao encontrado: {DRIVE_BASE}")
    print("\nTente um destes caminhos alternativos:")
    alt_paths = [
        Path('/content/drive/MyDrive/DSL-AG-hybrid'),
        Path('/content/drive/My Drive/DSL-AG-hybrid'),
        Path('/content/drive/Shareddrives/DSL-AG-hybrid'),
    ]
    for p in alt_paths:
        if p.exists():
            print(f"  [ENCONTRADO] {p}")
            DRIVE_BASE = p
            break
        else:
            print(f"  [X] {p}")

# Diretorios de dados
UNIFIED_CHUNKS_DIR = DRIVE_BASE / 'unified_chunks'
CHUNK_SIZES = ['chunk_500', 'chunk_1000', 'chunk_2000']

# Verificar estrutura
print("\n--- Verificar Estrutura de Dados ---")

for chunk_size in CHUNK_SIZES:
    chunk_dir = UNIFIED_CHUNKS_DIR / chunk_size
    if chunk_dir.exists():
        datasets = [d.name for d in chunk_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        print(f"[OK] {chunk_size}: {len(datasets)} datasets")
    else:
        print(f"[X] {chunk_size}: nao encontrado")

# Carregar metadata
metadata_file = UNIFIED_CHUNKS_DIR / 'metadata.json'
if metadata_file.exists():
    import json
    with open(metadata_file) as f:
        METADATA = json.load(f)
    print(f"\n[OK] Metadata carregado:")
    print(f"     Datasets: {len(METADATA['datasets'])}")
    print(f"     Chunks por tamanho: 500={METADATA['num_chunks_500']}, 1000={METADATA['num_chunks_1000']}, 2000={METADATA['num_chunks_2000']}")
else:
    print("\n[AVISO] metadata.json nao encontrado")
    METADATA = None

# Definir diretorio padrao de trabalho
DEFAULT_CHUNK_SIZE = 'chunk_2000'  # 6 chunks - mais rapido para testes
CHUNKS_DIR = UNIFIED_CHUNKS_DIR / DEFAULT_CHUNK_SIZE

print(f"\n--- Configuracao Padrao ---")
print(f"Chunk size: {DEFAULT_CHUNK_SIZE}")
print(f"Diretorio: {CHUNKS_DIR}")

# Listar datasets disponiveis
if CHUNKS_DIR.exists():
    AVAILABLE_DATASETS = sorted([d.name for d in CHUNKS_DIR.iterdir()
                                  if d.is_dir() and not d.name.startswith('.')
                                  and not d.name.endswith('_backup')])
    print(f"\nDatasets disponiveis ({len(AVAILABLE_DATASETS)}):")
    for i, d in enumerate(AVAILABLE_DATASETS[:10]):
        print(f"  {i+1}. {d}")
    if len(AVAILABLE_DATASETS) > 10:
        print(f"  ... e mais {len(AVAILABLE_DATASETS)-10}")
else:
    AVAILABLE_DATASETS = []

print("\n[OK] Setup concluido!")
print("Variaveis disponiveis: DRIVE_BASE, CHUNKS_DIR, AVAILABLE_DATASETS, METADATA")
