DRIVE_BASE = Path('/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid')

# Verificar se existe
if DRIVE_BASE.exists():
    print(f"[OK] DRIVE_BASE: {DRIVE_BASE}")
else:
    print(f"[ERRO] Path nao encontrado: {DRIVE_BASE}")
    # Tentar alternativas
    alternatives = [
        Path('/content/drive/MyDrive/DSL-AG-hybrid'),
        Path('/content/drive/Shareddrives/DSL-AG-hybrid'),
    ]
    for alt in alternatives:
        if alt.exists():
            DRIVE_BASE = alt
            print(f"[OK] Usando alternativa: {DRIVE_BASE}")
            break

# Diretorios de dados
UNIFIED_CHUNKS_DIR = DRIVE_BASE / 'unified_chunks'
EXPERIMENTS_DIR = DRIVE_BASE / 'experiments_unified'

# Diretorios de trabalho no Colab (temporarios)
WORK_DIR = Path('/content')
CDCMS_JARS_DIR = WORK_DIR / 'cdcms_jars'
ROSE_JARS_DIR = WORK_DIR / 'rose_jars'
TEST_DIR = WORK_DIR / 'cdcms_test_output'
TEMP_ARFF_DIR = WORK_DIR / 'cdcms_temp_arff'

# Criar diretorios de trabalho
for d in [CDCMS_JARS_DIR, ROSE_JARS_DIR, TEST_DIR, TEMP_ARFF_DIR]:
    d.mkdir(exist_ok=True)

# Tamanhos de chunk disponiveis
CHUNK_SIZES = {
    'chunk_500': {'size': 500, 'num_chunks': 24},
    'chunk_1000': {'size': 1000, 'num_chunks': 12},
    'chunk_2000': {'size': 2000, 'num_chunks': 6},
}

# Batches por chunk_size
BATCHES = {
    'chunk_500': ['batch_1', 'batch_2', 'batch_3'],
    'chunk_1000': ['batch_1', 'batch_2', 'batch_3', 'batch_4'],
    'chunk_2000': ['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5', 'batch_6', 'batch_7'],
}

# Janela de holdout para comparacao justa com EGIS
HOLDOUT_WINDOW_SIZE = 100  # Primeiras 100 instancias de cada chunk

# Timeout para execucao do CDCMS (segundos)
CDCMS_TIMEOUT = 600  # 10 minutos

# Datasets a excluir (problematicos)
EXCLUDE_DATASETS = ['IntelLabSensors', 'PokerHand']  # NaN ou muito lentos

# Datasets MULTICLASSE (CDCMS suporta apenas binario)
MULTICLASS_DATASETS = {
    'LED_Abrupt_Simple': 10,
    'LED_Gradual_Simple': 10,
    'LED_Stationary': 10,
    'WAVEFORM_Abrupt_Simple': 3,
    'WAVEFORM_Gradual_Simple': 3,
    'WAVEFORM_Stationary': 3,
    'CovType': 7,
    'Shuttle': 7,
    'RBF_Stationary': 4,
}
