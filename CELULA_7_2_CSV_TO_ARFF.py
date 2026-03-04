# =============================================================================
# CELULA 7.2: Funcoes para Converter CSV para ARFF
# =============================================================================
# O CDCMS requer arquivos ARFF. Esta celula converte CSV -> ARFF
# =============================================================================

import pandas as pd
from pathlib import Path

def csv_to_arff(csv_path, arff_path, relation_name=None):
    """
    Converte um arquivo CSV para formato ARFF.

    Args:
        csv_path: Caminho do arquivo CSV
        arff_path: Caminho de saida para o ARFF
        relation_name: Nome da relacao (default: nome do arquivo)

    Returns:
        True se sucesso, False se erro
    """
    try:
        df = pd.read_csv(csv_path)

        if relation_name is None:
            relation_name = Path(csv_path).stem

        with open(arff_path, 'w') as f:
            f.write(f"@relation {relation_name}\n\n")

            # Atributos (todas as colunas exceto a ultima que e a classe)
            for col in df.columns[:-1]:
                # Verificar se e numerico ou categorico
                if df[col].dtype in ['int64', 'float64']:
                    f.write(f"@attribute {col} numeric\n")
                else:
                    unique_vals = sorted(df[col].unique())
                    vals_str = ",".join(str(v) for v in unique_vals)
                    f.write(f"@attribute {col} {{{vals_str}}}\n")

            # Classe (ultima coluna)
            class_col = df.columns[-1]
            unique_classes = sorted(df[class_col].unique())
            class_str = ",".join(str(int(c)) for c in unique_classes)
            f.write(f"@attribute class {{{class_str}}}\n\n")

            # Dados
            f.write("@data\n")
            for _, row in df.iterrows():
                values = ",".join(str(v) for v in row)
                f.write(f"{values}\n")

        return True
    except Exception as e:
        print(f"[ERRO] {e}")
        return False


def load_dataset_as_arff(dataset_name, chunks_dir, output_dir, chunk_size='chunk_2000'):
    """
    Carrega todos os chunks de um dataset e concatena em um unico ARFF.

    Args:
        dataset_name: Nome do dataset (ex: 'SINE_Abrupt_Simple')
        chunks_dir: Diretorio base dos chunks (unified_chunks/)
        output_dir: Diretorio de saida para o ARFF
        chunk_size: Tamanho do chunk ('chunk_500', 'chunk_1000', 'chunk_2000')

    Returns:
        Path do arquivo ARFF ou None se erro
    """
    dataset_path = Path(chunks_dir) / chunk_size / dataset_name

    if not dataset_path.exists():
        print(f"[ERRO] Dataset nao encontrado: {dataset_path}")
        return None

    # Listar e ordenar chunks
    chunks = sorted(dataset_path.glob("chunk_*.csv"), key=lambda x: int(x.stem.split('_')[1]))

    if not chunks:
        print(f"[ERRO] Nenhum chunk encontrado em: {dataset_path}")
        return None

    print(f"Dataset: {dataset_name} ({len(chunks)} chunks)")

    # Concatenar todos os chunks
    all_data = []
    for chunk_file in chunks:
        df = pd.read_csv(chunk_file)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"  Instancias: {len(combined_df)}")
    print(f"  Features: {len(combined_df.columns)-1}")
    print(f"  Classes: {sorted(combined_df[combined_df.columns[-1]].unique())}")

    # Criar ARFF
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    arff_path = output_dir / f"{dataset_name}.arff"

    # Escrever ARFF
    with open(arff_path, 'w') as f:
        f.write(f"@relation {dataset_name}\n\n")

        for col in combined_df.columns[:-1]:
            if combined_df[col].dtype in ['int64', 'float64']:
                f.write(f"@attribute {col} numeric\n")
            else:
                unique_vals = sorted(combined_df[col].unique())
                vals_str = ",".join(str(v) for v in unique_vals)
                f.write(f"@attribute {col} {{{vals_str}}}\n")

        class_col = combined_df.columns[-1]
        unique_classes = sorted(combined_df[class_col].unique())
        class_str = ",".join(str(int(c)) for c in unique_classes)
        f.write(f"@attribute class {{{class_str}}}\n\n")

        f.write("@data\n")
        for _, row in combined_df.iterrows():
            values = ",".join(str(v) for v in row)
            f.write(f"{values}\n")

    print(f"  ARFF salvo: {arff_path}")
    return arff_path


def prepare_all_datasets(chunks_dir, output_dir, chunk_size='chunk_2000', max_datasets=None):
    """
    Prepara todos os datasets convertendo para ARFF.

    Args:
        chunks_dir: Diretorio base (unified_chunks/)
        output_dir: Diretorio de saida
        chunk_size: Tamanho do chunk
        max_datasets: Limite de datasets (None = todos)

    Returns:
        Lista de paths dos ARFFs criados
    """
    chunk_path = Path(chunks_dir) / chunk_size

    if not chunk_path.exists():
        print(f"[ERRO] Diretorio nao encontrado: {chunk_path}")
        return []

    datasets = sorted([d.name for d in chunk_path.iterdir()
                       if d.is_dir() and not d.name.startswith('.')
                       and not d.name.endswith('_backup')])

    if max_datasets:
        datasets = datasets[:max_datasets]

    print(f"Preparando {len(datasets)} datasets...")
    print("="*50)

    arff_files = []
    for i, dataset in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] {dataset}")
        arff_path = load_dataset_as_arff(dataset, chunks_dir, output_dir, chunk_size)
        if arff_path:
            arff_files.append(arff_path)

    print("\n" + "="*50)
    print(f"[OK] {len(arff_files)} ARFFs criados em: {output_dir}")

    return arff_files


print("[OK] Funcoes de conversao CSV -> ARFF carregadas:")
print("  - csv_to_arff(csv_path, arff_path)")
print("  - load_dataset_as_arff(dataset_name, chunks_dir, output_dir)")
print("  - prepare_all_datasets(chunks_dir, output_dir)")
