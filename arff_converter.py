"""
Conversor de Dados para Formato ARFF

Converte dados numpy/pandas para formato ARFF usado pelo MOA/ERulesD2S.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ARFFConverter:
    """Converte dados para formato ARFF (Attribute-Relation File Format)"""

    def __init__(self, relation_name: str = "dataset"):
        """
        Args:
            relation_name: Nome da relacao (dataset) no arquivo ARFF
        """
        self.relation_name = relation_name

    def convert_chunk(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        output_file: Union[str, Path] = "data.arff"
    ) -> Path:
        """
        Converte um chunk de dados para ARFF.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            feature_names: Nomes dos atributos (opcional)
            class_names: Nomes das classes (opcional)
            output_file: Caminho do arquivo de saida

        Returns:
            Path para arquivo ARFF criado
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_samples, n_features = X.shape

        # Gerar nomes de features se nao fornecidos
        if feature_names is None:
            feature_names = [f"att{i+1}" for i in range(n_features)]

        # Determinar classes unicas
        unique_classes = np.unique(y)
        if class_names is None:
            class_names = [str(int(c)) for c in unique_classes]

        logger.info(f"Convertendo chunk: {n_samples} instancias, {n_features} features")
        logger.info(f"Classes: {class_names}")

        # Escrever arquivo ARFF
        with open(output_path, 'w') as f:
            # Header
            f.write(f"@RELATION {self.relation_name}\n\n")

            # Attributes (features)
            for fname in feature_names:
                f.write(f"@ATTRIBUTE {fname} NUMERIC\n")

            # Classe (ultimo atributo)
            class_str = ','.join(class_names)
            f.write(f"@ATTRIBUTE class {{{class_str}}}\n")

            # Data section
            f.write("\n@DATA\n")

            # Instancias
            for i in range(n_samples):
                # Features
                features_str = ','.join([f"{X[i, j]:.6f}" for j in range(n_features)])
                # Classe
                class_label = class_names[list(unique_classes).index(y[i])]
                f.write(f"{features_str},{class_label}\n")

        logger.info(f"Arquivo ARFF criado: {output_path}")
        logger.info(f"Tamanho: {output_path.stat().st_size / 1024:.2f} KB")

        return output_path

    def convert_stream(
        self,
        chunks: List[tuple],  # List of (X, y) tuples
        output_dir: Union[str, Path],
        base_name: str = "chunk"
    ) -> List[Path]:
        """
        Converte multiplos chunks para arquivos ARFF separados.

        Args:
            chunks: Lista de tuplas (X, y)
            output_dir: Diretorio de saida
            base_name: Prefixo dos arquivos

        Returns:
            Lista de Paths dos arquivos ARFF criados
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        arff_files = []

        for idx, (X, y) in enumerate(chunks):
            output_file = output_dir / f"{base_name}_{idx}.arff"
            arff_path = self.convert_chunk(X, y, output_file=output_file)
            arff_files.append(arff_path)

        logger.info(f"Convertidos {len(arff_files)} chunks para ARFF")
        return arff_files

    def merge_chunks(
        self,
        chunks: List[tuple],  # List of (X, y) tuples
        output_file: Union[str, Path] = "merged.arff"
    ) -> Path:
        """
        Merge multiplos chunks em um unico arquivo ARFF.

        Args:
            chunks: Lista de tuplas (X, y)
            output_file: Arquivo de saida

        Returns:
            Path do arquivo ARFF merged
        """
        # Concatenar todos os chunks
        X_all = np.vstack([X for X, _ in chunks])
        y_all = np.hstack([y for _, y in chunks])

        return self.convert_chunk(X_all, y_all, output_file=output_file)


def validate_arff(arff_path: Union[str, Path]) -> bool:
    """
    Valida se um arquivo ARFF esta bem formado.

    Args:
        arff_path: Caminho do arquivo ARFF

    Returns:
        True se valido, False caso contrario
    """
    arff_path = Path(arff_path)

    if not arff_path.exists():
        logger.error(f"Arquivo nao encontrado: {arff_path}")
        return False

    try:
        with open(arff_path, 'r') as f:
            content = f.read()

        # Verificacoes basicas
        checks = [
            '@RELATION' in content,
            '@ATTRIBUTE' in content,
            '@DATA' in content,
            'class' in content.lower()
        ]

        if not all(checks):
            logger.error("Arquivo ARFF incompleto ou mal formado")
            return False

        # Contar linhas de dados
        data_section = content.split('@DATA')[1]
        data_lines = [line.strip() for line in data_section.split('\n') if line.strip() and not line.startswith('%')]

        logger.info(f"Arquivo ARFF validado: {len(data_lines)} instancias")
        return True

    except Exception as e:
        logger.error(f"Erro ao validar ARFF: {e}")
        return False


# Exemplo de uso
if __name__ == '__main__':
    # Configurar logging
    logging.basicConfig(level=logging.INFO)

    # Dados de exemplo
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    # Converter
    converter = ARFFConverter(relation_name="test_dataset")
    arff_path = converter.convert_chunk(X, y, output_file="test_data.arff")

    # Validar
    is_valid = validate_arff(arff_path)
    print(f"Arquivo ARFF valido: {is_valid}")
