"""
Testa ACDWM isoladamente em datasets multiclasse
para confirmar limitação.

OBJETIVO: Verificar se ACDWM retorna NaN em datasets com >2 classes

AUTOR: Claude Code
DATA: 2025-11-22
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s'
)
logger = logging.getLogger(__name__)

# Adicionar caminho do ACDWM
acdwm_path = Path('ACDWM').resolve()
if acdwm_path.exists():
    sys.path.insert(0, str(acdwm_path))
    logger.info(f"ACDWM path adicionado: {acdwm_path}")
else:
    logger.warning(f"ACDWM path não encontrado: {acdwm_path}")

from baseline_acdwm import ACDWMEvaluator


def test_acdwm_on_dataset(dataset_path, dataset_name, target_column='class'):
    """
    Testa ACDWM em um dataset específico.

    Args:
        dataset_path: Caminho para o CSV
        dataset_name: Nome do dataset (para exibição)
        target_column: Nome da coluna alvo

    Returns:
        dict com resultados do teste
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testando ACDWM em: {dataset_name}")
    logger.info(f"{'='*60}")

    results = {
        'dataset': dataset_name,
        'success': False,
        'error': None,
        'num_classes': 0,
        'gmean': None,
        'accuracy': None
    }

    try:
        # Carregar dataset
        df = pd.read_csv(dataset_path)
        logger.info(f"Shape: {df.shape}")

        # Verificar coluna target
        if target_column not in df.columns:
            raise ValueError(f"Coluna '{target_column}' não encontrada. Colunas disponíveis: {df.columns.tolist()}")

        # Informações sobre classes
        classes = sorted(df[target_column].unique())
        results['num_classes'] = len(classes)

        logger.info(f"Classes: {classes}")
        logger.info(f"Número de classes: {len(classes)}")
        logger.info(f"Distribuição:\n{df[target_column].value_counts()}")

        # Preparar dados (primeiro 1000 samples)
        sample_size = min(1000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)

        X = df_sample.drop(target_column, axis=1).values
        y = df_sample[target_column].values

        # Converter para formato River (dict)
        feature_names = df_sample.drop(target_column, axis=1).columns
        X_dict = [
            {feat: float(val) for feat, val in zip(feature_names, instance)}
            for instance in X
        ]

        # Dividir em treino/teste (50/50)
        split_idx = sample_size // 2
        X_train = X_dict[:split_idx]
        y_train = list(y[:split_idx])
        X_test = X_dict[split_idx:]
        y_test = list(y[split_idx:])

        logger.info(f"\nTreino: {len(X_train)} samples")
        logger.info(f"Teste: {len(X_test)} samples")

        # Tentar criar avaliador ACDWM
        logger.info("\nCriando ACDWMEvaluator...")
        try:
            evaluator = ACDWMEvaluator(
                acdwm_path='ACDWM',
                classes=classes,
                evaluation_mode='train-then-test'
            )
            logger.info("✓ ACDWMEvaluator criado com sucesso")
        except Exception as e:
            results['error'] = f"Erro ao criar evaluator: {e}"
            logger.error(f"✗ {results['error']}")
            return results

        # Treinar
        logger.info(f"\nTreinando em {len(X_train)} samples...")
        try:
            train_metrics = evaluator.train_on_chunk(X_train, y_train)
            logger.info(f"✓ Treino concluído")
        except Exception as e:
            results['error'] = f"Erro no treino: {e}"
            logger.error(f"✗ {results['error']}")
            import traceback
            traceback.print_exc()
            return results

        # Testar
        logger.info(f"\nTestando em {len(X_test)} samples...")
        try:
            test_metrics = evaluator.test_on_chunk(X_test, y_test)
            logger.info(f"✓ Teste concluído")

            # Extrair métricas
            results['gmean'] = test_metrics.get('gmean', None)
            results['accuracy'] = test_metrics.get('accuracy', None)

            logger.info(f"\nMétricas:")
            for key, value in test_metrics.items():
                logger.info(f"  {key}: {value}")

            # Verificar NaN
            if results['gmean'] is None or np.isnan(results['gmean']):
                logger.warning(f"\n⚠️ CONFIRMADO: ACDWM retorna NaN em G-mean!")
                results['success'] = False
                results['error'] = "G-mean = NaN"
            else:
                logger.info(f"\n✓ ACDWM funcionou corretamente")
                logger.info(f"  G-mean: {results['gmean']:.4f}")
                logger.info(f"  Accuracy: {results['accuracy']:.4f}")
                results['success'] = True

        except Exception as e:
            results['error'] = f"Erro no teste: {e}"
            logger.error(f"✗ {results['error']}")
            import traceback
            traceback.print_exc()
            return results

    except Exception as e:
        results['error'] = f"Erro geral: {e}"
        logger.error(f"✗ {results['error']}")
        import traceback
        traceback.print_exc()

    return results


def main():
    """Testa ACDWM em múltiplos datasets."""
    datasets_to_test = [
        ('datasets/processed/covertype_processed.csv', 'CovType', 'class'),
        ('datasets/processed/shuttle_processed.csv', 'Shuttle', 'class'),
        ('datasets/processed/intellabsensors_processed.csv', 'IntelLabSensors', 'class'),
    ]

    all_results = []

    for path, name, target_col in datasets_to_test:
        # Verificar se arquivo existe
        if not Path(path).exists():
            logger.warning(f"Arquivo não encontrado: {path}")
            all_results.append({
                'dataset': name,
                'success': False,
                'error': 'Arquivo não encontrado',
                'num_classes': 0,
                'gmean': None,
                'accuracy': None
            })
            continue

        # Executar teste
        result = test_acdwm_on_dataset(path, name, target_col)
        all_results.append(result)

    # Resumo final
    logger.info(f"\n{'='*60}")
    logger.info("RESUMO DOS TESTES")
    logger.info(f"{'='*60}")

    for result in all_results:
        status = "✓ OK" if result['success'] else "✗ FALHOU"
        error_msg = f" ({result['error']})" if result['error'] else ""
        logger.info(f"{result['dataset']} ({result['num_classes']} classes): {status}{error_msg}")
        if result['gmean'] is not None and not np.isnan(result['gmean']):
            logger.info(f"  G-mean: {result['gmean']:.4f}")

    # Salvar resultados em arquivo
    results_df = pd.DataFrame(all_results)
    output_file = "acdwm_multiclass_test_results.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Resultados salvos em: {output_file}")

    # Conclusão
    failed_datasets = [r for r in all_results if not r['success']]
    if failed_datasets:
        logger.warning(f"\n⚠️ LIMITAÇÃO CONFIRMADA: ACDWM falhou em {len(failed_datasets)} dataset(s)")
        logger.warning("Ação recomendada: Atribuir G-mean=0.0 para esses datasets")
    else:
        logger.info("\n✓ ACDWM funcionou em todos os datasets testados")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTeste interrompido pelo usuário")
    except Exception as e:
        logger.error(f"\nErro fatal: {e}")
        import traceback
        traceback.print_exc()
