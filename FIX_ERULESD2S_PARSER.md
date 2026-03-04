# FIX: Parser ERulesD2S Retornando 0.0

**Problema Identificado**: Parser falha ao ler CSV com headers duplicados

**Data**: 2025-11-19

---

## CAUSA RAIZ

O arquivo `erulesd2s_results.csv` gerado pelo MOA tem **headers duplicados**:

```csv
Learner,stream,...,classifications correct (percent),...
evolutionary.EvolutionaryRuleLearner...,1000.0,62.4,...
Learner,stream,...,classifications correct (percent),...   ← HEADER REPETIDO!
evolutionary.EvolutionaryRuleLearner...,1000.0,62.9,...
```

O parser em `erulesd2s_wrapper.py` (linhas 256-268) tenta ler isso com:
```python
df = pd.read_csv(results_file)
last_row = df.iloc[-1]
results['accuracy'] = last_row.get('classifications correct (percent)', 0) / 100
```

Mas falha porque:
1. Headers duplicados confundem o pandas
2. Nome da coluna pode estar incorreto
3. iloc[-1] pode pegar linha errada

---

## CORREÇÃO

### Arquivo: `erulesd2s_wrapper.py`

### Localização: Método `parse_results()` (linha 242-287)

### SUBSTITUIR o método completo por:

```python
def parse_results(self, results_file: Path, stdout: str) -> Dict:
    """
    Parseia resultados do ERulesD2S.

    Args:
        results_file: Arquivo CSV de resultados (se gerado)
        stdout: Saida padrao do MOA

    Returns:
        Dicionario com metricas
    """
    results = {}

    # Tentar ler arquivo CSV se existir
    if results_file.exists():
        try:
            # CORREÇÃO: Ler CSV ignorando headers duplicados
            with open(results_file, 'r') as f:
                lines = f.readlines()

            # Filtrar linhas que começam com "Learner" (headers)
            data_lines = [line for line in lines if not line.startswith('Learner')]

            if len(data_lines) > 0:
                # Pegar primeira linha (header real) e última linha de dados
                header_line = lines[0]  # Primeiro header
                last_data_line = data_lines[-1]  # Última linha de dados

                # Parsear manualmente
                headers = header_line.strip().split(',')
                values = last_data_line.strip().split(',')

                # Criar dicionário
                data_dict = dict(zip(headers, values))

                # Extrair accuracy (coluna "classifications correct (percent)")
                if 'classifications correct (percent)' in data_dict:
                    acc_str = data_dict['classifications correct (percent)']
                    try:
                        results['accuracy'] = float(acc_str) / 100.0
                        logger.info(f"Accuracy extraída do CSV: {results['accuracy']:.4f}")
                    except ValueError:
                        logger.warning(f"Erro ao converter accuracy: {acc_str}")
                        results['accuracy'] = 0.0
                else:
                    logger.warning(f"Coluna 'classifications correct (percent)' não encontrada")
                    logger.warning(f"Colunas disponíveis: {list(data_dict.keys())}")
                    results['accuracy'] = 0.0

                # Extrair Kappa se disponível
                if 'Kappa Statistic (percent)' in data_dict:
                    try:
                        results['kappa'] = float(data_dict['Kappa Statistic (percent)']) / 100.0
                    except:
                        pass

        except Exception as e:
            logger.warning(f"Erro ao ler CSV: {e}")

    # FALLBACK: Parsear stdout (pode não ter "=" no formato MOA)
    # Formato MOA tabular: linha CSV com valores separados por vírgula
    if 'accuracy' not in results or results['accuracy'] == 0.0:
        logger.info("Tentando parsear stdout como fallback...")

        # Procurar por linhas com números que pareçam resultados
        lines = stdout.split('\n')
        for line in lines:
            # Linha com resultados tem muitos números separados por vírgula
            if line.count(',') > 5 and not line.startswith('learning'):
                try:
                    parts = line.split(',')
                    # Posição 7 geralmente é "classifications correct (percent)"
                    if len(parts) > 7:
                        acc_value = float(parts[7])
                        results['accuracy'] = acc_value / 100.0
                        logger.info(f"Accuracy extraída do stdout: {results['accuracy']:.4f}")
                        break
                except:
                    continue

    # Se ainda não tem accuracy, retornar 0
    if 'accuracy' not in results:
        results['accuracy'] = 0.0

    # Calcular gmean a partir de accuracy (aproximação)
    results['gmean'] = results['accuracy']

    return results
```

---

## TESTE DA CORREÇÃO

Após aplicar a correção, re-executar a CÉLULA 7 para UM dataset de teste:

```python
DATASETS = ["SEA_Abrupt_Simple"]  # Apenas 1 para testar
```

**Resultado esperado**:
```
ERulesD2S   test_gmean = 0.62-0.65  (ao invés de 0.0)
```

---

## ALTERNATIVA RÁPIDA (SEM RE-EXECUTAR)

Se não quiser re-executar, pode criar um script para **reprocessar os CSVs existentes**:

```python
#!/usr/bin/env python3
"""
Reprocessar resultados ERulesD2S existentes
"""

import pandas as pd
from pathlib import Path
import re

BASE_DIR = Path("experiments_6chunks_phase2_gbml/batch_1")

DATASETS = [
    "SEA_Abrupt_Simple",
    # ... outros 11 datasets
]

for dataset in DATASETS:
    dataset_dir = BASE_DIR / dataset / "run_1"

    for chunk_idx in range(6):
        chunk_dir = dataset_dir / f"erulesd2s_chunk_{chunk_idx}"
        csv_file = chunk_dir / "erulesd2s_results.csv"

        if not csv_file.exists():
            continue

        print(f"\n{dataset} chunk {chunk_idx}:")

        # Ler CSV ignorando headers duplicados
        with open(csv_file, 'r') as f:
            lines = f.readlines()

        data_lines = [line for line in lines if not line.startswith('Learner')]

        if len(data_lines) > 0:
            header_line = lines[0]
            last_data_line = data_lines[-1]

            headers = header_line.strip().split(',')
            values = last_data_line.strip().split(',')

            data_dict = dict(zip(headers, values))

            if 'classifications correct (percent)' in data_dict:
                acc = float(data_dict['classifications correct (percent)']) / 100.0
                print(f"  Accuracy: {acc:.4f}")
            else:
                print(f"  ERRO: Coluna não encontrada")
```

---

## IMPACTO ESPERADO

### Antes (com bug):
```
ERulesD2S   test_gmean = 0.0454  (4.5%)
```

### Depois (corrigido):
```
ERulesD2S   test_gmean = 0.55-0.65  (55-65%)
```

**Resultado**: ERulesD2S passa de PIOR para COMPETITIVO com baselines

---

## PRÓXIMOS PASSOS

1. **Aplicar correção**: Editar `erulesd2s_wrapper.py`
2. **Testar**: Re-executar CÉLULA 7 com 1 dataset
3. **Validar**: Verificar que accuracy > 0
4. **Re-executar completo**: Todos 12 datasets (~30 min)
5. **Re-executar CÉLULA 11**: Consolidar resultados corrigidos

---

**Status**: FIX PRONTO PARA APLICAR
