# Plano de Ação Imediato - Finalização Fase 3

**Data:** 2025-11-22
**Decisão:** ✅ Aceitar protocolo atual e documentar diferenças
**Timeline:** 1-2 dias

---

## Decisão Tomada

✅ **ACEITAR PROTOCOLO CHUNK-WISE ATUAL**
- Todos os modelos usam mesmo protocolo básico
- Documentar diferenças metodológicas claramente
- Focar em análise de trade-offs entre abordagens
- NÃO re-executar experimentos

---

## Tarefas Hoje (4-6 horas)

### Tarefa 1: Confirmar NaN do ACDWM (2-3 horas)

#### 1.1. Criar Script de Teste Isolado
**Arquivo:** `test_acdwm_multiclass_isolated.py`

```python
"""
Testa ACDWM isoladamente em dataset multiclasse
para confirmar limitação.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Adicionar caminho do ACDWM
sys.path.insert(0, str(Path('ACDWM').resolve()))

from baseline_acdwm import ACDWMEvaluator

def test_acdwm_on_dataset(dataset_path, dataset_name):
    """Testa ACDWM em um dataset específico."""
    print(f"\n{'='*60}")
    print(f"Testando ACDWM em: {dataset_name}")
    print(f"{'='*60}")

    # Carregar dataset
    df = pd.read_csv(dataset_path)
    print(f"Shape: {df.shape}")
    print(f"Classes: {df['class'].unique()}")
    print(f"Distribuição:\n{df['class'].value_counts()}")

    # Preparar dados (primeiro 1000 samples)
    X = df.drop('class', axis=1).values[:1000]
    y = df['class'].values[:1000]

    classes = sorted(df['class'].unique())
    print(f"\nNúmero de classes: {len(classes)}")

    # Tentar criar avaliador ACDWM
    try:
        evaluator = ACDWMEvaluator(
            acdwm_path='ACDWM',
            classes=classes,
            evaluation_mode='train-then-test'
        )
        print("✓ ACDWMEvaluator criado com sucesso")
    except Exception as e:
        print(f"✗ ERRO ao criar evaluator: {e}")
        return False

    # Converter para formato River (dict)
    feature_names = df.drop('class', axis=1).columns
    X_dict = [
        {feat: float(val) for feat, val in zip(feature_names, instance)}
        for instance in X
    ]

    # Dividir em treino/teste
    split_idx = 500
    X_train = X_dict[:split_idx]
    y_train = list(y[:split_idx])
    X_test = X_dict[split_idx:]
    y_test = list(y[split_idx:])

    # Treinar
    print(f"\nTreinando em {len(X_train)} samples...")
    try:
        train_metrics = evaluator.train_on_chunk(X_train, y_train)
        print(f"✓ Treino concluído")
    except Exception as e:
        print(f"✗ ERRO no treino: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Testar
    print(f"\nTestando em {len(X_test)} samples...")
    try:
        test_metrics = evaluator.test_on_chunk(X_test, y_test)
        print(f"✓ Teste concluído")
        print(f"\nMétricas:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value}")

        # Verificar NaN
        if np.isnan(test_metrics.get('gmean', 0)):
            print(f"\n⚠️ CONFIRMADO: ACDWM retorna NaN em G-mean!")
            return False
        else:
            print(f"\n✓ ACDWM funcionou corretamente")
            return True

    except Exception as e:
        print(f"✗ ERRO no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    datasets_to_test = [
        ('datasets/processed/covertype_processed.csv', 'CovType (7 classes)'),
        ('datasets/processed/shuttle_processed.csv', 'Shuttle (7 classes)'),
        ('datasets/processed/intellabsensors_processed.csv', 'IntelLabSensors (2 classes)'),
    ]

    results = {}
    for path, name in datasets_to_test:
        success = test_acdwm_on_dataset(path, name)
        results[name] = "OK" if success else "FALHOU"

    print(f"\n{'='*60}")
    print("RESUMO DOS TESTES")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"{name}: {status}")
```

**Executar:**
```bash
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"
python test_acdwm_multiclass_isolated.py
```

#### 1.2. Verificar Logs do Batch 5
```bash
# Procurar mensagens de erro do ACDWM
grep -A 20 "ACDWM.*CovType\|ACDWM.*Shuttle\|ACDWM.*Intel" batch_5*.log
```

#### 1.3. Documentar Descobertas
**Arquivo:** `ACDWM_LIMITACAO_CONFIRMADA.md`

```markdown
# ACDWM: Limitação de Multiclasse Confirmada

**Data:** 2025-11-22
**Status:** [PREENCHER APÓS TESTE]

## Resultados dos Testes

### CovType (7 classes)
- Teste isolado: [OK/FALHOU]
- G-mean: [valor ou NaN]
- Erro: [descrição se houver]

### Shuttle (7 classes)
- Teste isolado: [OK/FALHOU]
- G-mean: [valor ou NaN]
- Erro: [descrição se houver]

### IntelLabSensors (2 classes)
- Teste isolado: [OK/FALHOU]
- G-mean: [valor ou NaN]
- Erro: [descrição se houver]

## Conclusão
[PREENCHER]

## Ação Tomada
Se confirmada limitação:
- Atribuir G-mean=0.0 para datasets falhados
- Documentar no paper: "ACDWM limitado a problemas binários"
```

---

### Tarefa 2: Ler Artigos (2-3 horas)

#### 2.1. ERulesD2S (paper-Bartosz.pdf)

**Seções para focar:**

1. **Abstract**
   - [ ] Qual o objetivo do ERulesD2S?
   - [ ] Qual a principal contribuição?

2. **Introduction**
   - [ ] Qual problema resolve?
   - [ ] Diferença para outros métodos?

3. **Methodology / Proposed Method**
   - [ ] Como funciona a evolução de regras?
   - [ ] Qual protocolo de avaliação?
   - [ ] Incremental ou batch?

4. **Experimental Setup**
   - [ ] Quantos datasets?
   - [ ] Tamanho dos datasets?
   - [ ] Protocolo de avaliação (prequential/holdout)?
   - [ ] Como dados foram divididos?

5. **Comparison with Baselines**
   - [ ] Quais baselines usados?
   - [ ] Mesmo protocolo para todos?
   - [ ] Como trataram concept drift?

**Criar tabela resumo:**
```markdown
| Aspecto | ERulesD2S |
|---------|-----------|
| Tipo | Rule-based evolutionary |
| Protocolo | [prequential/holdout/outro] |
| Treino | [incremental/batch/híbrido] |
| Window | [tamanho se houver] |
| Drift detection | [sim/não, método] |
```

#### 2.2. ACDWM (lu2020.pdf)

**Seções para focar:**

1. **Abstract**
   - [ ] Qual o objetivo do ACDWM?
   - [ ] O que é "adaptive chunk-based"?

2. **Methodology**
   - [ ] Como funciona o ensemble dinâmico?
   - [ ] Tamanho da window padrão?
   - [ ] Como pesos são atualizados?

3. **Experimental Setup**
   - [ ] Protocolo de avaliação usado?
   - [ ] Tamanho de chunks?
   - [ ] Prequential ou train-then-test?

4. **Limitations**
   - [ ] Menciona limitação de multiclasse?
   - [ ] Quais tipos de problemas testados?

**Criar tabela resumo:**
```markdown
| Aspecto | ACDWM |
|---------|-------|
| Tipo | Weighted ensemble |
| Window size | [valor padrão] |
| Protocolo | [prequential/outro] |
| Multiclasse | [suportado/não] |
| Chunk size | [valor usado] |
```

#### 2.3. Consolidar Descobertas

**Arquivo:** `ARTIGOS_RESUMO.md`

```markdown
# Resumo dos Artigos - ERulesD2S e ACDWM

## ERulesD2S (Bartosz et al.)
[Preencher após leitura]

### Protocolo usado no artigo:
- ...

### Comparação com nossa implementação:
- Igual: ...
- Diferente: ...

## ACDWM (Lu et al.)
[Preencher após leitura]

### Protocolo usado no artigo:
- ...

### Comparação com nossa implementação:
- Igual: ...
- Diferente: ...

## Implicações para nosso paper:
- ...
```

---

## Tarefas Amanhã (4-6 horas)

### Tarefa 3: Consolidar Resultados (2-3 horas)

#### 3.1. Verificar Estrutura de Resultados

```bash
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"

# Verificar resultados GBML
find experiments_6chunks_phase3_real -name "chunk_metrics.json" | wc -l
# Esperado: 17 (um por dataset)

# Verificar resultados baselines
find experiments_6chunks_phase3_real -name "*baseline*" -o -name "*ACDWM*" -o -name "*ARF*"
```

#### 3.2. Script de Consolidação

**Arquivo:** `consolidate_phase3_results.py`

```python
"""
Consolida todos os resultados da Fase 3
incluindo ACDWM com zeros para datasets falhados.
"""
import os
import json
import pandas as pd
from pathlib import Path

def consolidate_phase3():
    """Consolida resultados de todos os modelos."""
    base_dir = Path("experiments_6chunks_phase3_real")
    batches = ["batch_5", "batch_6", "batch_7"]

    # Modelos testados
    models = ['GBML', 'ACDWM', 'ARF', 'SRP', 'HAT', 'ERulesD2S']

    all_results = []

    for batch in batches:
        batch_dir = base_dir / batch

        # Listar datasets
        for dataset_dir in batch_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name

            # Para cada modelo
            for model in models:
                # GBML
                if model == 'GBML':
                    chunk_file = dataset_dir / "run_1" / "chunk_metrics.json"
                    if chunk_file.exists():
                        with open(chunk_file) as f:
                            chunks = json.load(f)
                        for chunk_data in chunks:
                            all_results.append({
                                'batch': batch,
                                'dataset': dataset_name,
                                'model': 'GBML',
                                'chunk': chunk_data['chunk'],
                                'test_gmean': chunk_data.get('test_gmean'),
                                'test_accuracy': chunk_data.get('test_accuracy'),
                            })

                # Outros modelos (procurar arquivos específicos)
                # [Implementar leitura de resultados de baselines]

    # Adicionar zeros para ACDWM em datasets falhados
    acdwm_failed = ['CovType', 'Shuttle', 'IntelLabSensors']  # Se confirmado

    for dataset in acdwm_failed:
        # Adicionar 5 chunks com G-mean=0.0
        for chunk in range(1, 6):
            all_results.append({
                'batch': 'batch_5',
                'dataset': dataset,
                'model': 'ACDWM',
                'chunk': chunk,
                'test_gmean': 0.0,
                'test_accuracy': 0.0,
            })

    # Criar DataFrame
    df = pd.DataFrame(all_results)

    # Salvar
    output_file = base_dir / "phase3_all_results_consolidated.csv"
    df.to_csv(output_file, index=False)

    print(f"✓ Consolidado: {len(df)} registros")
    print(f"✓ Salvo em: {output_file}")

    return df

if __name__ == "__main__":
    df = consolidate_phase3()

    # Estatísticas básicas
    print(f"\nEstatísticas:")
    print(f"  Total registros: {len(df)}")
    print(f"  Datasets: {df['dataset'].nunique()}")
    print(f"  Modelos: {df['model'].nunique()}")
    print(f"  Batches: {df['batch'].nunique()}")

    # Médias por modelo
    print(f"\nMédia G-mean por modelo:")
    print(df.groupby('model')['test_gmean'].mean().sort_values(ascending=False))
```

#### 3.3. Calcular Médias por Dataset

```python
# Após consolidação
df = pd.read_csv("experiments_6chunks_phase3_real/phase3_all_results_consolidated.csv")

# Calcular média por modelo-dataset
dataset_means = df.groupby(['model', 'dataset', 'batch']).agg({
    'test_gmean': ['mean', 'std'],
    'test_accuracy': ['mean', 'std']
}).reset_index()

dataset_means.columns = ['_'.join(col).strip('_') for col in dataset_means.columns]

# Salvar
dataset_means.to_csv(
    "experiments_6chunks_phase3_real/phase3_dataset_means.csv",
    index=False
)

print("✓ Médias por dataset salvas")
```

---

### Tarefa 4: Calcular Rankings (1 hora)

#### 4.1. Ranking Geral

```python
"""
Calcula ranking geral dos modelos.
"""
import pandas as pd

df_means = pd.read_csv("experiments_6chunks_phase3_real/phase3_dataset_means.csv")

# Ranking por G-mean
overall_ranking = df_means.groupby('model')['test_gmean_mean'].mean().sort_values(ascending=False)

print("Ranking Geral (Fase 3 - 17 datasets):")
for i, (model, gmean) in enumerate(overall_ranking.items(), 1):
    print(f"{i}. {model}: {gmean:.4f}")

# Comparar com Fase 2
print("\nComparação Fase 2 vs Fase 3:")
phase2_ranking = {
    'GBML': 0.7775,
    'ARF': 0.7240,
    'SRP': 0.7114,
    'ACDWM': 0.6998,
    'HAT': 0.6262,
    'ERulesD2S': 0.5511
}

for model in overall_ranking.index:
    phase3_score = overall_ranking[model]
    phase2_score = phase2_ranking.get(model, 0)
    diff = phase3_score - phase2_score
    print(f"{model}: {phase2_score:.4f} → {phase3_score:.4f} ({diff:+.4f})")
```

---

### Tarefa 5: Testes Estatísticos (1-2 horas)

#### 5.1. Script de Testes

```python
"""
Executa testes estatísticos completos.
"""
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

def friedman_test(df_means):
    """Friedman test para múltiplos modelos."""
    # Pivot: datasets × models
    pivot = df_means.pivot(index='dataset', columns='model', values='test_gmean_mean')

    # Friedman test
    statistic, p_value = stats.friedmanchisquare(*[pivot[col] for col in pivot.columns])

    print(f"Friedman Test:")
    print(f"  Statistic: {statistic:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Significativo: {'SIM' if p_value < 0.05 else 'NÃO'}")

    return statistic, p_value

def wilcoxon_pairwise(df_means, alpha=0.05):
    """Wilcoxon signed-rank pairwise com Bonferroni."""
    pivot = df_means.pivot(index='dataset', columns='model', values='test_gmean_mean')
    models = list(pivot.columns)

    n_comparisons = len(list(combinations(models, 2)))
    bonferroni_alpha = alpha / n_comparisons

    print(f"\nWilcoxon Pairwise (Bonferroni α={bonferroni_alpha:.6f}):")

    results = []
    for model1, model2 in combinations(models, 2):
        stat, p_value = stats.wilcoxon(pivot[model1], pivot[model2])
        significant = p_value < bonferroni_alpha

        results.append({
            'model1': model1,
            'model2': model2,
            'statistic': stat,
            'p_value': p_value,
            'significant': significant
        })

        print(f"  {model1} vs {model2}: p={p_value:.6f} {'*' if significant else ''}")

    return pd.DataFrame(results)

def cliffs_delta(x, y):
    """Calcula Cliff's Delta effect size."""
    n1, n2 = len(x), len(y)
    delta = 0
    for i in x:
        for j in y:
            if i > j:
                delta += 1
            elif i < j:
                delta -= 1

    return delta / (n1 * n2)

# Executar testes
df_means = pd.read_csv("experiments_6chunks_phase3_real/phase3_dataset_means.csv")

friedman_test(df_means)
wilcoxon_results = wilcoxon_pairwise(df_means)

# Salvar resultados
wilcoxon_results.to_csv(
    "experiments_6chunks_phase3_real/phase3_statistical_tests.csv",
    index=False
)
```

---

### Tarefa 6: Começar Atualização do Paper (1-2 horas)

#### 6.1. Seção Methodology - Adicionar

```latex
\subsection{Evaluation Protocol}

All models were evaluated using a \textbf{chunk-wise sequential train-then-test} protocol:
\begin{itemize}
    \item Data divided into 6 chunks of 1000 instances each
    \item For each iteration $i$ (0 to 4):
    \begin{itemize}
        \item Train on chunk $i$
        \item Test on chunk $i+1$
    \end{itemize}
    \item Total: 5 training rounds, 5 test evaluations
\end{itemize}

\subsubsection{Methodological Differences}

While all models use the same basic protocol, they differ in how they utilize information from previous chunks:

\begin{itemize}
    \item \textbf{GBML}: Re-executes genetic algorithm (200 generations) at each chunk, with population seeding from best individuals of previous chunks (memory-limited adaptation)

    \item \textbf{River models (ARF, SRP, HAT)}: Incremental learning via \texttt{learn\_one}, with complete model state persisting between chunks (continuous adaptation)

    \item \textbf{ACDWM}: Dynamic weighted ensemble with sliding window of recent samples (ensemble-based adaptation)

    \item \textbf{ERulesD2S}: Evolving rule-based system with incremental rule updates (rule-based adaptation)
\end{itemize}

These differences reflect the fundamental nature of each algorithm. GBML maintains a "notebook of best solutions" and re-thinks the problem from scratch at each chunk (using best ideas as inspiration), while River models maintain a "complete mental model" and adjust continuously with new information.
```

#### 6.2. Seção Results - Adicionar Nota

```latex
\subsection{ACDWM Limitation}

\textbf{Important note:} ACDWM failed to process multiclass datasets (>2 classes) in Batch 5, returning NaN for G-mean. Specifically:
\begin{itemize}
    \item CovType (7 classes): G-mean = 0.0
    \item Shuttle (7 classes): G-mean = 0.0
    \item [IntelLabSensors if confirmed]
\end{itemize}

This limitation was previously observed in Phase 2 with LED (10 classes) and WAVEFORM (3 classes) datasets. ACDWM's implementation uses binary label encoding (-1/+1), limiting it to binary classification problems. Following Phase 2 methodology, we assign G-mean=0.0 to failed datasets, ensuring a fair comparison where all models are evaluated on the same datasets.
```

---

## Checklist Geral

### Hoje
- [ ] Executar test_acdwm_multiclass_isolated.py
- [ ] Documentar resultados em ACDWM_LIMITACAO_CONFIRMADA.md
- [ ] Ler paper-Bartosz.pdf (ERulesD2S)
- [ ] Ler lu2020.pdf (ACDWM)
- [ ] Criar ARTIGOS_RESUMO.md

### Amanhã
- [ ] Executar consolidate_phase3_results.py
- [ ] Calcular médias por dataset
- [ ] Calcular ranking geral
- [ ] Executar testes estatísticos
- [ ] Atualizar Methodology no paper
- [ ] Atualizar Results no paper

---

## Arquivos a Criar

1. `test_acdwm_multiclass_isolated.py` - Teste ACDWM
2. `ACDWM_LIMITACAO_CONFIRMADA.md` - Documentação
3. `ARTIGOS_RESUMO.md` - Resumo dos artigos
4. `consolidate_phase3_results.py` - Consolidação
5. `calculate_rankings_phase3.py` - Rankings
6. `statistical_tests_phase3.py` - Testes
7. `paper_methodology_update.tex` - Atualização paper

---

## Entregas Finais (Esperadas em 2 dias)

1. ✅ ACDWM limitação confirmada e documentada
2. ✅ Artigos lidos e resumidos
3. ✅ Resultados Fase 3 consolidados com ACDWM zeros
4. ✅ Rankings finais calculados
5. ✅ Testes estatísticos completos
6. ✅ Paper Methodology atualizada
7. ✅ Paper Results atualizada com nota ACDWM

---

**Status:** PRONTO PARA EXECUTAR
**Próximo:** Criar test_acdwm_multiclass_isolated.py e executar
**Timeline:** 1-2 dias para completar tudo

**Criado por:** Claude Code
**Data:** 2025-11-22 11:00
