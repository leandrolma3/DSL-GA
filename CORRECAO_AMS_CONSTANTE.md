# Correção: AMS Constante na Figura de Transition Metrics Evolution

**Data:** 2026-01-28
**Arquivos modificados:** `collect_results_for_paper.py`

---

## Problema

A figura `fig4_transition_evolution.pdf` (STAGGER Abrupt Chain) mostrava AMS como linha constante em 0.25. Todas as 3395 transições em `paper_data/egis_transition_metrics.csv` tinham `AMS=0.25`.

## Causa Raiz

Em `collect_results_for_paper.py`, linhas 565-585, existia um bloco "Simplified transition metrics" que calculava RIR corretamente mas usava um placeholder fixo para AMS:

```python
# Código antigo (removido)
'AMS': 0.25,  # Default placeholder
'TCS': round(0.6 * rule_change + 0.4 * 0.25 * (1 - rule_change), 4),
```

A função `calculate_transition_metrics()` (linhas 358-465 do mesmo arquivo) já implementava o cálculo real de AMS via similaridade de Levenshtein entre regras, mas nunca era chamada.

## Correções Aplicadas

### 1. Substituição do bloco simplificado (linhas 565-585)

O bloco placeholder foi substituído por chamada à função existente:

```python
# Código novo
transitions = calculate_transition_metrics(rules_data)
for t in transitions:
    all_transitions.append({
        'config': config_name,
        'config_label': config['label'],
        'batch': batch_name,
        'dataset': dataset_name,
        'drift_type': drift_type,
        'transition': t['transition'],
        'chunk_from': t['chunk_from'],
        'chunk_to': t['chunk_to'],
        'RIR': t['RIR'],
        'AMS': t['AMS'],
        'TCS': t['TCS'],
        'rules_from': t.get('total_rules_from', 0),
        'rules_to': t.get('total_rules_to', 0)
    })
```

### 2. Aceleração do cálculo de similaridade (linhas 326-356)

A função `calculate_rule_similarity()` usava Levenshtein em Python puro, causando timeout (20+ minutos no primeiro config). Foi adicionado uso de `rapidfuzz` (já instalado) como acelerador:

```python
try:
    from rapidfuzz.distance import Levenshtein as rf_lev
    return rf_lev.normalized_similarity(rule1, rule2)
except ImportError:
    pass
# fallback para implementação Python pura
```

Tempo de execução: 20+ minutos → ~13 segundos.

## Resultados Antes/Depois

### AMS para STAGGER_Abrupt_Chain

| Métrica | Antes (placeholder) | Depois (real) |
|---------|-------------------|---------------|
| count   | 64                | 64            |
| mean    | 0.2500            | 0.1912        |
| std     | 0.0000            | 0.1355        |
| min     | 0.2500            | 0.0000        |
| 25%     | 0.2500            | 0.0000        |
| 50%     | 0.2500            | 0.2357        |
| 75%     | 0.2500            | 0.2937        |
| max     | 0.2500            | 0.4111        |

### Transition Metrics globais (3395 transições)

| Métrica | Média  | Desvio |
|---------|--------|--------|
| TCS     | 0.2349 | 0.1145 |
| RIR     | 0.2509 | 0.2117 |
| AMS     | 0.2720 | 0.0971 |

## Arquivos Regenerados

Nenhum destes arquivos foi editado manualmente — foram regenerados pelos scripts:

| Arquivo | Script gerador |
|---------|---------------|
| `paper_data/egis_transition_metrics.csv` | `collect_results_for_paper.py` |
| `paper/figures/fig4_transition_evolution.pdf` | `regenerate_figures.py` |
| `paper/figures/fig_violin_rir_ams_tcs.pdf` | `regenerate_figures.py` |
| `paper/tables/table_xiv_transitions.tex` | `generate_paper_tables.py` |
| `paper/main.pdf` | `pdflatex main.tex` |

## Comandos de Reprodução

```bash
# 1. Coletar dados com AMS real
python collect_results_for_paper.py

# 2. Verificar que AMS varia
python -c "import pandas as pd; df=pd.read_csv('paper_data/egis_transition_metrics.csv'); s=df[df['dataset']=='STAGGER_Abrupt_Chain']; print(s['AMS'].describe())"

# 3. Regenerar figuras e tabelas
python regenerate_figures.py
python generate_paper_tables.py

# 4. Recompilar paper
cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

## Verificação

1. `egis_transition_metrics.csv` — AMS varia entre 0.0 e 0.41 (não constante 0.25)
2. `fig4_transition_evolution.pdf` — linha AMS mostra variação visível
3. Nenhum outro arquivo `.py` ou `.tex` foi editado manualmente
