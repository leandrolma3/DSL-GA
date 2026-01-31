# Status do Artigo IEEE TKDE - EGIS

**Ultima atualizacao**: 2026-01-30
**Branch**: main
**Ultimo commit**: `dd21d12` - Fix TABLE VII: remove ACDWM/CDCMS from multiclass block

---

## 1. Estrutura Atual do Paper

**Arquivo principal**: `paper/main.tex` (1056 linhas, 22 paginas compiladas)

### Secoes:
| Secao | Conteudo | Status |
|-------|----------|--------|
| I - Introduction | Motivacao, RQs | Escrito |
| II - Related Work | Literatura | Escrito |
| III - EGIS Framework | Algoritmo proposto | Escrito |
| IV - Experimental Setup | Datasets, baselines, metricas | Escrito |
| V - Results and Discussion | Resultados experimentais | **Escrito, com inconsistencias pendentes** |
| VI - Conclusion | Conclusoes e trabalhos futuros | Escrito |
| Bibliography | 20 referencias | Escrito |

### Tabelas:
| Tabela | Label | Fonte | Status |
|--------|-------|-------|--------|
| TABLE VII - Summary Performance | `tab:summary_all` | Inline (main.tex:619-647) | **Corrigida em 30/01** |
| TABLE VIII - Performance by Drift Type | `tab:drift_performance` | Inline (main.tex:659-678) | Valores possivelmente desatualizados |
| TABLE - Binary Comparison (42 datasets) | `tab:binary_comparison` | `\input{tables/table_binary_comparison}` | OK |
| TABLE - Binary Comparison EXP-1000 | `tab:binary_comparison_1000` | `\input{tables/table_binary_comparison_1000}` | OK |
| TABLE IX - Friedman Ranking | `tab:complete_ranking` | `\input{tables/table_ix_ranking}` | OK (via script) |
| TABLE - Wilcoxon Tests | `tab:stat_tests` | Inline (main.tex:780-797) | Verificar |
| TABLE - Transition Metrics | `tab:transitions` | Inline (main.tex:805-828) | OK |
| TABLE - Rule Complexity | `tab:complexity_detailed` | Inline (main.tex:686-701) | Verificar |
| TABLE - Penalty Effect | `tab:penalty_effect` | Inline (main.tex:873-886) | Verificar |
| TABLE - Rule Examples | `tab:rule_examples` | Inline (main.tex:709-733) | OK (qualitativo) |

### Figuras (20 PDFs + 6 PNGs em `paper/figures/`):
- Todas referenciadas no main.tex
- Nota: `fig_tcs_timeseries.png` e `fig_rir_ams_scatter.png` sao .png (restante e .pdf)

---

## 2. Inconsistencias Conhecidas (ATENCAO NA RETOMADA)

### 2.1. CRITICO - Valores EGIS divergentes entre tabelas

A TABLE VII (corrigida) agora mostra EGIS com **0.858 (EXP-500)** e **0.868 (EXP-1000)** no bloco binary.

Porem, a TABLE VIII (Performance by Drift Type, linha 675) ainda mostra EGIS Overall = **0.898**. Esta e a tabela inline de drift por tipo que usa EXP-500.

**Possivel causa**: TABLE VII usa a media de EGIS sobre TODAS as configs (EXP-500, EXP-1000, EXP-2000 onde disponivel), enquanto TABLE VIII usa apenas EXP-500. Precisa verificar se 0.898 esta correto para EXP-500 only ou se tambem precisa atualizacao.

Textos que referenciam 0.898 e podem estar inconsistentes:
- Linha 680: "EGIS achieves G-Mean of 0.898 overall"
- Linha 985 (Conclusion, RQ3): "EGIS achieves G-Mean of 0.898 overall"

### 2.2. CRITICO - Conclusion usa valores antigos

A secao Conclusion (linhas 979-987) referencia:
- "average G-Mean of 0.80-0.81" (linha 981) -- possivelmente de all 48 datasets, verificar
- "0.803 vs 0.578" (linha 981) e "22.5 percentage points" -- verificar se batem com dados atuais
- "0.898 overall" (linha 985) -- ver item 2.1

### 2.3. CRITICO - Secao Discussion usa valores possivelmente antigos

Linha 914: "ensemble methods achieve higher G-Mean (0.88-0.91) than EGIS (0.80)"
Linha 920: "EGIS outperforms ERulesD2S by 22.5 percentage points (0.803 vs 0.578)"
Linha 918: "EGIS maintains consistent performance (G-Mean = 0.80-0.81)"

Esses valores (0.80, 0.803, 0.578) sao para all 48 datasets? Precisam ser verificados contra os dados regenerados.

### 2.4. MEDIO - Tabelas inline vs geradas por script

Varias tabelas no main.tex sao **inline** (hardcoded) e nao usam `\input`. O script `generate_paper_tables.py` gera versoes atualizadas em `paper/tables/`, mas o main.tex nao as referencia:

| Tabela inline | Arquivo gerado correspondente | Usa \input? |
|---------------|-------------------------------|-------------|
| TABLE VII (summary) | `table_vii_summary.tex` | NAO |
| TABLE VIII (drift) | `table_viii_drift.tex` | NAO |
| TABLE (wilcoxon) | `table_xi_wilcoxon.tex` | NAO |
| TABLE (penalty) | `table_xii_penalty.tex` | NAO |
| TABLE (complexity) | `table_xiii_complexity.tex` | NAO |
| TABLE (transitions) | `table_xiv_transitions.tex` | NAO |

**Risco**: Ao atualizar dados, as tabelas inline ficam desatualizadas. Considerar migrar para `\input{}`.

### 2.5. BAIXO - CDCMS ausente do bloco binary na TABLE VII

O script `generate_paper_tables.py` nao gera CDCMS no bloco binary porque CDCMS nao aparece em `consolidated_results.csv` com os model names esperados, ou nao esta em `MODELS_ORDER`. A TABLE VII inline anterior tinha CDCMS (0.843) mas a versao corrigida removeu. Verificar se CDCMS deveria estar no bloco binary (42 datasets) e se os dados existem.

---

## 3. Regras de Ouro para Retomada

### 3.1. Datasets excluidos (NUNCA incluir)
```
EXCLUDED_DATASETS = ['CovType', 'IntelLabSensors', 'PokerHand', 'Shuttle']
```
Definidos em `generate_paper_tables.py` linha 49. Essas 4 bases foram excluidas por problemas de qualidade/convergencia. O total de datasets e **48** (42 binary + 6 multiclass).

### 3.2. Modelos binary-only (NUNCA no bloco multiclass/all-48)
```
Binary-only: ACDWM, CDCMS
```
Esses modelos nao suportam classificacao multiclass. No bloco "All datasets (n=48)" da TABLE VII, apenas 6 modelos aparecem: EGIS, ARF, SRP, HAT, ROSE, ERulesD2S.

### 3.3. Datasets multiclass
```
LED_Abrupt_Simple, LED_Gradual_Simple, LED_Stationary,
WAVEFORM_Abrupt_Simple, WAVEFORM_Gradual_Simple, WAVEFORM_Stationary
```

### 3.4. Fonte de verdade para dados
- **Dados brutos**: `paper_data/consolidated_results.csv`
- **Script de geracao**: `generate_paper_tables.py`
- **Tabelas geradas**: `paper/tables/*.tex`
- **Estatisticas**: `paper_data/statistical_results.json`

Sempre regenerar via script ao invez de editar valores manualmente.

### 3.5. Ordem de modelos no paper
```python
MODELS_ORDER = ['EGIS', 'ARF', 'SRP', 'ROSE_Original', 'ROSE_ChunkEval', 'HAT', 'ACDWM', 'ERulesD2S']
```

### 3.6. Configuracoes experimentais
| Config | Chunk Size | Penalty |
|--------|-----------|---------|
| EXP-500-NP | 500 | gamma=0.0 |
| EXP-500-P | 500 | gamma=0.1 |
| EXP-1000-NP | 1000 | gamma=0.0 |
| EXP-1000-P | 1000 | gamma=0.1 |
| EXP-2000-NP | 2000 | gamma=0.0 |
| EXP-2000-P | 2000 | gamma=0.1 |

Nota: EXP-2000 so tem dados para EGIS (outros modelos mostram `--`).

---

## 4. Proximos Passos Sugeridos

1. **Resolver inconsistencia 2.1**: Verificar se TABLE VIII (0.898 para EGIS em EXP-500) esta correto ou precisa atualizacao. Se correto, a TABLE VII mostra media cross-config enquanto TABLE VIII mostra EXP-500 only -- documentar essa diferenca.

2. **Atualizar Conclusion e Discussion**: Alinhar todos os valores numericos citados no texto com as tabelas corrigidas.

3. **Decidir sobre tabelas inline vs \input**: Migrar tabelas inline para `\input{}` evitaria futuras inconsistencias.

4. **Verificar CDCMS**: Confirmar se CDCMS deve aparecer no bloco binary da TABLE VII e se os dados existem em `consolidated_results.csv`.

5. **Conferir numeracao de tabelas**: O paper usa numeracao romana automatica (padrao IEEE). Verificar se as referencias cruzadas (`\ref`) apontam para as tabelas corretas apos todas as edicoes.

6. **Verificar undefined references**: O LaTeX reporta "undefined references" -- rodar 2x `pdflatex` ou usar `latexmk`.

---

## 5. Comandos Uteis

```bash
# Regenerar todas as tabelas
python generate_paper_tables.py

# Compilar paper (2 passes para resolver referencias)
cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

# Verificar valores no CSV fonte
python -c "import pandas as pd; df=pd.read_csv('paper_data/consolidated_results.csv'); print(df[df['model']=='EGIS'].groupby('config_label')['gmean_mean'].mean())"
```

---

## 6. Historico de Mudancas Recentes

| Data | Commit | Descricao |
|------|--------|-----------|
| 2026-01-30 | `dd21d12` | Fix TABLE VII: remover ACDWM/CDCMS do bloco multiclass, atualizar valores |
| 2026-01-29 | `496a624` | Fix constant AMS (0.25 placeholder) in transition metrics |
| 2026-01-29 | `9da3cf7` | Adicionar scripts Python e configuracoes YAML |
