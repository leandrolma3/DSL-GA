# Status do Artigo IEEE TKDE - EGIS

**Ultima atualizacao**: 2026-01-30
**Branch**: main
**Ultimo commit**: `26b18df` - Add paper status documentation

---

## 0. CHECKPOINT DE SESSAO (30/01/2026)

### O que foi feito nesta sessao
1. **Corrigida TABLE VII** (`main.tex` linhas 619-647): removidos ACDWM e CDCMS do bloco "All datasets (n=48)" — sao modelos binary-only
2. **Atualizados valores EGIS** no bloco binary da TABLE VII: de 0.898/0.893 para **0.858/0.868** (dados reais do CSV)
3. **Atualizado texto** nos paragrafos antes e depois da TABLE VII para refletir novos valores
4. **`generate_paper_tables.py`**: corrigido `MULTICLASS_MODELS` (ROSE_Original em vez de ROSE_ChunkEval); segundo bloco agora calcula media sobre 48 datasets (nao apenas 6 multiclass)
5. **Regeneradas todas as tabelas** em `paper/tables/*.tex`
6. **Paper compilado** com sucesso (22 paginas, sem erros)

### O que ficou pendente (FAZER NA PROXIMA SESSAO)

**PRIORIDADE 1 — Inconsistencias numericas no texto do paper:**

A TABLE VIII (drift type, inline, linha 675) e varios paragrafos do paper ainda usam o valor antigo **0.898** para EGIS. O valor real no CSV para EGIS EXP-500-NP binary e **0.858**. Linhas afetadas:
- **Linha 675**: TABLE VIII Overall = 0.898 (deveria ser ~0.858)
- **Linha 669-674**: valores por drift type na TABLE VIII (todos possivelmente errados)
- **Linha 680**: texto "EGIS achieves G-Mean of 0.898 overall"
- **Linha 914**: Discussion "ensemble methods achieve higher G-Mean (0.88-0.91) than EGIS (0.80)"
- **Linha 918**: "EGIS maintains consistent performance (G-Mean = 0.80-0.81)"
- **Linha 920**: "EGIS outperforms ERulesD2S by 22.5 percentage points (0.803 vs 0.578)"
- **Linha 981**: Conclusion "average G-Mean of 0.80-0.81"
- **Linha 981**: "0.803 vs 0.578" e "22.5 percentage points"
- **Linha 985**: Conclusion RQ3 "G-Mean of 0.898 overall"

**PRIORIDADE 2 — CDCMS nao existe no CSV:**

CDCMS **nao aparece** em `paper_data/consolidated_results.csv`. No entanto o paper o referencia em:
- TABLE VIII (linha 667): coluna CDCMS com valores
- TABLE VII bloco binary (removido nesta sessao, mas texto ao redor pode mencionar)
- Linha 651: "CDCMS (31-10)" em referencia ao WLD
- TABLE de Wilcoxon inline (linha 780-797): nao mencionado mas verificar

Opcoes: (a) re-coletar dados CDCMS e adicionar ao CSV, ou (b) remover CDCMS de todo o paper.

**PRIORIDADE 3 — Valores corretos para usar (do CSV real):**

```
EGIS binary (42 datasets):
  EXP-500-NP:  0.858 ± 0.123
  EXP-500-P:   0.858 ± 0.121
  EXP-1000-NP: 0.868 ± 0.112
  EXP-1000-P:  0.869 ± 0.112
  EXP-2000-NP: 0.905 ± 0.082 (apenas 28 datasets)

EGIS EXP-500-NP binary, por drift_type:
  abrupt:     0.860 (n=14)
  gradual:    0.855 (n=9)
  noisy:      0.845 (n=8)
  real:       0.890 (n=4)
  stationary: 0.855 (n=7)
  Overall:    0.858

Modelos no CSV: ACDWM, ARF, EGIS, ERulesD2S, HAT, ROSE_ChunkEval, ROSE_Original, SRP
Modelos AUSENTES do CSV: CDCMS
Configs disponiveis: EXP-500-NP, EXP-500-P, EXP-1000-NP, EXP-1000-P, EXP-2000-NP
```

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

**CONFIRMADO**: O valor 0.898 esta INCORRETO. O CSV mostra EGIS EXP-500-NP binary overall = **0.858**. A TABLE VIII inteira precisa ser atualizada. Os valores por drift type tambem estao errados (ver checkpoint acima para valores corretos).

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

**CONFIRMADO**: CDCMS **nao existe** em `consolidated_results.csv`. Nenhuma variante de nome (CDC, cdc, CDCMS) foi encontrada. Os valores de CDCMS que estavam no paper (0.843, etc.) foram provavelmente inseridos manualmente de outra fonte. Decisao necessaria: re-coletar dados CDCMS ou remover do paper inteiro.

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
| 2026-01-30 | `26b18df` | Add paper status documentation |
| 2026-01-30 | `dd21d12` | Fix TABLE VII: remover ACDWM/CDCMS do bloco multiclass, atualizar valores |
| 2026-01-29 | `496a624` | Fix constant AMS (0.25 placeholder) in transition metrics |
| 2026-01-29 | `9da3cf7` | Adicionar scripts Python e configuracoes YAML |

---

## 7. Resumo para Prompt de Retomada

Ao retomar o trabalho com Claude Code, use o seguinte prompt para contexto:

> Estou trabalhando no artigo IEEE TKDE sobre EGIS (paper/main.tex).
> Leia o arquivo STATUS_ARTIGO_IEEE.md para entender o estado atual,
> as inconsistencias pendentes e as regras de ouro. A principal tarefa
> pendente e: (1) atualizar TABLE VIII e todos os textos do paper que
> referenciam valores antigos de EGIS (0.898 deve ser 0.858 para EXP-500),
> (2) decidir o que fazer com CDCMS (ausente do CSV mas citado no paper),
> (3) alinhar Discussion e Conclusion com os dados corretos.
