# Correção TABLE VII - Bloco Multiclass

**Data**: 2026-01-30

## Problema

O segundo bloco da TABLE VII no paper (`main.tex`) dizia "All datasets (n=48)" mas incluía ACDWM e CDCMS — modelos que **não suportam multiclass**. Isso era metodologicamente incorreto, pois as 6 bases multiclass (LED e WAVEFORM) não podem ser avaliadas por esses modelos.

Além disso, os valores hardcoded no `main.tex` estavam desatualizados em relação aos dados reais em `consolidated_results.csv`.

## Correções realizadas

### 1. `generate_paper_tables.py`

- **`MULTICLASS_MODELS`** (linha 173): corrigido de `ROSE_ChunkEval` para `ROSE_Original`
- **Segundo bloco**: mudado de "Multiclass only (n=6)" (média apenas sobre 6 bases multiclass) para "All datasets (n=48)" (média sobre todas as 48 bases), usando apenas os 6 modelos que suportam multiclass:
  - EGIS, ARF, SRP, HAT, ROSE, ERulesD2S
  - **Excluídos**: ACDWM (binary-only), CDCMS (binary-only)

### 2. `paper/main.tex`

- **Tabela inline** (linhas 619-650): atualizada com valores corretos do script
  - Bloco "Binary only (n=42)": removido CDCMS (não estava nos dados), valores EGIS atualizados (0.858/0.868)
  - Bloco "All datasets (n=48)": agora contém apenas 6 modelos (sem ACDWM, sem CDCMS)
- **Caption**: atualizada para explicar que ACDWM e CDCMS são excluídos do segundo bloco por serem binary-only
- **Texto** (parágrafos antes e depois da tabela): valores numéricos atualizados para refletir os dados corretos

### 3. Tabelas regeneradas

Executado `python generate_paper_tables.py`, regenerando todos os `.tex` em `paper/tables/`.

### 4. Paper recompilado

`pdflatex` executado 2x sem erros. Output: `paper/main.pdf` (22 páginas).

## Valores atualizados (EXP-500 / EXP-1000)

### Binary only (n=42) — todos os 8 modelos

| Modelo    | EXP-500 | EXP-1000 |
|-----------|---------|----------|
| ROSE      | 0.894   | 0.894    |
| ARF       | 0.879   | 0.880    |
| SRP       | 0.871   | 0.864    |
| ACDWM     | 0.860   | 0.818    |
| HAT       | 0.817   | 0.821    |
| **EGIS**  | 0.858   | 0.868    |
| ERulesD2S | 0.597   | 0.592    |

### All datasets (n=48) — apenas 6 modelos multiclass-capable

| Modelo    | EXP-500 | EXP-1000 |
|-----------|---------|----------|
| ARF       | 0.862   | 0.873    |
| **EGIS**  | 0.856   | 0.866    |
| ROSE      | 0.855   | 0.855    |
| SRP       | 0.846   | 0.848    |
| HAT       | 0.767   | 0.770    |
| ERulesD2S | 0.562   | 0.556    |

## Datasets excluídos de todas as análises

CovType, IntelLabSensors, PokerHand, Shuttle (definidos em `EXCLUDED_DATASETS`).

## Arquivos modificados

- `generate_paper_tables.py`
- `paper/main.tex`
- `paper/main.pdf`
- `paper/tables/*.tex` (regenerados)
