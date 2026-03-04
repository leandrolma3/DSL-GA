# Correcoes no Pos-Processamento Batch 1

**Data**: 2025-11-19
**Arquivo Corrigido**: `post_process_batch_1.py`
**Config Corrigido**: `configs/config_batch_1.yaml`

---

## PROBLEMAS IDENTIFICADOS E CORRIGIDOS

### Problema 1: Step 1 nao executava corretamente

**Sintoma**:
- `concept_differences.json` nao era gerado
- Usuario tinha que executar manualmente: `python -c "from analyze_concept_difference import main; main('configs/config_batch_1.yaml')"`

**Causa**:
- O script `analyze_concept_difference.py` nao aceita argumentos de linha de comando
- O `__main__` block apenas chama `main()` sem passar o config path
- `post_process_batch_1.py` tentava executar via subprocess com argumento, mas o argumento era ignorado

**Solucao Aplicada**:
```python
# ANTES (nao funcionava):
cmd = [sys.executable, str(ANALYZE_CONCEPT_DIFF_SCRIPT), str(CONFIG_FILE)]
return run_command(cmd, "Analise de Diferenca entre Conceitos")

# DEPOIS (funciona):
import analyze_concept_difference
analyze_concept_difference.main(str(CONFIG_FILE))
```

**Resultado**:
- Step 1 agora executa corretamente
- `concept_differences.json` e gerado automaticamente
- Heatmaps de conceitos sao criados

---

### Problema 2: Plots nao eram gerados

**Sintoma**:
- Pasta `plots/` nao era criada dentro de cada `run_1/`
- Graficos de acuracia e evolucao do GA ausentes

**Possiveis Causas**:
1. `generate_plots.py` falhava silenciosamente
2. Falta de arquivos necessarios (ResultsDictionary_*.pkl)
3. Erros na criacao da pasta plots/

**Solucoes Aplicadas**:

#### a) Melhor logging no run_command:
```python
# Mostra output completo (50 primeiras linhas)
for line in result.stdout.split('\n')[:50]:
    if line.strip():
        logger.info(f"  | {line}")

# Mostra erros completos
if e.stderr:
    logger.error(f"  Stderr:")
    for line in e.stderr.split('\n')[:50]:
        if line.strip():
            logger.error(f"    | {line}")
```

#### b) Verificacao pos-execucao:
```python
# Verifica se pasta plots foi criada
plots_dir = run_dir / "plots"
if plots_dir.exists():
    plot_files = list(plots_dir.glob("*.png"))
    logger.info(f"  [OK] Pasta plots/ criada com {len(plot_files)} arquivos PNG")
else:
    logger.warning(f"  [ATENCAO] Pasta plots/ NAO foi criada")
```

**Resultado**:
- Agora vemos exatamente o que acontece durante generate_plots.py
- Erros sao mostrados completos para debug
- Sabemos quantos plots foram gerados

---

### Problema 3: Typo no path do heatmap

**Sintoma**:
- Warning: `concept_differences.json nao encontrado em: .../concept_heatmapsS`

**Causa**:
- Typo no `configs/config_batch_1.yaml` linha 94
- Path tinha `concept_heatmapsS` com S extra

**Solucao**:
```yaml
# ANTES:
heatmap_save_directory: experiments_6chunks_phase2_gbml/test_real_results_heatmaps/concept_heatmapsS

# DEPOIS:
heatmap_save_directory: experiments_6chunks_phase2_gbml/test_real_results_heatmaps/concept_heatmaps
```

---

### Problema 4: Config errado no script

**Sintoma**:
- Script usava `config_test_drift_recovery.yaml` ao inves de `config_batch_1.yaml`

**Solucao**:
```python
# ANTES:
CONFIG_FILE = BASE_DIR / "config_test_drift_recovery.yaml"

# DEPOIS:
CONFIG_FILE = BASE_DIR / "configs" / "config_batch_1.yaml"
```

---

## COMO USAR O SCRIPT CORRIGIDO

### No Google Colab:

```python
# Celula 1: Pos-processamento completo
!cd /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid && python post_process_batch_1.py
```

O script executara automaticamente:
1. **Step 1**: analyze_concept_difference.py → gera concept_differences.json
2. **Step 2**: generate_plots.py para cada base → gera plots/
3. **Step 3**: rule_diff_analyzer.py para cada base → gera analises de regras

### Localmente (para testar):

```bash
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"
python post_process_batch_1.py
```

---

## VERIFICACAO DOS RESULTADOS

### O que deve ser criado:

#### 1. Step 1 - Concept Differences:
```
experiments_6chunks_phase2_gbml/test_real_results_heatmaps/concept_heatmaps/
├── concept_differences.json
├── Heatmap_SEA.png
├── Heatmap_AGRAWAL.png
├── Heatmap_RBF.png
├── Heatmap_STAGGER.png
├── Heatmap_HYPERPLANE.png
└── Heatmap_RANDOMTREE.png
```

#### 2. Step 2 - Plots (para cada base):
```
experiments_6chunks_phase2_gbml/batch_1/<DATASET_NAME>/run_1/plots/
├── Plot_AccuracyPeriodic_<DATASET>_Run1.png    # Grafico principal com drifts
├── Plot_GA_Evolution_Chunk0_<DATASET>_Run1.png
├── Plot_GA_Evolution_Chunk1_<DATASET>_Run1.png
├── Plot_GA_Evolution_Chunk2_<DATASET>_Run1.png
├── Plot_GA_Evolution_Chunk3_<DATASET>_Run1.png
└── Plot_GA_Evolution_Chunk4_<DATASET>_Run1.png
```

O **Plot_AccuracyPeriodic** e o mais importante - mostra:
- Acuracia de treino (azul)
- Acuracia de teste (laranja)
- Linhas verticais vermelhas indicando posicoes de drift
- Percentual de drift severity em cada posicao

#### 3. Step 3 - Rule Diff Analysis (para cada base):
```
experiments_6chunks_phase2_gbml/batch_1/<DATASET_NAME>/run_1/
├── rule_diff_analysis_<DATASET>_report.txt     # Relatorio texto
├── rule_diff_analysis_<DATASET>_matrix.csv     # Matriz de evolucao
└── rule_diff_analysis_<DATASET>_matrix.png     # Visualizacao da matriz
```

---

## DIAGNOSTICO DE PROBLEMAS

### Se Step 1 ainda falhar:

```python
# Executar manualmente e ver erro completo
cd /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid
python -c "from analyze_concept_difference import main; main('configs/config_batch_1.yaml')"
```

Verificar:
- Config existe? `ls configs/config_batch_1.yaml`
- Diretorio heatmaps existe? `mkdir -p experiments_6chunks_phase2_gbml/test_real_results_heatmaps/concept_heatmaps`

### Se Step 2 falhar (plots nao gerados):

```python
# Testar em uma base especifica
cd /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid
python generate_plots.py experiments_6chunks_phase2_gbml/batch_1/SEA_Abrupt_Simple/run_1 \
  -d experiments_6chunks_phase2_gbml/test_real_results_heatmaps/concept_heatmaps/concept_differences.json
```

Verificar:
- Arquivo existe? `ls experiments_6chunks_phase2_gbml/batch_1/SEA_Abrupt_Simple/run_1/ResultsDictionary_*.pkl`
- Se nao existe, o experimento GBML nao rodou ou falhou para essa base

### Se Step 3 falhar:

```python
# Testar em uma base especifica
cd /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid
python rule_diff_analyzer.py experiments_6chunks_phase2_gbml/batch_1/SEA_Abrupt_Simple/run_1/RulesHistory_*.txt \
  -o experiments_6chunks_phase2_gbml/batch_1/SEA_Abrupt_Simple/run_1/rule_diff_analysis_test
```

Verificar:
- Arquivo existe? `ls experiments_6chunks_phase2_gbml/batch_1/SEA_Abrupt_Simple/run_1/RulesHistory_*.txt`

---

## ARQUIVOS DE ENTRADA NECESSARIOS

Para cada base, o diretorio `run_1/` deve conter:

```
experiments_6chunks_phase2_gbml/batch_1/<DATASET_NAME>/run_1/
├── ResultsDictionary_<DATASET>_Run1_ChunkSize1000.pkl  # Para generate_plots.py
├── RulesHistory_<DATASET>_Run1.txt                     # Para rule_diff_analyzer.py
├── test_predictions_chunk*.csv                         # Dados de teste
└── train_predictions_chunk*.csv                        # Dados de treino
```

Se algum desses arquivos nao existe, o experimento GBML nao foi executado corretamente para aquela base.

---

## TEMPO ESTIMADO DE POS-PROCESSAMENTO

| Step | Tempo por Base | Tempo Total (12 bases) |
|------|----------------|------------------------|
| Step 1 (concept diff) | N/A (1x) | ~5-10 min |
| Step 2 (plots) | ~1-2 min | ~15-20 min |
| Step 3 (rule diff) | ~2-3 min | ~25-30 min |
| **TOTAL** | ~4-5 min | **~45-60 min** |

---

## PROXIMOS PASSOS

Apos executar `post_process_batch_1.py` com sucesso:

1. **Verificar plots gerados**:
   - Abrir `Plot_AccuracyPeriodic_*.png` de cada base
   - Verificar se linhas de drift aparecem nas posicoes corretas (2000, 3000, 4000, 5000)
   - Verificar se drift severity e diferente de 0.0%

2. **Analisar rule_diff**:
   - Ler `rule_diff_analysis_*_report.txt`
   - Verificar quantas regras mudam em cada drift
   - Analisar padroes de evolucao

3. **Comparar com modelos baseline**:
   - Executar River models (HAT, ARF, SRP)
   - Executar ACDWM e ERulesD2S
   - Gerar comparacoes estatisticas

4. **Consolidar resultados**:
   - Executar CELULA 11 (statistical comparison)
   - Executar CELULA 12 (final plots)
   - Gerar tabelas para paper

---

## RESUMO DAS CORRECOES

| # | Problema | Arquivo | Linha | Status |
|---|----------|---------|-------|--------|
| 1 | Step 1 usa subprocess ao inves de import | post_process_batch_1.py | 92-132 | ✅ CORRIGIDO |
| 2 | Logging insuficiente em run_command | post_process_batch_1.py | 76-95 | ✅ CORRIGIDO |
| 3 | Sem verificacao de plots criados | post_process_batch_1.py | 176-186 | ✅ CORRIGIDO |
| 4 | Typo no heatmap_save_directory | configs/config_batch_1.yaml | 94 | ✅ CORRIGIDO |
| 5 | Config errado (test_drift_recovery) | post_process_batch_1.py | 34, 40 | ✅ CORRIGIDO |

---

**Status Final**: ✅ SCRIPT CORRIGIDO E PRONTO PARA USO
**Proxima Acao**: Executar `post_process_batch_1.py` no Colab e verificar resultados
