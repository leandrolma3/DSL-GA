# Investigacao de Logs - EXP-B (chunk2000)

## Data: 2024-12-22

## Resumo do Problema

Tres datasets no EXP-B (TABLE V do paper) nao possuem resultados GBML (EGIS):
- PokerHand
- RANDOMTREE_Gradual_Noise
- WAVEFORM_Gradual_Simple

## Analise Detalhada

### 1. PokerHand

**Status:** GBML NUNCA FOI EXECUTADO no EXP-B

**Evidencias:**
- NAO aparece em nenhum log `chunk2000*.log`
- O diretorio `experiments_chunk2000_phase2/batch_5/PokerHand/` existe, mas contem apenas:
  - Dados de chunks (chunk_0_train.csv, etc.)
  - Resultados dos modelos comparativos (ROSE, HAT, ARF, SRP, ERulesD2S)
  - best_individual_trained_on_chunk_0.pkl e chunk_1.pkl (parciais)
- NAO possui `chunk_metrics.json` nem `RulesHistory*.json`

**Causa Raiz:**
- A configuracao `config_chunk2000_batch_1.yaml` tem `run_mode: drift_simulation`, que executa apenas experimentos sinteticos
- PokerHand esta listado em `standard_experiments`, NAO em `drift_simulation_experiments`
- O batch_5 (que deveria executar PokerHand) foi parcialmente executado - apenas IntelLabSensors completou

**Observacao:**
- PokerHand FOI executado com sucesso no EXP-C (balanced) - log `config_balanced_batch_6.log` mostra execucao completa em 26450.81 segundos

### 2. RANDOMTREE_Gradual_Noise

**Status:** GBML INICIOU MAS FOI INTERROMPIDO

**Evidencias:**
- Encontrado em `chunk2000_batch_3_full2.log` linhas 54443-62924
- Comecou: 2025-12-13 15:24:10
- Ultima entrada: 2025-12-13 18:55:26 (Gen 20 do Chunk 2)
- Total de 4 chunks gerados

**Progresso:**
- Chunk 0: Treinado e testado (Test G-mean: 0.7138)
- Chunk 1: Treinado e testado (Test G-mean: 0.6187, MILD DRIFT detectado)
- Chunk 2: Iniciou treinamento (Gen 20/200)
- Chunk 3: NAO processado

**Causa Raiz:**
- Sessao Colab expirou durante processamento
- Log termina abruptamente na linha 62924 (wc -l confirma 62924 linhas)

### 3. WAVEFORM_Gradual_Simple

**Status:** GBML INICIOU MAS FOI INTERROMPIDO

**Evidencias:**
- Encontrado em `chunk2000_batch_4_full.log` linhas 116357-124057
- Comecou: 2025-12-14 09:07:46
- Ultima entrada: 2025-12-14 12:08:33 (Gen 7 do Chunk 1)
- Total de 4 chunks gerados

**Progresso:**
- Chunk 0: Treinado e testado (Test G-mean: 0.5932)
- Chunk 1: Iniciou treinamento (Gen 7/25, modo recovery ativado)
- Chunks 2-3: NAO processados

**Causa Raiz:**
- Sessao Colab expirou durante processamento
- Log termina abruptamente na linha 124057 (wc -l confirma 124056 linhas)
- Erros de "Parallel eval failed" ocorreram antes (linhas 58983, 85888, 101247, 116298)

## Datasets com Baixo Desempenho

### 4. CovType (G-Mean: 0.360)

**Causa:** Concept drift severo
- Treino: Classes 2 (41%), 5 (26%), 1 (24%) dominam
- Teste: Classes 4, 6 dominam (pouco representadas no treino)
- Resultado: Modelo nao consegue generalizar para distribuicao diferente

### 5. Shuttle (G-Mean: 0.199)

**Causa:** Novas classes no stream
- Chunk 0: Test G-mean = 0.995 (excelente)
- Chunks 1-4: Test G-mean = 0.0 (colapso total)
- Classes 6 e 7 aparecem nos chunks de teste mas NAO estavam no treino
- EGIS (e qualquer modelo) nao pode prever classes nunca vistas

### 6. IntelLabSensors (G-Mean: 0.0)

**Causa:** Todos os modelos falharam
- ARF, SRP, ROSE, HAT, GBML: todos G-mean = 0.0
- Problema inerente ao dataset, nao especifico do EGIS

## Conclusoes

### Problemas de Infraestrutura (3 datasets):
1. **PokerHand**: Nunca foi executado no EXP-B
2. **RANDOMTREE_Gradual_Noise**: Sessao Colab expirou (chunk 2/4)
3. **WAVEFORM_Gradual_Simple**: Sessao Colab expirou (chunk 1/4)

### Problemas de Dados (2 datasets):
4. **CovType**: Concept drift severo entre treino/teste
5. **Shuttle**: Novas classes aparecem no stream

### Recomendacoes

1. **Re-executar GBML** para os 3 datasets incompletos:
   - PokerHand (do zero)
   - RANDOMTREE_Gradual_Noise (do zero)
   - WAVEFORM_Gradual_Simple (do zero)

2. **Adicionar nota no paper** explicando que:
   - CovType e Shuttle tem concept drift severo com mudanca de classes
   - Isso e uma limitacao conhecida de metodos baseados em regras

3. **Configuracao Colab**:
   - Usar sessoes mais longas ou checkpointing
   - Salvar chunk_metrics.json incrementalmente

## Arquivos de Log Analisados

| Log | Tamanho (linhas) | Conteudo |
|-----|-----------------|----------|
| chunk2000_batch_3_full2.log | 62,924 | RANDOMTREE_Gradual_Noise (interrompido) |
| chunk2000_batch_4_full.log | 124,056 | WAVEFORM_Gradual_Simple (interrompido) |
| chunk2000_batch_5_rerun.log | 2,575 | IntelLabSensors (completo) |
| chunk2000_batch_1_full.log | N/A | Nao contem PokerHand |
| config_balanced_batch_6.log | N/A | PokerHand EXP-C (completo) |
