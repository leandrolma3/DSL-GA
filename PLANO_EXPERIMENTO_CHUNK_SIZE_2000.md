# PLANO DE EXPERIMENTO - CHUNK SIZE 2000

**Data**: 2025-12-12
**Objetivo**: Criar novos YAMLs para testar todos os modelos com chunk_size dobrado (1000 -> 2000)
**Status**: YAMLS GERADOS - PRONTO PARA EXECUCAO

---

## 1. CONTEXTO

### 1.1 Experimento Anterior (Baseline)
- **chunk_size**: 1000 instancias
- **evaluation_period**: 1000
- **num_chunks**: 6
- **Batches executados**: 1-7 (12 datasets por batch para drift_simulation, 5-6 para standard)
- **Resultados em**: `experiments_6chunks_phase2_gbml/` e `experiments_6chunks_phase3_real/`

### 1.2 Novo Experimento
- **chunk_size**: 2000 instancias (DOBRADO)
- **evaluation_period**: 2000
- **num_chunks**: 6 (MANTIDO - mesmo numero de chunks, mas cada um maior)
- **Resultados em**: `experiments_chunk2000_phase1/` (nova pasta)

---

## 2. PARAMETROS A MODIFICAR

### 2.1 Parametros Obrigatorios (TODOS os YAMLs)

| Parametro | Valor Atual | Novo Valor | Localizacao |
|-----------|-------------|------------|-------------|
| `chunk_size` | 1000 | 2000 | `data_params.chunk_size` |
| `evaluation_period` | 1000 | 2000 | `experiment_settings.evaluation_period` |
| `base_results_dir` | `.../experiments_6chunks_phase2_gbml/batch_X` | `.../experiments_chunk2000/batch_X` | `experiment_settings.base_results_dir` |
| `heatmap_save_directory` | `experiments_6chunks_phase2_gbml/...` | `experiments_chunk2000/heatmaps` | `drift_analysis.heatmap_save_directory` |

### 2.2 Parametros Derivados (Verificar Consistencia)

| Parametro | Valor Atual | Novo Valor | Justificativa |
|-----------|-------------|------------|---------------|
| `max_instances` | 24000 | 24000 | MANTER - permite flexibilidade |
| `num_chunks` | 6 | 6 | MANTER - mesmo numero de chunks |
| `dt_seeding_sample_size_on_init` | 2000 | 2000 | MANTER - independente do chunk_size |
| `historical_reference_size` | 500 | 500 | MANTER - independente |

### 2.3 Parametros que NAO Devem Mudar

Os seguintes parametros devem permanecer IDENTICOS para garantir comparacao justa:
- `ga_params` (population_size, max_generations, etc.)
- `memory_params`
- `fitness_params`
- `parallelism`
- Definicoes de `drift_analysis.datasets`
- Definicoes de `experimental_streams`

---

## 3. LICOES APRENDIDAS DOS EXPERIMENTOS ANTERIORES

### 3.1 Erros Criticos a Evitar

1. **Caminhos incorretos no base_results_dir**
   - ERRADO: `/content/drive/MyDrive/DSL-AG-hybrid/...`
   - CORRETO: `/content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid/...`

2. **Formato incorreto para generators sinteticos**
   - ERRADO: `generator: SEAGenerator`
   - CORRETO: `class: river.datasets.synth.SEA`

3. **Parametros inexistentes em classes customizadas**
   - ERRADO: `balance: false` em AssetNegotiation
   - CORRETO: Apenas `seed` e `classification_function`

4. **Drifts fora da janela experimental**
   - Os `duration_chunks` ja foram corrigidos nos YAMLs atuais
   - Manter as mesmas configuracoes de concept_sequence

### 3.2 Boas Praticas Validadas

- Criar modelo NOVO a cada chunk (nao incremental)
- Usar campo `class:` com path completo para generators
- Testar com 1 dataset antes de rodar batch completo
- Salvar logs detalhados para debug

---

## 4. ESTRUTURA DOS NOVOS YAMLs

### 4.1 Arquivos a Criar

Baseado nos YAMLs existentes, criar os seguintes novos arquivos:

| YAML Original | Novo YAML | Tipo |
|---------------|-----------|------|
| `config_batch_1.yaml` | `config_chunk2000_batch_1.yaml` | drift_simulation (12 datasets) |
| `config_batch_2.yaml` | `config_chunk2000_batch_2.yaml` | drift_simulation |
| `config_batch_3.yaml` | `config_chunk2000_batch_3.yaml` | drift_simulation |
| `config_batch_4.yaml` | `config_chunk2000_batch_4.yaml` | drift_simulation |
| `config_batch_5.yaml` | `config_chunk2000_batch_5.yaml` | standard (datasets reais) |
| `config_batch_6.yaml` | `config_chunk2000_batch_6.yaml` | standard (sinteticos estacionarios) |
| `config_batch_7.yaml` | `config_chunk2000_batch_7.yaml` | standard (sinteticos estacionarios) |

### 4.2 Estrutura de Diretorios de Resultados

```
experiments_chunk2000/
  batch_1/
    SEA_Abrupt_Simple/
    SEA_Abrupt_Chain/
    ...
  batch_2/
  batch_3/
  batch_4/
  batch_5/
  batch_6/
  batch_7/
  heatmaps/
    concept_heatmaps/
```

---

## 5. CONSIDERACOES SOBRE DRIFT COM CHUNK_SIZE MAIOR

### 5.1 Impacto nos Drifts

Com chunk_size=2000 vs 1000:
- Cada chunk contem 2x mais instancias
- Os drifts ocorrerao nas MESMAS posicoes relativas (mesmo `duration_chunks`)
- Exemplo: SEA_Abrupt_Simple com duration_chunks=[3,3]
  - chunk_size=1000: drift em 3000 instancias
  - chunk_size=2000: drift em 6000 instancias

### 5.2 Hipoteses a Testar

1. **Modelos terao mais dados por chunk** -> Possivelmente melhor performance geral
2. **Menos avaliacoes** (6 chunks vs 6 chunks, mas cada avaliacao com mais dados)
3. **Drift detection pode ser diferente** -> Modelos podem ter mais tempo para adaptar

---

## 6. CHECKLIST DE VALIDACAO

### 6.1 Antes de Criar os YAMLs

- [x] Entender estrutura dos YAMLs existentes
- [x] Identificar parametros a modificar
- [x] Revisar erros anteriores
- [ ] Confirmar caminho base para resultados

### 6.2 Ao Criar Cada YAML

- [ ] Modificar `chunk_size: 2000`
- [ ] Modificar `evaluation_period: 2000`
- [ ] Modificar `base_results_dir` para nova pasta
- [ ] Modificar `heatmap_save_directory` para nova pasta
- [ ] Manter TODOS os outros parametros identicos
- [ ] Verificar formato correto de `drift_analysis.datasets`

### 6.3 Apos Criar os YAMLs

- [ ] Validar sintaxe YAML de todos os arquivos
- [ ] Comparar parametros criticos com originais
- [ ] Testar execucao com 1 dataset antes de rodar batch completo

---

## 7. PLANO DE EXECUCAO

### Fase 1: Criacao dos YAMLs (Atual)
1. Criar script Python para gerar YAMLs automaticamente
2. Gerar todos os 7 YAMLs de uma vez
3. Validar sintaxe e conteudo

### Fase 2: Validacao Local
1. Testar 1 dataset (ex: SEA_Abrupt_Simple) localmente
2. Verificar se resultados sao salvos no diretorio correto
3. Confirmar metricas estao sendo calculadas

### Fase 3: Execucao no Colab
1. Upload dos novos YAMLs para Google Drive
2. Executar batches sequencialmente
3. Monitorar logs e resultados

### Fase 4: Analise Comparativa
1. Comparar resultados chunk_size=2000 vs chunk_size=1000
2. Analisar impacto no desempenho dos modelos
3. Documentar descobertas

---

## 8. PROXIMOS PASSOS IMEDIATOS

1. **CRIAR** script Python para gerar os 7 YAMLs automaticamente
2. **EXECUTAR** script e gerar os arquivos
3. **VALIDAR** os arquivos gerados
4. **DOCUMENTAR** qualquer ajuste necessario

---

## 9. ARQUIVOS GERADOS (2025-12-12)

### 9.1 YAMLs Criados

| Arquivo | Tamanho | Tipo | Datasets |
|---------|---------|------|----------|
| `config_chunk2000_batch_1.yaml` | 22 KB | drift_simulation | 12 datasets abrupt |
| `config_chunk2000_batch_2.yaml` | 10 KB | drift_simulation | 9 datasets gradual |
| `config_chunk2000_batch_3.yaml` | 11 KB | drift_simulation | mixed |
| `config_chunk2000_batch_4.yaml` | 9 KB | drift_simulation | noise |
| `config_chunk2000_batch_5.yaml` | 3 KB | standard | 5 datasets reais |
| `config_chunk2000_batch_6.yaml` | 4 KB | standard | 6 sinteticos estacionarios |
| `config_chunk2000_batch_7.yaml` | 4 KB | standard | 6 sinteticos estacionarios |

### 9.2 Validacao das Modificacoes

Todos os YAMLs foram verificados com as seguintes mudancas:

| Parametro | Valor Original | Novo Valor |
|-----------|----------------|------------|
| `chunk_size` | 1000 | 2000 |
| `evaluation_period` | 1000 | 2000 |
| `base_results_dir` | `experiments_6chunks_phase2_gbml` | `experiments_chunk2000_phase1` |
| `base_results_dir` | `experiments_6chunks_phase3_real` | `experiments_chunk2000_phase2` |
| `heatmap_save_directory` | (atualizado conforme acima) | (atualizado conforme acima) |

### 9.3 Parametros Mantidos Identicos (Comparacao Justa)

- `ga_params` (population_size=120, max_generations=200, etc.)
- `memory_params`
- `fitness_params`
- `parallelism`
- `num_chunks` = 6
- `max_instances` = 24000
- Todas as definicoes de `drift_analysis.datasets`
- Todas as definicoes de `experimental_streams`

---

## 10. COMANDOS PARA EXECUCAO

### No Colab (exemplo para batch 1):

```python
# Celula 1: Montar Drive
from google.colab import drive
drive.mount('/content/drive')

# Celula 2: Ir para diretorio do projeto
%cd /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid

# Celula 3: Executar experimento
!python main.py --config configs/config_chunk2000_batch_1.yaml
```

### Sequencia recomendada:

1. Executar batch 5 primeiro (datasets reais, mais rapido para validar)
2. Depois batches 6 e 7 (sinteticos estacionarios)
3. Por ultimo batches 1-4 (drift simulation, mais demorados)

---

**Autor**: Claude Code
**Versao**: 2.0
**Status**: YAMLS GERADOS - PRONTO PARA EXECUCAO
**Script de geracao**: `generate_chunk2000_configs.py`
