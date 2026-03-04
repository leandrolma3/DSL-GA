# Documentacao de Atividades - Janeiro 2026

## Projeto: DSL-AG-hybrid (EGIS)
## Periodo: Janeiro 2026

---

## 1. Contexto Inicial

O projeto envolve experimentos de aprendizado de maquina em fluxos de dados (data streams) com deteccao de concept drift. Os experimentos utilizam o algoritmo EGIS com chunks pre-gerados de diferentes tamanhos (500, 1000, 2000 instancias).

### Estrutura de Experimentos Planejada
- **chunk_500**: 3 batches, 52 datasets, com e sem penalidade
- **chunk_1000**: 4 batches, 52 datasets, com e sem penalidade
- **chunk_2000**: 7 batches, 52 datasets, com e sem penalidade

---

## 2. Problemas Identificados e Solucoes Implementadas

### 2.1 Problema: Experimentos Interrompidos Perdiam Progresso

**Descricao:** Quando sessoes do Google Colab eram interrompidas (timeout de 24h), os experimentos reiniciavam do zero, perdendo todo o progresso.

**Solucao Implementada:** Sistema de resume em dois niveis no `main.py`:

#### Nivel 1 - Skip de Datasets Completos (linhas 1934-1959)
```python
# Verifica chunk_metrics.json
# Se tiver >= expected_chunks, faz SKIP do dataset
resume_enabled_main = exp_settings.get('resume_from_checkpoint', False)
if resume_enabled_main:
    if os.path.exists(metrics_file_check):
        if len(existing_metrics_check) >= expected_chunks_main:
            logger.info(f"[RESUME] SKIPPING '{stream_name}': Already complete")
            continue
```

#### Nivel 2 - Resume de Checkpoints Parciais (linhas 581-646)
```python
# Detecta checkpoints existentes (best_individual_trained_on_chunk_*.pkl)
# Determina start_chunk como ultimo_chunk + 1
# Carrega ultimo individuo para seeding
# Carrega metricas parciais de chunk_metrics_partial.json
```

#### Salvamento Incremental de Metricas (apos linha 1206)
```python
# Salva metricas parciais apos cada chunk processado
partial_metrics_path = os.path.join(run_results_dir, "chunk_metrics_partial.json")
with open(partial_metrics_path, 'w') as f_partial:
    json.dump(make_json_serializable(all_performance_metrics), f_partial, indent=2)
```

**Arquivos Modificados:**
- `main.py`: ~60 linhas de codigo adicionadas

**Configs Atualizados:**
- Todos os 28 arquivos YAML em `configs/` foram atualizados com:
```yaml
experiment_settings:
  resume_from_checkpoint: true
```

---

### 2.2 Problema: Metricas Perdidas em Experimentos Retomados

**Descricao:** Experimentos que foram retomados antes da implementacao do salvamento incremental tinham checkpoints completos mas metricas incompletas (ex: IntelLabSensors tinha 23 checkpoints mas apenas 3-4 metricas).

**Solucao Implementada:** Script `rebuild_metrics.py` para reconstruir metricas a partir dos checkpoints.

```python
def rebuild_metrics_for_experiment(experiment_dir, chunks_dir):
    """Reconstroi metricas a partir dos checkpoints .pkl"""
    # Carrega checkpoints
    # Carrega chunks pre-gerados
    # Recalcula train_gmean, test_gmean, test_f1
    # Salva em chunk_metrics.json ou chunk_metrics_partial.json
```

**Experimentos Reconstruidos:**
1. chunk_500/batch_3/IntelLabSensors
2. chunk_500_penalty/batch_3/IntelLabSensors
3. chunk_500/batch_2/WAVEFORM_Abrupt_Simple
4. chunk_500/batch_3/AGRAWAL_Stationary
5. chunk_500_penalty/batch_1/SEA_Gradual_Simple_Fast
6. chunk_500_penalty/batch_2/RBF_Gradual_Severe_Noise
7. chunk_500/batch_3/RBF_Stationary (parcial)
8. chunk_500_penalty/batch_1/RBF_Gradual_Severe (parcial)
9. chunk_500_penalty/batch_2/RANDOMTREE_Gradual_Noise (parcial)
10. chunk_500_penalty/batch_3/RBF_Stationary (parcial)

---

### 2.3 Problema: Erro na Lista de Datasets por Batch

**Descricao:** O dataset `SEA_Abrupt_Chain_Noise` estava incorretamente listado no batch_1, quando na verdade pertence ao batch_2.

**Correcao:** Listas de datasets corrigidas nos scripts de verificacao:

```python
# CORRETO
batch1 = ['SEA_Abrupt_Simple', 'SEA_Abrupt_Chain', 'SEA_Abrupt_Recurring', ...]
batch2 = ['SEA_Abrupt_Chain_Noise', 'AGRAWAL_Abrupt_Simple_Severe_Noise', ...]
```

---

## 3. Scripts de Monitoramento Criados

### 3.1 check_progress.py
Script para verificar progresso dos experimentos, contando:
- Experimentos completos
- Experimentos parciais
- Experimentos nao iniciados

### 3.2 rebuild_metrics.py
Script para reconstruir metricas perdidas a partir de checkpoints .pkl.

### 3.3 Script de Verificacao Inline
Codigo Python para verificacao rapida de progresso (executado via linha de comando).

---

## 4. Evolucao do Progresso dos Experimentos chunk_500

### Levantamento 1 (31/12/2025)
- Completos: 69
- Parciais: 10
- Progresso: ~65%

### Levantamento 2 (31/12/2025 - apos rebuild_metrics)
- Completos: 75
- Parciais: 4
- Nao iniciados: 27
- Progresso: 70.8%

### Levantamento 3 (01/01/2026)
- Completos: 83
- Parciais: 3
- Nao iniciados: 20
- Progresso: 78.3%

### Levantamento 4 (03/01/2026)
- Completos: 91
- Parciais: 3
- Nao iniciados: 10
- Progresso: 87.5%

### Levantamento 5 (05/01/2026)
- Completos: 101
- Parciais: 2
- Nao iniciados: 1
- Progresso: 97.1%

### Levantamento Final (06/01/2026)
- Completos: 104
- Parciais: 0
- Nao iniciados: 0
- **Progresso: 100%**

---

## 5. Conclusao dos Experimentos chunk_500

### Status Final por Batch

| Experimento | batch_1 | batch_2 | batch_3 | Total |
|-------------|---------|---------|---------|-------|
| chunk_500 | 18/18 | 17/17 | 17/17 | 52/52 |
| chunk_500_penalty | 18/18 | 17/17 | 17/17 | 52/52 |
| **Total** | 36/36 | 34/34 | 34/34 | **104/104** |

---

## 6. Preparacao para Experimentos chunk_1000

### Verificacao de Seguranca Realizada (06/01/2026)

#### Configs Verificados (8 arquivos)
- config_unified_chunk1000_batch_1.yaml
- config_unified_chunk1000_batch_2.yaml
- config_unified_chunk1000_batch_3.yaml
- config_unified_chunk1000_batch_4.yaml
- config_unified_chunk1000_penalty_batch_1.yaml
- config_unified_chunk1000_penalty_batch_2.yaml
- config_unified_chunk1000_penalty_batch_3.yaml
- config_unified_chunk1000_penalty_batch_4.yaml

#### Parametros Verificados
| Parametro | Valor |
|-----------|-------|
| chunk_size | 1000 |
| num_chunks | 12 |
| chunks processados esperados | 11 |
| resume_from_checkpoint | True |
| base_results_dir | .../chunk_1000/batch_X |
| pregenerated_chunks_base_dir | .../unified_chunks/chunk_1000 |
| feature_penalty (sem penalty) | 0.0 |
| feature_penalty (com penalty) | 0.1 |

#### Estrutura de Pastas
- Chunks pre-gerados: unified_chunks/chunk_1000 (53 datasets, 12 chunks cada)
- Resultados: experiments_unified/chunk_1000 e chunk_1000_penalty (a serem criadas)

**Conclusao:** Configs prontos para execucao.

---

## 7. Arquivos Criados/Modificados

### Arquivos Criados
1. `rebuild_metrics.py` - Script para reconstruir metricas
2. `check_progress.py` - Script para verificar progresso
3. `DOCUMENTACAO_JANEIRO_2026.md` - Este documento

### Arquivos Modificados
1. `main.py` - Sistema de resume implementado (~60 linhas)
2. `configs/*.yaml` - 28 arquivos com resume_from_checkpoint: true

---

## 8. Observacoes Importantes

### IntelLabSensors
- Dataset com 39 classes
- G-mean = 0.0 em todos os chunks de teste (problema de desbalanceamento extremo)
- F1-score entre 0.10-0.16 (baixo mas esperado para 39 classes)
- Tempo de execucao elevado (~2-3h por chunk)

### Mecanismo de Resume
- Funciona em dois niveis: skip de datasets completos e resume de chunks parciais
- Requer `resume_from_checkpoint: true` no config
- Salva metricas parciais em `chunk_metrics_partial.json`
- Carrega ultimo individuo de checkpoint para seeding

### Estrutura de Batches
- batch_1: 18 datasets (SEA, AGRAWAL, RBF, STAGGER, HYPERPLANE, RANDOMTREE)
- batch_2: 17 datasets (versoes com noise, gradual, recurring)
- batch_3: 17 datasets (stationary + datasets reais + AssetNegotiation)

---

## 9. Proximos Passos

1. Executar experimentos chunk_1000 (8 configs, ~104 experimentos)
2. Executar experimentos chunk_2000 (14 configs, ~104 experimentos)
3. Executar modelos comparativos (ARF, SRP, HAT, ROSE, ERulesD2S, ACDWM)
4. Consolidar resultados para analise estatistica
5. Gerar tabelas para paper

---

*Documento gerado em 06/01/2026*
