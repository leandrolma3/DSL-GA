# 📊 PLANEJAMENTO: Experimentos Completos em Drift Simulations

**Data**: 2025-10-28
**Status**: 🎯 **PLANEJAMENTO**
**Objetivo**: Executar GBML + River em todas as 41 drift simulations

---

## 🎯 VISÃO GERAL DO PROJETO

### Objetivos

1. **Executar GBML** em todas as 41 drift simulations
2. **Executar River baselines** (5 modelos) em todas as 41 simulações
3. **Comparar resultados** GBML vs River
4. **Gerar visualizações** e relatórios para cada simulação
5. **Consolidar análise** final com ranking de métodos

### Escopo

- **Datasets**: Apenas drift simulations (NÃO incluir datasets reais ainda)
- **Total**: **41 drift simulations** identificadas no config.yaml
- **Execuções**: 41 (GBML) + 41 (River) = **82 execuções totais**
- **Tempo estimado**: 12-18h por execução = **~1.000-1.500 horas total**
- **Paralelização**: Usar máquinas locais + Google Colab

---

## 📋 INVENTÁRIO: 41 DRIFT SIMULATIONS

### Categoria 1: Drifts Abruptos (11 streams)

| Nome | Dataset Base | Conceitos | Total Chunks | Ajuste Necessário |
|------|--------------|-----------|--------------|-------------------|
| SINE_Abrupt_Simple | SINE | 2 | 12 | ✅ Reduzir para 6 |
| SEA_Abrupt_Simple | SEA | 2 | 10 | ✅ Reduzir para 6 |
| SEA_Abrupt_Chain | SEA | 3 | 9 | ✅ Reduzir para 6 |
| AGRAWAL_Abrupt_Simple_Mild | AGRAWAL | 2 | 10 | ✅ Reduzir para 6 |
| AGRAWAL_Abrupt_Simple_Severe | AGRAWAL | 2 | 10 | ✅ Reduzir para 6 |
| RBF_Abrupt_Severe | RBF | 2 | 10 | ✅ Reduzir para 6 |
| STAGGER_Abrupt_Chain | STAGGER | 3 | 12 | ✅ Reduzir para 6 |
| HYPERPLANE_Abrupt_Simple | HYPERPLANE | 2 | 12 | ✅ Reduzir para 6 |
| AGRAWAL_Abrupt_Chain_Long | AGRAWAL | 5 | 10 | ✅ Reduzir para 6 |
| RANDOMTREE_Abrupt_Simple | RANDOMTREE | 2 | 10 | ✅ Reduzir para 6 |
| LED_Abrupt_Simple | LED | 2 | 10 | ✅ Reduzir para 6 |
| WAVEFORM_Abrupt_Simple | WAVEFORM | 2 | 10 | ✅ Reduzir para 6 |

---

### Categoria 2: Drifts Graduais (8 streams)

| Nome | Dataset Base | Conceitos | Total Chunks | Ajuste Necessário |
|------|--------------|-----------|--------------|-------------------|
| SEA_Gradual_Simple_Fast | SEA | 2 | 10 | ✅ Reduzir para 6 |
| SEA_Gradual_Simple_Slow | SEA | 2 | 12 | ✅ Reduzir para 6 |
| AGRAWAL_Gradual_Chain | AGRAWAL | 4 | 12 | ✅ Reduzir para 6 |
| RBF_Gradual_Moderate | RBF | 2 | 10 | ✅ Reduzir para 6 |
| RANDOMTREE_Gradual_Simple | RANDOMTREE | 2 | 10 | ✅ Reduzir para 6 |
| WAVEFORM_Gradual_Simple | WAVEFORM | 2 | 10 | ✅ Reduzir para 6 |
| LED_Gradual_Simple | LED | 2 | 10 | ✅ Reduzir para 6 |
| STAGGER_Gradual_Chain | STAGGER | 3 | 12 | ✅ Reduzir para 6 |
| HYPERPLANE_Gradual_Simple | HYPERPLANE | 2 | 12 | ✅ Reduzir para 6 |
| RBF_Gradual_Severe | RBF | 2 | 10 | ✅ Reduzir para 6 |
| AGRAWAL_Gradual_Mild_to_Severe | AGRAWAL | 3 | 12 | ✅ Reduzir para 6 |

---

### Categoria 3: Drifts Recorrentes e Blips (9 streams)

| Nome | Dataset Base | Conceitos | Total Chunks | Ajuste Necessário |
|------|--------------|-----------|--------------|-------------------|
| SINE_Gradual_Recurring | SINE | 2 recorrente | 13 | ✅ Reduzir para 6 |
| SINE_Abrupt_Recurring_Noise | SINE | 2 recorrente | 13 | ✅ Reduzir para 6 |
| SEA_Abrupt_Recurring | SEA | 2 recorrente | 13 | ✅ Reduzir para 6 |
| AGRAWAL_Gradual_Recurring | AGRAWAL | 2 recorrente | 13 | ✅ Reduzir para 6 |
| RBF_Abrupt_Blip | RBF | 1 blip | 13 | ✅ Reduzir para 6 |
| STAGGER_Mixed_Recurring | STAGGER | 3 recorrente | 12 | ✅ Reduzir para 6 |
| RBF_Severe_Gradual_Recurrent | RBF | 2 recorrente | 10 | ✅ Reduzir para 6 |
| SEA_Gradual_Recurring | SEA | 2 recorrente | 13 | ✅ Reduzir para 6 |
| AGRAWAL_Gradual_Blip | AGRAWAL | 1 blip | 13 | ✅ Reduzir para 6 |
| RANDOMTREE_Abrupt_Recurring | RANDOMTREE | 2 recorrente | 13 | ✅ Reduzir para 6 |

---

### Categoria 4: Drifts com Ruído (7 streams)

| Nome | Dataset Base | Conceitos | Total Chunks | Ajuste Necessário |
|------|--------------|-----------|--------------|-------------------|
| SEA_Abrupt_Chain_Noise | SEA | 3 | 9 | ✅ Já tem 6 úteis |
| AGRAWAL_Gradual_Recurring_Noise | AGRAWAL | 2 recorrente | 13 | ✅ Reduzir para 6 |
| RBF_Abrupt_Blip_Noise | RBF | 1 blip | 13 | ✅ Reduzir para 6 |
| HYPERPLANE_Gradual_Noise | HYPERPLANE | 2 | 12 | ✅ Reduzir para 6 |
| STAGGER_Abrupt_Chain_Noise | STAGGER | 3 | 12 | ✅ Reduzir para 6 |
| RBF_Gradual_Severe_Noise | RBF | 2 | 10 | ✅ Reduzir para 6 |
| RANDOMTREE_Gradual_Noise | RANDOMTREE | 2 | 10 | ✅ Reduzir para 6 |
| AGRAWAL_Abrupt_Simple_Severe_Noise | AGRAWAL | 2 | 10 | ✅ Reduzir para 6 |

---

### Categoria 5: Bartosz Paper Streams (3 streams)

| Nome | Dataset Base | Conceitos | Total Chunks | Ajuste Necessário |
|------|--------------|-----------|--------------|-------------------|
| Bartosz_RandomTree_drift | RANDOMTREE | 2 | 10 | ✅ Reduzir para 6 |
| Bartosz_Agrawal_recurring_drift | AGRAWAL | 2 recorrente | 10 | ✅ Reduzir para 6 |
| Bartosz_SEA_drift_noise | SEA | 4 | 8 | ✅ Reduzir para 6 |

---

## 🔧 CONFIGURAÇÕES NECESSÁRIAS

### 1. Ajustar config.yaml

#### Parâmetros Globais

```yaml
data_params:
  chunk_size: 6000
  num_chunks: 6  # REDUZIR de 8 para 6
  max_instances: 36000  # 6000 * 6

ga_params:
  population_size: 80  # REDUZIR de 120 para 80
  max_generations: 200
  max_generations_recovery: 25
  # Todos os outros parâmetros mantidos
```

**Justificativa**:
- População 80: Reduz tempo ~33% vs 120
- 6 chunks: Permite drift + recovery em <24h

---

#### Ajustar Cada Drift Simulation (41 configurações)

**Estratégia de ajuste**:

1. **Streams com 2 conceitos** (maioria):
   ```yaml
   concept_sequence:
     - { concept_id: 'c1', duration_chunks: 3 }  # 3 chunks conceito 1
     - { concept_id: 'c2', duration_chunks: 3 }  # 3 chunks conceito 2
   ```

2. **Streams com 3 conceitos**:
   ```yaml
   concept_sequence:
     - { concept_id: 'c1', duration_chunks: 2 }
     - { concept_id: 'c2', duration_chunks: 2 }
     - { concept_id: 'c3', duration_chunks: 2 }
   ```

3. **Streams com 4+ conceitos**:
   ```yaml
   concept_sequence:
     - { concept_id: 'c1', duration_chunks: 2 }
     - { concept_id: 'c2', duration_chunks: 1 }
     - { concept_id: 'c3', duration_chunks: 2 }
     - { concept_id: 'c4', duration_chunks: 1 }
   ```

4. **Streams recorrentes**:
   ```yaml
   concept_sequence:
     - { concept_id: 'c1', duration_chunks: 2 }
     - { concept_id: 'c2', duration_chunks: 2 }
     - { concept_id: 'c1', duration_chunks: 2 }  # Recorrência
   ```

5. **Streams com blip**:
   ```yaml
   concept_sequence:
     - { concept_id: 'c1', duration_chunks: 2 }
     - { concept_id: 'c2', duration_chunks: 1 }  # Blip rápido
     - { concept_id: 'c1', duration_chunks: 3 }  # Retorno
   ```

---

### 2. Criar config_experiments.yaml

Arquivo separado com configurações específicas para experimentos em massa:

```yaml
# config_experiments.yaml

experiment_settings:
  run_mode: 'drift_simulation'
  num_runs: 1
  base_results_dir: "experiments_mass"

  # Lista de streams a executar
  drift_simulations_to_run:
    # Categoria 1: Abrupt (12)
    - SINE_Abrupt_Simple
    - SEA_Abrupt_Simple
    - SEA_Abrupt_Chain
    - AGRAWAL_Abrupt_Simple_Mild
    - AGRAWAL_Abrupt_Simple_Severe
    - RBF_Abrupt_Severe
    - STAGGER_Abrupt_Chain
    - HYPERPLANE_Abrupt_Simple
    - AGRAWAL_Abrupt_Chain_Long
    - RANDOMTREE_Abrupt_Simple
    - LED_Abrupt_Simple
    - WAVEFORM_Abrupt_Simple

    # Categoria 2: Gradual (11)
    - SEA_Gradual_Simple_Fast
    - SEA_Gradual_Simple_Slow
    - AGRAWAL_Gradual_Chain
    - RBF_Gradual_Moderate
    - RANDOMTREE_Gradual_Simple
    - WAVEFORM_Gradual_Simple
    - LED_Gradual_Simple
    - STAGGER_Gradual_Chain
    - HYPERPLANE_Gradual_Simple
    - RBF_Gradual_Severe
    - AGRAWAL_Gradual_Mild_to_Severe

    # Categoria 3: Recurring/Blips (10)
    - SINE_Gradual_Recurring
    - SINE_Abrupt_Recurring_Noise
    - SEA_Abrupt_Recurring
    - AGRAWAL_Gradual_Recurring
    - RBF_Abrupt_Blip
    - STAGGER_Mixed_Recurring
    - RBF_Severe_Gradual_Recurrent
    - SEA_Gradual_Recurring
    - AGRAWAL_Gradual_Blip
    - RANDOMTREE_Abrupt_Recurring

    # Categoria 4: Noise (8)
    - SEA_Abrupt_Chain_Noise
    - AGRAWAL_Gradual_Recurring_Noise
    - RBF_Abrupt_Blip_Noise
    - HYPERPLANE_Gradual_Noise
    - STAGGER_Abrupt_Chain_Noise
    - RBF_Gradual_Severe_Noise
    - RANDOMTREE_Gradual_Noise
    - AGRAWAL_Abrupt_Simple_Severe_Noise

    # Categoria 5: Bartosz (3)
    - Bartosz_RandomTree_drift
    - Bartosz_Agrawal_recurring_drift
    - Bartosz_SEA_drift_noise

# Configuração River
river_models:
  - HoeffdingTreeClassifier
  - HoeffdingAdaptiveTreeClassifier
  - AdaptiveRandomForestClassifier
  - StreamingRandomPatchesClassifier
  - LeveragingBaggingClassifier

# Timeouts e limites
execution:
  max_time_per_experiment_hours: 24
  max_memory_gb: 16
  save_checkpoints: true
  checkpoint_frequency_chunks: 2
```

---

## 📊 ESTRUTURA DE SALVAMENTO

### Diretórios

```
experiments_mass/
├── GBML/
│   ├── SINE_Abrupt_Simple/
│   │   ├── run_1/
│   │   │   ├── results.csv
│   │   │   ├── metrics_per_chunk.csv
│   │   │   ├── plots/
│   │   │   │   ├── gmean_evolution.png
│   │   │   │   ├── drift_detection.png
│   │   │   │   ├── hc_activation.png
│   │   │   │   └── confusion_matrices.png
│   │   │   ├── best_individual.pkl
│   │   │   └── experiment_log.txt
│   ├── SEA_Abrupt_Simple/
│   │   └── run_1/...
│   └── ...
├── River/
│   ├── SINE_Abrupt_Simple/
│   │   ├── HoeffdingTree_results.csv
│   │   ├── HAT_results.csv
│   │   ├── ARF_results.csv
│   │   ├── SRP_results.csv
│   │   ├── LevBagging_results.csv
│   │   └── plots/
│   │       └── comparison.png
│   └── ...
└── Comparison/
    ├── SINE_Abrupt_Simple/
    │   ├── gbml_vs_river.csv
    │   └── comparison_plots.png
    └── ...
```

---

## 🤖 AUTOMAÇÃO: Scripts Necessários

### 1. adjust_config_for_mass_experiments.py

**Função**: Ajustar config.yaml automaticamente

```python
# Pseudo-código
def adjust_config():
    # 1. Ler config.yaml
    # 2. Ajustar parâmetros globais:
    #    - num_chunks: 6
    #    - population_size: 80
    #    - max_instances: 36000
    # 3. Ajustar cada drift_simulation para 6 chunks:
    #    - Identificar total de chunks atual
    #    - Reescalar duration_chunks proporcionalmente
    #    - Garantir soma = 6
    # 4. Salvar config_experiments_adjusted.yaml
```

---

### 2. run_mass_experiments.py

**Função**: Executar todos os experimentos GBML

```python
# Pseudo-código
def run_mass_experiments():
    streams = load_streams_list()  # 41 streams

    for stream_name in streams:
        print(f"Executando {stream_name}...")

        try:
            # 1. Executar GBML
            run_gbml_experiment(stream_name)

            # 2. Gerar gráficos
            generate_plots(stream_name)

            # 3. Salvar métricas
            save_metrics(stream_name)

            # 4. Checkpoint
            save_checkpoint(stream_name)

        except TimeoutError:
            log_timeout(stream_name)
        except MemoryError:
            log_memory_error(stream_name)
        except Exception as e:
            log_error(stream_name, e)
```

---

### 3. run_river_baselines.py

**Função**: Executar River em todos os streams

```python
# Pseudo-código (baseado em compare_gbml_vs_river.py)
def run_river_on_all_streams():
    streams = load_streams_list()
    models = ['HT', 'HAT', 'ARF', 'SRP', 'LevBagging']

    for stream_name in streams:
        for model in models:
            print(f"Executando River {model} em {stream_name}...")

            # 1. Carregar chunks (mesmos do GBML)
            chunks = load_chunks(stream_name)

            # 2. Treinar modelo River
            results = train_river_model(model, chunks)

            # 3. Salvar resultados
            save_river_results(stream_name, model, results)
```

---

### 4. compare_all_results.py

**Função**: Comparar GBML vs River em todos os streams

```python
# Pseudo-código
def compare_all():
    streams = load_streams_list()

    ranking_table = []

    for stream_name in streams:
        # 1. Carregar resultados GBML
        gbml_results = load_gbml_results(stream_name)

        # 2. Carregar resultados River (5 modelos)
        river_results = load_river_results(stream_name)

        # 3. Comparar
        comparison = compare(gbml_results, river_results)

        # 4. Gerar gráficos
        plot_comparison(stream_name, comparison)

        # 5. Adicionar ao ranking
        ranking_table.append(comparison)

    # 6. Gerar ranking final
    generate_final_ranking(ranking_table)
    generate_summary_plots(ranking_table)
```

---

### 5. parallel_executor.py

**Função**: Distribuir experimentos em múltiplas máquinas

```python
# Pseudo-código
def distribute_experiments():
    streams = load_streams_list()  # 41 streams
    machines = ['local1', 'local2', 'colab1', 'colab2', 'colab3']

    # Dividir streams entre máquinas
    allocation = allocate_streams_to_machines(streams, machines)

    for machine, assigned_streams in allocation.items():
        # 1. Criar config específico para máquina
        create_machine_config(machine, assigned_streams)

        # 2. Enviar arquivos
        upload_to_machine(machine, assigned_streams)

        # 3. Iniciar execução remota
        start_remote_execution(machine)

        # 4. Monitor progresso
        monitor(machine)
```

---

## ⏱️ ESTIMATIVA DE TEMPO

### Por Experimento

**GBML** (população 80, 6 chunks):
- Chunk estável: ~1.5-2h
- Chunk com drift: ~2-2.5h
- **Total médio**: 12-15h por stream

**River** (5 modelos, 6 chunks):
- Por modelo: ~10-15 min
- Total 5 modelos: ~1h por stream

---

### Total do Projeto

**GBML**: 41 streams × 13h média = **~533 horas** (~22 dias em 1 máquina)

**River**: 41 streams × 1h = **~41 horas** (~2 dias em 1 máquina)

**Total sequencial**: **~574 horas** (~24 dias)

---

### Com Paralelização (5 máquinas)

**GBML**: 533h / 5 = **~107h** (~4.5 dias)

**River**: 41h / 5 = **~8h** (~1 dia)

**Total paralelizado**: **~5-6 dias** ✅

---

## 📋 AÇÕES PRIORITÁRIAS

### Fase 1: Preparação (2-3 dias)

1. ✅ Criar `adjust_config_for_mass_experiments.py`
2. ✅ Ajustar todas as 41 configurações para 6 chunks
3. ✅ Reduzir população para 80
4. ✅ Validar salvamento de gráficos em main.py
5. ✅ Adaptar compare_gbml_vs_river.py para modo batch

---

### Fase 2: Automação (1-2 dias)

6. ✅ Criar `run_mass_experiments.py`
7. ✅ Criar `run_river_baselines.py`
8. ✅ Criar `parallel_executor.py`
9. ✅ Testar em 2-3 streams pequenos

---

### Fase 3: Execução (5-6 dias)

10. ✅ Distribuir streams entre máquinas
11. ✅ Executar GBML em todas as 41 simulações
12. ✅ Executar River em todas as 41 simulações
13. ✅ Monitorar e corrigir erros

---

### Fase 4: Análise (2-3 dias)

14. ✅ Consolidar resultados
15. ✅ Gerar gráficos comparativos
16. ✅ Criar ranking final
17. ✅ Documentar findings

---

## 🎯 PRÓXIMOS PASSOS IMEDIATOS

### Agora (Hoje)

1. Criar script `adjust_config_for_mass_experiments.py`
2. Ajustar 41 configurações no config.yaml
3. Validar salvamento de gráficos no main.py

### Amanhã

4. Criar scripts de automação
5. Testar em 2 streams (1 simples + 1 complexo)
6. Validar pipeline completo

### Depois de Amanhã

7. Iniciar execução paralela
8. Monitorar progresso
9. Consolidar primeiros resultados

---

## 📊 MÉTRICAS A COLETAR

### Por Stream

**GBML**:
- G-mean por chunk
- F1-score por chunk
- Drift detectado (tipo e severidade)
- HC taxa de aprovação
- Tempo de execução
- Memória usada

**River** (cada modelo):
- G-mean por chunk
- F1-score por chunk
- Tempo de execução
- Memória usada

---

### Consolidadas

- Ranking por stream (1º-6º lugar)
- Média geral GBML vs River
- Wins/Losses/Ties
- Melhores/piores streams para GBML
- Análise por tipo de drift (abrupt, gradual, recorrente)

---

## 🚦 CRITÉRIOS DE SUCESSO

### Técnicos

- [ ] 41 streams executados com sucesso (GBML)
- [ ] 205 execuções River (41 × 5 modelos)
- [ ] Gráficos gerados para todos os streams
- [ ] Nenhum timeout > 24h por stream
- [ ] Uso de memória < 16GB

---

### Científicos

- [ ] GBML supera pelo menos 1 baseline em ≥ 70% dos streams
- [ ] Identificar padrões: onde GBML é melhor/pior
- [ ] Validar contribuições (seeding 85%, HC 1.5%, etc.)
- [ ] Resultados reprodutíveis

---

## 📁 DOCUMENTAÇÃO A PRODUZIR

1. **EXPERIMENTOS_MASS_PROTOCOLO.md** - Protocolo detalhado
2. **EXPERIMENTOS_MASS_RESULTADOS.md** - Resultados consolidados
3. **EXPERIMENTOS_MASS_ANALISE.md** - Análise científica
4. **EXPERIMENTOS_MASS_RANKING.md** - Ranking de métodos

---

## 💡 CONSIDERAÇÕES IMPORTANTES

### Priorização de Streams

Se tempo/recursos limitados, priorizar:

**Prioridade ALTA** (validar primeiro, ~10 streams):
- RBF_Abrupt_Severe (já validado)
- SEA_Abrupt_Simple
- AGRAWAL_Abrupt_Simple_Severe
- RBF_Gradual_Moderate
- SEA_Gradual_Simple_Fast
- STAGGER_Abrupt_Chain
- SEA_Abrupt_Recurring
- RBF_Abrupt_Blip
- HYPERPLANE_Abrupt_Simple
- LED_Abrupt_Simple

**Prioridade MÉDIA** (~20 streams): Outros básicos

**Prioridade BAIXA** (~11 streams): Variações com noise

---

### Checkpoints

Salvar checkpoint a cada:
- 2 chunks executados
- 10 streams completos
- 1 categoria completa

---

### Tratamento de Erros

**Timeout (>24h)**:
- Salvar estado atual
- Marcar como "timeout"
- Continuar próximo stream

**Out of Memory**:
- Reduzir população para 60?
- Desabilitar algumas features?
- Marcar como "OOM"

**Erro de código**:
- Log detalhado
- Continuar próximo stream
- Revisar depois

---

## 🎓 LIÇÕES DO PROJETO ANTERIOR

1. ✅ População 80 é suficiente (vs 120)
2. ✅ 6 chunks permitem drift + recovery
3. ✅ Tolerância HC 1.5% funciona bem
4. ✅ Seeding 85% em SEVERE é crítico
5. ✅ Documentação detalhada essencial

---

**Criado por**: Claude Code
**Data**: 2025-10-28
**Status**: 🎯 **PLANEJAMENTO COMPLETO**
**Próximo passo**: Criar scripts de automação

**PRONTO PARA INICIAR IMPLEMENTAÇÃO!** 🚀
