# OTIMIZAÇÕES DE PERFORMANCE - FASE 1 (QUICK WINS)

**Data de Criação:** 2025-10-31
**Status:** 📝 PRÉ-IMPLEMENTAÇÃO
**Objetivo:** Reduzir tempo de execução em ~50% com implementação de 1-2 horas
**Autor:** Análise conjunta via Claude Code

---

## 📊 BASELINE ATUAL (ANTES DAS MUDANÇAS)

### Experimentos de Referência

| Experimento | Config | Tempo Total | Chunks | Avg Test G-mean | Status |
|-------------|--------|-------------|--------|-----------------|--------|
| **test_single** | RBF_Abrupt_Severe | 13.0h (46,923s) | 6 | 79.83% | ✅ Completo |
| **test_drift_recovery** | RBF_Drift_Recovery | 10.0h (36,034s) | 6 | 73.74% | ✅ Completo |
| **test_multi_drift** | RBF_Multi_Drift | 16.6h (59,779s) | 8 | 69.96% | ✅ Completo |

### Métricas Detalhadas (test_single como referência)

**Performance por Chunk:**
- Chunk 0: 87.12% G-mean, 6,415s (25 gens)
- Chunk 1: 87.89% G-mean, 9,511s (55 gens)
- Chunk 2: 90.49% G-mean, 12,265s (50 gens)
- Chunk 3: 89.68% G-mean, 5,846s (16 gens)
- Chunk 4: 43.98% G-mean, 12,879s (61 gens)

**Resumo Estatístico:**
- Avg Train G-mean: 92.65% ± 2.36%
- Avg Test G-mean: 79.83% ± 17.97%
- Avg Test F1: 80.29% ± 17.20%
- Tempo médio por chunk: 9,383s (~2.6h)
- Gerações médias: 41.4 gerações
- Tempo médio por geração: ~227s (~3.8min)

**Configuração Base:**
```yaml
ga_params:
  population_size: 100
  max_generations: 200
  max_generations_recovery: 25
  elitism_rate: 0.1
  intelligent_mutation_rate: 0.8
  hc_hierarchical_enabled: true
data_params:
  chunk_size: 6000
  num_chunks: 6
```

---

## 🎯 ANÁLISE DE GARGALOS IDENTIFICADOS

### Breakdown de Tempo por Geração (~227s)

| Componente | Tempo (s) | % Total | Complexidade | Prioridade Fix |
|------------|-----------|---------|--------------|----------------|
| **Fitness Evaluation** | 193 | 85% | O(P × N × C × R × D) | ⚡ CRÍTICO |
| **Deep Copy** | 27 | 12% | O(P × S) | 🔥 ALTA |
| **Hill Climbing** | 0-11 | 0-5% | O(V × N) quando ativo | 🔸 MÉDIA |
| **Operators (Crossover/Mutation)** | 7 | 3% | O(P × R × D) | 🔹 BAIXA |

**Quando HC ativo (a cada ~15 gerações de stagnation):**
- Tempo por geração: ~238s (+11s overhead HC)
- HC adiciona ~5% de tempo mas é crucial para escapar de estagnation

### Cálculos de Complexidade

**Por experimento completo (6 chunks, 50 gens médias):**
```
Operações de avaliação de nó:
= População × Gerações × Chunks × Dataset × Classes × Regras × Depth
= 100 × 50 × 6 × 6000 × 9 × 8 × 6
= 77.760.000.000 operações (78 bilhões)

Deep copies:
= População × Gerações × Chunks × Tamanho_Individual
= 100 × 50 × 6 × 54KB
= 1.62GB de alocações

HC ativações:
= (Gerações / 15) × Variantes × Dataset × Custo_Avaliação
= (50 / 15) × 15 × 6000 × custo
= ~300.000 avaliações extras por chunk
```

---

## 🚀 MUDANÇAS PLANEJADAS (QUICK WINS)

### OTIMIZAÇÃO #1: CACHE DE FITNESS ⚡

**Problema Identificado:**
- Indivíduos idênticos são re-avaliados múltiplas vezes
- Elite (12 indivíduos) não muda entre gerações mas é re-avaliada
- Filhos de crossover podem ser duplicados (~15-20% da população)

**Solução:**
Implementar cache baseado em hash da estrutura de regras.

**Localização:**
- **Arquivo:** `ga.py`
- **Função:** Loop de avaliação de fitness (linha ~840-865)
- **Inserção:** Antes do loop, criar cache dict

**Pseudocódigo:**
```python
# Adicionar antes do loop de gerações
fitness_cache = {}

# No loop de avaliação
for individual in population:
    ind_hash = hash_individual(individual)

    if ind_hash in fitness_cache:
        # CACHE HIT: Reutiliza resultado
        individual.fitness = fitness_cache[ind_hash]['fitness']
        individual.gmean = fitness_cache[ind_hash]['gmean']
        continue  # SKIP avaliação completa

    # CACHE MISS: Avalia e armazena
    evaluate_and_cache(individual)

# Limpa cache a cada 10 gerações (evita crescimento ilimitado)
if generation % 10 == 0:
    fitness_cache.clear()
```

**Estimativa de Ganho:**

| Métrica | Valor | Cálculo |
|---------|-------|---------|
| **Hit Rate Esperado** | ~25% | 12 elite + 13 duplicatas |
| **Avaliações Economizadas/Gen** | 25 | 25% de 100 |
| **Tempo Economizado/Gen** | 57s | 25 × 2.27s |
| **Ganho por Chunk** | 2,850s | 57s × 50 gens |
| **Ganho Total** | 17,100s (4.75h) | 2,850s × 6 chunks |
| **Redução de Tempo** | **-30%** | 13h → 9.1h |

**Complexidade de Implementação:**
- Linhas de código: ~25
- Tempo de implementação: ~30 minutos
- Risco de bugs: BAIXO (apenas lookup, não altera lógica)

**Critério de Sucesso:**
- Cache hit rate > 20%
- Tempo por geração com cache < 170s (vs 227s atual)
- Performance (G-mean) mantida (variação < 1%)

---

### OTIMIZAÇÃO #2: EARLY STOPPING SIMPLES EM FITNESS 🔥

**Problema Identificado:**
- Todos os 6000 exemplos são avaliados mesmo para indivíduos obviamente ruins
- Indivíduos com G-mean < 50% no 1º batch continuam sendo avaliados
- ~30-40% da população é descartável cedo

**Solução:**
Avaliar em batches incrementais e parar se score parcial já é pior que threshold.

**Localização:**
- **Arquivo:** `fitness.py`
- **Função:** `calculate_fitness()` (linha ~174-307)
- **Modificação:** Adicionar avaliação incremental com early stop

**Pseudocódigo:**
```python
def calculate_fitness(individual, data, target, ..., early_stop_threshold=None):
    if early_stop_threshold is None:
        early_stop_threshold = -float('inf')

    # Avaliação incremental em batches
    batch_size = 1000  # Avalia 1000 de cada vez
    all_predictions = []

    for batch_idx in range(num_batches):
        batch_data = data[start:end]
        batch_predictions = [individual._predict(inst) for inst in batch_data]
        all_predictions.extend(batch_predictions)

        # EARLY STOP: Após primeiro batch, verifica se vale continuar
        if batch_idx > 0:
            partial_gmean = calculate_gmean(target[:end], all_predictions)

            if partial_gmean < early_stop_threshold * 0.8:
                # Indivíduo ruim, retorna fitness mínimo
                return {'fitness': -float('inf'), 'g_mean': partial_gmean}

    # Se chegou aqui, calcula fitness completo
    return calculate_full_fitness(...)
```

**Passar threshold no GA:**
```python
# Em ga.py, antes de avaliar população
worst_elite_gmean = sorted(population, key=lambda x: x.gmean)[11].gmean
constant_args['early_stop_threshold'] = worst_elite_gmean
```

**Estimativa de Ganho:**

| Métrica | Valor | Cálculo |
|---------|-------|---------|
| **Indivíduos Parados Cedo** | ~35% | 30-40% da população |
| **Avaliações Economizadas** | 4000/indivíduo | 6000 - 2000 (primeiro batch) |
| **Tempo Economizado/Indivíduo** | 1.5s | 67% de 2.27s |
| **Economia por Geração** | 53s | 35 ind × 1.5s |
| **Ganho por Chunk** | 2,650s | 53s × 50 gens |
| **Ganho Total** | 15,900s (4.4h) | 2,650s × 6 chunks |
| **Redução de Tempo** | **-20%** | 13h → 10.4h |

**Complexidade de Implementação:**
- Linhas de código: ~50
- Tempo de implementação: ~40 minutos
- Risco de bugs: MÉDIO (pode descartar indivíduos bons se threshold mal calibrado)

**Critério de Sucesso:**
- Taxa de early stop > 25%
- Tempo por geração < 180s
- Performance (G-mean) mantida (variação < 2% - margem maior por ser heurística)

**Mitigação de Risco:**
- Usar threshold conservador (0.8 × worst_elite)
- Validar em primeiro batch (1000 exemplos = 16% dos dados)
- Monitorar se indivíduos bons estão sendo descartados (log warnings)

---

### OTIMIZAÇÃO #3: REDUZIR VARIANTES HILL CLIMBING 🔸

**Problema Identificado:**
- HC gera 12-18 variantes dependendo do nível (aggressive/moderate/fine_tuning)
- Cada variante precisa de fitness completo (6000 avaliações)
- Muitas variantes são similares e não adicionam valor

**Solução:**
Limitar número máximo de variantes retornadas para 8 (vs 12-18).

**Localização:**
- **Arquivo:** `hill_climbing_v2.py`
- **Função:** `hierarchical_hill_climbing()` (linha ~604)
- **Modificação:** Adicionar limite no retorno

**Código Atual:**
```python
# Linha 604
return all_variants[:num_variants]
```

**Código Modificado:**
```python
# Limita variantes ao mínimo entre configurado e 8
max_variants_to_return = min(8, level_config['num_variants_base'])
logger.info(f"       HC: Retornando {max_variants_to_return} variantes (de {len(all_variants)} geradas)")
return all_variants[:max_variants_to_return]
```

**Estimativa de Ganho:**

| Métrica | Valor | Cálculo |
|---------|-------|---------|
| **Variantes Reduzidas** | 12-18 → 8 | -33% a -56% |
| **Ativações HC/Chunk** | ~3 | A cada 15 gens de stagnation |
| **Tempo/Variante** | 2.27s | Fitness completo |
| **Economia por Ativação HC** | 9-23s | (12-18 - 8) × 2.27s |
| **Ganho por Chunk** | 27-69s | 3 ativações × economia |
| **Ganho Total** | 162-414s | × 6 chunks |
| **Redução de Tempo** | **-3 a -5%** | 13h → 12.6-12.4h |

**Complexidade de Implementação:**
- Linhas de código: ~10
- Tempo de implementação: ~20 minutos
- Risco de bugs: BAIXO (apenas limita lista, não altera lógica)

**Critério de Sucesso:**
- HC continua melhorando indivíduos (taxa de aprovação mantida)
- Número de variantes avaliadas ≤ 8 por ativação
- Tempo de HC reduzido mas efetividade mantida

---

### OTIMIZAÇÃO #4: SHALLOW COPY PARA ELITE 🔹

**Problema Identificado:**
- Elite (12 indivíduos) é deep copied a cada geração
- Elite não muda, então deep copy é desperdício
- Deep copy de 12 × 54KB = 648KB por geração = overhead desnecessário

**Solução:**
Usar referências diretas para elite + flag de proteção contra modificação.

**Localização:**
- **Arquivo:** `individual.py` (adicionar atributo)
- **Arquivo:** `ga.py` (modificar elitismo, linha ~1024)
- **Arquivo:** `ga_operators.py` (verificar proteção antes de modificar)

**Código em individual.py:**
```python
class Individual:
    def __init__(self, ...):
        # ... código existente ...
        self.is_protected = False  # NOVO: Flag de proteção
```

**Código em ga.py (elitismo):**
```python
# ANTES (linha 1024):
new_population.extend(copy.deepcopy(ind) for ind in final_elites)

# DEPOIS:
for elite in final_elites:
    elite.is_protected = True  # Marca como protegido
    new_population.append(elite)  # Usa referência direta (sem copy)
logger.debug(f"Elite preservation: {len(final_elites)} protected references added")
```

**Código em ga_operators.py (garantir cópia se modificar):**
```python
def crossover(parent1, parent2, ...):
    # Se pai é protegido, copia antes de modificar
    if getattr(parent1, 'is_protected', False):
        parent1 = copy.deepcopy(parent1)
        parent1.is_protected = False
    if getattr(parent2, 'is_protected', False):
        parent2 = copy.deepcopy(parent2)
        parent2.is_protected = False

    # ... resto do código normal ...
```

**Estimativa de Ganho:**

| Métrica | Valor | Cálculo |
|---------|-------|---------|
| **Deep Copies Eliminadas** | 12/geração | Elite não muda |
| **Tamanho por Copy** | 54KB | Estrutura Individual |
| **Overhead por Copy** | ~5ms | Python deepcopy overhead |
| **Economia por Geração** | 60ms | 12 × 5ms |
| **Ganho por Chunk** | 3s | 60ms × 50 gens |
| **Ganho Total** | 18s | 3s × 6 chunks |
| **Redução de Tempo** | **-2 a -3%** | 13h → 12.7h |

**Complexidade de Implementação:**
- Linhas de código: ~20
- Tempo de implementação: ~10 minutos
- Risco de bugs: MÉDIO (precisa garantir proteção em TODOS os operadores)

**Critério de Sucesso:**
- Elite preservada corretamente (mesmos indivíduos entre gerações)
- Nenhuma modificação acidental de elite
- Tempo de elitismo < 0.5s por geração (vs ~0.06s + overhead atual)

**Mitigação de Risco:**
- Adicionar asserts para verificar proteção
- Logar warnings se tentar modificar indivíduo protegido sem copiar
- Testar em experimento pequeno primeiro

---

### OTIMIZAÇÃO #5: REDUZIR LOGGING PARA WARNING ⚡

**Problema Identificado:**
- Logging level INFO gera ~1000 linhas por geração
- I/O de logging tem overhead, especialmente em loops
- Debug logs não são necessários em produção

**Solução:**
Mudar logging level de INFO para WARNING em produção.

**Localização:**
- **Arquivo:** `config.yaml` (ou `main.py`)
- **Parâmetro:** `logging_level`

**Código Atual (config.yaml):**
```yaml
experiment_settings:
  logging_level: INFO
```

**Código Modificado:**
```yaml
experiment_settings:
  logging_level: WARNING  # Era: INFO
```

**OU em main.py se logging configurado manualmente:**
```python
# ANTES:
logging.basicConfig(level=logging.INFO, ...)

# DEPOIS:
logging.basicConfig(level=logging.WARNING, ...)
```

**Estimativa de Ganho:**

| Métrica | Valor | Cálculo |
|---------|-------|---------|
| **Linhas Log Reduzidas** | ~1000/geração | INFO → WARNING |
| **Overhead por Log** | ~0.5ms | I/O + formatting |
| **Economia por Geração** | 0.5s | 1000 × 0.5ms |
| **Ganho por Chunk** | 25s | 0.5s × 50 gens |
| **Ganho Total** | 150s (2.5min) | 25s × 6 chunks |
| **Redução de Tempo** | **-1 a -2%** | 13h → 12.8h |

**Complexidade de Implementação:**
- Linhas de código: 1
- Tempo de implementação: ~2 minutos
- Risco de bugs: ZERO (apenas muda verbosidade)

**Critério de Sucesso:**
- Logs reduzidos (apenas warnings/errors)
- Tempo ligeiramente menor
- Informações críticas ainda visíveis

**Nota:** Para debug, pode-se manter INFO level em desenvolvimento.

---

## 📈 GANHO TOTAL ESPERADO (COMPOSTO)

### Cálculo de Ganho Acumulado

**IMPORTANTE:** Ganhos NÃO são aditivos (alguns se sobrepõem).

**Ganho Composto Calculado:**
```
Tempo Base: 13h (46,800s)

Após Otimização #1 (Cache -30%):
= 46,800s × 0.70 = 32,760s

Após Otimização #2 (Early Stop -20% do restante):
= 32,760s × 0.80 = 26,208s

Após Otimização #3 (HC -4% do restante):
= 26,208s × 0.96 = 25,160s

Após Otimização #4 (Shallow -2.5% do restante):
= 25,160s × 0.975 = 24,531s

Após Otimização #5 (Logging -1.5% do restante):
= 24,531s × 0.985 = 24,163s

TEMPO FINAL: 24,163s = 6.7h
REDUÇÃO TOTAL: 22,637s = 6.3h = -48.4%
```

### Tabela de Ganhos Acumulados

| Fase | Otimização | Tempo Após | Redução Absoluta | Redução % Acumulada |
|------|-----------|------------|------------------|---------------------|
| **Inicial** | - | 13.0h | - | 0% |
| **+Opt1** | Cache Fitness | 9.1h | -3.9h | -30% |
| **+Opt2** | Early Stop | 7.3h | -5.7h | -44% |
| **+Opt3** | HC Reduzido | 7.0h | -6.0h | -46% |
| **+Opt4** | Shallow Elite | 6.8h | -6.2h | -48% |
| **+Opt5** | Logging | **6.7h** | **-6.3h** | **-48%** |

### Ganho por Tipo de Experimento

| Experimento | Tempo Atual | Tempo Esperado | Economia | Aplicabilidade |
|-------------|-------------|----------------|----------|----------------|
| **test_single** (6 chunks) | 13.0h | 6.7h | 6.3h | ✅ 100% |
| **test_drift_recovery** (6 chunks) | 10.0h | 5.2h | 4.8h | ✅ 100% |
| **test_multi_drift** (8 chunks) | 16.6h | 8.6h | 8.0h | ✅ 100% |

**Economia Total Estimada por Rodada de 3 Experimentos:**
- Antes: 39.6h
- Depois: 20.5h
- **ECONOMIA: 19.1 horas (~1 dia de compute)**

---

## ✅ VALIDAÇÃO PLANEJADA

### Experimento de Validação

**Configuração:**
- **Experimento:** RBF_Abrupt_Severe (test_single)
- **Config:** `config_test_single_optimized.yaml` (cópia com otimizações)
- **Chunks:** 6 (36,000 instâncias)
- **População:** 100
- **Gerações:** 200 (com early stopping)

**Comparação Baseline:**
```
Baseline (experimento_test_single.log):
- Tempo: 13.0h (46,923s)
- Avg Test G-mean: 79.83%
- Chunks: 0-4 completos
```

### Critérios de Sucesso

| Métrica | Baseline | Meta | Range Aceitável | Prioridade |
|---------|----------|------|-----------------|-----------|
| **Tempo Total** | 13.0h | < 8.0h | 7.0-8.5h | 🔴 CRÍTICO |
| **Avg Test G-mean** | 79.83% | > 78% | 78-81% | 🔴 CRÍTICO |
| **Tempo/Chunk** | 9.4k s | < 5.0k s | 4.5-5.5k s | 🟠 ALTA |
| **Cache Hit Rate** | 0% | > 20% | 15-30% | 🟡 MÉDIA |
| **Early Stop Rate** | 0% | > 25% | 20-35% | 🟡 MÉDIA |
| **HC Variantes** | 12-18 | ≤ 8 | 6-8 | 🟢 BAIXA |

### Métricas a Monitorar

**Performance (NÃO pode degradar):**
- [ ] G-mean por chunk (vs baseline)
- [ ] F1-score por chunk
- [ ] Recall/Precision por classe
- [ ] Taxa de HC aprovação (deve manter ~30-40%)
- [ ] Gerações até convergência

**Tempo (DEVE melhorar):**
- [ ] Tempo total de execução
- [ ] Tempo por chunk
- [ ] Tempo por geração
- [ ] Tempo de fitness evaluation
- [ ] Tempo de Hill Climbing

**Otimizações (Verificar efetividade):**
- [ ] Cache hit rate por geração
- [ ] Taxa de early stop por geração
- [ ] Número de variantes HC geradas vs avaliadas
- [ ] Número de deep copies evitadas
- [ ] Tamanho dos logs gerados

### Plano de Rollback

**Se validação FALHAR (tempo > 8.5h OU G-mean < 78%):**

1. Analisar logs para identificar qual otimização causou problema
2. Desabilitar otimização problemática
3. Re-executar validação
4. Documentar em seção "RESULTADOS E AJUSTES"

**Comandos de rollback:**
```bash
# Reverter para versão anterior
git checkout HEAD~1 ga.py fitness.py hill_climbing_v2.py individual.py

# Ou desabilitar otimizações individualmente (se implementadas com flags)
# config.yaml:
enable_fitness_cache: false
enable_early_stopping: false
```

---

## 🔬 ANÁLISE DE RISCOS

### Riscos Identificados e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| **Cache retorna fitness desatualizado** | BAIXA | ALTO | Hash inclui TODAS as regras; cache limpo a cada 10 gens |
| **Early stop descarta bons indivíduos** | MÉDIA | MÉDIO | Threshold conservador (0.8×); valida em 16% dos dados |
| **Shallow copy causa mutação de elite** | MÉDIA | ALTO | Flag `is_protected`; verificação em todos operadores |
| **HC com 8 variantes perde efetividade** | BAIXA | BAIXO | 8 variantes ainda é suficiente; pode ajustar se necessário |
| **Logging WARNING perde info crítica** | BAIXA | BAIXO | Warnings/errors mantidos; pode voltar para INFO se debug |
| **Tempo não melhora como esperado** | MÉDIA | MÉDIO | Validação incremental; rollback se < 40% ganho |
| **G-mean degrada > 2%** | BAIXA | ALTO | Rollback imediato; análise de causa raiz |

### Contingências

**Se Tempo > 8.5h:**
- Investigar qual otimização não funcionou (profiler)
- Ajustar thresholds (cache, early stop)
- Considerar implementar apenas otimizações bem-sucedidas

**Se G-mean < 78%:**
- Rollback completo
- Testar otimizações individualmente (uma por vez)
- Aumentar threshold de early stop para mais conservador

**Se Cache Hit Rate < 15%:**
- Verificar se hash_individual() está correto
- Aumentar período de limpeza de cache (10 → 20 gerações)
- Logs adicionais para debug

---

## 📋 CHECKLIST DE IMPLEMENTAÇÃO

### Pré-Implementação
- [x] Documentar baseline atual
- [x] Calcular ganhos esperados
- [x] Definir critérios de sucesso
- [x] Criar plano de validação
- [ ] Backup de arquivos a modificar
- [ ] Criar branch git (se usando controle de versão)

### Implementação (Ordem Sugerida)

**Ordem otimizada para minimizar interdependências:**

1. [ ] **Otimização #5: Logging WARNING** (2min, ZERO risco)
   - Arquivo: config.yaml
   - Teste: Verificar que logs são reduzidos

2. [ ] **Otimização #3: HC Reduzido** (20min, BAIXO risco)
   - Arquivo: hill_climbing_v2.py
   - Teste: HC gera ≤8 variantes

3. [ ] **Otimização #4: Shallow Copy Elite** (10min, MÉDIO risco)
   - Arquivos: individual.py, ga.py, ga_operators.py
   - Teste: Elite preservada, sem modificações acidentais

4. [ ] **Otimização #1: Cache de Fitness** (30min, BAIXO risco)
   - Arquivo: ga.py
   - Teste: Cache hit rate > 20%, G-mean mantido

5. [ ] **Otimização #2: Early Stop** (40min, MÉDIO risco)
   - Arquivo: fitness.py, ga.py
   - Teste: Early stop rate > 25%, G-mean mantido

### Pós-Implementação
- [ ] Executar testes unitários (se houver)
- [ ] Executar experimento de validação
- [ ] Monitorar métricas durante execução
- [ ] Comparar resultados com baseline
- [ ] Atualizar este documento com resultados reais
- [ ] Criar documento de análise de impacto

---

## 📝 NOTAS TÉCNICAS

### Considerações de Implementação

**Hash de Individual (para Cache):**
```python
def hash_individual(individual):
    """
    Gera hash único baseado na estrutura de regras.
    Deve capturar TODA a informação relevante para fitness.
    """
    # Usa método existente que serializa todas as regras
    rules_string = individual.get_rules_as_string()

    # Adiciona default_class ao hash (pode afetar predições)
    hash_string = f"{rules_string}|{individual.default_class}"

    return hash(hash_string)
```

**Threshold de Early Stop:**
- Usar worst elite (12º melhor) como baseline
- Multiplicar por 0.8 para dar margem (conservador)
- Se população ainda não tem elite (geração 0), usar 0.0

**Limpeza de Cache:**
- A cada 10 gerações: `fitness_cache.clear()`
- Justificativa: Conceito muda entre chunks, cache deve ser resetado
- Alternativa: Limpar apenas ao mudar de chunk (se memória permitir)

**Proteção de Elite:**
- Flag `is_protected` deve ser resetada após deep copy
- Verificar proteção ANTES de qualquer modificação (crossover, mutation, HC)
- Usar `getattr(ind, 'is_protected', False)` para backward compatibility

---

## 🔗 REFERÊNCIAS

### Documentos Relacionados
- `ANALISE_EXPERIMENTO_COMPLETA.md` - Análise que identificou os gargalos
- `experimento_test_single.log` - Log baseline para comparação
- `experimento_test_drift_recovery.log` - Log de validação secundária
- `experimento_test_multi_drift.log` - Log de validação terciária

### Arquivos a Modificar
- `ga.py` - Cache de fitness, threshold early stop, shallow copy elite
- `fitness.py` - Early stopping incremental
- `hill_climbing_v2.py` - Reduzir variantes
- `individual.py` - Flag is_protected
- `ga_operators.py` - Verificar proteção em operators
- `config.yaml` - Logging level

### Funções-Chave
- `ga.py:evolve_population()` - Loop principal de GA
- `fitness.py:calculate_fitness()` - Avaliação de fitness
- `individual.py:_predict()` - Predição de indivíduo
- `hill_climbing_v2.py:hierarchical_hill_climbing()` - Hill climbing

---

## 📊 TEMPLATE PARA RESULTADOS (PREENCHER APÓS VALIDAÇÃO)

### RESULTADOS OBTIDOS

**Experimento de Validação:** `experimento_test_single_optimized_YYYYMMDD_HHMMSS.log`

**Dados Brutos:**
- Tempo total: ___h (___s)
- Redução real: ___%
- Avg Test G-mean: __.__%
- Avg Train G-mean: __.__%

**Comparação Before/After:**

| Métrica | Baseline | Otimizado | Delta | Status |
|---------|----------|-----------|-------|--------|
| Tempo Total | 13.0h | ___h | -___% | ✅/⚠️/❌ |
| Tempo/Chunk | 9,383s | ___s | -___% | ✅/⚠️/❌ |
| Tempo/Geração | 227s | ___s | -___% | ✅/⚠️/❌ |
| Avg Test G-mean | 79.83% | __.__% | ±__._%% | ✅/⚠️/❌ |
| Avg Train G-mean | 92.65% | __.__% | ±__._%% | ✅/⚠️/❌ |

**Análise de Impacto por Otimização:**

### Otimização #1: Cache de Fitness
- Cache hit rate: ___%
- Hits por geração: ___
- Misses por geração: ___
- Tempo economizado estimado: ___s
- Ganho real: -___% (esperado: -30%)
- Status: ✅/⚠️/❌

### Otimização #2: Early Stop
- Taxa de early stop: ___%
- Indivíduos parados cedo: ___ (de 100)
- Economia média de avaliações: ___
- Tempo economizado estimado: ___s
- Ganho real: -___% (esperado: -20%)
- Status: ✅/⚠️/❌

### Otimização #3: HC Reduzido
- Variantes geradas: ___
- Variantes avaliadas: ___
- Ativações HC: ___
- Tempo HC total: ___s
- Ganho real: -___% (esperado: -4%)
- Status: ✅/⚠️/❌

### Otimização #4: Shallow Copy Elite
- Deep copies eliminadas: ___
- Elite preservada corretamente: SIM/NÃO
- Overhead de elitismo: ___s
- Ganho real: -___% (esperado: -2.5%)
- Status: ✅/⚠️/❌

### Otimização #5: Logging
- Linhas de log: ___ (antes: ~50,000)
- Tamanho do log: ___KB (antes: ~5MB)
- Ganho real: -___% (esperado: -1.5%)
- Status: ✅/⚠️/❌

**Performance por Chunk:**

| Chunk | Baseline G-mean | Otimizado G-mean | Delta | Tempo Baseline | Tempo Otimizado | Delta Tempo |
|-------|-----------------|------------------|-------|----------------|-----------------|-------------|
| 0 | 87.12% | __.__% | ±__._%% | 6,415s | ___s | -___% |
| 1 | 87.89% | __.__% | ±__._%% | 9,511s | ___s | -___% |
| 2 | 90.49% | __.__% | ±__._%% | 12,265s | ___s | -___% |
| 3 | 89.68% | __.__% | ±__._%% | 5,846s | ___s | -___% |
| 4 | 43.98% | __.__% | ±__._%% | 12,879s | ___s | -___% |

---

## 🎯 CONCLUSÕES E PRÓXIMOS PASSOS (PREENCHER APÓS VALIDAÇÃO)

### Conclusões

**Sucesso Geral:** ✅/⚠️/❌

**Otimizações Bem-Sucedidas:**
- [ ] Cache de Fitness
- [ ] Early Stop
- [ ] HC Reduzido
- [ ] Shallow Copy Elite
- [ ] Logging

**Otimizações que Precisam Ajuste:**
- ...

**Problemas Encontrados:**
- ...

**Lições Aprendidas:**
- ...

### Próximos Passos

**Se Sucesso (Tempo < 8h E G-mean > 78%):**
1. [ ] Aplicar otimizações aos outros 2 experimentos (drift_recovery, multi_drift)
2. [ ] Validar ganhos são consistentes
3. [ ] Considerar implementar Fase 2 (Memory Recorrente, Drift Prediction)
4. [ ] Atualizar todos os configs para usar otimizações

**Se Parcialmente Bem-Sucedido (Tempo < 10h OU G-mean > 78%):**
1. [ ] Identificar otimização problemática
2. [ ] Ajustar parâmetros (thresholds, limites)
3. [ ] Re-validar com ajustes
4. [ ] Documentar trade-offs

**Se Falha (Tempo > 10h OU G-mean < 78%):**
1. [ ] Rollback completo
2. [ ] Testar otimizações individualmente
3. [ ] Identificar causa raiz
4. [ ] Re-planejar abordagem

---

**Documento criado por:** Claude Code
**Data:** 2025-10-31
**Status:** 📝 PRÉ-IMPLEMENTAÇÃO (Aguardando implementação e validação)
**Próxima Atualização:** Após validação experimental
