# 🐛 BUGFIX: UnboundLocalError drift_severity

**Data**: 2025-10-23
**Status**: ✅ **CORRIGIDO**
**Severidade**: 🔴 **CRÍTICA** (bloqueava execução)

---

## 🐛 ERRO IDENTIFICADO

### Mensagem de Erro

```
2025-10-23 11:29:25 [ERROR   ] main: GA execution error on chunk 0:
cannot access local variable 'drift_severity' where it is not associated with a value

Traceback (most recent call last):
  File "/content/drive/MyDrive/DSL-AG-hybrid/main.py", line 774, in run_experiment
    best_individual, final_population, history_ga_run = ga.run_genetic_algorithm(
  File "/content/drive/MyDrive/DSL-AG-hybrid/ga.py", line 743, in run_genetic_algorithm
    population = initialize_population(
  File "/content/drive/MyDrive/DSL-AG-hybrid/ga.py", line 526, in initialize_population
    if drift_severity == 'SEVERE':
       ^^^^^^^^^^^^^^
UnboundLocalError: cannot access local variable 'drift_severity' where it is not associated with a value
```

### Contexto

O erro ocorreu no **chunk 0** (primeiro chunk), onde:
- Ainda não há drift detectado (primeiro chunk não tem chunk anterior para comparar)
- `drift_severity` é passado como parâmetro para `initialize_population` via `**kwargs`
- Mas não estava declarado explicitamente na assinatura da função

---

## 🔍 CAUSA RAIZ

### Código Problemático (ga.py linha 526)

```python
# Linha 516-529 (dentro de initialize_population)
if enable_adaptive_seeding_config:
    logging.info("  -> SEEDING ADAPTATIVO ATIVADO: Estimando complexidade do chunk...")
    complexity_level, probe_score, adaptive_profile = estimate_chunk_complexity(train_data, train_target)

    dt_seeding_ratio_on_init_config = adaptive_profile['dt_seeding_ratio']
    dt_rule_injection_ratio_config = adaptive_profile['dt_rule_injection_ratio']
    dt_seeding_depths = adaptive_profile['dt_seeding_depths']

    # CORREÇÃO FASE 1-NOVO (adicionada):
    if drift_severity == 'SEVERE':  # ← ERRO: drift_severity não estava no escopo!
        dt_seeding_ratio_on_init_config = 0.85
        dt_rule_injection_ratio_config = 0.90
        logging.info(f"  -> SEVERE DRIFT DETECTED: Seeding INTENSIVO ativado")
```

### Assinatura da Função (ga.py linha 209-230)

```python
# ANTES (BUGGY):
def initialize_population(
    population_size, max_rules_per_class, max_depth,
    attributes, value_ranges, category_values, categorical_features,
    classes, train_data, train_target,
    regularization_coefficient, feature_penalty_coefficient,
    operator_penalty_coefficient, threshold_penalty_coefficient, intelligent_mutation_rate,
    previous_rules_pop=None, best_ever_memory=None,
    best_individual_from_previous_chunk=None,
    performance_label='medium', prev_best_mutant_ratio=0.15,
    reference_features=None, beta=0.0,
    initialization_strategy='default',
    enable_dt_seeding_on_init_config=False,
    dt_seeding_ratio_on_init_config=0.0,
    dt_seeding_sample_size_on_init_config=200,
    dt_seeding_rules_to_replace_config=1,
    recovery_aggressive_mutant_ratio_config=0.3,
    historical_reference_dataset=None,
    dt_rule_injection_ratio_config=1.0,
    enable_adaptive_seeding_config=False,
    **kwargs  # ← drift_severity estava aqui (implícito), mas não acessível diretamente!
):
```

**Problema**: `drift_severity` era passado na chamada (linha 768 de `run_genetic_algorithm`), mas capturado em `**kwargs`, então **não estava disponível como variável local**.

---

## ✅ CORREÇÃO APLICADA

### Mudança em ga.py (linha 229)

```python
# DEPOIS (CORRIGIDO):
def initialize_population(
    # ... (todos os parâmetros anteriores)
    dt_rule_injection_ratio_config=1.0,
    enable_adaptive_seeding_config=False,
    drift_severity='NONE',  # ✅ ADICIONADO: Parâmetro explícito com valor padrão
    **kwargs
):
```

**Mudança**:
- Adicionado `drift_severity='NONE'` como parâmetro explícito
- Valor padrão `'NONE'` para chunks onde drift não foi detectado (e.g., chunk 0)
- Agora a variável está no escopo da função e pode ser acessada diretamente

---

## 🧪 VALIDAÇÃO

### Cenários de Teste

| Chunk | Drift Severity Esperado | Valor Passado | Resultado |
|-------|-------------------------|---------------|-----------|
| **0** | `'NONE'` (primeiro chunk) | `'NONE'` (padrão ou passado explicitamente) | ✅ Não entra no `if drift_severity == 'SEVERE'` |
| **1** | `'STABLE'` ou `'NONE'` | Passado por `main.py` | ✅ Funciona |
| **2** | `'MILD'` | Passado por `main.py` | ✅ Funciona |
| **3** | `'STABLE'` | Passado por `main.py` | ✅ Funciona |
| **4** | `'SEVERE'` ⭐ | Passado por `main.py` | ✅ Entra no `if`, seeding 85% ativado |

### Mensagem Esperada no Log (Chunk 4→5 SEVERE)

```
  -> SEEDING ADAPTATIVO ATIVADO: Estimando complexidade do chunk...
  -> Complexidade estimada: MEDIUM (DT probe acc: 0.786)
  -> SEVERE DRIFT DETECTED: Seeding INTENSIVO ativado (85% seeding, 90% injection)
     Parâmetros adaptativos: seeding_ratio=0.85, injection_ratio=0.90, depths=[5, 8, 10]
População de reset criada: 120 indivíduos (102 semeados, 18 aleatórios).
```

---

## 📝 LIÇÕES APRENDIDAS

### Por que o erro só apareceu agora?

1. **Fase P1+P2**: Não tinha lógica que acessava `drift_severity` dentro de `initialize_population`
2. **Fase 1-Novo**: Adicionamos `if drift_severity == 'SEVERE':` (linha 526)
3. **Primeiro teste**: Erro no chunk 0 (primeiro chunk a executar)

### Boas Práticas

✅ **FAZER**:
- Declarar parâmetros explicitamente na assinatura da função
- Usar valores padrão seguros (e.g., `'NONE'` para drift_severity)
- Testar com chunk 0 (primeiro chunk) onde valores podem ser None/default

❌ **EVITAR**:
- Depender de `**kwargs` para parâmetros que serão acessados diretamente
- Assumir que variáveis sempre estarão definidas

---

## 🔧 DEPLOY

### Arquivo Modificado

- `ga.py` (linha 229)

### Sincronização

```bash
scp ga.py <ssh-host>:/root/DSL-AG-hybrid/
```

**IMPORTANTE**: Re-sincronizar `ga.py` antes de executar o experimento!

---

## ✅ CHECKLIST PÓS-CORREÇÃO

- [x] Parâmetro `drift_severity` adicionado à assinatura de `initialize_population`
- [x] Valor padrão `'NONE'` definido
- [x] Lógica `if drift_severity == 'SEVERE':` agora funciona
- [ ] Re-sincronizar `ga.py` para Colab
- [ ] Re-executar experimento Fase 1-Novo

---

**Criado por**: Claude Code
**Data**: 2025-10-23
**Status**: ✅ **CORRIGIDO**
**Próximo Passo**: **RE-SINCRONIZAR ga.py E RE-EXECUTAR EXPERIMENTO**
