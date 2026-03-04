# CHECKLIST DE SINCRONIZAÇÃO - GOOGLE COLAB

**Data de Criação:** 2025-11-01
**Objetivo:** Garantir que todos os arquivos com logging atualizado sejam sincronizados corretamente no Google Colab antes de rodar experimentos.

---

## ✅ PRÉ-UPLOAD: VALIDAÇÕES LOCAIS

### 1. Sintaxe Python Validada
- [x] `main.py` - Compilação OK (sem erros)
- [x] `utils.py` - Compilação OK (sem erros)

### 2. Arquivos Modificados (Layer 1 - Logging)

| Arquivo | Modificação | Linhas Alteradas | Status |
|---------|-------------|------------------|--------|
| **main.py** | Fase 2 logging WARNING | 553-621, 991-1001, 1088-1102 | ✅ Pronto |
| **main.py** | Timing logging | 505-523, 1081-1097, 1238-1254 | ✅ Pronto |
| **utils.py** | detect_recurring_concept() logging | 507-571 | ✅ Pronto |

### 3. Dependências de Sistema
- [x] `scipy` (para cosine similarity) - Verificar se está no Colab
- [x] `numpy` - Já disponível no Colab
- [x] `logging` - Built-in Python

---

## 📤 UPLOAD PARA GOOGLE COLAB

### Arquivos Obrigatórios (CRITICAL)

**Passo 1: Arquivos Python Atualizados**
```
[ ] main.py (versão com logging WARNING + timing)
[ ] utils.py (versão com logging WARNING em detect_recurring_concept)
```

**Passo 2: Configs YAML (se atualizados)**
```
[ ] config_test_single.yaml
[ ] config_test_drift_recovery.yaml
[ ] config_test_multi_drift.yaml
```

**Nota:** Se configs não foram alterados, não é necessário re-upload.

### Arquivos Opcionais (para referência)
```
[ ] ANALISE_EXPERIMENTOS_FASE1_FASE2.md
[ ] PLANO_DIAGNOSTICO_E_AJUSTES.md
[ ] CHECKLIST_SINCRONIZACAO_COLAB.md (este arquivo)
```

---

## 🔍 PÓS-UPLOAD: VALIDAÇÕES NO COLAB

### 1. Verificar Versões dos Arquivos

Execute no Colab para confirmar que os arquivos foram atualizados:

```bash
# Verificar se main.py tem logging WARNING de Fase 2
grep -n "logger.warning.*FASE 2" main.py | head -5

# Verificar se utils.py tem logging WARNING de similaridade
grep -n "logger.warning.*Similaridades calculadas" utils.py

# Verificar se main.py tem logging de timing
grep -n "experiment_start_time = time.time()" main.py
```

**Esperado:**
- `main.py`: Deve mostrar linhas ~553, ~560, ~568, etc. com `logger.warning("[FASE 2]...)`
- `utils.py`: Deve mostrar linha ~542 com `logging.warning(f"[FASE 2] Similaridades calculadas:")`
- `main.py`: Deve mostrar linha ~506 com `experiment_start_time = time.time()`

### 2. Verificar Dependências

```bash
# No Colab, confirmar que scipy está instalado
python -c "from scipy.spatial.distance import cosine; print('scipy OK')"
```

**Esperado:** "scipy OK"

Se falhar:
```bash
pip install scipy
```

### 3. Teste de Smoke (Opcional mas Recomendado)

Teste rápido com 2 chunks para validar que logging está funcionando:

```python
# No Colab
!python main.py config_test_drift_recovery.yaml --num_chunks 2 --run_number 999
```

**Validações esperadas no log:**
1. ✅ Mensagens `[FASE 2] Chunk X - Calculando concept fingerprint...`
2. ✅ Mensagens `[FASE 2] Similaridades calculadas:` com lista de conceitos
3. ✅ Mensagens `CHUNK X - INÍCIO` e `CHUNK X - FINAL` com timing
4. ✅ Mensagem `EXPERIMENTO FINALIZADO` com tempo total

Se **TODAS** as validações passarem → Prosseguir com experimentos completos.

Se **ALGUMA** falhar → Debug antes de rodar experimentos longos.

---

## 🚀 EXECUTAR EXPERIMENTOS

### Ordem Recomendada

**Experimento 1: TEST_SINGLE (Fase 1 apenas)**
```bash
python main.py config_test_single.yaml --run_number 3
```
**Objetivo:** Validar se logging de timing está funcionando.
**Tempo estimado:** ~6-8h

---

**Experimento 2: DRIFT_RECOVERY (Fase 1 + Fase 2)**
```bash
python main.py config_test_drift_recovery.yaml --run_number 3
```
**Objetivo:** Validar se Fase 2 está detectando recorrência (c1 → c5).
**Tempo estimado:** ~6-8h

---

**Experimento 3: MULTI_DRIFT (Fase 1 + Fase 2)**
```bash
python main.py config_test_multi_drift.yaml --run_number 3
```
**Objetivo:** Validar comportamento com múltiplos drifts.
**Tempo estimado:** ~8-10h

---

## 📊 ANÁLISE PÓS-EXPERIMENTO

### 1. Verificar Fase 2 Ativa

```bash
# Buscar por mensagens de Fase 2 nos logs
grep "FASE 2" experimento_test_drift_recovery3.log | head -20
grep "FASE 2" experimento_test_multi_drift3.log | head -20
```

**Esperado:**
- Deve aparecer `[FASE 2] Chunk 0 - Calculando concept fingerprint...`
- Deve aparecer `[FASE 2] Similaridades calculadas:` com valores numéricos
- Deve aparecer `[FASE 2] ✓ NOVO CONCEITO: 'concept_0'` ou `✓ CONCEITO RECORRENTE: 'concept_X'`

Se **ZERO mensagens** → Problema crítico (arquivos não sincronizados ou erro de execução)

### 2. Verificar Timing

```bash
# Extrair tempo total de cada experimento
grep "Tempo total:" experimento_test_single3.log
grep "Tempo total:" experimento_test_drift_recovery3.log
grep "Tempo total:" experimento_test_multi_drift3.log
```

**Esperado:**
- Linha com `Tempo total: XXXXs (X.XXh)`
- Linha com `Média por chunk: XXXs`

### 3. Verificar Similaridades (CRÍTICO para Fase 2)

```bash
# Extrair similaridades calculadas para DRIFT_RECOVERY
grep -A 10 "Similaridades calculadas:" experimento_test_drift_recovery3.log
```

**Esperado para DRIFT_RECOVERY:**
- Chunk 0: Nenhuma similaridade (primeiro conceito)
- Chunk 1-3: Similaridades baixas (conceitos diferentes)
- **Chunk 4:** Deve ter alta similaridade com concept_0 (recorrência de c1)

Se threshold = 0.85:
- `concept_0: 0.9XXX` → MATCH (recorrente)
- `concept_1: 0.4XXX` → no match
- etc.

Se **todas similaridades < 0.85** → Threshold muito alto, considerar reduzir para 0.75

---

## 🐛 TROUBLESHOOTING

### Problema 1: Fase 2 Não Aparece nos Logs

**Sintoma:** Zero mensagens com `[FASE 2]`

**Diagnóstico:**
```bash
# Verificar se arquivos foram realmente atualizados
grep "logger.warning.*FASE 2" main.py
grep "logger.warning.*Similaridades" utils.py
```

**Soluções:**
1. Re-upload de main.py e utils.py
2. Verificar que está executando a versão correta (não versão cached)
3. Adicionar `import importlib; importlib.reload(utils)` no main.py

---

### Problema 2: Erro "ModuleNotFoundError: No module named 'scipy'"

**Sintoma:** Exceção ao tentar importar scipy

**Solução:**
```bash
pip install scipy
```

Ou adicionar ao início do main.py:
```python
try:
    from scipy.spatial.distance import cosine
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'scipy'])
    from scipy.spatial.distance import cosine
```

---

### Problema 3: Timing Não Aparece

**Sintoma:** Não há mensagens de `EXPERIMENTO INICIADO` ou `CHUNK X - INÍCIO`

**Diagnóstico:**
```bash
# Verificar se logging de timing está no main.py
grep "experiment_start_time" main.py
grep "EXPERIMENTO INICIADO" main.py
```

**Soluções:**
1. Confirmar que main.py foi atualizado
2. Verificar que não há erro de sintaxe impedindo execução

---

### Problema 4: Similaridades Sempre Baixas (< 0.5)

**Sintoma:** Fase 2 funciona mas nunca detecta recorrência

**Diagnóstico:**
- Verificar nos logs quais são os valores de similaridade
- Verificar se fingerprint está sendo calculada corretamente

**Possíveis Causas:**
1. `numeric_features` vazio ou incorreto
2. Dados normalizados incorretamente
3. Threshold muito alto (0.85)

**Soluções:**
1. Adicionar logging de `numeric_features` no main.py
2. Verificar que `calculate_concept_fingerprint()` está usando features corretas
3. **Reduzir threshold de 0.85 para 0.75** (Layer 3 do plano)

---

## 📋 RESUMO DO FLUXO

```
┌─────────────────────────────────────┐
│ 1. VALIDAR SINTAXE LOCAL            │
│    ✅ main.py, utils.py             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. UPLOAD PARA COLAB                │
│    - main.py                         │
│    - utils.py                        │
│    - configs (se alterados)          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. VALIDAR PÓS-UPLOAD               │
│    - grep "FASE 2" main.py           │
│    - grep "Similaridades" utils.py   │
│    - Test scipy import               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 4. SMOKE TEST (2 chunks)            │
│    - Verificar logs aparecem         │
│    - Verificar timing funciona       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 5. EXECUTAR EXPERIMENTOS COMPLETOS  │
│    - TEST_SINGLE (run 3)             │
│    - DRIFT_RECOVERY (run 3)          │
│    - MULTI_DRIFT (run 3)             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 6. ANÁLISE DOS LOGS                 │
│    - Fase 2 ativa?                   │
│    - Similaridades aparecem?         │
│    - Recorrência detectada?          │
│    - Timing coerente?                │
└─────────────────────────────────────┘
```

---

## ✅ CHECKLIST FINAL PRÉ-EXECUÇÃO

Antes de rodar experimentos completos, confirmar:

- [ ] `main.py` e `utils.py` foram uploadados para o Colab
- [ ] Validação `grep "FASE 2" main.py` mostra linhas de código
- [ ] Validação `grep "Similaridades" utils.py` mostra linha de código
- [ ] Scipy está instalado (`from scipy.spatial.distance import cosine` funciona)
- [ ] Smoke test com 2 chunks passou (logs de Fase 2 e timing apareceram)
- [ ] Espaço em disco suficiente para logs (~500MB por experimento)
- [ ] Tempo disponível para execução (~24h total para os 3 experimentos)

**Se TODOS os itens estão checados → PROSSEGUIR COM EXPERIMENTOS**

**Se ALGUM item falhou → DEBUG antes de prosseguir**

---

**FIM DO CHECKLIST**

**Próxima ação:** Upload de arquivos para Google Colab e validação pós-upload.
