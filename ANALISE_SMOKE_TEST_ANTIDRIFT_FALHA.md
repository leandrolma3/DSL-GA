# ANALISE SMOKE TEST ANTI-DRIFT - FALHA CRITICA

**Data:** 2025-11-05
**Experimento:** smoke_test_antidrift (Run998)
**Status:** FALHA - Validacao cruzada nao funcionou
**Severidade:** CRITICA

---

## RESUMO EXECUTIVO

O smoke test foi executado mas a validacao cruzada FALHOU em todos os chunks devido a erro no codigo:

```
'Individual' object has no attribute 'predict'
```

**Resultado:** Validacao sempre assumiu match valido, Chunk 2 teve drift severo novamente.

---

## RESULTADOS DO SMOKE TEST

### Performance por Chunk

| Chunk | Test G-mean | Concept Match | Validation | Status |
|-------|-------------|---------------|------------|--------|
| 0 | 0.8802 | - | - | OK |
| 1 | 0.9184 | concept_0 (sim=0.9963) | ERRO | OK |
| 2 | **0.4155** | concept_0 (sim=0.9952) | ERRO | **CATASTROFICO** |
| 3 | 0.8692 | concept_0 (sim=0.9857) | ERRO | OK |
| 4 | 0.8098 | concept_0 (sim=0.9827) | ERRO | OK |

**Metricas finais:**
- Average Test G-mean: 0.7786
- Std Test G-mean: 0.1849

---

## COMPARACAO COM EXPERIMENTOS ANTERIORES

| Experimento | Chunks | Avg G-mean | Chunk 2 G-mean | Status |
|-------------|--------|------------|----------------|--------|
| Run5 | 5 | 0.7852 | 0.4377 | Drift severo |
| Run998 (smoke test) | 5 | **0.7786** | **0.4155** | **PIOR** |

**Conclusao:** Smoke test ficou PIOR que Run5 (-0.66 pontos).

---

## CAUSA RAIZ IDENTIFICADA

### Problema: Metodo predict() nao existe

**Codigo errado (main.py:635):**
```python
predictions = [best_memory_individual.predict(inst) for inst in validation_data]
```

**Erro gerado:**
```
[FASE 2]   Erro na validacao: 'Individual' object has no attribute 'predict'
```

**Consequencia:**
- Codigo cai no bloco except
- Log: "Mantendo match (validacao falhou, assumindo valido)"
- TODOS os matches foram aceitos sem validacao
- Chunk 2 herdou memoria ruim do concept_0

---

## SOLUCAO

### Individual tem metodo _predict, nao predict

**Arquivo:** individual.py:199-205

```python
def _predict(self, instance):
    """
    Preve o rotulo da classe para uma unica instancia usando a logica de desempate por especificidade.
    """
    # Implementacao...
```

**Correcao necessaria (main.py:635):**
```python
# ANTES (ERRADO):
predictions = [best_memory_individual.predict(inst) for inst in validation_data]

# DEPOIS (CORRETO):
predictions = [best_memory_individual._predict(inst) for inst in validation_data]
```

---

## IMPACTO DA FALHA

### 1. Validacao Cruzada Nunca Executou

**Evidencias do log:**
- 4x "Validando match com amostra do test set..." (chunks 1, 2, 3, 4)
- 4x "Erro na validacao: 'Individual' object has no attribute 'predict'"
- 0x "MATCH VALIDADO" ou "MATCH REJEITADO"

**Resultado:** Nenhum match foi validado, todos foram aceitos.

---

### 2. Chunk 2 Teve Drift Severo Novamente

**Chunk 2 (concept_0, sim=0.9952):**
- Train G-mean: (nao mostrado, mas presumivelmente alto)
- Test G-mean: 0.4155
- Severe drift detected: 0.918 -> 0.416 (drop 50.3%)

**Por que aconteceu:**
- Similarity 0.9952 >= 0.90: Match aceito
- Validacao cruzada falhou (erro no codigo)
- Match assumido valido (fallback)
- Herdou 50% da memoria do concept_0
- concept_0 era incompativel com Chunk 2
- Overfitting massivo

---

### 3. Performance Pior que Run5

**Run5 (sem anti-drift):**
- Avg G-mean: 0.7852
- Chunk 2: 0.4377

**Run998 (com anti-drift quebrado):**
- Avg G-mean: 0.7786 (-0.66 pontos)
- Chunk 2: 0.4155 (-0.22 pontos)

**Possivel causa da piora adicional:**
- Threshold 0.90 mais restritivo
- Mas validacao nao funcionou
- Pode ter afetado outros mecanismos

---

## LOGS CRITICOS

### Chunk 1
```
2025-11-04 19:33:12 [WARNING] [FASE 2]   Threshold similaridade: 0.90
2025-11-04 19:33:12 [WARNING] [FASE 2] Similaridades calculadas:
2025-11-04 19:33:12 [WARNING] [FASE 2] MATCH: 'concept_0' (sim=0.9963 >= 0.90)
2025-11-04 19:33:12 [WARNING] [FASE 2]   CONCEITO RECORRENTE: 'concept_0'
2025-11-04 19:33:12 [WARNING] [FASE 2] Validando match com amostra do test set...
2025-11-04 19:33:12 [WARNING] [FASE 2]   Erro na validacao: 'Individual' object has no attribute 'predict'
```

### Chunk 2 (Drift Severo)
```
2025-11-05 00:16:56 [WARNING] [FASE 2] MATCH: 'concept_0' (sim=0.9952 >= 0.90)
2025-11-05 00:16:56 [WARNING] [FASE 2]   CONCEITO RECORRENTE: 'concept_0'
2025-11-05 00:16:56 [WARNING] [FASE 2] Validando match com amostra do test set...
2025-11-05 00:16:56 [WARNING] [FASE 2]   Erro na validacao: 'Individual' object has no attribute 'predict'
2025-11-05 00:16:56 [WARNING] [FASE 2]   → Mantendo match (validacao falhou, assumindo valido)

[... treino ...]

2025-11-05 00:16:55 [WARNING] SEVERE DRIFT detected: 0.918 → 0.416 (drop: 50.3%)
2025-11-05 00:16:55 [WARNING]    → Inheritance REDUCED to 20% due to SEVERE drift (was 50%)
```

---

## VALIDACAO DO DIAGNOSTICO

### Threshold 0.90 Funcionou

```
grep "Threshold similaridade" smoke_test_antidrift.log
```

**Resultado:** 5 ocorrencias de "Threshold similaridade: 0.90" - OK

---

### Validacao Cruzada Tentou Executar

```
grep "Validando match" smoke_test_antidrift.log
```

**Resultado:** 4 ocorrencias (chunks 1, 2, 3, 4) - OK

---

### Validacao Sempre Falhou

```
grep "Erro na validacao" smoke_test_antidrift.log
```

**Resultado:** 4 ocorrencias de "'Individual' object has no attribute 'predict'" - PROBLEMA

---

### Nunca Houve Match Validado ou Rejeitado

```
grep "MATCH VALIDADO\|MATCH REJEITADO" smoke_test_antidrift.log
```

**Resultado:** 0 ocorrencias - PROBLEMA CONFIRMADO

---

## CORRECAO NECESSARIA

### Arquivo: main.py

**Linha 635:**

```python
# ANTES (ERRADO):
predictions = [best_memory_individual.predict(inst) for inst in validation_data]

# DEPOIS (CORRETO):
predictions = [best_memory_individual._predict(inst) for inst in validation_data]
```

**Justificativa:**
- Classe Individual tem metodo _predict (com underscore)
- Nao tem metodo predict (sem underscore)
- Confirmado em individual.py:199

---

## PROJECAO COM CORRECAO

### Chunk 2 com Validacao Funcionando

**Cenario esperado:**
1. Chunk 2: sim=0.9952 com concept_0 (>= 0.90, match aceito)
2. Validacao cruzada executa com 20% do test set
3. Validation G-mean: ~0.50 (ruim, pois conceitos incompativeis)
4. 0.50 < 0.70: Match rejeitado
5. Chunk 2 tratado como NOVO CONCEITO
6. Sem heranca do concept_0
7. Treino do zero
8. Test G-mean esperado: 0.80-0.85

**Impacto na media:**
- Chunk 2: 0.4155 -> 0.82 (+40 pontos)
- Avg G-mean: 0.7786 -> 0.86 (+8 pontos)

---

## PLANO DE ACAO

### Fase 1: Correcao Imediata (5 minutos)

1. Editar main.py linha 635
2. Trocar `.predict(` por `._predict(`
3. Validar sintaxe: `python -m py_compile main.py`

---

### Fase 2: Re-executar Smoke Test (3-4h)

**Comando:**
```bash
cd /content/drive/MyDrive/DSL-AG-hybrid && python main.py config_test_single.yaml --num_chunks 3 --run_number 999 2>&1 | tee smoke_test_antidrift_v2.log
```

**O que verificar:**
1. Nenhum erro de validacao
2. Logs de "MATCH VALIDADO" ou "MATCH REJEITADO"
3. Chunk 2 com Test G-mean >= 0.75
4. Avg G-mean >= 0.82

---

### Fase 3: Run6 Completo (se smoke test OK)

**Comando:**
```bash
cd /content/drive/MyDrive/DSL-AG-hybrid && python main.py config_test_single.yaml --num_chunks 5 --run_number 6 2>&1 | tee experimento_run6.log
```

---

## METRICAS DE SUCESSO (Smoke Test v2)

Correcao bem-sucedida se:

1. **Validacao cruzada executa sem erros:**
   - 0 logs de "Erro na validacao"

2. **Matches sendo validados ou rejeitados:**
   - Logs de "MATCH VALIDADO" OU "MATCH REJEITADO" aparecem

3. **Chunk 2 sem drift severo:**
   - Test G-mean >= 0.75
   - Delta < 0.15

4. **Avg G-mean melhora:**
   - >= 0.82 (vs 0.7786 atual)

---

## TROUBLESHOOTING

### Se validacao ainda falhar

**Verificar:**
1. validation_data esta no formato correto (lista de dicionarios)
2. Metodo _predict aceita esse formato
3. Classes estao definidas corretamente

**Alternativa:**
- Usar mecanismo de predicao existente em fitness.py
- Avaliar todo o individuo de uma vez

---

## LICAO APRENDIDA

1. **SEMPRE testar com dados reais antes de experimento longo**
   - Smoke test revelou bug critico
   - Economizou 6+ horas de Run6 com bug

2. **Validar chamadas de metodos privados**
   - Metodos com underscore (_predict) nao sao publicos
   - Verificar assinatura antes de usar

3. **Nao assumir que except resolve tudo**
   - Fallback "assumindo valido" escondeu o problema
   - Melhor: logar erro e abortar experimento

---

**Status:** DIAGNOSTICO COMPLETO, CORRECAO IDENTIFICADA
**Prioridade:** CRITICA - Corrigir antes de qualquer experimento
**Tempo estimado:** 5min correcao + 3-4h smoke test v2
