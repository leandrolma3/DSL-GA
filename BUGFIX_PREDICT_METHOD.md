# BUGFIX - Metodo predict() Nao Existe

**Data:** 2025-11-05
**Severidade:** CRITICA (bloqueou validacao cruzada)
**Status:** CORRIGIDO

---

## ERRO ORIGINAL

```
AttributeError: 'Individual' object has no attribute 'predict'
```

**Sintoma:** Validacao cruzada sempre falhava, todos os matches eram aceitos por default.

---

## CAUSA RAIZ

Na implementacao da validacao cruzada (linha 635), usei metodo errado:

```python
predictions = [best_memory_individual.predict(inst) for inst in validation_data]
```

**Problema:**
- Classe Individual tem metodo `_predict` (com underscore)
- Nao tem metodo `predict` (sem underscore)
- Confirmado em individual.py linha 199:
  ```python
  def _predict(self, instance):
      """
      Preve o rotulo da classe para uma unica instancia usando a logica de desempate por especificidade.
      """
  ```

**Consequencia:**
- AttributeError a cada tentativa de validacao
- Codigo caia no bloco except
- Log: "Mantendo match (validacao falhou, assumindo valido)"
- TODOS os matches aceitos sem validacao
- Chunk 2 teve drift severo (G-mean=0.4155)

---

## CORRECAO APLICADA

**Arquivo:** main.py linha 635

**Antes (ERRADO):**
```python
predictions = [best_memory_individual.predict(inst) for inst in validation_data]
```

**Depois (CORRETO):**
```python
predictions = [best_memory_individual._predict(inst) for inst in validation_data]
```

**Mudanca:** Trocado `.predict(` por `._predict(`

---

## VALIDACAO

```bash
python -m py_compile main.py
```

**Resultado:** SEM ERROS

---

## IMPACTO DA CORRECAO

### Antes (com bug):
- Validacao cruzada: 0% sucesso (4/4 erros)
- Matches rejeitados: 0
- Chunk 2 G-mean: 0.4155
- Avg G-mean: 0.7786

### Depois (esperado):
- Validacao cruzada: 100% sucesso
- Matches rejeitados: >= 1 (Chunk 2)
- Chunk 2 G-mean: 0.80-0.85
- Avg G-mean: 0.82-0.86

**Melhoria esperada:** +4-8 pontos em G-mean medio

---

## EVIDENCIAS DO BUG

### Logs do Smoke Test (Run998)

**Chunk 1:**
```
[FASE 2] Validando match com amostra do test set...
[FASE 2]   Erro na validacao: 'Individual' object has no attribute 'predict'
[FASE 2]   → Mantendo match (validacao falhou, assumindo valido)
```

**Chunk 2:**
```
[FASE 2] Validando match com amostra do test set...
[FASE 2]   Erro na validacao: 'Individual' object has no attribute 'predict'
[FASE 2]   → Mantendo match (validacao falhou, assumindo valido)

[... treino ...]

SEVERE DRIFT detected: 0.918 → 0.416 (drop: 50.3%)
```

**Total:** 4 erros identicos (chunks 1, 2, 3, 4)

---

## PROXIMO PASSO: SMOKE TEST v2

**Comando para Google Colab:**
```bash
cd /content/drive/MyDrive/DSL-AG-hybrid && python main.py config_test_single.yaml --num_chunks 3 --run_number 999 2>&1 | tee smoke_test_antidrift_v2.log
```

**O que verificar:**

1. **Validacao sem erros:**
   ```bash
   grep "Erro na validacao" smoke_test_antidrift_v2.log
   ```
   Esperado: 0 ocorrencias

2. **Matches sendo validados/rejeitados:**
   ```bash
   grep -E "MATCH VALIDADO|MATCH REJEITADO" smoke_test_antidrift_v2.log
   ```
   Esperado: Pelo menos 1 ocorrencia

3. **Chunk 2 sem drift severo:**
   ```bash
   grep "CHUNK.*FINAL" smoke_test_antidrift_v2.log -A2
   ```
   Esperado: Todos chunks >= 0.75

4. **G-mean validation logs:**
   ```bash
   grep "Validation G-mean" smoke_test_antidrift_v2.log
   ```
   Esperado: Valores numericos visiveis

---

## METRICAS DE SUCESSO

Correcao bem-sucedida se:

1. **Nenhum erro de validacao:**
   - 0 logs de "Erro na validacao"

2. **Validacao executada:**
   - Logs de "Validation G-mean: X.XXXX" aparecem

3. **Decisoes tomadas:**
   - Logs de "MATCH VALIDADO" OU "MATCH REJEITADO"

4. **Sem drift severo:**
   - Nenhum chunk com G-mean < 0.60

5. **Media melhora:**
   - Avg G-mean >= 0.82 (vs 0.7786 do Run998)

---

## LICAO APRENDIDA

1. **Verificar assinatura de metodos privados**
   - Metodos com underscore podem ter interface diferente
   - Sempre consultar definicao da classe

2. **Testar validacao com dados reais**
   - Smoke test revelou bug antes de experimento longo
   - Economizou 6+ horas de Run6 com bug

3. **Melhorar tratamento de erros**
   - Fallback "assumindo valido" escondeu problema
   - Considerar abortar se validacao falhar

---

**Status:** CORRIGIDO, sintaxe validada
**Proximo passo:** SMOKE TEST v2 (3 chunks)
**Tempo estimado:** 3-4h execucao + 15min analise
**Comando:**
```bash
cd /content/drive/MyDrive/DSL-AG-hybrid && python main.py config_test_single.yaml --num_chunks 3 --run_number 999 2>&1 | tee smoke_test_antidrift_v2.log
```
