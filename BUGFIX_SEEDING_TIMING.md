# 🐛 BUGFIX: Seeding 85% - Problema de Timing

**Data**: 2025-10-24
**Status**: ✅ **CORRIGIDO** (com limitações)
**Severidade**: 🔴 **CRÍTICA**

---

## 🐛 PROBLEMA IDENTIFICADO

### Observação no Log (Fase 1-Novo, Chunk 4)

**Esperado**:
```
  -> SEVERE DRIFT DETECTED: Seeding INTENSIVO ativado (85% seeding, 90% injection)
     Parâmetros adaptativos: seeding_ratio=0.85, injection_ratio=0.90
População de reset criada: 120 indivíduos (102 semeados, 18 aleatórios).
```

**Observado**:
```
  -> Complexidade estimada: MEDIUM (DT probe acc: 0.790)
     Parâmetros adaptativos: seeding_ratio=0.6, injection_ratio=0.6
População de reset criada: 120 indivíduos (72 semeados, 48 aleatórios).
```

**Resultado**: Seeding 85% **NÃO foi aplicado** no chunk 4 (drift SEVERE).

---

## 🔍 CAUSA RAIZ: Problema de Timing

### Fluxo Atual (INCORRETO para chunk 4)

```
1. Chunk 3→4:
   - Treina GA no chunk 3
   - Testa no chunk 4
   - Resultado: G-mean = 87.40%
   - Drift detectado: STABLE ou IMPROVED (performance boa)

2. Chunk 4→5:
   - INICIA treino do GA no chunk 4
   - drift_severity_to_pass = 'STABLE' (do chunk anterior)
   - Seeding: 60% (complexidade MEDIUM)
   - Executa GA completo
   - TESTA no chunk 5
   - Resultado: G-mean = 39.00% (COLAPSO!)
   - Drift detectado: SEVERE (mas GA já executou!)
   - drift_severity = 'SEVERE' (guardado para próximo chunk)

3. Chunk 5→6 (NÃO EXISTE no experimento):
   - Seria aqui que usaria seeding 85%!
   - drift_severity_to_pass = 'SEVERE'
```

### Problema Fundamental

O sistema detecta drift **APÓS** executar o chunk, então:
- **Drift detectado no chunk N** só é usado no **chunk N+1**
- **Chunk 4** (onde queríamos seeding 85%) usa drift do **chunk 3** (STABLE)
- **Seeding 85% seria usado no chunk 5** (que não existe)

---

## ✅ CORREÇÃO APLICADA

### Mudança 1: Heurística Preditiva (main.py linha 779-782)

```python
# HEURÍSTICA PREDITIVA: Se não há histórico ainda, mas performance foi muito baixa, assume SEVERE
if len(historical_gmean) >= 1 and historical_gmean[-1] < 0.50:
    logger.warning(f"Chunk {i}: Previous chunk had very low G-mean ({historical_gmean[-1]:.3f}) - assuming SEVERE drift preventively")
    drift_severity_to_pass = 'SEVERE'
```

**Lógica**:
- Se o chunk **anterior** teve G-mean < 50% (muito ruim), assume drift SEVERE
- **Chunk 5** (se existisse) receberia `drift_severity='SEVERE'` e usaria seeding 85%

**Limitação**:
- Só funciona se houver um **próximo chunk** após o drift
- No experimento atual (5 chunks), **não há chunk 5** para aplicar seeding 85%

---

### Mudança 2: Log de Drift Passado (main.py linha 773-777)

```python
# Log do drift sendo passado para o GA
if drift_severity_to_pass != 'NONE':
    logger.info(f"Chunk {i}: Using drift_severity='{drift_severity_to_pass}' from previous chunk for GA adaptation")
```

**Propósito**: Diagnóstico - ver qual drift está sendo passado para cada chunk.

---

## 📊 IMPACTO DA CORREÇÃO

### Cenário 1: Experimento com 6+ chunks

Se tivéssemos **chunk 5** (treinar após chunk 4→5 SEVERE):

| Chunk | Drift Anterior | drift_severity_to_pass | Seeding | Observação |
|-------|----------------|------------------------|---------|------------|
| 0 | N/A | `'NONE'` | 60% | Primeiro chunk |
| 1 | `'NONE'` | `'NONE'` | 60% | Ainda sem drift |
| 2 | `'STABLE'` | `'STABLE'` | 60% | Performance boa |
| 3 | `'MILD'` | `'MILD'` | 60% | Drift leve |
| 4 | `'STABLE'` | `'STABLE'` | 60% | ❌ Chunk crítico, mas sem adaptação |
| **5** | **`'SEVERE'`** | **`'SEVERE'`** | **85%** ✅ | **Seeding intensivo aplicado!** |

**Resultado esperado**:
- Chunk 4→5: 39.00% (igual, não tem jeito)
- **Chunk 5→6**: **Melhor** que seria sem correção (seeding 85% ajuda recovery)

---

### Cenário 2: Experimento atual (5 chunks) ⚠️

**Problema**: Não há chunk 5, então seeding 85% **nunca é usado**.

| Chunk | Observação |
|-------|------------|
| 0-3 | Performance boa, sem drift SEVERE |
| 4 | ❌ SEVERE drift, mas já é último chunk |
| 5 | ❌ **Não existe** - seeding 85% não é testado |

**Conclusão**: Com apenas 5 chunks, **a correção não tem efeito prático**.

---

## 🎯 SOLUÇÃO IDEAL (Não implementada)

### Opção A: Detecção de Drift Preditiva

Detectar drift **ANTES** de treinar, usando:
- Análise da distribuição dos dados de treino (concept drift detection)
- Comparar chunk atual com chunks anteriores (e.g., KS-test, JS-divergence)
- Se divergência > threshold, assume drift e usa seeding 85%

**Vantagens**: Seeding 85% usado no chunk **onde o drift ocorre**
**Desvantagens**: Complexo, requer bibliotecas extras (e.g., skmultiflow)

---

### Opção B: Teste em 2 Fases

1. **Fase 1**: Treina GA com seeding padrão (60%)
2. **Teste intermediário**: Avalia no chunk de teste
3. **Fase 2**: Se G-mean < 50%, re-inicializa com seeding 85%

**Vantagens**: Seeding 85% usado **no mesmo chunk** onde drift ocorre
**Desvantagens**:
- Requer executar GA 2 vezes (custo computacional)
- Lógica mais complexa

---

### Opção C: Experimento com 7-8 Chunks ⭐ **RECOMENDADO**

Adicionar mais chunks ao experimento para ter:
- Chunk 4→5: SEVERE drift (G-mean 39%)
- **Chunk 5**: Usa seeding 85% ✅
- **Chunk 6**: Testa se seeding 85% ajudou recovery

**Vantagens**:
- Testa correção sem mudanças complexas
- Valida se seeding 85% realmente ajuda

**Desvantagens**:
- Experimento mais longo (~16-18h)

---

## ✅ VALIDAÇÃO DA CORREÇÃO

### Como Validar (Experimento com 6+ chunks)

**Esperado no log do Chunk 5**:
```
2025-XX-XX XX:XX:XX [INFO] main: Chunk 5: Using drift_severity='SEVERE' from previous chunk for GA adaptation
2025-XX-XX XX:XX:XX [INFO] root:   -> SEEDING ADAPTATIVO ATIVADO: Estimando complexidade do chunk...
2025-XX-XX XX:XX:XX [INFO] root:   -> Complexidade estimada: MEDIUM (DT probe acc: 0.XXX)
2025-XX-XX XX:XX:XX [INFO] root:   -> SEVERE DRIFT DETECTED: Seeding INTENSIVO ativado (85% seeding, 90% injection)
2025-XX-XX XX:XX:XX [INFO] root:      Parâmetros adaptativos: seeding_ratio=0.85, injection_ratio=0.90, depths=[5, 8, 10]
2025-XX-XX XX:XX:XX [INFO] root:   -> Seeding Probabilístico ATIVADO: Injetando 90% das regras DT
2025-XX-XX XX:XX:XX [INFO] root: População de reset criada: 120 indivíduos (102 semeados, 18 aleatórios).
```

**Checklist**:
- [ ] Mensagem "Using drift_severity='SEVERE'" aparece
- [ ] Mensagem "SEVERE DRIFT DETECTED: Seeding INTENSIVO" aparece
- [ ] `seeding_ratio=0.85` e `injection_ratio=0.90`
- [ ] População tem **102 semeados**, não 72

---

### Como Validar (Heurística Preditiva)

**Se chunk N teve G-mean < 50%**, chunk N+1 deve mostrar:
```
2025-XX-XX XX:XX:XX [WARNING] main: Chunk 5: Previous chunk had very low G-mean (0.390) - assuming SEVERE drift preventively
2025-XX-XX XX:XX:XX [INFO] main: Chunk 5: Using drift_severity='SEVERE' from previous chunk for GA adaptation
```

---

## 🚦 DECISÃO: O QUE FAZER?

### **OPÇÃO A**: Executar Experimento com 7-8 Chunks ⭐ **RECOMENDADO**

**Ação**:
1. Modificar config.yaml para gerar 7-8 chunks
2. Re-executar experimento (~16-18h)
3. Validar se chunk 5 usa seeding 85%
4. Ver se chunk 5→6 tem melhor recovery que 4→5

**Meta**: Chunk 5→6: G-mean ≥ 55-60% (melhor que 39% do 4→5)

---

### **OPÇÃO B**: Implementar Detecção Preditiva

**Ação**:
1. Implementar concept drift detection (KS-test ou JS-divergence)
2. Detectar drift ANTES de treinar
3. Se drift > threshold, usar seeding 85%

**Complexidade**: Alta (requer bibliotecas extras)
**Tempo**: ~3-4h implementação + teste

---

### **OPÇÃO C**: Aceitar Limitação e Focar em HC

**Ação**:
1. Aceitar que chunk 4→5 terá performance ruim (~39-40%)
2. Focar em melhorar HC (tolerância 2%, 25 variantes, **18 variantes corrigido**)
3. Tentar atingir Avg G-mean 83-84% (ao invés de 85%)

**Justificativa**: Drift SEVERE pode ser **inerentemente difícil** para o método atual

---

## 📝 RECOMENDAÇÃO FINAL

**Escolher OPÇÃO A** (experimento com 7-8 chunks) porque:
1. ✅ Testa correção sem mudanças complexas no código
2. ✅ Valida hipótese de seeding 85%
3. ✅ Mais chunks = mais dados para análise
4. ⏱️ Custo: ~18h (aceitável para validação final)

**SE opção A falhar** (seeding 85% não ajudar):
- Considerar **OPÇÃO C** (aceitar limitação, focar em HC)
- Ou tentar **OPÇÃO B** (detecção preditiva) como último recurso

---

**Criado por**: Claude Code
**Data**: 2025-10-24
**Status**: ✅ **CORRIGIDO** (heurística preditiva)
**Limitação**: ⚠️ **Requer experimento com 6+ chunks para validar**
**Próximo Passo**: **Executar experimento com 7-8 chunks OU aceitar limitação e focar em HC**
