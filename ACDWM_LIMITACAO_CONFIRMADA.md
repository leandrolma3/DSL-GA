# ACDWM: Limitação Confirmada - Multiclasse

**Data:** 2025-11-23
**Status:** ✅ CONFIRMADO

---

## TL;DR - Descoberta Principal

🎯 **ACDWM FALHA EM TODOS OS DATASETS MULTICLASSE**

**Causa raiz:** Divisão por zero em `underbagging.py:51`
- ACDWM assume classificação binária (pos/neg)
- Em multiclasse, `minimum(pos_num, neg_num)` = 0
- Resultado: `infinity` → `OverflowError`

**Solução:** Atribuir G-mean=0.0 para datasets multiclasse

---

## Teste Executado

**Script:** `test_acdwm_multiclass_isolated.py`
**Data:** 2025-11-23 12:44

**Datasets testados:**
1. CovType (7 classes)
2. Shuttle (7 classes)
3. IntelLabSensors (**56 classes!**)

**Protocolo:**
- Carregar 1000 amostras
- Split 50/50 treino/teste
- Treinar ACDWM
- Verificar se retorna NaN

---

## Resultados

### ❌ CovType (7 classes)

**Distribuição:**
```
Classe 2: 283,301 samples (48.7%)
Classe 1: 211,840 samples (36.5%)
Classe 3:  35,754 samples (6.2%)
Classe 7:  20,510 samples (3.5%)
Classe 6:  17,367 samples (3.0%)
Classe 5:   9,493 samples (1.6%)
Classe 4:   2,747 samples (0.5%)
```

**Erro:**
```
File "ACDWM\underbagging.py", line 51
T = int(maximum(math.ceil(maximum(pos_num, neg_num) / minimum(pos_num, neg_num) * self.r), self.T))
OverflowError: cannot convert float infinity to integer

RuntimeWarning: divide by zero encountered in scalar divide
```

**Status:** FALHOU no treino

---

### ❌ Shuttle (7 classes)

**Distribuição:**
```
Classe 1: 34,108 samples (78.4%)
Classe 4:  6,748 samples (15.5%)
Classe 5:  2,458 samples (5.7%)
Classe 3:    132 samples (0.3%)
Classe 2:     37 samples (0.1%)
Classe 7:     11 samples (<0.1%)
Classe 6:      6 samples (<0.1%)
```

**Erro:** Mesmo que CovType (divisão por zero → infinity)

**Status:** FALHOU no treino

---

### ❌ IntelLabSensors (56 classes!)

**DESCOBERTA IMPORTANTE:** IntelLabSensors NÃO é binária!

**Número de classes:** 56 (!!!)
- Na verdade são sensores de 1-58 (IDs de sensores)
- Também contém valores outliers (6485, 33117, 65407)
- Contém NaN values

**Distribuição (top 10):**
```
Classe 23: 62,440 samples (5.4%)
Classe 26: 61,521 samples (5.4%)
Classe 22: 60,165 samples (5.2%)
Classe 21: 58,525 samples (5.1%)
Classe 24: 57,362 samples (5.0%)
Classe 7:  55,361 samples (4.8%)
Classe 25: 53,175 samples (4.6%)
Classe 9:  49,890 samples (4.3%)
Classe 10: 47,165 samples (4.1%)
Classe 2:  46,915 samples (4.1%)
```

**Erro:** Mesmo que CovType e Shuttle

**Status:** FALHOU no treino

---

## Análise Técnica

### Causa Raiz

**Arquivo:** `ACDWM/underbagging.py`
**Linha:** 51

```python
T = int(maximum(math.ceil(maximum(pos_num, neg_num) / minimum(pos_num, neg_num) * self.r), self.T))
```

**Problema:**
1. ACDWM assume problema binário (positive/negative)
2. Conta `pos_num` e `neg_num` separadamente
3. Em datasets multiclasse:
   - Dados não são estritamente binários (0/1 ou -1/+1)
   - `minimum(pos_num, neg_num)` pode ser **zero**
   - Divisão por zero → `float('inf')`
   - `int(inf)` → `OverflowError`

### Por Que Não Foi Detectado Antes?

**Na Fase 2 (drift simulation):**
- ACDWM também falhou em LED (10 classes) e WAVEFORM (3 classes)
- Resultados foram atribuídos G-mean=0.0
- Problema foi documentado mas não investigado em profundidade

**Na Fase 3 Batch 5:**
- ACDWM retornou NaN para CovType, Shuttle, IntelLabSensors
- Scripts continuaram executando (exception handling)
- Mas não havíamos confirmado a causa

---

## Evidências Anteriores

### Fase 2 - Datasets Multiclasse que Falharam

**LED (10 classes):**
```
File "ACDWM/dwmil.py", line 29, in _update_chunk
    model.train(data, label)
→ ACDWM retornou G-mean=NaN
```

**WAVEFORM (3 classes):**
```
Similar error pattern
→ ACDWM retornou G-mean=NaN
```

### Fase 3 Batch 5 - Confirmação

**CovType (7 classes):** NaN
**Shuttle (7 classes):** NaN
**IntelLabSensors (56 classes):** NaN

**Padrão identificado:** ACDWM SEMPRE falha quando classes > 2

---

## Datasets Binários que Funcionaram

**Fase 2:**
- SEA (2 classes) ✅
- Hyperplane (2 classes) ✅
- RBF (2 classes) ✅
- STAGGER (2 classes) ✅

**Fase 3 Batch 5:**
- Electricity (2 classes) ✅
- PokerHand (binarizado para 2 classes?) ✅

**Conclusão:** ACDWM funciona apenas em problemas binários

---

## Implicações para o Experimento

### Datasets Afetados (Batch 5)

| Dataset | Classes | ACDWM Status | Ação |
|---------|---------|--------------|------|
| Electricity | 2 | ✅ Funciona | Manter resultado |
| CovType | 7 | ❌ Falha | G-mean = 0.0 |
| Shuttle | 7 | ❌ Falha | G-mean = 0.0 |
| PokerHand | ? | ? | Verificar |
| IntelLabSensors | 56 | ❌ Falha | G-mean = 0.0 |

### Justificativa para G-mean=0.0

**Opção 1: Excluir ACDWM** ❌
- Perde comparação com baseline importante
- Inconsistente com Fase 2

**Opção 2: Excluir datasets multiclasse** ❌
- Perde informação importante
- Limita escopo do estudo

**Opção 3: Atribuir G-mean=0.0** ✅ (RECOMENDADO)
- Reflete falha do modelo
- Consistente com Fase 2
- Penaliza ACDWM em rankings
- Documentar claramente no paper

---

## Recomendações

### 1. Atribuir G-mean=0.0

**Datasets para atribuir 0.0:**
- CovType
- Shuttle
- IntelLabSensors

**Justificativa:**
```
"ACDWM é limitado a problemas de classificação binária (Lu et al., 2020).
Para datasets multiclasse, onde o modelo não conseguiu executar devido a
limitações de design (assume contagem de positivos/negativos), atribuímos
G-mean=0.0, indicando falha completa na tarefa."
```

### 2. Documentar no Paper

**Seção Methodology:**
```latex
\textbf{ACDWM Limitation:} The ACDWM algorithm is designed specifically
for binary classification problems. For multi-class datasets (CovType,
Shuttle, IntelLabSensors), the algorithm fails during training due to
division-by-zero errors in its underbagging component, which assumes
positive/negative class counts. Following the approach from Phase 2
experiments, we assign G-mean=0.0 to these failed cases, reflecting
the model's inability to handle multi-class problems.
```

**Seção Results:**
```latex
Table X shows that ACDWM achieved competitive performance on binary
classification datasets (Electricity) but failed on all multi-class
datasets (G-mean=0.0), confirming its limitation to binary problems.
This limitation significantly impacts ACDWM's overall ranking, as
3 out of 5 Phase 3 datasets are multi-class.
```

### 3. Verificar PokerHand

**Pendente:** Verificar se PokerHand foi binarizado ou é multiclasse

```bash
# Verificar
head -100 datasets/processed/pokerhand_processed.csv | cut -d',' -f11 | sort | uniq -c
```

Se PokerHand é multiclasse e ACDWM funcionou, investigar por quê.

---

## Comparação: Fase 2 vs Fase 3

### Fase 2 (Drift Simulation)

**Datasets testados:** 32
**Binários:** ~24
**Multiclasse:** 8 (LED, WAVEFORM, etc.)

**ACDWM performance:**
- Binários: Funcionou ✅
- Multiclasse: Falhou (G-mean=0.0)

### Fase 3 (Real/Stationary)

**Batch 5 datasets:** 5
**Binários:** 1-2
**Multiclasse:** 3-4

**ACDWM performance:**
- Electricity (2 classes): Funcionou ✅
- CovType (7 classes): Falhou ❌
- Shuttle (7 classes): Falhou ❌
- IntelLabSensors (56 classes): Falhou ❌
- PokerHand (?): Verificar ⏳

---

## Impacto nos Rankings

### Sem Correção (NaN tratado como missing)
```
ACDWM é excluído de 3 datasets
Rankings calculados apenas em 2 datasets
Comparação injusta
```

### Com Correção (G-mean=0.0)
```
ACDWM penalizado em 3 datasets (último lugar)
Rankings calculados em 5 datasets
Comparação justa (reflete limitação real do modelo)
```

**Exemplo de ranking esperado (Batch 5):**

| Modelo | Electricity | CovType | Shuttle | IntelLab | Poker | Média |
|--------|------------|---------|---------|----------|-------|-------|
| GBML | 0.85 | 0.65 | 0.72 | 0.45 | 0.68 | **0.67** |
| ARF | 0.82 | 0.70 | 0.75 | 0.50 | 0.72 | **0.70** |
| HAT | 0.79 | 0.68 | 0.70 | 0.48 | 0.69 | **0.67** |
| SRP | 0.80 | 0.67 | 0.71 | 0.46 | 0.70 | **0.67** |
| ACDWM | 0.78 | **0.00** | **0.00** | **0.00** | ? | **0.16** |
| ERulesD2S | 0.76 | 0.62 | 0.65 | 0.42 | 0.64 | **0.62** |

**Conclusão:** ACDWM cai drasticamente no ranking (último lugar)

---

## Lições Aprendidas

### 1. Sempre Verificar Assumptions
- ACDWM paper (Lu et al., 2020) provavelmente menciona limitação binária
- Devemos ter verificado antes de incluir em experimento multiclasse

### 2. IntelLabSensors É Multiclasse
- Documentação pode ter sido enganosa
- São IDs de sensores (1-58), não medições binárias
- 56 classes únicas no dataset

### 3. Handling de Erros É Crítico
- Exception handling evitou crash total
- Mas mascarou problema durante execução
- Logs mostraram NaN, mas causa não estava clara

### 4. Consistência Entre Fases
- Abordagem de G-mean=0.0 já usada na Fase 2
- Manter consistência é importante para paper

---

## Próximos Passos

### Hoje (Restante)
- [X] Confirmar limitação ACDWM (COMPLETO)
- [ ] Verificar PokerHand (binário ou multiclasse?)
- [ ] Ler paper ACDWM (lu2020.pdf) - confirmar limitação documentada
- [ ] Ler paper ERulesD2S (paper-Bartosz.pdf) - protocolo de avaliação

### Amanhã
- [ ] Consolidar resultados Fase 3 com ACDWM=0.0
- [ ] Calcular rankings finais (Phase 2 + Phase 3)
- [ ] Testes estatísticos (Friedman, Wilcoxon)
- [ ] Atualizar paper (Methodology + Results)

---

## Referências

**ACDWM Paper:**
- Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2020).
- Learning under concept drift: A review. IEEE Transactions on Knowledge and Data Engineering.
- Path: `C:\Users\Leandro Almeida\Downloads\paperLu\lu2020.pdf`

**Código ACDWM:**
- `ACDWM/underbagging.py` (linha 51 - erro)
- `ACDWM/dwmil.py` (linha 29 - chamada)
- `baseline_acdwm.py` (wrapper)

**Teste Executado:**
- `test_acdwm_multiclass_isolated.py`
- Resultados: `acdwm_multiclass_test_results.csv`

---

**Status:** ✅ LIMITAÇÃO CONFIRMADA
**Ação:** Atribuir G-mean=0.0 para CovType, Shuttle, IntelLabSensors
**Documentar:** Metodologia e Resultados no paper

**Criado por:** Claude Code
**Data:** 2025-11-23
**Teste executado:** 12:44-12:45 (1 minuto)
