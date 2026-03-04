# Investigacao ROSE vs GBML - Analise Profunda

**Data:** 2025-11-26
**Status:** CONCLUIDO

---

## 1. Resumo Executivo

Esta investigacao identificou **DOIS PROBLEMAS CRITICOS** na execucao atual do ROSE:

### Problema 1: Evaluator Errado para Multiclasse
O ROSE esta usando `WindowAUCImbalancedPerformanceEvaluator` que e para **BINARIO**.
Para multiclasse deveria usar `WindowAUCMultiClassImbalancedPerformanceEvaluator`.

**Impacto:** ROSE zera em todos os datasets multiclasse (LED, WAVEFORM, Shuttle, CovType).

### Problema 2: Granularidade de Avaliacao Diferente
- **ROSE:** Prequential **instance-by-instance** (aprende apos cada exemplo)
- **GBML:** Prequential **chunk-based** (aprende apos chunk inteiro de 1000+ exemplos)

**Impacto:** ROSE tem vantagem significativa pois adapta mais rapidamente.

---

## 2. Analise Detalhada: Estrategias de Avaliacao

### 2.1 GBML (e River: HAT/ARF/SRP)

```
Protocolo Prequential Chunk-Based:
  Para cada chunk N:
    1. Se N > 0: TESTAR modelo atual no chunk N (metricas)
    2. TREINAR modelo no chunk N inteiro (1000-6000 instancias)
    3. Passar para chunk N+1
```

**Caracteristicas:**
- Treina em batch (todas instancias do chunk de uma vez)
- So atualiza modelo apos processar chunk inteiro
- Heranca entre chunks via memoria de elite e seeding

### 2.2 ROSE (via MOA)

```
Protocolo Prequential Instance-Based (EvaluateInterleavedTestThenTrain):
  Para cada instancia i:
    1. PREDIZER classe de i (antes de treinar)
    2. ATUALIZAR metricas com predicao
    3. TREINAR modelo com (i, classe_real)
    4. Passar para instancia i+1
```

**Caracteristicas:**
- Treina incrementalmente (uma instancia por vez)
- Atualiza modelo apos CADA exemplo
- Adapta muito mais rapidamente a mudancas

### 2.3 Comparacao Visual

```
Timeline de 10 instancias:

GBML (chunk_size=5):
  Inst: [1, 2, 3, 4, 5] -> Treino batch -> [6, 7, 8, 9, 10] -> Treino batch
  Atualizacoes: 2

ROSE:
  Inst: 1->treino, 2->treino, 3->treino, ... 10->treino
  Atualizacoes: 10 (5x mais!)
```

---

## 3. Por que ROSE Supera em Alguns Casos?

### Datasets com Drift (STAGGER, SEA)

```
Log observado:
  STAGGER_Abrupt_Recurring:
    GBML:  0.9564
    ROSE:  1.0000  <- Perfeito!

  SEA_Abrupt_Chain:
    GBML:  0.9651
    ROSE:  0.9913
```

**Explicacao:**
1. ROSE atualiza modelo apos CADA instancia
2. Quando drift ocorre, ROSE adapta em 1-10 instancias
3. GBML so adapta apos proximo chunk (1000+ instancias)
4. ROSE tambem usa ADWIN para detectar drift por classificador

### Por que ROSE Performa 1.0000?

Em datasets como STAGGER:
- Padroes simples (apenas 3 features categoricas)
- ROSE com ensemble de 10 HoeffdingTrees
- Cada arvore com subset de features
- ADWIN detecta drift e substitui arvores rapidamente
- Resultado: adaptacao quase perfeita

---

## 4. Por que ROSE Zera em Multiclasse?

### Datasets com G-mean = 0

```
LED (10 classes):         ROSE = 0.0000
WAVEFORM (3 classes):     ROSE = 0.0000
Shuttle (7 classes):      ROSE = 0.0000
CovType (7 classes):      ROSE = 0.0000
```

### Causa Raiz: Evaluator Errado

**WindowAUCImbalancedPerformanceEvaluator**:
- Calcula: G-mean = sqrt(Sensitivity * Specificity)
- Sensitivity = TP / (TP + FN) <- so para classe positiva
- Specificity = TN / (TN + FP) <- so para classe negativa
- **NAO FUNCIONA para multiclasse!**

**WindowAUCMultiClassImbalancedPerformanceEvaluator**:
- Calcula G-mean multiclasse corretamente
- Usa average ou macro-averaging
- **ESTE E O CORRETO para multiclasse!**

### Referencia do Repositorio ROSE

Do GitHub oficial:
```
Experimento 5 lista datasets multiclasse:
- covtype (7 classes)
- shuttle (7 classes)
- poker (10 classes)

Todos usam: WindowAUCMultiClassImbalancedPerformanceEvaluator
```

---

## 5. Como Tornar a Comparacao Justa?

### Opcao A: Ajustar ROSE para Chunk-Based (RECOMENDADO)

```python
# Processar ROSE chunk-by-chunk como GBML
Para cada chunk:
  1. Criar arquivo ARFF do chunk
  2. Executar ROSE com max_instances = chunk_size
  3. Extrair metricas finais do chunk
  4. Resetar ROSE (novo modelo) para proximo chunk
```

**Pros:** Comparacao direta e justa
**Cons:** ROSE perde vantagem de adaptacao continua

### Opcao B: Executar GBML Instance-by-Instance

```python
# Modificar GBML para avaliacao instance-by-instance
Para cada instancia:
  1. Predizer com modelo atual
  2. Atualizar metricas
  3. Acumular instancia no buffer
  Se buffer cheio (mini-batch):
    4. Re-treinar GA (muito caro!)
```

**Pros:** Comparacao mais justa
**Cons:** GA e MUITO caro para rodar por instancia

### Opcao C: Aceitar Diferenca e Documentar (PRAGMATICO)

```
Documentar no paper:
- ROSE: prequential instance-based (protocolo original)
- GBML: prequential chunk-based (protocolo original)
- Diferenca de granularidade favorece ROSE em adaptacao rapida
- Comparacao valida mas com ressalva metodologica
```

**Pros:** Mantem protocolos originais dos papers
**Cons:** Comparacao nao e 100% justa

---

## 6. Correcao Urgente: Evaluator Multiclasse

### Modificacao Necessaria no rose_wrapper.py

```python
def get_evaluator_for_dataset(n_classes: int) -> str:
    """Retorna o evaluator correto baseado no numero de classes."""
    if n_classes == 2:
        return "WindowAUCImbalancedPerformanceEvaluator"
    else:
        return "WindowAUCMultiClassImbalancedPerformanceEvaluator"
```

### Datasets que Precisam do Evaluator Multiclasse

| Dataset | Classes | Evaluator Atual | Evaluator Correto |
|---------|---------|-----------------|-------------------|
| LED_* | 10 | Binary (ERRADO) | MultiClass |
| WAVEFORM_* | 3 | Binary (ERRADO) | MultiClass |
| Shuttle | 7 | Binary (ERRADO) | MultiClass |
| CovType | 7 | Binary (ERRADO) | MultiClass |
| PokerHand | 10 | Binary (ERRADO) | MultiClass |

### Datasets Binarios (Evaluator Atual OK)

| Dataset | Classes |
|---------|---------|
| SEA_* | 2 |
| STAGGER_* | 2 |
| AGRAWAL_* | 2 |
| RBF_* | 2 |
| SINE_* | 2 |
| HYPERPLANE_* | 2 |
| RANDOMTREE_* | 2 |
| Electricity | 2 |
| IntelLabSensors | 2 |

---

## 7. Conclusoes

### 7.1 Por que ROSE performa muito bem em binarios com drift?

1. **Adaptacao instance-by-instance**: ROSE atualiza apos cada exemplo
2. **ADWIN drift detection**: Detecta drift por classificador
3. **Background learners**: Substitui classificadores degradados
4. **Self-adjusting bagging**: Ajusta pesos para classes minoritarias
5. **Per-class sliding windows**: Nao esquece classes raras

### 7.2 Por que ROSE zera em multiclasse?

**Evaluator errado!** `WindowAUCImbalancedPerformanceEvaluator` e para binario.
Deve usar `WindowAUCMultiClassImbalancedPerformanceEvaluator`.

### 7.3 A comparacao e justa?

**Parcialmente.** Ha diferenca de granularidade:
- ROSE: 1 atualizacao por instancia
- GBML: 1 atualizacao por chunk (1000+ instancias)

Isso da vantagem ao ROSE em cenarios de drift rapido.

---

## 8. Recomendacoes de Acao

### Imediato (Corrigir Experimento):

1. **Re-executar datasets multiclasse** com evaluator correto
2. **Adicionar deteccao automatica** de numero de classes no wrapper
3. **Documentar diferenca de protocolo** na metodologia do paper

### Medio Prazo (Validacao):

4. **Testar ROSE chunk-based** para comparacao mais justa
5. **Analisar curvas de aprendizado** para entender adaptacao

### No Paper:

6. **Reconhecer vantagem do ROSE** em adaptacao rapida
7. **Destacar vantagem do GBML** em interpretabilidade (regras explicaveis)
8. **Mencionar trade-off** velocidade vs explicabilidade

---

## 9. Referencias

- [ROSE GitHub](https://github.com/canoalberto/ROSE)
- Paper: Cano & Krawczyk (2022) - Machine Learning, Vol. 111(7), pp. 2561-2599
- [MOA Framework](https://moa.cms.waikato.ac.nz/)

---

**Criado por:** Claude Code
**Data:** 2025-11-26
**Status:** Investigacao concluida
