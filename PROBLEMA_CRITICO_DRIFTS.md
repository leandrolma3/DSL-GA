# PROBLEMA CRITICO - Drifts Fora do Range

**Data**: 2025-11-19
**Descoberta**: TODOS os 42 experimentos precisam de correcao

---

## SITUACAO

**Batch 1**: Corrigimos 12 experimentos no `config_batch_1.yaml` ✅
**Problema**: O `config.yaml` principal ainda tem os valores INCORRETOS
**Impacto**: Todos os outros 30 experimentos tambem precisam de correcao

---

## ANALISE COMPLETA

**Total experimentos**: 42
**Precisam correcao**: 42 (100%)
**Ja corretos**: 0

### Experimentos que PODEM ser corrigidos: 35

Estes podem ser ajustados para caber em 6 chunks.

### Experimentos IMPOSSIVEIS de corrigir: 7

Estes tem **muitos conceitos + transicoes graduais longas** que nao cabem em 6 chunks:

1. **AGRAWAL_Gradual_Chain**
   - 4 conceitos, width 2
   - Precisa: 4 + (3*2) = 10 chunks
   - **IMPOSSIVEL** em 6 chunks

2. **AGRAWAL_Gradual_Mild_to_Severe**
   - 3 conceitos, width 2
   - Precisa: 3 + (2*2) = 7 chunks
   - **IMPOSSIVEL** em 6 chunks

3. **AGRAWAL_Gradual_Recurring**
   - 3 conceitos, width 2
   - Precisa: 7 chunks
   - **IMPOSSIVEL**

4. **AGRAWAL_Gradual_Recurring_Noise**
   - 3 conceitos, width 2
   - Precisa: 7 chunks
   - **IMPOSSIVEL**

5. **AGRAWAL_Gradual_Blip**
   - 3 conceitos, width 1
   - Precisa: 3 + (2*1) = 5 chunks
   - Teoricamente possivel mas muito apertado

6. **RBF_Severe_Gradual_Recurrent**
   - 3 conceitos, width 2
   - Precisa: 7 chunks
   - **IMPOSSIVEL**

7. **STAGGER_Mixed_Recurring**
   - Mix de abrupt e gradual
   - Precisa verificar em detalhes

---

## SOLUCOES POSSIVEIS

### Opcao 1: REDUZIR WIDTH das transicoes graduais

Para experimentos impossiveis, **reduzir gradual_drift_width_chunks**:

**AGRAWAL_Gradual_Chain (4 conceitos):**
- Atual: width = 2
- **Solucao**: width = 1
- Chunks: 4 + (3*1) = 7 chunks → **AINDA IMPOSSIVEL**
- **Solucao REAL**: width = 0 (tornar abrupt) OU remover 1 conceito

**AGRAWAL_Gradual_Mild_to_Severe (3 conceitos):**
- Atual: width = 2
- **Solucao**: width = 1
- Chunks: 3 + (2*1) = 5 chunks + transicao = **POSSIVEL**

**AGRAWAL_Gradual_Recurring (3 conceitos):**
- Atual: width = 2
- **Solucao**: width = 1
- Chunks: 5 chunks → **POSSIVEL**

### Opcao 2: REMOVER conceitos

**AGRAWAL_Gradual_Chain:**
- Atual: 4 conceitos (f1, f3, f5, f7)
- **Solucao**: 3 conceitos (f1, f3, f5) com width 1
- Chunks: 3 + 2 = 5 chunks → **POSSIVEL**

### Opcao 3: ACEITAR que alguns experimentos NAO SERAO EXECUTADOS

Simplesmente **nao incluir** nos batches os experimentos impossiveis.

**Reducao**:
- De 42 experimentos para ~35-38 experimentos
- Ainda teremos boa cobertura

---

## RECOMENDACAO

**Abordagem Pragmatica**:

1. **Corrigir config.yaml principal** com valores que funcionam
2. **Reduzir width** dos gradual para 1 quando necessario
3. **Remover 1-2 conceitos** de chains muito longas
4. **Excluir** 2-3 experimentos mais problematicos se necessario

**Justificativa**:
- Precisamos de **resultados validos** mais do que cobertura completa
- Melhor ter 35 experimentos corretos que 42 incorretos
- Podemos argumentar no paper que limitamos a 6 chunks por design

---

## PROPOSTA DE CORRECOES

### Batch 2 (Gradual) - Correcoes Sugeridas:

#### PODEM ser corrigidos (9 experimentos):

1. **SEA_Gradual_Simple_Fast** (2 conceitos, width 1)
   - Atual: [5, 5]
   - **Corrigir**: [2, 3]
   - Drift em 2000-3000

2. **SEA_Gradual_Simple_Slow** (2 conceitos, width 2)
   - Atual: [5, 5]
   - **Corrigir**: [2, 2]
   - Drift em 2000-4000

3. **SEA_Gradual_Recurring** (3 conceitos, width 1)
   - Atual: [4, 5, 4]
   - **Corrigir**: [2, 1, 2]
   - Drifts em 2000-3000, 4000-5000

4. **STAGGER_Gradual_Chain** (3 conceitos, width 1)
   - Atual: [4, 4, 4]
   - **Corrigir**: [2, 1, 2]

5. **RBF_Gradual_Moderate** (2 conceitos, width 2)
   - Atual: [5, 5]
   - **Corrigir**: [2, 2]

6. **RBF_Gradual_Severe** (2 conceitos, width 2)
   - Atual: [5, 5]
   - **Corrigir**: [2, 2]

7. **HYPERPLANE_Gradual_Simple** (2 conceitos, width 3)
   - Atual: [6, 6]
   - **PROBLEMA**: width 3 precisa 2 + 3 = 5 chunks minimo
   - **Corrigir**: [1, 2] OU reduzir width para 2 e usar [2, 2]

8. **RANDOMTREE_Gradual_Simple** (2 conceitos, width 2)
   - Atual: [5, 5]
   - **Corrigir**: [2, 2]

9. **LED_Gradual_Simple** (2 conceitos, width 2)
   - Atual: [5, 5]
   - **Corrigir**: [2, 2]

#### NAO PODEM ser corrigidos (3 experimentos):

1. **AGRAWAL_Gradual_Chain** - EXCLUIR ou reduzir para 3 conceitos
2. **AGRAWAL_Gradual_Mild_to_Severe** - Reduzir width para 1
3. **AGRAWAL_Gradual_Blip** - Reduzir width para 0 ou ajustar conceitos

**BATCH 2 REVISADO**: 9 experimentos (ao inves de 12)

---

## BATCH 2 ALTERNATIVO (Viavel)

Se reduzirmos width e ajustarmos conceitos:

**9 Experimentos Graduais Viaveis**:
1. SEA_Gradual_Simple_Fast
2. SEA_Gradual_Simple_Slow
3. SEA_Gradual_Recurring
4. STAGGER_Gradual_Chain
5. RBF_Gradual_Moderate
6. RBF_Gradual_Severe
7. HYPERPLANE_Gradual_Simple (com width reduzido)
8. RANDOMTREE_Gradual_Simple
9. LED_Gradual_Simple

**Tempo Estimado**: ~10-12h (ainda viavel para Colab)

---

## PROXIMA ACAO RECOMENDADA

1. **Criar config_batch_2.yaml** com os 9 experimentos viaveis
2. **Aplicar correcoes** de duration_chunks e width
3. **Documentar exclusoes** dos 3 experimentos problematicos
4. **Validar** com script de verificacao
5. **Executar** Batch 2

**Status**: AGUARDANDO DECISAO DO USUARIO
- Optar por 9 experimentos viaveis? OU
- Tentar ajustar os 3 problematicos com reducao de width/conceitos?
