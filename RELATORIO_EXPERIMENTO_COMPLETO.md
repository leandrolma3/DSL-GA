# Relatorio Final - Experimento Completo

**Data**: 2025-11-11
**Experimento**: Comparacao GBML vs ACDWM vs River (HAT, ARF, SRP)
**Datasets**: 3 (RBF_Abrupt_Severe, RBF_Abrupt_Moderate, RBF_Gradual_Moderate)
**Chunks**: 3 por dataset (6000 instancias cada)
**Metodologia**: Train-then-test (Chunk i treina, Chunk i+1 testa)

---

## Resumo Executivo

### Ranking Geral (G-mean media - 8 avaliacoes)

| Posicao | Modelo | G-mean Media | Diferenca vs Melhor |
|---------|--------|--------------|---------------------|
| 1       | ACDWM  | 0.7746       | -                   |
| 2       | ARF    | 0.7730       | -0.0016 (-0.2%)     |
| 3       | SRP    | 0.7715       | -0.0031 (-0.4%)     |
| 4       | HAT    | 0.7130       | -0.0616 (-8.0%)     |
| 5       | GBML   | 0.7072       | -0.0675 (-8.7%)     |

### Principais Resultados

1. **ACDWM obteve o melhor desempenho geral**, com G-mean medio de 0.7746
2. **ARF ficou muito proximo** (diferenca de apenas 0.16%)
3. **SRP tambem competitivo** (diferenca de 0.4%)
4. **GBML ficou em ultimo lugar**, 8.7% abaixo do melhor modelo
5. **Todos os modelos enfrentam dificuldade em concept drift abrupto** (chunk 1->2)

---

## Analise por Dataset

### 1. RBF_Abrupt_Moderate

**Caracteristicas**: Drift abrupto com severidade moderada

#### Tabela de Resultados (G-mean por Avaliacao)

| Avaliacao    | GBML   | ACDWM  | HAT    | ARF    | SRP    |
|--------------|--------|--------|--------|--------|--------|
| Chunk 0 -> 1 | 0.8782 | 0.9139 | 0.8616 | 0.9244 | 0.9268 |
| Chunk 1 -> 2 | 0.4695 | 0.5040 | 0.4198 | 0.4768 | 0.4271 |
| Chunk 2 -> 3 | 0.8477 | 0.9185 | 0.7966 | 0.8973 | 0.9079 |
| **Media**    | 0.7318 | 0.7788 | 0.6927 | 0.7662 | 0.7539 |

#### Ranking

1. **ACDWM**: 0.7788
2. **ARF**: 0.7662 (-1.6%)
3. **SRP**: 0.7539 (-3.2%)
4. **GBML**: 0.7318 (-6.0%)
5. **HAT**: 0.6927 (-11.1%)

#### Observacoes

- **ACDWM venceu neste dataset**
- **SRP obteve melhor resultado na primeira avaliacao** (0.9268)
- **Todos os modelos colapsaram no Chunk 1->2** (drift abrupto)
  - Melhor: ACDWM (0.5040)
  - Pior: HAT (0.4198)
- **ACDWM se recuperou melhor** no Chunk 2->3 (0.9185)

---

### 2. RBF_Abrupt_Severe

**Caracteristicas**: Drift abrupto com severidade alta

#### Tabela de Resultados (G-mean por Avaliacao)

| Avaliacao    | GBML   | ACDWM  | HAT    | ARF    | SRP    |
|--------------|--------|--------|--------|--------|--------|
| Chunk 0 -> 1 | 0.8799 | 0.9138 | 0.8600 | 0.9166 | 0.9152 |
| Chunk 1 -> 2 | 0.4289 | 0.4481 | 0.5809 | 0.4925 | 0.5028 |
| Chunk 2 -> 3 | 0.6998 | 0.9039 | 0.7806 | 0.8871 | 0.8952 |
| **Media**    | 0.6695 | 0.7553 | 0.7405 | 0.7654 | 0.7711 |

#### Ranking

1. **SRP**: 0.7711
2. **ARF**: 0.7654 (-0.7%)
3. **ACDWM**: 0.7553 (-2.0%)
4. **HAT**: 0.7405 (-4.0%)
5. **GBML**: 0.6695 (-13.2%)

#### Observacoes

- **SRP venceu neste dataset** (mais severo)
- **ARF tambem performou muito bem**
- **ACDWM foi o unico que superou 0.90 no Chunk 2->3** (0.9039)
- **HAT foi o melhor no Chunk 1->2** (0.5809) - mais resistente ao drift severo
- **GBML teve o pior desempenho geral** (-13.2% vs melhor)

---

### 3. RBF_Gradual_Moderate

**Caracteristicas**: Drift gradual com severidade moderada

#### Tabela de Resultados (G-mean por Avaliacao)

| Avaliacao    | GBML   | ACDWM  | HAT    | ARF    | SRP    |
|--------------|--------|--------|--------|--------|--------|
| Chunk 0 -> 1 | 0.6820 | 0.7113 | 0.6541 | 0.6850 | 0.7026 |
| Chunk 1 -> 2 | 0.7714 | 0.8836 | 0.7503 | 0.9043 | 0.8945 |
| **Media**    | 0.7267 | 0.7974 | 0.7022 | 0.7947 | 0.7985 |

#### Ranking

1. **SRP**: 0.7985
2. **ACDWM**: 0.7974 (-0.1%)
3. **ARF**: 0.7947 (-0.5%)
4. **GBML**: 0.7267 (-9.0%)
5. **HAT**: 0.7022 (-12.1%)

#### Observacoes

- **SRP e ACDWM praticamente empataram** (diferenca de 0.1%)
- **ARF obteve melhor resultado no Chunk 1->2** (0.9043)
- **Drift gradual e mais facil para todos os modelos**
  - Nenhum modelo teve colapso severo
  - Todos melhoraram do Chunk 0->1 para 1->2
- **GBML performou relativamente melhor** neste dataset (-9% vs -13.2% no Severe)

---

## Analise Comparativa

### Desempenho por Tipo de Drift

| Modelo | Abrupt Severe | Abrupt Moderate | Gradual Moderate | Media |
|--------|---------------|-----------------|------------------|-------|
| ACDWM  | 0.7553 (3o)   | 0.7788 (1o)     | 0.7974 (2o)      | 0.7746 |
| ARF    | 0.7654 (2o)   | 0.7662 (2o)     | 0.7947 (3o)      | 0.7730 |
| SRP    | 0.7711 (1o)   | 0.7539 (3o)     | 0.7985 (1o)      | 0.7715 |
| HAT    | 0.7405 (4o)   | 0.6927 (5o)     | 0.7022 (5o)      | 0.7130 |
| GBML   | 0.6695 (5o)   | 0.7318 (4o)     | 0.7267 (4o)      | 0.7072 |

### Estabilidade (Desvio Padrao)

| Modelo | Desvio Padrao | Interpretacao |
|--------|---------------|---------------|
| HAT    | 0.1430        | Mais estavel  |
| GBML   | 0.1649        | Estavel       |
| ARF    | 0.1815        | Moderado      |
| ACDWM  | 0.1844        | Moderado      |
| SRP    | 0.1902        | Mais variavel |

**Observacao**: HAT e o mais estavel, mas tem performance inferior. SRP e o mais variavel, mas tem performance superior.

### Melhor e Pior Resultado Individual

| Modelo | Melhor G-mean | Dataset/Chunk      | Pior G-mean | Dataset/Chunk      |
|--------|---------------|--------------------|-------------|--------------------|
| ACDWM  | 0.9185        | Abrupt_Mod (2->3)  | 0.4481      | Abrupt_Sev (1->2)  |
| ARF    | 0.9244        | Abrupt_Mod (0->1)  | 0.4768      | Abrupt_Mod (1->2)  |
| SRP    | 0.9268        | Abrupt_Mod (0->1)  | 0.4271      | Abrupt_Mod (1->2)  |
| HAT    | 0.8616        | Abrupt_Mod (0->1)  | 0.4198      | Abrupt_Mod (1->2)  |
| GBML   | 0.8799        | Abrupt_Sev (0->1)  | 0.4289      | Abrupt_Sev (1->2)  |

---

## Padroes Identificados

### 1. Colapso no Drift Abrupto (Chunk 1->2)

**Todos os modelos sofreram queda drastica** nos datasets com drift abrupto:

- **RBF_Abrupt_Moderate (Chunk 1->2)**:
  - SRP: 0.9268 -> 0.4271 (queda de 54%)
  - ARF: 0.9244 -> 0.4768 (queda de 48%)
  - GBML: 0.8782 -> 0.4695 (queda de 47%)
  - ACDWM: 0.9139 -> 0.5040 (queda de 45%)
  - HAT: 0.8616 -> 0.4198 (queda de 51%)

- **RBF_Abrupt_Severe (Chunk 1->2)**:
  - SRP: 0.9152 -> 0.5028 (queda de 45%)
  - ARF: 0.9166 -> 0.4925 (queda de 46%)
  - ACDWM: 0.9138 -> 0.4481 (queda de 51%)
  - GBML: 0.8799 -> 0.4289 (queda de 51%)
  - HAT: 0.8600 -> 0.5809 (queda de 32%)

**HAT foi o mais resistente** ao drift abrupto severo (32% de queda vs 45-51% dos outros).

### 2. Capacidade de Recuperacao (Chunk 2->3)

Apos o drift abrupto, **ACDWM demonstrou melhor recuperacao**:

- **RBF_Abrupt_Moderate (Chunk 2->3)**:
  - ACDWM: 0.9185 (recuperacao de 82% desde 0.5040)
  - SRP: 0.9079 (recuperacao de 81% desde 0.4271)
  - ARF: 0.8973 (recuperacao de 88% desde 0.4768)

- **RBF_Abrupt_Severe (Chunk 2->3)**:
  - ACDWM: 0.9039 (recuperacao de 102% desde 0.4481)
  - SRP: 0.8952 (recuperacao de 78% desde 0.5028)
  - ARF: 0.8871 (recuperacao de 80% desde 0.4925)

**ACDWM foi o unico modelo que superou o desempenho inicial** apos o drift.

### 3. Drift Gradual e Mais Facil

No dataset **RBF_Gradual_Moderate**, todos os modelos **melhoraram** do Chunk 0->1 para 1->2:

| Modelo | Chunk 0->1 | Chunk 1->2 | Melhoria |
|--------|------------|------------|----------|
| ARF    | 0.6850     | 0.9043     | +32%     |
| SRP    | 0.7026     | 0.8945     | +27%     |
| ACDWM  | 0.7113     | 0.8836     | +24%     |
| GBML   | 0.6820     | 0.7714     | +13%     |
| HAT    | 0.6541     | 0.7503     | +15%     |

**ARF teve a maior melhoria** com drift gradual.

---

## Conclusoes

### 1. ACDWM e o Melhor Modelo Geral

Com **G-mean medio de 0.7746**, ACDWM:
- Venceu em 1 dos 3 datasets (RBF_Abrupt_Moderate)
- Ficou em 2o lugar nos outros 2 datasets
- Demonstrou **melhor capacidade de recuperacao** apos drift abrupto
- Particularmente forte em cenarios com drift moderado

### 2. River Models (ARF, SRP) Sao Competitivos

- **ARF**: 2o lugar geral (0.7730), apenas 0.2% abaixo de ACDWM
- **SRP**: 3o lugar geral (0.7715), apenas 0.4% abaixo de ACDWM
- **Venceram em datasets com drift severo e gradual**
- **Mais rapidos que GBML** (nao necessitam evolucao GA)

### 3. HAT e Mais Resistente a Drift Abrupto

- **Menor queda de performance** no drift abrupto severo (32% vs 45-51%)
- Mas **performance geral inferior** (4o lugar)
- **Mais estavel** (menor desvio padrao)

### 4. GBML Precisa de Melhorias

- **Ultimo lugar geral** (0.7072), 8.7% abaixo do melhor
- **Maior queda** no drift abrupto
- **Menor recuperacao** apos drift
- Possibilidades de melhoria:
  - Ajustar hiperparametros GA
  - Melhorar deteccao de drift
  - Implementar estrategias de adaptacao mais rapidas

### 5. Todos os Modelos Sofrem com Drift Abrupto

- **Queda media de 45-50%** de performance
- Necessidade de estrategias de deteccao e adaptacao mais rapidas
- **ACDWM mostrou melhor recuperacao**, sugerindo que ensemble dinamico e efetivo

---

## Recomendacoes

### Para Uso Pratico

1. **Drift Abrupto Moderado**: Use **ACDWM**
2. **Drift Abrupto Severo**: Use **SRP** ou **ARF**
3. **Drift Gradual**: Use **SRP** ou **ACDWM** (praticamente empatados)
4. **Necessita Estabilidade**: Use **HAT** (menor variancia)
5. **Necessita Performance**: Evite **GBML** no estado atual

### Para Pesquisa Futura

1. **Melhorar GBML**:
   - Investigar por que esta performando pior
   - Otimizar hiperparametros GA
   - Implementar deteccao de drift mais rapida

2. **Analisar Recuperacao de ACDWM**:
   - Entender por que ACDWM se recupera melhor
   - Aplicar estrategia similar em outros modelos

3. **Combinar Modelos**:
   - Usar HAT para drift abrupto severo (mais resistente)
   - Usar ACDWM para recuperacao pos-drift
   - Ensemble: ACDWM + ARF + SRP

4. **Investigar Chunk 1->2**:
   - Por que todos os modelos colapsam?
   - Implementar deteccao precoce de drift
   - Testar estrategias de re-treinamento rapido

---

## Informacoes do Experimento

**Log analisado**: experiment_comparison_full2.log
**Linhas totais**: 110,214
**Tempo total**: ~7.3 horas
**Modelos testados**: 5 (GBML, ACDWM, HAT, ARF, SRP)
**Avaliacoes totais**: 8 por modelo (24 avaliacoes no total)
**Seed**: 42 (reproducivel)

**Scripts de analise**:
- analyze_experiment_log.py: Analise basica
- analyze_experiment_detailed.py: Analise detalhada
- experiment_detailed_analysis.txt: Relatorio tecnico

**Arquivos gerados**:
- experiment_analysis_report.txt
- experiment_detailed_analysis.txt
- RELATORIO_EXPERIMENTO_COMPLETO.md (este arquivo)
