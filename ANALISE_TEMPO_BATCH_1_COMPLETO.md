# Analise de Tempo - Batch 1 Completo (12 Bases)

**Data da Execucao**: 2025-11-18 a 2025-11-19
**Log Analisado**: `batch_1_full.log`
**Config Utilizado**: `config_test_drift_recovery.yaml`
**Resultados em**: `experiments_6chunks_phase2_gbml/batch_1`

---

## TEMPO TOTAL DE EXECUCAO

| Metrica | Valor |
|---------|-------|
| **Inicio** | 2025-11-18 19:03:56 |
| **Fim** | 2025-11-19 08:43:59 |
| **Duracao Total** | **13 horas, 40 minutos, 3 segundos** |
| **Total em segundos** | **49,203 segundos** |
| **Total em minutos** | **820 minutos** |

---

## TEMPO POR DATASET (12 BASES)

| # | Dataset | Tempo (s) | Tempo (min) | Tempo (h:mm) | % do Total |
|---|---------|-----------|-------------|--------------|------------|
| 1 | SEA_Abrupt_Simple | 1,998.21 | 33.3 | 0:33 | 4.1% |
| 2 | SEA_Abrupt_Chain | 2,218.52 | 37.0 | 0:37 | 4.5% |
| 3 | SEA_Abrupt_Recurring | 1,952.27 | 32.5 | 0:33 | 4.0% |
| 4 | AGRAWAL_Abrupt_Simple_Mild | 3,563.87 | 59.4 | 0:59 | 7.2% |
| 5 | AGRAWAL_Abrupt_Simple_Severe | 3,055.45 | 50.9 | 0:51 | 6.2% |
| 6 | AGRAWAL_Abrupt_Chain_Long | 3,022.54 | 50.4 | 0:50 | 6.1% |
| 7 | RBF_Abrupt_Severe | 8,088.23 | 134.8 | 2:15 | 16.4% |
| 8 | RBF_Abrupt_Blip | 6,359.48 | 106.0 | 1:46 | 12.9% |
| 9 | STAGGER_Abrupt_Chain | 1,505.98 | 25.1 | 0:25 | 3.1% |
| 10 | STAGGER_Abrupt_Recurring | 1,466.09 | 24.4 | 0:24 | 3.0% |
| 11 | HYPERPLANE_Abrupt_Simple | 7,983.89 | 133.1 | 2:13 | 16.2% |
| 12 | RANDOMTREE_Abrupt_Simple | 7,988.74 | 133.1 | 2:13 | 16.2% |
| **TOTAL** | **12 bases** | **49,203** | **820** | **13:40** | **100%** |

---

## ANALISE POR CATEGORIA

### Por Velocidade de Execucao

**RAPIDAS (< 30 min)**
- STAGGER_Abrupt_Recurring: 24.4 min
- STAGGER_Abrupt_Chain: 25.1 min
- SEA_Abrupt_Recurring: 32.5 min
- SEA_Abrupt_Simple: 33.3 min

**MEDIAS (30-60 min)**
- SEA_Abrupt_Chain: 37.0 min
- AGRAWAL_Abrupt_Chain_Long: 50.4 min
- AGRAWAL_Abrupt_Simple_Severe: 50.9 min
- AGRAWAL_Abrupt_Simple_Mild: 59.4 min

**LENTAS (> 60 min)**
- RBF_Abrupt_Blip: 106.0 min (1h46)
- HYPERPLANE_Abrupt_Simple: 133.1 min (2h13)
- RANDOMTREE_Abrupt_Simple: 133.1 min (2h13)
- RBF_Abrupt_Severe: 134.8 min (2h15)

### Por Tipo de Dataset

| Dataset Base | Numero de Bases | Tempo Total (min) | Tempo Medio (min) |
|--------------|-----------------|-------------------|-------------------|
| **SEA** | 3 | 102.8 | 34.3 |
| **AGRAWAL** | 3 | 160.7 | 53.6 |
| **RBF** | 2 | 240.8 | 120.4 |
| **STAGGER** | 2 | 49.5 | 24.8 |
| **HYPERPLANE** | 1 | 133.1 | 133.1 |
| **RANDOMTREE** | 1 | 133.1 | 133.1 |

### Por Padrao de Drift

| Padrao | Numero de Bases | Tempo Total (min) | Tempo Medio (min) |
|--------|-----------------|-------------------|-------------------|
| **Simple** (1 drift em 3000) | 5 | 383.8 | 76.8 |
| **Chain** (2+ drifts sequenciais) | 3 | 112.5 | 37.5 |
| **Recurring** (conceito retorna) | 2 | 56.9 | 28.5 |
| **Blip** (conceito temporario) | 1 | 106.0 | 106.0 |
| **Chain Long** (4 conceitos) | 1 | 50.4 | 50.4 |

---

## COMPARACAO COM ESTIMATIVAS ANTERIORES

### Batch 1 Original (5 bases) - Execucoes Anteriores

| Dataset | Tempo Anterior (s) | Tempo Atual (s) | Diferenca |
|---------|-------------------|-----------------|-----------|
| SEA_Abrupt_Simple | 1,582 | 1,998 | +26% |
| AGRAWAL_Abrupt_Simple_Severe | 4,471 | 3,055 | -32% |
| RBF_Abrupt_Severe | 8,116 | 8,088 | -0.3% |
| STAGGER_Abrupt_Chain | 1,690 | 1,506 | -11% |
| HYPERPLANE_Abrupt_Simple | 10,941 | 7,984 | -27% |
| **TOTAL (5 bases)** | **26,800 (7.4h)** | **22,631 (6.3h)** | **-16%** |

**Observacao**: Os tempos atuais sao menores porque as correcoes de `duration_chunks`
resultaram em experimentos mais curtos (6 chunks ao inves de 8-10 chunks efetivos).

---

## PROJECOES PARA BATCHES FUTUROS

### Baseado nos Tempos Medios Observados

**Tempo medio por base**: 820 min / 12 bases = **68.3 min/base**

### Batch 2: Gradual Drifts (12 bases estimadas)

| Categoria | Estimativa | Justificativa |
|-----------|------------|---------------|
| Bases rapidas (SEA, STAGGER) | 5 x 30 min = 150 min | Similar ao Batch 1 |
| Bases medias (AGRAWAL) | 3 x 55 min = 165 min | Similar ao Batch 1 |
| Bases lentas (RBF, HYPERPLANE) | 4 x 120 min = 480 min | Similar ao Batch 1 |
| **TOTAL ESTIMADO** | **~13-14 horas** | Dentro do limite Colab |

### Batch 3: Noise & Mixed (10 bases estimadas)

| Categoria | Estimativa |
|-----------|------------|
| Bases rapidas | 4 x 30 min = 120 min |
| Bases medias | 4 x 55 min = 220 min |
| Bases lentas | 2 x 120 min = 240 min |
| **TOTAL ESTIMADO** | **~10-11 horas** |

### Batch 4: Complementary (11 bases estimadas)

| Categoria | Estimativa |
|-----------|------------|
| Bases rapidas | 4 x 30 min = 120 min |
| Bases medias | 4 x 55 min = 220 min |
| Bases lentas | 3 x 120 min = 360 min |
| **TOTAL ESTIMADO** | **~12-13 horas** |

---

## RECOMENDACOES PARA PROXIMAS EXECUCOES

### 1. Uso Otimo do Google Colab

**Colab Free (limite ~12h)**
- Executar batches com 10-12 bases
- Priorizar bases rapidas e medias
- Deixar bases lentas para Colab Pro se disponivel

**Colab Pro (limite ~24h)**
- Executar batches maiores (15-20 bases)
- Agrupar bases lentas em uma unica execucao

### 2. Estrategia de Agrupamento

**OPCAO A: Maximizar bases por execucao**
- Grupo 1: 6 bases rapidas + 3 bases medias (~6h)
- Grupo 2: 3 bases medias + 2 bases lentas (~6h)
- **Resultado**: 14 bases em ~12h (2 execucoes Colab Free)

**OPCAO B: Balancear tempo**
- Grupo 1: Mix de rapidas/medias para ~6-7h
- Grupo 2: Mix de rapidas/medias para ~6-7h
- **Resultado**: 12 bases em ~13h (mais estavel)

### 3. Ordem de Prioridade para Proximos Batches

**Prioridade ALTA (bases fundamentais)**
1. Batch 2 - Gradual Drifts (12 bases) - **13-14h**
   - Complementa Batch 1 com drifts graduais
   - Essencial para comparacao abrupt vs gradual

**Prioridade MEDIA (bases complementares)**
2. Batch 3 - Noise & Mixed (10 bases) - **10-11h**
   - Adiciona complexidade com ruido
   - Testa robustez dos modelos

**Prioridade BAIXA (bases extras)**
3. Batch 4 - Complementary (11 bases) - **12-13h**
   - Casos especiais e variantes
   - Completa cobertura experimental

---

## METRICAS DE EFICIENCIA

### Aproveitamento do Tempo Colab

| Metrica | Valor |
|---------|-------|
| Tempo utilizado | 13h 40min |
| Limite Colab Free | ~12h |
| **Excesso** | **+1h 40min** |
| Taxa de aproveitamento | 114% do limite |

**Conclusao**: O Batch 1 com 12 bases EXCEDE ligeiramente o limite do Colab Free.

### Sugestao de Ajuste

**Para Colab Free (12h limite):**
- Remover as 2 bases mais lentas (RANDOMTREE, HYPERPLANE)
- Total: 10 bases em ~11h
- Executar as 2 removidas em outra sessao (4.5h)

**OU usar Colab Pro:**
- Executar todas as 12 bases confortavelmente
- Tempo total: 13h40 bem dentro do limite de 24h

---

## CUSTO-BENEFICIO POR DATASET

### Melhor Custo-Beneficio (rapidas, alta informacao)

1. **STAGGER** (ambas): 25 min cada
   - Drift detection facil
   - Bom para validacao

2. **SEA** (todas): 33-37 min cada
   - Dataset classico, bem documentado
   - Referencia importante

### Pior Custo-Beneficio (lentas)

1. **RANDOMTREE**: 133 min (2h13)
   - Tempo similar a RBF e HYPERPLANE
   - Menos usado na literatura

2. **RBF_Abrupt_Severe**: 135 min (2h15)
   - Mais lento do batch
   - Mas importante para severidade alta

---

## TEMPO DE POS-PROCESSAMENTO (ESTIMATIVA)

### Scripts de Analise

| Script | Tempo por Base | Tempo Total (12 bases) |
|--------|----------------|------------------------|
| analyze_concept_difference.py | ~30s | ~6 min (1x) |
| generate_plots.py | ~1-2 min | ~15-20 min |
| rule_diff_analyzer.py | ~2-3 min | ~25-30 min |
| **TOTAL** | ~4-5 min | **~45-60 min** |

**Conclusao**: Pos-processamento adiciona ~1h ao tempo total.

---

## TIMELINE COMPLETA - BATCH 1

```
19:03:56 - Inicio Batch 1
19:37:14 - SEA_Abrupt_Simple concluido (33 min)
20:14:13 - SEA_Abrupt_Chain concluido (37 min)
20:46:45 - SEA_Abrupt_Recurring concluido (33 min)
21:46:09 - AGRAWAL_Abrupt_Simple_Mild concluido (59 min)
22:37:04 - AGRAWAL_Abrupt_Simple_Severe concluido (51 min)
23:27:27 - AGRAWAL_Abrupt_Chain_Long concluido (50 min)
01:42:15 - RBF_Abrupt_Severe concluido (135 min) [+DIA]
03:28:14 - RBF_Abrupt_Blip concluido (106 min)
03:53:20 - STAGGER_Abrupt_Chain concluido (25 min)
04:17:46 - STAGGER_Abrupt_Recurring concluido (24 min)
06:30:50 - HYPERPLANE_Abrupt_Simple concluido (133 min)
08:43:59 - RANDOMTREE_Abrupt_Simple concluido (133 min)
08:43:59 - Fim Batch 1 (13h40min total)
```

---

## CONCLUSOES E PROXIMOS PASSOS

### Sucesso do Batch 1

- [OK] 12 bases executadas com sucesso
- [OK] Drifts corrigidos para 0-5000 instancias
- [OK] Tempo total: 13h40 (proximo do estimado)
- [OK] Todos os resultados salvos corretamente

### Para Replicar em Outros Batches

1. **Usar mesma estrategia de correcoes**
   - Ajustar duration_chunks para 6 chunks
   - Validar posicoes de drift antes de executar

2. **Otimizar agrupamento**
   - Bases rapidas: 4-5 por grupo
   - Bases medias: 3-4 por grupo
   - Bases lentas: 2-3 por grupo

3. **Monitorar tempo**
   - Checkpoint a cada base concluida
   - Ajustar se aproximando do limite

4. **Pos-processamento**
   - Executar em celula separada
   - ~1h adicional apos experimento

### Viabilidade de Executar Todos os Batches

| Batch | Bases | Tempo Estimado | Sessoes Colab Free | Sessoes Colab Pro |
|-------|-------|----------------|---------------------|-------------------|
| Batch 1 (Abrupt) | 12 | 13h40 | 2 sessoes | 1 sessao |
| Batch 2 (Gradual) | 12 | 13-14h | 2 sessoes | 1 sessao |
| Batch 3 (Noise) | 10 | 10-11h | 1 sessao | 1 sessao |
| Batch 4 (Comp) | 11 | 12-13h | 2 sessoes | 1 sessao |
| **TOTAL** | **45** | **~50h** | **7 sessoes** | **4 sessoes** |

**Cronograma Sugerido (Colab Free):**
- Semana 1: Batch 1 completo (2 sessoes)
- Semana 2: Batch 2 completo (2 sessoes)
- Semana 3: Batch 3 completo (1 sessao)
- Semana 4: Batch 4 completo (2 sessoes)

**Cronograma Sugerido (Colab Pro):**
- Dia 1: Batch 1 (13h40)
- Dia 2: Batch 2 (13-14h)
- Dia 3: Batch 3 (10-11h)
- Dia 4: Batch 4 (12-13h)
- **Total: 4 dias completos**

---

**Status Final**: BATCH 1 CONCLUIDO COM SUCESSO - PRONTO PARA POS-PROCESSAMENTO
**Proximo Passo**: Executar `post_process_batch_1.py` para gerar plots e analises
