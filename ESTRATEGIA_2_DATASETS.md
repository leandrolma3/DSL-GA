# ESTRATEGIA: 2 DATASETS POR CONFIG

## CALCULO DE VIABILIDADE

### Com Colab Pro
- **Limite de tempo**: 24 horas continuas
- **Tempo por dataset**: 11 horas
- **2 datasets**: 22 horas
- **Margem de seguranca**: 2 horas

**CONCLUSAO**: VIAVEL!

---

## DISTRIBUICAO BATCH 1 (TESTE)

### Execucao 1 (22h estimadas)
**Config**: config_batch_1_exec_1.yaml
**Datasets**:
1. SEA_Abrupt_Simple
2. AGRAWAL_Abrupt_Simple_Severe

### Execucao 2 (22h estimadas)
**Config**: config_batch_1_exec_2.yaml
**Datasets**:
1. RBF_Abrupt_Severe
2. HYPERPLANE_Abrupt_Simple

### Execucao 3 (11h estimadas)
**Config**: config_batch_1_exec_3.yaml
**Datasets**:
1. STAGGER_Abrupt_Chain

**TOTAL BATCH 1**: 3 execucoes, 55 horas distribuidas

---

## TESTE INICIAL (RECOMENDADO)

### Fase de Teste: Executar apenas Execucao 1

**Objetivo**: Validar que 2 datasets cabem em 24h

**Datasets de teste**:
1. SEA_Abrupt_Simple (tipicamente mais rapido)
2. AGRAWAL_Abrupt_Simple_Severe (drift severo)

**Resultado esperado**: 18-22 horas

**Validacoes**:
- [ ] Tempo total < 24h
- [ ] Ambos datasets executados com sucesso
- [ ] Resultados salvos corretamente
- [ ] Qualidade mantida (G-mean similar ao teste anterior)

**Se o teste passar**: Escalar para as 3 execucoes completas do Batch 1

---

## ESCALAMENTO PARA MULTIPLAS CONTAS

### Estrategia com 3 Contas

**Conta 1**: Execucao 1 (2 datasets)
**Conta 2**: Execucao 2 (2 datasets)
**Conta 3**: Execucao 3 (1 dataset)

**Resultado**: Batch 1 completo em 22 horas!

### Estrategia para 12 Batches

**Total de execucoes**: 12 batches × 3 execucoes = 36 execucoes

**Com 3 contas em paralelo**:
- 36 execucoes / 3 contas = 12 rodadas
- 12 rodadas × 22h = 264 horas
- **Total**: 11 dias (executando 24/7)

**Com 6 contas em paralelo**:
- 36 execucoes / 6 contas = 6 rodadas
- 6 rodadas × 22h = 132 horas
- **Total**: 5.5 dias (executando 24/7)

---

## CRONOGRAMA PROPOSTO

### Semana 1: Teste e Validacao
- **Dia 1**: Executar Batch 1 - Execucao 1 (teste)
- **Dia 2**: Validar resultados, ajustar se necessario
- **Dia 3**: Executar Batch 1 - Execucao 2
- **Dia 4**: Executar Batch 1 - Execucao 3
- **Dia 5**: Consolidar Batch 1, validar estrutura

### Semana 2-3: Escalar para Todos os Batches
- Executar em paralelo com multiplas contas
- Monitorar e validar resultados
- Backup incremental

### Semana 4: Consolidacao e Fase 2
- Consolidar todos os 57 datasets
- Iniciar Fase 2 (River models)

---

## ESTRUTURA DE RESULTADOS

```
experiments_6chunks_phase1_gbml/
  batch_1/
    SEA_Abrupt_Simple/
      run_1/
        chunk_data/
        plots/
        ...
    AGRAWAL_Abrupt_Simple_Severe/
      run_1/
        ...
    RBF_Abrupt_Severe/
    HYPERPLANE_Abrupt_Simple/
    STAGGER_Abrupt_Chain/
  batch_2/
    ...
  ...
  batch_12/
```

---

## SCRIPTS NECESSARIOS

### 1. Gerar Configs (3 por batch)
- Script: `generate_2dataset_configs.py`
- Output: 36 configs (12 batches × 3 execucoes)

### 2. Script de Execucao Colab
- Monta Drive
- Copia config
- Executa main.py
- Valida resultados
- Envia notificacao

### 3. Script de Monitoramento
- Verifica progresso
- Estima tempo restante
- Detecta erros

---

## VANTAGENS DESTA ESTRATEGIA

1. **Viavel no Colab Pro**: 22h < 24h
2. **Paralelizavel**: Multiplas contas
3. **Recuperavel**: Se uma execucao falhar, perde apenas 2 datasets
4. **Flexivel**: Pode rodar 1, 2 ou 3 execucoes por vez
5. **Qualidade**: Mantem todos os parametros otimizados

---

## PROXIMOS PASSOS

1. Criar 3 configs para Batch 1
2. Testar Execucao 1 (2 datasets)
3. Validar tempo e qualidade
4. Se OK, gerar configs para todos os 12 batches
5. Escalar com multiplas contas

---

**Data**: 2025-11-17
**Status**: PRONTO PARA GERAR CONFIGS DE TESTE
