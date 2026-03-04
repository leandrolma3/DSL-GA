# Análise: ERulesD2S com Performance Muito Baixa

**Data**: 2025-11-19
**Problema**: ERulesD2S apresenta test_gmean = 4.54% (muito abaixo dos outros modelos)

---

## SINTOMAS

### Performance Observada (test_gmean médio):

| Modelo | Test G-mean | Performance |
|--------|-------------|-------------|
| ACDWM | 80.42% | EXCELENTE |
| GBML | 78.72% | EXCELENTE |
| ARF | 74.62% | BOM |
| SRP | 74.06% | BOM |
| HAT | 62.52% | RAZOÁVEL |
| **ERulesD2S** | **4.54%** | **CRÍTICO** |

**Resultado**: ERulesD2S está essencialmente falhando (4.5% é próximo de aleatório para classificação binária que seria 50%, ou seja, está PIOR que aleatório!)

---

## POSSÍVEIS CAUSAS

### 1. Parâmetros Muito Baixos

Na CÉLULA 7:
```python
'--erulesd2s-pop', '25',   # População = 25 indivíduos
'--erulesd2s-gen', '50',   # Gerações = 50
```

**Problema**: ERulesD2S pode precisar de MUITO mais gerações e população para convergir, especialmente em datasets complexos.

**Comparação**:
- GBML usa: population=120, generations=200 (4x mais!)
- ERulesD2S usa: population=25, generations=50

**Solução**: Aumentar parâmetros:
```python
'--erulesd2s-pop', '100',   # 4x mais
'--erulesd2s-gen', '200',   # 4x mais
```

**Impacto**: Tempo de execução 4-8x maior (~10-15 min por dataset)

---

### 2. Timeout Silencioso

ERulesD2S pode estar atingindo timeout e retornando modelo não convergido.

**Verificar**: Logs da execução dos 7 novos datasets

**Solução**: Aumentar timeout na CÉLULA 7:
```python
result = subprocess.run(
    cmd,
    timeout=7200  # 2 horas (ao invés de 1 hora)
)
```

---

### 3. Problema na Conversão de Formato

ERulesD2S pode estar reportando métricas em formato diferente que não está sendo convertido corretamente.

**Verificar**: Executar `diagnostico_erulesd2s.py` para inspecionar arquivos CSV originais

**Investigar**:
- Valores originais nos CSVs
- Colunas corretas
- Conversão de chunks (1-6 → 0-4)

---

### 4. Falha em Alguns Datasets

Alguns datasets podem ter características que ERulesD2S não consegue lidar.

**Verificar**: Performance por dataset (usar `diagnostico_erulesd2s.py`)

---

## AÇÕES RECOMENDADAS

### AÇÃO 1: Diagnóstico Rápido (5 min)

Execute no Colab/local:
```python
!python diagnostico_erulesd2s.py
```

Isso mostrará:
- Quais datasets têm resultados ERulesD2S
- Valores de G-mean por dataset
- Se há zeros ou valores muito baixos
- Primeiras linhas dos CSVs

### AÇÃO 2: Verificar Logs da CÉLULA 7

Revisar logs da execução para ver se há:
- Timeouts
- Erros do Java
- Warnings do ERulesD2S
- Mensagens de convergência

### AÇÃO 3: Re-executar com Parâmetros Maiores (OPCIONAL)

Se diagnóstico indicar problema de convergência:

1. Atualizar CÉLULA 7:
```python
'--erulesd2s-pop', '100',   # Era 25
'--erulesd2s-gen', '200',   # Era 50
```

2. Re-executar apenas datasets problemáticos

**Tempo**: ~2-3 horas (12 datasets × 10-15 min cada)

### AÇÃO 4: Aceitar e Documentar (se necessário)

Se ERulesD2S continuar com problemas após ajustes:

1. Documentar no paper que ERulesD2S não convergiu adequadamente
2. Excluir ERulesD2S das comparações estatísticas
3. Focar em baselines mais robustos (ACDWM, ARF, SRP, HAT)

**Justificativa válida**:
- Parâmetros padrão do ERulesD2S podem não ser adequados para concept drift
- Outros baselines são mais estabelecidos e robustos

---

## IMPACTO NA ANÁLISE ATUAL

### Testes Estatísticos

ERulesD2S provavelmente NÃO foi incluído nos testes porque:
```python
models_to_compare = ['GBML', 'ACDWM', 'ARF', 'SRP', 'HAT']
# ERulesD2S ausente (provavelmente por escolha ou por dados insuficientes)
```

**Resultado**: A análise estatística atual (GBML vs baselines) está CORRETA e não é afetada pelo problema do ERulesD2S.

### Comparação Geral

Na tabela de comparação, ERulesD2S aparece mas com performance muito baixa:
```
HYPERPLANE_Abrupt_Simple      0.7398  0.7455     0.0000  0.7133  0.8160  0.7071
```

Note: `ERulesD2S = 0.0000` em HYPERPLANE (falha completa!)

---

## CONCLUSÃO

**Status Atual**:
- GBML: OK
- ACDWM, ARF, SRP, HAT: OK
- **ERulesD2S: PROBLEMÁTICO**

**Recomendações**:

1. **CURTO PRAZO**: Execute `diagnostico_erulesd2s.py` para entender o problema
2. **MÉDIO PRAZO**: Se tiver tempo, re-execute com parâmetros maiores
3. **LONGO PRAZO**: Se persistir, exclua ERulesD2S e documente no paper

**Para o paper**:
- Análise estatística atual é válida (usa apenas modelos robustos)
- ERulesD2S pode ser mencionado como "não convergiu com parâmetros padrão"
- Foco em GBML vs ACDWM, ARF, SRP, HAT (4 baselines sólidos)

---

**Status**: PROBLEMA IDENTIFICADO - AGUARDANDO DIAGNÓSTICO
