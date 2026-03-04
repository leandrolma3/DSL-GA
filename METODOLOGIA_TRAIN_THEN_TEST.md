# METODOLOGIA TRAIN-THEN-TEST - CORRECAO IMPLEMENTADA

**Data**: 2025-11-18
**Ajuste**: Garantir comparacao justa entre GBML e modelos comparativos

---

## PROBLEMA IDENTIFICADO

### Versao Anterior (INCORRETA)

**River/ACDWM** salvavam APENAS metricas de TESTE:
- Chunk 0: treina → testa no chunk 1 → salva APENAS test_gmean
- Chunk 1: treina → testa no chunk 2 → salva APENAS test_gmean
- etc.

**GBML** salvava AMBAS metricas de treino E teste:
- Chunk 0: treina → avalia no chunk 0 (train_gmean) E no chunk 1 (test_gmean)
- Chunk 1: treina → avalia no chunk 1 (train_gmean) E no chunk 2 (test_gmean)
- etc.

**Consequencia**: Comparacao INJUSTA - faltavam metricas de treino nos modelos comparativos.

---

## CORRECAO IMPLEMENTADA

### Metodologia Train-Then-Test (CORRIGIDA)

Para TODOS os modelos (GBML, River, ACDWM):

```
Para cada chunk i (de 0 a 4):
    1. Treinar modelo no chunk i
    2. Avaliar no chunk i → train_gmean, train_accuracy, train_f1
    3. Avaliar no chunk i+1 → test_gmean, test_accuracy, test_f1
    4. Salvar AMBOS train e test metrics
```

### Exemplo Concreto (Dataset SEA)

**Chunk 0**:
- Treina com chunk_0_train.csv (1000 instancias)
- Avalia em chunk_0_train.csv → train_gmean = 0.9886
- Avalia em chunk_1_test.csv → test_gmean = 0.9529
- **Salva**: chunk=0, train_gmean=0.9886, test_gmean=0.9529

**Chunk 1**:
- Treina com chunk_1_test.csv (1000 instancias)
- Avalia em chunk_1_test.csv → train_gmean = 0.9850
- Avalia em chunk_2_test.csv → test_gmean = 0.9702
- **Salva**: chunk=1, train_gmean=0.9850, test_gmean=0.9702

**E assim por diante...**

---

## ESTRUTURA DE DADOS ATUALIZADA

### Arquivo CSV de Resultados (River/ACDWM)

Colunas ANTES (INCORRETO):
```
chunk,train_chunk,test_chunk,model,train_size,test_size,accuracy,f1_weighted,f1_macro,gmean
```

Colunas DEPOIS (CORRETO):
```
chunk,train_chunk,test_chunk,model,train_size,test_size,
train_gmean,train_accuracy,train_f1,
test_gmean,test_accuracy,test_f1,test_f1_macro
```

### Comparacao com GBML (chunk_metrics.json)

**GBML**:
```json
{
  "chunk": 0,
  "train_gmean": 0.9886,
  "test_gmean": 0.9529,
  "test_f1": 0.9599
}
```

**River/ACDWM (CORRIGIDO)**:
```csv
chunk,train_gmean,test_gmean,train_accuracy,test_accuracy,train_f1,test_f1
0,0.9886,0.9529,0.9920,0.9600,0.9900,0.9580
```

---

## ARQUIVOS MODIFICADOS

### 1. run_comparative_on_existing_chunks.py

**Funcao**: `run_river_model_on_chunks()`
- **Linha 161-162**: Adicionada avaliacao no chunk de treino
- **Linha 164-165**: Avaliacao no chunk de teste (ja existia)
- **Linha 174-182**: Salva AMBOS train e test metrics

**Funcao**: `run_acdwm_on_chunks()`
- **Linha 232-233**: Adicionada avaliacao no chunk de treino
- **Linha 235-236**: Avaliacao no chunk de teste
- **Linha 245-253**: Salva AMBOS train e test metrics

---

## JUSTIFICATIVA CIENTIFICA

### Por que avaliar no chunk de treino?

1. **Deteccao de overfitting**:
   - train_gmean >> test_gmean = overfitting
   - train_gmean ≈ test_gmean = generalizacao boa

2. **Comparacao justa com GBML**:
   - GBML ja salvava train metrics
   - Comparacao estatistica requer mesma estrutura de dados

3. **Analise de adaptacao a drift**:
   - Queda brusca em test_gmean = drift detectado
   - train_gmean estavel + test_gmean baixo = modelo nao adaptou

4. **Testes estatisticos validos**:
   - Wilcoxon, Friedman, Nemenyi requerem MESMOS dados
   - Sem train metrics = comparacao invalida

---

## IMPACTO NA EXECUCAO

### Tempo de Execucao

**ANTES**:
- 1 avaliacao por chunk (apenas teste)
- Tempo: ~T

**DEPOIS**:
- 2 avaliacoes por chunk (treino + teste)
- Tempo estimado: ~1.5T a 2T

Para River (modelos rapidos): impacto minimo (~10-20% mais lento)
Para ACDWM (modelo lento): impacto maior (~50-100% mais lento)

### Espaco em Disco

Colunas adicionais por chunk:
- train_gmean
- train_accuracy
- train_f1

Impacto: +50% no tamanho dos CSVs de resultados (insignificante)

---

## PROXIMOS PASSOS

1. **Re-executar modelos comparativos** com codigo corrigido
2. **Validar estrutura de dados** - comparar com GBML
3. **Executar analises estatisticas** com dados completos
4. **Gerar plots comparativos** usando train e test metrics

---

## VALIDACAO

Para validar que a correcao funcionou:

```python
import pandas as pd

# Ler resultados River
df_river = pd.read_csv("river_HAT_results.csv")

# Verificar colunas
print("Colunas:", df_river.columns.tolist())
# Deve conter: train_gmean, train_accuracy, train_f1, test_gmean, test_accuracy, test_f1

# Verificar valores
print("\nPrimeiras linhas:")
print(df_river[['chunk', 'train_gmean', 'test_gmean']].head())

# Verificar se train > test (geralmente esperado)
print(f"\nMedia train_gmean: {df_river['train_gmean'].mean():.4f}")
print(f"Media test_gmean: {df_river['test_gmean'].mean():.4f}")
```

---

**Autor**: Claude Code
**Versao**: 2.0 (corrigida)
**Status**: Pronto para re-execucao no Colab
