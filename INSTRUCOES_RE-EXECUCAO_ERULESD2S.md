# Instruções para Re-executar ERulesD2S com Parser Corrigido

**Data**: 2025-11-19
**Correção aplicada**: erulesd2s_wrapper.py - método parse_results()

---

## CORREÇÃO APLICADA COM SUCESSO

O arquivo `erulesd2s_wrapper.py` foi atualizado localmente com o parser corrigido.

**Mudança**: O método `parse_results()` agora lida corretamente com headers duplicados no CSV.

---

## OPÇÃO 1: Re-executar no Colab (RECOMENDADO)

### Passo 1: Atualizar arquivo no Drive

O arquivo corrigido está em:
```
C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid\erulesd2s_wrapper.py
```

Este arquivo será copiado para o Drive quando você executar a CÉLULA 2 do notebook.

### Passo 2: Abrir notebook no Colab

Abra: `Batch_1_Comparative_Models.ipynb`

### Passo 3: Re-executar células necessárias

Execute as células NA ORDEM:

**CÉLULA 1**: Montar Drive
```python
# Monta Google Drive
```

**CÉLULA 2**: Copiar repositório
```python
# Copia DSL-AG-hybrid do Drive para /content
# Isso vai copiar o erulesd2s_wrapper.py CORRIGIDO
```

**CÉLULA 3**: Instalar dependências
```python
# pip install -q river scikit-learn pyyaml matplotlib seaborn pandas numpy
```

**CÉLULA 4**: Clonar ACDWM
```python
# git clone ACDWM
```

**CÉLULA 8**: Instalar Java e Maven
```python
# apt-get install openjdk-11-jdk maven
```

**CÉLULA 9**: Setup ERulesD2S
```python
# python setup_erulesd2s.py
```

### Passo 4: Re-executar CÉLULA 7 (COMPLETA - ~30 min)

```python
DATASETS = [
    "SEA_Abrupt_Simple",
    "SEA_Abrupt_Chain",
    "SEA_Abrupt_Recurring",
    "AGRAWAL_Abrupt_Simple_Mild",
    "AGRAWAL_Abrupt_Simple_Severe",
    "AGRAWAL_Abrupt_Chain_Long",
    "RBF_Abrupt_Severe",
    "RBF_Abrupt_Blip",
    "STAGGER_Abrupt_Chain",
    "STAGGER_Abrupt_Recurring",
    "HYPERPLANE_Abrupt_Simple",
    "RANDOMTREE_Abrupt_Simple"
]

# Executar TODOS os modelos em TODOS os datasets
```

**Tempo estimado**: ~30-35 minutos

**Saída esperada por dataset**:
```
[INFO] Accuracy extraída do CSV: 0.6290  ← NOVO! Era 0.0000 antes
SEA_Abrupt_Simple: SUCESSO (2.22 min)
```

### Passo 5: Re-executar CÉLULA 11 (Consolidação)

```python
# Consolidar todos os resultados incluindo ERulesD2S corrigido
```

**Saída esperada**:
```
ERulesD2S   test_gmean = 0.55-0.65  (ao invés de 0.0454)
```

---

## OPÇÃO 2: Testar com 1 Dataset Primeiro (RÁPIDO - 5 min)

Para validar a correção antes de rodar tudo:

### Modificar CÉLULA 7 temporariamente:

```python
DATASETS = [
    "SEA_Abrupt_Simple"  # APENAS 1 para teste
]
```

Execute CÉLULA 7.

**Resultado esperado**:
```
[INFO] Accuracy extraída do CSV: 0.6290
SEA_Abrupt_Simple: SUCESSO (2.22 min)
```

Se funcionar, volte a lista completa e execute novamente.

---

## VERIFICAÇÃO DE SUCESSO

### Durante execução da CÉLULA 7:

Procure por mensagens como:
```
[INFO] Accuracy extraída do CSV: 0.6290  ← BOM! Parser funcionando
```

Se continuar vendo `0.0000`, o arquivo pode não ter sido copiado corretamente.

### Após CÉLULA 11:

Compare os valores:

**ANTES (com bug)**:
```
ERulesD2S   test_gmean = 0.0454  (4.5%)
```

**DEPOIS (corrigido)**:
```
ERulesD2S   test_gmean = 0.55-0.65  (55-65%)
```

---

## TROUBLESHOOTING

### Problema: Ainda retorna 0.0

**Causa**: Arquivo erulesd2s_wrapper.py não foi atualizado no Colab

**Solução**:
1. Verificar se CÉLULA 2 foi executada após salvar a correção
2. Confirmar que o arquivo no Drive foi atualizado
3. Executar manualmente:
   ```python
   !cat /content/DSL-AG-hybrid/erulesd2s_wrapper.py | grep "CORRIGIDO"
   # Deve mostrar: "CORRIGIDO: Lida com headers duplicados"
   ```

### Problema: ImportError após correção

**Causa**: Sintaxe incorreta no arquivo corrigido

**Solução**:
1. Verificar que não há erros de indentação
2. Confirmar que imports estão corretos (Path, Dict)
3. Re-copiar o arquivo parse_results_CORRIGIDO.py

---

## RESULTADOS ESPERADOS FINAIS

### Comparação Modelos (test_gmean):

```
Rank   Model      Mean Test G-mean
------------------------------------------
1      ACDWM      0.8042
2      GBML       0.7872
3      ARF        0.7462
4      SRP        0.7406
5      ERulesD2S  0.55-0.65  ← CORRIGIDO (era 0.0454)
6      HAT        0.6252
```

### Testes Estatísticos:

ERulesD2S agora deve ser incluído ou excluído baseado em critério de performance, não por erro de parser.

---

## TEMPO TOTAL ESTIMADO

- Setup células 1-4, 8-9: ~10 min
- CÉLULA 7 (12 datasets): ~30-35 min
- CÉLULA 11 (consolidação): ~1 min

**Total**: ~45 minutos

---

## BACKUP

Antes de começar, faça backup do log atual:
```python
!cp batch_1_all_models_with_gbml.csv batch_1_all_models_with_gbml_BEFORE_FIX.csv
```

Assim você pode comparar antes/depois.

---

**Status**: PRONTO PARA RE-EXECUÇÃO
**Próximo passo**: Executar CÉLULA 2 no Colab para copiar arquivo corrigido
