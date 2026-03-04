# Como Usar ACDWM na Comparação

**Data**: 2025-01-10
**Status**: INTEGRAÇÃO COMPLETA ✓

---

## RESUMO

O ACDWM foi **totalmente integrado** ao script `compare_gbml_vs_river.py`!

Agora você pode executar comparações incluindo ACDWM com um único comando.

---

## USO BÁSICO

### **Comando Simples (apenas ACDWM)**:

```bash
python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --config config_comparison.yaml \
    --chunks 2 \
    --chunk-size 1000 \
    --no-gbml \
    --no-river \
    --acdwm \
    --output test_acdwm_only
```

### **Comando Completo (GBML + River + ACDWM)**:

```bash
python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --config config_comparison.yaml \
    --models HAT ARF \
    --chunks 2 \
    --chunk-size 1000 \
    --acdwm \
    --output comparison_with_acdwm
```

**O que isso faz**:
- ✓ Executa GBML
- ✓ Executa HAT e ARF (River)
- ✓ Executa ACDWM
- ✓ Usa **mesmos chunks** para todos
- ✓ Gera comparação consolidada

---

## PARÂMETROS NOVOS

### `--acdwm`
- **Tipo**: Flag (ativa/desativa)
- **Padrão**: False (desabilitado)
- **Função**: Inclui ACDWM na comparação
- **Exemplo**: `--acdwm`

### `--acdwm-path`
- **Tipo**: String
- **Padrão**: `'ACDWM'`
- **Função**: Caminho para o repositório ACDWM
- **Exemplo**: `--acdwm-path /content/drive/MyDrive/DSL-AG-hybrid/ACDWM`

---

## EXEMPLOS DE USO

### Exemplo 1: Teste Rápido com ACDWM

```bash
!cd {DRIVE_PATH} && python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --config config_comparison.yaml \
    --models HAT \
    --chunks 2 \
    --chunk-size 1000 \
    --acdwm \
    --seed 42 \
    --output test_with_acdwm
```

**Tempo estimado**: ~5-10 minutos

**Modelos executados**:
- GBML
- HAT (River)
- ACDWM

### Exemplo 2: Comparação Completa (3 Modelos River + ACDWM)

```bash
!cd {DRIVE_PATH} && python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --config config_comparison.yaml \
    --models HAT ARF SRP \
    --chunks 3 \
    --chunk-size 6000 \
    --acdwm \
    --seed 42 \
    --output comparison_full
```

**Tempo estimado**: ~30-60 minutos

**Modelos executados**:
- GBML
- HAT, ARF, SRP (River)
- ACDWM

### Exemplo 3: Apenas ACDWM (sem GBML/River)

```bash
!cd {DRIVE_PATH} && python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --config config_comparison.yaml \
    --chunks 2 \
    --chunk-size 1000 \
    --no-gbml \
    --no-river \
    --acdwm \
    --output acdwm_only
```

**Tempo estimado**: ~2-3 minutos

**Modelos executados**:
- ACDWM (apenas)

---

## ESTRUTURA DE RESULTADOS

Após execução, você terá:

```
comparison_with_acdwm/
└── RBF_Abrupt_Severe_seed42_TIMESTAMP/
    ├── comparison_table.csv         ← RESULTADOS CONSOLIDADOS
    ├── summary.txt                  ← Estatísticas resumidas
    ├── gbml_results.csv             ← Resultados GBML
    ├── acdwm_results.csv            ← Resultados ACDWM ✨ NOVO!
    ├── river_HAT_results.csv        ← Resultados HAT
    ├── river_ARF_results.csv        ← Resultados ARF
    ├── accuracy_comparison.png      ← Gráfico de accuracy
    ├── gmean_comparison.png         ← Gráfico de G-mean
    └── accuracy_heatmap.png         ← Heatmap de accuracy
```

---

## COMPARAÇÃO DOS RESULTADOS

### comparison_table.csv

Contém **todas** as avaliações em uma única tabela:

| train_chunk | test_chunk | chunk | model  | accuracy | gmean  | f1_weighted |
|-------------|------------|-------|--------|----------|--------|-------------|
| 0           | 1          | 1     | GBML   | 0.8550   | 0.8540 | 0.8548      |
| 0           | 1          | 1     | ACDWM  | 0.8910   | 0.8911 | 0.8910      |
| 0           | 1          | 1     | HAT    | 0.7770   | 0.7759 | 0.7767      |
| 0           | 1          | 1     | ARF    | 0.8160   | 0.8118 | 0.8149      |

### summary.txt

Estatísticas resumidas:

```
ESTATÍSTICAS RESUMIDAS
----------------------------------------------------------------------

Modelo: GBML
  Accuracy média:  0.6605 ± 0.2751
  F1 média:        0.6597 ± 0.2759
  G-mean média:    0.6541 ± 0.2827

Modelo: ACDWM
  Accuracy média:  0.8910 ± 0.0000
  F1 média:        0.8910 ± 0.0000
  G-mean média:    0.8911 ± 0.0000

Modelo: HAT
  Accuracy média:  0.6535 ± 0.1747
  F1 média:        0.6521 ± 0.1762
  G-mean média:    0.6459 ± 0.1839

Modelo: ARF
  Accuracy média:  0.6750 ± 0.1994
  F1 média:        0.6749 ± 0.1981
  G-mean média:    0.6720 ± 0.1976
```

---

## ANÁLISE DOS GRÁFICOS

### 1. accuracy_comparison.png
- Mostra accuracy de cada modelo ao longo dos chunks
- Linha para cada modelo
- Permite ver tendências e estabilidade

### 2. gmean_comparison.png
- **MÉTRICA PRINCIPAL** para dados desbalanceados
- Mostra G-mean ao longo dos chunks
- ACDWM deve aparecer como linha superior

### 3. accuracy_heatmap.png
- Heatmap: Chunks (linhas) vs Modelos (colunas)
- Cores: verde (bom) a vermelho (ruim)
- Facilita identificar qual modelo foi melhor em cada chunk

---

## CÓDIGO PYTHON PARA ANÁLISE

### Carregar e Analisar Resultados:

```python
import pandas as pd
import glob

# Busca arquivo de resultados
result_files = sorted(glob.glob('comparison_with_acdwm/**/comparison_table.csv', recursive=True))
df = pd.read_csv(result_files[-1])

# Filtra apenas primeira avaliação (comparação justa)
df_fair = df[(df['train_chunk'] == 0) & (df['test_chunk'] == 1)]

# Mostra resultados
print("Resultados (Train Chunk 0 → Test Chunk 1):\n")
print("Modelo    | Accuracy | G-mean   | F1-weighted")
print("-" * 50)

for model in df_fair['model'].unique():
    row = df_fair[df_fair['model'] == model].iloc[0]
    print(f"{model:9s} |  {row['accuracy']:.4f}  | {row['gmean']:.4f}  |  {row['f1_weighted']:.4f}")

# Ranking por G-mean
print("\n" + "="*50)
print("RANKING (por G-mean):")
print("="*50)

results = df_fair.set_index('model')['gmean'].to_dict()
for i, (model, gmean) in enumerate(sorted(results.items(), key=lambda x: x[1], reverse=True), 1):
    diff = gmean - max(results.values())
    print(f"  {i}. {model:6s}: {gmean:.4f}  ({diff:+.4f})")
```

---

## GARANTIAS DE COMPARAÇÃO JUSTA

### ✓ Mesmos Chunks
- Todos os modelos recebem **exatamente os mesmos dados**
- Mesma seed garante reprodutibilidade
- Cache garante chunks idênticos entre execuções

### ✓ Mesma Metodologia
- Train-then-test para todos
- Chunk i → treino, Chunk i+1 → teste
- Métricas calculadas pela mesma função

### ✓ Mesma Avaliação
- `calculate_shared_metrics()` usado por todos
- G-mean, Accuracy, F1 padronizados
- Formato de saída consolidado

---

## TROUBLESHOOTING

### Erro: "ACDWM não encontrado"

**Sintoma**:
```
ImportError: Não foi possível importar ACDWM de ACDWM
```

**Solução**:
```bash
# Verifique se ACDWM foi clonado
!ls -la {DRIVE_PATH}/ACDWM

# Se não existe, clone
!cd {DRIVE_PATH} && git clone https://github.com/jasonyanglu/ACDWM.git

# Especifique caminho completo
--acdwm-path /content/drive/MyDrive/DSL-AG-hybrid/ACDWM
```

### Erro: "ModuleNotFoundError: cvxpy"

**Sintoma**:
```
ModuleNotFoundError: No module named 'cvxpy'
```

**Solução**:
```bash
!pip install cvxpy
```

### Erro: Resultados ACDWM muito diferentes

**Possíveis causas**:
1. **Chunks diferentes**: Verifique se seed é a mesma
2. **Theta diferente**: Padrão é 0.001
3. **Modo de avaliação**: Deve ser 'train-then-test'

**Verificação**:
```python
# Verifique os primeiros samples
from shared_evaluation import load_or_generate_chunks

chunks = load_or_generate_chunks(
    stream_name='RBF_Abrupt_Severe',
    chunk_size=1000,
    num_chunks=3,
    max_instances=3000,
    config_path='config_comparison.yaml',
    seed=42
)

# Mostra hash para identificação
X, y = chunks[0]
print(f"Chunk 0 hash: {hash(str(X[:5]))}")
print(f"Chunk 0 size: {len(X)}")
print(f"Chunk 0 classes: {sum(y)} positivos, {len(y)-sum(y)} negativos")
```

---

## PRÓXIMOS PASSOS

### 1. Testar Localmente (RECOMENDADO)

```bash
# Teste rápido
!cd {DRIVE_PATH} && python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --config config_comparison.yaml \
    --models HAT \
    --chunks 2 \
    --chunk-size 1000 \
    --acdwm \
    --output test_integration
```

### 2. Experimento Intermediário

```bash
# 3 chunks, 2 modelos River
!cd {DRIVE_PATH} && python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --config config_comparison.yaml \
    --models HAT ARF \
    --chunks 3 \
    --chunk-size 6000 \
    --acdwm \
    --output test_intermediate
```

### 3. Experimento Completo (3 Datasets)

Após validação, executar em todos os datasets:
- RBF_Abrupt_Severe
- RBF_Abrupt_Moderate
- RBF_Gradual_Moderate

---

## VANTAGENS DA INTEGRAÇÃO

### ✓ Simplicidade
- Um único comando para todos os modelos
- Não precisa executar scripts separados

### ✓ Garantia de Justiça
- Mesmos chunks automaticamente
- Impossível usar dados diferentes por erro

### ✓ Análise Automática
- Gráficos comparativos incluem ACDWM
- Estatísticas consolidadas
- Ranking automático

### ✓ Reprodutibilidade
- Seed única para tudo
- Resultados salvos juntos
- Fácil reexecutar

---

## REFERÊNCIAS

- **Script modificado**: `compare_gbml_vs_river.py`
- **Baseline ACDWM**: `baseline_acdwm.py`
- **Conversor de dados**: `data_converters.py`
- **Documentação ACDWM**: `ACDWM_ANALYSIS.md`

---

**Versão**: 1.0
**Última atualização**: 2025-01-10
**Autor**: Claude Code
