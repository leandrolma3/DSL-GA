# 📊 Guia de Uso: Comparação GBML vs River

**Autor:** Claude Code
**Data:** 2025-01-06
**Versão:** 1.0

---

## 🎯 Objetivo

Este guia explica como usar o **sistema unificado de comparação** entre o framework GBML e os modelos clássicos do River, **garantindo que ambos usem exatamente os mesmos dados e metodologia**.

---

## 📁 Arquivos Criados

```
DSL-AG-hybrid/
├── shared_evaluation.py         # Interface unificada de avaliação
├── baseline_river.py            # Implementação dos modelos River
├── gbml_evaluator.py            # Wrapper do GBML
├── compare_gbml_vs_river.py     # Script principal de comparação
└── COMPARISON_USAGE.md          # Este arquivo
```

---

## ⚙️ Instalação de Dependências

```bash
# Instalar River (se ainda não instalado)
pip install river

# Outras dependências necessárias
pip install numpy pandas matplotlib seaborn pyyaml scikit-learn
```

---

## 🚀 Uso Básico

### **1. Comparação Rápida (SEA com 5 chunks)**

```bash
python compare_gbml_vs_river.py \
    --stream SEA_Abrupt_Simple \
    --chunks 5 \
    --chunk-size 6000 \
    --seed 42
```

**O que este comando faz:**
1. Carrega/gera 6 chunks de 6000 instâncias do stream SEA
2. Executa GBML nos chunks
3. Executa HAT, ARF e SRP (padrão) nos **mesmos chunks**
4. Salva resultados, gráficos e tabelas comparativas

---

### **2. Comparação com Modelos Específicos**

```bash
python compare_gbml_vs_river.py \
    --stream AGRAWAL_Abrupt_Simple \
    --models HAT ARF SRP ADWIN_BAGGING \
    --chunks 10 \
    --seed 42
```

---

### **3. Apenas River (sem GBML)**

Útil para testar rapidamente os baselines:

```bash
python compare_gbml_vs_river.py \
    --stream CovType \
    --models HAT ARF \
    --no-gbml
```

---

### **4. Apenas GBML (sem River)**

```bash
python compare_gbml_vs_river.py \
    --stream PokerHand \
    --no-river
```

---

## 📊 Resultados Gerados

Após executar a comparação, os seguintes arquivos são criados:

```
comparison_results/
└── SEA_Abrupt_Simple_seed42_20250106_143022/
    ├── gbml_results.csv              # Resultados detalhados do GBML
    ├── river_HAT_results.csv         # Resultados do HAT
    ├── river_ARF_results.csv         # Resultados do ARF
    ├── river_SRP_results.csv         # Resultados do SRP
    ├── comparison_table.csv          # Tabela unificada
    ├── summary.txt                   # Estatísticas resumidas
    ├── accuracy_comparison.png       # Gráfico de accuracy
    ├── gmean_comparison.png          # Gráfico de G-mean
    └── accuracy_heatmap.png          # Heatmap de performance
```

---

## 🔬 Uso Programático (Python)

### **Exemplo 1: Comparação Simples**

```python
from compare_gbml_vs_river import run_full_comparison

results = run_full_comparison(
    stream_name='SEA_Abrupt_Simple',
    chunk_size=6000,
    num_chunks=5,
    seed=42
)

print(f"Resultados salvos em: {results['experiment_dir']}")
```

### **Exemplo 2: Usando Chunks Pré-gerados**

```python
from shared_evaluation import load_or_generate_chunks
from baseline_river import create_river_model

# Gera/carrega chunks (serão cacheados)
chunks = load_or_generate_chunks(
    stream_name='AGRAWAL_Abrupt_Simple',
    chunk_size=6000,
    num_chunks=5,
    max_instances=36000,
    config_path='config.yaml',
    seed=42
)

# Cria e avalia modelo River
hat = create_river_model('HAT', classes=[0, 1, 2])
results_df = hat.evaluate_prequential(chunks)

print(results_df)
```

### **Exemplo 3: Teste Rápido com Dados Sintéticos**

```python
import numpy as np
from baseline_river import create_river_model

# Cria chunks fictícios
fake_chunks = [
    (
        [{'x1': np.random.rand(), 'x2': np.random.rand()} for _ in range(1000)],
        [np.random.randint(0, 2) for _ in range(1000)]
    )
    for _ in range(5)
]

# Testa modelo
model = create_river_model('ARF', classes=[0, 1], n_models=5)
results = model.evaluate_prequential(fake_chunks)

print(f"Accuracy média: {results['accuracy'].mean():.4f}")
```

---

## 🎛️ Parâmetros da Linha de Comando

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `--stream` | Nome do stream (obrigatório) | - |
| `--config` | Caminho para config.yaml | `config.yaml` |
| `--models` | Lista de modelos River | `HAT ARF SRP` |
| `--chunks` | Número de chunks | `5` |
| `--chunk-size` | Tamanho dos chunks | `6000` |
| `--seed` | Seed para reprodutibilidade | `42` |
| `--output` | Diretório de saída | `comparison_results` |
| `--no-gbml` | Não executar GBML | `False` |
| `--no-river` | Não executar River | `False` |
| `--force` | Forçar regeneração de chunks | `False` |

---

## 📝 Modelos River Disponíveis

| Código | Nome Completo | Descrição |
|--------|---------------|-----------|
| `HAT` | Hoeffding Adaptive Tree | Árvore incremental com adaptação a drift |
| `ARF` | Adaptive Random Forest | Ensemble de árvores com ADWIN |
| `SRP` | Streaming Random Patches | Random subspaces para streams |
| `ADWIN_BAGGING` | ADWIN Bagging | Bagging com detector ADWIN |
| `LEVERAGING_BAGGING` | Leveraging Bagging | Bagging com leveraging |

---

## 🔒 Garantias de Reprodutibilidade

O sistema **GARANTE** que:

✅ **Mesmos Chunks**: GBML e River usam exatamente os mesmos chunks
✅ **Mesma Ordem**: Instâncias na mesma ordem
✅ **Mesmas Seeds**: Aleatoriedade controlada
✅ **Mesmo Cache**: Chunks são salvos e reutilizados
✅ **Mesmas Métricas**: Cálculos idênticos (accuracy, F1, G-mean)

**Exemplo de Cache:**
```
chunks_cache/
├── SEA_Abrupt_Simple_cs6000_nc6_seed42.pkl
└── AGRAWAL_Abrupt_Simple_cs6000_nc5_seed42.pkl
```

---

## 🐛 Troubleshooting

### **Erro: River não instalado**
```bash
pip install river
```

### **Erro: Chunks inválidos**
```bash
# Força regeneração
python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --force
```

### **Erro: Memória insuficiente**
```bash
# Reduz chunk_size
python compare_gbml_vs_river.py --stream CovType --chunk-size 3000
```

### **Aviso: Cache corrompido**
```bash
# Remove cache
rm -rf chunks_cache/
# Regenera
python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --force
```

---

## 📈 Análise dos Resultados

### **Lendo a Tabela Comparativa**

```python
import pandas as pd

df = pd.read_csv('comparison_results/.../comparison_table.csv')

# Mostra resumo por modelo
summary = df.groupby('model')[['accuracy', 'f1_weighted', 'gmean']].agg(['mean', 'std'])
print(summary)
```

### **Comparando Dois Modelos**

```python
gbml = df[df['model'] == 'GBML']
hat = df[df['model'] == 'HAT']

# Teste t para diferença estatística
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(gbml['accuracy'], hat['accuracy'])

print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
```

---

## 🔄 Workflow Recomendado

### **Para Desenvolvimento/Debug**
```bash
# 1. Teste rápido com stream pequeno
python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --chunks 3

# 2. Valida que resultados são salvos corretamente
ls comparison_results/

# 3. Valida que cache funciona (segunda execução deve ser rápida)
python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --chunks 3
```

### **Para Experimentos Científicos**
```bash
# 1. Streams sintéticos com drift controlado
python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --chunks 10 --seed 42
python compare_gbml_vs_river.py --stream AGRAWAL_Gradual_Chain --chunks 10 --seed 42
python compare_gbml_vs_river.py --stream RBF_Abrupt_Severe --chunks 10 --seed 42

# 2. Datasets reais
python compare_gbml_vs_river.py --stream Electricity --chunks 20 --seed 42
python compare_gbml_vs_river.py --stream CovType --chunks 15 --seed 42

# 3. Múltiplas seeds para validação estatística
for seed in 42 123 456 789 1000; do
    python compare_gbml_vs_river.py --stream SEA_Abrupt_Simple --seed $seed
done
```

---

## 🧪 Testes Unitários

```bash
# Testa shared_evaluation.py
python shared_evaluation.py

# Testa baseline_river.py
python baseline_river.py

# Testa gbml_evaluator.py
python gbml_evaluator.py
```

---

## 📚 Próximos Passos

1. **Executar comparação em datasets pequenos** (validação)
2. **Analisar diferenças de performance**
3. **Identificar bugs no GBML** (se houver)
4. **Refatorar módulos core** (se necessário)
5. **Executar suite completa** de experimentos

---

## 📧 Suporte

Para dúvidas ou problemas:
1. Verifique os logs gerados
2. Tente com `--force` para regenerar dados
3. Reduza `--chunks` para debugging
4. Verifique se `config.yaml` está correto

---

**🎉 Sistema pronto para uso!**

O pipeline unificado garante comparações científicas válidas entre GBML e River.
