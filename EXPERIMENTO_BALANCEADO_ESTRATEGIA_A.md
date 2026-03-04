# EXPERIMENTO BALANCEADO - ESTRATEGIA A (SUAVE)

**Data**: 2025-12-15
**Objetivo**: Testar GBML com balanceamento entre performance e complexidade
**Status**: PRONTO PARA EXECUCAO

---

## 1. MODIFICACAO NO CODIGO

### 1.1 Arquivo Modificado

**fitness.py, linhas 382-389**

**ANTES (foco total em performance):**
```python
fitness_score = performance_score - (0 * total_penalty)
```

**DEPOIS (balanceado):**
```python
# PENALTY_WEIGHT: Controla o balanceamento entre performance e complexidade
# 0.0 = Foco total em performance (ignora complexidade)
# 0.1 = Estrategia Suave (10% peso para penalidades) - BALANCEADO
# 0.3 = Estrategia Moderada (30% peso para penalidades)
# 0.5 = Estrategia Forte (50% peso para penalidades)
PENALTY_WEIGHT = 0.1  # Estrategia A (Suave) - Balanceamento performance vs complexidade

fitness_score = performance_score - (PENALTY_WEIGHT * total_penalty)
```

### 1.2 Efeito da Modificacao

| Componente | Peso Anterior | Peso Atual |
|------------|---------------|------------|
| Performance (G-Mean + F1) | 100% | 100% |
| Penalidade de Complexidade | 0% | 10% |
| Penalidade de Features | 0% | 10% |
| Penalidade de Estabilidade | 0% | 10% |

---

## 2. YAMLS GERADOS

### 2.1 Arquivos Criados

| Arquivo | Tipo | Datasets |
|---------|------|----------|
| `config_balanced_batch_1.yaml` | drift_simulation | 12 (abrupt) |
| `config_balanced_batch_2.yaml` | drift_simulation | 9 (gradual) |
| `config_balanced_batch_3.yaml` | drift_simulation | 8 (ruido) |
| `config_balanced_batch_4.yaml` | drift_simulation | 6 (SINE/LED/WAVEFORM) |
| `config_balanced_batch_5.yaml` | standard | 5 (reais) |
| `config_balanced_batch_6.yaml` | standard | 6 (sinteticos) |
| `config_balanced_batch_7.yaml` | standard | 6 (sinteticos) |

### 2.2 Validacao de Consistencia

As unicas diferencas entre os YAMLs originais e balanceados sao:
- `base_results_dir`: experiments_chunk2000_* -> experiments_balanced_*
- `heatmap_save_directory`: experiments_chunk2000_* -> experiments_balanced_*

**Todos os outros parametros sao IDENTICOS**, garantindo comparacao justa:
- chunk_size: 2000
- num_chunks: 6
- population_size: 120
- max_generations: 200
- elitism_rate: 0.1
- initial_regularization_coefficient: 0.001
- feature_penalty_coefficient: 0.1
- gmean_bonus_coefficient: 0.1
- (e todos os demais)

---

## 3. ESTRUTURA DE DIRETORIOS

### 3.1 Experimentos Anteriores (PENALTY_WEIGHT=0.0)

```
experiments_chunk2000_phase1/
  batch_1/  (12 datasets abrupt drift)
  batch_2/  (9 datasets gradual drift)
  batch_3/  (8 datasets com ruido)
  batch_4/  (6 datasets SINE/LED/WAVEFORM)

experiments_chunk2000_phase2/
  batch_5/  (5 datasets reais)
  batch_6/  (6 sinteticos estacionarios)
  batch_7/  (6 sinteticos estacionarios)
```

### 3.2 Experimentos Balanceados (PENALTY_WEIGHT=0.1)

```
experiments_balanced_phase1/
  batch_1/  (12 datasets abrupt drift)
  batch_2/  (9 datasets gradual drift)
  batch_3/  (8 datasets com ruido)
  batch_4/  (6 datasets SINE/LED/WAVEFORM)

experiments_balanced_phase2/
  batch_5/  (5 datasets reais)
  batch_6/  (6 sinteticos estacionarios)
  batch_7/  (6 sinteticos estacionarios)
```

---

## 4. COMANDOS PARA EXECUCAO

### 4.1 No Google Colab

```python
# Celula 1: Montar Drive
from google.colab import drive
drive.mount('/content/drive')

# Celula 2: Ir para diretorio do projeto
%cd /content/drive/Othercomputers/Laptop-CIn/Downloads/DSL-AG-hybrid

# Celula 3: Executar batch desejado
!python main.py --config configs/config_balanced_batch_1.yaml
```

### 4.2 Sequencia Recomendada

1. **Batch 5** (datasets reais) - mais rapido para validar
2. **Batches 6 e 7** (sinteticos estacionarios)
3. **Batches 1-4** (drift simulation - mais demorados)

---

## 5. HIPOTESES DO EXPERIMENTO

### 5.1 Hipotese Principal

Com PENALTY_WEIGHT=0.1, esperamos:
- Reducao no numero medio de regras (de ~25 para ~15-20)
- Reducao no numero de condicoes por regra
- G-Mean similar ou levemente menor (queda maxima esperada: 2-5%)

### 5.2 Metricas de Comparacao

| Metrica | Exp. Anterior (0.0) | Exp. Balanceado (0.1) |
|---------|---------------------|----------------------|
| G-Mean medio | ~0.85-0.95 | Esperado: ~0.83-0.93 |
| Regras/chunk | ~25 | Esperado: ~15-20 |
| Condicoes/regra | ~5.6 | Esperado: ~4-5 |
| Features usadas | ~8-10 | Esperado: ~6-8 |

---

## 6. ANALISE POS-EXPERIMENTO

### 6.1 Scripts de Analise

Apos execucao, usar os mesmos scripts de analise:
- `analyze_explainability.py`
- `generate_plots.py`
- Scripts de consolidacao

### 6.2 Comparacao entre Experimentos

Para comparar resultados:
1. Extrair metricas de ambos os experimentos
2. Comparar G-Mean, numero de regras, complexidade
3. Avaliar trade-off performance vs explicabilidade

---

## 7. ARQUIVOS RELACIONADOS

| Arquivo | Descricao |
|---------|-----------|
| `fitness.py` | Codigo modificado (PENALTY_WEIGHT=0.1) |
| `generate_balanced_configs.py` | Script de geracao dos YAMLs |
| `configs/config_balanced_batch_*.yaml` | YAMLs do experimento |
| `ANALISE_SISTEMA_FITNESS_GBML.md` | Analise detalhada do sistema de fitness |

---

## 8. CONSIDERACOES IMPORTANTES

### 8.1 Reversibilidade

Para voltar ao comportamento anterior (foco total em performance):
- Alterar `PENALTY_WEIGHT = 0.0` em fitness.py

### 8.2 Ajustes Futuros

Se os resultados mostrarem:
- **Queda excessiva de G-Mean**: Reduzir PENALTY_WEIGHT para 0.05
- **Pouca reducao de complexidade**: Aumentar PENALTY_WEIGHT para 0.2 ou 0.3
- **Necessidade de ajuste fino**: Modificar coeficientes individuais no YAML

---

**Autor**: Claude Code
**Script de geracao**: `generate_balanced_configs.py`
**Versao**: 1.0
