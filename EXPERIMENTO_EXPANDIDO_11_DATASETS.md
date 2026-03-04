# Experimento Expandido: 11 Datasets com Concept Drift

**Data:** 2025-11-12
**Status:** Pronto para Execução
**Validação Local:** ✅ Completada com Sucesso

---

## 📊 Resumo Executivo

Este documento descreve o experimento expandido com 11 datasets de drift simulation, projetado para rodar no Google Colab Pro (24h).

### Configuração Validada

**Baseado em:**
- Validação local com RBF_Abrupt_Severe
- chunk_size=3000 (redução de 50% vs 6000)
- **Tempo real: 15 min/chunk** (58% mais rápido que estimado!)

**Resultado:**
- ✅ GBML teve **melhor performance** com chunk_size=3000 (+1.6%)
- ✅ Margem de segurança **75%** no Colab Pro

---

## 🎯 11 Datasets Selecionados

### Distribuição por Gerador

| Gerador | Datasets | Drift Types |
|---------|----------|-------------|
| **RBF** | 2 | Abrupt Severe, Gradual Moderate |
| **SEA** | 3 | Abrupt Simple, Gradual Fast, Abrupt Recurring |
| **AGRAWAL** | 2 | Abrupt Severe, Gradual Chain |
| **HYPERPLANE** | 1 | Abrupt Simple |
| **STAGGER** | 1 | Abrupt Chain |
| **SINE** | 2 | Abrupt Simple, Gradual Recurring |

### Lista Completa

1. **RBF_Abrupt_Severe** ✅ Validado
   - Drift: Abrupto
   - Severidade: Severa
   - Chunks: 4 (c1:2 → c2_severe:2)

2. **RBF_Gradual_Moderate**
   - Drift: Gradual (2 chunks de transição)
   - Severidade: Moderada
   - Chunks: 4 (c1:2 → c3_moderate:2)

3. **SEA_Abrupt_Simple**
   - Drift: Abrupto
   - Complexity: Simples
   - Chunks: 4 (f1:2 → f3:2)

4. **SEA_Gradual_Simple_Fast**
   - Drift: Gradual (1 chunk de transição)
   - Speed: Rápido
   - Chunks: 4 (f1:2 → f3:2)

5. **SEA_Abrupt_Recurring**
   - Drift: Abrupto Recorrente
   - Pattern: f1 → f3 → f1
   - Chunks: 4 (f1:1 → f3:2 → f1:1)

6. **AGRAWAL_Abrupt_Simple_Severe**
   - Drift: Abrupto
   - Severidade: Severa (f1 → f6)
   - Chunks: 4 (f1:2 → f6:2)

7. **AGRAWAL_Gradual_Chain**
   - Drift: Gradual em Cadeia
   - Pattern: f1 → f3 → f5 → f7
   - Chunks: 4 (1 chunk cada)

8. **HYPERPLANE_Abrupt_Simple**
   - Drift: Abrupto
   - Dimensões: 10 features, 4 classes
   - Chunks: 4 (plane1:2 → plane2:2)

9. **STAGGER_Abrupt_Chain**
   - Drift: Abrupto em Cadeia
   - Pattern: f1 → f2 → f3
   - Chunks: 4 (f1:1 → f2:2 → f3:1)

10. **SINE_Abrupt_Simple**
    - Drift: Abrupto
    - Types: sum → prod
    - Chunks: 4 (f1_sum:2 → f2_prod:2)

11. **SINE_Gradual_Recurring**
    - Drift: Gradual Recorrente
    - Pattern: f1_sum → f3_sum_alt → f1_sum
    - Chunks: 4 (f1:1 → f3:2 → f1:1)

---

## ⏱️ Projeção de Tempo

### Configuração Base (5 Modelos Python)

| Componente | Cálculo | Tempo |
|------------|---------|-------|
| Total de chunks | 11 datasets × 4 chunks | 44 chunks |
| GBML | 44 × 15 min | **660 min** |
| ACDWM | 44 × 1 min | 44 min |
| HAT, ARF, SRP | 44 × 1 min × 3 | 132 min |
| **Total** | | **11 horas** |
| **Margem Colab Pro** | 24h - 11h | **13h (54%)** ✅✅ |

### Com ERulesD2S (6 Modelos)

| Componente | Cálculo | Tempo |
|------------|---------|-------|
| Python models | | 11 horas |
| ERulesD2S | 44 × 4 min | 176 min (~3h) |
| **Total** | | **14 horas** |
| **Margem Colab Pro** | 24h - 14h | **10h (42%)** ✅ |

---

## 📈 Poder Estatístico

### Comparação com Experimento Anterior

| Métrica | Anterior | Expandido | Aumento |
|---------|----------|-----------|---------|
| Datasets | 3 | 11 | +267% |
| Geradores | 1 (RBF) | 5 | +400% |
| Avaliações/modelo | 8 | 33 | +312% |
| Chunk size | 6000 | 3000 | -50% |
| Tempo total | 7.3h | 11h | +51% |

### Benefícios Estatísticos

1. **Maior Robustez:** 33 avaliações vs 8 anteriores
2. **Maior Diversidade:** 5 geradores vs 1
3. **Cobertura Abrangente:** 7 abrupt + 4 gradual
4. **Padrões Variados:** Simple, Chain, Recurring
5. **Intervalos de Confiança:** Redução de ~40% na largura

---

## 🤖 Integração do ERulesD2S (Bartoz)

### Características do ERulesD2S

| Aspecto | Detalhes |
|---------|----------|
| Tecnologia | Java + MOA + GPU (CUDA) |
| População | 25 indivíduos |
| Gerações | 50 |
| Chunk size | 1000 (padrão) |
| Tempo/chunk | ~4 min (25% do GBML) |
| Explicabilidade | ✅ Regras interpretáveis |

### Desafios de Integração

1. **Framework Diferente:** Java vs Python
2. **Formato de Dados:** ARFF vs numpy arrays
3. **Execução:** Subprocess para MOA
4. **Parsing:** Resultados em formato MOA
5. **GPU:** Requer CUDA (já disponível no Colab)

### Duas Opções de Execução

#### **Opção A (Recomendada): Python Models Primeiro**

**Vantagens:**
- ✅ Menor risco técnico
- ✅ Margem de segurança maior (54%)
- ✅ Resultados Python consolidados
- ✅ Pode adicionar ERulesD2S depois

**Execução:**
1. Rodar experimento com 5 modelos Python (11h)
2. Analisar resultados
3. Preparar integração ERulesD2S em experimento separado
4. Rodar ERulesD2S focado (3-5 datasets selecionados)

**Comando:**
```bash
python compare_gbml_vs_river.py \
  --config config_experiment_expanded.yaml \
  --models HAT ARF SRP \
  --acdwm \
  --seed 42 \
  --output experiment_expanded_results
```

#### **Opção B (Ambiciosa): Integração Completa**

**Vantagens:**
- ✅ Resultados completos em uma execução
- ✅ Comparação direta com ERulesD2S
- ✅ Ainda viável (42% margem)

**Desvantagens:**
- ⚠️ Requer wrapper Python→Java/MOA
- ⚠️ Maior complexidade técnica
- ⚠️ Risco de falhas na integração
- ⚠️ Debugging mais difícil

**Passos Necessários:**
1. Criar wrapper Python para ERulesD2S
2. Converter dados para ARFF
3. Executar via subprocess
4. Parsear resultados MOA
5. Integrar no pipeline de comparação

**Tempo de Desenvolvimento:** ~4-6 horas

---

## 🚀 Guia de Execução

### Pré-requisitos

1. **Google Colab Pro**
   - GPU: T4 ou superior
   - RAM: 12GB+
   - Tempo: 24h session

2. **Google Drive**
   - Espaço: ~5GB para resultados
   - Montar drive para persistência

3. **Arquivos Necessários**
   - `config_experiment_expanded.yaml` ✅
   - `compare_gbml_vs_river.py`
   - Todos os módulos Python (ga.py, river_evaluator.py, etc.)

### Passo a Passo (Opção A)

#### 1. Setup Inicial
```python
# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clonar repositório ou fazer upload
!git clone <repo-url>
# OU
!unzip /content/drive/MyDrive/DSL-AG-hybrid.zip

# Navegar para diretório
%cd DSL-AG-hybrid
```

#### 2. Instalar Dependências
```python
!pip install river scikit-learn pandas numpy matplotlib seaborn pyyaml xgboost
```

#### 3. Verificar Configuração
```python
!python -c "import river; print(f'River: {river.__version__}')"
!python -c "import yaml; config = yaml.safe_load(open('config_experiment_expanded.yaml')); print(f'Datasets: {len(config[\"experiment_settings\"][\"drift_simulation_experiments\"])}')"
```

#### 4. Executar Experimento
```python
# Execução completa com todos os modelos
!python compare_gbml_vs_river.py \
  --config config_experiment_expanded.yaml \
  --models HAT ARF SRP \
  --acdwm \
  --seed 42 \
  --output experiment_expanded_results
```

#### 5. Monitorar Progresso
```python
# Em outra célula (executar periodicamente)
!tail -n 50 experiment_expanded_results/latest.log

# Verificar resultados parciais
!ls -lh experiment_expanded_results/*/comparison_table.csv
```

#### 6. Salvar Resultados
```python
# Copiar para Drive regularmente
!cp -r experiment_expanded_results /content/drive/MyDrive/

# Compactar resultados
!tar -czf experiment_expanded_results.tar.gz experiment_expanded_results/
!cp experiment_expanded_results.tar.gz /content/drive/MyDrive/
```

---

## 📊 Resultados Esperados

### Arquivos Gerados por Dataset

Para cada um dos 11 datasets:
```
experiment_expanded_results/
├── RBF_Abrupt_Severe_seed42_<timestamp>/
│   ├── comparison_table.csv          # Métricas consolidadas
│   ├── gbml_results.csv               # Resultados GBML
│   ├── acdwm_results.csv              # Resultados ACDWM
│   ├── river_HAT_results.csv          # Resultados HAT
│   ├── river_ARF_results.csv          # Resultados ARF
│   ├── river_SRP_results.csv          # Resultados SRP
│   ├── best_individual_chunk_*.pkl    # Melhores indivíduos GBML
│   ├── gmean_comparison.png           # Gráfico G-mean
│   ├── accuracy_comparison.png        # Gráfico Accuracy
│   └── summary.txt                    # Resumo textual
```

### Métricas Coletadas

Para cada avaliação (33 por modelo):
- **Accuracy**
- **G-mean** (métrica principal)
- **F1-score weighted**
- **Precision/Recall por classe**
- **Tempo de treinamento**
- **Tempo de predição**

### Análise Consolidada

Script de análise final:
```python
python analyze_experiment_expanded.py
```

Outputs:
1. **Tabela Consolidada:** Todos os resultados (11 datasets × 5 modelos)
2. **Rankings:** Por dataset e global
3. **Análise Estatística:** Testes de significância
4. **Gráficos Comparativos:** Boxplots, heatmaps, rankings
5. **Relatório PDF:** Documento completo com análises

---

## 🔍 Análise Estatística Planejada

### Testes a Realizar

1. **Teste Global**
   - Kruskal-Wallis: Diferenças entre 5 modelos
   - α = 0.05

2. **Comparações Pareadas**
   - Mann-Whitney U: GBML vs cada modelo
   - Correção de Bonferroni: α = 0.005

3. **Tamanho de Efeito**
   - Cohen's d: Magnitude das diferenças
   - Interpretação: negligível/pequeno/médio/grande

4. **Intervalos de Confiança**
   - 95% CI para G-mean médio
   - Overlap analysis

5. **Análise por Tipo de Drift**
   - Abrupt vs Gradual
   - Modelos melhores em cada tipo?

6. **Análise por Gerador**
   - Diferenças entre RBF, SEA, AGRAWAL, etc.
   - Consistência dos rankings

### Hipóteses a Testar

**H1:** GBML é estatisticamente equivalente aos outros modelos
**H2:** GBML tem melhor explicabilidade com performance competitiva
**H3:** Modelos ensemble (ACDWM, ARF, SRP) têm menor variância
**H4:** HAT adapta-se melhor a drift gradual
**H5:** GBML recupera-se mais rápido de drift abrupto

---

## 💾 Tamanho Estimado dos Resultados

| Componente | Tamanho Unitário | Quantidade | Total |
|------------|------------------|------------|-------|
| CSVs | 10 KB | 66 | 660 KB |
| PKLs (GBML) | 50 KB | 33 | 1.65 MB |
| PNGs | 300 KB | 33 | 9.9 MB |
| Logs | 500 KB | 11 | 5.5 MB |
| **Total** | | | **~20 MB** |

Compactado: ~5 MB

---

## ⚠️ Troubleshooting

### Problema 1: Sessão desconecta

**Solução:**
```python
# Adicionar ao notebook para manter conexão
import time
from IPython.display import Javascript

def keep_alive():
    display(Javascript('''
        function KeepClicking(){
            console.log("Keeping alive...");
            document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click()
        }
        setInterval(KeepClicking, 60000)
    '''))

keep_alive()
```

### Problema 2: Out of Memory

**Solução:**
- Reduzir `population_size` para 75
- Reduzir `max_generations` para 150
- Executar em batches de 5-6 datasets

### Problema 3: Timeout de chunk

**Solução:**
- Aumentar timeout no código
- Verificar se há chunks travados
- Restart runtime e retomar

### Problema 4: Resultados não salvam

**Solução:**
- Verificar permissões do Drive
- Copiar manualmente para Drive a cada 2-3 datasets
- Usar checkpoints

---

## 📅 Cronograma Estimado

| Fase | Duração | Descrição |
|------|---------|-----------|
| Setup | 10 min | Montar Drive, instalar deps |
| Dataset 1 | 1h | RBF_Abrupt_Severe |
| Dataset 2 | 1h | RBF_Gradual_Moderate |
| Dataset 3 | 1h | SEA_Abrupt_Simple |
| Dataset 4 | 1h | SEA_Gradual_Simple_Fast |
| Dataset 5 | 1h | SEA_Abrupt_Recurring |
| Dataset 6 | 1h | AGRAWAL_Abrupt_Simple_Severe |
| Dataset 7 | 1h | AGRAWAL_Gradual_Chain |
| Dataset 8 | 1h | HYPERPLANE_Abrupt_Simple |
| Dataset 9 | 1h | STAGGER_Abrupt_Chain |
| Dataset 10 | 1h | SINE_Abrupt_Simple |
| Dataset 11 | 1h | SINE_Gradual_Recurring |
| **Total** | **11h** | |
| Backup | 10 min | Copiar para Drive |

**Conclusão:** 12:00 - 23:30 (mesmo dia)

---

## 🎯 Próximos Passos Após Execução

### Imediato (Dia 1)
1. ✅ Verificar completude dos resultados
2. ✅ Fazer backup completo no Drive
3. ✅ Executar análise consolidada
4. ✅ Gerar relatório preliminar

### Curto Prazo (Semana 1)
1. Análise estatística completa
2. Comparação com experimento anterior (3 datasets)
3. Identificar padrões e insights
4. Preparar gráficos para publicação

### Médio Prazo (Semana 2-3)
1. **Fase 4:** Integração do ERulesD2S
2. Executar ERulesD2S em 3-5 datasets selecionados
3. Comparação ERulesD2S vs outros modelos
4. Análise comparativa de explicabilidade

### Longo Prazo (Mês 1-2)
1. Redigir artigo científico
2. Preparar apresentação
3. Submeter para conferência/journal

---

## 📚 Referências

1. **Validação Local:** validation_local_results/
2. **Configuração:** config_experiment_expanded.yaml
3. **Plano Original:** PLANO_EXPERIMENTO_EXPANDIDO.md
4. **Análise Estatística Anterior:** CONCLUSAO_ESTATISTICA_GBML.md
5. **Instruções ERulesD2S:** Bartoz/INSTRUÇÕES_ERULESD2S.md

---

## ✅ Checklist Pré-Execução

- [ ] Config file validado: `config_experiment_expanded.yaml`
- [ ] Google Drive montado e com espaço (5GB+)
- [ ] GPU habilitada no Colab Pro (T4+)
- [ ] Dependências instaladas (river, sklearn, xgboost)
- [ ] Todos os arquivos Python presentes
- [ ] Keep-alive script ativado
- [ ] Backup strategy definida
- [ ] Monitor de progresso configurado

---

**Documento criado:** 2025-11-12
**Última atualização:** 2025-11-12
**Versão:** 1.0
**Status:** ✅ Pronto para Execução
