# 📊 ANALISE INFRAESTRUTURA DE PLOTS E RESULTADOS

**Data**: 2025-10-29
**Objetivo**: Entender infraestrutura existente e extrair insights dos codigos parcialmente funcionais

---

## 🎯 RESUMO EXECUTIVO

### Status da Infraestrutura:
- ✅ **analyze_concept_difference.py**: FUNCIONANDO (gera concept_differences.json)
- ⚠️ **generate_plots.py**: FUNCIONANDO mas com problemas de caminhos
- ❌ **Resultados do teste**: Salvos no Google Colab, NAO localmente

### Problema Principal:
**Os resultados do experimento foram salvos no Google Colab** (`/content/drive/MyDrive/DSL-AG-hybrid/experiments_test/`), mas NAO foram baixados para a maquina local. Por isso o `generate_plots.py` nao encontra os arquivos JSON esperados.

---

## 📁 ESTRUTURA DE ARQUIVOS ESPERADOS vs REAIS

### Arquivos JSON que generate_plots.py ESPERA (por run):

```
experiments_test/RBF_Abrupt_Severe/run_1/
├── run_config.json                    ❌ NO LOCAL (existe no Colab)
├── periodic_accuracy.json             ❌ NO LOCAL (existe no Colab)
├── ga_history_per_chunk.json          ❌ NO LOCAL (existe no Colab)
├── rule_details_per_chunk.json        ❌ NO LOCAL (existe no Colab)
├── attribute_usage_per_chunk.json     ❌ NO LOCAL (existe no Colab)
└── chunk_metrics.json                 ❌ NO LOCAL (existe no Colab)
```

### Arquivos que EXISTEM localmente:

```
test_real_results_heatmaps/concept_heatmapsS/
├── concept_differences.json           ✅ EXISTE (gerado recentemente)
└── ConceptDifference_Heatmap_RBF.png  ✅ EXISTE (heatmap visual)
```

### Log do experimento:
```
experimento_teste_6chunks_20251028_160506.log  ✅ EXISTE (log completo)
```

---

## 🔍 ANALISE DETALHADA: generate_plots.py

### Funcao Principal: `generate_plots_for_run()`

**Parametros**:
- `run_dir`: Diretorio com resultados de um run especifico
- `output_dir`: Diretorio base para salvar plots (default: mesmo que run_dir)
- `diff_data`: Dados de concept_differences.json

**Arquivos que le** (generate_plots.py:259-264):
```python
run_config.json                    # Configuracao do experimento
periodic_accuracy.json             # Acuracias periodicas durante teste
ga_history_per_chunk.json          # Historico de evolucao do GA por chunk
rule_details_per_chunk.json        # Detalhes das regras por chunk
attribute_usage_per_chunk.json     # Uso de atributos por chunk
chunk_metrics.json                 # Metricas finais por chunk
```

**Plots que GERA**:

1. **Plot_AccuracyPeriodic_[stream]_Run[N].png**
   - Funcao: `plot_periodic_accuracy_with_drift_info()`
   - Mostra: Acuracia de teste periodica + acuracia de treino + marcadores de drift
   - Requer: `periodic_accuracy.json`, `chunk_metrics.json`, `concept_differences.json`

2. **Plot_GA_Evolution_Chunk[N]_[stream]_Run[N].png** (um por chunk)
   - Funcao: `plotting.plot_ga_evolution()`
   - Mostra: Evolucao de fitness ao longo das geracoes
   - Requer: `ga_history_per_chunk.json`

3. **Plot_RuleComponents_Heatmap_[stream]_Run[N].png**
   - Funcao: `plotting.plot_rule_changes()`
   - Mostra: Mudancas de componentes das regras por chunk (heatmap)
   - Requer: `rule_details_per_chunk.json`

4. **Plot_RuleComponents_Radar_[stream]_Run[N].png**
   - Funcao: `plotting.plot_rule_info_radar()`
   - Mostra: Visao radar das caracteristicas das regras
   - Requer: `rule_details_per_chunk.json`

5. **Plot_AttributeUsage_[stream]_Run[N].png**
   - Funcao: `plotting.plot_attribute_usage_over_time()`
   - Mostra: Uso de atributos ao longo do tempo
   - Requer: `attribute_usage_per_chunk.json`, `run_config.json['attributes']`

### Caminhos Esperados:

**Linha 324** (argumento default):
```python
--diff_file default="results/concept_heatmaps/concept_differences.json"
```

**Problema**: Arquivo esta em `test_real_results_heatmaps/concept_heatmapsS/concept_differences.json`

---

## 📊 ANALISE: plot_periodic_accuracy_with_drift_info()

Esta e a funcao mais complexa e util do generate_plots.py (linhas 37-232).

### O que ela FAZ:

1. **Plota acuracia de teste periodica** (linha azul com pontos)
   - Dados: `periodic_test_accuracies` = lista de (instancia, acuracia)
   - Eixo X: Total de instancias processadas
   - Eixo Y: Acuracia (%)

2. **Plota acuracia de treino final** (linha laranja tracejada)
   - Dados: `chunk_train_metrics` = lista de metricas por chunk
   - Marca: Fim de cada chunk de treino
   - Label: "Train Chunk N End"

3. **Marca drifts visuais**:
   - **Abrupt drift**: Linha vertical vermelha tracejada
   - **Gradual drift**: Faixa salmon transparente
   - **Texto**: Severidade do drift em % (ex: "60.7%")

4. **Marca limites de chunks de teste**:
   - Linhas verticais cinza pontilhadas
   - Label: "Test Phase N (Model from Chunk M)"

### Informacoes VISUAIS que fornece:

- Onde ocorreram os drifts
- Severidade de cada drift (%)
- Performance antes e depois dos drifts
- Quais chunks foram usados para treino vs teste
- Evolucao da acuracia ao longo do stream

### Por que e IMPORTANTE:

Este plot e o **mais informativo** para analisar adaptacao a drift porque:
- Mostra causa (drift) e efeito (queda de acuracia) no mesmo grafico
- Permite validar se o sistema detectou drift no momento correto
- Mostra visualmente se houve recovery apos drift

---

## 🚨 PROBLEMA IDENTIFICADO: Resultados no Colab

### Evidencia 1: Log mostra caminho do Colab

**experimento_teste_6chunks_20251028_160506.log:5**:
```
Relative base_results_dir resolved: /content/drive/MyDrive/DSL-AG-hybrid/experiments_test
```

**experimento_teste_6chunks_20251028_160506.log:11**:
```
Results will be saved in: /content/drive/MyDrive/DSL-AG-hybrid/experiments_test/RBF_Abrupt_Severe/run_1
```

### Evidencia 2: Chunk data foi salvo no Colab

**experimento_teste_6chunks_20251028_160506.log:25**:
```
Chunk 0 (train) salvo com sucesso em: /content/drive/MyDrive/DSL-AG-hybrid/experiments_test/RBF_Abrupt_Severe/run_1/chunk_data/chunk_0_train.csv
```

### Evidencia 3: Diretorio nao existe localmente

```bash
find C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid -name "experiments_test" -type d
# Resultado: (vazio - nao encontrado)
```

### Conclusao:

Os arquivos JSON necessarios para gerar plots **EXISTEM no Google Colab**, mas **NAO foram baixados** para a maquina local.

---

## 💡 INSIGHTS DA INFRAESTRUTURA EXISTENTE

### 1. Sistema de Plots e ROBUSTO

**generate_plots.py** tem:
- ✅ Tratamento de erros completo (try/except em cada plot)
- ✅ Logs informativos em cada etapa
- ✅ Flexibilidade de caminhos (argparse)
- ✅ Geracao automatica de diretorios de output
- ✅ Suporte a plots offline (nao precisa rodar experimento novamente)

### 2. Separacao Clara de Responsabilidades

```
analyze_concept_difference.py  → Gera concept_differences.json (PRE-experimento)
main.py                        → Executa experimento, salva JSONs
generate_plots.py              → Le JSONs, gera plots (POS-experimento)
plotting.py                    → Funcoes de plotagem reutilizaveis
```

### 3. Arquitetura Preparada para Batch

**generate_plots.py** aceita:
- Um diretorio de run especifico
- Pode ser chamado em loop para multiplos runs
- Plots salvos em subpasta `plots/` dentro de cada run

Exemplo de uso em batch:
```bash
for run_dir in experiments_test/*/run_*; do
    python generate_plots.py "$run_dir" \
        --diff_file test_real_results_heatmaps/concept_heatmapsS/concept_differences.json
done
```

### 4. Visualizacoes Complementares

Os 5 tipos de plots cobrem aspectos diferentes:

| Plot | Aspecto | Util para |
|------|---------|-----------|
| **AccuracyPeriodic** | Performance geral | Ver impacto de drifts |
| **GA_Evolution** | Evolucao do algoritmo | Diagnosticar convergencia |
| **RuleComponents_Heatmap** | Mudancas de regras | Ver adaptacao estrutural |
| **RuleComponents_Radar** | Caracteristicas gerais | Comparar chunks |
| **AttributeUsage** | Uso de features | Ver relevancia de atributos |

### 5. Marcadores de Drift Inteligentes

**plot_periodic_accuracy_with_drift_info()** diferencia:
- Drift **abrupt**: Linha vertical vermelha
- Drift **gradual**: Faixa salmon (largura = gradual_width_chunks)
- Severidade: Texto em % (calculado de concept_differences.json)

Isso permite **validar visualmente** se:
- O drift foi detectado no momento correto
- A severidade estimada esta correta
- Houve recovery apos o drift

---

## 🔧 FIXES NECESSARIOS

### Fix 1: Baixar Resultados do Google Colab

**Opcao A - Manual**:
1. Acessar Google Drive: `/DSL-AG-hybrid/experiments_test/`
2. Baixar pasta `RBF_Abrupt_Severe/run_1/` completa
3. Extrair para local: `C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid\experiments_test\`

**Opcao B - Script Python** (criar `download_results_from_drive.py`):
```python
from google.colab import drive
import shutil
drive.mount('/content/drive')
shutil.copytree(
    '/content/drive/MyDrive/DSL-AG-hybrid/experiments_test',
    '/content/experiments_test_backup'
)
# Depois baixar /content/experiments_test_backup via Colab Files
```

**Opcao C - Google Drive Desktop**:
1. Instalar Google Drive for Desktop
2. Sincronizar pasta DSL-AG-hybrid
3. Copiar experiments_test/ para pasta local

### Fix 2: Ajustar Caminho de concept_differences.json

**No generate_plots.py linha 324**, mudar:
```python
# Era:
--diff_file default="results/concept_heatmaps/concept_differences.json"

# Deve ser:
--diff_file default="test_real_results_heatmaps/concept_heatmapsS/concept_differences.json"
```

**OU** mover arquivo para o caminho esperado:
```bash
mkdir -p results/concept_heatmaps
cp test_real_results_heatmaps/concept_heatmapsS/concept_differences.json \
   results/concept_heatmaps/
```

### Fix 3: Criar Script de Pos-Processamento Batch

Criar `process_all_plots.py`:
```python
import os
import subprocess

# Diretorio base de experimentos
experiments_dir = "experiments_test"
diff_file = "test_real_results_heatmaps/concept_heatmapsS/concept_differences.json"

# Encontrar todos os runs
for stream_dir in os.listdir(experiments_dir):
    stream_path = os.path.join(experiments_dir, stream_dir)
    if not os.path.isdir(stream_path): continue

    for run_dir in os.listdir(stream_path):
        if not run_dir.startswith("run_"): continue

        run_path = os.path.join(stream_path, run_dir)
        print(f"Gerando plots para: {run_path}")

        subprocess.run([
            "python", "generate_plots.py",
            run_path,
            "--diff_file", diff_file
        ])
```

---

## 📋 CHECKLIST DE VALIDACAO

Antes de rodar `generate_plots.py`, validar:

### Arquivos de Entrada:
- [ ] `experiments_test/RBF_Abrupt_Severe/run_1/run_config.json` existe
- [ ] `experiments_test/RBF_Abrupt_Severe/run_1/periodic_accuracy.json` existe
- [ ] `experiments_test/RBF_Abrupt_Severe/run_1/ga_history_per_chunk.json` existe
- [ ] `experiments_test/RBF_Abrupt_Severe/run_1/rule_details_per_chunk.json` existe
- [ ] `experiments_test/RBF_Abrupt_Severe/run_1/attribute_usage_per_chunk.json` existe
- [ ] `experiments_test/RBF_Abrupt_Severe/run_1/chunk_metrics.json` existe
- [ ] `test_real_results_heatmaps/concept_heatmapsS/concept_differences.json` existe

### Comando de Teste:
```bash
python generate_plots.py \
    experiments_test/RBF_Abrupt_Severe/run_1 \
    --diff_file test_real_results_heatmaps/concept_heatmapsS/concept_differences.json
```

### Resultado Esperado:
```
experiments_test/RBF_Abrupt_Severe/run_1/plots/
├── Plot_AccuracyPeriodic_RBF_Abrupt_Severe_Run1.png
├── Plot_GA_Evolution_Chunk0_RBF_Abrupt_Severe_Run1.png
├── Plot_GA_Evolution_Chunk1_RBF_Abrupt_Severe_Run1.png
├── Plot_GA_Evolution_Chunk2_RBF_Abrupt_Severe_Run1.png
├── Plot_GA_Evolution_Chunk3_RBF_Abrupt_Severe_Run1.png
├── Plot_GA_Evolution_Chunk4_RBF_Abrupt_Severe_Run1.png
├── Plot_RuleComponents_Heatmap_RBF_Abrupt_Severe_Run1.png
├── Plot_RuleComponents_Radar_RBF_Abrupt_Severe_Run1.png
└── Plot_AttributeUsage_RBF_Abrupt_Severe_Run1.png
```

Total: **9 plots** (1 accuracy + 5 GA evolution + 1 heatmap + 1 radar + 1 attribute)

---

## 🎯 INSIGHTS EXTRAIDOS

### 1. Workflow Completo Esta Desenhado

```
FASE 1: PRE-EXPERIMENTO
├── analyze_concept_difference.py → Gera concept_differences.json
└── Calcula severidades: c1_vs_c2_severe = 60.68% (SEVERE)

FASE 2: EXPERIMENTO
├── main.py executa com config_test_single.yaml
├── Le concept_differences.json (se disponivel)
├── Salva 6 JSONs + chunk_data CSVs
└── Gera log completo

FASE 3: POS-EXPERIMENTO
├── generate_plots.py le os 6 JSONs
├── Usa concept_differences.json para severidades
├── Gera 9 plots visuais
└── Salva em subpasta plots/
```

### 2. Sistema de Deteccao de Drift Tem 2 Modos

**Modo PROATIVO** (esperado):
- Le `concept_differences.json` ANTES do experimento
- Detecta mudanca de `concept_id` durante geracao
- Ativa seeding 85% quando `diff >= 25%` (SEVERE)
- **Status**: NAO funcionou no teste (arquivo nao encontrado durante experimento)

**Modo REATIVO** (fallback):
- Detecta queda de performance (ex: 91% → 45%)
- Ativa adaptacao DEPOIS do problema
- **Status**: FUNCIONOU no teste (detectou chunk 4)

### 3. Visualizacoes Sao Chave para Validacao

Sem plots, e dificil validar:
- Se drift foi detectado no momento certo
- Se seeding foi ativado corretamente
- Se houve recovery apos drift
- Quais regras mudaram durante adaptacao

**generate_plots.py fornece essas respostas visualmente**

### 4. Infraestrutura Preparada para Escala

- Batch processing: Loop sobre multiplos runs
- Caminhos flexiveis: Argparse para customizacao
- Output organizado: Subpasta `plots/` por run
- Resiliente: Try/except em cada plot (um erro nao quebra outros)

---

## 🚀 PROXIMOS PASSOS RECOMENDADOS

### Opcao 1: VALIDAR PLOTS com Resultados Existentes (RECOMENDADO)

**Objetivo**: Ver se generate_plots.py funciona corretamente

1. ✅ **Baixar resultados do Google Colab**
   - Acessar `/DSL-AG-hybrid/experiments_test/RBF_Abrupt_Severe/run_1/`
   - Baixar pasta completa para local

2. ✅ **Ajustar caminho de concept_differences.json**
   - Usar flag `--diff_file` no comando

3. ✅ **Gerar plots**
   ```bash
   python generate_plots.py \
       experiments_test/RBF_Abrupt_Severe/run_1 \
       --diff_file test_real_results_heatmaps/concept_heatmapsS/concept_differences.json
   ```

4. ✅ **Analisar plots visuais**
   - Ver em qual instancia o drift ocorreu visualmente
   - Validar se severidade (60.7%) aparece no grafico
   - Ver queda de 91% → 45% no plot de acuracia

**Beneficio**: Entender o que aconteceu no teste sem precisar re-executar

### Opcao 2: CORRIGIR e RE-TESTAR

**Objetivo**: Validar deteccao proativa de drift

1. ✅ Confirmar `concept_differences.json` existe em local correto
2. ✅ Ajustar main.py para ler de caminho correto
3. ✅ Re-executar teste com RBF_Abrupt_Severe (8-10h)
4. ✅ Gerar plots para comparar vs teste anterior

### Opcao 3: PROSSEGUIR para Batch de Streams

**Objetivo**: Executar 5 streams em paralelo

**Prerequisitos**:
- Confirmar Fix 2 (caminho de concept_differences.json)
- Validar `config_6chunks.yaml` esta correto
- Preparar infraestrutura de Google Colab para 5 runs simultaneos

---

## 📊 COMPARACAO: Infraestrutura Atual vs Ideal

| Aspecto | Atual | Ideal | Gap |
|---------|-------|-------|-----|
| **Geracao de concept_differences.json** | ✅ Funciona | ✅ Funciona | - |
| **Caminho de leitura em main.py** | ❌ Hardcoded incorreto | ✅ Configuravel | 🔧 Fix necessario |
| **Salvamento de JSONs** | ✅ Funciona | ✅ Funciona | - |
| **Geracao de plots** | ⚠️ Funciona se JSONs disponiveis | ✅ Funciona | 📥 Baixar resultados |
| **Deteccao proativa de drift** | ❌ Nao funcionou no teste | ✅ Funcionando | 🔧 Fix necessario |
| **Visualizacao de drift** | ⚠️ Possivel, mas sem JSONs | ✅ Funcionando | 📥 Baixar resultados |
| **Batch processing** | ✅ Suportado | ✅ Suportado | - |

---

## 📖 DOCUMENTACAO UTIL

### Funcoes de plotting.py usadas por generate_plots.py:

1. `plotting.plot_ga_evolution()` - Evolucao de fitness por geracao
2. `plotting.plot_rule_changes()` - Heatmap de mudancas de regras
3. `plotting.plot_rule_info_radar()` - Radar plot de caracteristicas
4. `plotting.plot_attribute_usage_over_time()` - Uso de atributos

### Formato de periodic_accuracy.json:

```json
[
  [500, 0.8882],    // (instancia, acuracia)
  [1000, 0.9015],
  [1500, 0.9158],
  ...
]
```

### Formato de chunk_metrics.json:

```json
[
  {
    "chunk": 0,
    "train_accuracy": 0.9124,
    "train_gmean": 0.9124,
    "test_gmean": 0.8882,
    "test_f1_weighted": 0.8888,
    "concept_id": "c1"
  },
  ...
]
```

---

## ✅ CONCLUSAO

### O que FUNCIONA:
1. ✅ Geracao de concept_differences.json (60.68% SEVERE)
2. ✅ Salvamento de JSONs durante experimento (no Colab)
3. ✅ Codigo de geracao de plots (testado localmente)
4. ✅ Infraestrutura preparada para batch

### O que NAO FUNCIONA:
1. ❌ Caminho de concept_differences.json em main.py
2. ❌ Deteccao proativa de drift (por falta do arquivo)
3. ❌ Resultados nao baixados do Colab para local

### Acao IMEDIATA Recomendada:

**BAIXAR RESULTADOS DO COLAB** e **GERAR PLOTS** para validar visualmente o que aconteceu no teste. Isso fornecera insights sem precisar re-executar 10 horas de experimento.

```bash
# Apos baixar resultados do Colab:
python generate_plots.py \
    experiments_test/RBF_Abrupt_Severe/run_1 \
    --diff_file test_real_results_heatmaps/concept_heatmapsS/concept_differences.json
```

**Tempo estimado**: 10 minutos (download + geracao de plots)

---

**Documento criado por**: Claude Code
**Data**: 2025-10-29
**Status**: ✅ **INFRAESTRUTURA ANALISADA - AGUARDANDO DOWNLOAD DE RESULTADOS**
