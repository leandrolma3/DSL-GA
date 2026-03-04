# Correção dos Configs Batch 6 e Batch 7

**Data:** 2025-11-21
**Problema:** ImportError ao executar batches 6 e 7
**Status:** CORRIGIDO

---

## Problema Identificado

### Erro Observado
```
ImportError: Could not import class 'None'.

Traceback:
  File "data_handling.py", line 81, in get_class_from_string
    module_path, class_name = class_path.rsplit('.', 1)
AttributeError: 'NoneType' object has no attribute 'rsplit'
```

### Causa Raiz
Os configs dos **batches 6 e 7** estavam usando formato **incorreto** na seção `drift_analysis.datasets`:

**FORMATO INCORRETO (usado inicialmente):**
```yaml
drift_analysis:
  datasets:
    SEA_STATIONARY:
      generator: SEAGenerator      # ❌ ERRADO
      n_classes: 2
```

**FORMATO CORRETO (necessário):**
```yaml
drift_analysis:
  datasets:
    SEA_STATIONARY:
      class: river.datasets.synth.SEA  # ✅ CORRETO
      n_features: 3
      feature_bounds:
      - [0, 10]
```

### Por Que Ocorreu
- Os configs foram criados com base em suposição de formato
- Não foi verificado contra os configs da Fase 2 (batches 1-4)
- O campo `generator:` não existe no código; deve ser `class:`
- O código espera o caminho completo da classe River (e.g., `river.datasets.synth.SEA`)

---

## Correções Aplicadas

### config_batch_6.yaml

**Datasets corrigidos:**
1. **SEA_STATIONARY**
   - ❌ `generator: SEAGenerator`
   - ✅ `class: river.datasets.synth.SEA`
   - Adicionado: `n_features: 3`, `feature_bounds: [[0, 10]]`

2. **AGRAWAL_STATIONARY**
   - ❌ `generator: AGRAWALGenerator`
   - ✅ `class: river.datasets.synth.Agrawal`
   - Adicionado: `n_features: 9`, `feature_bounds: [[0, 1]]`

3. **RBF_STATIONARY**
   - ❌ `generator: RandomRBFGenerator`
   - ✅ `class: river.datasets.synth.RandomRBF`
   - Mantido: `n_classes: 5`
   - Adicionado: `n_features: 10`, `feature_bounds: [[0, 1]]`

4. **LED_STATIONARY**
   - ❌ `generator: LEDGenerator`
   - ✅ `class: river.datasets.synth.LED`
   - Adicionado: `n_features: 24`, `feature_bounds: [[0, 1]]`

5. **HYPERPLANE_STATIONARY**
   - ❌ `generator: HyperplaneGenerator`
   - ✅ `class: river.datasets.synth.Hyperplane`
   - Mantido: `n_classes: 2`
   - Adicionado: `n_features: 10`, `feature_bounds: [[0, 1]]`

6. **RANDOMTREE_STATIONARY**
   - ❌ `generator: RandomTreeGenerator`
   - ✅ `class: river.datasets.synth.RandomTree`
   - Corrigido: `n_classes: 2` (era 5)
   - Adicionado: `n_features: 10`, `feature_bounds: [[0, 1]]`

**Novo tamanho:** 3.9 KB (era 3.5 KB)

---

### config_batch_7.yaml

**Datasets corrigidos:**
1. **STAGGER_STATIONARY**
   - ❌ `generator: STAGGERGenerator`
   - ✅ `class: river.datasets.synth.STAGGER`
   - Adicionado: `n_features: 3`, `feature_bounds: [[1, 3]]` (3 features)

2. **WAVEFORM_STATIONARY**
   - ❌ `generator: WaveformGenerator`
   - ✅ `class: river.datasets.synth.Waveform`
   - Adicionado: `n_features: 40`, `feature_bounds: [[0, 6]]`

3. **SINE_STATIONARY**
   - ❌ `generator: SineGenerator`
   - ✅ `class: river.datasets.synth.Sine`
   - Adicionado: `n_features: 5`, `feature_bounds: [[0, 6.28]]` (5 features)

4. **ASSETNEGOTIATION_F2**
   - ❌ `generator: AssetNegotiation`
   - ✅ `class: custom_generators.AssetNegotiation`
   - Reorganizado: `params:` com `seed: 42`, `classification_function: 1`, `balance: false`

5. **ASSETNEGOTIATION_F3**
   - ❌ `generator: AssetNegotiation`
   - ✅ `class: custom_generators.AssetNegotiation`
   - Reorganizado: `params:` com `classification_function: 2`

6. **ASSETNEGOTIATION_F4**
   - ❌ `generator: AssetNegotiation`
   - ✅ `class: custom_generators.AssetNegotiation`
   - Reorganizado: `params:` com `classification_function: 3`

**Novo tamanho:** 4.0 KB (era 3.7 KB)

---

## Referência: Formato Correto para Generators River

### Generators Sintéticos River (Stationary)

```yaml
drift_analysis:
  datasets:
    # SEA Generator
    SEA_STATIONARY:
      class: river.datasets.synth.SEA
      n_features: 3
      feature_bounds:
      - [0, 10]
      - [0, 10]
      - [0, 10]

    # AGRAWAL Generator
    AGRAWAL_STATIONARY:
      class: river.datasets.synth.Agrawal
      n_features: 9
      feature_bounds:
      - [0, 1]

    # RBF Generator
    RBF_STATIONARY:
      class: river.datasets.synth.RandomRBF
      n_features: 10
      n_classes: 5
      feature_bounds:
      - [0, 1]

    # LED Generator
    LED_STATIONARY:
      class: river.datasets.synth.LED
      n_features: 24
      feature_bounds:
      - [0, 1]

    # HYPERPLANE Generator
    HYPERPLANE_STATIONARY:
      class: river.datasets.synth.Hyperplane
      n_features: 10
      n_classes: 2
      feature_bounds:
      - [0, 1]

    # RANDOMTREE Generator
    RANDOMTREE_STATIONARY:
      class: river.datasets.synth.RandomTree
      n_features: 10
      n_classes: 2
      feature_bounds:
      - [0, 1]

    # STAGGER Generator
    STAGGER_STATIONARY:
      class: river.datasets.synth.STAGGER
      n_features: 3
      feature_bounds:
      - [1, 3]
      - [1, 3]
      - [1, 3]

    # WAVEFORM Generator
    WAVEFORM_STATIONARY:
      class: river.datasets.synth.Waveform
      n_features: 40
      feature_bounds:
      - [0, 6]

    # SINE Generator
    SINE_STATIONARY:
      class: river.datasets.synth.Sine
      n_features: 5
      feature_bounds:
      - [0, 6.28]
      - [0, 6.28]
      - [0, 6.28]
      - [0, 6.28]
      - [0, 6.28]
```

### Generator Customizado (AssetNegotiation)

```yaml
    ASSETNEGOTIATION_F2:
      class: custom_generators.AssetNegotiation
      params:
        seed: 42
        classification_function: 1  # F2
        balance: false

    ASSETNEGOTIATION_F3:
      class: custom_generators.AssetNegotiation
      params:
        seed: 42
        classification_function: 2  # F3
        balance: false

    ASSETNEGOTIATION_F4:
      class: custom_generators.AssetNegotiation
      params:
        seed: 42
        classification_function: 3  # F4
        balance: false
```

---

## Verificação Pós-Correção

### Comandos para Testar

**Batch 6:**
```bash
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"
python main.py --config configs/config_batch_6.yaml
```

**Batch 7:**
```bash
python main.py --config configs/config_batch_7.yaml
```

### O Que Observar
- ✅ Carregamento correto dos datasets sintéticos
- ✅ Criação de 6 chunks por dataset
- ✅ Treino no chunk 0
- ✅ 5 avaliações (chunks 1-5)
- ✅ Nenhum erro de import
- ✅ Métricas salvas corretamente

---

## Comparação com Batch 5 (Funcionando)

**Batch 5** não teve esse problema porque usa datasets **reais** (CSV ou River datasets), não generators sintéticos:

```yaml
# Batch 5 - FUNCIONA (datasets reais)
datasets:
  ELECTRICITY:
    class: river.datasets.Elec2  # Dataset real do River
  SHUTTLE:
    loader: local_csv            # CSV local
    source_path: datasets/processed/...
```

**Batches 6 e 7** usam generators sintéticos, que requerem formato específico:
```yaml
# Batches 6 e 7 - CORRIGIDO (generators sintéticos)
datasets:
  SEA_STATIONARY:
    class: river.datasets.synth.SEA  # Generator sintético
    n_features: 3
    feature_bounds: [[0, 10]]
```

---

## Lições Aprendidas

### 1. Sempre Verificar Configs de Referência
Antes de criar novos configs, sempre verificar configs existentes que funcionam (batches 1-4).

### 2. Campo `class:` é Obrigatório
O código `data_handling.py` espera o campo `class:` com o caminho completo da classe.

### 3. Generators Sintéticos Requerem Parâmetros
Generators sintéticos precisam de:
- `class:` com path completo
- `n_features:` número de features
- `feature_bounds:` limites de cada feature
- `n_classes:` (opcional) número de classes

### 4. Testar com 1 Dataset Primeiro
Sempre testar com 1 dataset antes de executar batch completo.

---

## Checklist de Verificação para Novos Configs

Ao criar novos configs, verificar:

- [ ] Campo `class:` presente (não `generator:`)
- [ ] Caminho completo da classe especificado
- [ ] Para River: `river.datasets.synth.CLASSNAME`
- [ ] Para custom: `custom_generators.CLASSNAME`
- [ ] `n_features:` especificado
- [ ] `feature_bounds:` especificado (lista)
- [ ] `n_classes:` especificado (se multiclass)
- [ ] `params:` usado para parâmetros customizados (AssetNegotiation)
- [ ] Testar com 1 dataset antes de rodar batch completo

---

## Status Final

### config_batch_6.yaml ✅
- **Status:** CORRIGIDO
- **Tamanho:** 3.9 KB
- **Datasets:** 6 sintéticos estacionários (parte 1)
- **Pronto para:** Execução

### config_batch_7.yaml ✅
- **Status:** CORRIGIDO
- **Tamanho:** 4.0 KB
- **Datasets:** 6 sintéticos estacionários (parte 2)
- **Pronto para:** Execução

### Próximos Passos
1. Executar Batch 6: `python main.py --config configs/config_batch_6.yaml`
2. Executar Batch 7: `python main.py --config configs/config_batch_7.yaml`
3. Verificar resultados em `experiments_6chunks_phase3_real/batch_6/` e `batch_7/`

---

---

## Segunda Correção - AssetNegotiation (2025-11-22)

### Problema Adicional Descoberto

**Erro observado após primeira correção:**
```
TypeError: AssetNegotiation.__init__() got an unexpected keyword argument 'balance'
```

### Causa
Os três datasets **AssetNegotiation** (F2, F3, F4) incluíam parâmetro `balance: false` que **não existe** na classe.

**Assinatura correta da classe:**
```python
def __init__(self, classification_function: int = 1, seed: int = None):
```

**Parâmetros aceitos:**
- ✅ `classification_function: int` (1, 2 ou 3)
- ✅ `seed: int`
- ❌ `balance` (NÃO EXISTE)

### Correção Aplicada

**❌ INCORRETO (causava erro):**
```yaml
ASSETNEGOTIATION_F2:
  class: custom_generators.AssetNegotiation
  params:
    seed: 42
    classification_function: 1
    balance: false  # ❌ Parâmetro inexistente!
```

**✅ CORRETO (aplicado):**
```yaml
ASSETNEGOTIATION_F2:
  class: custom_generators.AssetNegotiation
  params:
    seed: 42
    classification_function: 1  # Sem 'balance'
```

### Datasets Corrigidos
1. ✅ ASSETNEGOTIATION_F2 - Removido `balance: false`
2. ✅ ASSETNEGOTIATION_F3 - Removido `balance: false`
3. ✅ ASSETNEGOTIATION_F4 - Removido `balance: false`

### Status Pós-Correção
- **Batch 5:** ✅ Executado com sucesso (5/5 datasets)
- **Batch 6:** ✅ Executado com sucesso (6/6 datasets)
- **Batch 7:** ✅ CORRIGIDO - Pronto para re-executar os 3 AssetNegotiation

---

**Primeira Correção por:** Claude Code
**Data:** 2025-11-21
**Tempo de Diagnóstico:** ~5 minutos
**Tempo de Correção:** ~10 minutos

**Segunda Correção por:** Claude Code
**Data:** 2025-11-22
**Tempo de Diagnóstico:** ~2 minutos
**Tempo de Correção:** ~5 minutos
