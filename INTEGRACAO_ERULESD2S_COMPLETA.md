# Integração Completa do ERulesD2S

**Data:** 2025-11-12
**Status:** ✅ Infraestrutura Pronta para Teste
**Opção:** B (Integração Completa)

---

## 📦 Arquivos Criados

### Módulos de Integração

1. **arff_converter.py** - Conversor de dados Python → ARFF
   - Converte numpy/pandas para formato ARFF (MOA)
   - Suporta chunks individuais ou streams completos
   - Validação de arquivos ARFF

2. **erulesd2s_wrapper.py** - Wrapper Python → Java/MOA
   - Executa ERulesD2S via subprocess
   - Configuração de parâmetros (população, gerações, etc.)
   - Parsing de resultados MOA
   - Timeout e tratamento de erros

3. **setup_erulesd2s.py** - Script de instalação
   - Clona repositório ERulesD2S do GitHub
   - Compila com Maven
   - Cria links simbólicos e configurações

4. **test_erulesd2s_integration.py** - Teste unitário
   - Teste rápido com dados sintéticos
   - Valida conversão ARFF + execução + parsing

5. **test_erulesd2s_local_validation.py** - Teste completo
   - Testa com RBF_Abrupt_Severe (dataset validado)
   - Compara 6 modelos: GBML, ACDWM, HAT, ARF, SRP, ERulesD2S
   - Gera relatório consolidado

---

## 🚀 Guia de Execução Passo a Passo

### Passo 1: Setup do ERulesD2S (15-20 min)

```bash
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"

# Instalar ERulesD2S
python setup_erulesd2s.py
```

**O que faz:**
- Verifica Java 11+ e Maven
- Clona https://github.com/canoalberto/ERulesD2S
- Compila projeto (mvn clean compile && mvn package)
- Cria `erulesd2s.jar` (link para JAR compilado)
- Gera `erulesd2s_config.env`

**Pré-requisitos (Windows):**
```powershell
# Instalar Java 11+ (se não tiver)
winget install EclipseAdoptium.Temurin.11.JDK

# Instalar Maven (se não tiver)
winget install Apache.Maven

# Verificar
java -version
mvn --version
```

**Pré-requisitos (Linux/Colab):**
```bash
# Instalar Java e Maven
apt-get update
apt-get install -y openjdk-11-jdk-headless maven

# Verificar
java -version
mvn --version
```

---

### Passo 2: Teste Unitário (2-3 min)

```bash
python test_erulesd2s_integration.py
```

**O que faz:**
- Gera 1000 instâncias de teste (2 classes)
- Converte para ARFF
- Executa ERulesD2S (pop=10, gen=20)
- Valida resultados

**Output esperado:**
```
===========================================
TESTE DE INTEGRACAO ERULESD2S
==========================================

[INFO] Gerando dados de teste: 1000 samples, 5 features
[INFO] Arquivo ARFF criado: test_erulesd2s_output/test_data.arff
[INFO] ARFF validado com sucesso
[INFO] JAR encontrado: erulesd2s.jar
[INFO] Wrapper ERulesD2S criado
[INFO] Executando ERulesD2S...
[INFO] ERulesD2S executado com sucesso!
[INFO] Accuracy: 0.8542

Status: SUCESSO
```

**Se falhar:**
- Verificar logs em `test_erulesd2s_output/erulesd2s_run/erulesd2s_log.txt`
- Verificar se Java consegue executar: `java -jar erulesd2s.jar`
- Verificar classpath e bibliotecas MOA

---

### Passo 3: Teste Local Completo (45-60 min)

```bash
python test_erulesd2s_local_validation.py
```

**O que faz:**
1. **Fase 1:** Executa modelos Python (GBML, ACDWM, HAT, ARF, SRP)
   - Reutiliza resultados se já existirem
   - Ou executa `compare_gbml_vs_river.py` com RBF_Abrupt_Severe

2. **Fase 2:** Executa ERulesD2S
   - Converte chunks para ARFF
   - Executa ERulesD2S (pop=25, gen=50)
   - 3 chunks × ~4 min = 12 min

3. **Fase 3:** Consolida resultados
   - Merge Python + ERulesD2S
   - Gera ranking por G-mean
   - Salva `validation_local_results_with_erulesd2s.csv`

**Output esperado:**
```
===============================================
RESUMO DOS RESULTADOS
===============================================

G-mean medio por modelo:
------------------------------------------------------------
         mean    std    min    max
model
SRP     0.7223  0.2567  0.5408  0.9039
ARF     0.7030  0.2961  0.4936  0.9123
ACDWM   0.6872  0.3035  0.4726  0.9018
GBML    0.6800  0.2653  0.4923  0.8676
ERulesD2S 0.6950  0.2820  0.4850  0.9100
HAT     0.6658  0.1867  0.5337  0.7978

Ranking por G-mean medio:
  1. SRP           : 0.7223
  2. ARF           : 0.7030
  3. ERulesD2S     : 0.6950
  4. ACDWM         : 0.6872
  5. GBML          : 0.6800
  6. HAT           : 0.6658
```

---

## ⏱️ Projeção de Tempo para Experimento Completo

### Com 6 Modelos (incluindo ERulesD2S)

| Componente | Cálculo | Tempo |
|------------|---------|-------|
| **GBML** | 44 chunks × 15 min | 660 min (11h) |
| **ACDWM** | 44 chunks × 1 min | 44 min |
| **HAT, ARF, SRP** | 44 chunks × 1 min × 3 | 132 min |
| **ERulesD2S** | 44 chunks × 4 min | 176 min (3h) |
| **Total** | | **14 horas** |
| **Margem Colab Pro** | 24h - 14h | **10h (42%)** ✅ |

### Tempo por Dataset

Cada dataset (4 chunks):
- GBML: 4 × 15 min = 60 min
- Python models: 4 × 4 min = 16 min
- ERulesD2S: 4 × 4 min = 16 min
- **Total: ~1h 30min por dataset**

11 datasets × 1.5h = **16.5 horas**

---

## 🎯 Decisão: Executar Teste Local Primeiro

### Próximos Passos

1. ✅ **AGORA:** Executar teste local completo
   ```bash
   python setup_erulesd2s.py           # 15-20 min
   python test_erulesd2s_integration.py  # 2-3 min
   python test_erulesd2s_local_validation.py  # 45-60 min
   ```

2. ✅ **Analisar resultados locais:**
   - ERulesD2S está competitivo?
   - Tempo real por chunk?
   - Qualidade das métricas?

3. ✅ **Se teste local bem-sucedido:**
   - Integrar ERulesD2S no `compare_gbml_vs_river.py`
   - Executar experimento completo no Colab (11 datasets)

---

## 📊 Comparação: ERulesD2S vs Outros Modelos

### Características

| Modelo | Tecnologia | Explicabilidade | Tempo/Chunk | Parâmetros |
|--------|-----------|----------------|-------------|------------|
| **GBML** | Python/GA | ✅✅✅ Regras | ~15 min | pop=100, gen=200 |
| **ACDWM** | Python/Ensemble | ⚠️ Médio | ~1 min | Dynamic ensemble |
| **HAT** | Python/River | ❌ Baixo | ~1 min | Tree |
| **ARF** | Python/River | ❌ Baixo | ~1 min | Random forest |
| **SRP** | Python/River | ❌ Baixo | ~1 min | Tree |
| **ERulesD2S** | Java/GP/GPU | ✅✅ Regras | ~4 min | pop=25, gen=50 |

### Vantagens do ERulesD2S

1. **Explicabilidade:** Regras interpretáveis (similar ao GBML)
2. **GPU:** Paralelização em GPU (mais rápido que GBML)
3. **Compacidade:** ~31 regras (vs 252 VFDR, 732 G-eRules)
4. **Paper:** Publicado em Pattern Recognition (2019)
5. **Benchmark:** 82.14% accuracy média em 5 datasets

### Desvantagens

1. **Tecnologia diferente:** Java vs Python
2. **Integração complexa:** Conversão ARFF, subprocess
3. **Dependências:** Java, Maven, MOA
4. **Menos testado:** No pipeline atual

---

## 🔧 Troubleshooting

### Problema 1: Maven build falha

**Erro:**
```
[ERROR] Failed to execute goal
```

**Solução:**
```bash
cd ERulesD2S
rm -rf ~/.m2/repository
mvn clean install -U -DskipTests
```

### Problema 2: Java OutOfMemoryError

**Erro:**
```
java.lang.OutOfMemoryError: Java heap space
```

**Solução:**
- Aumentar memória: `-Xmx8g` → `-Xmx12g`
- Ou reduzir parâmetros: `population_size=25` → `15`

### Problema 3: ARFF conversion falha

**Erro:**
```
ValueError: Cannot convert data to ARFF
```

**Solução:**
- Verificar tipos de dados (tudo deve ser numérico)
- Verificar NaN/Inf values
- Validar shape: `X.shape[0] == y.shape[0]`

### Problema 4: ERulesD2S timeout

**Erro:**
```
subprocess.TimeoutExpired
```

**Solução:**
- Aumentar timeout: `timeout=120` → `timeout=300`
- Reduzir gerações: `num_generations=50` → `30`
- Verificar se processo travou (kill manual)

### Problema 5: Resultados parsing falha

**Erro:**
```
No accuracy found in output
```

**Solução:**
- Verificar logs em `erulesd2s_log.txt`
- MOA pode ter formato de output diferente
- Ajustar regex patterns em `parse_results()`

---

## 📁 Estrutura de Arquivos Após Testes

```
DSL-AG-hybrid/
├── arff_converter.py                  # Módulo conversão
├── erulesd2s_wrapper.py              # Módulo wrapper
├── setup_erulesd2s.py                # Script instalação
├── test_erulesd2s_integration.py     # Teste unitário
├── test_erulesd2s_local_validation.py # Teste completo
│
├── ERulesD2S/                         # Repositório clonado
│   ├── src/
│   ├── target/
│   │   └── erulesd2s-1.0-SNAPSHOT.jar
│   └── pom.xml
│
├── erulesd2s.jar                      # Link simbólico
├── erulesd2s_config.env              # Configurações
│
├── test_erulesd2s_output/            # Teste unitário
│   ├── test_data.arff
│   └── erulesd2s_run/
│       ├── erulesd2s_results.csv
│       └── erulesd2s_log.txt
│
├── arff_chunks/                       # Chunks convertidos
│   ├── rbf_chunk_0.arff
│   ├── rbf_chunk_1.arff
│   └── rbf_chunk_2.arff
│
├── validation_local_results/          # Resultados Python
│   └── RBF_Abrupt_Severe_seed42_*/
│       └── comparison_table.csv
│
├── validation_local_results_erulesd2s/ # Resultados ERulesD2S
│   ├── chunk_0/
│   ├── chunk_1/
│   ├── chunk_2/
│   └── erulesd2s_results.csv
│
└── validation_local_results_with_erulesd2s.csv # Consolidado
```

---

## ✅ Checklist de Execução

### Antes do Teste Local

- [ ] Java 11+ instalado (`java -version`)
- [ ] Maven instalado (`mvn --version`)
- [ ] Git instalado
- [ ] Python 3.8+ com dependências (river, numpy, pandas, etc.)
- [ ] Espaço em disco: ~2GB para ERulesD2S

### Durante o Teste Local

- [ ] `setup_erulesd2s.py` executado com sucesso
- [ ] `erulesd2s.jar` criado
- [ ] `test_erulesd2s_integration.py` passou
- [ ] `test_erulesd2s_local_validation.py` em execução

### Após o Teste Local

- [ ] Arquivo `validation_local_results_with_erulesd2s.csv` criado
- [ ] 6 modelos com resultados (incluindo ERulesD2S)
- [ ] G-mean de ERulesD2S entre 0.5-0.9
- [ ] Tempo por chunk ERulesD2S: 3-5 min

---

## 🎓 Próxima Fase: Experimento Completo no Colab

Se teste local for bem-sucedido, próximos passos:

1. **Adaptar para Colab:**
   - Instalar Java, Maven no Colab
   - Executar `setup_erulesd2s.py` no início do notebook
   - Modificar `compare_gbml_vs_river.py` para incluir ERulesD2S

2. **Executar 11 datasets:**
   - Usar `config_experiment_expanded.yaml`
   - 6 modelos × 44 chunks = **14 horas**
   - Monitorar progresso

3. **Análise final:**
   - 33 avaliações por modelo (vs 8 anteriores)
   - Comparação completa incluindo ERulesD2S
   - Análise estatística expandida

---

**Status:** ✅ Pronto para Teste Local

**Comando para iniciar:**
```bash
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"
python setup_erulesd2s.py
python test_erulesd2s_local_validation.py
```

**Tempo estimado total:** ~1h 15min (setup + testes)
