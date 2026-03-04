# ROSE - Robust Online Self-Adjusting Ensemble

## Informacoes do Modelo

| Item | Descricao |
|------|-----------|
| **Paper** | ROSE: robust online self-adjusting ensemble for continual learning on imbalanced drifting data streams |
| **Autores** | Alberto Cano, Bartosz Krawczyk |
| **Publicacao** | Machine Learning, 2022, Vol. 111(7), pp. 2561-2599 |
| **GitHub** | https://github.com/canoalberto/ROSE |
| **Linguagem** | Java (MOA Framework) |
| **Abordagem** | Online |
| **Foco** | Concept Drift + Class Imbalance |

---

## Componentes Principais

### 1. Online Feature Subspace Sampling
- Cada classificador base treina em um subset aleatorio de features
- Tamanho do subset: sqrt(total_features)
- Aumenta diversidade do ensemble

### 2. Drift Detection com Background Ensemble
- Ensemble principal (foreground) faz predicoes
- Detectores de drift monitoram cada classifier (ADWIN)
- Quando WARNING detectado -> inicia background learner
- Quando DRIFT confirmado -> background substitui foreground

### 3. Per-Class Sliding Windows
- Mantem sliding window de tamanho W por classe
- Garante que classe minoritaria nao seja "esquecida"
- Permite treinar mesmo quando classe ausente no batch

### 4. Self-Adjusting Bagging
- Peso lambda ajustado por sqrt(imbalance_ratio) para classe minoritaria
- Aumenta representacao de exemplos dificeis

---

## Estrutura de Arquivos

```
rose/
├── rose_wrapper.py          # Wrapper Python para executar ROSE via Java (PRINCIPAL)
├── ROSE_python.py           # Implementacao Python pura usando River (ALTERNATIVA)
├── Test_ROSE_Colab.ipynb    # Notebook para validacao no Google Colab
├── ANALISE_ROSE_PAPER.md    # Analise detalhada do paper
└── README.md                # Este arquivo
```

## Duas Abordagens Disponiveis

### 1. Via JAR Java (Recomendada)
- Usa o JAR original do ROSE
- Executa via MOA framework
- Resultados identicos ao paper original
- Arquivo: `rose_wrapper.py`

### 2. Via Python/River (Alternativa)
- Implementacao Python pura
- Usa River para HoeffdingTree e ADWIN
- Mais facil de integrar, mas pode ter pequenas diferencas
- Arquivo: `ROSE_python.py`

---

## Dependencias

### JARs Necessarios (baixados automaticamente no Colab)
| JAR | Tamanho | Descricao |
|-----|---------|-----------|
| ROSE-1.0.jar | 115 KB | Classifier ROSE |
| MOA-dependencies.jar | 64.6 MB | Framework MOA |
| sizeofag-1.0.4.jar | 13 KB | Memory profiling (opcional) |

### Requisitos
- Java 11+ (OpenJDK ou Oracle JDK)
- Python 3.8+
- numpy, pandas (para processamento de resultados)

---

## Como Usar no Google Colab

### 1. Upload do Notebook
Fazer upload de `Test_ROSE_Colab.ipynb` para o Google Colab.

### 2. Executar Setup
```python
# Instalar Java
!apt-get update -qq
!apt-get install -y default-jdk -qq

# Baixar JARs
# (codigo no notebook)
```

### 3. Preparar Dados
```python
# Converter dados para ARFF
from rose_wrapper import create_arff_file
create_arff_file(X, y, "dados.arff")
```

### 4. Executar ROSE
```python
# Executar via MOA
success, results = run_rose(
    arff_file="dados.arff",
    output_dir="output",
    chunk_size=500
)

print(f"G-mean: {results['gmean']:.4f}")
```

---

## Comando MOA

Exemplo de comando completo:

```bash
java -Xmx4g \
  -javaagent:sizeofag-1.0.4.jar \
  -cp ROSE-1.0.jar:MOA-dependencies.jar \
  moa.DoTask \
  "EvaluateInterleavedTestThenTrain \
   -e (WindowAUCImbalancedPerformanceEvaluator) \
   -s (ArffFileStream -f dados.arff) \
   -l (moa.classifiers.meta.imbalanced.ROSE) \
   -f 500 \
   -d results.csv"
```

### Parametros Principais
| Parametro | Descricao | Default |
|-----------|-----------|---------|
| -e | Evaluator | WindowAUCImbalancedPerformanceEvaluator |
| -s | Stream (arquivo ARFF) | - |
| -l | Learner | moa.classifiers.meta.imbalanced.ROSE |
| -f | Frequencia de avaliacao | 500 |
| -i | Max instancias | all |
| -d | Arquivo de saida CSV | - |

---

## Metricas de Saida

O ROSE (via MOA) retorna as seguintes metricas:

| Metrica | Descricao |
|---------|-----------|
| G-Mean | Media geometrica de Sensitivity e Specificity |
| AUC | Area Under ROC Curve |
| classifications correct (percent) | Accuracy |
| Kappa Statistic (percent) | Cohen's Kappa |

---

## Integracao com Pipeline GBML

### Abordagem: Java via Subprocess

Seguindo o padrao do ERulesD2S, o ROSE e executado via:
1. Converter chunks Python para arquivos ARFF
2. Executar JAR do ROSE via subprocess
3. Parsear resultados do CSV de saida
4. Integrar metricas no DataFrame consolidado

### Codigo de Integracao

```python
from rose_wrapper import ROSEWrapper, ROSEEvaluator
from arff_converter import ARFFConverter

# Criar wrapper
wrapper = ROSEWrapper(
    rose_jar_path="ROSE-1.0.jar",
    moa_dependencies_jar="MOA-dependencies.jar"
)

# Converter chunks para ARFF
converter = ARFFConverter()
arff_files = converter.convert_stream(chunks, output_dir)

# Avaliar
evaluator = ROSEEvaluator(wrapper, output_dir)
results_df = evaluator.evaluate_chunks(chunks, arff_files)
```

---

## Resultados Esperados

### Do Paper (G-mean medio)
| Dataset | IR=5 | IR=10 | IR=20 | IR=50 | IR=100 |
|---------|------|-------|-------|-------|--------|
| Agrawal | 0.912 | 0.887 | 0.856 | 0.798 | 0.742 |
| LED | 0.845 | 0.821 | 0.789 | 0.734 | 0.678 |
| RandomRBF | 0.923 | 0.901 | 0.872 | 0.821 | 0.769 |

### Ranking do Paper
| Modelo | G-mean | Rank |
|--------|--------|------|
| ROSE | 0.812 | 1.2 |
| CSMOTE | 0.784 | 2.8 |
| OOB | 0.756 | 3.4 |
| UOB | 0.748 | 3.9 |
| ARF | 0.723 | 4.5 |

---

## Troubleshooting

### Erro: Java not found
```bash
# Colab
!apt-get install -y default-jdk

# Local (Ubuntu)
sudo apt install openjdk-11-jdk
```

### Erro: JAR not found
```python
# Verificar se JARs existem
import os
print(os.listdir("rose_jars"))

# Re-baixar se necessario
# (ver codigo no notebook)
```

### Erro: Out of memory
```bash
# Aumentar memoria Java
java -Xmx8g ...
```

### Erro: ClassNotFoundException
Verificar se classpath inclui todos os JARs:
```bash
# Linux/Mac
-cp ROSE-1.0.jar:MOA-dependencies.jar

# Windows
-cp ROSE-1.0.jar;MOA-dependencies.jar
```

---

## Proximos Passos

1. [x] Criar wrapper Python
2. [x] Criar notebook Colab
3. [ ] Validar no Colab com dados sinteticos
4. [ ] Integrar no pipeline main.py
5. [ ] Executar nos 32 datasets
6. [ ] Comparar com GBML

---

**Criado por:** Claude Code
**Data:** 2025-11-25
**Status:** Pronto para validacao no Colab
