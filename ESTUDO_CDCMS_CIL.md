# Estudo: CDCMS.CIL - Concept Drift Handling Based on Clustering in the Model Space for Class-Imbalanced Learning

**Data:** 2026-01-23
**Repositorio:** https://github.com/michaelchiucw/CDCMS.CIL
**Paper:** [IEEE DSAA 2025](https://ieeexplore.ieee.org/document/11247989/)

---

## 1. VISAO GERAL

### 1.1 O que e CDCMS.CIL?

CDCMS.CIL (Concept Drift handling based on Clustering in the Model Space for Class-Imbalanced Learning) e um framework de ensemble learning para data streams que aborda simultaneamente:

1. **Concept Drift**: Mudancas no padrao subjacente dos dados ao longo do tempo
2. **Class Imbalance**: Distribuicao desigual de classes nos dados

### 1.2 Contribuicao Principal

O paper introduz a analise de dois tipos de diversidade em ensembles:

| Tipo de Diversidade | Descricao | Quando e Melhor |
|---------------------|-----------|-----------------|
| **Homogenea** | Modelos treinados no mesmo conceito | Periodos estaveis (sem drift) |
| **Heterogenea** | Modelos de conceitos diferentes | Cenarios altamente desbalanceados com drift |

**Insight chave:** A diversidade heterogenea pode auxiliar significativamente na adaptacao em cenarios desbalanceados, pois modelos de conceitos anteriores podem ter aprendido melhor as classes minoritarias.

---

## 2. ARQUITETURA DO ALGORITMO

### 2.1 Estrutura de Ensembles

O CDCMS.CIL mantem tres ensembles em estados diferentes:

```
+------------------+     +------------------+     +------------------+
|   ensemble_NL    |     |   ensemble_OL    |     |   ensemble_NH    |
| (Normal Learner) |     | (Out-of-control) |     | (New Hypothesis) |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
   Modelos do               Modelos salvos            Modelos recuperados
 conceito atual             antes do drift            do repositorio
```

### 2.2 Repositorio de Modelos

- Armazena modelos historicos de conceitos anteriores
- Tamanho: `ensembleSize * repositorySizeMultiple` (ex: 10 * 10 = 100)
- Quando drift e detectado, clustering recupera modelos relevantes

### 2.3 Fluxo de Estados

```
                    +------------+
                    |   NORMAL   |
                    +------------+
                          |
                    drift detectado
                          |
                          v
                    +------------+
                    | OUTCONTROL |
                    +------------+
                          |
              clustering + recuperacao
                          |
                          v
                    +------------+
                    |   NORMAL   |
                    +------------+
```

---

## 3. PARAMETROS CONFIGURAVEIS

### 3.1 Parametros Principais

| Parametro | Tipo | Padrao | Descricao |
|-----------|------|--------|-----------|
| `baseLearner` | Classifier | HoeffdingTree | Classificador base do ensemble |
| `ensembleSize` | int | 10 | Tamanho maximo de cada ensemble |
| `repositorySizeMultiple` | int | 10 | Multiplicador do repositorio (n*k) |
| `timeStepsInterval` | int | 500 | Intervalo para avaliacao/atualizacao |
| `fadingFactor` | float | 0.999 | Fator de decaimento para metricas |
| `similarityThreshold` | float | 0.8 | Limiar para Q-Statistics (similaridade) |
| `driftDetector` | Detector | ADWIN | Detector de concept drift |
| `clusterer` | Clusterer | EM | Algoritmo de clustering (Weka) |

### 3.2 Parametros Especificos da Versao GMean

| Parametro | Padrao | Descricao |
|-----------|--------|-----------|
| `numClasses` | 2 | Numero de classes para calculo G-Mean |

---

## 4. IMPLEMENTACOES DISPONIVEIS

O repositorio oferece 4 variantes:

| Arquivo | Ponderacao | Balanceamento | Melhor Para |
|---------|------------|---------------|-------------|
| `CDCMS_CIL.java` | Accuracy | Nenhum | Datasets balanceados |
| `CDCMS_CIL_OSUS.java` | Accuracy | Over/Undersampling | Desbalanceamento moderado |
| `CDCMS_CIL_GMean.java` | G-Mean | Nenhum | **Nosso caso** (metrica G-Mean) |
| `CDCMS_CIL_GMean_OSUS.java` | G-Mean | Over/Undersampling | Desbalanceamento severo |

**Recomendacao para nosso projeto:** Usar `CDCMS_CIL_GMean.java` para compatibilidade com nossa metrica de avaliacao.

---

## 5. DEPENDENCIAS E REQUISITOS

### 5.1 Ambiente

- **Java:** 11.0.1 (mesmo que usamos para ROSE e ERulesD2S)
- **MOA:** 2018.6.0
- **Weka:** Integrado via MOA (para clustering)

### 5.2 Dependencias MOA

```java
moa.classifiers.AbstractClassifier
moa.classifiers.Classifier
moa.core.driftdetection.ChangeDetector
moa.clusterers.Clusterer
com.yahoo.labs.samoa.instances.*
weka.clusterers.AbstractClusterer
```

### 5.3 Detectores de Drift Suportados

- `ADWINChangeDetector` (padrao)
- `DDM_GMean` (especifico para G-Mean)
- `DDM_OCI` (Out-of-Control Interval)

---

## 6. PLANO DE INTEGRACAO

### 6.1 Estrategia

Similar a ROSE e ERulesD2S, usaremos wrapper Python que executa o classificador MOA via subprocess.

### 6.2 Passos de Integracao

1. **Compilar CDCMS.CIL**
   - Clonar repositorio
   - Compilar com Maven/Gradle
   - Gerar JAR executavel

2. **Criar Wrapper Python**
   - Converter dados para ARFF
   - Executar via `java -cp` com MOA
   - Parsear resultados

3. **Adicionar ao Notebook Comparativo**
   - Funcao `run_cdcms_cil()`
   - Suporte a cache de resultados
   - Integracao com pipeline existente

### 6.3 Comando MOA Esperado

```bash
java -Xmx4g -cp cdcms_cil.jar:moa.jar moa.DoTask \
  "EvaluateInterleavedTestThenTrain \
   -l (moa.classifiers.meta.CDCMS_CIL_GMean \
       -s 10 -t 500 -f 0.999) \
   -s (ArffFileStream -f dataset.arff) \
   -f 1000 \
   -d output.csv"
```

---

## 7. COMPARACAO COM OUTROS MODELOS

### 7.1 Tabela Comparativa

| Caracteristica | CDCMS.CIL | ARF | ACDWM | ERulesD2S | ROSE | GBML |
|----------------|-----------|-----|-------|-----------|------|------|
| Tipo | Ensemble | Ensemble | Ensemble | Rules | Ensemble | Rules |
| Drift Detection | Sim | Sim | Sim | Nao | Sim | Sim |
| Class Imbalance | **Foco** | Parcial | Sim | Nao | **Foco** | Parcial |
| Multiclass | Sim | Sim | **Nao** | Sim | Sim | Sim |
| Explicabilidade | Baixa | Baixa | Baixa | Media | Baixa | **Alta** |
| Linguagem | Java/MOA | Python | Python | Java | Java | Python |

### 7.2 Por que CDCMS.CIL e Relevante?

1. **Class Imbalance Focus:** Projetado especificamente para dados desbalanceados
2. **Diversidade Heterogenea:** Abordagem inovadora de reutilizar modelos de conceitos anteriores
3. **G-Mean Native:** Versao com suporte nativo a G-Mean como metrica
4. **State-of-the-art:** Publicado em IEEE DSAA 2025

---

## 8. PROXIMOS PASSOS

### Imediato
1. [ ] Clonar repositorio CDCMS.CIL
2. [ ] Compilar JAR com dependencias
3. [ ] Testar execucao basica com dataset simples

### Notebook
4. [ ] Criar funcao `run_cdcms_cil()` no notebook
5. [ ] Integrar ao pipeline de modelos comparativos
6. [ ] Executar em experiments_unified

### Validacao
7. [ ] Comparar resultados com outros modelos
8. [ ] Verificar consistencia de metricas

---

## 9. REFERENCIAS

- **Paper:** [The Value of Diversity for Dealing with Concept Drift in Class-Imbalanced Data Streams](https://ieeexplore.ieee.org/document/11247989/) - IEEE DSAA 2025
- **Repositorio:** https://github.com/michaelchiucw/CDCMS.CIL
- **Autores:**
  - [Chun Wai Chiu (Michael)](https://www.linkedin.com/in/michaelchiucw/)
  - [Leandro Minku](https://www.researchgate.net/profile/Chun-Wai-Chiu)

---

## 10. NOTAS TECNICAS

### 10.1 Calculo do G-Mean na Versao GMean

```java
// Pseudocodigo do calculo
gmean = 1.0
for each class i:
    recall_i = estimations[i] / b[i]  // com fading factor
    gmean *= recall_i
gmean = pow(gmean, 1.0/numClasses)  // raiz n-esima
```

### 10.2 Q-Statistics para Similaridade

O algoritmo usa Q-Statistics para medir similaridade entre modelos:

- Q = 1: Modelos identicos
- Q = 0: Modelos independentes
- Q = -1: Modelos opostos

Modelos com Q > `similarityThreshold` sao considerados do mesmo cluster.

---

**FIM DO DOCUMENTO**
