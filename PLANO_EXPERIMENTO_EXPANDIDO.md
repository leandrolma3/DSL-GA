# Plano de Acao: Experimento Expandido com 10 Datasets

**Data**: 2025-11-11
**Objetivo**: Rodar experimento comparativo expandido com 6 modelos em 10 datasets
**Contexto**: Resultados estatisticos mostraram que GBML e competitivo - agora expandir analise

---

## 1. MOTIVACAO

### Descoberta Estatistica

A analise estatistica revelou que:
- **GBML e estatisticamente EQUIVALENTE a todos os outros modelos** (p > 0.13)
- Diferenca de 6-7% NAO e significativa (tamanho de efeito: pequeno)
- Intervalos de confianca tem overlap de 82-87%

### Objetivos do Experimento Expandido

1. **Aumentar poder estatistico**: Mais datasets (10 vs 3) e mais avaliacoes
2. **Adicionar novo competidor**: ERulesD2S (Bartoz) - tambem baseado em regras
3. **Reduzir tempo de execucao**: chunk_size de 6000 para 3000
4. **Ampliar escopo**: Diferentes tipos de drift e geradores

---

## 2. MODELOS A COMPARAR

### 2.1. GBML (Atual)
- **Tipo**: Genetic-Based Machine Learning
- **Explicabilidade**: Alta (regras interpretaveis)
- **Custo computacional**: Alto (evolucao GA)
- **Status**: Implementado e funcionando

### 2.2. ACDWM
- **Tipo**: Adaptive Chunk-based Dynamic Weighted Majority
- **Explicabilidade**: Media (ensemble dinamico)
- **Custo computacional**: Moderado
- **Status**: Integrado

### 2.3. River Models (HAT, ARF, SRP)
- **Tipo**: Adaptive Random Forest variants
- **Explicabilidade**: Baixa (arvores ensembles)
- **Custo computacional**: Baixo a Moderado
- **Status**: Integrado

### 2.4. ERulesD2S (Bartoz) - NOVO
- **Tipo**: Evolving Rule-based classifier
- **Explicabilidade**: Alta (regras interpretaveis)
- **Tecnologia**: Programacao genetica com GPU (Java/MOA)
- **Custo computacional**: Alto (GP + GPU)
- **Status**: Necessita integracao

#### Caracteristicas ERulesD2S

```
Parametros padrao:
- population_size: 25 (vs 100 do GBML)
- num_generations: 50 (vs 200 do GBML)
- rules_per_class: 5
- chunk_size: 1000
- Fitness: sensitivity × specificity
- Paralelizacao: GPU (CUDA)
```

**Desafio de Integracao**:
- ERulesD2S usa framework MOA (Java)
- GBML/River/ACDWM usam Python
- Possibilidades:
  1. Wrapper Python -> Java via subprocess
  2. Executar separadamente e consolidar resultados
  3. Adaptar codigo MOA para interface Python (PyJNIus)

---

## 3. SELECAO DE DATASETS

### 3.1. Criterios de Selecao

Para ter 10 datasets diversificados:
1. **Diferentes tipos de drift**: abrupto, gradual, recorrente
2. **Diferentes severidades**: leve, moderado, severo
3. **Diferentes geradores**: RBF, SEA, AGRAWAL, HYPERPLANE, SINE
4. **Balanco**: Sinteticos conhecidos da literatura

### 3.2. Datasets Selecionados (10)

#### Categoria 1: Drift Abrupto (4 datasets)

1. **RBF_Abrupt_Severe**
   - Gerador: RandomRBF
   - Drift: Abrupto severo
   - Features: 10
   - Chunks: 2 conceitos × 2 chunks = 4 chunks
   - Justificativa: Baseline atual, drift muito forte

2. **RBF_Abrupt_Moderate** (renomear de RBF_Abrupt_Moderate)
   - Gerador: RandomRBF
   - Drift: Abrupto moderado
   - Features: 10
   - Chunks: 2 conceitos × 2 chunks = 4 chunks
   - Justificativa: Baseline atual, drift medio

3. **SEA_Abrupt_Simple**
   - Gerador: SEA
   - Drift: Abrupto
   - Features: 3
   - Chunks: f1 (5 chunks) -> f3 (5 chunks) = 10 chunks
   - Justificativa: Benchmark classico, poucos features

4. **AGRAWAL_Abrupt_Simple_Severe**
   - Gerador: AGRAWAL
   - Drift: Abrupto severo
   - Features: 9
   - Chunks: f1 (5 chunks) -> f6 (5 chunks) = 10 chunks
   - Justificativa: Drift severo, mais features

#### Categoria 2: Drift Gradual (3 datasets)

5. **RBF_Gradual_Moderate**
   - Gerador: RandomRBF
   - Drift: Gradual (width=2 chunks)
   - Features: 10
   - Chunks: c1 (5 chunks) -> c3 (5 chunks) = 10 chunks
   - Justificativa: Baseline atual

6. **SEA_Gradual_Simple_Fast**
   - Gerador: SEA
   - Drift: Gradual rapido (width=1 chunk)
   - Features: 3
   - Chunks: f1 (5 chunks) -> f3 (5 chunks) = 10 chunks
   - Justificativa: Drift gradual rapido

7. **AGRAWAL_Gradual_Chain**
   - Gerador: AGRAWAL
   - Drift: Gradual (width=2 chunks)
   - Features: 9
   - Chunks: f1->f3->f5->f7 (4 conceitos × 3 chunks) = 12 chunks
   - Justificativa: Multiplos drifts graduais em cadeia

#### Categoria 3: Cenarios Complexos (3 datasets)

8. **HYPERPLANE_Abrupt_Simple**
   - Gerador: Hyperplane
   - Drift: Abrupto
   - Features: 10
   - Chunks: plane1 (6 chunks) -> plane2 (6 chunks) = 12 chunks
   - Justificativa: Hiperplano rotativo, diferente de RBF

9. **SINE_Gradual_Recurring**
   - Gerador: SINE
   - Drift: Gradual + Recorrente
   - Features: 5
   - Chunks: f1 (4) -> f3 (5) -> f1 (4) = 13 chunks
   - Justificativa: Conceito recorrente (volta ao inicio)

10. **STAGGER_Abrupt_Chain**
    - Gerador: STAGGER
    - Drift: Abrupto em cadeia
    - Features: 3 (categoricos)
    - Chunks: f1 (4) -> f2 (4) -> f3 (4) = 12 chunks
    - Justificativa: Features categoricos, multiplos drifts

### 3.3. Resumo dos Datasets

| ID | Dataset | Gerador | Drift | Features | Chunks | Avaliacoes |
|----|---------|---------|-------|----------|--------|------------|
| 1  | RBF_Abrupt_Severe | RBF | Abrupto Severo | 10 | 4 | 3 |
| 2  | RBF_Abrupt_Moderate | RBF | Abrupto Moderado | 10 | 4 | 3 |
| 3  | SEA_Abrupt_Simple | SEA | Abrupto | 3 | 10 | 9 |
| 4  | AGRAWAL_Abrupt_Simple_Severe | AGRAWAL | Abrupto Severo | 9 | 10 | 9 |
| 5  | RBF_Gradual_Moderate | RBF | Gradual | 10 | 10 | 9 |
| 6  | SEA_Gradual_Simple_Fast | SEA | Gradual Rapido | 3 | 10 | 9 |
| 7  | AGRAWAL_Gradual_Chain | AGRAWAL | Gradual Cadeia | 9 | 12 | 11 |
| 8  | HYPERPLANE_Abrupt_Simple | HYPERPLANE | Abrupto | 10 | 12 | 11 |
| 9  | SINE_Gradual_Recurring | SINE | Gradual Recorrente | 5 | 13 | 12 |
| 10 | STAGGER_Abrupt_Chain | STAGGER | Abrupto Cadeia | 3 | 12 | 11 |

**Total**: 97 chunks → 87 avaliacoes (train-then-test)

---

## 4. ESTIMATIVA DE TEMPO DE EXECUCAO

### 4.1. Dados do Experimento Anterior (chunk_size=6000)

| Dataset | Tempo (min) | Chunks | Tempo/chunk |
|---------|-------------|--------|-------------|
| RBF_Abrupt_Severe | 231.4 | 4 | 57.9 min |
| RBF_Abrupt_Moderate | 208.7 | 4 | 52.2 min |
| **Media** | **220.0** | **4** | **55.0 min** |

### 4.2. Estimativa com chunk_size=3000

**Impacto da reducao do chunk_size**:

1. **Evolucao GA mais rapida**: Menos instancias → menos avaliacoes de fitness
   - Estimativa: 60-70% do tempo original
   - Tempo/chunk: 55 × 0.65 ≈ **35.8 min/chunk**

2. **Numero de chunks**: Mesma quantidade de dados, chunks menores
   - Para mesma cobertura: precisa de mais chunks
   - Mas datasets ja tem numero de chunks definido

3. **Overhead fixo**: Setup, load, save resultados (constante)

**Estimativa conservadora**: 35-40 min/chunk com chunk_size=3000

### 4.3. Tempo Total Estimado

#### Cenario 1: Apenas GBML (sem ERulesD2S ainda)

```
Modelos: GBML + ACDWM + HAT + ARF + SRP = 5 modelos
Datasets: 10
Total chunks: 97
Total avaliacoes: 87

Tempo por chunk (estimado):
- GBML: 35 min
- ACDWM: 2 min
- River (HAT, ARF, SRP): 3 × 1 min = 3 min
- Total: 40 min/chunk

Tempo total: 97 chunks × 40 min = 3880 min ≈ 64.7 horas
```

**Conclusao**: **NAO VIAVEL no Colab gratuito (12h) ou Pro (24h)**

#### Cenario 2: Reducao para Caber no Colab Pro (24h)

**Ajustes necessarios**:
1. Reduzir numero de datasets: 10 → **6 datasets**
2. Reduzir chunks por dataset: Usar datasets com menos chunks

**Selecao reduzida (6 datasets)**:

| ID | Dataset | Chunks | Avaliacoes | Tempo (40min/chunk) |
|----|---------|--------|------------|---------------------|
| 1  | RBF_Abrupt_Severe | 4 | 3 | 160 min (2.7h) |
| 2  | RBF_Abrupt_Moderate | 4 | 3 | 160 min (2.7h) |
| 3  | SEA_Abrupt_Simple | 10 | 9 | 400 min (6.7h) |
| 5  | RBF_Gradual_Moderate | 10 | 9 | 400 min (6.7h) |
| 6  | SEA_Gradual_Simple_Fast | 10 | 9 | 400 min (6.7h) |
| 8  | HYPERPLANE_Abrupt_Simple | 12 | 11 | 480 min (8.0h) |

**Total**: 50 chunks × 40 min = **2000 min ≈ 33.3 horas**

Ainda **NAO CABE em 24h do Colab Pro**!

#### Cenario 3: Estrategia Hibrida (VIAVEL)

**Opcao A: Executar em 2 sessoes Colab**

**Sessao 1 (22h)**:
- Datasets 1-3: RBF_Abrupt_Severe, RBF_Abrupt_Moderate, SEA_Abrupt_Simple
- Total: 18 chunks × 40 min = 720 min (12h)
- Margem de seguranca: OK

**Sessao 2 (22h)**:
- Datasets 4-6: RBF_Gradual_Moderate, SEA_Gradual_Simple_Fast, HYPERPLANE_Abrupt_Simple
- Total: 32 chunks × 40 min = 1280 min (21.3h)
- Margem de seguranca: OK

**Opcao B: Executar localmente (sem Colab)**

Se tiver maquina local com recursos:
- Total: 50 chunks × 40 min = 33.3 horas
- Pode rodar overnight (2 noites)
- Sem restricao de tempo

**Opcao C: Reduzir chunks mantendo 6 datasets**

Modificar config para:
- Cada dataset: 3-4 chunks (vs 10-12)
- Total: 6 datasets × 4 chunks = 24 chunks
- Tempo: 24 × 40 = 960 min (16h) - **CABE no Colab Pro 24h**

---

## 5. PLANO DE ACAO RECOMENDADO

### 5.1. Plano Conservador (OPCAO C - Recomendado)

**Configuracao**:
- Chunk size: 3000
- Numero de datasets: 6
- Chunks por dataset: 3-4
- Modelos: GBML + ACDWM + HAT + ARF + SRP (5 modelos)
- ERulesD2S: Adicionar em fase posterior

**Datasets Selecionados (6)**:

1. RBF_Abrupt_Severe (4 chunks)
2. RBF_Abrupt_Moderate (4 chunks)
3. RBF_Gradual_Moderate (4 chunks)
4. SEA_Abrupt_Simple (4 chunks)
5. AGRAWAL_Abrupt_Simple_Severe (4 chunks)
6. HYPERPLANE_Abrupt_Simple (4 chunks)

**Total**: 24 chunks × 40 min = **960 min (16 horas)**

**Viabilidade**: **SIM - Cabe no Colab Pro 24h com margem de 8h**

**Avaliacoes totais**: 24 - 6 = **18 avaliacoes por modelo** (vs 8 no experimento anterior)

**Beneficios**:
- Mais que dobra poder estatistico (18 vs 8 avaliacoes)
- Diversidade de geradores (RBF, SEA, AGRAWAL, HYPERPLANE)
- Viavel em uma unica sessao Colab Pro

### 5.2. Roadmap de Implementacao

#### Fase 1: Preparacao (1-2 horas)

**Tarefa 1.1**: Atualizar config_comparison.yaml
- Reduzir chunk_size: 6000 → 3000
- Atualizar num_chunks para datasets selecionados
- Listar 6 datasets em drift_simulation_experiments

**Tarefa 1.2**: Validar geracao de chunks
- Testar cada dataset com novo chunk_size
- Verificar que chunks sao gerados corretamente
- Salvar checksums para reproducibilidade

**Tarefa 1.3**: Teste rapido local
- Executar 1 dataset com 2 chunks
- Validar que todos os 5 modelos funcionam
- Tempo esperado: 80 min (2 chunks × 40 min)

#### Fase 2: Execucao Experimento Principal (16 horas)

**Tarefa 2.1**: Configurar Colab Pro
- Verificar runtime GPU/CPU
- Montar Google Drive
- Upload de arquivos atualizados

**Tarefa 2.2**: Executar experimento
- Comando: python run_comparison_colab.py
- Monitorar progresso via logs
- Checkpoints a cada dataset completo

**Tarefa 2.3**: Backup incremental
- Salvar resultados a cada dataset
- Copia de seguranca no Drive
- Em caso de desconexao: retomar do ultimo checkpoint

#### Fase 3: Analise de Resultados (2-3 horas)

**Tarefa 3.1**: Analise estatistica completa
- Executar statistical_analysis.py
- Comparar 18 avaliacoes vs 8 anteriores
- Testes de significancia atualizados

**Tarefa 3.2**: Analise por tipo de drift
- Separar: Abrupto vs Gradual
- Verificar se GBML e melhor/pior em cada contexto
- Identificar padroes

**Tarefa 3.3**: Relatorio consolidado
- Atualizar RELATORIO_EXPERIMENTO_COMPLETO.md
- Incluir novos datasets
- Conclusoes estatisticas robustas

#### Fase 4: Integracao ERulesD2S (Opcional - 5-8 horas)

**Tarefa 4.1**: Investigar integracao
- Testar wrapper Python -> Java
- Avaliar viabilidade tecnica
- Estimar overhead de comunicacao

**Tarefa 4.2**: Implementar adapter
- Criar baseline_erulesd2s.py
- Interface compativel com framework
- Conversao de formatos (Python ↔ Java/MOA)

**Tarefa 4.3**: Executar comparacao com ERulesD2S
- Rodar subset de datasets (3 datasets)
- Comparar com GBML (ambos baseados em regras)
- Analise de explicabilidade vs performance

---

## 6. ESTIMATIVA DE GANHO ESTATISTICO

### 6.1. Poder Estatistico Atual (8 avaliacoes)

```
N = 8 avaliacoes por modelo
Alpha = 0.05 (com Bonferroni: 0.005)
Resultado: Nenhuma diferenca significativa detectada
Intervalos de confianca: Largos (0.29-0.34)
```

### 6.2. Poder Estatistico Proposto (18 avaliacoes)

```
N = 18 avaliacoes por modelo (aumento de 125%)
Alpha = 0.05 (com Bonferroni: 0.005)
Beneficios esperados:
- Intervalos de confianca: 50% mais estreitos
- Deteccao de diferencas: Efeito medio (d=0.5) detectavel
- Robustez: Resultados mais confiaveis
```

**Calculo de IC**:
- IC width ~ 1/sqrt(n)
- Atual (n=8): width ~ 1/2.83 = 0.35
- Proposto (n=18): width ~ 1/4.24 = 0.24
- Reducao: 32% mais estreito

### 6.3. Comparacao

| Aspecto | Experimento Atual | Experimento Proposto | Ganho |
|---------|-------------------|----------------------|-------|
| Avaliacoes/modelo | 8 | 18 | +125% |
| Datasets | 3 | 6 | +100% |
| Diversidade | Apenas RBF | RBF, SEA, AGRAWAL, HYPERPLANE | +300% |
| IC width (relativo) | 1.00 | 0.68 | -32% |
| Poder detectar d=0.5 | ~40% | ~80% | +100% |
| Tempo execucao | 7.3h | 16h | +119% |
| Viavel Colab Pro? | SIM (7.3h < 24h) | SIM (16h < 24h) | - |

---

## 7. DECISOES A TOMAR

### 7.1. Decisao Imediata: Numero de Datasets

**Opcao A: 6 datasets (Recomendado)**
- Pros: Cabe no Colab Pro (16h), dobra poder estatistico
- Contras: Nao atinge meta de 10 datasets

**Opcao B: 10 datasets (2 sessoes)**
- Pros: Maior diversidade, meta original
- Contras: Requer 2 sessoes Colab, mais complexo

**Opcao C: 4 datasets (Conservador)**
- Pros: Margem de seguranca maxima (10h)
- Contras: Poder estatistico menor

**RECOMENDACAO**: **Opcao A (6 datasets)**

### 7.2. Decisao Curto Prazo: ERulesD2S

**Opcao A: Incluir agora (junto com 6 datasets)**
- Pros: Comparacao completa desde inicio
- Contras: Maior complexidade, risco de falhas, +tempo

**Opcao B: Adicionar depois (Fase 4)**
- Pros: Validar framework primeiro, menos risco
- Contras: Precisa rodar experimentos novamente

**RECOMENDACAO**: **Opcao B (Adicionar depois)**

Justificativa:
1. ERulesD2S usa tecnologia diferente (Java/MOA vs Python)
2. Integracao nao e trivial (requer wrapper ou adaptacao)
3. Risco de bugs compromete experimento principal
4. Melhor validar 5 modelos Python primeiro

### 7.3. Decisao Medio Prazo: Chunk Size

**Opcao A: chunk_size=3000 (Proposto)**
- Pros: GA mais rapido, mais datasets viaveis
- Contras: Menos instancias por treino

**Opcao B: chunk_size=4000 (Meio termo)**
- Pros: Balanco treino vs tempo
- Contras: Ainda pesado

**Opcao C: chunk_size=2000 (Agressivo)**
- Pros: GA muito mais rapido
- Contras: Pode prejudicar aprendizado

**RECOMENDACAO**: **Opcao A (chunk_size=3000)**

Justificativa: Reducao de 50% ja e significativa, mantendo aprendizado adequado

---

## 8. PROXIMOS PASSOS IMEDIATOS

### Passo 1: Validacao Local (2-3 horas)

```bash
# Atualizar config
vim config_comparison.yaml
# chunk_size: 3000
# datasets: [RBF_Abrupt_Severe, RBF_Abrupt_Moderate]

# Teste rapido (2 datasets, 2 chunks cada)
python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --config config_comparison.yaml \
    --models HAT ARF \
    --chunks 2 \
    --chunk-size 3000 \
    --acdwm \
    --output test_3000_local
```

**Verificar**:
- Tempo de execucao (deve ser ~70 min para 2 chunks)
- Metricas geradas corretamente
- Todos os 5 modelos funcionam

### Passo 2: Criar Config Final (30 min)

Arquivo: config_experiment_expanded.yaml

```yaml
experiment_settings:
  run_mode: drift_simulation
  drift_simulation_experiments:
  - RBF_Abrupt_Severe
  - RBF_Abrupt_Moderate
  - RBF_Gradual_Moderate
  - SEA_Abrupt_Simple
  - AGRAWAL_Abrupt_Simple_Severe
  - HYPERPLANE_Abrupt_Simple

data_params:
  chunk_size: 3000
  num_chunks: 5  # 4 chunks principais + 1 extra
  max_instances: 15000

# ... resto igual ao config_comparison.yaml
```

### Passo 3: Preparar Colab (30 min)

1. Upload arquivos para Google Drive
2. Atualizar notebooks se necessario
3. Verificar dependencias instaladas
4. Configurar runtime (CPU ou GPU)

### Passo 4: Executar no Colab Pro (16 horas)

```python
!cd {DRIVE_PATH} && python run_comparison_colab.py \
    --config config_experiment_expanded.yaml
```

### Passo 5: Analisar Resultados (2-3 horas)

```bash
# Analise estatistica
python statistical_analysis.py

# Analise detalhada
python analyze_experiment_detailed.py

# Gerar relatorios
# - statistical_analysis_report.txt
# - experiment_detailed_analysis.txt
# - CONCLUSAO_ESTATISTICA_GBML.md (atualizado)
```

---

## 9. CRONOGRAMA

### Semana 1: Validacao e Preparacao
- Dia 1: Teste local com chunk_size=3000 (2-3h)
- Dia 2: Ajustes baseados no teste, criar configs finais (2h)
- Dia 3: Preparar ambiente Colab, upload arquivos (1h)

### Semana 2: Execucao Principal
- Dia 4: Executar experimento no Colab Pro (16h overnight)
- Dia 5: Analisar resultados, gerar relatorios (3h)
- Dia 6: Revisar analise estatistica, documentar conclusoes (2h)

### Semana 3 (Opcional): ERulesD2S
- Dia 7-8: Investigar integracao ERulesD2S (4h)
- Dia 9-10: Implementar adapter, testar (4h)
- Dia 11: Executar comparacao subset (8h)
- Dia 12: Analise comparativa GBML vs ERulesD2S (2h)

**Tempo total**: 2-3 semanas

---

## 10. RISCOS E MITIGACOES

### Risco 1: Colab desconectar antes de terminar

**Probabilidade**: Media
**Impacto**: Alto

**Mitigacao**:
- Salvar checkpoints a cada dataset completo
- Implementar resume_from_checkpoint()
- Backup incremental no Google Drive
- Dividir em 2 sessoes se necessario

### Risco 2: chunk_size=3000 prejudicar aprendizado

**Probabilidade**: Baixa
**Impacto**: Medio

**Mitigacao**:
- Validar em teste local primeiro
- Comparar metricas com chunk_size=6000
- Se necessario: ajustar para 4000

### Risco 3: Integracao ERulesD2S muito complexa

**Probabilidade**: Alta
**Impacto**: Baixo (nao bloqueia experimento principal)

**Mitigacao**:
- Deixar ERulesD2S para Fase 4 (opcional)
- Focar primeiro nos 5 modelos Python
- Se necessario: rodar ERulesD2S separadamente

### Risco 4: Resultados nao melhorarem significancia

**Probabilidade**: Baixa
**Impacto**: Medio

**Mitigacao**:
- 18 avaliacoes vs 8 ja e melhoria substancial
- Mesmo sem significancia, mais dados = mais robusto
- Analise por tipo de drift pode revelar padroes

---

## 11. METRICAS DE SUCESSO

### Sucesso Minimo
- [x] Experimento completa 6 datasets
- [x] 18 avaliacoes por modelo coletadas
- [x] Analise estatistica executada
- [x] Intervalos de confianca reduzidos em 30%

### Sucesso Esperado
- [x] Sucesso minimo +
- [x] Identificar contextos onde GBML e melhor (drift abrupto vs gradual)
- [x] Relatorio consolidado documentado
- [x] Poder estatistico suficiente para publicacao

### Sucesso Ideal
- [x] Sucesso esperado +
- [x] ERulesD2S integrado e comparado
- [x] 10 datasets completados (2 sessoes)
- [x] Artigo cientifico submetido

---

## 12. RECURSOS NECESSARIOS

### Computacionais
- Google Colab Pro: $10/mes (sessao 24h)
- Google Drive: 15GB espaco (resultados)
- Maquina local (opcional): Para testes rapidos

### Tempo
- Desenvolvimento: 8-10 horas
- Execucao: 16 horas (Colab)
- Analise: 5-6 horas
- **Total**: 29-32 horas

### Humanos
- 1 pesquisador: Todas as fases
- Revisor (opcional): Validar analise estatistica

---

## CONCLUSAO

Este plano propoe um experimento expandido **viavel e robusto** que:

1. **Mais que dobra o poder estatistico** (18 vs 8 avaliacoes)
2. **Aumenta diversidade** (6 datasets, 4 geradores diferentes)
3. **Cabe no Colab Pro** (16h < 24h com margem de seguranca)
4. **Mantem todos os modelos atuais** (GBML, ACDWM, River)
5. **Deixa caminho para ERulesD2S** (Fase 4 opcional)

**RECOMENDACAO FINAL**: Executar Plano Conservador (Opcao A) com 6 datasets e chunk_size=3000.

ERulesD2S pode ser adicionado posteriormente em experimento focado (comparacao de modelos baseados em regras: GBML vs ERulesD2S).
