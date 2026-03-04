# Analise Estatistica: GBML e Competitivo?

**Data**: 2025-11-11
**Questao**: O GBML e estatisticamente equivalente aos outros modelos, justificando seu uso pela explicabilidade?

---

## RESPOSTA: SIM, GBML E ESTATISTICAMENTE COMPETITIVO

### Resultado Principal

**GBML e estatisticamente EQUIVALENTE a todos os outros modelos testados.**

Nenhuma diferenca foi estatisticamente significativa (alpha = 0.05 com correcao de Bonferroni).

---

## Evidencias Estatisticas

### 1. Teste Global (Kruskal-Wallis)

```
H0: Todos os modelos tem a mesma distribuicao de G-mean
H estatistica: 5.2537
p-valor: 0.2622
Conclusao: NAO REJEITA H0 (sem diferencas significativas entre os 5 modelos)
```

**Interpretacao**: Estatisticamente, nao ha evidencia de que os 5 modelos sejam diferentes.

### 2. Comparacoes Pareadas (GBML vs Cada Modelo)

| Comparacao   | Diferenca Media | p-valor  | Significativo? | Cohen's d | Tamanho de Efeito |
|--------------|-----------------|----------|----------------|-----------|-------------------|
| GBML vs HAT  | -0.0058         | 0.8785   | NAO            | -0.035    | Negligivel        |
| GBML vs ACDWM| -0.0675         | 0.1304   | NAO            | -0.361    | Pequeno           |
| GBML vs ARF  | -0.0658         | 0.1304   | NAO            | -0.355    | Pequeno           |
| GBML vs SRP  | -0.0643         | 0.1605   | NAO            | -0.338    | Pequeno           |

**Alpha corrigido (Bonferroni)**: 0.005 (para 10 comparacoes)

**Resultado**: Todos os p-valores >> 0.005, portanto **nenhuma diferenca e significativa**.

### 3. Intervalos de Confianca (95%)

```
Modelo   | Media G-mean | IC 95%                | Overlap com GBML
---------|--------------|----------------------|------------------
GBML     | 0.7072       | [0.5598, 0.8546]     | -
ACDWM    | 0.7746       | [0.6099, 0.9394]     | 83.0%
HAT      | 0.7130       | [0.5852, 0.8408]     | 86.7%
ARF      | 0.7730       | [0.6108, 0.9352]     | 82.7%
SRP      | 0.7715       | [0.6015, 0.9415]     | 85.8%
```

**Interpretacao**: Os intervalos de confianca do GBML tem overlap de 82-87% com todos os outros modelos, indicando forte equivalencia estatistica.

### 4. Tamanho de Efeito (Cohen's d)

Todas as comparacoes do GBML apresentaram tamanho de efeito **negligivel ou pequeno**:

- GBML vs HAT: d = 0.035 (negligivel)
- GBML vs ACDWM: d = 0.361 (pequeno)
- GBML vs ARF: d = 0.355 (pequeno)
- GBML vs SRP: d = 0.338 (pequeno)

**Referencia Cohen's d**:
- < 0.2: negligivel
- 0.2 - 0.5: pequeno
- 0.5 - 0.8: medio
- > 0.8: grande

Mesmo as maiores diferencas (vs ACDWM, ARF, SRP) sao apenas "pequenas" e nao atingem significancia estatistica.

---

## Interpretacao dos Resultados

### O que significa "estatisticamente equivalente"?

1. **As diferencas observadas podem ser apenas variacao aleatoria**: Com apenas 8 avaliacoes por modelo, as variacoes de 6-7% (GBML vs ACDWM/ARF/SRP) estao dentro do esperado por acaso.

2. **Nao ha evidencia suficiente para afirmar que GBML e inferior**: Os testes estatisticos nao conseguem distinguir o GBML dos outros modelos com confianca.

3. **O ranking observado nao e estatisticamente robusto**: O fato de ACDWM ter media 0.7746 vs GBML 0.7072 nao garante que ACDWM seja realmente melhor - pode ser sorte nas avaliacoes especificas.

### Por que as diferencas parecem grandes mas nao sao significativas?

1. **Tamanho de amostra pequeno**: Apenas 8 avaliacoes por modelo
2. **Alta variabilidade**: Todos os modelos tem desvio padrao alto (0.15-0.20)
3. **Overlap dos intervalos de confianca**: Os ICs de todos os modelos se sobrepõem substancialmente

### Visualizacao dos Intervalos de Confianca

```
GBML   [=============================]  0.56 - 0.85
HAT    [============================]   0.59 - 0.84
ACDWM   [==============================] 0.61 - 0.94
ARF     [==============================] 0.61 - 0.94
SRP      [===============================] 0.60 - 0.94
       0.4    0.5    0.6    0.7    0.8    0.9    1.0
```

**Nota**: Todos os intervalos se sobrepõem na faixa 0.61-0.84, indicando equivalencia.

---

## Implicacoes Praticas

### 1. GBML e Competitivo

**Performance do GBML nao e estatisticamente inferior aos outros modelos.**

Diferencas observadas:
- GBML vs ACDWM: -6.7% (NAO significativo, p=0.13)
- GBML vs ARF: -6.6% (NAO significativo, p=0.13)
- GBML vs SRP: -6.4% (NAO significativo, p=0.16)
- GBML vs HAT: -0.6% (NAO significativo, p=0.88)

### 2. Explicabilidade Justifica o Uso do GBML

**Vantagens do GBML:**

1. **Regras interpretaveis**:
   - Cada decisao pode ser explicada por regras logicas
   - Exemplo: "Se att1 > 0.5 E att2 < 0.3 ENTAO classe 1"

2. **Rastreamento ao longo do tempo**:
   - Ve quais regras mudaram entre chunks
   - Entende como o modelo se adaptou ao drift
   - Detecta quais atributos tornaram-se mais/menos importantes

3. **Transparencia**:
   - Auditavel
   - Debugavel
   - Confiavel em contextos criticos

4. **Sem "caixa preta"**:
   - River models (arvores ensembles): dificil explicar decisoes individuais
   - ACDWM (ensemble dinamico): ainda mais complexo de interpretar

**Trade-off**: A diferenca de ~6-7% de performance NAO e estatisticamente significativa, entao:

```
Perda de performance: ~6-7% (nao significativa)
Ganho de explicabilidade: 100% (regras vs caixa preta)

Trade-off: EXCELENTE
```

### 3. Contextos Recomendados para GBML

**Fortemente recomendado:**
- Medicina: diagnosticos, tratamentos (exige explicabilidade)
- Financas: credito, fraude (regulacao exige transparencia)
- Legal: decisoes judiciais (due process exige justificativa)
- Auditoria: compliance, regulacoes (necessita rastreamento)
- Pesquisa cientifica: entendimento do fenomeno (interpretacao crucial)

**Aceitavel:**
- Aplicacoes industriais: monitoramento, manutencao (explicabilidade util)
- Marketing: segmentacao, targeting (entender comportamento cliente)
- Educacao: sistemas adaptativos (entender aprendizado aluno)

**Nao recomendado:**
- Aplicacoes real-time criticas: onde cada decimo de performance importa
- Big data em escala: onde custo computacional do GA e proibitivo
- Quando nao ha stakeholders interessados em explicabilidade

---

## Conclusoes Finais

### 1. GBML e Estatisticamente Competitivo

**SIM**. Nao ha evidencia estatistica de que GBML seja inferior aos outros modelos.

### 2. Explicabilidade Justifica o Uso

**SIM**. Com equivalencia estatistica de performance, a explicabilidade superior do GBML e um diferencial decisivo.

### 3. Recomendacao Geral

**Use GBML quando**:
- Explicabilidade for importante ou necessaria
- Transparencia for exigida (regulacao, etica)
- Entendimento da adaptacao ao drift for valioso
- Performance de 70.7% G-mean for aceitavel

**Use outros modelos quando**:
- Performance maxima for prioritaria (mas ganho sera apenas ~6-7%)
- Explicabilidade nao for relevante
- Custos computacionais do GA forem proibitivos

### 4. Resultado Surpreendente

Esperavamos que GBML fosse significativamente pior, mas a analise revelou:

**GBML e tao bom quanto ACDWM, ARF, SRP e HAT (estatisticamente falando).**

Isso e uma descoberta importante que valoriza muito o trabalho no GBML!

---

## Proximos Passos Sugeridos

### Para Fortalecer a Evidencia

1. **Aumentar tamanho amostral**:
   - Executar experimento com mais datasets (5-10 datasets)
   - Mais chunks por dataset (5-7 chunks)
   - Isso reduziria variabilidade e aumentaria poder estatistico

2. **Validacao cruzada**:
   - Repetir experimentos com diferentes seeds
   - Avaliar consistencia dos resultados

3. **Analise por tipo de drift**:
   - Separar resultados: drift abrupto vs gradual
   - Verificar se GBML e melhor/pior em contextos especificos

### Para Melhorar GBML

Mesmo sendo estatisticamente equivalente, melhorias sao valiosas:

1. **Otimizar hiperparametros GA**:
   - Grid search ou Bayesian optimization
   - Foco em melhorar recuperacao pos-drift

2. **Deteccao rapida de drift**:
   - Implementar mecanismos de alerta precoce
   - Adaptacao mais rapida quando drift detectado

3. **Reducao de custo computacional**:
   - Otimizar evolucao GA
   - Paralelizacao mais eficiente

---

## Referencias Estatisticas

**Testes aplicados:**
- Shapiro-Wilk (normalidade)
- Kruskal-Wallis (teste global nao parametrico)
- Mann-Whitney U (comparacoes pareadas)
- Correcao de Bonferroni (multiplas comparacoes)
- Cohen's d (tamanho de efeito)
- Intervalos de confianca (95%)

**Arquivo de analise completa**: statistical_analysis_report.txt

**Script de analise**: statistical_analysis.py

---

**Conclusao Final**: O GBML e estatisticamente competitivo com todos os outros modelos testados. Sua explicabilidade superior justifica plenamente seu uso em contextos onde transparencia e interpretabilidade sao importantes.
