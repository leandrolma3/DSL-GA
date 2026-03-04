# Guia para Testar MCMO no Google Colab

## Passo a Passo

### 1. Preparar Google Drive

Primeiro, faça upload da pasta `DSL-AG-hybrid` para o Google Drive:

```
Google Drive/
└── DSL-AG-hybrid/
    ├── mcmo/
    │   ├── __init__.py
    │   ├── MCMO.py
    │   ├── GMM.py
    │   ├── OptAlgorithm.py
    │   ├── baseline_mcmo.py
    │   └── README.md
    ├── Test_MCMO_Adapter.ipynb  ← Notebook criado
    └── datasets/  (opcional)
        └── Electricity.csv
```

**Opções:**
- Upload manual via interface do Google Drive
- Usar Google Drive for Desktop para sincronizar pasta local

### 2. Abrir Notebook no Colab

1. No Google Drive, navegue até `DSL-AG-hybrid/`
2. Clique com botão direito em `Test_MCMO_Adapter.ipynb`
3. Selecione: **Abrir com → Google Colaboratory**

**Ou:**
- Acesse diretamente: https://colab.research.google.com
- Menu: File → Upload notebook
- Selecione `Test_MCMO_Adapter.ipynb`

### 3. Ajustar Path (Importante!)

Na **Seção 1** do notebook, ajuste o path para o local correto:

```python
# AJUSTE ESTE PATH
DSL_PATH = '/content/drive/MyDrive/DSL-AG-hybrid'
```

**Exemplos de paths comuns:**
- `'/content/drive/MyDrive/DSL-AG-hybrid'` (raiz do Drive)
- `'/content/drive/MyDrive/Projetos/DSL-AG-hybrid'` (subpasta)
- `'/content/drive/Shareddrives/NomeDrive/DSL-AG-hybrid'` (drive compartilhado)

### 4. Executar Notebook

Execute as células sequencialmente:

#### Célula 1: Montar Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
- Clique no link que aparece
- Faça login na sua conta Google
- Copie o código de autorização
- Cole no campo indicado

#### Célula 2: Instalar Dependências
```python
!pip install geatpy==2.7.0 scikit-multiflow==0.5.3
```
- Aguarde ~2-3 minutos
- Ignore warnings de compatibilidade

#### Células 3-10: Executar Testes
- Execute sequencialmente (Shift+Enter)
- Acompanhe os resultados
- Verifique gráficos gerados

### 5. Testes Incluídos

O notebook executa 4 testes principais:

#### Teste 1: Dados Sintéticos
- Gera 10 chunks de 500 amostras
- Simula drift gradual
- Testa MCMOAdapter com n_sources=3
- **Tempo estimado:** 3-5 minutos

#### Teste 2: Electricity Dataset
- Carrega Electricity.csv (se disponível)
- Divide em chunks de 1000 amostras
- Testa em dados reais
- **Tempo estimado:** 5-10 minutos (depende do tamanho)

#### Teste 3: Comparação com Baseline
- Compara MCMO vs HoeffdingTree simples
- Gera gráfico comparativo
- Calcula diferença de performance
- **Tempo estimado:** 2-3 minutos

#### Teste 4: Análise de Parâmetros
- Testa n_sources = [2, 3, 4, 5]
- Identifica melhor configuração
- **Tempo estimado:** 5-8 minutos

**Tempo total estimado:** 15-30 minutos

### 6. Resultados Esperados

#### Outputs do Notebook:

1. **Gráficos:**
   - Accuracy por chunk (sintético)
   - Accuracy por chunk (Electricity)
   - Comparação MCMO vs Baseline
   - Impacto de n_sources

2. **Métricas:**
   - Mean accuracy por teste
   - Class distribution por chunk
   - Estatísticas globais

3. **Arquivo CSV:**
   - `mcmo_test_results.csv` com resultados detalhados
   - Salvo em: `/content/` e `DSL-AG-hybrid/`

#### Exemplo de Resultado:

```
RESUMO DOS TESTES - MCMO ADAPTER
======================================================================

1. DADOS SINTÉTICOS
----------------------------------------------------------------------
   MCMO Adapter:         0.8542
   Baseline (HT):        0.8123
   Diferença:            +4.19 p.p.

2. ELECTRICITY DATASET
----------------------------------------------------------------------
   MCMO Adapter:         0.7891
   Total de samples:     45312

3. ANÁLISE DE PARÂMETROS
----------------------------------------------------------------------
   Melhor n_sources:     3
   Accuracy com best:    0.8542

4. CONCLUSÕES
----------------------------------------------------------------------
   ✓ MCMO Adapter demonstrou superioridade sobre baseline simples
   Temporal splitting funcionou corretamente
   Adapter está pronto para integração no pipeline principal
```

### 7. Troubleshooting

#### Erro: "Pasta não encontrada"
**Solução:** Ajuste `DSL_PATH` na Seção 1 para o caminho correto no seu Drive.

#### Erro: "No module named 'geatpy'"
**Solução:** Execute novamente a célula de instalação:
```python
!pip install geatpy==2.7.0 scikit-multiflow==0.5.3
```

#### Erro: "Electricity dataset não encontrado"
**Opções:**
1. Fazer upload do Electricity.csv para `DSL-AG-hybrid/datasets/`
2. Ajustar path na Seção 5
3. Pular teste de Electricity (só usar sintético)

#### Warning: "MCMO dependencies not available"
**Causa:** Dependências não instaladas ainda.
**Solução:** Execute Seção 2 (instalação) antes de continuar.

#### Erro de memória no Colab
**Soluções:**
- Reduzir tamanho dos chunks: `chunk_size = 500`
- Reduzir número de chunks: `X_chunks[:5]` (usar apenas 5)
- Usar Colab Pro para mais RAM

### 8. Otimizações

#### Para Datasets Grandes:
```python
# Reduzir número de chunks
n_chunks_elec = 20  # ao invés de usar todos

# Ou amostrar aleatoriamente
X_chunks_elec = X_chunks_elec[:20]
y_chunks_elec = y_chunks_elec[:20]
```

#### Para Execução Mais Rápida:
```python
# Reduzir gerações NSGA-II (modificar MCMO.py)
MAXGEN=25  # ao invés de 50

# Reduzir tamanho população
NIND=25  # ao invés de 50
```

#### Modo Debug:
```python
# Ativar verbose para ver detalhes
adapter = MCMOAdapter(n_sources=3, verbose=True)
```

### 9. Próximos Passos Após Teste

Se os testes forem bem-sucedidos:

1. **Baixar Resultados:**
   - Download `mcmo_test_results.csv` do Colab
   - Analisar métricas detalhadas

2. **Integrar no Pipeline:**
   - Adicionar MCMO em `main.py`
   - Executar Phase 3 experiments

3. **Análise Estatística:**
   - Friedman test
   - Wilcoxon pairwise comparisons
   - Atualizar paper com resultados

### 10. Estrutura de Arquivos no Drive

Após execução completa:

```
Google Drive/DSL-AG-hybrid/
├── mcmo/
│   ├── MCMO.py
│   ├── GMM.py
│   ├── OptAlgorithm.py
│   ├── baseline_mcmo.py
│   ├── __init__.py
│   └── README.md
├── Test_MCMO_Adapter.ipynb
├── mcmo_test_results.csv  ← Gerado pelo notebook
├── MCMO_API_DOCUMENTATION.md
├── SUMARIO_EXPLORACAO_MCMO.md
└── GUIA_TESTE_MCMO_COLAB.md  ← Este arquivo
```

## Comandos Úteis no Colab

```python
# Ver arquivos disponíveis
!ls /content/drive/MyDrive/DSL-AG-hybrid/

# Verificar memória
!free -h

# Ver versão Python
!python --version

# Verificar instalação de pacote
!pip show geatpy

# Limpar memória
import gc
gc.collect()
```

## FAQ

**Q: Preciso ter GPU no Colab?**
A: Não, CPU é suficiente. MCMO usa principalmente CPU (NSGA-II).

**Q: Quanto tempo demora o teste completo?**
A: 15-30 minutos dependendo do tamanho do dataset.

**Q: Posso pausar e retomar?**
A: Sim, mas variáveis são perdidas. Execute células do início novamente.

**Q: Resultados ficam salvos?**
A: Sim, em `mcmo_test_results.csv` no Drive.

**Q: O que fazer se der erro de timeout?**
A: Reduza número de chunks ou tamanho do dataset.

## Contato

Para dúvidas ou problemas:
1. Verificar logs de erro no notebook
2. Conferir paths e dependências
3. Consultar documentação em `mcmo/README.md`

---

**Criado por:** Claude Code
**Data:** 2025-11-24
**Versão:** 1.0
