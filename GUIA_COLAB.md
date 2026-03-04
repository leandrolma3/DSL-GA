# Guia: Executando Experimentos no Google Colab

**Data**: 2025-01-07
**Versão**: 1.0

---

## VISÃO GERAL

Este guia explica como executar a comparação **GBML vs River vs ACDWM** no Google Colab.

### Notebooks Disponíveis:

1. **`setup_acdwm_colab.ipynb`** - Setup inicial (execute PRIMEIRO)
2. **`experimento_comparacao_colab.ipynb`** - Execução de experimentos

---

## PARTE 1: PREPARAÇÃO (NO SEU COMPUTADOR)

### Passo 1.1: Upload dos Arquivos para Google Drive

Faça upload da pasta **DSL-AG-hybrid** completa para seu Google Drive:

```
MyDrive/
└── DSL-AG-hybrid/              ← Sua pasta principal
    ├── baseline_acdwm.py
    ├── baseline_river.py
    ├── compare_gbml_vs_river.py
    ├── config_comparison.yaml
    ├── data_converters.py
    ├── shared_evaluation.py
    ├── gbml_evaluator.py
    ├── metrics.py
    ├── run_comparison_colab.py
    ├── setup_acdwm_colab.ipynb        ← Notebook 1
    ├── experimento_comparacao_colab.ipynb  ← Notebook 2
    └── ... (outros arquivos)
```

**Métodos de upload**:
- **Opção A**: Arrastar e soltar no Google Drive web
- **Opção B**: Usar Google Drive Desktop (sincronização automática)
- **Opção C**: Usar `rclone` ou `gdrive` (linha de comando)

### Passo 1.2: Verificar Estrutura

Acesse seu Google Drive e confirme que todos os arquivos estão na pasta correta.

---

## PARTE 2: SETUP NO GOOGLE COLAB

### Passo 2.1: Abrir Notebook de Setup

1. No Google Drive, navegue até `DSL-AG-hybrid/`
2. Clique com botão direito em **`setup_acdwm_colab.ipynb`**
3. Selecione **"Abrir com" → "Google Colaboratory"**

Se "Google Colaboratory" não aparecer:
- Vá em "Conectar mais apps"
- Busque "Colaboratory"
- Instale e conecte

### Passo 2.2: Configurar Runtime

No Colab, vá em **Runtime → Change runtime type**:

- **Runtime type**: Python 3
- **Hardware accelerator**: None (ou GPU se disponível)
- **Runtime shape**: Standard (ou High-RAM se tiver acesso)

Clique em **Save**.

### Passo 2.3: Ajustar Caminho da Pasta

Na **Célula 2** do notebook (`# 2. Configurar Caminhos`), ajuste a linha:

```python
DRIVE_PATH = '/content/drive/MyDrive/DSL-AG-hybrid'
```

**IMPORTANTE**:
- Se sua pasta está em outro local, ajuste o caminho
- Exemplo: `/content/drive/MyDrive/Projetos/DSL-AG-hybrid`

### Passo 2.4: Executar Setup

Execute **TODAS as células** do notebook em ordem:

1. **Célula 1**: Montar Google Drive
   - Clique no link de autorização
   - Faça login com sua conta Google
   - Copie o código de autorização

2. **Célula 2**: Configurar caminhos
   - Deve mostrar: `[OK] Pasta encontrada`

3. **Célula 3**: Instalar dependências
   - Leva ~2-3 minutos

4. **Célula 4**: Verificar instalação
   - Todos os pacotes devem mostrar `[OK]`

5. **Célula 5**: Clonar ACDWM
   - Se já existe, mostra mensagem
   - Senão, clona do GitHub

6. **Célula 6**: Testar importações
   - Deve mostrar: `[OK] TODOS OS MODULOS IMPORTADOS`

7. **Célula 7**: Teste rápido ACDWM
   - Executa 2 chunks sintéticos
   - Valida que ACDWM funciona

8. **Célula 8**: Verificar configuração
   - Lista datasets disponíveis

9. **Célula 9**: Resumo
   - Mostra status geral

### Passo 2.5: Validar Setup

Ao final, você deve ver:

```
======================================================================
AMBIENTE CONFIGURADO E PRONTO PARA USO!
======================================================================
```

Se tudo passou, o setup está **COMPLETO**! ✓

---

## PARTE 3: EXECUTAR EXPERIMENTOS

### Passo 3.1: Abrir Notebook de Experimentos

1. No Drive, abra **`experimento_comparacao_colab.ipynb`**
2. **Runtime → Change runtime type** (mesmo do setup)

### Passo 3.2: Executar Setup Rápido

Execute **Célula 1** (Setup Inicial):
- Monta Drive
- Configura caminhos

### Passo 3.3: TESTE RÁPIDO (RECOMENDADO)

**SEMPRE comece com o teste rápido!**

Execute **Célula 2** (Teste Rápido):

```python
!python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --config config_comparison.yaml \
    --models HAT \
    --chunks 2 \
    --chunk-size 1000 \
    --seed 42 \
    --output test_quick_results
```

**O que acontece**:
- Dataset: RBF_Abrupt_Severe
- 2 chunks de 1000 samples cada
- Modelos: GBML + HAT
- Tempo: ~5-10 minutos

Execute **Célula 3** para ver resultados:
- Tabela de métricas
- Comparação GBML vs HAT

### Passo 3.4: Teste Intermediário (Opcional)

Se o teste rápido funcionou, execute **Célula 4**:

```python
!python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --config config_comparison.yaml \
    --models HAT ARF \
    --chunks 3 \
    --chunk-size 6000 \
    --seed 42 \
    --output test_intermediate_results
```

**O que muda**:
- 3 chunks (configuração final)
- 6000 samples por chunk
- 2 modelos River (HAT + ARF)
- Tempo: ~30-60 minutos

Execute **Célula 5** para analisar resultados.

### Passo 3.5: Experimento Completo (Produção)

⚠️ **ATENÇÃO**: Execute apenas se os testes anteriores funcionaram!

Execute **Célula 6**:

```python
!python run_comparison_colab.py
```

**Configuração completa**:
- 3 datasets (RBF_Abrupt_Severe, RBF_Abrupt_Moderate, RBF_Gradual_Moderate)
- 3 chunks por dataset
- 6000 samples por chunk
- Modelos: GBML + HAT + ARF + SRP

**Tempo estimado**: 10-15 horas

**DICA**:
- Execute à noite ou quando puder deixar rodando
- Colab pode desconectar após 12h (versão gratuita)
- Considere Colab Pro se precisar de sessões mais longas

---

## PARTE 4: ANALISAR RESULTADOS

### Passo 4.1: Verificar Arquivos Gerados

```
comparison_results/
└── experiment_TIMESTAMP/
    ├── experiment_config.json          ← Configuração do experimento
    ├── experiment_summary.json         ← Resumo (duração, status)
    ├── consolidated_results.csv        ← RESULTADOS CONSOLIDADOS
    ├── summary_statistics.txt          ← Estatísticas resumidas
    └── [dataset]_seed42/
        ├── comparison_table.csv        ← Resultados por chunk
        ├── GBML_results.json
        └── River_[model]_results.json
```

### Passo 4.2: Ver Resultados Consolidados

Execute **Célula 7** (Analisar Resultados Completos):

Mostra:
- Tabela consolidada de todos os datasets
- Melhor modelo por dataset
- Ranking geral por G-mean

### Passo 4.3: Gerar Gráficos

Execute **Célula 8** (Gerar Gráficos):

Gera:
- Gráfico de barras: G-mean por dataset e modelo
- Boxplot: Distribuição de G-mean por modelo

### Passo 4.4: Salvar Resultados

Execute **Célula 9** (Salvar Resultados no Drive):

Cria backup com timestamp:
```
comparison_results_backup_20250107_143022/
```

Os resultados ficam salvos no Google Drive automaticamente!

---

## PARTE 5: INCLUIR ACDWM NA COMPARAÇÃO

**STATUS**: ✓ ACDWM TOTALMENTE INTEGRADO!

O ACDWM foi integrado ao `compare_gbml_vs_river.py` e agora pode ser executado junto com GBML e River.

### Como Usar:

Adicione a flag `--acdwm` ao comando de comparação:

```bash
!cd {DRIVE_PATH} && python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --config config_comparison.yaml \
    --models HAT ARF \
    --chunks 2 \
    --chunk-size 1000 \
    --acdwm \
    --seed 42 \
    --output test_with_acdwm
```

### Opções Disponíveis:

- `--acdwm`: Inclui ACDWM na comparação (flag on/off)
- `--acdwm-path`: Caminho para o repositório ACDWM (padrão: 'ACDWM')

### Exemplo: Apenas ACDWM (sem GBML/River)

```bash
!cd {DRIVE_PATH} && python compare_gbml_vs_river.py \
    --stream RBF_Abrupt_Severe \
    --config config_comparison.yaml \
    --chunks 2 \
    --chunk-size 1000 \
    --no-gbml \
    --no-river \
    --acdwm \
    --output acdwm_only
```

### Garantia de Comparação Justa:

✓ Todos os modelos recebem **exatamente os mesmos chunks**
✓ Mesma seed garante reprodutibilidade
✓ Mesma metodologia (train-then-test)
✓ Resultados consolidados automaticamente

### Documentação Completa:

Veja `COMO_USAR_ACDWM.md` para documentação detalhada sobre:
- Exemplos de comandos
- Estrutura de resultados
- Análise de gráficos
- Troubleshooting

---

## TROUBLESHOOTING

### Problema 1: "Pasta não encontrada"

**Erro**: `[X] ERRO: Pasta nao encontrada: /content/drive/MyDrive/DSL-AG-hybrid`

**Solução**:
- Verifique se a pasta está no lugar correto no Drive
- Ajuste a variável `DRIVE_PATH` na Célula 2
- Execute `!ls -la /content/drive/MyDrive/` para ver pastas disponíveis

### Problema 2: "ModuleNotFoundError"

**Erro**: `ModuleNotFoundError: No module named 'river'`

**Solução**:
- Re-execute a Célula 3 (Instalar Dependências)
- Verifique se instalação completou sem erros
- Execute Célula 4 para verificar versões

### Problema 3: "ACDWM não encontrado"

**Erro**: `[X] Diretorio ACDWM nao encontrado`

**Solução**:
- Execute Célula 5 (Clonar ACDWM)
- Se já existe, verifique: `!ls -la /content/drive/MyDrive/DSL-AG-hybrid/ACDWM/`

### Problema 4: Runtime desconectou

**Erro**: Colab desconectou durante experimento longo

**Solução**:
- Colab gratuito desconecta após ~12h
- Execute experimentos em lotes menores
- Considere Colab Pro ($9.99/mês) para sessões de 24h
- Use checkpointing: salve progresso intermediário

### Problema 5: Memória insuficiente

**Erro**: `RuntimeError: Out of memory`

**Solução**:
- Reduza `chunk_size` (ex: 6000 → 3000)
- Reduza número de chunks
- Mude para High-RAM runtime (se disponível)

### Problema 6: Arquivo de configuração não encontrado

**Erro**: `FileNotFoundError: config_comparison.yaml`

**Solução**:
- Verifique se `config_comparison.yaml` está na pasta DSL-AG-hybrid
- Execute: `!ls -lh *.yaml` para confirmar

---

## DICAS E BOAS PRÁTICAS

### Performance:

1. **Sempre execute teste rápido primeiro**
   - Valida que tudo funciona
   - Detecta problemas cedo

2. **Monitore recursos**
   - Vá em **Runtime → Manage sessions**
   - Veja RAM e disk usage

3. **Salve resultados frequentemente**
   - Execute Célula 9 periodicamente
   - Resultados já estão no Drive automaticamente

### Organização:

1. **Use nomes descritivos**
   - Modifique `--output` para incluir data/hora
   - Exemplo: `--output results_20250107_teste1`

2. **Documente experimentos**
   - Anote parâmetros testados
   - Salve logs importantes

3. **Backup de configurações**
   - Faça cópia de `config_comparison.yaml` antes de modificar

### Colaboração:

1. **Compartilhe notebooks**
   - Clique em "Share" no Colab
   - Adicione comentários no código

2. **Versione resultados**
   - Use `comparison_results_backup_TIMESTAMP`
   - Mantenha histórico de experimentos

---

## ESTRUTURA DE ARQUIVOS FINAL

```
MyDrive/DSL-AG-hybrid/
├── ACDWM/                              ← Clonado automaticamente
│   ├── dwmil.py
│   ├── chunk_size_select.py
│   └── ...
├── comparison_results/                 ← Gerado por experimentos
│   └── experiment_20250107_143022/
│       ├── consolidated_results.csv
│       └── ...
├── comparison_results_backup_*/        ← Backups
├── test_quick_results/                 ← Testes rápidos
├── test_intermediate_results/          ← Testes intermediários
├── baseline_acdwm.py
├── baseline_river.py
├── compare_gbml_vs_river.py
├── config_comparison.yaml
├── data_converters.py
├── shared_evaluation.py
├── gbml_evaluator.py
├── metrics.py
├── run_comparison_colab.py
├── setup_acdwm_colab.ipynb             ← Notebook 1
├── experimento_comparacao_colab.ipynb  ← Notebook 2
├── GUIA_COLAB.md                       ← Este arquivo
└── ... (outros arquivos)
```

---

## PRÓXIMOS PASSOS

### Fase 4: Integração Completa ACDWM

- [ ] Modificar `compare_gbml_vs_river.py` para suportar ACDWM
- [ ] Criar função `run_acdwm_comparison()`
- [ ] Adicionar flag `--include-acdwm`
- [ ] Testar comparação GBML vs River vs ACDWM
- [ ] Atualizar `run_comparison_colab.py`

### Fase 5: Análise de Resultados

- [ ] Gerar tabelas comparativas
- [ ] Criar visualizações avançadas
- [ ] Análise estatística (testes de hipótese)
- [ ] Escrever relatório final

---

## RECURSOS ADICIONAIS

### Google Colab:
- Documentação: https://colab.research.google.com/
- FAQ: https://research.google.com/colaboratory/faq.html
- Colab Pro: https://colab.research.google.com/signup

### Tutoriais:
- Mounting Google Drive: https://colab.research.google.com/notebooks/io.ipynb
- Using GPU/TPU: https://colab.research.google.com/notebooks/gpu.ipynb

### Suporte:
- Issues no GitHub: https://github.com/jasonyanglu/ACDWM/issues
- Documentação River: https://riverml.xyz/
- Documentação DEAP: https://deap.readthedocs.io/

---

**Versão do Guia**: 1.0
**Última Atualização**: 2025-01-07
**Autor**: Claude Code
