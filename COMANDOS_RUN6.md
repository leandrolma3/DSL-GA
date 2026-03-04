# COMANDOS PARA EXECUTAR RUN6

**Objetivo:** Executar experimento completo com Layer 1 Paralelo (5 chunks)

---

## GOOGLE COLAB (LINUX/BASH)

### Comando Correto para Colab

```bash
cd /content/drive/MyDrive/DSL-AG-hybrid

python main.py config_test_single.yaml --num_chunks 5 --run_number 6 2>&1 | tee experimento_run6.log
```

**Observacao:** Use `tee` (minusculo) no Linux, nao `Tee-Object`

---

### Alternativa: Redirecionar apenas para arquivo

```bash
cd /content/drive/MyDrive/DSL-AG-hybrid

python main.py config_test_single.yaml --num_chunks 5 --run_number 6 > experimento_run6.log 2>&1
```

**Diferenca:**
- `tee`: Exibe no terminal E salva em arquivo
- `>`: Apenas salva em arquivo (nao exibe no terminal)

---

### Monitorar progresso enquanto executa

```bash
# Terminal 1: Executar experimento
cd /content/drive/MyDrive/DSL-AG-hybrid
python main.py config_test_single.yaml --num_chunks 5 --run_number 6 > experimento_run6.log 2>&1 &

# Terminal 2: Monitorar log em tempo real
tail -f /content/drive/MyDrive/DSL-AG-hybrid/experimento_run6.log
```

---

## WINDOWS (POWERSHELL)

### Comando para Windows Local

```powershell
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"

python main.py config_test_single.yaml --num_chunks 5 --run_number 6 2>&1 | Tee-Object -FilePath "experimento_run6.log"
```

---

## PARAMETROS DO EXPERIMENTO

### Config Atual (config_test_single.yaml)

```yaml
genetic_algorithm:
  population_size: 100
  num_generations: 25
  elitism_rate: 0.1
  use_parallel: true  # IMPORTANTE: Modo paralelo ativado
```

### Parametros da Linha de Comando

- `--num_chunks 5`: Executar 5 chunks
- `--run_number 6`: Identificador do experimento

---

## TEMPO ESTIMADO

**Baseado em Run997 (77.8min por chunk):**
- 5 chunks x 77.8min = 389min
- **Tempo total estimado: 6.5h**

**Comparacao:**
- Run5 (Layer1 quebrado): 12.9h
- Run3 (Baseline): 9.9h
- Run6 (Layer1 funcionando): ~6.5h (estimado)

---

## VERIFICACAO DURANTE EXECUCAO

### Ver progresso do experimento

```bash
# Ver ultimas linhas do log
tail -n 50 experimento_run6.log

# Ver chunks completados
grep "CHUNK.*FINAL" experimento_run6.log

# Ver metricas de cache
grep "\[CACHE\] Gen" experimento_run6.log | tail -n 10

# Ver metricas de early stop
grep "\[EARLY STOP\] Gen" experimento_run6.log | tail -n 10
```

---

## APOS CONCLUSAO

### Extrair metricas principais

```bash
# Resumo dos chunks
grep -E "CHUNK [0-9]+ - FINAL|Tempo total:|Train G-mean:|Test G-mean:" experimento_run6.log

# Resumo final
grep -E "Avg Test G-mean:|Std Test G-mean:|Tempo total:" experimento_run6.log | tail -n 5
```

### Analisar resultados

```bash
# Copiar log para maquina local (se necessario)
# Colab -> Google Drive (ja esta salvo)
# Depois baixar do Drive

# Executar analise
cd /content/drive/MyDrive/DSL-AG-hybrid
python analyze_run6.py  # Script de analise (criar depois)
```

---

## TROUBLESHOOTING

### Problema: Log muito grande (truncado)

**Solucao:** Desativar debug logging antes de Run6

```python
# ga.py linha 31
DEBUG_LAYER1 = False  # Desativar debug detalhado

# fitness.py linha 10
DEBUG_LAYER1_FITNESS = False  # Desativar debug detalhado
```

**Logs que serao mantidos:**
- [CACHE] Gen X: Hits=...
- [EARLY STOP] Gen X: Descartados=...
- CHUNK X - FINAL
- Metricas de tempo e G-mean

**Logs que serao removidos:**
- [DEBUG L1] Gen X: Avaliando...
- [DEBUG L1] Hash gerado...
- [DEBUG L1] CACHE HIT #...
- [DEBUG L1 FITNESS] Early stop check...

---

### Problema: Experimento muito lento

**Diagnostico:**

```bash
# Ver tempo por geracao
grep "Gen [0-9]*/25.*Time:" experimento_run6.log | tail -n 10

# Ver se cache esta funcionando
grep "\[CACHE\] Gen" experimento_run6.log | tail -n 5

# Ver se early stop esta funcionando
grep "\[EARLY STOP\] Gen" experimento_run6.log | tail -n 5
```

**Se cache ou early stop nao aparecerem:**
- Verificar se use_parallel=true no config
- Verificar se codigo foi modificado corretamente

---

## COMANDO FINAL RECOMENDADO (COLAB)

```bash
# 1. Ir para diretorio
cd /content/drive/MyDrive/DSL-AG-hybrid

# 2. (OPCIONAL) Desativar debug logging para evitar truncamento
# Editar ga.py linha 31: DEBUG_LAYER1 = False
# Editar fitness.py linha 10: DEBUG_LAYER1_FITNESS = False

# 3. Executar experimento
python main.py config_test_single.yaml --num_chunks 5 --run_number 6 2>&1 | tee experimento_run6.log

# 4. Aguardar conclusao (~6.5h estimado)

# 5. Analisar resultados
grep -E "CHUNK [0-9]+ - FINAL|Tempo total:|Test G-mean:" experimento_run6.log
```

---

## METRICAS DE SUCESSO (RUN6)

Experimento bem-sucedido se:

1. **Tempo total < 7.5h** (vs 12.9h Run5)
2. **Tempo/chunk < 90min** (vs 154.3min Run5)
3. **Test G-mean >= 0.775** (vs 0.7852 Run5)
4. **Cache hit rate >= 8%** (geracoes 2+)
5. **Early stop rate >= 5%** (geracoes 2+)
6. **Logs de cache aparecem** em todas as geracoes
7. **Logs de early stop aparecem** em geracoes 2+

---

**Status:** PRONTO PARA EXECUCAO
**Plataforma:** Google Colab (Linux/Bash)
**Comando:** `python main.py config_test_single.yaml --num_chunks 5 --run_number 6 2>&1 | tee experimento_run6.log`
