# ANÁLISE: SEEDING 85% E COMPATIBILIDADE COM 6 CHUNKS

**Data**: 2025-10-28
**Pergunta**: Precisamos ajustar seeding trigger antes de testar com 6 chunks?

---

## ✅ RESPOSTA: NÃO É NECESSÁRIO AJUSTAR!

O seeding 85% **já está adaptativo** e **não depende de chunk_index fixo**.

---

## 🔍 ANÁLISE DO CÓDIGO

### Localização: `ga.py` linhas 526-530

```python
# CORREÇÃO: Aumenta seeding intensivo para 85% em drift SEVERE
if drift_severity == 'SEVERE':
    dt_seeding_ratio_on_init_config = 0.85
    dt_rule_injection_ratio_config = 0.90
    logging.info(f"  -> SEVERE DRIFT DETECTED: Seeding INTENSIVO ativado (85% seeding, 90% injection)")
```

### Como Funciona:

1. **Trigger baseado em `drift_severity`**, não em `chunk_index`
2. `drift_severity` é calculado em `main.py` (linhas 613-637) com base na diferença entre conceitos
3. Classificação de severidade (main.py:634-637):
   ```python
   is_severe_drift = False
   if drift_transition_detected and drift_severity_numeric >= 0.25:
       is_severe_drift = True
       logger.warning(f"SEVERE DRIFT detected (severity: {drift_severity_numeric:.2%}) for chunk {i}.")
   ```

4. **Independente do número total de chunks!** ✅

---

## 📊 COMO FUNCIONA COM 6 CHUNKS

### Exemplo: `RBF_Abrupt_Severe` (6 chunks)

```yaml
concept_sequence:
  - {concept_id: 'c1', duration_chunks: 3}      # Chunks 0-2
  - {concept_id: 'c2_severe', duration_chunks: 3}  # Chunks 3-5
```

**Timeline**:
- **Chunk 0-2**: Conceito `c1` (STABLE)
  - `drift_severity = 'STABLE'`
  - Seeding normal (ex: 60-80% dependendo de adaptive profile)

- **Chunk 3**: Transição `c1` → `c2_severe` detectada
  - `drift_severity_numeric` = diferença entre c1 e c2_severe (ex: 0.45 = 45%)
  - Como 0.45 >= 0.25: `is_severe_drift = True`
  - `drift_severity = 'SEVERE'`
  - **Seeding 85% ATIVADO** ✅

- **Chunk 4-5**: Conceito `c2_severe` estável
  - `drift_severity = 'STABLE'` (sem transição)
  - Seeding volta ao normal

### Exemplo: `SEA_Gradual_Simple_Fast` (6 chunks, gradual)

```yaml
concept_sequence:
  - {concept_id: 'f1', duration_chunks: 3}
  - {concept_id: 'f3', duration_chunks: 3}
gradual_drift_width_chunks: 1
```

**Timeline**:
- **Chunk 0-2**: Conceito `f1` (STABLE)
- **Chunk 3**: Transição gradual `f1` → `f3` (janela de 1 chunk)
  - Drift detectado e classificado por severidade
  - Se severidade >= 25%: Seeding 85% ativado
- **Chunk 4-5**: Conceito `f3` estável

---

## ✅ VALIDAÇÃO

### Comportamento Esperado com 6 Chunks:

| Chunk | Conceito | Drift Transition? | Severity | Seeding |
|-------|----------|-------------------|----------|---------|
| 0 | c1 | ❌ | STABLE | Normal (60-80%) |
| 1 | c1 | ❌ | STABLE | Normal |
| 2 | c1 | ❌ | STABLE | Normal |
| 3 | c1→c2 | ✅ | **SEVERE** (45%) | **85%** ✅ |
| 4 | c2 | ❌ | STABLE | Normal |
| 5 | c2 | ❌ | STABLE | Normal |

### Ponto Crítico:

O seeding 85% **só ativa quando há drift SEVERE**, independentemente de qual chunk seja. Pode ser:
- Chunk 1 (drift muito cedo)
- Chunk 3 (meio da stream)
- Chunk 5 (drift tardio)

**Conclusão**: Funciona perfeitamente com 6 chunks! ✅

---

## 🎯 RECOMENDAÇÃO

### Você pode **TESTAR IMEDIATAMENTE** sem ajustes!

1. **Escolher stream de teste**: `RBF_Abrupt_Severe` ou `SEA_Abrupt_Simple`
2. **Executar com config_6chunks.yaml**
3. **Validar logs**:
   ```bash
   grep "SEVERE DRIFT DETECTED: Seeding INTENSIVO" experimento.log
   ```
4. **Validar chunk de ativação**: Deve ativar no chunk onde o drift ocorre (ex: chunk 3 para 2 conceitos iguais)

---

## 📋 COMANDOS PARA TESTE

### 1. Preparar config para teste único:

```bash
cd "C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid"

# Criar config de teste
python -c "
import yaml
with open('config_6chunks.yaml') as f:
    config = yaml.safe_load(f)

# Executar apenas RBF_Abrupt_Severe
config['experiment_settings']['drift_simulation_experiments'] = ['RBF_Abrupt_Severe']
config['experiment_settings']['num_runs'] = 1

with open('config_test_single.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print('Config de teste criado: config_test_single.yaml')
"
```

### 2. Executar teste:

```bash
python main.py
# Certifique-se que main.py carrega o config correto
# Ou edite experiment_settings -> drift_simulation_experiments no config_6chunks.yaml
```

### 3. Validar saída:

```bash
# Verificar chunks gerados
grep "Starting GA for chunk" <log-file>

# Verificar seeding 85%
grep "SEVERE DRIFT DETECTED: Seeding INTENSIVO" <log-file>

# Verificar número de chunks processados
grep "Chunk [0-9]" <log-file> | wc -l  # Deve ser 6
```

---

## 🚨 O QUE OBSERVAR NO LOG

### Logs Esperados:

```
[INFO] Starting GA for chunk 0 (dataset: RBF_Abrupt_Severe)
[INFO] Conceito atual: c1, drift_severity: STABLE
[INFO] Seeding ratio: 0.60 (adaptive profile: SIMPLE/MEDIUM/COMPLEX)

[INFO] Starting GA for chunk 1...
[INFO] Seeding ratio: 0.60

[INFO] Starting GA for chunk 2...
[INFO] Seeding ratio: 0.60

[INFO] Starting GA for chunk 3...
[WARNING] SEVERE DRIFT detected (severity: 45.00%) for chunk 3
[INFO]   -> SEVERE DRIFT DETECTED: Seeding INTENSIVO ativado (85% seeding, 90% injection)  # ✅ AQUI!
[INFO] Parâmetros adaptativos: seeding_ratio=0.85, injection_ratio=0.90

[INFO] Starting GA for chunk 4...
[INFO] Seeding ratio: 0.60  # Volta ao normal (sem drift)

[INFO] Starting GA for chunk 5...
[INFO] Seeding ratio: 0.60
```

---

## ✅ CONCLUSÃO FINAL

**NÃO PRECISA AJUSTAR NADA!**

O código já está:
- ✅ Adaptativo por severidade (não por chunk_index)
- ✅ Compatível com qualquer número de chunks
- ✅ Testado e validado na Fase 2 (8 chunks)
- ✅ Funcionará perfeitamente com 6 chunks

**Pode prosseguir direto para o teste!** 🚀

---

**Próximo passo recomendado**:
1. Testar com `RBF_Abrupt_Severe` (1 run, 6 chunks)
2. Validar seeding 85% ativa no chunk 3
3. Validar performance ≥ 85% G-mean
4. Se tudo OK → executar batch de 3-5 streams
5. Se tudo OK → executar 41 streams completos

---

**Criado por**: Claude Code
**Data**: 2025-10-28
**Status**: ✅ PRONTO PARA TESTE IMEDIATO
