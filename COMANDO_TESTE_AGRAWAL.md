# 🚀 Comando Correto para Teste AGRAWAL

**Data:** 07/10/2025

---

## ❌ Problema Identificado

```bash
# COMANDO ERRADO
python compare_gbml_vs_river.py --stream AGRAWAL_Abrupt_Simple ...
```

**Erro:**
```
ValueError: Stream 'AGRAWAL_Abrupt_Simple' not found in 'experimental_streams'.
```

---

## ✅ Comandos Corretos

### **Opção 1: AGRAWAL Mild (Recomendado - mais fácil)**
```bash
python compare_gbml_vs_river.py --stream AGRAWAL_Abrupt_Simple_Mild --chunks 2 --chunk-size 1000 --no-river
```

**Características:**
- Dataset: AGRAWAL (9 atributos vs 3 do SEA)
- Drift: Abrupt (abrupto)
- Severidade: Mild (suave - drift f1 → f2)
- Instâncias: 2 chunks × 1000 = 2000
- Tempo estimado: **~20-30 min** (50 gerações)

---

### **Opção 2: AGRAWAL Severe (Mais difícil)**
```bash
python compare_gbml_vs_river.py --stream AGRAWAL_Abrupt_Simple_Severe --chunks 2 --chunk-size 1000 --no-river
```

**Características:**
- Dataset: AGRAWAL (9 atributos)
- Drift: Abrupt (abrupto)
- Severidade: Severe (severo - drift f1 → f6)
- Instâncias: 2 chunks × 1000 = 2000
- Tempo estimado: **~20-30 min** (50 gerações)

---

## 📊 Streams AGRAWAL Disponíveis no Config.yaml

| Nome do Stream | Tipo Drift | Severidade | Conceitos |
|----------------|------------|------------|-----------|
| `AGRAWAL_Abrupt_Simple_Mild` | Abrupt | Mild | f1 → f2 |
| `AGRAWAL_Abrupt_Simple_Severe` | Abrupt | Severe | f1 → f6 |
| `AGRAWAL_Gradual_Chain` | Gradual | Medium | f1 → f2 → f3 → f4 |
| `AGRAWAL_Gradual_Recurring` | Gradual | Medium | f1 → f6 → f1 |
| `AGRAWAL_Gradual_Recurring_Noise` | Gradual | Medium + Noise | f1 → f6 → f1 (45% noise) |
| `AGRAWAL_Abrupt_Chain_Long` | Abrupt | Long | f1 → f2 → f3 → f4 → f5 |

---

## 🎯 O Que Observar no Log

### **1. Evolução do Fitness**
```
Gen 1:  BestFit: 1.XXXX (G-mean: 0.XXX)
Gen 10: BestFit: 1.YYYY (G-mean: 0.YYY)  ← Deve AUMENTAR!
Gen 25: BestFit: 1.ZZZZ (G-mean: 0.ZZZ)  ← Deve continuar melhorando
Gen 50: BestFit: 1.WWWW (G-mean: 0.WWW)
```

**Comparar com SEA:**
- SEA: Gen 1 (1.2916) → Gen 200 (1.2916) = ZERO melhora ❌
- AGRAWAL: Se melhorar → Dataset é o problema ✅
- AGRAWAL: Se também estagna → GA é o problema ⚠️

---

### **2. Hill Climbing**
```
Gen 15/50 - BestFit: 1.XXXX (G-mean: 0.XXX)
Estagnação detectada (15 gerações)! Ativando mecanismos de resgate...
  -> Ativando Hill Climbing para refinar o melhor indivíduo...
Gen 16/50 - BestFit: 1.YYYY (G-mean: 0.YYY)  ← DEVE MELHORAR!
```

---

### **3. Accuracy Final**
```
Chunk 0 - Acc: 0.XXXX, F1: 0.YYYY, G-mean: 0.ZZZZ
Chunk 1 - Acc: 0.AAAA, F1: 0.BBBB, G-mean: 0.CCCC

MÉDIA: Acc: X.XX% ± Y.YY%
```

**Comparar:**
- SEA: 94.90% ± 3.82%
- AGRAWAL: ? (esperamos 85-92% por ser mais complexo)

---

## 📝 Informações Técnicas

### **AGRAWAL Dataset:**
- **Atributos:** 9 (vs 3 do SEA)
- **Classes:** 2 (binário)
- **Funções:** f1-f9 (9 conceitos diferentes)
- **Complexidade:** Maior que SEA (não-linear)

### **Config Atual:**
```yaml
max_generations: 50        # Reduzido de 200
population_size: 100
enable_dt_seeding_on_init: true  # Robust Seeding ativo
```

---

## 🔍 Análise Esperada

### **Cenário A: AGRAWAL Evolui** ✅
- Fitness melhora ao longo das gerações
- Hill Climbing funciona
- **Conclusão:** SEA é trivial demais, sistema funciona!

### **Cenário B: AGRAWAL Também Estagna** ⚠️
- Fitness não melhora após Gen 1
- Hill Climbing não funciona
- **Conclusão:** Robust Seeding é muito forte, precisa ajustar

### **Ações Caso B:**
1. Reduzir `dt_seeding_ratio_on_init: 0.3` (de 0.5)
2. Testar com seeding desabilitado (`enable_dt_seeding_on_init: false`)
3. Aumentar mutação (`mutation_rate: 0.25`)

---

## ✅ Comando Final Recomendado

```bash
python compare_gbml_vs_river.py \
    --stream AGRAWAL_Abrupt_Simple_Mild \
    --chunks 2 \
    --chunk-size 1000 \
    --no-river
```

**Tempo:** ~20-30 minutos
**Objetivo:** Validar se evolução funciona em dataset mais complexo

---

**🚀 Execute e envie o log completo para análise!**
