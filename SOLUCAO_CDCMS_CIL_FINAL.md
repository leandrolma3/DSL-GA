# Solucao Final para CDCMS.CIL

**Data:** 2026-01-24
**Status:** Solucao identificada

---

## Problema Identificado

O CDCMS.CIL nao estava funcionando devido a **conflitos de classpath**:

1. **53+ JARs fragmentados** no classpath causando conflitos
2. **cdcms_cil.jar continha classes MOA duplicadas** que conflitavam com moa.jar
3. **AutoClassDiscovery** do CDCMS.CIL falhava silenciosamente ao tentar encontrar clusterers Weka

---

## Solucao: Usar MOA-dependencies.jar do ROSE

### Insight Principal

O projeto ROSE (https://github.com/canoalberto/ROSE) funciona perfeitamente usando apenas **2 JARs**:
- ROSE-1.0.jar (115 KB) - apenas o codigo ROSE
- MOA-dependencies.jar (64.6 MB) - UBER JAR com MOA + Weka + todas dependencias bundled

### Aplicacao ao CDCMS.CIL

Usar a mesma estrategia:
1. **cdcms_cil_clean.jar** - JAR limpo com apenas classes CDCMS (sem MOA/Weka duplicadas)
2. **MOA-dependencies.jar** - UBER JAR do ROSE com tudo bundled

---

## Arquivos Criados

### 1. CELULA_CDCMS_USANDO_ROSE_DEPS.py
Celula principal que:
- Baixa MOA-dependencies.jar do ROSE se necessario
- Usa classpath minimo: cdcms_cil.jar + MOA-dependencies.jar
- Segue exatamente o formato de comando do ROSE
- Inclui sizeofag como javaagent

### 2. CELULA_DIAGNOSTICO_CONFLITOS_JAR.py
Celula de diagnostico que:
- Analisa classes dentro do cdcms_cil.jar
- Identifica classes MOA/Weka duplicadas
- Cria cdcms_cil_clean.jar (versao sem conflitos)
- Testa a versao limpa

---

## Passos para Executar no Colab

### 1. Executar celulas de setup (1.1 a 1.4)
Instalar Java, montar Drive, etc.

### 2. Executar celula de download do ROSE JARs
```python
# Baixar MOA-dependencies.jar do ROSE
import urllib.request
from pathlib import Path

rose_dir = Path('rose_jars')
rose_dir.mkdir(exist_ok=True)

jars = ["MOA-dependencies.jar", "sizeofag-1.0.4.jar"]
for jar in jars:
    jar_path = rose_dir / jar
    if not jar_path.exists():
        url = f"https://github.com/canoalberto/ROSE/raw/master/{jar}"
        urllib.request.urlretrieve(url, jar_path)
        print(f"Baixado: {jar}")
```

### 3. Executar celula de compilacao do CDCMS.CIL
Compilar cdcms_cil.jar do codigo fonte.

### 4. Executar CELULA_DIAGNOSTICO_CONFLITOS_JAR.py
Criar cdcms_cil_clean.jar sem classes duplicadas.

### 5. Executar CELULA_CDCMS_USANDO_ROSE_DEPS.py
Executar CDCMS.CIL com classpath limpo.

---

## Formato do Comando

```bash
java -Xmx4g \
  -javaagent:rose_jars/sizeofag-1.0.4.jar \
  -cp cdcms_jars/cdcms_cil_clean.jar:rose_jars/MOA-dependencies.jar \
  moa.DoTask \
  "EvaluateInterleavedTestThenTrain \
   -s (ArffFileStream -f dados.arff) \
   -l (moa.classifiers.meta.CDCMS_CIL_GMean -s 10 -t 500 -c 2) \
   -f 500 \
   -d results.csv"
```

### Diferencas Chave do Formato

| Aspecto | Antes (nao funcionava) | Depois (deve funcionar) |
|---------|------------------------|-------------------------|
| JARs | 53+ fragmentados | 2 apenas |
| MOA | moa.jar + lib/*.jar | MOA-dependencies.jar |
| cdcms | com classes duplicadas | cdcms_cil_clean.jar |
| sizeofag | nao usado | usado como javaagent |
| Task | argumentos separados | string unica |

---

## Verificacao de Sucesso

Apos executar, verificar:

1. **Arquivo de saida existe:**
   ```python
   if output_file.exists() and output_file.stat().st_size > 0:
       print("SUCESSO!")
   ```

2. **Return code = 0:**
   ```python
   if result.returncode == 0:
       print("Execucao OK!")
   ```

3. **Metricas validas no CSV:**
   ```python
   df = pd.read_csv(output_file)
   print(df.columns)
   print(df['G-Mean'].mean())
   ```

---

## Troubleshooting

### Erro: ClassNotFoundException
- Verificar se cdcms_cil_clean.jar contem moa/classifiers/meta/CDCMS_CIL_GMean.class
- Verificar se MOA-dependencies.jar foi baixado completamente (64.6 MB)

### Erro: NoClassDefFoundError weka/clusterers/EM
- MOA-dependencies.jar deve conter todas as classes Weka
- Verificar: `jar tf MOA-dependencies.jar | grep weka/clusterers/EM`

### Erro: Task not recognized
- Usar formato de task como string unica entre aspas
- Garantir que nao ha espacos extras

---

## Referencias

- **ROSE:** https://github.com/canoalberto/ROSE
- **CDCMS.CIL:** https://github.com/michaelchiucw/CDCMS.CIL
- **MOA:** https://moa.cms.waikato.ac.nz/

---

**Criado por:** Claude Code
**Baseado em:** Analise do Execute_All_Comparative_Models.ipynb e rose_wrapper.py
