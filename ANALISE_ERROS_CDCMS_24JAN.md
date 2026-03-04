# Analise Detalhada dos Erros CDCMS.CIL - 24/01/2026

## Contexto

Apos multiplas tentativas de fazer o CDCMS.CIL funcionar, incluindo:
- Limpeza do JAR (removendo classes MOA/Weka duplicadas)
- Uso do MOA-dependencies.jar do ROSE
- Diferentes formatos de comando
- Testes com shell=True vs shell=False
- Criacao de JAR minimal com apenas classes CDCMS

O sistema ainda falha com os mesmos erros.

---

## Descoberta Critica

### ROSE funciona, CDCMS nao funciona

Usando o MESMO formato de comando e o MESMO MOA-dependencies.jar:

| Teste | Resultado |
|-------|-----------|
| NaiveBayes + MOA-deps | **FUNCIONA** (2158 bytes) |
| ROSE + MOA-deps | **FUNCIONA** |
| CDCMS_CIL_GMean + MOA-deps | **FALHA** |
| CDCMS minimal + MOA-deps | **FALHA** |

Isso prova que o problema NAO esta no:
- Formato do comando
- MOA-dependencies.jar
- Ambiente Java/Colab

O problema ESTA nas classes CDCMS.CIL em si.

---

## Erro Principal

```
java.lang.Exception: Class named 'EvaluateInterleavedTestThenTrain' is not an instance of moa.tasks.meta.MetaMainTask
    at moa.DoTask.main(DoTask.java:96)
```

### Erro Secundario

```
java.lang.Exception: Class not found: WriteCommandLineTemplate
    at moa.options.ClassOption.getClass(ClassOption.java:82)
```

---

## Causa Raiz Identificada

O CDCMS.CIL foi compilado contra uma **versao diferente do MOA** do que a presente no MOA-dependencies.jar do ROSE.

### Evidencias:

1. **ROSE funciona**: O ROSE-1.0.jar foi compilado **junto** com o MOA-dependencies.jar, garantindo compatibilidade
2. **ERulesD2S usa MOA 2020.07.1**: Conforme pom.xml do ERulesD2S
3. **CDCMS provavelmente usa versao diferente**: O cdcms_cil.jar foi compilado contra outra versao

### Consequencia Tecnica:

Quando o classloader Java carrega as classes CDCMS, elas referenciam interfaces/classes MOA de uma versao diferente. Isso causa:
- Conflitos de tipo em runtime
- Falha no mecanismo de descoberta de Tasks do MOA
- O MOA nao reconhece EvaluateInterleavedTestThenTrain como Task valida

---

## Solucoes

### Solucao 1: Verificar e Baixar MOA Correto (CELULA_VERIFICAR_CDCMS_POM.py)

Esta celula:
1. Baixa pom.xml do repositorio CDCMS.CIL
2. Identifica a versao exata do MOA usada
3. Baixa o MOA correto do Maven Central
4. Testa a combinacao CDCMS + MOA correto

### Solucao 2: Recompilar CDCMS do Fonte (CELULA_SOLUCAO_DEFINITIVA_CDCMS.py)

Esta celula:
1. Clona o repositorio CDCMS.CIL
2. Compila as classes contra o MOA-dependencies.jar
3. Cria novo JAR compativel
4. Testa a execucao

### Solucao 3: Contatar Autores

Se as solucoes automaticas falharem:
- Abrir issue no GitHub: https://github.com/michaelchiucw/CDCMS.CIL/issues
- Perguntar qual ambiente/versao MOA eles usam
- Pedir JAR pre-compilado funcional

---

## Ordem de Execucao Recomendada

1. **Execute primeiro**: `CELULA_VERIFICAR_CDCMS_POM.py`
   - Identifica versao MOA necessaria
   - Tenta baixar e testar com versao correta

2. **Se falhar, execute**: `CELULA_SOLUCAO_DEFINITIVA_CDCMS.py`
   - Recompila do fonte contra MOA-dependencies.jar
   - Garante compatibilidade binaria

3. **Se ambos falharem**: Contatar autores

---

## Arquivos Criados

| Arquivo | Funcao |
|---------|--------|
| CELULA_VERIFICAR_CDCMS_POM.py | Identifica versao MOA e tenta baixar correta |
| CELULA_SOLUCAO_DEFINITIVA_CDCMS.py | Recompila CDCMS do fonte |
| CELULA_DIAGNOSTICO_MOA_DEPS.py | Diagnostico do MOA-dependencies.jar |
| CELULA_ALTERNATIVA_MOA_OFICIAL.py | Tenta usar MOA oficial do Waikato |

---

## Comparacao com ROSE

| Aspecto | ROSE | CDCMS |
|---------|------|-------|
| JAR tamanho | 112.6 KB | 2.1 MB (cheio de duplicatas) |
| Classes proprias | ~50 | ~25 |
| Dependencias bundled | Nenhuma | 1232 MOA + 5 Weka |
| Compatibilidade | Compilado junto com MOA-deps | Compilado separadamente |
| Resultado | **FUNCIONA** | **NAO FUNCIONA** |

---

## Resumo Tecnico

O cdcms_cil.jar contem:
- Classes CDCMS proprias: ~25
- Classes MOA duplicadas: 1232
- Classes Weka duplicadas: 5

Mesmo apos criar `cdcms_cil_minimal.jar` com apenas as 25 classes CDCMS, o problema persiste porque essas classes foram compiladas com `.class` files que referenciam uma API MOA diferente.

A unica solucao definitiva e **recompilar** ou **usar o MOA da versao correta**.

---

**Criado por:** Claude Code
**Data:** 2026-01-24
**Status:** Duas celulas de solucao criadas para teste
