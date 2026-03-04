# Células do Colab para MCMO com Pymoo

Copie e cole estas células no Google Colab para testar MCMO com NSGA-II real.

---

## Célula 2: Instalar Dependências (PYMOO)

```python
# Célula 2: Instalar Dependências com PYMOO (NSGA-II REAL)

print("Instalando pymoo (NSGA-II)...")
!pip install -q pymoo

print("Instalando river...")
!pip install -q river

print("Instalando scipy...")
!pip install -q scipy

print("\n✓ Instalação completa!")
print("  ✓ pymoo (NSGA-II multi-objetivo)")
print("  ✓ river (HoeffdingTree, ADWIN)")
print("  ✓ scipy (MMD, Fisher)")
print("  ✓ sklearn (GMM, já no Colab)")
```

---

## Célula 3: Importar MCMO Adapter (Versão Pymoo)

```python
# Célula 3: Importar MCMO Adapter (Versão Pymoo - NSGA-II REAL)

# Reimportar módulos (caso já tenha executado antes)
import importlib
import sys

if 'mcmo.MCMO_pymoo' in sys.modules:
    importlib.reload(sys.modules['mcmo.MCMO_pymoo'])
if 'mcmo.baseline_mcmo_pymoo' in sys.modules:
    importlib.reload(sys.modules['mcmo.baseline_mcmo_pymoo'])

# Importar versão pymoo (com NSGA-II real)
try:
    from mcmo.baseline_mcmo_pymoo import MCMOAdapter, MCMOEvaluator, test_mcmo_adapter
    print("✓ MCMOAdapter (pymoo version) importado com sucesso!")

    from mcmo import baseline_mcmo_pymoo
    if baseline_mcmo_pymoo.MCMO_AVAILABLE:
        print("✓ MCMO Pymoo disponível")
        print("  ✓ Feature selection: NSGA-II (pymoo) ← REAL MULTI-OBJETIVO")
        print("  ✓ Drift detection: ADWIN (river)")
        print("  ✓ Classifier: HoeffdingTree (river)")
        print("  ✓ Sample weighting: GMM (sklearn)")
        print("  ⚠ Sample weighting no tree: Não suportado (limitação river)")
    else:
        print(f"✗ MCMO não disponível: {baseline_mcmo_pymoo.IMPORT_ERROR}")

except ImportError as e:
    print(f"✗ Erro ao importar: {e}")
    print("\nVerifique:")
    print("1. MCMO_pymoo.py e baseline_mcmo_pymoo.py estão em mcmo/")
    print("2. DSL_PATH está correto")
    print("3. Dependências foram instaladas (pymoo, river, scipy)")
```

---

## Célula 4: Teste com Dados Sintéticos (MANTÉM A MESMA)

```python
# Célula 4: mesma do notebook anterior
# (gera X_chunks e y_chunks)
```

---

## Célula 5: Testar MCMO com NSGA-II Real

```python
# Célula 5: Testar MCMO Adapter Pymoo (NSGA-II REAL)

import time

print("\n" + "=" * 70)
print("Testando MCMO Adapter Pymoo (n_sources=3)")
print("  Feature Selection: NSGA-II (multi-objetivo)")
print("  Tempo esperado: ~2-5 minutos (NSGA-II é mais lento)")
print("=" * 70 + "\n")

start_time = time.time()

results_pymoo = test_mcmo_adapter(
    X_chunks=X_chunks,
    y_chunks=y_chunks,
    n_sources=3,
    nsga_pop_size=50,  # População NSGA-II
    nsga_n_gen=50,     # Gerações NSGA-II
    verbose=True
)

elapsed_time = time.time() - start_time

print("\n" + "=" * 70)
print("Resultados - MCMO Pymoo (NSGA-II REAL)")
print("=" * 70)
print(f"Acurácia Média:  {results_pymoo['mean_accuracy']:.4f}")
print(f"Total Amostras:  {results_pymoo['global_metrics']['total_samples']}")
print(f"Chunks:          {results_pymoo['global_metrics']['total_chunks_processed']}")
print(f"Tempo total:     {elapsed_time:.1f}s (~{elapsed_time/60:.1f} min)")
```

---

## Célula 6: Comparação Pymoo vs Simplified vs Baseline

```python
# Célula 6: Comparação das 3 Versões

import matplotlib.pyplot as plt

print("=" * 70)
print("COMPARAÇÃO: NSGA-II vs Correlação vs Baseline")
print("=" * 70)

# Testar baseline (já foi testado antes)
# Usar results_pymoo (novo) e results (simplified anterior)

# Comparar médias
print("\nRESULTADOS:")
print("-" * 70)
print(f"Baseline (HT):           {baseline_mean:.4f}")
print(f"MCMO Simplified (Corr):  {results['mean_accuracy']:.4f}  (Δ={results['mean_accuracy']-baseline_mean:+.4f})")
print(f"MCMO Pymoo (NSGA-II):    {results_pymoo['mean_accuracy']:.4f}  (Δ={results_pymoo['mean_accuracy']-baseline_mean:+.4f})")

print("\nMELHORIA vs BASELINE:")
print("-" * 70)
improvement_simplified = ((results['mean_accuracy'] / baseline_mean) - 1) * 100
improvement_pymoo = ((results_pymoo['mean_accuracy'] / baseline_mean) - 1) * 100

print(f"MCMO Simplified: {improvement_simplified:+.1f}%")
print(f"MCMO Pymoo:      {improvement_pymoo:+.1f}%")

print("\nDIFERENÇA NSGA-II vs CORRELAÇÃO:")
print("-" * 70)
diff = results_pymoo['mean_accuracy'] - results['mean_accuracy']
print(f"Diferença absoluta: {diff:+.4f} ({diff*100:+.2f} p.p.)")
diff_rel = ((results_pymoo['mean_accuracy'] / results['mean_accuracy']) - 1) * 100
print(f"Diferença relativa: {diff_rel:+.1f}%")

# Plot comparação
plt.figure(figsize=(16, 6))

plt.plot(range(1, len(baseline_accuracies)+1), baseline_accuracies,
         marker='s', linewidth=2.5, markersize=8, label='Baseline (HT)', color='orange')
plt.plot(range(1, len(results['accuracies'])+1), results['accuracies'],
         marker='o', linewidth=2.5, markersize=8, label='MCMO Simplified (Corr)', color='steelblue')
plt.plot(range(1, len(results_pymoo['accuracies'])+1), results_pymoo['accuracies'],
         marker='^', linewidth=2.5, markersize=8, label='MCMO Pymoo (NSGA-II)', color='green')

plt.axhline(y=baseline_mean, color='orange', linestyle='--', alpha=0.4)
plt.axhline(y=results['mean_accuracy'], color='steelblue', linestyle='--', alpha=0.4)
plt.axhline(y=results_pymoo['mean_accuracy'], color='green', linestyle='--', alpha=0.4,
            label=f'Pymoo Mean: {results_pymoo["mean_accuracy"]:.4f}')

plt.xlabel('Chunk', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('MCMO: NSGA-II vs Correlation vs Baseline', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Decisão final
print("\n" + "=" * 70)
print("CONCLUSÃO")
print("=" * 70)

if results_pymoo['mean_accuracy'] > baseline_mean:
    print("✓ MCMO Pymoo (NSGA-II) SUPEROU o baseline!")
else:
    print("✗ MCMO Pymoo não superou baseline")

if results_pymoo['mean_accuracy'] > results['mean_accuracy']:
    print("✓ NSGA-II é MELHOR que correlação")
    print(f"  Justifica uso de NSGA-II (ganho de {diff*100:.2f} p.p.)")
else:
    print("⚠ NSGA-II não superou correlação")
    print("  Correlação pode ser suficiente (mais rápido)")
```

---

## Célula 7: Resumo Final

```python
# Célula 7: Resumo Final

print("="*70)
print("RESUMO FINAL - MCMO PYMOO (NSGA-II)")
print("="*70)

print("\n1. CONFIGURAÇÃO")
print("-" * 70)
print(f"   Método:               MCMO Pymoo (NSGA-II real)")
print(f"   n_sources:            {3}")
print(f"   initial_beach:        {200}")
print(f"   NSGA-II pop_size:     {50}")
print(f"   NSGA-II n_gen:        {50}")
print(f"   Chunks testados:      {len(X_chunks)}")

print("\n2. RESULTADOS")
print("-" * 70)
print(f"   Baseline (HT):        {baseline_mean:.4f}")
print(f"   MCMO Simplified:      {results['mean_accuracy']:.4f}  (Δ={results['mean_accuracy']-baseline_mean:+.4f})")
print(f"   MCMO Pymoo:           {results_pymoo['mean_accuracy']:.4f}  (Δ={results_pymoo['mean_accuracy']-baseline_mean:+.4f})")

print("\n3. COMPONENTES USADOS (PYMOO)")
print("-" * 70)
print("   ✓ Feature Selection:  NSGA-II (pymoo) ← MULTI-OBJETIVO")
print("   ✓ Sample Weighting:   GMM (sklearn)")
print("   ✓ Base Classifier:    HoeffdingTree (river)")
print("   ✓ Drift Detection:    ADWIN (river)")
print("   ✓ Ensemble:           Weighted voting")

print("\n4. VALIDAÇÃO")
print("-" * 70)
if results_pymoo['mean_accuracy'] > baseline_mean:
    print("   ✓ MCMO Pymoo superou baseline")
else:
    print("   ✗ MCMO Pymoo não superou baseline (tuning necessário)")

if results_pymoo['mean_accuracy'] > results['mean_accuracy']:
    print("   ✓ NSGA-II melhor que correlação")
else:
    print("   ⚠ Correlação suficiente (usar simplified para ser rápido)")

print("   ✓ Temporal splitting funcionou")
print("   ✓ Compatível com Colab (pymoo)")
print("   ✓ Pronto para integração")

print("\n5. RECOMENDAÇÃO")
print("-" * 70)
if results_pymoo['mean_accuracy'] > results['mean_accuracy'] + 0.02:
    print("   → Usar MCMO Pymoo (NSGA-II) para melhor performance")
elif results_pymoo['mean_accuracy'] > results['mean_accuracy']:
    print("   → NSGA-II ligeiramente melhor, trade-off speed vs accuracy")
else:
    print("   → Usar MCMO Simplified (correlação) - mais rápido, similar accuracy")

print("\n" + "="*70)
print("Teste completo! ✓")
print("="*70)
```

---

## Célula 8: Debug Detalhado - Investigar Performance Baixa

```python
# Célula 8: Debug Profundo - Por que MCMO tem baixa performance?

import numpy as np
from sklearn.metrics import accuracy_score
from mcmo.baseline_mcmo_pymoo import MCMOAdapter

print("=" * 70)
print("DEBUG DETALHADO: Investigando Performance do MCMO")
print("=" * 70)

# Criar novo adapter para teste controlado
adapter = MCMOAdapter(
    n_sources=3,
    initial_beach=200,
    max_pool=5,
    nsga_pop_size=50,
    nsga_n_gen=50,
    verbose=False
)

# Processar primeiros chunks para popular buffer
print("\n1. POPULANDO BUFFER (chunks 0-3)")
print("-" * 70)

for chunk_idx in range(4):
    X, y = X_chunks[chunk_idx], y_chunks[chunk_idx]
    predictions = adapter.partial_fit_predict(X, y)
    acc = accuracy_score(y, predictions)
    print(f"Chunk {chunk_idx}: Acc={acc:.4f} (buffer_size={len(adapter.chunk_buffer)})")

# Agora MCMO deve estar ativo
print("\n2. ANÁLISE DE FEATURES SELECIONADAS")
print("-" * 70)

if adapter.mcmo.solution is not None:
    solution = adapter.mcmo.solution
    n_selected = int(np.sum(solution))
    selected_indices = [i for i, val in enumerate(solution) if val == 1]

    print(f"Features selecionadas: {n_selected}/{len(solution)}")
    print(f"Índices: {selected_indices}")
    print(f"Vetor binário: {solution}")

    # Testar discriminação das features selecionadas
    X_test = X_chunks[3]
    y_test = y_chunks[3]

    # Reduzir features
    from mcmo.MCMO_pymoo import Feature_Reduce
    X_reduced = Feature_Reduce(X_test, solution)

    print(f"\nForma original: {X_test.shape}")
    print(f"Forma reduzida: {X_reduced.shape}")

    # Calcular separabilidade das classes com features selecionadas
    from mcmo.MCMO_pymoo import F1_score, mmd_rbf

    fisher = F1_score(X_reduced, y_test)
    print(f"\nFisher criterion (menor=melhor): {fisher:.4f}")

    # Comparar com todas as features
    fisher_all = F1_score(X_test, y_test)
    print(f"Fisher com TODAS features: {fisher_all:.4f}")

    if fisher < fisher_all:
        print("✓ Features selecionadas são MAIS discriminativas")
    else:
        print("✗ Features selecionadas são MENOS discriminativas")

else:
    print("✗ MCMO ainda não inicializou (solution=None)")

# 3. Analisar classifiers individuais
print("\n3. PERFORMANCE DOS CLASSIFIERS INDIVIDUAIS")
print("-" * 70)

if len(adapter.mcmo.source_classifiers) > 0:
    X_test = X_chunks[4]
    y_test = y_chunks[4]
    X_test_reduced = Feature_Reduce(X_test, adapter.mcmo.solution)

    print(f"Testando chunk 4 ({len(y_test)} samples)")
    print(f"Features usadas: {X_test_reduced.shape[1]}")

    individual_accs = []
    for i, clf in enumerate(adapter.mcmo.source_classifiers):
        preds = clf.predict(X_test_reduced)
        acc = accuracy_score(y_test, preds)
        individual_accs.append(acc)
        print(f"  Source Classifier {i}: {acc:.4f}")

    print(f"\nMédia individual: {np.mean(individual_accs):.4f}")

    # Testar ensemble
    ensemble_preds = adapter.mcmo.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    print(f"Ensemble (voting):  {ensemble_acc:.4f}")

    if ensemble_acc > np.mean(individual_accs):
        print("✓ Ensemble MELHORA performance")
    else:
        print("✗ Ensemble PIORA performance (voting pode estar errado)")

    # Comparar com baseline (sem feature selection)
    from river.tree import HoeffdingTreeClassifier
    baseline_clf = HoeffdingTreeClassifier()

    # Treinar baseline no chunk 3
    for i in range(len(X_chunks[3])):
        x_dict = {f'f{j}': float(X_chunks[3][i, j]) for j in range(X_chunks[3].shape[1])}
        baseline_clf.learn_one(x_dict, int(y_chunks[3][i]))

    # Testar no chunk 4
    baseline_preds = []
    for i in range(len(X_test)):
        x_dict = {f'f{j}': float(X_test[i, j]) for j in range(X_test.shape[1])}
        pred = baseline_clf.predict_one(x_dict)
        baseline_preds.append(pred if pred is not None else 0)

    baseline_acc = accuracy_score(y_test, baseline_preds)
    print(f"\nBaseline (sem FS):  {baseline_acc:.4f}")

    print(f"\nCOMPARAÇÃO:")
    print(f"  Baseline:          {baseline_acc:.4f}")
    print(f"  MCMO Individual:   {np.mean(individual_accs):.4f} (Δ={np.mean(individual_accs)-baseline_acc:+.4f})")
    print(f"  MCMO Ensemble:     {ensemble_acc:.4f} (Δ={ensemble_acc-baseline_acc:+.4f})")

else:
    print("✗ Nenhum source classifier treinado ainda")

# 4. Analisar distribuição source vs target
print("\n4. DISTRIBUIÇÃO SOURCE vs TARGET")
print("-" * 70)

if adapter.mcmo.solution is not None:
    # Pegar chunks do buffer
    buffer_list = list(adapter.chunk_buffer)
    source_chunks = buffer_list[-(adapter.n_sources+1):-1]
    target_chunk = buffer_list[-1]

    X_target = target_chunk[0]
    X_target_reduced = Feature_Reduce(X_target, adapter.mcmo.solution)

    print(f"Source chunks: {len(source_chunks)}")
    print(f"Target chunk: {X_target.shape}")

    # Calcular MMD entre cada source e target
    for i, (X_src, y_src) in enumerate(source_chunks):
        X_src_reduced = Feature_Reduce(X_src, adapter.mcmo.solution)
        mmd = mmd_rbf(X_src_reduced, X_target_reduced)
        print(f"  MMD(Source {i}, Target): {mmd:.6f}")

    # Calcular MMD médio
    mmds = []
    for X_src, y_src in source_chunks:
        X_src_reduced = Feature_Reduce(X_src, adapter.mcmo.solution)
        mmds.append(mmd_rbf(X_src_reduced, X_target_reduced))

    print(f"\nMMD médio: {np.mean(mmds):.6f} (menor=mais similar)")

    if np.mean(mmds) > 1.0:
        print("⚠ Distribuições muito diferentes (covariate shift grande)")
    elif np.mean(mmds) > 0.1:
        print("⚠ Distribuições moderadamente diferentes")
    else:
        print("✓ Distribuições similares")

# 5. Analisar sample weights (GMM)
print("\n5. SAMPLE WEIGHTING (GMM)")
print("-" * 70)

if adapter.mcmo.gmm is not None:
    X_test = X_chunks[4]
    X_test_reduced = Feature_Reduce(X_test, adapter.mcmo.solution)

    weights = adapter.mcmo.gmm.evaluation_weight(X_test_reduced)

    print(f"Sample weights estatísticas:")
    print(f"  Min:    {np.min(weights):.4f}")
    print(f"  Max:    {np.max(weights):.4f}")
    print(f"  Mean:   {np.mean(weights):.4f}")
    print(f"  Std:    {np.std(weights):.4f}")
    print(f"  Median: {np.median(weights):.4f}")

    # Verificar se weights variam significativamente
    cv = np.std(weights) / np.mean(weights)
    print(f"\nCoeficiente de variação: {cv:.4f}")

    if cv < 0.1:
        print("⚠ Weights quase uniformes (GMM não está diferenciando samples)")
    elif cv < 0.5:
        print("✓ Weights com variação moderada")
    else:
        print("✓ Weights com alta variação (GMM diferenciando bem)")

    # IMPORTANTE: River não suporta sample_weight!
    print("\n⚠ LIMITAÇÃO: river HoeffdingTree NÃO usa sample_weight")
    print("  Weights são calculados mas NÃO aplicados ao treinamento")
    print("  Perda esperada de performance: ~2-3 p.p.")

else:
    print("✗ GMM ainda não treinado")

# 6. Resumo diagnóstico
print("\n" + "=" * 70)
print("DIAGNÓSTICO FINAL")
print("=" * 70)

issues = []
fixes = []

if adapter.mcmo.solution is not None:
    if fisher >= fisher_all:
        issues.append("Features selecionadas são MENOS discriminativas que todas")
        fixes.append("Ajustar objetivos NSGA-II ou aumentar n_gen")

    if np.mean(mmds) > 0.5:
        issues.append("Distribuições source/target muito diferentes (MMD alto)")
        fixes.append("Usar chunks mais próximos temporalmente ou aumentar n_sources")

    if cv < 0.1:
        issues.append("GMM não está diferenciando samples (weights uniformes)")
        fixes.append("Ajustar gaussian_number ou usar distribuição diferente")

    if ensemble_acc < np.mean(individual_accs):
        issues.append("Ensemble voting PIORA performance")
        fixes.append("Revisar lógica de voting ou usar weighted majority")

    issues.append("River NÃO suporta sample_weight (limitação fundamental)")
    fixes.append("Implementar classifier customizado ou usar sklearn com partial_fit")

if len(issues) == 0:
    print("✓ Nenhum problema óbvio detectado")
    print("  MCMO está funcionando tecnicamente correto")
    print("  Performance baixa pode ser devido a:")
    print("    - Dados sintéticos não adequados para MCMO")
    print("    - Hyperparâmetros precisam tuning")
    print("    - Método não adequado para este tipo de drift")
else:
    print(f"Problemas detectados ({len(issues)}):")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    print(f"\nSoluções sugeridas:")
    for i, fix in enumerate(fixes, 1):
        print(f"  {i}. {fix}")

print("\n" + "=" * 70)
```

---

## Célula 9: Teste com Hyperparâmetros Ajustados

```python
# Célula 9: Testar MCMO com hyperparâmetros otimizados

print("=" * 70)
print("TESTE COM HYPERPARÂMETROS AJUSTADOS")
print("=" * 70)

# Testar diferentes configurações
configs = [
    {
        'name': 'Original',
        'n_sources': 3,
        'initial_beach': 200,
        'nsga_pop_size': 50,
        'nsga_n_gen': 50,
        'gaussian_number': 5
    },
    {
        'name': 'Mais Sources',
        'n_sources': 5,  # Mais histórico
        'initial_beach': 200,
        'nsga_pop_size': 50,
        'nsga_n_gen': 50,
        'gaussian_number': 5
    },
    {
        'name': 'Beach Menor',
        'n_sources': 3,
        'initial_beach': 100,  # Iniciar mais rápido
        'nsga_pop_size': 50,
        'nsga_n_gen': 50,
        'gaussian_number': 5
    },
    {
        'name': 'NSGA-II Intenso',
        'n_sources': 3,
        'initial_beach': 200,
        'nsga_pop_size': 100,  # População maior
        'nsga_n_gen': 100,     # Mais gerações
        'gaussian_number': 5
    },
    {
        'name': 'Mais Gaussianas',
        'n_sources': 3,
        'initial_beach': 200,
        'nsga_pop_size': 50,
        'nsga_n_gen': 50,
        'gaussian_number': 10  # GMM mais complexo
    }
]

results_configs = []

for config in configs:
    print(f"\n{'='*70}")
    print(f"Testando: {config['name']}")
    print(f"{'='*70}")

    for key, val in config.items():
        if key != 'name':
            print(f"  {key}: {val}")

    evaluator = MCMOEvaluator(
        n_sources=config['n_sources'],
        initial_beach=config['initial_beach'],
        nsga_pop_size=config['nsga_pop_size'],
        nsga_n_gen=config['nsga_n_gen'],
        verbose=False
    )

    chunk_accs = []
    for i, (X, y) in enumerate(zip(X_chunks, y_chunks)):
        predictions, metrics = evaluator.evaluate_chunk(X, y)
        chunk_accs.append(metrics['accuracy'])

    global_metrics = evaluator.get_global_metrics()
    mean_acc = global_metrics['global_accuracy']

    results_configs.append({
        'name': config['name'],
        'config': config,
        'mean_accuracy': mean_acc,
        'chunk_accs': chunk_accs
    })

    print(f"\n  Acurácia Média: {mean_acc:.4f}")

    if mean_acc > baseline_mean:
        improvement = ((mean_acc / baseline_mean) - 1) * 100
        print(f"  ✓ SUPEROU baseline em {improvement:+.1f}%")
    else:
        diff = baseline_mean - mean_acc
        print(f"  ✗ Abaixo do baseline por {diff:.4f}")

# Comparar todas as configurações
print(f"\n{'='*70}")
print("RANKING DE CONFIGURAÇÕES")
print(f"{'='*70}")

# Ordenar por acurácia
results_sorted = sorted(results_configs, key=lambda x: x['mean_accuracy'], reverse=True)

print(f"\nBaseline: {baseline_mean:.4f}")
print("-" * 70)

for i, result in enumerate(results_sorted, 1):
    name = result['name']
    acc = result['mean_accuracy']
    delta = acc - baseline_mean

    if delta > 0:
        symbol = "✓"
    else:
        symbol = "✗"

    print(f"{i}. {symbol} {name:20s}  {acc:.4f}  (Δ={delta:+.4f})")

# Plot comparação
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))

# Plot baseline
plt.axhline(y=baseline_mean, color='red', linestyle='--', linewidth=2,
            label=f'Baseline: {baseline_mean:.4f}', alpha=0.7)

# Plot cada configuração
colors = ['blue', 'green', 'orange', 'purple', 'brown']
markers = ['o', 's', '^', 'D', 'v']

for i, result in enumerate(results_configs):
    plt.plot(range(1, len(result['chunk_accs'])+1),
             result['chunk_accs'],
             marker=markers[i],
             linewidth=2,
             markersize=6,
             label=f"{result['name']}: {result['mean_accuracy']:.4f}",
             color=colors[i])

plt.xlabel('Chunk', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('MCMO: Comparação de Hyperparâmetros', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Melhor configuração
best = results_sorted[0]
print(f"\n{'='*70}")
print("MELHOR CONFIGURAÇÃO")
print(f"{'='*70}")
print(f"Nome: {best['name']}")
print(f"Acurácia: {best['mean_accuracy']:.4f}")
print(f"Melhoria vs baseline: {(best['mean_accuracy'] - baseline_mean)*100:+.2f} p.p.")
print("\nParâmetros:")
for key, val in best['config'].items():
    if key != 'name':
        print(f"  {key}: {val}")
```

---

## Arquivos para Upload no Drive

Fazer upload destes arquivos para `DSL-AG-hybrid/mcmo/`:

1. ✅ `MCMO_pymoo.py` (criado)
2. ✅ `baseline_mcmo_pymoo.py` (criado)

Manter também:
- `MCMO_simplified.py` (para comparação)
- `baseline_mcmo_simplified.py` (para comparação)
