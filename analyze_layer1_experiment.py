"""
Análise detalhada do experimento Layer 1 (5 chunks)
Extrai e analisa métricas de desempenho, cache, early stopping e hill climbing
"""

import re
from datetime import datetime
import statistics

def parse_log_file(filepath):
    """Parse do log completo do experimento"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Estrutura de dados
    chunks = []
    current_chunk = None

    # Regex patterns
    chunk_start = re.compile(r'CHUNK (\d+) - INÍCIO')
    chunk_final = re.compile(r'CHUNK (\d+) - FINAL')
    tempo_total = re.compile(r'Tempo total: ([\d.]+)s \(([\d.]+)min\)')
    train_gmean = re.compile(r'Train G-mean: ([\d.]+)')
    test_gmean = re.compile(r'Test G-mean:\s+([\d.]+)')
    test_f1 = re.compile(r'Test F1:\s+([\d.]+)')
    delta_pattern = re.compile(r'Delta:\s+([-\d.]+)')
    best_fitness = re.compile(r'Best Fitness: ([\d.]+)')

    # Padrões Layer 1
    early_stop_threshold = re.compile(r'\[EARLY STOP\] Gen (\d+): threshold=([\d.]+)')
    early_stop_descartados = re.compile(r'\[EARLY STOP\] Gen (\d+): Descartados=(\d+)/(\d+) \(([\d.]+)%\)')
    cache_hit = re.compile(r'\[CACHE\] Gen (\d+): Hits=(\d+)/(\d+) \(([\d.]+)%\)')
    cache_final = re.compile(r'\[CACHE FINAL\] Hits=(\d+), Misses=(\d+), Hit Rate=([\d.]+)%')
    hc_aplicando = re.compile(r'\[HC\] Aplicando Hill Climbing \(estagnação=(\d+), elite_gmean=([\d.]+)\)')
    hc_pulando = re.compile(r'\[HC\] PULANDO Hill Climbing')
    hc_variantes = re.compile(r'\[HC\] Geradas (\d+) variantes')
    hc_aprovadas = re.compile(r'\[HC\] Aprovadas: (\d+)/(\d+) variantes \(([\d.]+)%\)')

    # Gen pattern
    gen_pattern = re.compile(r'Gen (\d+)/(\d+) - BestFit: ([\d.]+) \(G-mean: ([\d.]+)\)')
    gen_summary = re.compile(r'\[GEN (\d+)\] Best: Fit=([\d.]+), Gmean=([\d.]+)')

    # Fase 2 patterns
    concept_pattern = re.compile(r'CONCEITO RECORRENTE|NOVO CONCEITO')
    similarity_pattern = re.compile(r'concept_\d+: ([\d.]+)')
    severe_drift = re.compile(r'SEVERE DRIFT detected.*drop: ([\d.]+)%')

    lines = content.split('\n')

    for line in lines:
        # Início de chunk
        match = chunk_start.search(line)
        if match:
            chunk_id = int(match.group(1))
            current_chunk = {
                'id': chunk_id,
                'tempo_s': 0,
                'tempo_min': 0,
                'train_gmean': 0,
                'test_gmean': 0,
                'test_f1': 0,
                'delta': 0,
                'best_fitness': 0,
                'early_stop_count': 0,
                'early_stop_avg_pct': [],
                'cache_hits_total': 0,
                'cache_misses_total': 0,
                'cache_hit_rate': 0,
                'hc_aplicado': 0,
                'hc_pulado': 0,
                'hc_variantes_total': 0,
                'hc_aprovadas_total': 0,
                'geracoes': [],
                'concept_recorrente': False,
                'similarity': 0,
                'severe_drift': False,
                'drift_drop': 0
            }
            continue

        # Final de chunk
        match = chunk_final.search(line)
        if match and current_chunk:
            chunks.append(current_chunk)
            current_chunk = None
            continue

        if current_chunk is None:
            continue

        # Tempo total
        match = tempo_total.search(line)
        if match:
            current_chunk['tempo_s'] = float(match.group(1))
            current_chunk['tempo_min'] = float(match.group(2))

        # Métricas finais
        match = train_gmean.search(line)
        if match:
            current_chunk['train_gmean'] = float(match.group(1))

        match = test_gmean.search(line)
        if match:
            current_chunk['test_gmean'] = float(match.group(1))

        match = test_f1.search(line)
        if match:
            current_chunk['test_f1'] = float(match.group(1))

        match = delta_pattern.search(line)
        if match:
            current_chunk['delta'] = float(match.group(1))

        match = best_fitness.search(line)
        if match:
            current_chunk['best_fitness'] = float(match.group(1))

        # Early stop descartados
        match = early_stop_descartados.search(line)
        if match:
            current_chunk['early_stop_count'] += 1
            pct = float(match.group(4))
            current_chunk['early_stop_avg_pct'].append(pct)

        # Cache final
        match = cache_final.search(line)
        if match:
            current_chunk['cache_hits_total'] = int(match.group(1))
            current_chunk['cache_misses_total'] = int(match.group(2))
            current_chunk['cache_hit_rate'] = float(match.group(3))

        # HC aplicado
        match = hc_aplicando.search(line)
        if match:
            current_chunk['hc_aplicado'] += 1

        # HC pulado
        match = hc_pulando.search(line)
        if match:
            current_chunk['hc_pulado'] += 1

        # HC variantes
        match = hc_variantes.search(line)
        if match:
            current_chunk['hc_variantes_total'] += int(match.group(1))

        # HC aprovadas
        match = hc_aprovadas.search(line)
        if match:
            current_chunk['hc_aprovadas_total'] += int(match.group(1))

        # Gerações
        match = gen_pattern.search(line)
        if match:
            gen_num = int(match.group(1))
            gmean = float(match.group(4))
            current_chunk['geracoes'].append({'gen': gen_num, 'gmean': gmean})

        # Conceito recorrente
        if 'CONCEITO RECORRENTE' in line:
            current_chunk['concept_recorrente'] = True

        # Similarity
        match = similarity_pattern.search(line)
        if match:
            current_chunk['similarity'] = float(match.group(1))

        # Severe drift
        match = severe_drift.search(line)
        if match:
            current_chunk['severe_drift'] = True
            current_chunk['drift_drop'] = float(match.group(1))

    return chunks


def analyze_experiment(chunks):
    """Análise completa do experimento"""

    print("=" * 80)
    print("ANÁLISE DETALHADA - EXPERIMENTO LAYER 1 (5 CHUNKS)")
    print("=" * 80)
    print()

    # 1. RESUMO GERAL
    print("1. RESUMO GERAL")
    print("-" * 80)

    total_tempo_h = sum(c['tempo_min'] for c in chunks) / 60
    avg_tempo_min = statistics.mean([c['tempo_min'] for c in chunks])

    print(f"Total de chunks: {len(chunks)}")
    print(f"Tempo total: {total_tempo_h:.2f}h ({sum(c['tempo_min'] for c in chunks):.1f}min)")
    print(f"Tempo médio por chunk: {avg_tempo_min:.1f}min ({avg_tempo_min/60:.2f}h)")
    print()

    # 2. DESEMPENHO (G-MEAN E DELTA)
    print("2. DESEMPENHO (G-MEAN E OVERFITTING)")
    print("-" * 80)

    train_gmeans = [c['train_gmean'] for c in chunks]
    test_gmeans = [c['test_gmean'] for c in chunks]
    deltas = [c['delta'] for c in chunks]

    print(f"Train G-mean: {statistics.mean(train_gmeans):.4f} +/- {statistics.stdev(train_gmeans):.4f}")
    print(f"Test G-mean:  {statistics.mean(test_gmeans):.4f} +/- {statistics.stdev(test_gmeans):.4f}")
    print(f"Delta (Train-Test): {statistics.mean(deltas):.4f} +/- {statistics.stdev(deltas):.4f}")
    print()

    print("Por chunk:")
    for c in chunks:
        overfitting = "OK" if abs(c['delta']) < 0.10 else "ALTO"
        print(f"  Chunk {c['id']}: Train={c['train_gmean']:.3f}, Test={c['test_gmean']:.3f}, Delta={c['delta']:.3f} [{overfitting}]")
    print()

    # 3. CACHE (LAYER 1 FIX 2)
    print("3. CACHE SHA256 (LAYER 1 FIX 2)")
    print("-" * 80)

    cache_hit_rates = [c['cache_hit_rate'] for c in chunks if c['cache_hit_rate'] > 0]

    if cache_hit_rates:
        print(f"Hit rate médio: {statistics.mean(cache_hit_rates):.1f}%")
        print(f"Hit rate range: {min(cache_hit_rates):.1f}% - {max(cache_hit_rates):.1f}%")
        print()

        print("Por chunk:")
        for c in chunks:
            total_ops = c['cache_hits_total'] + c['cache_misses_total']
            if total_ops > 0:
                status = "EXCELENTE" if c['cache_hit_rate'] > 40 else "BOM" if c['cache_hit_rate'] > 30 else "REGULAR"
                print(f"  Chunk {c['id']}: {c['cache_hit_rate']:.1f}% ({c['cache_hits_total']}/{total_ops} ops) [{status}]")
        print()
    else:
        print("PROBLEMA: Nenhum log de cache encontrado!")
        print()

    # 4. EARLY STOPPING (LAYER 1 FIX 1)
    print("4. EARLY STOPPING (LAYER 1 FIX 1)")
    print("-" * 80)

    print("Por chunk:")
    for c in chunks:
        if c['early_stop_avg_pct']:
            avg_pct = statistics.mean(c['early_stop_avg_pct'])
            count = c['early_stop_count']
            status = "EXCELENTE" if avg_pct > 30 else "BOM" if avg_pct > 20 else "REGULAR"
            print(f"  Chunk {c['id']}: {avg_pct:.1f}% descartados (apareceu {count}x no log) [{status}]")
        else:
            print(f"  Chunk {c['id']}: SEM DADOS (não descartou ou logging invisível)")
    print()

    # 5. HILL CLIMBING (LAYER 1 FIX 3)
    print("5. HILL CLIMBING SELETIVO (LAYER 1 FIX 3)")
    print("-" * 80)

    print("Por chunk:")
    for c in chunks:
        total_hc = c['hc_aplicado'] + c['hc_pulado']
        if total_hc > 0:
            pct_pulado = (c['hc_pulado'] / total_hc) * 100
            approval_rate = (c['hc_aprovadas_total'] / c['hc_variantes_total'] * 100) if c['hc_variantes_total'] > 0 else 0
            print(f"  Chunk {c['id']}: Aplicado={c['hc_aplicado']}, Pulado={c['hc_pulado']} ({pct_pulado:.0f}% economia)")
            print(f"            Variantes geradas={c['hc_variantes_total']}, Aprovadas={c['hc_aprovadas_total']} ({approval_rate:.1f}%)")
        else:
            print(f"  Chunk {c['id']}: Nenhum HC (sem estagnação suficiente)")
    print()

    # 6. FASE 2 (CONCEPT FINGERPRINTING)
    print("6. FASE 2 - CONCEPT FINGERPRINTING")
    print("-" * 80)

    print("Por chunk:")
    for c in chunks:
        if c['concept_recorrente']:
            print(f"  Chunk {c['id']}: CONCEITO RECORRENTE (similarity={c['similarity']:.4f})")
        else:
            print(f"  Chunk {c['id']}: NOVO CONCEITO")

        if c['severe_drift']:
            print(f"            SEVERE DRIFT detectado (drop={c['drift_drop']:.1f}%)")
    print()

    # 7. EVOLUÇÃO DAS GERAÇÕES
    print("7. EVOLUÇÃO DO G-MEAN POR CHUNK")
    print("-" * 80)

    for c in chunks:
        if c['geracoes']:
            first_gen = c['geracoes'][0]['gmean']
            last_gen = c['geracoes'][-1]['gmean']
            improvement = last_gen - first_gen
            print(f"  Chunk {c['id']}: Gen 1={first_gen:.3f} -> Gen final={last_gen:.3f} (melhoria={improvement:+.3f})")
    print()

    # 8. COMPARAÇÃO COM BASELINE (RUN3)
    print("8. COMPARAÇÃO COM BASELINE (RUN3 - SEM LAYER 1)")
    print("-" * 80)

    # Dados Run3 (do histórico)
    run3_tempo_h = 9.9  # TEST_SINGLE 5 chunks
    run3_gmean = 0.7763

    run4_tempo_h = total_tempo_h
    run4_gmean = statistics.mean(test_gmeans)

    tempo_reduction = ((run3_tempo_h - run4_tempo_h) / run3_tempo_h) * 100
    gmean_diff = run4_gmean - run3_gmean

    print(f"Run3 (baseline): {run3_tempo_h:.1f}h, G-mean={run3_gmean:.4f}")
    print(f"Run4 (Layer 1):  {run4_tempo_h:.1f}h, G-mean={run4_gmean:.4f}")
    print()
    print(f"Redução de tempo: {tempo_reduction:+.1f}% (meta: -40-55%)")
    print(f"Delta G-mean: {gmean_diff:+.4f} (meta: neutro ou leve melhora)")
    print()

    if tempo_reduction < -30:
        print("STATUS: TEMPO PIOR QUE BASELINE (PROBLEMA!)")
    elif tempo_reduction < -10:
        print("STATUS: TEMPO LEVEMENTE PIOR")
    elif tempo_reduction > 30:
        print("STATUS: TEMPO EXCELENTE (META ALCANÇADA!)")
    elif tempo_reduction > 15:
        print("STATUS: TEMPO BOM (ABAIXO DA META)")
    else:
        print("STATUS: TEMPO REGULAR")
    print()

    # 9. ANÁLISE DE CHUNK PROBLEMÁTICO
    print("9. CHUNK PROBLEMÁTICO (CHUNK 2)")
    print("-" * 80)

    chunk2 = chunks[2]
    print(f"Test G-mean: {chunk2['test_gmean']:.4f} (MUITO BAIXO)")
    print(f"Delta: {chunk2['delta']:.4f} (overfitting SEVERO)")
    print(f"Tempo: {chunk2['tempo_min']:.1f}min ({chunk2['tempo_min']/60:.2f}h)")
    print()
    print("Diagnóstico:")
    print("  - Severe drift detectado no chunk seguinte (chunk 3)")
    print("  - Train G-mean alto (0.9405) mas test muito baixo (0.4389)")
    print("  - Overfitting extremo (delta = -0.50)")
    print("  - Chunk 3 ativou contramedidas (SEVERE DRIFT countermeasures)")
    print()

    # 10. SUMÁRIO EXECUTIVO
    print("=" * 80)
    print("10. SUMÁRIO EXECUTIVO")
    print("=" * 80)

    print()
    print("LAYER 1 - OTIMIZAÇÕES:")
    print()

    # Fix 1: Early Stop
    if any(c['early_stop_avg_pct'] for c in chunks):
        avg_early_stop = statistics.mean([statistics.mean(c['early_stop_avg_pct']) for c in chunks if c['early_stop_avg_pct']])
        print(f"  Fix 1 (Early Stop): FUNCIONANDO ({avg_early_stop:.1f}% descartados em média)")
    else:
        print(f"  Fix 1 (Early Stop): SEM DADOS (logging invisível ou não ativado)")

    # Fix 2: Cache
    if cache_hit_rates:
        print(f"  Fix 2 (Cache SHA256): FUNCIONANDO ({statistics.mean(cache_hit_rates):.1f}% hit rate)")
    else:
        print(f"  Fix 2 (Cache SHA256): SEM DADOS")

    # Fix 3: HC
    total_hc_aplicado = sum(c['hc_aplicado'] for c in chunks)
    total_hc_pulado = sum(c['hc_pulado'] for c in chunks)
    if total_hc_aplicado + total_hc_pulado > 0:
        hc_pct_pulado = (total_hc_pulado / (total_hc_aplicado + total_hc_pulado)) * 100
        print(f"  Fix 3 (HC Seletivo): FUNCIONANDO ({hc_pct_pulado:.0f}% economia)")
    else:
        print(f"  Fix 3 (HC Seletivo): SEM DADOS")

    print()
    print("DESEMPENHO:")
    print()
    print(f"  Tempo: {run4_tempo_h:.2f}h (Run3: {run3_tempo_h:.1f}h, delta: {tempo_reduction:+.1f}%)")
    print(f"  G-mean: {run4_gmean:.4f} (Run3: {run3_gmean:.4f}, delta: {gmean_diff:+.4f})")
    print()

    if tempo_reduction > 20:
        print("  RESULTADO: SUCESSO PARCIAL (redução de tempo significativa)")
    elif tempo_reduction > 0:
        print("  RESULTADO: MELHORA LEVE (abaixo da meta)")
    else:
        print("  RESULTADO: TEMPO PIOROU (investigar)")

    print()
    print("PROBLEMAS IDENTIFICADOS:")
    print()
    print("  1. Chunk 2: Overfitting extremo (delta=-0.50)")
    print("  2. Test G-mean chunk 2: 0.4389 (muito baixo)")
    print("  3. Severe drift detectado após chunk 2")
    print("  4. Tempo ainda maior que Run3 (esperava-se redução)")
    print()
    print("RECOMENDAÇÕES:")
    print()
    print("  1. Investigar por que chunk 2 teve overfitting severo")
    print("  2. Verificar se early stopping está sendo agressivo demais")
    print("  3. Analisar threshold adaptativo na fase 2")
    print("  4. Considerar implementar Layer 2 (penalizar overfitting)")
    print()

    print("=" * 80)


def main():
    filepath = "C:\\Users\\Leandro Almeida\\Downloads\\DSL-AG-hybrid\\novo_experimento.txt"

    print("Parseando log...")
    chunks = parse_log_file(filepath)

    print(f"Encontrados {len(chunks)} chunks")
    print()

    analyze_experiment(chunks)

    # Salvar resultados
    output_file = "C:\\Users\\Leandro Almeida\\Downloads\\DSL-AG-hybrid\\ANALISE_LAYER1_RUN4.txt"

    import sys
    original_stdout = sys.stdout

    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        analyze_experiment(chunks)

    sys.stdout = original_stdout

    print()
    print(f"Análise completa salva em: {output_file}")


if __name__ == "__main__":
    main()
