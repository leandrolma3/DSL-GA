"""
Analise do Run997 (Smoke Test Layer 1 Paralelo)
Compara com Run5 para verificar melhorias
"""

import re
import statistics

def parse_run997_log(filepath):
    """Parse do log truncado do Run997"""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = []
    current_chunk = None

    # Patterns
    chunk_start = re.compile(r'CHUNK (\d+) - INÍCIO')
    chunk_final = re.compile(r'CHUNK (\d+) - FINAL')
    tempo_total = re.compile(r'Tempo total: ([\d.]+)s \(([\d.]+)min\)')
    train_gmean = re.compile(r'Train G-mean: ([\d.]+)')
    test_gmean = re.compile(r'Test G-mean:\s+([\d.]+)')

    # Cache patterns
    cache_gen = re.compile(r'\[CACHE\] Gen (\d+): Hits=(\d+)/(\d+) \(([\d.]+)%\)')
    early_stop_gen = re.compile(r'\[EARLY STOP\] Gen (\d+): Descartados=(\d+)/(\d+) \(([\d.]+)%\)')
    early_stop_threshold = re.compile(r'\[EARLY STOP\] Gen (\d+): threshold=([\d.]+)')

    # Debug patterns
    debug_cache_hit = re.compile(r'\[DEBUG L1\].*CACHE HIT')
    debug_early_stopped = re.compile(r'\[DEBUG L1\].*EARLY STOPPED individual')

    lines = content.split('\n')
    chunk_being_finalized = None

    # Metricas globais
    cache_logs = []
    early_stop_logs = []

    for line in lines:
        match = chunk_start.search(line)
        if match:
            chunk_id = int(match.group(1))
            current_chunk = {
                'id': chunk_id,
                'tempo_s': 0,
                'tempo_min': 0,
                'train_gmean': 0,
                'test_gmean': 0,
                'cache_logs': [],
                'early_stop_logs': [],
                'early_stop_thresholds': []
            }
            continue

        match = chunk_final.search(line)
        if match and current_chunk:
            chunk_being_finalized = current_chunk
            current_chunk = None
            continue

        if chunk_being_finalized:
            match = tempo_total.search(line)
            if match:
                chunk_being_finalized['tempo_s'] = float(match.group(1))
                chunk_being_finalized['tempo_min'] = float(match.group(2))

            match = train_gmean.search(line)
            if match:
                chunk_being_finalized['train_gmean'] = float(match.group(1))

            match = test_gmean.search(line)
            if match:
                chunk_being_finalized['test_gmean'] = float(match.group(1))
                chunks.append(chunk_being_finalized)
                chunk_being_finalized = None

        # Extrair metricas de cache e early stop (globais)
        match = cache_gen.search(line)
        if match:
            cache_logs.append({
                'gen': int(match.group(1)),
                'hits': int(match.group(2)),
                'total': int(match.group(3)),
                'hit_rate': float(match.group(4))
            })

        match = early_stop_gen.search(line)
        if match:
            early_stop_logs.append({
                'gen': int(match.group(1)),
                'descartados': int(match.group(2)),
                'total': int(match.group(3)),
                'pct': float(match.group(4))
            })

        match = early_stop_threshold.search(line)
        if match:
            if current_chunk:
                current_chunk['early_stop_thresholds'].append(float(match.group(2)))

    return chunks, cache_logs, early_stop_logs


def analyze_run997():
    """Analise completa do Run997"""

    filepath = "C:\\Users\\Leandro Almeida\\Downloads\\DSL-AG-hybrid\\novo_experimento3.txt"

    print("=" * 80)
    print("ANALISE RUN997 - SMOKE TEST LAYER 1 PARALELO")
    print("=" * 80)
    print()

    chunks, cache_logs, early_stop_logs = parse_run997_log(filepath)

    # 1. RESUMO GERAL
    print("1. RESUMO GERAL")
    print("-" * 80)

    if chunks:
        total_tempo_h = sum(c['tempo_min'] for c in chunks) / 60
        avg_tempo_min = statistics.mean([c['tempo_min'] for c in chunks])

        train_gmeans = [c['train_gmean'] for c in chunks if c['train_gmean'] > 0]
        test_gmeans = [c['test_gmean'] for c in chunks if c['test_gmean'] > 0]

        print(f"Chunks completados: {len(chunks)}")
        print(f"Tempo total: {total_tempo_h:.2f}h ({sum(c['tempo_min'] for c in chunks):.1f}min)")
        print(f"Tempo medio por chunk: {avg_tempo_min:.1f}min")
        print()

        if train_gmeans:
            print(f"Train G-mean: {statistics.mean(train_gmeans):.4f} +/- {statistics.stdev(train_gmeans) if len(train_gmeans) > 1 else 0:.4f}")
        if test_gmeans:
            print(f"Test G-mean:  {statistics.mean(test_gmeans):.4f} +/- {statistics.stdev(test_gmeans) if len(test_gmeans) > 1 else 0:.4f}")
        print()

        print("Por chunk:")
        for c in chunks:
            print(f"  Chunk {c['id']}: {c['tempo_min']:.1f}min, Train={c['train_gmean']:.4f}, Test={c['test_gmean']:.4f}")
    else:
        print("AVISO: Log truncado - nenhum chunk completado")
        print("Analisando metricas parciais disponiveis...")
    print()

    # 2. CACHE FUNCIONOU?
    print("2. CACHE SHA256 - FUNCIONAMENTO")
    print("-" * 80)

    if cache_logs:
        print(f"Total de geracoes com cache: {len(cache_logs)}")
        print()

        # Estatisticas de hit rate
        hit_rates = [c['hit_rate'] for c in cache_logs]
        avg_hit_rate = statistics.mean(hit_rates)
        max_hit_rate = max(hit_rates)

        # Cache hits por geracao
        gen_1_cache = [c for c in cache_logs if c['gen'] == 1]
        gen_2plus_cache = [c for c in cache_logs if c['gen'] >= 2]

        if gen_1_cache:
            print(f"Geracao 1: Hits={gen_1_cache[0]['hits']}/{gen_1_cache[0]['total']} ({gen_1_cache[0]['hit_rate']:.1f}%)")

        if gen_2plus_cache:
            avg_hit_rate_2plus = statistics.mean([c['hit_rate'] for c in gen_2plus_cache])
            print(f"Geracoes 2+: Hit rate medio = {avg_hit_rate_2plus:.1f}%")
            print(f"             Hit rate maximo = {max_hit_rate:.1f}%")
        print()

        print("Primeiras 10 geracoes:")
        for c in cache_logs[:10]:
            print(f"  Gen {c['gen']}: Hits={c['hits']}/{c['total']} ({c['hit_rate']:.1f}%)")
        print()

        print("STATUS: CACHE FUNCIONANDO")
        print(f"  Hit rate medio: {avg_hit_rate:.1f}%")
        print(f"  Hit rate maximo: {max_hit_rate:.1f}%")
    else:
        print("STATUS: CACHE NAO FUNCIONOU (nenhum log encontrado)")
    print()

    # 3. EARLY STOP FUNCIONOU?
    print("3. EARLY STOP ADAPTATIVO - FUNCIONAMENTO")
    print("-" * 80)

    if early_stop_logs:
        print(f"Total de geracoes com early stop: {len(early_stop_logs)}")
        print()

        # Estatisticas de descarte
        descarte_pcts = [e['pct'] for e in early_stop_logs]
        avg_descarte = statistics.mean(descarte_pcts)
        max_descarte = max(descarte_pcts)

        # Early stop por geracao
        gen_2_early = [e for e in early_stop_logs if e['gen'] == 2]
        gen_3plus_early = [e for e in early_stop_logs if e['gen'] >= 3]

        if gen_2_early:
            print(f"Geracao 2: Descartados={gen_2_early[0]['descartados']}/{gen_2_early[0]['total']} ({gen_2_early[0]['pct']:.1f}%)")

        if gen_3plus_early:
            avg_descarte_3plus = statistics.mean([e['pct'] for e in gen_3plus_early])
            print(f"Geracoes 3+: Descarte medio = {avg_descarte_3plus:.1f}%")
            print(f"             Descarte maximo = {max_descarte:.1f}%")
        print()

        print("Primeiras 10 geracoes:")
        for e in early_stop_logs[:10]:
            print(f"  Gen {e['gen']}: Descartados={e['descartados']}/{e['total']} ({e['pct']:.1f}%)")
        print()

        print("STATUS: EARLY STOP FUNCIONANDO")
        print(f"  Descarte medio: {avg_descarte:.1f}%")
        print(f"  Descarte maximo: {max_descarte:.1f}%")
    else:
        print("STATUS: EARLY STOP NAO FUNCIONOU (nenhum log encontrado)")
    print()

    # 4. COMPARACAO COM RUN5
    print("4. COMPARACAO COM RUN5 (DEBUG)")
    print("-" * 80)

    # Dados do Run5
    run5_tempo_chunk = 154.3  # minutos (media)
    run5_gmean = 0.7852
    run5_cache_hits = 0
    run5_early_stops = 0

    print(f"Run5 (Debug - Layer1 quebrado):")
    print(f"  Tempo medio/chunk: {run5_tempo_chunk:.1f}min")
    print(f"  Test G-mean: {run5_gmean:.4f}")
    print(f"  Cache hits: {run5_cache_hits}")
    print(f"  Early stop descartes: {run5_early_stops}")
    print()

    if chunks:
        run997_tempo_chunk = statistics.mean([c['tempo_min'] for c in chunks])
        run997_gmean = statistics.mean([c['test_gmean'] for c in chunks if c['test_gmean'] > 0])
        run997_cache_hits = len([c for c in cache_logs if c['hits'] > 0])
        run997_early_stops = len(early_stop_logs)

        print(f"Run997 (Smoke Test - Layer1 paralelo):")
        print(f"  Tempo medio/chunk: {run997_tempo_chunk:.1f}min")
        print(f"  Test G-mean: {run997_gmean:.4f}")
        print(f"  Cache hits: {run997_cache_hits} geracoes")
        print(f"  Early stop descartes: {run997_early_stops} geracoes")
        print()

        delta_tempo = ((run997_tempo_chunk - run5_tempo_chunk) / run5_tempo_chunk) * 100
        delta_gmean = run997_gmean - run5_gmean

        print(f"DELTA Run5 -> Run997:")
        print(f"  Tempo: {delta_tempo:+.1f}%")
        print(f"  G-mean: {delta_gmean:+.4f}")
        print(f"  Cache: {run5_cache_hits} -> {run997_cache_hits} geracoes")
        print(f"  Early stop: {run5_early_stops} -> {run997_early_stops} geracoes")
        print()

        if delta_tempo < 0:
            print(f"MELHORIA DE TEMPO: {abs(delta_tempo):.1f}% mais rapido")
        else:
            print(f"PIORA DE TEMPO: {delta_tempo:.1f}% mais lento")
    else:
        print("Run997: Dados incompletos (log truncado)")
    print()

    # 5. VALIDACAO METRICAS DE SUCESSO
    print("5. VALIDACAO - METRICAS DE SUCESSO")
    print("-" * 80)

    success_criteria = {
        'Sintaxe valida': True,  # Ja validado
        'Logs de cache aparecem': len(cache_logs) >= 5,
        'Logs de early stop aparecem': len(early_stop_logs) >= 3,
        'Tempo reduz (< 90min/chunk)': chunks and statistics.mean([c['tempo_min'] for c in chunks]) < 90 if chunks else False,
        'G-mean mantido (>= 0.75)': chunks and statistics.mean([c['test_gmean'] for c in chunks if c['test_gmean'] > 0]) >= 0.75 if chunks else False
    }

    for criterio, passou in success_criteria.items():
        status = "OK" if passou else "FALHOU"
        print(f"  [{status}] {criterio}")
    print()

    total_passou = sum(success_criteria.values())
    total_criterios = len(success_criteria)

    print(f"RESULTADO: {total_passou}/{total_criterios} criterios atendidos")
    print()

    if total_passou >= 4:
        print("STATUS: IMPLEMENTACAO BEM-SUCEDIDA")
        print("PROXIMO PASSO: Executar experimento completo (Run6 com 5+ chunks)")
    elif total_passou >= 3:
        print("STATUS: IMPLEMENTACAO PARCIALMENTE BEM-SUCEDIDA")
        print("PROXIMO PASSO: Investigar criterios que falharam")
    else:
        print("STATUS: IMPLEMENTACAO FALHOU")
        print("PROXIMO PASSO: Revisar codigo e debugar")
    print()

    # 6. ECONOMIA ESTIMADA
    print("6. ECONOMIA ESTIMADA (Layer 1)")
    print("-" * 80)

    if cache_logs and early_stop_logs:
        # Economia de cache
        total_evals_sem_cache = sum(c['total'] for c in cache_logs)
        total_cache_hits = sum(c['hits'] for c in cache_logs)
        economia_cache_pct = (total_cache_hits / total_evals_sem_cache * 100) if total_evals_sem_cache > 0 else 0

        # Economia de early stop
        total_evals_sem_early = sum(e['total'] for e in early_stop_logs)
        total_early_stops = sum(e['descartados'] for e in early_stop_logs)
        economia_early_pct = (total_early_stops / total_evals_sem_early * 100) if total_evals_sem_early > 0 else 0

        print(f"Cache SHA256:")
        print(f"  Avaliacoes evitadas: {total_cache_hits}/{total_evals_sem_cache} ({economia_cache_pct:.1f}%)")
        print()

        print(f"Early Stop Adaptativo:")
        print(f"  Avaliacoes descartadas: {total_early_stops}/{total_evals_sem_early} ({economia_early_pct:.1f}%)")
        print()

        economia_total = economia_cache_pct + economia_early_pct
        print(f"Economia total estimada: {economia_total:.1f}%")
        print(f"  (Cache: {economia_cache_pct:.1f}% + Early Stop: {economia_early_pct:.1f}%)")
    else:
        print("Dados insuficientes para calcular economia")
    print()

    print("=" * 80)
    print("FIM DA ANALISE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_run997()
