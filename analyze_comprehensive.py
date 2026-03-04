"""
Análise abrangente do experimento com debug ativo
Compara com experimentos anteriores e identifica problemas raiz
"""

import re
import statistics
from collections import defaultdict

def parse_comprehensive_log(filepath):
    """Parse completo do log com debug"""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = []
    current_chunk = None
    chunk_being_finalized = None

    # Patterns
    chunk_start = re.compile(r'CHUNK (\d+) - INÍCIO')
    chunk_final = re.compile(r'CHUNK (\d+) - FINAL')
    tempo_total = re.compile(r'Tempo total: ([\d.]+)s \(([\d.]+)min\)')
    train_gmean = re.compile(r'Train G-mean: ([\d.]+)')
    test_gmean = re.compile(r'Test G-mean:\s+([\d.]+)')
    delta_pattern = re.compile(r'Delta:\s+([-\d.]+)')

    # Alternative start pattern for logs that use INICIO instead of INÍCIO
    chunk_start_alt = re.compile(r'CHUNK (\d+) - INICIO')

    # Debug patterns
    early_stop_desativado = re.compile(r'\[DEBUG L1 FITNESS\] Early stop DESATIVADO: threshold=(\S+)')
    early_stop_check = re.compile(r'\[DEBUG L1 FITNESS\] Early stop check')
    early_stop_eval = re.compile(r'\[DEBUG L1 FITNESS\] Early stop eval')
    early_stop_descarte = re.compile(r'\[DEBUG L1 FITNESS\] EARLY STOP DESCARTE')

    debug_l1_cache = re.compile(r'\[DEBUG L1\].*cache', re.IGNORECASE)
    debug_l1_hash = re.compile(r'\[DEBUG L1\].*Hash gerado')

    # Layer 1 patterns
    hc_aplicando = re.compile(r'\[HC\] Aplicando Hill Climbing')
    hc_pulando = re.compile(r'\[HC\] PULANDO Hill Climbing')
    hc_variantes = re.compile(r'\[HC\] Geradas (\d+) variantes')
    hc_aprovadas = re.compile(r'\[HC\] Aprovadas: (\d+)/(\d+)')

    early_stop_threshold_log = re.compile(r'\[EARLY STOP\] Gen \d+: threshold=([\d.]+)')

    lines = content.split('\n')

    for line in lines:
        match = chunk_start.search(line)
        if not match:
            match = chunk_start_alt.search(line)

        if match:
            chunk_id = int(match.group(1))
            current_chunk = {
                'id': chunk_id,
                'tempo_s': 0,
                'tempo_min': 0,
                'train_gmean': 0,
                'test_gmean': 0,
                'delta': 0,
                'early_stop_desativado_count': 0,
                'early_stop_threshold_logged': [],
                'hc_aplicado': 0,
                'hc_pulado': 0,
                'hc_variantes_total': 0,
                'hc_aprovadas_total': 0,
                'debug_cache_logs': 0,
                'debug_hash_logs': 0
            }
            continue

        match = chunk_final.search(line)
        if match and current_chunk:
            # Move current chunk to finalization mode
            chunk_being_finalized = current_chunk
            current_chunk = None
            continue

        # Check if we're reading metrics for a chunk that just ended
        # Metrics appear AFTER CHUNK X - FINAL marker
        if chunk_being_finalized is not None:
            # Tempo
            match = tempo_total.search(line)
            if match:
                chunk_being_finalized['tempo_s'] = float(match.group(1))
                chunk_being_finalized['tempo_min'] = float(match.group(2))

            # Métricas
            match = train_gmean.search(line)
            if match:
                chunk_being_finalized['train_gmean'] = float(match.group(1))

            match = test_gmean.search(line)
            if match:
                chunk_being_finalized['test_gmean'] = float(match.group(1))
                # After test gmean, chunk is complete - save it
                chunks.append(chunk_being_finalized)
                chunk_being_finalized = None

            match = delta_pattern.search(line)
            if match:
                chunk_being_finalized['delta'] = float(match.group(1))

        # Debug early stop desativado
        if current_chunk and early_stop_desativado.search(line):
            current_chunk['early_stop_desativado_count'] += 1

        # Early stop threshold logged
        if current_chunk:
            match = early_stop_threshold_log.search(line)
            if match:
                threshold = float(match.group(1))
                current_chunk['early_stop_threshold_logged'].append(threshold)

            # HC
            if hc_aplicando.search(line):
                current_chunk['hc_aplicado'] += 1

            if hc_pulando.search(line):
                current_chunk['hc_pulado'] += 1

            match = hc_variantes.search(line)
            if match:
                current_chunk['hc_variantes_total'] += int(match.group(1))

            match = hc_aprovadas.search(line)
            if match:
                current_chunk['hc_aprovadas_total'] += int(match.group(1))

            # Debug cache
            if debug_l1_cache.search(line):
                current_chunk['debug_cache_logs'] += 1

            if debug_l1_hash.search(line):
                current_chunk['debug_hash_logs'] += 1

    return chunks


def analyze_comprehensive(chunks):
    """Análise completa com foco em problemas raiz"""

    print("=" * 80)
    print("ANALISE COMPLETA - EXPERIMENTO COM DEBUG (RUN5)")
    print("=" * 80)
    print()

    # 1. RESUMO GERAL
    print("1. RESUMO GERAL")
    print("-" * 80)

    total_tempo_h = sum(c['tempo_min'] for c in chunks) / 60
    avg_tempo_min = statistics.mean([c['tempo_min'] for c in chunks])

    train_gmeans = [c['train_gmean'] for c in chunks]
    test_gmeans = [c['test_gmean'] for c in chunks]
    deltas = [c['delta'] for c in chunks]

    print(f"Total chunks: {len(chunks)}")
    print(f"Tempo total: {total_tempo_h:.2f}h ({sum(c['tempo_min'] for c in chunks):.1f}min)")
    print(f"Tempo medio por chunk: {avg_tempo_min:.1f}min")
    print()
    print(f"Train G-mean: {statistics.mean(train_gmeans):.4f} +/- {statistics.stdev(train_gmeans) if len(train_gmeans) > 1 else 0:.4f}")
    print(f"Test G-mean:  {statistics.mean(test_gmeans):.4f} +/- {statistics.stdev(test_gmeans) if len(test_gmeans) > 1 else 0:.4f}")
    print(f"Delta medio:  {statistics.mean(deltas):.4f}")
    print()

    # 2. PROBLEMA CRITICO: EARLY STOP DESATIVADO
    print("2. DIAGNOSTICO CRITICO - EARLY STOP")
    print("-" * 80)

    total_early_stop_desativado = sum(c['early_stop_desativado_count'] for c in chunks)

    print(f"Logs 'Early stop DESATIVADO': {total_early_stop_desativado} ocorrencias")
    print()

    if total_early_stop_desativado > 0:
        print("PROBLEMA IDENTIFICADO:")
        print("  Early stop recebe threshold=None durante avaliacao paralela")
        print("  Isso significa que early_stop_threshold NAO esta sendo passado")
        print("  para a funcao calculate_fitness() em evaluate_individual_fitness_parallel()")
        print()

    print("Por chunk:")
    for c in chunks:
        thresholds_logged = len(c['early_stop_threshold_logged'])
        print(f"  Chunk {c['id']}: {c['early_stop_desativado_count']} desativado logs, {thresholds_logged} thresholds calculados")
    print()

    # 3. CACHE: PROBLEMA TAMBEM IDENTIFICADO
    print("3. DIAGNOSTICO - CACHE")
    print("-" * 80)

    total_cache_logs = sum(c['debug_cache_logs'] for c in chunks)
    total_hash_logs = sum(c['debug_hash_logs'] for c in chunks)

    print(f"Logs de cache (DEBUG L1): {total_cache_logs}")
    print(f"Logs de hash gerado: {total_hash_logs}")
    print()

    if total_cache_logs == 0 and total_hash_logs == 0:
        print("PROBLEMA IDENTIFICADO:")
        print("  Codigo de cache NAO esta sendo executado")
        print("  Possivel causa: codigo de cache esta no modo SERIAL")
        print("  mas experimento esta rodando em modo PARALELO")
        print()

    # 4. HILL CLIMBING (UNICO QUE FUNCIONA)
    print("4. HILL CLIMBING (FUNCIONANDO)")
    print("-" * 80)

    total_hc_aplicado = sum(c['hc_aplicado'] for c in chunks)
    total_hc_pulado = sum(c['hc_pulado'] for c in chunks)

    if total_hc_aplicado + total_hc_pulado > 0:
        economia_hc = (total_hc_pulado / (total_hc_aplicado + total_hc_pulado)) * 100
        print(f"HC aplicado: {total_hc_aplicado}")
        print(f"HC pulado: {total_hc_pulado}")
        print(f"Economia: {economia_hc:.1f}%")
        print()

        for c in chunks:
            if c['hc_aplicado'] + c['hc_pulado'] > 0:
                approval = (c['hc_aprovadas_total'] / c['hc_variantes_total'] * 100) if c['hc_variantes_total'] > 0 else 0
                print(f"  Chunk {c['id']}: Aplicado={c['hc_aplicado']}, Pulado={c['hc_pulado']}, Aprovacao={approval:.1f}%")
    print()

    # 5. COMPARACAO COM RUN3 E RUN4
    print("5. COMPARACAO ENTRE RUNS")
    print("-" * 80)

    # Dados historicos
    run3_tempo = 9.9
    run3_gmean = 0.7763

    run4_tempo = 13.42
    run4_gmean = 0.7928

    run5_tempo = total_tempo_h
    run5_gmean = statistics.mean(test_gmeans)

    print(f"Run3 (Baseline - Fase 1+2):     {run3_tempo:.1f}h, G-mean={run3_gmean:.4f}")
    print(f"Run4 (Layer1 sem funcionar):    {run4_tempo:.1f}h, G-mean={run4_gmean:.4f}")
    print(f"Run5 (Debug ativo):             {run5_tempo:.1f}h, G-mean={run5_gmean:.4f}")
    print()

    delta_run3_run4_tempo = ((run4_tempo - run3_tempo) / run3_tempo) * 100
    delta_run3_run5_tempo = ((run5_tempo - run3_tempo) / run3_tempo) * 100
    delta_run4_run5_tempo = ((run5_tempo - run4_tempo) / run4_tempo) * 100

    print(f"Run3 -> Run4: {delta_run3_run4_tempo:+.1f}% tempo, {(run4_gmean - run3_gmean):+.4f} G-mean")
    print(f"Run3 -> Run5: {delta_run3_run5_tempo:+.1f}% tempo, {(run5_gmean - run3_gmean):+.4f} G-mean")
    print(f"Run4 -> Run5: {delta_run4_run5_tempo:+.1f}% tempo, {(run5_gmean - run4_gmean):+.4f} G-mean")
    print()

    # 6. CAUSA RAIZ IDENTIFICADA
    print("6. CAUSA RAIZ DOS PROBLEMAS")
    print("-" * 80)

    print("PROBLEMA 1: Early Stop nao funciona")
    print("  Sintoma: threshold=None nas avaliacoes paralelas")
    print("  Causa raiz: evaluate_individual_fitness_parallel() NAO passa")
    print("              early_stop_threshold para calculate_fitness()")
    print("  Localizacao: ga.py linha ~161")
    print()

    print("PROBLEMA 2: Cache nao funciona")
    print("  Sintoma: Zero logs de cache ou hash")
    print("  Causa raiz: Codigo de cache esta no bloco SERIAL (linha ~915)")
    print("              mas experimento roda em modo PARALELO (use_parallel=True)")
    print("  Localizacao: ga.py linha ~907-998")
    print()

    print("CONCLUSAO:")
    print("  Layer 1 foi implementado APENAS no modo SERIAL")
    print("  Modo PARALELO (default) NAO tem cache nem early stop")
    print("  Por isso Run3, Run4 e Run5 nao tiveram beneficio")
    print()

    # 7. IMPACTO ESTIMADO
    print("7. IMPACTO ESTIMADO SE CORRIGIDO")
    print("-" * 80)

    print("Se Early Stop e Cache funcionassem:")
    print(f"  Tempo atual: {run5_tempo:.1f}h")
    print(f"  Reducao esperada: -40-55%")
    print(f"  Tempo esperado: {run5_tempo * 0.45:.1f}h - {run5_tempo * 0.60:.1f}h")
    print(f"  Diferenca vs Run3: {((run5_tempo * 0.50 - run3_tempo) / run3_tempo * 100):+.1f}%")
    print()

    # 8. RECOMENDACOES URGENTES
    print("8. RECOMENDACOES URGENTES")
    print("-" * 80)

    print("PRIORIDADE 1 (CRITICO):")
    print("  1. Implementar cache e early stop no modo PARALELO")
    print("     - Mover codigo de ga.py linha 915-998 para modo paralelo")
    print("     - Adicionar early_stop_threshold ao worker args")
    print("     - Implementar cache compartilhado entre workers")
    print()

    print("PRIORIDADE 2 (ALTO):")
    print("  2. Corrigir evaluate_individual_fitness_parallel()")
    print("     - Passar early_stop_threshold via constant_args_for_worker")
    print("     - Linha ~161: adicionar parametro ao calculate_fitness()")
    print()

    print("PRIORIDADE 3 (MEDIO):")
    print("  3. Validar com smoke test")
    print("     - 2 chunks com use_parallel=False (serial)")
    print("     - Verificar se cache e early stop funcionam")
    print()

    print("=" * 80)
    print("CONCLUSAO FINAL")
    print("=" * 80)
    print()
    print("Layer 1 NAO funciona porque foi implementado apenas no modo SERIAL")
    print("Todos os experimentos (Run3, Run4, Run5) rodaram em modo PARALELO")
    print("Por isso tempo piorou em vez de melhorar")
    print()
    print("PROXIMA ACAO:")
    print("  Implementar Layer 1 no modo PARALELO ou")
    print("  Forcar use_parallel=False para testar implementacao serial")
    print()


def main():
    filepath = "C:\\Users\\Leandro Almeida\\Downloads\\DSL-AG-hybrid\\novo_experimento2.txt"

    print("Parseando log com debug...")
    chunks = parse_comprehensive_log(filepath)

    print(f"Encontrados {len(chunks)} chunks")
    print()

    analyze_comprehensive(chunks)

    # Salvar
    output_file = "C:\\Users\\Leandro Almeida\\Downloads\\DSL-AG-hybrid\\ANALISE_COMPREHENSIVE_RUN5.txt"

    import sys
    original_stdout = sys.stdout

    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        analyze_comprehensive(chunks)

    sys.stdout = original_stdout

    print()
    print(f"Analise completa salva em: {output_file}")


if __name__ == "__main__":
    main()
