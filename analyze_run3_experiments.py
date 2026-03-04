#!/usr/bin/env python3
"""
Análise Completa dos Experimentos Run 3 (com Fase 1 + Fase 2 + Logging)
Compara com baselines e experimentos anteriores para identificar problemas de performance
"""

import re
import sys
from pathlib import Path

def extract_final_metrics(log_file):
    """Extrai métricas finais do experimento"""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Buscar seção EXPERIMENTO FINALIZADO
    match = re.search(r'EXPERIMENTO FINALIZADO.*?Tempo total: ([\d.]+)s \(([\d.]+)h\).*?'
                      r'Média por chunk: ([\d.]+)s.*?'
                      r'Chunks processados: (\d+).*?'
                      r'Avg Test G-mean: ([\d.]+).*?'
                      r'Std Test G-mean: ([\d.]+)', content, re.DOTALL)

    if match:
        return {
            'tempo_total_s': float(match.group(1)),
            'tempo_total_h': float(match.group(2)),
            'tempo_medio_chunk_s': float(match.group(3)),
            'num_chunks': int(match.group(4)),
            'avg_test_gmean': float(match.group(5)),
            'std_test_gmean': float(match.group(6))
        }
    return None

def extract_chunk_times(log_file):
    """Extrai tempo de cada chunk"""
    chunks_time = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'CHUNK (\d+) - FINAL.*', line)
            if match:
                chunk_id = int(match.group(1))
                # Pegar próxima linha com tempo
                continue
            match_time = re.search(r'Tempo total: ([\d.]+)s', line)
            if match_time and 'CHUNK' not in line:
                chunks_time.append(float(match_time.group(1)))
    return chunks_time

def extract_chunk_metrics(log_file):
    """Extrai Train/Test G-mean de cada chunk"""
    chunks = []
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Buscar seções CHUNK X - FINAL
    pattern = r'CHUNK (\d+) - FINAL.*?Tempo total: ([\d.]+)s.*?Train G-mean: ([\d.]+).*?Test G-mean:\s+([\d.]+).*?Test F1:\s+([\d.]+)'
    for match in re.finditer(pattern, content, re.DOTALL):
        chunks.append({
            'chunk_id': int(match.group(1)),
            'tempo_s': float(match.group(2)),
            'train_gmean': float(match.group(3)),
            'test_gmean': float(match.group(4)),
            'test_f1': float(match.group(5))
        })

    return chunks

def extract_phase2_activity(log_file):
    """Extrai atividade da Fase 2 (similaridades, matches, etc)"""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Contar eventos de Fase 2
    novos_conceitos = len(re.findall(r'NOVO CONCEITO:', content))
    conceitos_recorrentes = len(re.findall(r'CONCEITO RECORRENTE:', content))

    # Extrair todas as similaridades
    similarities = []
    for match in re.finditer(r'concept_(\d+): (0\.\d+)', content):
        similarities.append({
            'concept_id': int(match.group(1)),
            'similarity': float(match.group(2))
        })

    # Contar MATCH vs NO MATCH
    matches = len(re.findall(r'✓ MATCH:', content))
    no_matches = len(re.findall(r'✗ NO MATCH:', content))

    return {
        'novos_conceitos': novos_conceitos,
        'conceitos_recorrentes': conceitos_recorrentes,
        'total_similarities': len(similarities),
        'matches': matches,
        'no_matches': no_matches,
        'similarities': similarities
    }

def extract_generation_times(log_file, num_samples=10):
    """Extrai tempos de gerações (primeiras N gerações de cada chunk)"""
    times = []
    with open(log_file, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            match = re.search(r'Gen \d+/\d+ -.*Time: ([\d.]+)s', line)
            if match and count < num_samples:
                times.append(float(match.group(1)))
                count += 1
            if count >= num_samples:
                break
    return times

def compare_with_baseline(run3_metrics, baseline_metrics, exp_name):
    """Compara Run3 com baseline"""
    print(f"\n{'='*80}")
    print(f"  COMPARACAO: {exp_name} - Run3 vs Baseline")
    print(f"{'='*80}\n")

    if not run3_metrics or not baseline_metrics:
        print("[X] Metricas incompletas para comparacao")
        return

    # Performance
    delta_gmean = run3_metrics['avg_test_gmean'] - baseline_metrics['avg_test_gmean']
    delta_gmean_pct = (delta_gmean / baseline_metrics['avg_test_gmean']) * 100

    print(f">> PERFORMANCE (G-mean):")
    print(f"   Baseline:  {baseline_metrics['avg_test_gmean']:.4f} ({baseline_metrics['avg_test_gmean']*100:.2f}%)")
    print(f"   Run3:      {run3_metrics['avg_test_gmean']:.4f} ({run3_metrics['avg_test_gmean']*100:.2f}%)")
    print(f"   Delta:     {delta_gmean:+.4f} ({delta_gmean_pct:+.2f}%)", end="")

    if delta_gmean > 0.01:
        print(" [OK] MELHORA")
    elif delta_gmean < -0.01:
        print(" [X] PIORA SIGNIFICATIVA")
    else:
        print(" [~] ESTAVEL")

    # Timing
    delta_time = run3_metrics['tempo_total_h'] - baseline_metrics['tempo_total_h']
    delta_time_pct = (delta_time / baseline_metrics['tempo_total_h']) * 100

    print(f"\n>> TEMPO DE EXECUCAO:")
    print(f"   Baseline:  {baseline_metrics['tempo_total_h']:.2f}h")
    print(f"   Run3:      {run3_metrics['tempo_total_h']:.2f}h")
    print(f"   Delta:     {delta_time:+.2f}h ({delta_time_pct:+.1f}%)", end="")

    if delta_time < -0.5:
        print(" [OK] MAIS RAPIDO")
    elif delta_time > 0.5:
        print(" [X] MAIS LENTO")
    else:
        print(" [~] SIMILAR")

    # Variância
    print(f"\n>> ESTABILIDADE (Std G-mean):")
    print(f"   Baseline:  {baseline_metrics['std_test_gmean']:.4f}")
    print(f"   Run3:      {run3_metrics['std_test_gmean']:.4f}")
    delta_std = run3_metrics['std_test_gmean'] - baseline_metrics['std_test_gmean']
    print(f"   Delta:     {delta_std:+.4f}", end="")

    if delta_std < -0.01:
        print(" [OK] MAIS ESTAVEL")
    elif delta_std > 0.01:
        print(" [X] MENOS ESTAVEL")
    else:
        print(" [~] SIMILAR")

def analyze_experiment_deep(log_file, exp_name):
    """Análise profunda de um experimento"""
    print(f"\n{'#'*80}")
    print(f"  ANÁLISE PROFUNDA: {exp_name}")
    print(f"{'#'*80}\n")

    # 1. Métricas finais
    final_metrics = extract_final_metrics(log_file)
    if final_metrics:
        print(f">> METRICAS FINAIS:")
        print(f"   Avg Test G-mean: {final_metrics['avg_test_gmean']:.4f} ({final_metrics['avg_test_gmean']*100:.2f}%)")
        print(f"   Std Test G-mean: {final_metrics['std_test_gmean']:.4f}")
        print(f"   Tempo total:     {final_metrics['tempo_total_h']:.2f}h ({final_metrics['tempo_total_s']:.0f}s)")
        print(f"   Tempo/chunk:     {final_metrics['tempo_medio_chunk_s']/60:.1f}min ({final_metrics['tempo_medio_chunk_s']:.0f}s)")
    else:
        print("[X] Nao foi possivel extrair metricas finais")
        return None

    # 2. Métricas por chunk
    chunks = extract_chunk_metrics(log_file)
    if chunks:
        print(f"\n>> METRICAS POR CHUNK:")
        print(f"   {'Chunk':<8} {'Train':<8} {'Test':<8} {'F1':<8} {'Tempo':<12}")
        print(f"   {'-'*50}")
        for c in chunks:
            print(f"   {c['chunk_id']:<8} {c['train_gmean']:.4f}   {c['test_gmean']:.4f}   "
                  f"{c['test_f1']:.4f}   {c['tempo_s']/60:.1f}min ({c['tempo_s']:.0f}s)")

    # 3. Fase 2 activity
    phase2 = extract_phase2_activity(log_file)
    print(f"\n>> ATIVIDADE FASE 2:")
    print(f"   Novos conceitos criados:    {phase2['novos_conceitos']}")
    print(f"   Conceitos recorrentes:      {phase2['conceitos_recorrentes']}")
    print(f"   Total de similaridades:     {phase2['total_similarities']}")
    print(f"   Matches (>=0.85):           {phase2['matches']}")
    print(f"   No Matches (<0.85):         {phase2['no_matches']}")

    if phase2['similarities']:
        print(f"\n   Similaridades detectadas:")
        for sim in phase2['similarities'][:20]:  # Primeiras 20
            print(f"      concept_{sim['concept_id']}: {sim['similarity']:.4f}")

    phase2_working = (phase2['novos_conceitos'] > 0 or phase2['conceitos_recorrentes'] > 0)
    if phase2_working:
        print(f"\n   [OK] FASE 2 ATIVA E FUNCIONANDO")
    else:
        print(f"\n   [X] FASE 2 NAO DETECTADA")

    # 4. Timing de gerações (sample)
    gen_times = extract_generation_times(log_file, num_samples=20)
    if gen_times:
        import statistics
        print(f"\n>> TIMING DE GERACOES (sample das primeiras 20):")
        print(f"   Media:   {statistics.mean(gen_times):.1f}s")
        print(f"   Mediana: {statistics.median(gen_times):.1f}s")
        print(f"   Min:     {min(gen_times):.1f}s")
        print(f"   Max:     {max(gen_times):.1f}s")

    return final_metrics

if __name__ == '__main__':
    print("="*80)
    print("  ANÁLISE COMPLETA - RUN 3 (FASE 1 + FASE 2 + LOGGING)")
    print("="*80)

    # ========================================
    # ANÁLISE INDIVIDUAL DOS EXPERIMENTOS
    # ========================================

    # TEST_SINGLE Run3
    print("\n" + "#"*80)
    print("  1. TEST_SINGLE - RUN 3")
    print("#"*80)
    single3_metrics = analyze_experiment_deep(
        'experimento_test_single3.log',
        'TEST_SINGLE Run3'
    )

    # DRIFT_RECOVERY Run3
    print("\n" + "#"*80)
    print("  2. DRIFT_RECOVERY - RUN 3")
    print("#"*80)
    recovery3_metrics = analyze_experiment_deep(
        'experimento_test_recovery3.log',
        'DRIFT_RECOVERY Run3'
    )

    # ========================================
    # COMPARAÇÕES COM BASELINES E RUN2
    # ========================================

    print("\n\n" + "="*80)
    print("  COMPARAÇÕES COM EXPERIMENTOS ANTERIORES")
    print("="*80)

    # Baseline metrics (do documento ANALISE_EXPERIMENTOS_FASE1_FASE2.md)
    single_baseline = {
        'avg_test_gmean': 0.7983,
        'std_test_gmean': 0.1797,
        'tempo_total_h': 8.0  # Estimativa baseada em experimentos anteriores
    }

    single_run2 = {
        'avg_test_gmean': 0.7738,
        'std_test_gmean': 0.1952,
        'tempo_total_h': 8.5  # Estimativa
    }

    recovery_baseline = {
        'avg_test_gmean': 0.7374,
        'std_test_gmean': 0.1975,
        'tempo_total_h': 10.0  # Estimativa
    }

    recovery_run2 = {
        'avg_test_gmean': 0.7243,
        'std_test_gmean': 0.2087,
        'tempo_total_h': 11.0  # Estimativa
    }

    # Comparação TEST_SINGLE
    if single3_metrics:
        compare_with_baseline(single3_metrics, single_baseline, "TEST_SINGLE vs Baseline")
        compare_with_baseline(single3_metrics, single_run2, "TEST_SINGLE vs Run2")

    # Comparação DRIFT_RECOVERY
    if recovery3_metrics:
        compare_with_baseline(recovery3_metrics, recovery_baseline, "DRIFT_RECOVERY vs Baseline")
        compare_with_baseline(recovery3_metrics, recovery_run2, "DRIFT_RECOVERY vs Run2")

    # ========================================
    # RESUMO EXECUTIVO
    # ========================================

    print("\n\n" + "="*80)
    print("  RESUMO EXECUTIVO")
    print("="*80 + "\n")

    if single3_metrics and recovery3_metrics:
        # Performance summary
        print(">> PERFORMANCE (G-mean):")
        print(f"   TEST_SINGLE Run3:     {single3_metrics['avg_test_gmean']:.4f} ({single3_metrics['avg_test_gmean']*100:.2f}%)")
        print(f"   DRIFT_RECOVERY Run3:  {recovery3_metrics['avg_test_gmean']:.4f} ({recovery3_metrics['avg_test_gmean']*100:.2f}%)")

        # Timing summary
        print(f"\n>> TEMPO DE EXECUCAO:")
        print(f"   TEST_SINGLE Run3:     {single3_metrics['tempo_total_h']:.2f}h")
        print(f"   DRIFT_RECOVERY Run3:  {recovery3_metrics['tempo_total_h']:.2f}h")

        # Delta vs baseline
        single_delta = single3_metrics['avg_test_gmean'] - single_baseline['avg_test_gmean']
        recovery_delta = recovery3_metrics['avg_test_gmean'] - recovery_baseline['avg_test_gmean']

        print(f"\n>> DELTA VS BASELINE:")
        print(f"   TEST_SINGLE:     {single_delta:+.4f} ({(single_delta/single_baseline['avg_test_gmean'])*100:+.2f}%)")
        print(f"   DRIFT_RECOVERY:  {recovery_delta:+.4f} ({(recovery_delta/recovery_baseline['avg_test_gmean'])*100:+.2f}%)")

        # Conclusões
        print(f"\n>> CONCLUSOES:")

        if single_delta < -0.01 and recovery_delta < -0.01:
            print("   [X] AMBOS EXPERIMENTOS PIORARAM em relacao ao baseline")
            print("   [!] FASE 1 + FASE 2 nao trouxeram ganhos esperados")
        elif single_delta < -0.01:
            print("   [X] TEST_SINGLE piorou, DRIFT_RECOVERY OK")
            print("   [!] Possivel problema com FASE 1 (early stop, cache)")
        elif recovery_delta < -0.01:
            print("   [X] DRIFT_RECOVERY piorou, TEST_SINGLE OK")
            print("   [!] Possivel problema com FASE 2 (threshold, fingerprint)")
        else:
            print("   [OK] Resultados estaveis ou melhorados")

        # Timing assessment
        if single3_metrics['tempo_total_h'] > 9.5:
            print(f"   [X] TEST_SINGLE muito lento ({single3_metrics['tempo_total_h']:.1f}h vs ~8h esperado)")
            print("   [!] FASE 1 NAO esta reduzindo tempo conforme esperado (-48%)")

        if recovery3_metrics['tempo_total_h'] > 13:
            print(f"   [X] DRIFT_RECOVERY muito lento ({recovery3_metrics['tempo_total_h']:.1f}h vs ~10h esperado)")
            print("   [!] FASE 2 pode estar adicionando overhead excessivo")

    print("\n" + "="*80)
    print("  FIM DA ANÁLISE")
    print("="*80 + "\n")
