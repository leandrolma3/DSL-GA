#!/usr/bin/env python3
"""
Análise detalhada dos logs de experimentos - Fases 1 e 2
"""

import re
import sys

def parse_summary_table(log_file):
    """Extrai tabela de summary do log"""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Procura por "Stream Name" e pega as próximas linhas
    match = re.search(r'Stream Name.*?\n(.*?)\n=', content, re.DOTALL)
    if match:
        data_line = match.group(1).strip()
        parts = data_line.split()
        return {
            'stream': parts[0],
            'avg_train': float(parts[2]),
            'avg_test': float(parts[4]),
            'avg_f1': float(parts[6])
        }
    return None

def extract_chunk_metrics(log_file):
    """Extrai métricas por chunk do log"""
    chunks = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Procura por "Chunk X Results:"
            match = re.search(r'Chunk (\d+) Results: TrainGmean([0-9.]+), TestGmean=([0-9.]+)', line)
            if match:
                chunks.append({
                    'chunk': int(match.group(1)),
                    'train': float(match.group(2)),
                    'test': float(match.group(3))
                })
    return chunks

def check_phase2_activity(log_file):
    """Verifica se Fase 2 está ativa"""
    phase2_markers = {
        'novo_conceito': 0,
        'recuperando': 0,
        'recorrente_detectado': 0,
        'memory_preservada': 0
    }

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    phase2_markers['novo_conceito'] = len(re.findall(r'NOVO conceito detectado', content))
    phase2_markers['recuperando'] = len(re.findall(r'RECUPERANDO memória', content))
    phase2_markers['recorrente_detectado'] = len(re.findall(r'CONCEITO RECORRENTE DETECTADO', content))
    phase2_markers['memory_preservada'] = len(re.findall(r'Memory PRESERVADA', content))

    return phase2_markers

def count_severe_drifts(log_file):
    """Conta quantos SEVERE drifts foram detectados"""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    return len(re.findall(r'SEVERE DRIFT detected:', content))

def analyze_experiment(log_file, exp_name):
    """Análise completa de um experimento"""
    print(f"\n{'='*80}")
    print(f"  {exp_name}")
    print(f"{'='*80}")

    # 1. Summary metrics
    summary = parse_summary_table(log_file)
    if summary:
        print(f"\n📊 MÉTRICAS FINAIS:")
        print(f"   Stream: {summary['stream']}")
        print(f"   Avg Train G-mean: {summary['avg_train']:.4f} ({summary['avg_train']*100:.2f}%)")
        print(f"   Avg Test G-mean:  {summary['avg_test']:.4f} ({summary['avg_test']*100:.2f}%)")
        print(f"   Avg Test F1:      {summary['avg_f1']:.4f} ({summary['avg_f1']*100:.2f}%)")

    # 2. Drift detection
    severe_count = count_severe_drifts(log_file)
    print(f"\n🔴 SEVERE DRIFTS DETECTADOS: {severe_count}")

    # 3. Phase 2 activity
    phase2 = check_phase2_activity(log_file)
    print(f"\n🧠 ATIVIDADE FASE 2 (Memory Recorrente):")
    print(f"   Novos conceitos criados: {phase2['novo_conceito']}")
    print(f"   Memórias recuperadas: {phase2['recuperando']}")
    print(f"   Detecções de recorrência: {phase2['recorrente_detectado']}")
    print(f"   Memory preservada (drift): {phase2['memory_preservada']}")

    phase2_working = any(v > 0 for v in phase2.values())
    if phase2_working:
        print(f"   ✅ FASE 2 ATIVA")
    else:
        print(f"   ❌ FASE 2 NÃO DETECTADA (possível problema)")

    # 4. Per-chunk metrics (se disponível)
    chunks = extract_chunk_metrics(log_file)
    if chunks:
        print(f"\n📈 MÉTRICAS POR CHUNK:")
        for c in chunks:
            print(f"   Chunk {c['chunk']}: Train={c['train']:.3f}, Test={c['test']:.3f}")

    return summary, phase2_working

def compare_baseline_vs_optimized(baseline_log, optimized_log, exp_name):
    """Compara baseline vs versão otimizada"""
    print(f"\n{'='*80}")
    print(f"  COMPARAÇÃO: {exp_name}")
    print(f"{'='*80}")

    baseline_summary = parse_summary_table(baseline_log)
    optimized_summary = parse_summary_table(optimized_log)

    if not baseline_summary or not optimized_summary:
        print("❌ Não foi possível extrair métricas")
        return

    # Calcula deltas
    delta_test = optimized_summary['avg_test'] - baseline_summary['avg_test']
    delta_test_pct = (delta_test / baseline_summary['avg_test']) * 100

    print(f"\n📊 BASELINE (sem otimizações):")
    print(f"   Avg Test G-mean: {baseline_summary['avg_test']:.4f} ({baseline_summary['avg_test']*100:.2f}%)")

    print(f"\n📊 OTIMIZADO (Fase 1 + Fase 2):")
    print(f"   Avg Test G-mean: {optimized_summary['avg_test']:.4f} ({optimized_summary['avg_test']*100:.2f}%)")

    print(f"\n📈 DELTA:")
    print(f"   Absoluto: {delta_test:+.4f} ({delta_test*100:+.2f} pontos percentuais)")
    print(f"   Relativo: {delta_test_pct:+.2f}%")

    if delta_test > 0:
        print(f"   ✅ MELHORIA")
    elif delta_test < -0.01:
        print(f"   ❌ PIORA SIGNIFICATIVA")
    else:
        print(f"   ⚠️  ESTÁVEL (sem mudança significativa)")

    return delta_test

if __name__ == '__main__':
    print("="*80)
    print("  ANÁLISE DETALHADA - EXPERIMENTOS FASE 1 E FASE 2")
    print("="*80)

    # Análise individual de cada experimento OTIMIZADO
    analyze_experiment('experimento_test_single2.log', 'TEST_SINGLE (Fase 1)')
    analyze_experiment('experimento_test_drift_recovery2.log', 'DRIFT_RECOVERY (Fase 1 + Fase 2)')
    analyze_experiment('experimento_test_multi_drift2.log', 'MULTI_DRIFT (Fase 1 + Fase 2)')

    # Comparações
    print(f"\n\n{'#'*80}")
    print(f"  COMPARAÇÕES BASELINE VS OTIMIZADO")
    print(f"{'#'*80}")

    compare_baseline_vs_optimized(
        'experimento_test_single.log',
        'experimento_test_single2.log',
        'TEST_SINGLE'
    )

    compare_baseline_vs_optimized(
        'experimento_test_drift_recovery.log',
        'experimento_test_drift_recovery2.log',
        'DRIFT_RECOVERY'
    )

    compare_baseline_vs_optimized(
        'experimento_test_multi_drift.log',
        'experimento_test_multi_drift2.log',
        'MULTI_DRIFT'
    )

    print(f"\n{'='*80}")
    print("  FIM DA ANÁLISE")
    print(f"{'='*80}\n")
