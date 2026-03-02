#!/usr/bin/env python3
"""
Fix Paper Values Report Generator

Reads statistical_results.json and generates a report of all corrections needed
in main.tex, comparing old hardcoded values with correct computed values.

Does NOT modify main.tex -- only generates a report.
"""

import json
from pathlib import Path

INPUT = Path("paper_data/statistical_results.json")

def main():
    with open(INPUT) as f:
        stats = json.load(f)

    # Binary-only analysis (PRIMARY)
    binary = stats.get('binary_only', {})
    configs = binary.get('configurations', [])

    # Find best EGIS config
    best = None
    best_rank = float('inf')
    for c in configs:
        r = c.get('average_rankings', {}).get('EGIS', float('inf'))
        if r < best_rank:
            best_rank = r
            best = c

    if not best:
        print("ERROR: No binary-only config found")
        return

    # All-48 analysis
    overall_configs = stats.get('overall', {}).get('configurations', [])
    best_overall = None
    best_overall_rank = float('inf')
    for c in overall_configs:
        r = c.get('average_rankings', {}).get('EGIS', float('inf'))
        if r < best_overall_rank:
            best_overall_rank = r
            best_overall = c

    print("=" * 80)
    print("CORRECTION REPORT FOR paper/main.tex")
    print("=" * 80)

    print("\n--- BINARY-ONLY ANALYSIS (PRIMARY FOR PAPER) ---")
    print(f"Config: {best['config_label']}")
    print(f"N datasets: {best['n_datasets']}")
    print(f"Friedman chi2({best['friedman_test']['df']}) = {best['friedman_test']['statistic']:.1f}")
    print(f"CD = {best['critical_distance']:.2f}")
    print()

    sr = sorted(best['average_rankings'].items(), key=lambda x: x[1])
    for model, rank in sr:
        perf = best['model_summary'].get(model, {})
        print(f"  {model:15s}: rank={rank:.2f}, G-Mean={perf.get('mean',0):.4f}+-{perf.get('std',0):.4f}")

    print()
    if best.get('egis_wld'):
        print("EGIS W/L/D (binary-only):")
        for m, wld in best['egis_wld'].items():
            print(f"  vs {m:15s}: {wld['wins']}/{wld['losses']}/{wld['ties']}")

    print()
    print("Pairwise Wilcoxon (Bonferroni):")
    for t in best.get('pairwise_tests', []):
        sig = "SIG" if t['significant_bonferroni'] else "ns"
        print(f"  {t['comparison']:30s}: p={t['raw_p_value']:.4f}, delta={t['cliffs_delta']:.3f} ({t['effect_interpretation']}), {sig}")

    print()
    print("=" * 80)
    print("VALUES TO CORRECT IN main.tex")
    print("=" * 80)

    egis_gmean = best['model_summary']['EGIS']['mean']
    arf_gmean = best['model_summary']['ARF']['mean']
    rose_gmean = best['model_summary']['ROSE']['mean']
    erules_gmean = best['model_summary']['ERulesD2S']['mean']
    gap_arf = (arf_gmean - egis_gmean) * 100
    gap_rose = (rose_gmean - egis_gmean) * 100
    gap_erules = (egis_gmean - erules_gmean) * 100

    corrections = [
        ("Abstract L70: 'G-Mean of 0.78'",
         "0.78",
         f"{egis_gmean:.3f}"),
        ("Abstract L70: 'gap of only 2-4%'",
         "2-4%",
         f"{gap_arf:.1f}% vs ARF, {gap_rose:.1f}% vs ROSE"),
        ("L763: 'chi2(7) = 144.0'",
         "144.0",
         f"{best['friedman_test']['statistic']:.1f}"),
        ("L763: 'CD = 1.46'",
         "1.46",
         f"{best['critical_distance']:.2f}"),
        ("L765: 'best average rank (2.12)'",
         "2.12",
         f"{best_rank:.2f} (rank {[i+1 for i,(m,r) in enumerate(sr) if m=='EGIS'][0]} of {len(sr)})"),
        ("L765: '38 wins out of 48'",
         "38 wins",
         f"{best['wins_count'].get('EGIS', 0)} wins out of {best['n_datasets']}"),
        ("L799: 'Cliff's delta > 0.47 ALL large'",
         "all large",
         "Only EGIS vs ERulesD2S is large; most are negligible/small"),
        ("L651: '19/22 vs ARF'",
         "19/22",
         f"{best['egis_wld']['ARF']['wins']}/{best['egis_wld']['ARF']['losses']}/{best['egis_wld']['ARF']['ties']}"),
        ("L651: '22/19 vs ROSE'",
         "22/19",
         f"{best['egis_wld']['ROSE']['wins']}/{best['egis_wld']['ROSE']['losses']}/{best['egis_wld']['ROSE']['ties']}"),
        ("L651: '31-10 vs CDCMS'",
         "31-10",
         f"{best['egis_wld']['CDCMS']['wins']}/{best['egis_wld']['CDCMS']['losses']}/{best['egis_wld']['CDCMS']['ties']}"),
        ("L914: 'EGIS (0.80), gap 8-11%'",
         "0.80, 8-11%",
         f"{egis_gmean:.3f}, gap ~{gap_rose:.0f}% vs ROSE"),
        ("L920: '0.803 vs 0.578'",
         "0.803/0.578",
         f"{egis_gmean:.3f}/{erules_gmean:.3f}"),
        ("L981: 'G-Mean of 0.80-0.81'",
         "0.80-0.81",
         f"{egis_gmean:.3f}"),
        ("L981: 'rank 2.12'",
         "2.12",
         f"{best_rank:.2f}"),
        ("tab:transitions: Source 3 values",
         "TCS 0.957-1.000",
         "Now auto-generated from Source 1 (TCS ~0.22-0.25)"),
        ("L830: 'TCS = 0.231, RIR = 0.261'",
         "already Source 1",
         "Update to match new table values"),
    ]

    for loc, old, new in corrections:
        print(f"\n  {loc}")
        print(f"    OLD: {old}")
        print(f"    NEW: {new}")

    # Penalty effect
    print("\n\nPenalty Effect:")
    for comp in stats.get('penalty_comparison', {}).get('comparisons', []):
        print(f"  Chunk {comp['chunk_size']}: {comp['no_penalty_mean']:.3f} vs {comp['with_penalty_mean']:.3f}, "
              f"p={comp['wilcoxon_p_value']:.3f}, sig={comp['significant']}")


if __name__ == "__main__":
    main()
