"""
Diagnóstico focado: Por que Cache e Early Stop não aparecem nos logs?
"""

import re

def diagnosticar_logs(filepath):
    """Verifica presença de todos os tipos de logs Layer 1"""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    print("=" * 80)
    print("DIAGNOSTICO LAYER 1 - LOGS")
    print("=" * 80)
    print()

    # 1. EARLY STOP - Threshold
    print("1. EARLY STOP - THRESHOLD")
    print("-" * 80)
    threshold_logs = re.findall(r'\[EARLY STOP\] Gen (\d+): threshold=([\d.]+)', content)
    print(f"Total de logs de threshold: {len(threshold_logs)}")
    if threshold_logs:
        print("Primeiros 5:")
        for gen, thresh in threshold_logs[:5]:
            print(f"  Gen {gen}: threshold={thresh}")
    print()

    # 2. EARLY STOP - Descartados
    print("2. EARLY STOP - DESCARTADOS")
    print("-" * 80)
    descartados_logs = re.findall(r'\[EARLY STOP\].*Descartados=(\d+)/(\d+)', content)
    print(f"Total de logs de descartados: {len(descartados_logs)}")
    if descartados_logs:
        print("Primeiros 5:")
        for desc, total in descartados_logs[:5]:
            pct = (int(desc) / int(total)) * 100
            print(f"  Descartados={desc}/{total} ({pct:.1f}%)")
    else:
        print("  NENHUM LOG ENCONTRADO")
        print("  Possível causa:")
        print("    - early_stopped_count sempre 0 (nenhum indivíduo descartado)")
        print("    - if early_stopped_count > 0: não executou")
    print()

    # 3. CACHE - Por geração
    print("3. CACHE - HIT RATE POR GERACAO")
    print("-" * 80)
    cache_gen_logs = re.findall(r'\[CACHE\] Gen (\d+): Hits=(\d+)/(\d+)', content)
    print(f"Total de logs de cache por geração: {len(cache_gen_logs)}")
    if cache_gen_logs:
        print("Primeiros 5:")
        for gen, hits, total in cache_gen_logs[:5]:
            hit_rate = (int(hits) / int(total)) * 100 if int(total) > 0 else 0
            print(f"  Gen {gen}: Hits={hits}/{total} ({hit_rate:.1f}%)")
    else:
        print("  NENHUM LOG ENCONTRADO")
        print("  Possível causa:")
        print("    - if cache_hits > 0 or cache_misses > 0: não executou")
        print("    - Código de logging não foi executado")
    print()

    # 4. CACHE - Final
    print("4. CACHE - FINAL (POR CHUNK)")
    print("-" * 80)
    cache_final_logs = re.findall(r'\[CACHE FINAL\] Hits=(\d+), Misses=(\d+), Hit Rate=([\d.]+)%', content)
    print(f"Total de logs de cache final: {len(cache_final_logs)}")
    if cache_final_logs:
        for i, (hits, misses, rate) in enumerate(cache_final_logs):
            print(f"  Chunk {i}: Hits={hits}, Misses={misses}, Rate={rate}%")
    else:
        print("  NENHUM LOG ENCONTRADO")
        print("  Possível causa:")
        print("    - total_cache_ops = 0 (cache não foi usado)")
        print("    - Código de logging não foi executado")
    print()

    # 5. HC - Aplicando/Pulando
    print("5. HILL CLIMBING - APLICADO/PULADO")
    print("-" * 80)
    hc_aplicado = re.findall(r'\[HC\] Aplicando Hill Climbing', content)
    hc_pulado = re.findall(r'\[HC\] PULANDO Hill Climbing', content)
    print(f"HC Aplicado: {len(hc_aplicado)}x")
    print(f"HC Pulado: {len(hc_pulado)}x")
    if len(hc_aplicado) + len(hc_pulado) > 0:
        pct = (len(hc_pulado) / (len(hc_aplicado) + len(hc_pulado))) * 100
        print(f"Economia: {pct:.1f}%")
    print()

    # 6. GEN - Resumos
    print("6. GENERATION SUMMARY")
    print("-" * 80)
    gen_summary = re.findall(r'\[GEN (\d+)\]', content)
    print(f"Total de resumos de geração: {len(gen_summary)}")
    if gen_summary:
        print(f"Gerações logadas: {', '.join(gen_summary[:10])}...")
    print()

    # 7. Análise de contadores
    print("7. ANALISE DE POSSÍVEIS PROBLEMAS")
    print("-" * 80)

    problemas = []

    if len(threshold_logs) > 0 and len(descartados_logs) == 0:
        problemas.append("EARLY STOP: Threshold calculado MAS nenhum descarte")
        problemas.append("  -> early_stopped_count sempre 0")
        problemas.append("  -> Threshold muito alto ou early stop não funciona")

    if len(cache_gen_logs) == 0:
        problemas.append("CACHE: Nenhum log de cache por geração")
        problemas.append("  -> cache_hits e cache_misses sempre 0")
        problemas.append("  -> Cache não está sendo usado")

    if len(cache_final_logs) == 0:
        problemas.append("CACHE FINAL: Nenhum log de cache final")
        problemas.append("  -> total_cache_ops = 0")
        problemas.append("  -> Confirma que cache não foi usado")

    if not problemas:
        print("  Nenhum problema óbvio detectado (todos logs presentes)")
    else:
        for i, prob in enumerate(problemas, 1):
            print(f"  {i}. {prob}")

    print()
    print("=" * 80)
    print("RECOMENDACAO")
    print("=" * 80)
    print()

    if len(descartados_logs) == 0:
        print("EARLY STOP:")
        print("  1. Verificar ga.py linha 914-916: early_stopped_count incremento")
        print("  2. Verificar fitness.py linha 231: early_stopped flag sendo setado")
        print("  3. Adicionar debug: print(f'early_stopped_count = {early_stopped_count}')")
        print()

    if len(cache_gen_logs) == 0:
        print("CACHE:")
        print("  1. Verificar ga.py linha 899-943: cache_hit vs cache_miss lógica")
        print("  2. Verificar ga.py linha 955-956: if cache_hits > 0 or cache_misses > 0")
        print("  3. Adicionar debug: print(f'cache_hits={cache_hits}, cache_misses={cache_misses}')")
        print()

    print("PRÓXIMO PASSO:")
    print("  Execute smoke test (2 chunks) com debug logging adicional")
    print()


def main():
    filepath = "C:\\Users\\Leandro Almeida\\Downloads\\DSL-AG-hybrid\\novo_experimento.txt"

    diagnosticar_logs(filepath)


if __name__ == "__main__":
    main()
