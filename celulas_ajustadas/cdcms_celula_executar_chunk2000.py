# =============================================================================
# Executar CDCMS em chunk_2000 (Todos os Batches)
# =============================================================================
# AVISO: 7 batches x ~41 datasets binarios = pode demorar bastante

print("="*70)
print("EXECUTAR CDCMS EM chunk_2000")
print("="*70)

results_chunk2000 = run_cdcms_all_batches('chunk_2000', skip_existing=True)
