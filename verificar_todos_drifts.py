#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para Verificar TODOS os Drifts
Identifica quais experimentos precisam de correcao para caber em 6 chunks
"""

import yaml
from pathlib import Path

CONFIG_FILE = Path("config.yaml")
CHUNK_SIZE = 1000
NUM_CHUNKS = 6
TRAIN_END_INSTANCE = 5000  # chunks 0-4 para treino

def calculate_drift_positions(concept_sequence, gradual_width=0):
    """Calcula posicoes de drift baseado na sequencia de conceitos"""
    drifts = []
    current_instance = 0

    for i, concept in enumerate(concept_sequence[:-1]):  # Exceto ultimo
        duration = concept['duration_chunks']
        current_instance += duration * CHUNK_SIZE

        # Drift comeca aqui
        drift_start = current_instance
        drift_end = current_instance + (gradual_width * CHUNK_SIZE)

        drifts.append({
            'position': drift_start,
            'end': drift_end if gradual_width > 0 else drift_start,
            'width': gradual_width,
            'before': concept_sequence[i]['concept_id'],
            'after': concept_sequence[i+1]['concept_id']
        })

        current_instance = drift_end if gradual_width > 0 else drift_start

    return drifts, current_instance

def needs_correction(drifts, total_instance):
    """Verifica se precisa correcao"""
    if not drifts:
        return False, "Sem drifts"

    # Verificar se algum drift esta fora do range de treino
    for drift in drifts:
        if drift['position'] >= TRAIN_END_INSTANCE:
            return True, f"Drift em {drift['position']} (FORA! limite={TRAIN_END_INSTANCE})"
        if drift['end'] > TRAIN_END_INSTANCE:
            return True, f"Drift termina em {drift['end']} (FORA! limite={TRAIN_END_INSTANCE})"

    # Verificar se experimento vai alem de 6 chunks
    if total_instance > NUM_CHUNKS * CHUNK_SIZE:
        return True, f"Total {total_instance} instancias (EXCEDE {NUM_CHUNKS} chunks)"

    return False, "OK - Drifts dentro do range"

def suggest_correction(concept_sequence, gradual_width):
    """Sugere valores corrigidos de duration_chunks"""
    num_concepts = len(concept_sequence)

    # Total de chunks disponiveis para conceitos (excluindo transicoes graduais)
    total_chunks_available = NUM_CHUNKS - (num_concepts - 1) * gradual_width

    if total_chunks_available <= 0:
        return None, "IMPOSSIVEL: Muitas transicoes graduais para 6 chunks"

    # Distribuir chunks igualmente
    chunks_per_concept = total_chunks_available / num_concepts

    if chunks_per_concept < 1:
        return None, f"IMPOSSIVEL: Precisa de pelo menos 1 chunk por conceito ({chunks_per_concept:.2f})"

    # Criar sugestao
    suggested = []
    remaining_chunks = total_chunks_available

    for i in range(num_concepts):
        if i == num_concepts - 1:
            # Ultimo conceito pega o resto
            chunks = remaining_chunks
        else:
            # Arredondar para baixo
            chunks = int(chunks_per_concept)

        suggested.append(chunks)
        remaining_chunks -= chunks

    return suggested, f"Distribuicao: {suggested}"

# Carregar config
with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

experimental_streams = config.get('experimental_streams', {})

# Filtrar drift experiments
drift_keywords = ['Abrupt', 'Gradual', 'Incremental', 'Recurring', 'Blip', 'Mixed']
drift_experiments = {k: v for k, v in experimental_streams.items()
                     if any(kw in k for kw in drift_keywords)}

print("="*100)
print("VERIFICACAO DE DRIFTS - TODOS OS EXPERIMENTOS")
print("="*100)
print(f"Total experimentos: {len(drift_experiments)}")
print(f"Range de treino: 0-{TRAIN_END_INSTANCE} instancias (chunks 0-4)")
print(f"Chunks totais: {NUM_CHUNKS}")
print()

# Analisar cada experimento
needs_fix = []
already_ok = []

for exp_name, exp_config in sorted(drift_experiments.items()):
    concept_seq = exp_config.get('concept_sequence', [])
    drift_type = exp_config.get('drift_type', 'unknown')
    gradual_width = exp_config.get('gradual_drift_width_chunks', 0)

    if not concept_seq:
        continue

    drifts, total_inst = calculate_drift_positions(concept_seq, gradual_width)
    needs_fix_flag, reason = needs_correction(drifts, total_inst)

    if needs_fix_flag:
        needs_fix.append(exp_name)

        print(f"\n[PRECISA CORRECAO] {exp_name}")
        print(f"  Tipo: {drift_type}, Width: {gradual_width} chunks")
        print(f"  Conceitos: {len(concept_seq)}")
        print(f"  Duration atual: {[c['duration_chunks'] for c in concept_seq]}")
        print(f"  Razao: {reason}")

        # Mostrar drifts
        for i, drift in enumerate(drifts, 1):
            status = "FORA" if drift['position'] >= TRAIN_END_INSTANCE else "OK"
            print(f"  Drift {i}: {drift['before']} -> {drift['after']} em {drift['position']}", end="")
            if drift['width'] > 0:
                print(f"-{drift['end']}", end="")
            print(f" [{status}]")

        # Sugerir correcao
        suggested, msg = suggest_correction(concept_seq, gradual_width)
        if suggested:
            print(f"  SUGESTAO: duration_chunks = {suggested}")
        else:
            print(f"  PROBLEMA: {msg}")
    else:
        already_ok.append(exp_name)

# Resumo
print()
print("="*100)
print("RESUMO")
print("="*100)
print(f"Total experimentos: {len(drift_experiments)}")
print(f"Precisam correcao: {len(needs_fix)}")
print(f"Ja corretos: {len(already_ok)}")
print()

if already_ok:
    print("JA CORRETOS:")
    for exp in already_ok:
        print(f"  - {exp}")
    print()

print(f"\nPRECISAM CORRECAO ({len(needs_fix)} experimentos):")
for exp in needs_fix:
    print(f"  - {exp}")

# Salvar lista
with open('experimentos_precisam_correcao.txt', 'w') as f:
    f.write("EXPERIMENTOS QUE PRECISAM CORRECAO\n")
    f.write("="*100 + "\n\n")
    for exp in needs_fix:
        f.write(f"{exp}\n")

print(f"\nLista salva em: experimentos_precisam_correcao.txt")
