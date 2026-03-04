#!/usr/bin/env python3
"""
Diagnostico detalhado do formato do cache de chunks
"""

import pickle
import numpy as np
from pathlib import Path

cache_file = Path('chunks_cache/RBF_Abrupt_Severe_cs3000_nc5_seed42.pkl')

print("="*80)
print("DIAGNOSTICO DO CACHE")
print("="*80)

with open(cache_file, 'rb') as f:
    chunks = pickle.load(f)

print(f"\nCache: {cache_file}")
print(f"Tipo: {type(chunks)}")
print(f"Numero de chunks: {len(chunks)}")

print("\n" + "-"*80)
print("ESTRUTURA DOS CHUNKS")
print("-"*80)

for i, chunk in enumerate(chunks[:2]):  # Apenas primeiros 2 chunks
    print(f"\nChunk {i}:")
    print(f"  Tipo: {type(chunk)}")
    print(f"  Len: {len(chunk)}")

    if isinstance(chunk, (list, tuple)):
        for j, elem in enumerate(chunk):
            print(f"  Elemento {j}:")
            print(f"    Tipo: {type(elem)}")

            if isinstance(elem, (list, tuple)):
                print(f"    Len: {len(elem)}")
                if len(elem) > 0:
                    print(f"    Primeiro item tipo: {type(elem[0])}")
                    if isinstance(elem[0], (list, tuple)):
                        print(f"    Primeiro item len: {len(elem[0])}")
                        if len(elem[0]) > 0:
                            print(f"    Primeiro item[0] tipo: {type(elem[0][0])}")

            if isinstance(elem, np.ndarray):
                print(f"    Shape: {elem.shape}")
                print(f"    Dtype: {elem.dtype}")

            # Tentar converter para array
            try:
                arr = np.array(elem)
                print(f"    Como array: shape={arr.shape}, dtype={arr.dtype}")
            except:
                print(f"    Nao conversivel para array")

print("\n" + "="*80)
print("TENTATIVA DE RECONSTRUCAO")
print("="*80)

# Tentar diferentes formas de reconstruir
chunk0 = chunks[0]

print("\nOpcao 1: chunk[0] e chunk[1]")
try:
    X = np.array(chunk0[0])
    y = np.array(chunk0[1])
    print(f"  X.shape: {X.shape}, y.shape: {y.shape}")
except Exception as e:
    print(f"  ERRO: {e}")

print("\nOpcao 2: Lista de listas transposta")
try:
    # Se cada elemento de chunk[0] for uma feature
    if isinstance(chunk0[0], (list, tuple)) and len(chunk0[0]) > 0:
        if isinstance(chunk0[0][0], (list, tuple)):
            # Transposer
            X = np.array(chunk0[0]).T
            y = np.array(chunk0[1])
            print(f"  X.shape: {X.shape}, y.shape: {y.shape}")
except Exception as e:
    print(f"  ERRO: {e}")

print("\nOpcao 3: Chunk completo como array")
try:
    full = np.array(chunk0)
    print(f"  full.shape: {full.shape}")
except Exception as e:
    print(f"  ERRO: {e}")
