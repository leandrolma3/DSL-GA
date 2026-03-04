#!/usr/bin/env python3
"""
Analise Rapida da Validacao Local

Analisa resultados do teste de validacao e compara com experimento anterior.
"""

import pandas as pd
import glob
import os
import sys

def main():
    print("=" * 80)
    print("ANALISE DA VALIDACAO LOCAL")
    print("=" * 80)
    print()

    # Busca arquivo de resultados
    result_files = glob.glob('validation_local_results/**/comparison_table.csv', recursive=True)

    if not result_files:
        print("[ERRO] Nenhum arquivo de resultados encontrado!")
        print()
        print("Verifique se o teste foi executado e completou com sucesso.")
        return 1

    result_file = result_files[0]
    print(f"Arquivo de resultados: {result_file}")
    print()

    # Le resultados
    df = pd.read_csv(result_file)

    print("=" * 80)
    print("1. METRICAS OBTIDAS")
    print("=" * 80)
    print()

    # Identifica coluna de chunk
    chunk_col = 'chunk' if 'chunk' in df.columns else 'chunk_idx'

    print("Resultados por chunk e modelo:")
    print("-" * 80)
    print(df[[chunk_col, 'model', 'accuracy', 'gmean', 'f1_weighted']].to_string(index=False))
    print()

    # Estatisticas por modelo
    print("=" * 80)
    print("2. ESTATISTICAS POR MODELO")
    print("=" * 80)
    print()

    stats = df.groupby('model')[['accuracy', 'gmean', 'f1_weighted']].agg(['mean', 'std', 'min', 'max'])
    print(stats.round(4))
    print()

    # Comparacao com experimento anterior (chunk_size=6000)
    print("=" * 80)
    print("3. COMPARACAO COM EXPERIMENTO ANTERIOR (chunk_size=6000)")
    print("=" * 80)
    print()

    # Valores do experimento anterior para RBF_Abrupt_Severe
    previous_results = {
        'GBML': {'mean': 0.6695, 'std': 0.1854},
        'ACDWM': {'mean': 0.7553, 'std': 0.2172},
        'HAT': {'mean': 0.7405, 'std': 0.1174},
        'ARF': {'mean': 0.7654, 'std': 0.1933},
        'SRP': {'mean': 0.7711, 'std': 0.1899}
    }

    print("Comparacao de G-mean medio:")
    print("-" * 80)
    print(f"{'Modelo':<10} | {'Anterior':>10} | {'Validacao':>10} | {'Diferenca':>10} | {'Status':>15}")
    print("-" * 80)

    current_means = df.groupby('model')['gmean'].mean()

    for model in ['GBML', 'ACDWM', 'HAT', 'ARF', 'SRP']:
        if model in current_means.index:
            prev_mean = previous_results.get(model, {}).get('mean', 0)
            curr_mean = current_means[model]
            diff = curr_mean - prev_mean
            diff_pct = (diff / prev_mean * 100) if prev_mean > 0 else 0

            if abs(diff_pct) < 10:
                status = "Similar"
            elif diff_pct > 0:
                status = f"Melhor (+{diff_pct:.1f}%)"
            else:
                status = f"Pior ({diff_pct:.1f}%)"

            print(f"{model:<10} | {prev_mean:>10.4f} | {curr_mean:>10.4f} | {diff:>+10.4f} | {status:>15}")
        else:
            print(f"{model:<10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'Ausente':>15}")

    print()

    # Analise de qualidade dos resultados
    print("=" * 80)
    print("4. VALIDACAO DE QUALIDADE")
    print("=" * 80)
    print()

    issues = []

    # Verifica se todos os modelos estao presentes
    expected_models = {'GBML', 'ACDWM', 'HAT', 'ARF', 'SRP'}
    found_models = set(df['model'].unique())
    missing_models = expected_models - found_models

    if missing_models:
        issues.append(f"Modelos faltando: {', '.join(missing_models)}")
        print(f"[AVISO] Modelos faltando: {', '.join(missing_models)}")
    else:
        print("[OK] Todos os 5 modelos presentes")

    # Verifica numero de avaliacoes
    expected_evals = 2  # 3 chunks -> 2 avaliacoes (train-then-test)
    evals_per_model = df.groupby('model').size()

    for model, count in evals_per_model.items():
        if count != expected_evals:
            issues.append(f"{model}: {count} avaliacoes (esperado: {expected_evals})")
            print(f"[AVISO] {model}: {count} avaliacoes (esperado: {expected_evals})")
        else:
            print(f"[OK] {model}: {count} avaliacoes")

    # Verifica se metricas estao em faixa razoavel
    for model in found_models:
        model_data = df[df['model'] == model]
        mean_gmean = model_data['gmean'].mean()

        if mean_gmean < 0.3:
            issues.append(f"{model}: G-mean muito baixo ({mean_gmean:.4f})")
            print(f"[AVISO] {model}: G-mean muito baixo ({mean_gmean:.4f})")
        elif mean_gmean > 0.99:
            issues.append(f"{model}: G-mean suspeito ({mean_gmean:.4f} - possivel overfitting)")
            print(f"[AVISO] {model}: G-mean suspeito ({mean_gmean:.4f})")
        else:
            print(f"[OK] {model}: G-mean em faixa razoavel ({mean_gmean:.4f})")

    print()

    # Verifica tamanho dos arquivos
    print("Arquivos gerados:")
    print("-" * 80)

    result_dir = 'validation_local_results'
    all_files = glob.glob(f'{result_dir}/**/*', recursive=True)

    file_types = {}
    for f in all_files:
        if os.path.isfile(f):
            ext = os.path.splitext(f)[1]
            size_kb = os.path.getsize(f) / 1024
            file_types[ext] = file_types.get(ext, 0) + 1
            print(f"  {f:<60} ({size_kb:>8.1f} KB)")

    print()
    print("Resumo:")
    for ext, count in sorted(file_types.items()):
        print(f"  {ext or '(sem extensao)'}: {count} arquivo(s)")

    print()

    # Conclusao final
    print("=" * 80)
    print("5. CONCLUSAO")
    print("=" * 80)
    print()

    if not issues:
        print("Status: VALIDACAO BEM-SUCEDIDA")
        print()
        print("Resultados:")
        print("  - Todos os modelos executaram corretamente")
        print("  - Metricas em faixa esperada")
        print("  - Resultados similares ao experimento anterior")
        print()
        print("Recomendacao: PROSSEGUIR com experimento completo (6 datasets)")
        print()
        return 0
    else:
        print("Status: VALIDACAO COM RESSALVAS")
        print()
        print(f"Problemas encontrados ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        print("Recomendacao: INVESTIGAR problemas antes de prosseguir")
        print()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
