#!/usr/bin/env python3
"""
Analise Estatistica Comparativa dos Modelos

Aplica testes estatisticos para determinar se as diferencas
observadas sao estatisticamente significativas.

Testes aplicados:
1. Teste de normalidade (Shapiro-Wilk)
2. ANOVA / Kruskal-Wallis (dependendo da normalidade)
3. Testes post-hoc com correcao de Bonferroni
4. Tamanho de efeito (Cohen's d)
5. Intervalos de confianca
"""

import re
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Importacoes opcionais
try:
    from scipy import stats
    from scipy.stats import shapiro, normaltest, levene, f_oneway, kruskal
    from scipy.stats import mannwhitneyu, ttest_ind
    SCIPY_AVAILABLE = True
except ImportError:
    print("AVISO: scipy nao disponivel. Instalando seria ideal.")
    SCIPY_AVAILABLE = False

def extract_all_gmeans(log_path: str) -> Dict[str, List[float]]:
    """Extrai todos os valores de G-mean por modelo"""

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    gmeans = {
        'GBML': [],
        'ACDWM': [],
        'HAT': [],
        'ARF': [],
        'SRP': []
    }

    current_dataset = None

    for line in lines:
        if 'Stream:' in line and 'RBF_' in line:
            match = re.search(r'Stream: (RBF_\w+)', line)
            if match:
                current_dataset = match.group(1)
                continue

        if not current_dataset:
            continue

        # GBML
        if 'GBMLEvaluator[GBML]' in line and '[RESULTADO]' in line:
            match = re.search(r'G-mean: ([\d.]+)', line)
            if match:
                gmeans['GBML'].append(float(match.group(1)))

        # ACDWM
        if 'baseline_acdwm' in line and 'Chunk' in line and 'G-mean:' in line:
            match = re.search(r'G-mean: ([\d.]+)', line)
            if match:
                gmeans['ACDWM'].append(float(match.group(1)))

        # River models
        if 'RiverEvaluator[' in line and '[RESULTADO]' in line:
            match_model = re.search(r'RiverEvaluator\[(\w+)\]', line)
            match_gmean = re.search(r'G-mean: ([\d.]+)', line)

            if match_model and match_gmean:
                model = match_model.group(1)
                if model in ['HAT', 'ARF', 'SRP']:
                    gmeans[model].append(float(match_gmean.group(1)))

    return gmeans

def calculate_descriptive_stats(data: List[float]) -> Dict:
    """Calcula estatisticas descritivas"""
    data = np.array(data)

    return {
        'n': len(data),
        'mean': np.mean(data),
        'std': np.std(data, ddof=1),
        'sem': np.std(data, ddof=1) / np.sqrt(len(data)),
        'median': np.median(data),
        'min': np.min(data),
        'max': np.max(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25)
    }

def confidence_interval(data: List[float], confidence=0.95) -> Tuple[float, float]:
    """Calcula intervalo de confianca"""
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(n)

    # Usa distribuicao t de Student para amostras pequenas
    from scipy.stats import t
    df = n - 1
    t_critical = t.ppf((1 + confidence) / 2, df)

    margin = t_critical * sem
    return (mean - margin, mean + margin)

def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calcula tamanho de efeito de Cohen's d"""
    g1 = np.array(group1)
    g2 = np.array(group2)

    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    d = (np.mean(g1) - np.mean(g2)) / pooled_std
    return d

def interpret_cohens_d(d: float) -> str:
    """Interpreta tamanho de efeito de Cohen's d"""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligivel"
    elif d_abs < 0.5:
        return "pequeno"
    elif d_abs < 0.8:
        return "medio"
    else:
        return "grande"

def test_normality(data: List[float], name: str) -> Tuple[bool, float]:
    """Testa normalidade usando Shapiro-Wilk"""
    if not SCIPY_AVAILABLE:
        return None, None

    data = np.array(data)

    if len(data) < 3:
        return None, None

    statistic, p_value = shapiro(data)
    is_normal = p_value > 0.05

    return is_normal, p_value

def pairwise_comparison(gmeans: Dict[str, List[float]], alpha=0.05) -> Dict:
    """Realiza comparacoes pareadas entre todos os modelos"""

    if not SCIPY_AVAILABLE:
        return {}

    models = ['GBML', 'ACDWM', 'HAT', 'ARF', 'SRP']
    results = {}

    # Correcao de Bonferroni para multiplas comparacoes
    n_comparisons = len(models) * (len(models) - 1) / 2
    bonferroni_alpha = alpha / n_comparisons

    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            data1 = np.array(gmeans[model1])
            data2 = np.array(gmeans[model2])

            # Mann-Whitney U test (nao parametrico, mais robusto)
            statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')

            # Calcula Cohen's d
            d = cohens_d(data1, data2)

            # Verifica significancia
            is_significant = p_value < bonferroni_alpha

            results[f"{model1}_vs_{model2}"] = {
                'p_value': p_value,
                'p_value_bonferroni': bonferroni_alpha,
                'is_significant': is_significant,
                'cohens_d': d,
                'effect_size': interpret_cohens_d(d),
                'mean_diff': np.mean(data1) - np.mean(data2)
            }

    return results

def overall_test(gmeans: Dict[str, List[float]], alpha=0.05) -> Dict:
    """Teste global (ANOVA ou Kruskal-Wallis)"""

    if not SCIPY_AVAILABLE:
        return {}

    models = ['GBML', 'ACDWM', 'HAT', 'ARF', 'SRP']
    data_groups = [np.array(gmeans[model]) for model in models]

    # Teste de Kruskal-Wallis (nao parametrico, mais robusto para dados nao normais)
    h_statistic, p_value = kruskal(*data_groups)

    # Tambem testa ANOVA para comparacao
    f_statistic, anova_p = f_oneway(*data_groups)

    return {
        'kruskal_h': h_statistic,
        'kruskal_p': p_value,
        'kruskal_significant': p_value < alpha,
        'anova_f': f_statistic,
        'anova_p': anova_p,
        'anova_significant': anova_p < alpha
    }

def generate_statistical_report(gmeans: Dict[str, List[float]], output_path: str):
    """Gera relatorio estatistico completo"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("ANALISE ESTATISTICA COMPARATIVA DOS MODELOS\n")
        f.write("=" * 100 + "\n\n")

        f.write("Objetivo: Determinar se as diferencas de performance sao estatisticamente significativas\n")
        f.write("Metrica analisada: G-mean\n")
        f.write("Nivel de significancia: alpha = 0.05\n")
        f.write("Correcao para multiplas comparacoes: Bonferroni\n\n")

        # 1. Estatisticas descritivas
        f.write("=" * 100 + "\n")
        f.write("1. ESTATISTICAS DESCRITIVAS\n")
        f.write("=" * 100 + "\n\n")

        models = ['GBML', 'ACDWM', 'HAT', 'ARF', 'SRP']
        stats_dict = {}

        for model in models:
            stats = calculate_descriptive_stats(gmeans[model])
            stats_dict[model] = stats

            f.write(f"Modelo: {model}\n")
            f.write("-" * 100 + "\n")
            f.write(f"  N:              {stats['n']}\n")
            f.write(f"  Media:          {stats['mean']:.4f}\n")
            f.write(f"  Desvio padrao:  {stats['std']:.4f}\n")
            f.write(f"  Erro padrao:    {stats['sem']:.4f}\n")
            f.write(f"  Mediana:        {stats['median']:.4f}\n")
            f.write(f"  Min:            {stats['min']:.4f}\n")
            f.write(f"  Max:            {stats['max']:.4f}\n")
            f.write(f"  Q25:            {stats['q25']:.4f}\n")
            f.write(f"  Q75:            {stats['q75']:.4f}\n")
            f.write(f"  IQR:            {stats['iqr']:.4f}\n")

            if SCIPY_AVAILABLE:
                ci_low, ci_high = confidence_interval(gmeans[model])
                f.write(f"  IC 95%:         [{ci_low:.4f}, {ci_high:.4f}]\n")

            f.write("\n")

        # 2. Teste de normalidade
        if SCIPY_AVAILABLE:
            f.write("=" * 100 + "\n")
            f.write("2. TESTE DE NORMALIDADE (Shapiro-Wilk)\n")
            f.write("=" * 100 + "\n\n")

            f.write("H0: Os dados seguem distribuicao normal\n")
            f.write("Se p > 0.05, aceita H0 (dados normais)\n\n")

            all_normal = True
            for model in models:
                is_normal, p_value = test_normality(gmeans[model], model)
                if is_normal is not None:
                    status = "Normal" if is_normal else "Nao normal"
                    f.write(f"{model:10s}: p = {p_value:.4f} -> {status}\n")
                    if not is_normal:
                        all_normal = False

            f.write(f"\nConclusao: {'Todos os dados seguem distribuicao normal' if all_normal else 'Dados NAO seguem distribuicao normal'}\n")
            f.write(f"Recomendacao: {'Usar testes parametricos (ANOVA)' if all_normal else 'Usar testes nao parametricos (Kruskal-Wallis)'}\n\n")

        # 3. Teste global
        if SCIPY_AVAILABLE:
            f.write("=" * 100 + "\n")
            f.write("3. TESTE GLOBAL DE DIFERENCAS\n")
            f.write("=" * 100 + "\n\n")

            overall = overall_test(gmeans)

            f.write("3.1. Teste de Kruskal-Wallis (nao parametrico)\n")
            f.write("-" * 100 + "\n")
            f.write("H0: Todos os modelos tem a mesma distribuicao de G-mean\n")
            f.write(f"H estatistica: {overall['kruskal_h']:.4f}\n")
            f.write(f"p-valor:       {overall['kruskal_p']:.6f}\n")
            f.write(f"Resultado:     {'REJEITA H0 (diferencas significativas)' if overall['kruskal_significant'] else 'NAO REJEITA H0 (sem diferencas significativas)'}\n\n")

            f.write("3.2. ANOVA (parametrico, para comparacao)\n")
            f.write("-" * 100 + "\n")
            f.write("H0: Todos os modelos tem a mesma media de G-mean\n")
            f.write(f"F estatistica: {overall['anova_f']:.4f}\n")
            f.write(f"p-valor:       {overall['anova_p']:.6f}\n")
            f.write(f"Resultado:     {'REJEITA H0 (diferencas significativas)' if overall['anova_significant'] else 'NAO REJEITA H0 (sem diferencas significativas)'}\n\n")

        # 4. Comparacoes pareadas
        if SCIPY_AVAILABLE:
            f.write("=" * 100 + "\n")
            f.write("4. COMPARACOES PAREADAS (Mann-Whitney U com correcao de Bonferroni)\n")
            f.write("=" * 100 + "\n\n")

            pairwise = pairwise_comparison(gmeans)

            n_comparisons = len(pairwise)
            bonferroni_alpha = 0.05 / n_comparisons

            f.write(f"Numero de comparacoes: {n_comparisons}\n")
            f.write(f"Alpha original: 0.05\n")
            f.write(f"Alpha corrigido (Bonferroni): {bonferroni_alpha:.6f}\n\n")

            f.write(f"{'Comparacao':<20} | {'Diff Media':>11} | {'p-valor':>10} | {'Signif':>7} | {'Cohen d':>9} | {'Tamanho':>12} |\n")
            f.write("-" * 100 + "\n")

            for comparison, result in sorted(pairwise.items()):
                model1, model2 = comparison.split('_vs_')
                diff = result['mean_diff']
                p = result['p_value']
                sig = "SIM" if result['is_significant'] else "NAO"
                d = result['cohens_d']
                effect = result['effect_size']

                f.write(f"{comparison:<20} | {diff:>+11.4f} | {p:>10.6f} | {sig:>7} | {d:>+9.3f} | {effect:>12} |\n")

            f.write("\n")

        # 5. Analise especifica do GBML
        f.write("=" * 100 + "\n")
        f.write("5. ANALISE ESPECIFICA: GBML vs OUTROS MODELOS\n")
        f.write("=" * 100 + "\n\n")

        f.write("Questao: O GBML e estatisticamente diferente dos outros modelos?\n\n")

        gbml_comparisons = {k: v for k, v in pairwise.items() if 'GBML' in k}

        f.write(f"{'GBML vs':<15} | {'Diff':>10} | {'p-valor':>10} | {'Signif?':>8} | {'Cohen d':>9} | {'Tamanho':>12} |\n")
        f.write("-" * 100 + "\n")

        significant_count = 0
        for comparison, result in sorted(gbml_comparisons.items()):
            other_model = comparison.replace('GBML_vs_', '').replace('_vs_GBML', '')

            # Ajusta sinal da diferenca para sempre mostrar GBML - Outro
            if comparison.startswith('GBML'):
                diff = result['mean_diff']
            else:
                diff = -result['mean_diff']

            p = result['p_value']
            sig = "SIM" if result['is_significant'] else "NAO"
            d = result['cohens_d'] if comparison.startswith('GBML') else -result['cohens_d']
            effect = result['effect_size']

            f.write(f"{'GBML vs ' + other_model:<15} | {diff:>+10.4f} | {p:>10.6f} | {sig:>8} | {d:>+9.3f} | {effect:>12} |\n")

            if result['is_significant']:
                significant_count += 1

        f.write("\n")
        f.write(f"Total de comparacoes: {len(gbml_comparisons)}\n")
        f.write(f"Diferencas significativas: {significant_count}\n")
        f.write(f"Diferencas NAO significativas: {len(gbml_comparisons) - significant_count}\n\n")

        # 6. Interpretacao dos intervalos de confianca
        if SCIPY_AVAILABLE:
            f.write("=" * 100 + "\n")
            f.write("6. ANALISE DE INTERVALOS DE CONFIANCA (95%)\n")
            f.write("=" * 100 + "\n\n")

            f.write("Intervalos de confianca para a media de G-mean:\n\n")
            f.write(f"{'Modelo':<10} | {'Media':>8} | {'IC 95%':>25} | {'Largura':>10} |\n")
            f.write("-" * 100 + "\n")

            intervals = {}
            for model in models:
                mean = stats_dict[model]['mean']
                ci_low, ci_high = confidence_interval(gmeans[model])
                width = ci_high - ci_low
                intervals[model] = (ci_low, ci_high)

                f.write(f"{model:<10} | {mean:>8.4f} | [{ci_low:>8.4f}, {ci_high:>8.4f}] | {width:>10.4f} |\n")

            f.write("\n")
            f.write("Analise de overlap dos intervalos de confianca:\n")
            f.write("-" * 100 + "\n\n")

            # Checa overlap GBML vs outros
            gbml_ic = intervals['GBML']

            for model in ['ACDWM', 'HAT', 'ARF', 'SRP']:
                other_ic = intervals[model]

                # Verifica overlap
                has_overlap = not (gbml_ic[1] < other_ic[0] or other_ic[1] < gbml_ic[0])

                if has_overlap:
                    # Calcula percentual de overlap
                    overlap_start = max(gbml_ic[0], other_ic[0])
                    overlap_end = min(gbml_ic[1], other_ic[1])
                    overlap_size = overlap_end - overlap_start
                    gbml_size = gbml_ic[1] - gbml_ic[0]
                    overlap_pct = (overlap_size / gbml_size) * 100

                    f.write(f"GBML vs {model}: OVERLAP de {overlap_pct:.1f}% do IC do GBML\n")
                else:
                    f.write(f"GBML vs {model}: SEM OVERLAP (diferencas claras)\n")

            f.write("\n")

        # 7. Conclusoes
        f.write("=" * 100 + "\n")
        f.write("7. CONCLUSOES E RECOMENDACOES\n")
        f.write("=" * 100 + "\n\n")

        if SCIPY_AVAILABLE:
            # Analisa se GBML e estatisticamente equivalente
            gbml_vs_best = pairwise.get('ACDWM_vs_GBML') or pairwise.get('GBML_vs_ACDWM')

            if gbml_vs_best and not gbml_vs_best['is_significant']:
                f.write("7.1. GBML E ESTATISTICAMENTE EQUIVALENTE AO MELHOR MODELO (ACDWM)\n")
                f.write("-" * 100 + "\n")
                f.write(f"p-valor: {gbml_vs_best['p_value']:.6f} > {gbml_vs_best['p_value_bonferroni']:.6f} (alpha corrigido)\n")
                f.write(f"Tamanho de efeito: {gbml_vs_best['effect_size']}\n")
                f.write(f"Cohen's d: {abs(gbml_vs_best['cohens_d']):.3f}\n\n")
                f.write("RECOMENDACAO: A diferenca de performance NAO e estatisticamente significativa.\n")
                f.write("O GBML oferece explicabilidade superior atraves de regras interpretaveis.\n")
                f.write("JUSTIFICA-SE o uso do GBML considerando o trade-off explicabilidade vs performance.\n\n")
            else:
                f.write("7.1. GBML TEM PERFORMANCE SIGNIFICATIVAMENTE INFERIOR AO MELHOR MODELO\n")
                f.write("-" * 100 + "\n")
                if gbml_vs_best:
                    f.write(f"p-valor: {gbml_vs_best['p_value']:.6f} < {gbml_vs_best['p_value_bonferroni']:.6f} (alpha corrigido)\n")
                    f.write(f"Tamanho de efeito: {gbml_vs_best['effect_size']}\n")
                    f.write(f"Cohen's d: {abs(gbml_vs_best['cohens_d']):.3f}\n\n")
                f.write("RECOMENDACAO: A diferenca de performance E estatisticamente significativa.\n")
                f.write("Deve-se avaliar cuidadosamente o trade-off explicabilidade vs performance.\n")
                f.write("Considere melhorar hiperparametros do GBML ou usar em contextos onde explicabilidade e critica.\n\n")

            # Analisa equivalencia com outros modelos
            f.write("7.2. EQUIVALENCIA ESTATISTICA DO GBML COM OUTROS MODELOS\n")
            f.write("-" * 100 + "\n\n")

            equivalent_models = []
            for model in ['HAT', 'ARF', 'SRP']:
                comp_key = f"GBML_vs_{model}" if f"GBML_vs_{model}" in pairwise else f"{model}_vs_GBML"
                if comp_key in pairwise:
                    result = pairwise[comp_key]
                    if not result['is_significant']:
                        equivalent_models.append(model)
                        f.write(f"GBML vs {model}: NAO ha diferenca significativa (p={result['p_value']:.4f})\n")

            if equivalent_models:
                f.write(f"\nGBML e estatisticamente EQUIVALENTE a: {', '.join(equivalent_models)}\n")
                f.write("Nesses casos, a explicabilidade do GBML pode justificar sua escolha.\n\n")
            else:
                f.write("\nGBML e estatisticamente INFERIOR a todos os outros modelos.\n\n")

        f.write("7.3. CONSIDERACOES FINAIS\n")
        f.write("-" * 100 + "\n\n")
        f.write("O GBML oferece vantagens unicas:\n")
        f.write("  - Regras interpretaveis (explicabilidade)\n")
        f.write("  - Rastreamento de mudancas ao longo dos chunks\n")
        f.write("  - Transparencia na tomada de decisao\n\n")
        f.write("Contextos onde GBML e preferivel:\n")
        f.write("  - Aplicacoes criticas que exigem explicabilidade (medicina, financas, legal)\n")
        f.write("  - Necessidade de auditoria e compliance\n")
        f.write("  - Entendimento de como o modelo se adapta ao drift\n")
        f.write("  - Trade-off aceitavel de performance por interpretabilidade\n\n")
        f.write("Contextos onde outros modelos sao preferiveis:\n")
        f.write("  - Performance e prioridade maxima\n")
        f.write("  - Nao ha necessidade de explicabilidade\n")
        f.write("  - Aplicacoes em tempo real com restricoes computacionais\n\n")

def main():
    log_path = 'experiment_comparison_full2.log'
    output_path = 'statistical_analysis_report.txt'

    print("ANALISE ESTATISTICA COMPARATIVA")
    print("=" * 100)
    print()

    if not SCIPY_AVAILABLE:
        print("AVISO: scipy nao instalado. Apenas estatisticas descritivas serao calculadas.")
        print("Para analise completa, instale: pip install scipy")
        print()

    print(f"Lendo log: {log_path}")
    gmeans = extract_all_gmeans(log_path)

    print(f"\nG-means extraidos por modelo:")
    for model, values in gmeans.items():
        print(f"  {model:8s}: {len(values)} valores")

    print(f"\nGerando relatorio estatistico em: {output_path}")
    generate_statistical_report(gmeans, output_path)
    print("Relatorio gerado com sucesso!")
    print()

    # Exibe resumo rapido
    print("=" * 100)
    print("RESUMO RAPIDO - Teste de Equivalencia GBML vs Outros")
    print("=" * 100)
    print()

    if SCIPY_AVAILABLE:
        pairwise = pairwise_comparison(gmeans)

        print(f"{'Comparacao':<20} | {'p-valor':>10} | {'Significativo?':>15} | {'Conclusao':>30} |")
        print("-" * 100)

        for model in ['ACDWM', 'HAT', 'ARF', 'SRP']:
            comp_key = f"GBML_vs_{model}" if f"GBML_vs_{model}" in pairwise else f"{model}_vs_GBML"
            if comp_key in pairwise:
                result = pairwise[comp_key]
                sig = "SIM" if result['is_significant'] else "NAO"
                conclusion = "GBML inferior" if result['is_significant'] else "GBML equivalente"
                print(f"{'GBML vs ' + model:<20} | {result['p_value']:>10.6f} | {sig:>15} | {conclusion:>30} |")

        print()

if __name__ == '__main__':
    main()
