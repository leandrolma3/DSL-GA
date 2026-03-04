#!/usr/bin/env python3
"""Update LaTeX tables to include ACDWM column."""

import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    df = pd.read_csv("all_models_results.csv")
    return df

def format_val(v, best_val=None, is_best_interp=False, is_missing=False):
    if is_missing or pd.isna(v):
        return '--'
    s = f"{v:.3f}"
    if best_val is not None and abs(v - best_val) < 0.0005:
        s = f"\\textbf{{{s}}}"
    return s

def generate_comparison_table(df, config, label_suffix, caption_suffix):
    models = ['EGIS', 'ARF', 'HAT', 'SRP', 'ROSE', 'eRulesD2S', 'CDCMS', 'ACDWM']
    model_headers = ['EGIS', 'ARF', 'HAT', 'SRP', 'ROSE', 'eRulesD2S', 'CDCMS', 'ACDWM']

    subset = df[df['config'] == config].copy()

    drift_groups = [
        ('Abrupt Drift', 'abrupt', True),
        ('Gradual Drift', 'gradual', True),
        ('Noisy Drift', 'noisy', True),
        ('Stationary Drift', 'stationary', True),
        ('Real Drift', 'real', True),
        ('Multiclass', None, False),
    ]

    lines = []
    lines.append(r'\begin{table*}[ht]')
    lines.append(r'\centering')

    if config == 'chunk_500':
        lines.append(r'\caption{Performance Comparison (G-Mean) - EXP-500 Configuration. Results for 42 binary datasets and 10 multiclass datasets organized by drift type. CDCMS and ACDWM have no multiclass support (shown as --). Summary statistics shown for binary-only (n=42) and all datasets (n=52). Best performance per row in bold.}')
        lines.append(r'\label{tab:binary_comparison}')
    else:
        lines.append(r'\caption{Performance Comparison (G-Mean) - EXP-1000 Configuration. Results for 42 binary datasets and 10 multiclass datasets organized by drift type. Larger chunk size (1000 instances) provides more training data per evolutionary cycle but potentially delays drift response. CDCMS and ACDWM have no multiclass support (shown as --). Summary statistics shown for binary-only (n=42) and all datasets (n=52). Best performance per row in bold.}')
        lines.append(r'\label{tab:binary_comparison_1000}')

    lines.append(r'\scriptsize')
    lines.append(r'\begin{tabular}{l|cccccccc}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Dataset} & \textbf{EGIS} & \textbf{ARF} & \textbf{HAT} & \textbf{SRP} & \textbf{ROSE} & \textbf{eRulesD2S} & \textbf{CDCMS} & \textbf{ACDWM} \\')
    lines.append(r'\midrule')

    binary_data = subset[subset['is_binary'] == True]
    multi_data = subset[subset['is_binary'] == False]

    for group_name, drift_type, is_binary in drift_groups:
        if is_binary:
            group = binary_data[binary_data['drift_type'] == drift_type].sort_values('dataset')
            count = len(group)
        else:
            group = multi_data.sort_values('dataset')
            count = len(group)

        lines.append(f'\\multicolumn{{9}}{{l}}{{\\textbf{{{group_name} ({count})}}}} \\\\')

        for _, row in group.iterrows():
            ds_name = row['dataset'].replace('_', r'\_')
            vals = []
            valid_vals = []
            for m in models:
                v = row.get(m, np.nan)
                if pd.notna(v) and v != '':
                    try:
                        valid_vals.append(float(v))
                    except (ValueError, TypeError):
                        pass

            best = max(valid_vals) if valid_vals else None

            cells = []
            for m in models:
                v = row.get(m, np.nan)
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    v = np.nan

                is_missing = pd.isna(v)
                # ACDWM and CDCMS missing for multiclass
                if not is_binary and m in ['CDCMS', 'ACDWM'] and is_missing:
                    cells.append('--')
                elif is_missing:
                    cells.append('--')
                else:
                    cells.append(format_val(v, best))

            lines.append(f'{ds_name} & {" & ".join(cells)} \\\\')

    # Summary statistics
    lines.append(r'\midrule')
    lines.append(r"\multicolumn{9}{l}{\textit{Binary only (n=42):}} \\")

    # W/L/D for EGIS vs each
    wld_parts = ['--']
    for m in models[1:]:
        egis_vals = binary_data['EGIS']
        m_vals = binary_data[m]
        mask = egis_vals.notna() & m_vals.notna()
        if m == 'ACDWM':
            # ACDWM has 41 binary (missing RBF_Stationary)
            mask = mask & (egis_vals != '') & (m_vals != '')
        w = ((egis_vals[mask].astype(float) > m_vals[mask].astype(float))).sum()
        l = ((egis_vals[mask].astype(float) < m_vals[mask].astype(float))).sum()
        d = mask.sum() - w - l
        wld_parts.append(f'{w}/{l}/{d}')
    lines.append(f'\\textbf{{EGIS W/L/D}} & {" & ".join(wld_parts)} \\\\')

    # Mean
    mean_parts = []
    for m in models:
        vals = binary_data[m].dropna().astype(float)
        mean_parts.append(f'{vals.mean():.3f}' if len(vals) > 0 else '--')
    lines.append(f'\\textbf{{Mean}} & {" & ".join(mean_parts)} \\\\')

    # Std
    std_parts = []
    for m in models:
        vals = binary_data[m].dropna().astype(float)
        std_parts.append(f'{vals.std():.3f}' if len(vals) > 0 else '--')
    lines.append(f'\\textbf{{Std}} & {" & ".join(std_parts)} \\\\')

    # Avg Rank
    rank_data = binary_data[models].copy()
    for m in models:
        rank_data[m] = pd.to_numeric(rank_data[m], errors='coerce')
    ranks = rank_data.rank(axis=1, ascending=False, method='average')
    avg_ranks = ranks.mean()
    rank_parts = [f'{avg_ranks[m]:.2f}' if not pd.isna(avg_ranks[m]) else '--' for m in models]
    lines.append(f'\\textbf{{Avg Rank}} & {" & ".join(rank_parts)} \\\\')

    # All datasets n=52
    lines.append(r'\midrule')
    lines.append(r"\multicolumn{9}{l}{\textit{All datasets (n=52):}} \\")

    # W/L/D n=52
    wld_parts52 = ['--']
    for m in models[1:]:
        egis_vals = subset['EGIS']
        m_vals = subset[m]
        mask = egis_vals.notna() & m_vals.notna()
        try:
            e = egis_vals[mask].astype(float)
            mv = m_vals[mask].astype(float)
            w = (e > mv).sum()
            l = (e < mv).sum()
            d = len(e) - w - l
            wld_parts52.append(f'{w}/{l}/{d}')
        except:
            wld_parts52.append('--')
    lines.append(f'\\textbf{{EGIS W/L/D}} & {" & ".join(wld_parts52)} \\\\')

    # Mean n=52 - for models without multiclass, count those as 0
    mean52_parts = []
    for m in models:
        if m in ['CDCMS', 'ACDWM']:
            # These don't support multiclass - use 0 for missing
            vals = subset[m].fillna(0).astype(float)
        else:
            vals = subset[m].dropna().astype(float)
        mean52_parts.append(f'{vals.mean():.3f}' if len(vals) > 0 else '--')
    lines.append(f'\\textbf{{Mean}} & {" & ".join(mean52_parts)} \\\\')

    std52_parts = []
    for m in models:
        if m in ['CDCMS', 'ACDWM']:
            vals = subset[m].fillna(0).astype(float)
        else:
            vals = subset[m].dropna().astype(float)
        std52_parts.append(f'{vals.std():.3f}' if len(vals) > 0 else '--')
    lines.append(f'\\textbf{{Std}} & {" & ".join(std52_parts)} \\\\')

    # Avg Rank n=52
    rank52_data = subset[models].copy()
    for m in models:
        rank52_data[m] = pd.to_numeric(rank52_data[m], errors='coerce')
    # For n=52 ranking, NaN models get worst rank
    ranks52 = rank52_data.rank(axis=1, ascending=False, method='average', na_option='bottom')
    avg_ranks52 = ranks52.mean()
    rank52_parts = [f'{avg_ranks52[m]:.2f}' for m in models]
    lines.append(f'\\textbf{{Avg Rank}} & {" & ".join(rank52_parts)} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')

    return '\n'.join(lines)

def compute_table_vii_values(df):
    """Compute summary values for Table VII including ACDWM."""
    models = ['EGIS', 'ARF', 'HAT', 'SRP', 'ROSE', 'eRulesD2S', 'CDCMS', 'ACDWM']

    for cfg in ['chunk_500', 'chunk_1000']:
        subset = df[df['config'] == cfg]
        binary = subset[subset['is_binary'] == True]

        print(f"\n=== {cfg} ===")
        print("Binary (n=42):")
        for m in models:
            vals = binary[m].dropna().astype(float)
            if len(vals) > 0:
                print(f"  {m}: {vals.mean():.3f} ± {vals.std():.3f} (n={len(vals)})")

        print("All (n=52):")
        for m in models:
            if m in ['CDCMS', 'ACDWM']:
                vals = subset[m].fillna(0).astype(float)
            else:
                vals = subset[m].dropna().astype(float)
            if len(vals) > 0:
                print(f"  {m}: {vals.mean():.3f} ± {vals.std():.3f}")

def compute_drift_performance_with_acdwm(df):
    """Compute Table IX drift performance including ACDWM."""
    models = ['EGIS', 'ARF', 'HAT', 'SRP', 'ROSE', 'eRulesD2S', 'CDCMS', 'ACDWM']
    binary = df[(df['config'] == 'chunk_500') & (df['is_binary'] == True)]

    print("\n=== Drift Performance with ACDWM (EXP-500) ===")
    for dt in ['abrupt', 'gradual', 'noisy', 'stationary', 'real']:
        group = binary[binary['drift_type'] == dt]
        parts = []
        for m in models:
            vals = group[m].dropna().astype(float)
            parts.append(f"{vals.mean():.3f}" if len(vals) > 0 else "--")
        print(f"  {dt.capitalize()} ({len(group)}): {' | '.join(parts)}")

    # Overall
    parts = []
    for m in models:
        vals = binary[m].dropna().astype(float)
        parts.append(f"{vals.mean():.3f}" if len(vals) > 0 else "--")
    print(f"  Overall ({len(binary)}): {' | '.join(parts)}")

def main():
    df = load_data()

    # Generate tables
    table_500 = generate_comparison_table(df, 'chunk_500', '', 'EXP-500')
    table_1000 = generate_comparison_table(df, 'chunk_1000', '_1000', 'EXP-1000')

    Path("paper/tables/table_binary_comparison.tex").write_text(table_500)
    Path("paper/tables/table_binary_comparison_1000.tex").write_text(table_1000)

    print("Saved updated comparison tables")

    compute_table_vii_values(df)
    compute_drift_performance_with_acdwm(df)

if __name__ == '__main__':
    main()
