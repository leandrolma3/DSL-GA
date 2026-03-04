#!/usr/bin/env python3
"""Regenerate figures for paper improvements."""

import pandas as pd
import numpy as np
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
})

OUT_DIR = 'paper/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Datasets excluded from all analyses
EXCLUDED_DATASETS = ['CovType', 'IntelLabSensors', 'PokerHand', 'Shuttle']


def regenerate_metrics_by_drift():
    """Regenerate fig_metrics_by_drift.pdf with ACDWM included."""
    df = pd.read_csv('all_models_results.csv')
    df = df[~df['dataset'].isin(EXCLUDED_DATASETS)]
    binary = df[(df['config'] == 'chunk_500') & (df['is_binary'] == True)]

    drift_order = ['abrupt', 'gradual', 'noisy', 'stationary', 'real']
    drift_labels = ['Abrupt', 'Gradual', 'Noisy', 'Stationary', 'Real']
    models = ['EGIS', 'ARF', 'HAT', 'SRP', 'ROSE', 'eRulesD2S', 'CDCMS', 'ACDWM']
    model_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#795548', '#607D8B', '#00BCD4']

    data = []
    for dt in drift_order:
        subset = binary[binary['drift_type'] == dt]
        for m in models:
            vals = pd.to_numeric(subset[m], errors='coerce').dropna()
            data.append({'drift': dt, 'model': m, 'gmean': vals.mean() if len(vals) > 0 else 0})

    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(drift_order))
    width = 0.10
    offsets = np.arange(len(models)) - len(models) / 2 + 0.5

    for i, m in enumerate(models):
        vals = [plot_df[(plot_df['drift'] == dt) & (plot_df['model'] == m)]['gmean'].values[0]
                for dt in drift_order]
        ax.bar(x + offsets[i] * width, vals, width, label=m, color=model_colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(drift_labels)
    ax.set_ylabel('G-Mean')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=len(models), frameon=False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_metrics_by_drift.pdf')
    fig.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def generate_violin_plot():
    """Violin plot of RIR, AMS, and TCS by drift type."""
    try:
        trans = pd.read_csv('paper_data/egis_transition_metrics.csv')
    except FileNotFoundError:
        print("ERROR: egis_transition_metrics.csv not found")
        return

    # Exclude removed datasets
    trans = trans[~trans['dataset'].isin(EXCLUDED_DATASETS)]

    # Filter to chunk_500
    if 'config' in trans.columns:
        trans = trans[trans['config'] == 'chunk_500']

    # Reclassify noisy datasets
    noise_datasets = [
        'SEA_Abrupt_Chain_Noise', 'STAGGER_Abrupt_Chain_Noise',
        'AGRAWAL_Abrupt_Simple_Severe_Noise', 'SINE_Abrupt_Recurring_Noise',
        'RBF_Abrupt_Blip_Noise', 'RBF_Gradual_Severe_Noise',
        'HYPERPLANE_Gradual_Noise', 'RANDOMTREE_Gradual_Noise'
    ]
    trans.loc[trans['dataset'].isin(noise_datasets), 'drift_type'] = 'noisy'

    drift_order = ['abrupt', 'gradual', 'noisy', 'stationary', 'real']
    drift_labels = ['Abrupt', 'Gradual', 'Noisy', 'Stationary', 'Real']

    # Build data for violin with 3 metrics
    data = []
    for _, row in trans.iterrows():
        dt = row.get('drift_type', '')
        if dt not in drift_order:
            continue
        if pd.notna(row.get('RIR')):
            data.append({'Drift Type': dt.capitalize(), 'Metric': 'RIR', 'Value': row['RIR']})
        if pd.notna(row.get('AMS')):
            data.append({'Drift Type': dt.capitalize(), 'Metric': 'AMS', 'Value': row['AMS']})
        if pd.notna(row.get('TCS')):
            data.append({'Drift Type': dt.capitalize(), 'Metric': 'TCS', 'Value': row['TCS']})

    vdf = pd.DataFrame(data)
    if len(vdf) == 0:
        print("No data for violin plot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = {'RIR': '#2196F3', 'AMS': '#4CAF50', 'TCS': '#FF9800'}

    sns.violinplot(data=vdf, x='Drift Type', y='Value', hue='Metric',
                   split=False, inner='quartile', palette=palette,
                   order=[d.capitalize() for d in drift_order], ax=ax,
                   density_norm='width', cut=0)

    ax.set_ylabel('Metric Value')
    ax.set_xlabel('Drift Type')
    ax.set_ylim(0, 1.05)
    ax.legend(title='', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, 'fig_violin_rir_ams_tcs.pdf')
    fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved violin plot: {path}")


def generate_transition_evolution_stagger():
    """Point 7: Regenerate fig4_transition_evolution.pdf with STAGGER_Abrupt_Chain."""
    try:
        trans = pd.read_csv('paper_data/egis_transition_metrics.csv')
    except FileNotFoundError:
        print("ERROR: egis_transition_metrics.csv not found")
        return

    # Filter to STAGGER_Abrupt_Chain, chunk_500
    mask = (trans['dataset'] == 'STAGGER_Abrupt_Chain')
    if 'config' in trans.columns:
        mask = mask & (trans['config'] == 'chunk_500')
    dataset_df = trans[mask].copy()

    if len(dataset_df) < 2:
        print(f"Insufficient data for STAGGER_Abrupt_Chain (found {len(dataset_df)} rows)")
        # Fallback to original
        return

    dataset_df = dataset_df.sort_values('chunk_from')

    fig, ax = plt.subplots(figsize=(8, 5))
    chunks = dataset_df['chunk_from'].values + 0.5
    tcs = dataset_df['TCS'].values
    rir = dataset_df['RIR'].values
    ams = dataset_df['AMS'].values

    ax.plot(chunks, tcs, 'b-o', label='TCS', linewidth=2, markersize=6)
    ax.plot(chunks, rir, 'r-s', label='RIR', linewidth=2, markersize=5)
    ax.plot(chunks, ams, 'g-^', label='AMS', linewidth=2, markersize=5)

    # STAGGER concept changes at chunks ~7 and ~14 (for 500-instance chunks in 11500-instance stream)
    for dp in [6.5, 13.5]:
        ax.axvline(x=dp, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
        ax.annotate('Drift', xy=(dp, 0.95), fontsize=8, alpha=0.7, ha='center')

    ax.set_xlabel('Chunk Transition', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Transition Metrics Evolution (STAGGER Abrupt Chain)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_transition_evolution.pdf')
    fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def copy_rule_evolution_matrix():
    """Point 7: Copy STAGGER rule evolution matrix to paper figures."""
    src = Path("analysis_batch_output/rule_diff_reports/chunk_500/batch_1/STAGGER_Abrupt_Chain_run_1_matrix.png")
    dst = Path(OUT_DIR) / "fig_rule_evolution_matrix.png"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")
    else:
        print(f"WARNING: {src} not found")


def generate_performance_over_time():
    """G-Mean per chunk for multiple models (Krawczyk-style). X-axis in instances."""
    base = Path("experiments_unified/chunk_500/batch_1")
    datasets = ['STAGGER_Abrupt_Chain', 'SEA_Abrupt_Simple']
    dataset_labels = ['STAGGER Abrupt Chain', 'SEA Abrupt Simple']
    egis_chunk_size = 500
    # Known drift points (in instances) for chunk_500
    drift_points_instances = {
        'STAGGER_Abrupt_Chain': [(3500, 'Drift'), (7000, 'Drift')],
        'SEA_Abrupt_Simple': [(5500, 'Drift')],
    }

    model_colors = {
        'EGIS': '#2196F3', 'ARF': '#4CAF50', 'HAT': '#FF9800',
        'SRP': '#9C27B0', 'ROSE (prequential)': '#F44336', 'eRulesD2S': '#795548',
        'CDCMS': '#607D8B', 'ACDWM': '#00BCD4'
    }
    model_markers = {
        'EGIS': 'o', 'ARF': 's', 'HAT': '^', 'SRP': 'D',
        'ROSE (prequential)': 'v', 'eRulesD2S': 'P', 'ACDWM': '*'
    }
    model_linestyles = {
        'EGIS': '-', 'ARF': '--', 'HAT': '-.', 'SRP': ':',
        'ROSE (prequential)': '-', 'eRulesD2S': '--', 'ACDWM': '-.'
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

    for idx, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
        ax = axes[idx]
        ds_dir = base / ds / "run_1"

        if not ds_dir.exists():
            ax.set_title(f"{ds_label} (no data)")
            continue

        # EGIS: chunk_metrics.json (0-indexed chunks)
        cm_file = ds_dir / "chunk_metrics.json"
        if cm_file.exists():
            with open(cm_file) as f:
                cm = json.load(f)
            x_egis = [(c['chunk'] + 1) * egis_chunk_size for c in cm]
            gmean_egis = [c['test_gmean'] for c in cm]
            ax.plot(x_egis, gmean_egis, color=model_colors['EGIS'], label='EGIS',
                    linewidth=2, marker=model_markers['EGIS'], markersize=5,
                    markevery=3, linestyle=model_linestyles['EGIS'])

        # River models: ARF, HAT, SRP (1-indexed chunks)
        for model_name in ['ARF', 'HAT', 'SRP']:
            river_file = ds_dir / f"river_{model_name}_results.csv"
            if river_file.exists():
                rdf = pd.read_csv(river_file)
                x_river = rdf['chunk'] * egis_chunk_size
                ax.plot(x_river, rdf['test_gmean'], color=model_colors[model_name],
                        label=model_name, linewidth=1.5, alpha=0.8,
                        marker=model_markers[model_name], markersize=5,
                        markevery=3, linestyle=model_linestyles[model_name])

        # ROSE: rose_chunk_eval_results.csv (cumulative/prequential G-Mean)
        rose_file = ds_dir / "rose_chunk_eval_results.csv"
        if rose_file.exists():
            rdf = pd.read_csv(rose_file)
            if 'G-Mean' in rdf.columns and 'learning evaluation instances' in rdf.columns:
                rdf['learning evaluation instances'] = pd.to_numeric(rdf['learning evaluation instances'], errors='coerce')
                rdf['G-Mean'] = pd.to_numeric(rdf['G-Mean'], errors='coerce')
                rdf = rdf.dropna(subset=['learning evaluation instances', 'G-Mean'])
                x_rose = rdf['learning evaluation instances']
                ax.plot(x_rose, rdf['G-Mean'], color=model_colors['ROSE (prequential)'],
                        label='ROSE (prequential)', linewidth=1.5, alpha=0.8,
                        marker=model_markers['ROSE (prequential)'], markersize=5,
                        markevery=3, linestyle=model_linestyles['ROSE (prequential)'])

        # eRulesD2S: convert chunk index to instances
        erul_file = ds_dir / "erulesd2s_results.csv"
        if erul_file.exists():
            edf = pd.read_csv(erul_file)
            n_erules_chunks = len(edf)
            total_instances_map = {
                'STAGGER_Abrupt_Chain': 12000,
                'SEA_Abrupt_Simple': 12000,
            }
            total_inst = total_instances_map.get(ds, n_erules_chunks * egis_chunk_size)
            erules_chunk_size = total_inst / n_erules_chunks
            x_erules = edf['chunk'] * erules_chunk_size
            ax.plot(x_erules, edf['test_gmean'], color=model_colors['eRulesD2S'],
                    label='eRulesD2S', linewidth=1.5, alpha=0.8,
                    marker=model_markers['eRulesD2S'], markersize=5,
                    markevery=3, linestyle=model_linestyles['eRulesD2S'])

        # ACDWM (1-indexed chunks)
        acdwm_file = ds_dir / "acdwm_results.csv"
        if acdwm_file.exists():
            adf = pd.read_csv(acdwm_file)
            x_acdwm = adf['chunk'] * egis_chunk_size
            ax.plot(x_acdwm, adf['test_gmean'], color=model_colors['ACDWM'],
                    label='ACDWM', linewidth=1.5, alpha=0.8,
                    marker=model_markers['ACDWM'], markersize=5,
                    markevery=3, linestyle=model_linestyles['ACDWM'])

        # Drift annotations with text boxes
        for dp, label in drift_points_instances.get(ds, []):
            ax.axvline(x=dp, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.annotate(label, xy=(dp, 1.02), fontsize=7, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='red', alpha=0.8))

        ax.set_title(ds_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Instances')
        if idx == 0:
            ax.set_ylabel('G-Mean')
        ax.set_ylim(0, 1.05)
        ax.grid(which='major', alpha=0.4, linestyle='-')
        ax.grid(which='minor', alpha=0.15, linestyle=':')
        ax.minorticks_on()

    # Shared legend below panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.08), frameon=True, fontsize=8)

    plt.subplots_adjust(bottom=0.18)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    path = os.path.join(OUT_DIR, 'fig_performance_over_time.pdf')
    fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def generate_transition_matrix_2x3():
    """Generate 2x3 composite figure: STAGGER_Abrupt_Chain + SEA_Gradual_Simple_Slow.
    Columns: Transition Metrics Evolution | Rule Evolution Matrix | Rule Components Heatmap
    """
    import json
    from pathlib import Path
    from collections import defaultdict

    datasets = ['STAGGER_Abrupt_Chain', 'SEA_Gradual_Simple_Slow']
    labels = ['STAGGER Abrupt Chain', 'SEA Gradual Simple Slow']
    drift_points = {
        'STAGGER_Abrupt_Chain': [6.5, 13.5],
        'SEA_Gradual_Simple_Slow': [5.5, 14.5],
    }
    base = Path("experiments_unified/chunk_500/batch_1")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    for row_idx, (ds, label) in enumerate(zip(datasets, labels)):
        # --- Column 0: Transition Metrics Evolution ---
        ax = axes[row_idx, 0]
        try:
            trans = pd.read_csv('paper_data/egis_transition_metrics.csv')
            trans = trans[~trans['dataset'].isin(EXCLUDED_DATASETS)]
            mask = (trans['dataset'] == ds)
            if 'config' in trans.columns:
                mask = mask & (trans['config'] == 'chunk_500')
            ds_df = trans[mask].sort_values('chunk_from')

            if len(ds_df) >= 2:
                chunks = ds_df['chunk_from'].values + 0.5
                ax.plot(chunks, ds_df['TCS'].values, 'b-o', label='TCS', linewidth=2, markersize=4)
                ax.plot(chunks, ds_df['RIR'].values, 'r-s', label='RIR', linewidth=2, markersize=4)
                ax.plot(chunks, ds_df['AMS'].values, 'g-^', label='AMS', linewidth=2, markersize=4)
                for dp in drift_points.get(ds, []):
                    ax.axvline(x=dp, color='gray', linestyle='--', alpha=0.6)
                ax.set_ylim(0, 1.05)
                ax.legend(fontsize=7, loc='upper right')
            ax.set_title(f'{label}\nTransition Metrics', fontsize=10)
            ax.set_xlabel('Chunk Transition', fontsize=9)
            ax.set_ylabel('Metric Value', fontsize=9)
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'No data\n{e}', ha='center', va='center', transform=ax.transAxes)

        # --- Column 1: Rule Evolution Matrix ---
        ax = axes[row_idx, 1]
        try:
            run_dir = base / ds / "run_1"
            hist_files = list(run_dir.glob("RulesHistory_*.txt"))
            if hist_files:
                with open(hist_files[0], 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse chunks - extract rules per chunk
                chunk_sections = re.split(r'--- Chunk (\d+) \(Trained\)', content)
                chunks_rules = {}
                for i in range(1, len(chunk_sections), 2):
                    chunk_num = int(chunk_sections[i])
                    section = chunk_sections[i + 1] if i + 1 < len(chunk_sections) else ""
                    rules = re.findall(r'IF (.+?) THEN Class (\d+)', section)
                    chunks_rules[chunk_num] = [f"IF {r[0]} THEN Class {r[1]}" for r in rules]

                # Compute evolution counts
                sorted_chunks = sorted(chunks_rules.keys())
                transitions = []
                counts_data = {'Unchanged': [], 'Modified': [], 'New': [], 'Deleted': []}

                for i in range(len(sorted_chunks) - 1):
                    prev_rules = set(chunks_rules[sorted_chunks[i]])
                    curr_rules = set(chunks_rules[sorted_chunks[i + 1]])
                    unchanged = len(prev_rules & curr_rules)
                    deleted = len(prev_rules - curr_rules)
                    new = len(curr_rules - prev_rules)
                    modified = 0  # Simplified: exact match only
                    counts_data['Unchanged'].append(unchanged)
                    counts_data['Modified'].append(modified)
                    counts_data['New'].append(new)
                    counts_data['Deleted'].append(deleted)
                    transitions.append(f'{sorted_chunks[i]}->{sorted_chunks[i+1]}')

                df_matrix = pd.DataFrame(counts_data, index=transitions).T
                sns.heatmap(df_matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                            linewidths=0.5, cbar_kws={'shrink': 0.8})
                ax.set_title(f'{label}\nRule Evolution Matrix', fontsize=10)
                ax.set_xlabel('Chunk Transition', fontsize=9)
                ax.set_ylabel('')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
            else:
                ax.text(0.5, 0.5, 'No RulesHistory', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error\n{e}', ha='center', va='center', transform=ax.transAxes)

        # --- Column 2: Rule Components Heatmap ---
        ax = axes[row_idx, 2]
        try:
            json_path = base / ds / "run_1" / "rule_details_per_chunk.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    rule_details = json.load(f)

                num_chunks = len(rule_details)
                logical_ops = [len(c.get("logical_ops", [])) for c in rule_details]
                comp_ops = [len(c.get("comparison_ops", [])) for c in rule_details]
                thresholds = [len(c.get("numeric_thresholds", [])) for c in rule_details]
                features = [len(c.get("features", [])) for c in rule_details]
                cat_values = [len(c.get("categorical_values_used", [])) for c in rule_details]

                data_matrix = np.array([logical_ops, comp_ops, thresholds, features, cat_values])
                y_labels = ["Logical Ops", "Comparison Ops", "Numeric Thresh.", "Features", "Categorical"]

                sns.heatmap(data_matrix, annot=True, fmt="d", cmap="viridis", ax=ax,
                            xticklabels=[f'C{i}' for i in range(num_chunks)],
                            yticklabels=y_labels, cbar_kws={'shrink': 0.8})
                ax.set_title(f'{label}\nRule Components', fontsize=10)
                ax.set_xlabel('Chunk', fontsize=9)
                ax.set_ylabel('')
            else:
                ax.text(0.5, 0.5, 'No rule_details', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error\n{e}', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_transition_matrix_2x3.pdf')
    fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    print("=== Regenerating fig_metrics_by_drift.pdf (with ACDWM) ===")
    regenerate_metrics_by_drift()

    print("\n=== Point 4: Generating RIR/AMS violin plot ===")
    generate_violin_plot()

    print("\n=== Point 7: Regenerating transition evolution with STAGGER ===")
    generate_transition_evolution_stagger()

    print("\n=== Point 7: Copying rule evolution matrix ===")
    copy_rule_evolution_matrix()

    print("\n=== Point 8: Generating performance over time figure ===")
    generate_performance_over_time()

    print("\n=== Generating 2x3 transition matrix figure ===")
    generate_transition_matrix_2x3()

    print("\nAll figures done!")
