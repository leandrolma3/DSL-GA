#!/usr/bin/env python3
"""
Generate Supplementary Material for IEEE TKDE Paper.

Reads data from paper_data/ and generates paper/supplementary_material.tex.
The SupMat contains:
  S1: Experimental Details
  S2: Complete Performance Tables
  S3: Extended Statistical Analysis
  S4: Extended Transition Metrics
  S5: EGIS Configuration Analysis
  S6: Per-Dataset Visual Analysis (optional, ~336 pages)
"""

import json
import csv
import os
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

BASE = Path(".")
PAPER_DATA = BASE / "paper_data"
PAPER = BASE / "paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
PER_DATASET = PAPER_DATA / "per_dataset_analysis"

CONFIGS = [
    "chunk_500", "chunk_1000", "chunk_2000",
    "chunk_500_penalty", "chunk_1000_penalty", "chunk_2000_penalty",
    "chunk_500_penalty_03"
]

CONFIG_LABELS = {
    "chunk_500": "EXP-500-NP",
    "chunk_1000": "EXP-1000-NP",
    "chunk_2000": "EXP-2000-NP",
    "chunk_500_penalty": "EXP-500-P",
    "chunk_1000_penalty": "EXP-1000-P",
    "chunk_2000_penalty": "EXP-2000-P",
    "chunk_500_penalty_03": "EXP-500-P03",
}

MULTICLASS_DATASETS = [
    "LED_Abrupt_Simple", "LED_Gradual_Simple", "LED_Stationary",
    "RBF_Stationary", "WAVEFORM_Abrupt_Simple",
    "WAVEFORM_Gradual_Simple", "WAVEFORM_Stationary",
]

MODELS = ["EGIS", "ARF", "SRP", "HAT", "ROSE", "ACDWM", "ERulesD2S", "CDCMS"]

DRIFT_TYPE_ORDER = ["abrupt", "gradual", "noisy", "stationary", "real"]


def escape_latex(s):
    """Escape LaTeX special characters in a string."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

def generate_preamble():
    return r"""\documentclass[lettersize,10pt]{article}

\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage[caption=false,font=normalsize,labelfont=sf,textfont=sf]{subfig}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{hyperref}
\usepackage[protrusion=true,expansion=true]{microtype}

\title{Supplementary Material:\\An Explainable Evolutionary Grammar Approach\\for Interpretable Data Stream Classification with Concept Drift}
\author{Leandro Maciel Almeida and Leandro L. Minku}
\date{}

\begin{document}
\maketitle

\noindent This supplementary material provides extended experimental results, detailed statistical analyses, and per-dataset visualizations supporting the main paper. All sections are numbered with the prefix ``S'' to distinguish them from the main manuscript.

\tableofcontents
\newpage
"""


def generate_s1_experimental_details():
    """S1: Experimental Details (dataset dimensions, configs, hyperparameters)."""
    lines = []
    lines.append(r"\section{Experimental Details}")
    lines.append(r"\label{sec:s1}")

    # S1.1 Dataset Dimensions
    lines.append(r"\subsection{Dataset Dimensions}")
    lines.append(r"\label{sec:s1_datasets}")
    lines.append(
        r"The experimental evaluation encompasses 48 datasets: 41 binary classification "
        r"and 7 multiclass datasets. Table~\ref{tab:s_drift_counts} shows the distribution "
        r"across drift types."
    )
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Dataset Distribution by Drift Type}")
    lines.append(r"\label{tab:s_drift_counts}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Drift Type} & \textbf{All} & \textbf{Binary Only} \\")
    lines.append(r"\midrule")
    lines.append(r"Abrupt & 16 & 14 \\")
    lines.append(r"Gradual & 11 & 9 \\")
    lines.append(r"Noisy & 8 & 8 \\")
    lines.append(r"Stationary & 9 & 6 \\")
    lines.append(r"Real & 4 & 4 \\")
    lines.append(r"\midrule")
    lines.append(r"Total & 48 & 41 \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    lines.append(
        r"The 7 multiclass datasets are: LED\_Abrupt\_Simple, LED\_Gradual\_Simple, "
        r"LED\_Stationary, RBF\_Stationary, WAVEFORM\_Abrupt\_Simple, "
        r"WAVEFORM\_Gradual\_Simple, and WAVEFORM\_Stationary. "
        r"IntelLabSensors was excluded from all analyses due to data quality issues."
    )
    lines.append("")

    # S1.2 Experimental Configurations
    lines.append(r"\subsection{Experimental Configurations}")
    lines.append(r"\label{sec:s1_configs}")
    lines.append(
        r"Table~\ref{tab:s_configs} summarizes the seven EGIS configurations evaluated. "
        r"All configurations use a stream of 12{,}000 instances divided into chunks of "
        r"the specified size."
    )
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{EGIS Experimental Configurations}")
    lines.append(r"\label{tab:s_configs}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Label} & \textbf{Chunk Size} & \textbf{Penalty ($\gamma$)} & \textbf{Num Chunks} & \textbf{Total Instances} \\")
    lines.append(r"\midrule")
    for cfg, label in sorted(CONFIG_LABELS.items(), key=lambda x: x[1]):
        cs = cfg.split("_")[1]
        pen = "0.3" if "03" in cfg else ("0.1" if "penalty" in cfg else "0.0")
        nc = str(12000 // int(cs))
        lines.append(f"{label} & {cs} & {pen} & {nc} & 12{{,}}000 \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # S1.3 EGIS Hyperparameters
    lines.append(r"\subsection{EGIS Hyperparameters}")
    lines.append(r"\label{sec:s1_params}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{EGIS Hyperparameter Settings}")
    lines.append(r"\label{tab:s_params}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Parameter} & \textbf{Value} \\")
    lines.append(r"\midrule")
    params = [
        ("Population size", "120"),
        ("Max generations", "200"),
        ("Elitism rate", "0.1"),
        ("Intelligent mutation rate", "0.8"),
        ("Tournament size (initial/final)", "2 / 5"),
        ("Crossover rate", "0.9"),
        ("Hill climbing iterations", "50"),
        ("Gene Therapy injection rate", "0.3"),
        ("Drift detection method", "ADWIN"),
        ("Memory management", "Adaptive sliding window"),
    ]
    for p, v in params:
        lines.append(f"{p} & {v} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def generate_perdataset_table_2000(consolidated_csv_path):
    """Generate a per-dataset G-Mean longtable for EXP-2000-NP from consolidated_results.csv.

    Returns a list of LaTeX lines.
    """
    import statistics as stat

    rows = load_csv(consolidated_csv_path)

    # Filter: config_label == EXP-2000-NP, binary only
    filtered = [
        r for r in rows
        if r["config_label"] == "EXP-2000-NP" and r["dataset"] not in MULTICLASS_DATASETS
    ]

    # Build lookup: dataset -> model -> gmean_mean
    data = {}
    drift_map = {}
    for r in filtered:
        ds = r["dataset"]
        model = r["model"]
        try:
            val = float(r["gmean_mean"])
        except (ValueError, KeyError):
            val = None
        data.setdefault(ds, {})[model] = val
        drift_map[ds] = r.get("drift_type", "unknown")

    # Group datasets by drift type
    drift_groups = defaultdict(list)
    for ds in sorted(data.keys()):
        dt = drift_map.get(ds, "unknown")
        drift_groups[dt].append(ds)

    lines = []
    # Header
    model_cols = " & ".join([f"\\textbf{{{m}}}" for m in MODELS])
    col_spec = "l" + "c" * len(MODELS)
    lines.append(f"\\begin{{longtable}}{{{col_spec}}}")
    lines.append(
        r"\caption{Per-Dataset G-Mean: Binary Datasets, Chunk Size 2000 (EXP-2000-NP)}"
        r"\label{tab:s_binary_2000} \\"
    )
    lines.append(r"\toprule")
    lines.append(f"\\textbf{{Dataset}} & {model_cols} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(f"\\textbf{{Dataset}} & {model_cols} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    all_vals = {m: [] for m in MODELS}

    for dt in DRIFT_TYPE_ORDER:
        if dt not in drift_groups:
            continue
        # Drift type group header
        ncols = 1 + len(MODELS)
        lines.append(
            f"\\midrule \\multicolumn{{{ncols}}}{{l}}"
            f"{{\\textit{{{dt.capitalize()} Drift}}}} \\\\"
        )
        lines.append(r"\midrule")

        for ds in drift_groups[dt]:
            row_vals = []
            for m in MODELS:
                v = data[ds].get(m)
                if v is not None:
                    row_vals.append(v)
                    all_vals[m].append(v)
                else:
                    row_vals.append(None)

            # Find best (highest) value in this row
            valid = [v for v in row_vals if v is not None]
            best_val = max(valid) if valid else None

            cells = []
            for v in row_vals:
                if v is None:
                    cells.append("--")
                elif best_val is not None and abs(v - best_val) < 1e-9:
                    cells.append(f"\\textbf{{{v:.3f}}}")
                else:
                    cells.append(f"{v:.3f}")

            ds_escaped = escape_latex(ds)
            cells_str = " & ".join(cells)
            lines.append(f"{ds_escaped} & {cells_str} \\\\")

    # Summary row: Mean +/- Std
    lines.append(r"\midrule")
    summary_cells = []
    for m in MODELS:
        vals = all_vals[m]
        if vals:
            mn = stat.mean(vals)
            sd = stat.stdev(vals) if len(vals) > 1 else 0.0
            summary_cells.append(f"{mn:.3f}$\\pm${sd:.3f}")
        else:
            summary_cells.append("--")
    summary_str = " & ".join(summary_cells)
    lines.append(f"\\textbf{{Mean $\\pm$ Std}} & {summary_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")

    return lines


def generate_s2_performance_tables():
    """S2: Complete Performance Tables (per-dataset G-Mean)."""
    lines = []
    lines.append(r"\section{Complete Performance Tables}")
    lines.append(r"\label{sec:s2}")
    lines.append(
        r"This section provides the complete per-dataset G-Mean performance tables "
        r"for all configurations and models. These tables extend the summary statistics "
        r"presented in the main paper."
    )
    lines.append("")

    # Include existing per-dataset tables
    for table_file, label, caption in [
        ("table_binary_comparison.tex", "EXP-500-NP", "Binary Datasets, Chunk Size 500"),
        ("table_binary_comparison_1000.tex", "EXP-1000-NP", "Binary Datasets, Chunk Size 1000"),
    ]:
        tpath = TABLES / table_file
        if tpath.exists():
            lines.append(f"\\subsection{{Per-Dataset G-Mean: {caption}}}")
            lines.append(f"\\label{{sec:s2_{label.lower().replace('-','_')}}}")
            lines.append("")
            # Read the table content and include it
            content = read_file(tpath)
            # Remove the outer table environment if it has one, to allow resizing
            lines.append(content)
            lines.append("\\clearpage")
            lines.append("")

    # Generate chunk 2000 per-dataset table programmatically
    consolidated_csv = PAPER_DATA / "consolidated_results.csv"
    if consolidated_csv.exists():
        lines.append(r"\subsection{Per-Dataset G-Mean: Binary Datasets, Chunk Size 2000}")
        lines.append(r"\label{sec:s2_exp_2000_np}")
        lines.append("")
        table_lines = generate_perdataset_table_2000(consolidated_csv)
        lines.extend(table_lines)
        lines.append("\\clearpage")
        lines.append("")

    # Generate summary table from consolidated_results.csv
    lines.append(r"\subsection{Summary Statistics by Configuration and Model}")
    lines.append(r"\label{sec:s2_summary}")
    lines.append("")

    csv_path = PAPER_DATA / "summary_by_config_model.csv"
    if csv_path.exists():
        rows = load_csv(csv_path)
        lines.append(r"\begin{longtable}{llccc}")
        lines.append(r"\caption{Mean G-Mean by Configuration and Model}\label{tab:s_summary_config} \\")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Config} & \textbf{Model} & \textbf{Mean G-Mean} & \textbf{Std} & \textbf{N} \\")
        lines.append(r"\midrule")
        lines.append(r"\endfirsthead")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Config} & \textbf{Model} & \textbf{Mean G-Mean} & \textbf{Std} & \textbf{N} \\")
        lines.append(r"\midrule")
        lines.append(r"\endhead")
        for row in rows:
            cfg = escape_latex(row.get("config_label", row.get("config", "")))
            model = row.get("model", "")
            mean = float(row.get("gmean_avg", row.get("gmean_mean_mean", 0)))
            std = float(row.get("gmean_std", row.get("gmean_mean_std", 0)))
            n = row.get("n_datasets", row.get("count", ""))
            lines.append(f"{cfg} & {model} & {mean:.4f} & {std:.4f} & {n} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")
    lines.append("")

    return "\n".join(lines)


def generate_s3_statistical_analysis():
    """S3: Extended Statistical Analysis (all-48 ranking, detailed pairwise)."""
    lines = []
    lines.append(r"\section{Extended Statistical Analysis}")
    lines.append(r"\label{sec:s3}")
    lines.append(
        r"The main paper presents statistical analysis on the 41 binary datasets as the "
        r"primary comparison. This section provides the complementary all-48 dataset analysis "
        r"and detailed pairwise comparisons across all configurations."
    )
    lines.append("")

    stats_path = PAPER_DATA / "statistical_results.json"
    if not stats_path.exists():
        lines.append(r"\textit{Statistical results not available.}")
        return "\n".join(lines)

    stats = load_json(stats_path)

    # S3.1 All-48 Friedman ranking
    lines.append(r"\subsection{All-48 Datasets Friedman Ranking}")
    lines.append(r"\label{sec:s3_all48}")
    lines.append("")

    overall = stats.get("overall", {})
    configs = overall.get("configurations", [])

    # Use primary config EXP-500-NP for consistency with main paper
    best = None
    for c in configs:
        if c.get("config_label") == "EXP-500-NP":
            best = c
            break
    if best is None and configs:
        best = configs[0]  # fallback

    if best:
        ft = best["friedman_test"]
        cd = best["critical_distance"]
        n = best["n_datasets"]
        lines.append(
            f"On all {n} datasets, the Friedman test yields "
            f"$\\chi^2({ft['df']}) = {ft['statistic']:.1f}$ ($p < 0.001$) with "
            f"Nemenyi critical distance CD~=~{cd:.2f}."
        )
        lines.append("")
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{Model Ranking on All {n} Datasets (EXP-500-NP)}}")
        lines.append(r"\label{tab:s_ranking_all48}")
        lines.append(r"\footnotesize")
        lines.append(r"\begin{tabular}{clcc}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Rank} & \textbf{Model} & \textbf{Avg Rank} & \textbf{Mean G-Mean} \\")
        lines.append(r"\midrule")
        sr = sorted(best["average_rankings"].items(), key=lambda x: x[1])
        for i, (model, rank) in enumerate(sr, 1):
            perf = best.get("model_summary", {}).get(model, {})
            mean = perf.get("mean", 0)
            bold = r"\textbf{" + model + "}" if model == "EGIS" else model
            lines.append(f"{i} & {bold} & {rank:.2f} & {mean:.3f} \\\\")
        lines.append(r"\midrule")
        friedman_line = (
            r"\multicolumn{4}{l}{\textit{Friedman: $\chi^2("
            + str(ft['df']) + ") = " + f"{ft['statistic']:.1f}"
            + "$, p < 0.001, CD = " + f"{cd:.2f}" + r"}} \\"
        )
        lines.append(friedman_line)
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

        # CD diagram for all-48
        cd_fig = FIGURES / "fig_critical_difference_all48.pdf"
        if cd_fig.exists():
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"\centering")
            lines.append(r"\includegraphics[width=0.95\textwidth]{figures/fig_critical_difference_all48.pdf}")
            lines.append(
                f"\\caption{{Critical difference diagram for all {n} datasets "
                f"(EXP-500-NP). Methods connected by a horizontal bar "
                f"are not significantly different (Nemenyi, $\\alpha = 0.05$).}}"
            )
            lines.append(r"\label{fig:s_cd_all48}")
            lines.append(r"\end{figure}")
            lines.append("")

    # S3.2 Pairwise tests per configuration
    lines.append(r"\subsection{Pairwise Wilcoxon Tests Across Configurations}")
    lines.append(r"\label{sec:s3_pairwise}")
    lines.append(
        r"Table~\ref{tab:s_pairwise_all} summarizes pairwise Wilcoxon tests (EGIS vs each "
        r"baseline) across all configurations, including raw p-values, Bonferroni-adjusted "
        r"p-values, and Cliff's delta effect sizes."
    )
    lines.append("")

    # Binary-only configs
    binary = stats.get("binary_only", {})
    binary_configs = binary.get("configurations", [])

    if binary_configs:
        lines.append(r"\begin{longtable}{llccccc}")
        lines.append(r"\caption{Pairwise Wilcoxon Tests: EGIS vs Baselines (Binary Datasets)}\label{tab:s_pairwise_all} \\")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Config} & \textbf{Comparison} & \textbf{Raw p} & \textbf{Adj. p} & \textbf{Sig.} & \textbf{$\delta$} & \textbf{Effect} \\")
        lines.append(r"\midrule")
        lines.append(r"\endfirsthead")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Config} & \textbf{Comparison} & \textbf{Raw p} & \textbf{Adj. p} & \textbf{Sig.} & \textbf{$\delta$} & \textbf{Effect} \\")
        lines.append(r"\midrule")
        lines.append(r"\endhead")

        sorted_configs = sorted(binary_configs, key=lambda c: c["config_label"])
        for cfg in sorted_configs:
            clabel = escape_latex(cfg["config_label"])
            for t in cfg.get("pairwise_tests", []):
                comp = t["comparison"].replace("EGIS vs ", "")
                raw_p = t["raw_p_value"]
                adj_p = t.get("adjusted_p_value", raw_p * len(cfg.get("pairwise_tests", [])))
                sig = "Yes" if t.get("significant_bonferroni", False) else "No"
                delta = t["cliffs_delta"]
                eff = t["effect_interpretation"]
                p_str = f"$<$0.0001" if raw_p < 0.0001 else f"{raw_p:.4f}"
                ap_str = f"$<$0.0001" if adj_p < 0.0001 else f"{min(adj_p, 1.0):.4f}"
                lines.append(
                    f"{clabel} & {comp} & {p_str} & {ap_str} & {sig} & {delta:.3f} & {eff} \\\\"
                )
            # Add separator between configs (but not after the last one)
            if cfg != sorted_configs[-1]:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")
    lines.append("")

    return "\n".join(lines)


def generate_s4_transition_metrics():
    """S4: Extended Transition Metrics."""
    lines = []
    lines.append(r"\section{Extended Transition Metrics}")
    lines.append(r"\label{sec:s4}")
    lines.append(
        r"This section provides detailed transition metric analysis across all "
        r"configurations, extending the EXP-500-NP results presented in the main paper."
    )
    lines.append("")

    # S4.1 Transition metrics by drift type and config
    lines.append(r"\subsection{Transition Metrics by Configuration}")
    lines.append(r"\label{sec:s4_by_config}")

    csv_path = PAPER_DATA / "egis_transition_metrics.csv"
    if csv_path.exists():
        rows = load_csv(csv_path)
        # Aggregate by config_label and drift_type
        agg = defaultdict(lambda: defaultdict(lambda: {"TCS": [], "RIR": [], "AMS": []}))
        for row in rows:
            cl = row.get("config_label", "")
            dt = row.get("drift_type", "")
            for m in ["TCS", "RIR", "AMS"]:
                try:
                    agg[cl][dt][m].append(float(row[m]))
                except (ValueError, KeyError):
                    pass

        lines.append("")
        lines.append(r"\begin{longtable}{llccc}")
        lines.append(r"\caption{Transition Metrics by Configuration and Drift Type}\label{tab:s_transitions_all} \\")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Config} & \textbf{Drift Type} & \textbf{TCS} & \textbf{RIR} & \textbf{AMS} \\")
        lines.append(r"\midrule")
        lines.append(r"\endfirsthead")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Config} & \textbf{Drift Type} & \textbf{TCS} & \textbf{RIR} & \textbf{AMS} \\")
        lines.append(r"\midrule")
        lines.append(r"\endhead")

        sorted_keys = sorted(agg.keys())
        for idx, cl in enumerate(sorted_keys):
            for dt in ["abrupt", "gradual", "noisy", "stationary", "real"]:
                if dt in agg[cl]:
                    d = agg[cl][dt]
                    import statistics
                    tcs_m = statistics.mean(d["TCS"]) if d["TCS"] else 0
                    tcs_s = statistics.stdev(d["TCS"]) if len(d["TCS"]) > 1 else 0
                    rir_m = statistics.mean(d["RIR"]) if d["RIR"] else 0
                    rir_s = statistics.stdev(d["RIR"]) if len(d["RIR"]) > 1 else 0
                    ams_m = statistics.mean(d["AMS"]) if d["AMS"] else 0
                    ams_s = statistics.stdev(d["AMS"]) if len(d["AMS"]) > 1 else 0
                    lines.append(
                        f"{escape_latex(cl)} & {dt.capitalize()} & "
                        f"{tcs_m:.3f}$\\pm${tcs_s:.3f} & "
                        f"{rir_m:.3f}$\\pm${rir_s:.3f} & "
                        f"{ams_m:.3f}$\\pm${ams_s:.3f} \\\\"
                    )
            if idx < len(sorted_keys) - 1:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{longtable}")
    lines.append("")

    # S4.2 TCS timeseries figure
    tcs_fig = FIGURES / "fig_tcs_timeseries.png"
    if tcs_fig.exists():
        lines.append(r"\subsection{TCS Time Series}")
        lines.append(r"\label{sec:s4_tcs_ts}")
        lines.append(r"\begin{figure*}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.95\textwidth]{figures/fig_tcs_timeseries.png}")
        lines.append(
            r"\caption{Transition Change Score (TCS) time series by drift type. "
            r"Abrupt drift scenarios show characteristic spikes at drift points; "
            r"gradual drift exhibits smoother transitions; stationary streams "
            r"maintain consistently low TCS values.}"
        )
        lines.append(r"\label{fig:s_tcs_timeseries}")
        lines.append(r"\end{figure*}")
        lines.append("")

    # S4.3 TCS comparison
    tcs_comp = FIGURES / "fig_tcs_comparison.pdf"
    if tcs_comp.exists():
        lines.append(r"\subsection{TCS Comparison Across Configurations}")
        lines.append(r"\label{sec:s4_tcs_comp}")
        lines.append(r"\begin{figure*}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.95\textwidth]{figures/fig_tcs_comparison.pdf}")
        lines.append(
            r"\caption{TCS comparison across drift types and configurations. "
            r"Box plots show distributions of transition scores for each scenario.}"
        )
        lines.append(r"\label{fig:s_tcs_comparison}")
        lines.append(r"\end{figure*}")
        lines.append("")

    # S4.4 Evolution heatmaps
    for dtype, dname in [("abrupt", "Abrupt"), ("gradual", "Gradual"),
                         ("stationary", "Stationary"), ("real", "Real-world")]:
        fig = FIGURES / f"fig_evolution_{dtype}.pdf"
        if fig.exists():
            lines.append(f"\\begin{{figure}}[htbp]")
            lines.append(r"\centering")
            lines.append(f"\\includegraphics[width=0.95\\columnwidth]{{figures/fig_evolution_{dtype}.pdf}}")
            lines.append(
                f"\\caption{{Rule evolution heatmap for {dname.lower()} drift, "
                f"showing feature importance changes over time.}}"
            )
            lines.append(f"\\label{{fig:s_evolution_{dtype}}}")
            lines.append(r"\end{figure}")
            lines.append("")

    return "\n".join(lines)


def generate_s5_config_analysis():
    """S5: EGIS Configuration Analysis."""
    lines = []
    lines.append(r"\section{EGIS Configuration Analysis}")
    lines.append(r"\label{sec:s5}")
    lines.append(
        r"This section provides detailed analysis of how EGIS performance varies "
        r"across chunk size and penalty configurations."
    )
    lines.append("")

    # S5.1 Chunk size effect
    fig = FIGURES / "fig_chunk_size_effect.pdf"
    if fig.exists():
        lines.append(r"\subsection{Chunk Size Effect}")
        lines.append(r"\label{sec:s5_chunk}")
        lines.append(r"\begin{figure*}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.95\textwidth]{figures/fig_chunk_size_effect.pdf}")
        lines.append(
            r"\caption{Chunk size sensitivity analysis showing EGIS performance (G-Mean) "
            r"across configurations. Larger chunks improve performance on stationary and "
            r"gradual drift, while smaller chunks benefit abrupt drift adaptation.}"
        )
        lines.append(r"\label{fig:s_chunk_size}")
        lines.append(r"\end{figure*}")
        lines.append("")

    # S5.2 Config comparison
    fig = FIGURES / "fig_config_comparison.pdf"
    if fig.exists():
        lines.append(r"\subsection{Configuration Comparison}")
        lines.append(r"\label{sec:s5_config}")
        lines.append(r"\begin{figure*}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.95\textwidth]{figures/fig_config_comparison.pdf}")
        lines.append(
            r"\caption{Configuration comparison showing the combined effect of chunk size "
            r"and complexity penalty on EGIS performance. Error bars indicate standard "
            r"deviation across datasets.}"
        )
        lines.append(r"\label{fig:s_config_comp}")
        lines.append(r"\end{figure*}")
        lines.append("")

    # S5.3 Performance heatmap
    fig = FIGURES / "fig7_drift_heatmap.pdf"
    if fig.exists():
        lines.append(r"\subsection{Performance Heatmap by Drift Type}")
        lines.append(r"\label{sec:s5_heatmap}")
        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.95\columnwidth]{figures/fig7_drift_heatmap.pdf}")
        lines.append(
            r"\caption{Performance heatmap by drift type and model (G-Mean) for binary "
            r"datasets. Darker colors indicate higher performance.}"
        )
        lines.append(r"\label{fig:s_heatmap}")
        lines.append(r"\end{figure}")
        lines.append("")

    # S5.4 Metrics by drift type
    fig = FIGURES / "fig_metrics_by_drift.pdf"
    if fig.exists():
        lines.append(r"\subsection{Performance by Drift Type}")
        lines.append(r"\label{sec:s5_drift}")
        lines.append(r"\begin{figure*}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.95\textwidth]{figures/fig_metrics_by_drift.pdf}")
        lines.append(
            r"\caption{Performance comparison by drift type showing G-Mean for each model "
            r"across drift categories.}"
        )
        lines.append(r"\label{fig:s_metrics_drift}")
        lines.append(r"\end{figure*}")
        lines.append("")

    return "\n".join(lines)


def generate_s6_per_dataset(include_s6, s6_figures_dir="paper/figures/case_studies"):
    """S6: Per-Dataset Visual Analysis with composite 2x2 case study figures."""
    lines = []
    lines.append(r"\section{Per-Dataset Visual Analysis}")
    lines.append(r"\label{sec:s6}")

    if not include_s6:
        lines.append(
            r"\textit{Per-dataset plots are available in the online supplementary archive. "
            r"Each dataset has four visualizations: rule evolution matrix, attribute usage, "
            r"rule components heatmap, and accuracy with detected drifts.}"
        )
        return "\n".join(lines)

    lines.append(
        r"This section presents detailed case study visualizations for each of the 48 datasets "
        r"across three no-penalty EGIS configurations (EXP-500-NP, EXP-1000-NP, EXP-2000-NP). "
        r"Each figure is a composite 2$\times$2 panel showing: (a) rule evolution counts (stacked area), "
        r"(b) rule complexity over time (dual-axis line plot), (c) transition metrics (TCS, RIR, AMS) "
        r"over chunk transitions, and (d) rule turnover proportions (normalized stacked bar). "
        r"Drift annotations are shown where applicable."
    )
    lines.append("")

    # Get dataset list from consolidated_results.csv
    consolidated_csv = PAPER_DATA / "consolidated_results.csv"
    if not consolidated_csv.exists():
        lines.append(r"\textit{consolidated\_results.csv not found.}")
        return "\n".join(lines)

    rows = load_csv(consolidated_csv)

    # Collect unique datasets with their drift_type
    dataset_drift = {}
    for r in rows:
        ds = r["dataset"]
        dt = r.get("drift_type", "unknown")
        if ds not in dataset_drift:
            dataset_drift[ds] = dt

    # Organize by drift type
    drift_groups = defaultdict(list)
    for ds in sorted(dataset_drift.keys()):
        drift_groups[dataset_drift[ds]].append(ds)

    drift_type_titles = {
        "abrupt": "Abrupt Drift Datasets",
        "gradual": "Gradual Drift Datasets",
        "noisy": "Noisy Drift Datasets",
        "stationary": "Stationary Datasets",
        "real": "Real-World Datasets",
    }

    np_configs = [
        ("EXP-500-NP", 500),
        ("EXP-1000-NP", 1000),
        ("EXP-2000-NP", 2000),
    ]

    subsection_counter = 0
    for dt in DRIFT_TYPE_ORDER:
        if dt not in drift_groups:
            continue
        subsection_counter += 1
        title = drift_type_titles.get(dt, f"{dt.capitalize()} Datasets")
        lines.append(f"\\subsection{{{title}}}")
        lines.append(f"\\label{{sec:s6_{dt}}}")
        lines.append("")

        for ds in drift_groups[dt]:
            ds_escaped = escape_latex(ds)
            dt_display = dt.capitalize() if dt != "real" else "real-world"

            lines.append(f"\\subsubsection{{{ds_escaped}}}")
            lines.append(r"\begin{figure*}[htbp]")
            lines.append(r"\centering")

            for i, (config_label, chunk_size) in enumerate(np_configs):
                # Path relative to paper/ directory
                fig_path = f"figures/case_studies/{config_label}/{ds}.pdf"
                newline = r"\\" if i < len(np_configs) - 1 else ""
                lines.append(
                    f"\\subfloat[{config_label} (chunk={chunk_size})]"
                    f"{{\\includegraphics[width=0.95\\textwidth,height=0.27\\textheight,keepaspectratio]{{{fig_path}}}}}{newline}"
                )

            lines.append(
                f"\\caption{{Case study: {ds_escaped} ({dt_display} drift). "
                f"Composite 2$\\times$2 panels show rule evolution counts, rule complexity, "
                f"transition metrics, and rule turnover proportions for each chunk configuration.}}"
            )
            # Label: sanitize dataset name for LaTeX label
            label_ds = ds.lower().replace("-", "_")
            lines.append(f"\\label{{fig:s6_{label_ds}}}")
            lines.append(r"\end{figure*}")
            lines.append(r"\clearpage")
            lines.append("")

    return "\n".join(lines)


def generate_s7_rule_evolution():
    """S7: Rule Evolution Details."""
    lines = []
    lines.append(r"\section{Rule Evolution Details}")
    lines.append(r"\label{sec:s7}")
    lines.append(
        r"This section provides summary statistics on rule evolution patterns "
        r"across configurations."
    )
    lines.append("")

    # Use evolution_analysis_summary.csv
    csv_path = PAPER_DATA / "evolution_analysis_summary.csv"
    if csv_path.exists():
        rows = load_csv(csv_path)

        # Aggregate by config_label
        agg = defaultdict(lambda: {"unchanged": [], "modified": [], "new": [], "deleted": []})
        for row in rows:
            cl = row.get("config_label", row.get("config", ""))
            for k in ["unchanged", "modified", "new", "deleted"]:
                try:
                    agg[cl][k].append(float(row[k]))
                except (ValueError, KeyError):
                    pass

        if agg:
            import statistics
            lines.append(r"\begin{table}[htbp]")
            lines.append(r"\centering")
            lines.append(r"\caption{Rule Evolution Summary by Configuration (Mean Counts per Transition)}")
            lines.append(r"\label{tab:s_rule_evo}")
            lines.append(r"\footnotesize")
            lines.append(r"\begin{tabular}{lcccc}")
            lines.append(r"\toprule")
            lines.append(r"\textbf{Config} & \textbf{Unchanged} & \textbf{Modified} & \textbf{New} & \textbf{Deleted} \\")
            lines.append(r"\midrule")
            for cl in sorted(agg.keys()):
                d = agg[cl]
                unch = statistics.mean(d["unchanged"]) if d["unchanged"] else 0
                mod = statistics.mean(d["modified"]) if d["modified"] else 0
                new = statistics.mean(d["new"]) if d["new"] else 0
                dele = statistics.mean(d["deleted"]) if d["deleted"] else 0
                lines.append(
                    f"{escape_latex(cl)} & {unch:.1f} & {mod:.1f} & {new:.1f} & {dele:.1f} \\\\"
                )
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def generate_postamble():
    return r"""
\end{document}
"""


def main():
    parser = argparse.ArgumentParser(description="Generate Supplementary Material")
    parser.add_argument("--output", default="paper/supplementary_material.tex",
                        help="Output .tex file path")
    parser.add_argument("--include-s6", action="store_true",
                        help="Include per-dataset plots (S6, ~336 pages)")
    parser.add_argument("--s6-figures-dir", default="paper/figures/case_studies",
                        help="Directory containing case study figures for S6")
    args = parser.parse_args()

    print("Generating Supplementary Material...")
    print(f"  Output: {args.output}")
    print(f"  Include S6 per-dataset plots: {args.include_s6}")
    print(f"  S6 figures dir: {args.s6_figures_dir}")

    sections = [
        generate_preamble(),
        generate_s1_experimental_details(),
        generate_s2_performance_tables(),
        generate_s3_statistical_analysis(),
        generate_s4_transition_metrics(),
        generate_s5_config_analysis(),
        generate_s6_per_dataset(args.include_s6, s6_figures_dir=args.s6_figures_dir),
        generate_s7_rule_evolution(),
        generate_postamble(),
    ]

    output = "\n\n".join(sections)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\nGenerated: {out_path}")
    print(f"  Size: {len(output):,} characters")
    line_count = output.count("\n")
    print(f"  Lines: {line_count:,}")


if __name__ == "__main__":
    main()
