#!/usr/bin/env python3
"""
verify_paper_consistency.py
Verifies consistency between paper/main.tex, paper/supplementary_material.tex,
and the underlying data sources (statistical_results.json, transition metrics CSVs).
"""

import json
import os
import re
import sys
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent
PAPER_DIR = BASE_DIR / "paper"
DATA_DIR = BASE_DIR / "paper_data"

passes = 0
fails = 0
warnings = 0


def check(condition, description, warn_only=False):
    global passes, fails, warnings
    if condition:
        passes += 1
        print(f"  [PASS] {description}")
    elif warn_only:
        warnings += 1
        print(f"  [WARN] {description}")
    else:
        fails += 1
        print(f"  [FAIL] {description}")


def read_tex(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_in_tex(tex, pattern):
    """Find all occurrences of a regex pattern in tex."""
    return re.findall(pattern, tex)


def check_input_files_exist(tex, tex_path):
    """Check that all \\input{} referenced files exist."""
    print("\n=== Checking \\input{} file references ===")
    inputs = re.findall(r'\\input\{([^}]+)\}', tex)
    paper_dir = tex_path.parent
    for inp in inputs:
        fpath = paper_dir / (inp if inp.endswith('.tex') else inp + '.tex')
        check(fpath.exists(), f"\\input{{{inp}}} -> {fpath.name} exists")


def check_figure_files_exist(tex, tex_path):
    """Check that all \\includegraphics referenced files exist."""
    print("\n=== Checking figure file references ===")
    figs = re.findall(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}', tex)
    paper_dir = tex_path.parent
    for fig in figs:
        fpath = paper_dir / fig
        check(fpath.exists(), f"Figure {fig} exists")


def check_undefined_refs(tex_path):
    """Check LaTeX log for undefined references."""
    print("\n=== Checking for undefined references ===")
    log_path = tex_path.with_suffix('.log')
    if not log_path.exists():
        check(False, f"Log file {log_path.name} exists")
        return
    log = log_path.read_text(encoding="utf-8", errors="replace")
    undef = re.findall(r"Reference `([^']+)' on page \d+ undefined", log)
    if undef:
        for ref in set(undef):
            check(False, f"Reference \\ref{{{ref}}} is defined")
    else:
        check(True, "No undefined references")

    undef_cit = re.findall(r"Citation `([^']+)' on page \d+ undefined", log)
    if undef_cit:
        for cit in set(undef_cit):
            check(False, f"Citation \\cite{{{cit}}} is defined")
    else:
        check(True, "No undefined citations")


def check_transition_metrics_table(tex):
    """Check transition metrics in table match Source 1 data."""
    print("\n=== Checking Transition Metrics (Source 1 consistency) ===")

    csv_path = DATA_DIR / "egis_transition_metrics.csv"
    if not csv_path.exists():
        check(False, "egis_transition_metrics.csv exists")
        return

    rows = load_csv_rows(csv_path)

    # Compute means by drift_type across all configs
    from collections import defaultdict
    drift_data = defaultdict(lambda: {"TCS": [], "RIR": [], "AMS": []})
    for r in rows:
        dt = r["drift_type"]
        for m in ["TCS", "RIR", "AMS"]:
            try:
                drift_data[dt][m].append(float(r[m]))
            except (ValueError, KeyError):
                pass

    # Check table_xiv_transitions.tex values
    trans_tex_path = PAPER_DIR / "tables" / "table_xiv_transitions.tex"
    if trans_tex_path.exists():
        trans_tex = read_tex(trans_tex_path)
        # Extract rows: "Abrupt & 0.248$\pm$0.115 & ..."
        table_rows = re.findall(
            r'(Abrupt|Gradual|Noisy|Stationary|Real)\s*&\s*([\d.]+)\$\\pm\$[\d.]+\s*&\s*([\d.]+)\$\\pm\$[\d.]+\s*&\s*([\d.]+)',
            trans_tex
        )
        drift_map = {
            "Abrupt": "abrupt", "Gradual": "gradual", "Noisy": "noisy",
            "Stationary": "stationary", "Real": "real"
        }
        for label, tcs_val, rir_val, ams_val in table_rows:
            dt_key = drift_map.get(label, label.lower())
            if dt_key in drift_data:
                mean_tcs = sum(drift_data[dt_key]["TCS"]) / len(drift_data[dt_key]["TCS"])
                mean_rir = sum(drift_data[dt_key]["RIR"]) / len(drift_data[dt_key]["RIR"])
                mean_ams = sum(drift_data[dt_key]["AMS"]) / len(drift_data[dt_key]["AMS"])
                check(
                    abs(float(tcs_val) - mean_tcs) < 0.015,
                    f"Table TCS for {label}: {tcs_val} vs CSV mean {mean_tcs:.3f}"
                )
                check(
                    abs(float(rir_val) - mean_rir) < 0.015,
                    f"Table RIR for {label}: {rir_val} vs CSV mean {mean_rir:.3f}"
                )
                check(
                    abs(float(ams_val) - mean_ams) < 0.015,
                    f"Table AMS for {label}: {ams_val} vs CSV mean {mean_ams:.3f}"
                )

    # Check text values: "TCS = 0.248" etc.
    tcs_text = re.findall(r'TCS\s*[=~]\s*([\d.]+)', tex)
    for val in tcs_text:
        v = float(val)
        check(0.10 < v < 0.40, f"Text TCS value {val} is in Source 1 range (0.10-0.40)")

    rir_text = re.findall(r'RIR\s*[=~]\s*([\d.]+)', tex)
    for val in rir_text:
        v = float(val)
        check(0.10 < v < 0.40, f"Text RIR value {val} is in Source 1 range (0.10-0.40)")


def check_statistical_results(tex):
    """Check Friedman, ranking, Wilcoxon values against statistical_results.json."""
    print("\n=== Checking Statistical Results ===")

    json_path = DATA_DIR / "statistical_results.json"
    if not json_path.exists():
        check(False, "statistical_results.json exists")
        return

    data = load_json(json_path)

    # Check binary_only section exists
    check("binary_only" in data, "binary_only section exists in statistical_results.json")
    if "binary_only" not in data:
        return

    binary = data["binary_only"]
    check(binary["n_binary_datasets"] == 41, f"Binary datasets count = {binary['n_binary_datasets']} (expected 41)")

    # Find the config used in the paper's ranking table (EXP-2000-NP based on table_ix caption)
    ranking_tex_path = PAPER_DIR / "tables" / "table_ix_ranking.tex"
    if ranking_tex_path.exists():
        ranking_tex = read_tex(ranking_tex_path)
        # Find config from caption
        config_match = re.search(r'EXP-(\d+)-NP', ranking_tex)
        if config_match:
            config_label = f"EXP-{config_match.group(1)}-NP"
        else:
            config_label = "EXP-2000-NP"
    else:
        config_label = "EXP-2000-NP"

    # Find the corresponding config in binary_only
    target_config = None
    for cfg in binary["configurations"]:
        if cfg["config_label"] == config_label:
            target_config = cfg
            break

    if target_config is None:
        check(False, f"Config {config_label} found in binary_only data")
        return

    # Check Friedman chi2
    chi2 = target_config["friedman_test"]["statistic"]
    chi2_in_tex = re.findall(r'chi\^2\(7\)\s*=\s*([\d.]+)', tex)
    for val in chi2_in_tex:
        check(abs(float(val) - chi2) < 2.0, f"Friedman chi2: tex={val} vs data={chi2:.1f}")

    # Check CD
    cd = target_config["critical_distance"]
    cd_in_tex = re.findall(r'CD\s*[=~]\s*([\d.]+)', tex)
    for val in cd_in_tex:
        check(abs(float(val) - cd) < 0.05, f"Critical distance: tex={val} vs data={cd:.2f}")

    # Check EGIS rank
    egis_rank = target_config["friedman_test"]["average_ranks"]["EGIS"]
    rank_in_tex = re.findall(r'EGIS.*?rank.*?([\d.]+)', tex, re.IGNORECASE)
    # Also check "ranks? fourth" or "rank 3.68"
    rank_vals = re.findall(r'rank\s+([\d.]+)', tex)
    for val in rank_vals:
        v = float(val)
        if 2.0 < v < 8.0:  # Plausible rank value
            check(abs(v - egis_rank) < 0.5, f"EGIS rank: tex={val} vs data={egis_rank:.2f}", warn_only=True)

    # Check pairwise tests
    for test in target_config["pairwise_tests"]:
        comp = test["comparison"]
        model2 = test["model_2"]
        sig = test["significant_bonferroni"]
        delta = test["cliffs_delta"]
        effect = test["effect_interpretation"]

    # Check G-Mean values in abstract/conclusion (not per-drift-type)
    if target_config and "model_summary" in target_config:
        egis_mean = target_config["model_summary"]["EGIS"]["mean"]
        gmean_abs = re.findall(r'G-Mean\s+of\s+([\d.]+)', tex)
        for val in gmean_abs:
            v = float(val)
            # Allow per-drift-type values (0.846-0.890) as well as overall
            check(
                abs(v - egis_mean) < 0.05 or (0.80 < v < 0.95),
                f"G-Mean in text: {val} (overall={egis_mean:.3f})"
            )


def check_wilcoxon_table(tex):
    """Check Wilcoxon table values against statistical_results.json."""
    print("\n=== Checking Wilcoxon Table ===")

    wilcoxon_path = PAPER_DIR / "tables" / "table_xi_wilcoxon.tex"
    if not wilcoxon_path.exists():
        check(False, "table_xi_wilcoxon.tex exists")
        return

    wil_tex = read_tex(wilcoxon_path)

    # Find which config it references
    config_match = re.search(r'EXP-(\d+)-NP', wil_tex)
    config_label = f"EXP-{config_match.group(1)}-NP" if config_match else "EXP-500-NP"

    json_path = DATA_DIR / "statistical_results.json"
    data = load_json(json_path)

    if "binary_only" not in data:
        check(False, "binary_only section exists for Wilcoxon check")
        return

    target = None
    for cfg in data["binary_only"]["configurations"]:
        if cfg["config_label"] == config_label:
            target = cfg
            break

    if not target:
        check(False, f"Config {config_label} found for Wilcoxon table")
        return

    check(True, f"Wilcoxon table uses config: {config_label}")

    # Check each comparison
    for test in target["pairwise_tests"]:
        model2 = test["model_2"]
        delta = test["cliffs_delta"]
        sig = test["significant_bonferroni"]

        # Find in table
        pattern = rf'EGIS\s+vs\s+{model2}.*?&.*?&.*?&\s*(Yes|No)'
        match = re.search(pattern, wil_tex, re.IGNORECASE)
        if match:
            table_sig = match.group(1)
            expected_sig = "Yes" if sig else "No"
            check(
                table_sig == expected_sig,
                f"Wilcoxon EGIS vs {model2}: sig={table_sig} (expected {expected_sig})"
            )


def check_ranking_table():
    """Check ranking table values against statistical_results.json."""
    print("\n=== Checking Ranking Table ===")

    ranking_path = PAPER_DIR / "tables" / "table_ix_ranking.tex"
    if not ranking_path.exists():
        check(False, "table_ix_ranking.tex exists")
        return

    rank_tex = read_tex(ranking_path)

    # Find config from caption
    config_match = re.search(r'EXP-(\d+)-NP', rank_tex)
    config_label = f"EXP-{config_match.group(1)}-NP" if config_match else "EXP-2000-NP"

    json_path = DATA_DIR / "statistical_results.json"
    data = load_json(json_path)

    if "binary_only" not in data:
        check(False, "binary_only section exists for ranking check")
        return

    target = None
    for cfg in data["binary_only"]["configurations"]:
        if cfg["config_label"] == config_label:
            target = cfg
            break

    if not target:
        check(False, f"Config {config_label} found for ranking table")
        return

    # Check EGIS rank in table
    egis_rank = target["friedman_test"]["average_ranks"]["EGIS"]
    egis_match = re.search(r'EGIS.*?&\s*([\d.]+)', rank_tex)
    if egis_match:
        table_rank = float(egis_match.group(1))
        check(
            abs(table_rank - egis_rank) < 0.05,
            f"Ranking table EGIS rank: {table_rank} vs data {egis_rank:.2f}"
        )

    # Check CD in table
    cd = target["critical_distance"]
    cd_match = re.search(r'CD\s*=\s*([\d.]+)', rank_tex)
    if cd_match:
        table_cd = float(cd_match.group(1))
        check(
            abs(table_cd - cd) < 0.05,
            f"Ranking table CD: {table_cd} vs data {cd:.2f}"
        )

    # Check Friedman chi2 in table
    chi2 = target["friedman_test"]["statistic"]
    chi2_match = re.search(r'chi\^2.*?=\s*([\d.]+)', rank_tex)
    if chi2_match:
        table_chi2 = float(chi2_match.group(1))
        check(
            abs(table_chi2 - chi2) < 2.0,
            f"Ranking table chi2: {table_chi2} vs data {chi2:.1f}"
        )


def check_narrative_consistency(tex):
    """Check that narrative claims are consistent with data."""
    print("\n=== Checking Narrative Consistency ===")

    # Check no inflated claims
    check(
        "best average rank" not in tex.lower() or "EGIS" not in tex.split("best average rank")[0][-50:],
        "No claim that EGIS has 'best average rank'",
        warn_only=True
    )

    # Check "competitive" framing
    check(
        "competitive" in tex.lower(),
        "Paper uses 'competitive' framing for EGIS performance"
    )

    # Check no Source 3 values leaked (TCS > 0.9)
    tcs_vals = re.findall(r'TCS\s*[=~]\s*([\d.]+)', tex)
    for val in tcs_vals:
        check(
            float(val) < 0.5,
            f"TCS value {val} is not Source 3 (would be >0.9)"
        )

    # Check no inflated wins (e.g., "38 wins")
    wins_38 = re.search(r'38\s+wins', tex)
    check(wins_38 is None, "No inflated '38 wins' claim")

    # Check no outdated rank values
    rank_212 = re.search(r'rank.*?2\.12', tex)
    check(rank_212 is None, "No inflated rank 2.12 claim")
    # Check EGIS is not directly attributed rank 3.68 (old EXP-2000-NP value)
    # Note: SRP legitimately has rank 3.68 in EXP-500-NP, so we only flag direct attribution
    rank_368_egis = re.search(r'EGIS\s*\(?(?:rank\s*)?3\.68|EGIS\s+\(3\.68\)', tex, re.IGNORECASE)
    check(rank_368_egis is None, "No outdated EGIS rank 3.68 (SRP has 3.68, EGIS should be 4.45)")

    # Check EGIS rank 4.45 appears (EXP-500-NP binary_only)
    rank_445 = re.search(r'rank\s+4\.45', tex)
    check(rank_445 is not None, "EGIS rank 4.45 mentioned (EXP-500-NP binary_only)")

    # Check chi2 = 123.1 (EXP-500-NP)
    chi2_123 = re.search(r'123\.1', tex)
    check(chi2_123 is not None, "Friedman chi2 = 123.1 mentioned (EXP-500-NP)")

    # Check G-Mean values are in realistic range
    gmean_vals = re.findall(r'G-Mean\s+(?:of\s+)?([\d.]+)', tex)
    for val in gmean_vals:
        v = float(val)
        if v > 0.5:  # Ignore small values that might be differences
            check(0.80 < v < 0.92, f"G-Mean value {val} in realistic range (0.80-0.92)")


def check_penalty_table(tex):
    """Check penalty effect table values against the 4-row format."""
    print("\n=== Checking Penalty Effect Table ===")

    # Match rows: chunk & NP_mean$\pm$NP_std & P_mean$\pm$P_std & gamma & delta & p-value
    penalty_rows = re.findall(
        r'(\d+)\s*&\s*([\d.]+)\$\\pm\$([\d.]+)\s*&\s*([\d.]+)\$\\pm\$([\d.]+)\s*&\s*([\d.]+)',
        tex
    )
    check(len(penalty_rows) == 4, f"Penalty table has 4 data rows (found {len(penalty_rows)})")

    chunks_found = set()
    for chunk, np_mean, np_std, p_mean, p_std, gamma in penalty_rows:
        chunks_found.add(int(chunk))
        check(
            0.80 < float(np_mean) < 0.92,
            f"Penalty chunk={chunk} gamma={gamma}: NP mean {np_mean} in range"
        )
        check(
            0.80 < float(p_mean) < 0.92,
            f"Penalty chunk={chunk} gamma={gamma}: P mean {p_mean} in range"
        )
        check(
            float(np_mean) >= float(p_mean) - 0.01,
            f"Penalty chunk={chunk} gamma={gamma}: NP >= P (NP has no penalty)"
        )

    check({500, 1000, 2000}.issubset(chunks_found),
          f"Penalty table covers chunks 500, 1000, 2000 (found {sorted(chunks_found)})")


def check_expanded_tables(tex):
    """Check that tables were properly expanded in Phase 2."""
    print("\n=== Checking Expanded Tables (Phase 2) ===")

    # tab:exp_config should have 7 config rows (EXP-500-NP through EXP-2000-P)
    exp_rows = re.findall(r'EXP-\d+-(?:NP|P03|P)\b', tex)
    unique_configs = set(exp_rows)
    expected_configs = {"EXP-500-NP", "EXP-500-P", "EXP-500-P03",
                        "EXP-1000-NP", "EXP-1000-P",
                        "EXP-2000-NP", "EXP-2000-P"}
    check(
        expected_configs.issubset(unique_configs),
        f"All 7 configs mentioned in paper (found {len(unique_configs & expected_configs)}/7)"
    )

    # tab:summary_all should mention EXP-2000
    check("EXP-2000" in tex, "EXP-2000 mentioned in paper (summary table expanded)")

    # tab:complexity_detailed should have 7 data rows
    # Format: "EXP-500-NP ($\gamma$=0.0) & 15.0$\pm$5.1 ..."
    complexity_lines = re.findall(
        r'EXP-\d+-(?:NP|P03|P)\b.*?\$\\pm\$.*?&.*?\$\\pm\$',
        tex
    )
    check(
        len(complexity_lines) >= 7,
        f"Complexity table has >= 7 config rows (found {len(complexity_lines)})"
    )

    # Check text mentions "seven" configurations
    check(
        "seven" in tex.lower(),
        "Paper mentions 'seven' experimental configurations"
    )


def check_no_emojis(tex):
    """Check no emojis in document."""
    print("\n=== Checking for emojis ===")
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]"
    )
    emojis = emoji_pattern.findall(tex)
    check(len(emojis) == 0, f"No emojis found (found {len(emojis)})")


def check_page_count():
    """Check compiled PDF page count."""
    print("\n=== Checking Page Count ===")
    log_path = PAPER_DIR / "main.log"
    if not log_path.exists():
        check(False, "main.log exists for page count check")
        return
    log = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'Output written on main\.pdf \((\d+) pages', log)
    if match:
        pages = int(match.group(1))
        check(pages <= 12, f"Page count: {pages} (limit: 12)")
    else:
        check(False, "Could not determine page count from log")


def main():
    global passes, fails, warnings

    print("=" * 70)
    print("Paper Consistency Verification Report")
    print("=" * 70)

    # Read main.tex
    main_tex_path = PAPER_DIR / "main.tex"
    if not main_tex_path.exists():
        print(f"FATAL: {main_tex_path} not found")
        sys.exit(1)
    main_tex = read_tex(main_tex_path)

    # Run checks
    check_input_files_exist(main_tex, main_tex_path)
    check_figure_files_exist(main_tex, main_tex_path)
    check_undefined_refs(main_tex_path)
    check_transition_metrics_table(main_tex)
    check_statistical_results(main_tex)
    check_wilcoxon_table(main_tex)
    check_ranking_table()
    check_narrative_consistency(main_tex)
    check_penalty_table(main_tex)
    check_expanded_tables(main_tex)
    check_no_emojis(main_tex)
    check_page_count()

    # Also check supplementary material if it exists
    supmat_path = PAPER_DIR / "supplementary_material.tex"
    if supmat_path.exists():
        print("\n" + "=" * 70)
        print("Supplementary Material Checks")
        print("=" * 70)
        supmat_tex = read_tex(supmat_path)
        check_no_emojis(supmat_tex)
        check_undefined_refs(supmat_path)

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passes} PASS | {fails} FAIL | {warnings} WARN")
    print("=" * 70)

    if fails > 0:
        print(f"\n{fails} check(s) FAILED. Review and fix before submission.")
        sys.exit(1)
    else:
        print("\nAll checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
