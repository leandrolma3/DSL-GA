# rule_diff_analyzer.py (Corrigido - Segundo SyntaxError no try/except)

import re
import os
import argparse
import logging
import Levenshtein # Ou import editdistance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# --- Função de Parsing (Corrigida Estrutura Try/Except para Class Label) ---

def parse_rules_history(file_path):
    """
    Parses the rule history text file, including performance metrics,
    into a structured dictionary, handling multi-line rules.
    """
    if not os.path.exists(file_path): logging.error(f"History file not found: {file_path}"); return None

    chunks_data = {}
    current_chunk = -1
    parsing_state = "LOOKING_FOR_CHUNK"
    current_rule_condition_lines = []

    # Regex patterns (sem alterações)
    chunk_header_re = re.compile(r"^-+ Chunk (\d+) \(Trained\)")
    train_perf_re = re.compile(r"^\s*Train Perf \(Chunk \d+\): TrainAcc=([-\d.]+)")
    test_perf_re = re.compile(r"Test Perf \(Chunk \d+\): TestAcc=([-\d.]+),\s*TestF1=([-\d.]+)")
    fitness_re = re.compile(r"^\s*Fitness:\s*([-\d.]+)", re.IGNORECASE)
    default_class_re = re.compile(r"^\s*Default Class:\s*(\S+)", re.IGNORECASE)
    rule_start_re = re.compile(r"^\s*(\d+):\s*IF (.*)")
    rule_end_re = re.compile(r"(.*?) THEN Class\s+(\S+)$")
    class_header_re = re.compile(r"^\s*Class\s+(\S+):")
    separator_re = re.compile(r"^-+$")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            rules_by_class = defaultdict(list)

            for line_num, raw_line in enumerate(f):
                line = raw_line.strip();
                if not line: continue

                # 1. Checa cabeçalho de chunk
                chunk_match = chunk_header_re.match(raw_line)
                if chunk_match:
                    if current_chunk != -1: # Salva chunk anterior
                        chunks_data[current_chunk].setdefault('fitness', float('nan')); chunks_data[current_chunk].setdefault('default_class', 'Unknown')
                        chunks_data[current_chunk].setdefault('train_acc', float('nan')); chunks_data[current_chunk].setdefault('test_acc_next', float('nan'))
                        chunks_data[current_chunk].setdefault('test_f1_next', float('nan')); chunks_data[current_chunk]['rules'] = dict(rules_by_class)
                        if parsing_state == "PARSING_RULE": logging.warning(f"Chunk boundary while parsing rule in chunk {current_chunk}."); current_rule_condition_lines = []
                    # Inicia novo chunk
                    current_chunk = int(chunk_match.group(1)); logging.debug(f"Parsing Chunk {current_chunk}...")
                    chunks_data[current_chunk] = {}; rules_by_class = defaultdict(list); parsing_state = "IN_CHUNK_HEADER"
                    # Tenta extrair Test Perf da mesma linha
                    test_match = test_perf_re.search(raw_line)
                    if test_match:
                         try: chunks_data[current_chunk]['test_acc_next'] = float(test_match.group(1)); chunks_data[current_chunk]['test_f1_next'] = float(test_match.group(2)); logging.debug(f"  Parsed Test Perf: TestAcc={test_match.group(1)}, TestF1={test_match.group(2)}")
                         except (ValueError, IndexError) as e: logging.warning(f"L{line_num+1}: Could not parse test perf: {e}")
                    else: logging.warning(f"L{line_num+1}: Test Perf data not found in header line.")
                    continue

                # --- Processa linha baseado no estado ---
                if parsing_state == "IN_CHUNK_HEADER":
                     train_match = train_perf_re.search(line)
                     if train_match:
                         try: chunks_data[current_chunk]['train_acc'] = float(train_match.group(1)); logging.debug(f"  Parsed Train Perf: TrainAcc={train_match.group(1)}")
                         except(ValueError, IndexError): logging.warning(f"L{line_num+1}: Invalid TrainAcc value {train_match.group(1)}")
                         continue # Continua esperando '---' ou outra meta
                     elif separator_re.match(line):
                          parsing_state = "IN_CHUNK_META"; continue # Muda estado APÓS separador
                     else: # Se não for TrainPerf nem '---', assume que já é meta/regra
                          parsing_state = "IN_CHUNK_META"; # Não dá continue, reprocessa linha

                if parsing_state == "IN_CHUNK_META":
                    fitness_match = fitness_re.match(line)
                    if fitness_match:
                         try: chunks_data[current_chunk]['fitness'] = float(fitness_match.group(1))
                         except ValueError: logging.warning(f"L{line_num+1}: Invalid fitness value {fitness_match.group(1)}")
                         continue

                    # Default Class (ESTRUTURA TRY/EXCEPT CORRIGIDA)
                    default_class_match = default_class_re.match(line)
                    if default_class_match:
                        dc_str = default_class_match.group(1)
                        try:
                            chunks_data[current_chunk]['default_class'] = int(dc_str)
                        except ValueError:
                            try:
                                chunks_data[current_chunk]['default_class'] = float(dc_str)
                            except ValueError:
                                chunks_data[current_chunk]['default_class'] = dc_str
                        continue

                    class_header_match = class_header_re.match(line)
                    if class_header_match: continue

                    # INÍCIO de uma regra
                    rule_start_match = rule_start_re.match(line)
                    if rule_start_match:
                        initial_condition_part = rule_start_match.group(2).strip()
                        rule_end_match = rule_end_re.search(initial_condition_part)
                        if rule_end_match: # Regra de linha única
                            full_condition = rule_end_match.group(1).strip(); class_label_str = rule_end_match.group(2).strip()
                            # --- Bloco try/except CORRIGIDO para class_label ---
                            try:
                                class_label = int(class_label_str)
                            except ValueError:
                                try:
                                    class_label = float(class_label_str)
                                except ValueError:
                                    class_label = class_label_str # Mantém como string
                            # ----------------------------------------------------
                            rules_by_class[class_label].append(full_condition)
                        else: # Início de regra multi-linha
                            current_rule_condition_lines = [initial_condition_part]; parsing_state = "PARSING_RULE"
                        continue

                elif parsing_state == "PARSING_RULE":
                    rule_end_match = rule_end_re.search(line)
                    if rule_end_match: # Fim da regra
                        final_condition_part = rule_end_match.group(1).strip(); class_label_str = rule_end_match.group(2).strip()
                        if final_condition_part or not current_rule_condition_lines: current_rule_condition_lines.append(final_condition_part)
                         # --- Bloco try/except CORRIGIDO para class_label ---
                        try:
                            class_label = int(class_label_str)
                        except ValueError:
                            try:
                                class_label = float(class_label_str)
                            except ValueError:
                                class_label = class_label_str # Mantém como string
                        # ----------------------------------------------------
                        full_condition = " ".join(current_rule_condition_lines).strip()
                        rules_by_class[class_label].append(full_condition)
                        current_rule_condition_lines = []; parsing_state = "IN_CHUNK_META"
                    else: # Linha intermediária
                        current_rule_condition_lines.append(line)
                    continue

            # --- Fim do Arquivo: Salva o último chunk ---
            if current_chunk != -1:
                chunks_data[current_chunk].setdefault('fitness', float('nan')); chunks_data[current_chunk].setdefault('default_class', 'Unknown')
                chunks_data[current_chunk].setdefault('train_acc', float('nan')); chunks_data[current_chunk].setdefault('test_acc_next', float('nan'))
                chunks_data[current_chunk].setdefault('test_f1_next', float('nan')); chunks_data[current_chunk]['rules'] = dict(rules_by_class)

        if not chunks_data: logging.warning(f"No chunks successfully parsed from file: {file_path}"); return None
        logging.info(f"Successfully parsed {len(chunks_data)} chunks from {file_path}")
        # Log para verificar performance parseada
        for chk_idx, chk_data in chunks_data.items(): logging.debug(f"Chunk {chk_idx} Parsed Perf: Train={chk_data.get('train_acc')}, TestAcc={chk_data.get('test_acc_next')}, TestF1={chk_data.get('test_f1_next')}")
        return chunks_data

    except FileNotFoundError: logging.error(f"History file not found: {file_path}"); return None
    except Exception as e: logging.error(f"Error parsing file {file_path}: {e}", exc_info=True); return None


# --- Função de Comparação ---
def compare_chunk_rules(chunk_data_i, chunk_data_i_plus_1, similarity_threshold=0.35):
    # ... (código como antes) ...
    if not chunk_data_i or not chunk_data_i_plus_1: logging.error("Invalid input data for comparison."); return None
    rules_i = chunk_data_i.get('rules'); rules_i_plus_1 = chunk_data_i_plus_1.get('rules')
    if not isinstance(rules_i, dict) or not isinstance(rules_i_plus_1, dict): logging.error("Invalid 'rules' structure."); return {'unchanged': {}, 'modified': {}, 'new': {}, 'deleted': {}}
    all_classes = set(rules_i.keys()) | set(rules_i_plus_1.keys())
    diff_results = {'unchanged': defaultdict(list), 'modified': defaultdict(list), 'new': defaultdict(list), 'deleted': defaultdict(list)}
    for class_label in all_classes:
        logging.debug(f"Comparing rules for class {class_label}...")
        ruleset_i_list = rules_i.get(class_label, []); ruleset_i_plus_1_list = rules_i_plus_1.get(class_label, [])
        remaining_i = list(ruleset_i_list); remaining_i_plus_1 = list(ruleset_i_plus_1_list); unchanged_found = []
        list1, list2 = (remaining_i, remaining_i_plus_1) if len(remaining_i) < len(remaining_i_plus_1) else (remaining_i_plus_1, remaining_i)
        for rule in list(list1):
             if rule in list2: unchanged_found.append(rule); list1.remove(rule); list2.remove(rule)
        if unchanged_found: diff_results['unchanged'][class_label].extend(unchanged_found); logging.debug(f"  Class {class_label}: Found {len(unchanged_found)} unchanged rules.")
        potential_matches = defaultdict(lambda: {'best_prev_idx': -1, 'min_norm_dist': 1.1})
        used_prev_indices = set(); used_next_indices = set(); current_modified_pairs = []
        if remaining_i and remaining_i_plus_1:
            for idx_next, r_next in enumerate(remaining_i_plus_1):
                best_match_idx = -1; min_norm_dist = 1.1
                for idx_prev, r_prev in enumerate(remaining_i):
                    try: len_max = max(len(r_prev), len(r_next), 1); distance = Levenshtein.distance(r_prev, r_next); norm_dist = distance / len_max
                    except Exception as e: logging.warning(f"Levenshtein error: {e}"); norm_dist = 1.1
                    if norm_dist < similarity_threshold and norm_dist < min_norm_dist: min_norm_dist = norm_dist; best_match_idx = idx_prev
                if best_match_idx != -1: potential_matches[idx_next] = {'best_prev_idx': best_match_idx, 'min_norm_dist': min_norm_dist}
            sorted_potential_matches = sorted(potential_matches.items(), key=lambda item: item[1]['min_norm_dist'])
            for idx_next, match_info in sorted_potential_matches:
                if idx_next in used_next_indices: continue
                best_prev_idx = match_info['best_prev_idx']
                if best_prev_idx not in used_prev_indices:
                    r_prev = remaining_i[best_prev_idx]; r_next = remaining_i_plus_1[idx_next] # type: ignore
                    current_modified_pairs.append((r_prev, r_next)); used_prev_indices.add(best_prev_idx); used_next_indices.add(idx_next)
            if current_modified_pairs: diff_results['modified'][class_label].extend(current_modified_pairs); logging.debug(f"  Class {class_label}: Found {len(current_modified_pairs)} modified rules (Threshold: {similarity_threshold:.2f}).")
        final_new = [r for idx, r in enumerate(remaining_i_plus_1) if idx not in used_next_indices]
        if final_new: diff_results['new'][class_label].extend(final_new); logging.debug(f"  Class {class_label}: Found {len(final_new)} new rules.")
        final_deleted = [r for idx, r in enumerate(remaining_i) if idx not in used_prev_indices]
        if final_deleted: diff_results['deleted'][class_label].extend(final_deleted); logging.debug(f"  Class {class_label}: Found {len(final_deleted)} deleted rules.")
    return diff_results


# --- Função de Geração de Relatório ---
def generate_diff_report(diff_results, chunk_data_i, chunk_data_i_plus_1, chunk_i_index):
    # ... (código como antes) ...
    report_lines = []; report_lines.append("\n" + "="*70); report_lines.append(f"--- Diff Chunk {chunk_i_index} -> Chunk {chunk_i_index + 1} ---"); report_lines.append("="*70)
    train_acc_i = chunk_data_i.get('train_acc', float('nan')); test_acc_next = chunk_data_i.get('test_acc_next', float('nan')); test_f1_next = chunk_data_i.get('test_f1_next', float('nan'))
    fit_i = chunk_data_i.get('fitness', float('nan')); fit_i_plus_1 = chunk_data_i_plus_1.get('fitness', float('nan'))
    dc_i = chunk_data_i.get('default_class', 'N/A'); dc_i_plus_1 = chunk_data_i_plus_1.get('default_class', 'N/A')
    report_lines.append(f"Performance:")
    report_lines.append(f"  - Train Acc (Chunk {chunk_i_index}):      {train_acc_i:.4f}")
    report_lines.append(f"  - Test Acc (Chunk {chunk_i_index+1}):     {test_acc_next:.4f}")
    report_lines.append(f"  - Test F1 (Chunk {chunk_i_index+1}):      {test_f1_next:.4f}")
    report_lines.append(f"Fitness:       {fit_i:.6f} -> {fit_i_plus_1:.6f}")
    report_lines.append(f"Default Class: {dc_i} -> {dc_i_plus_1}"); report_lines.append("-"*70)
    all_classes = set(diff_results['unchanged'].keys()) | set(diff_results['modified'].keys()) | set(diff_results['new'].keys()) | set(diff_results['deleted'].keys())
    if not all_classes: report_lines.append("No rules found in either chunk for comparison."); return "\n".join(report_lines)
    for class_label in sorted(list(all_classes)):
        report_lines.append(f"\n=== Class {class_label} ==="); has_content_for_class = False
        unchanged = diff_results['unchanged'].get(class_label, [])
        if unchanged: has_content_for_class=True; report_lines.append("\n  Unchanged Rules:"); [report_lines.append(f"    IF {rule} THEN Class {class_label}") for rule in sorted(unchanged)]
        modified = diff_results['modified'].get(class_label, [])
        if modified: has_content_for_class=True; report_lines.append("\n  Modified Rules:"); [report_lines.extend([f"  - Old: IF {old_rule} THEN Class {class_label}", f"  + New: IF {new_rule} THEN Class {class_label}", f"    (Similarity: {1 - Levenshtein.distance(old_rule, new_rule) / max(len(old_rule), len(new_rule), 1):.2f})"]) for old_rule, new_rule in sorted(modified, key=lambda pair: pair[1])]
        new = diff_results['new'].get(class_label, [])
        if new: has_content_for_class=True; report_lines.append("\n  New Rules:"); [report_lines.append(f"  + IF {rule} THEN Class {class_label}") for rule in sorted(new)]
        deleted = diff_results['deleted'].get(class_label, [])
        if deleted: has_content_for_class=True; report_lines.append("\n  Deleted Rules:"); [report_lines.append(f"  - IF {rule} THEN Class {class_label}") for rule in sorted(deleted)]
        if not has_content_for_class: report_lines.append("  (No changes or rules for this class in this transition)")
    return "\n".join(report_lines)


# --- Novas Funções para Sumários ---
def calculate_diff_counts(diff_results):
    # ... (código como antes, com 'remain' = unchanged + modified + new) ...
    counts = {'unchanged': 0, 'modified': 0, 'new': 0, 'deleted': 0, 'remain': 0}
    if not diff_results: return counts
    counts['unchanged'] = sum(len(rules) for rules in diff_results['unchanged'].values())
    counts['modified'] = sum(len(pairs) for pairs in diff_results['modified'].values())
    counts['new'] = sum(len(rules) for rules in diff_results['new'].values())
    counts['deleted'] = sum(len(rules) for rules in diff_results['deleted'].values())
    counts['remain'] = counts['unchanged'] + counts['modified'] + counts['new'] # Cálculo atualizado
    return counts

def generate_evolution_matrix_table(all_counts_per_transition):
    # ... (código como antes, com label 'Remain Rules (Chunk i+1)') ...
    if not all_counts_per_transition: logging.warning("No transition count data for evolution matrix."); return None
    data_for_df = defaultdict(list); transitions = []
    row_names = ["Unchanged", "Modified", "New", "Deleted", "Remain Rules (Chunk i+1)"] # Label atualizado
    for i, counts in enumerate(all_counts_per_transition):
        transition_label = f"{i}->{i+1}"; transitions.append(transition_label)
        data_for_df[row_names[0]].append(counts['unchanged']); data_for_df[row_names[1]].append(counts['modified']); data_for_df[row_names[2]].append(counts['new']); data_for_df[row_names[3]].append(counts['deleted']); data_for_df[row_names[4]].append(counts['remain'])
    try:
        df_matrix = pd.DataFrame(data_for_df, index=transitions).T
        print("\n" + "="*70); print("Rule Evolution Matrix (Counts per Transition)"); print("="*70); print(df_matrix.to_string()); print("="*70 + "\n"); return df_matrix
    except Exception as e: logging.error(f"Failed to create evolution matrix DataFrame: {e}"); return None

def plot_evolution_matrix(df_matrix, source_file_name, save_path=None):
     # ... (código como antes, com label 'Remain Rules (Chunk i+1)') ...
     if df_matrix is None or df_matrix.empty: logging.warning("DataFrame for matrix is empty. Skipping plot."); return
     try:
         plt.figure(figsize=(max(8, df_matrix.shape[1] * 0.8), max(5, df_matrix.shape[0]*0.6)))
         ordered_index = ["Unchanged", "Modified", "New", "Deleted", "Remain Rules (Chunk i+1)"] # Label atualizado
         existing_rows = [idx for idx in ordered_index if idx in df_matrix.index]
         if not existing_rows: logging.warning("No valid rows found to plot matrix heatmap."); plt.close(); return
         df_to_plot = df_matrix.reindex(existing_rows)
         sns.heatmap(df_to_plot, annot=True, fmt="d", cmap="Blues", linewidths=.5)
         plt.title(f"Rule Evolution Matrix - {source_file_name}"); plt.xlabel("Chunk Transition"); plt.ylabel("Change Category")
         plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
         if save_path: plot_dir = os.path.dirname(save_path);
         if plot_dir: os.makedirs(plot_dir, exist_ok=True); plt.savefig(save_path, bbox_inches='tight'); logging.info(f"Evolution matrix heatmap saved to: {save_path}") # type: ignore
         plt.close()
     except Exception as e: logging.error(f"Failed to generate evolution matrix heatmap: {e}", exc_info=True); plt.close()


def print_performance_summary(parsed_data):
    # ... (código como antes) ...
    if not parsed_data: return
    train_accs = [data.get('train_acc', np.nan) for data in parsed_data.values()]; test_accs = [data.get('test_acc_next', np.nan) for data in parsed_data.values()]; test_f1s = [data.get('test_f1_next', np.nan) for data in parsed_data.values()]
    avg_train_acc, std_train_acc = (np.nanmean(train_accs), np.nanstd(train_accs)) if not np.all(np.isnan(train_accs)) else (np.nan, np.nan)
    avg_test_acc, std_test_acc = (np.nanmean(test_accs), np.nanstd(test_accs)) if not np.all(np.isnan(test_accs)) else (np.nan, np.nan)
    avg_test_f1, std_test_f1 = (np.nanmean(test_f1s), np.nanstd(test_f1s)) if not np.all(np.isnan(test_f1s)) else (np.nan, np.nan)
    count = sum(1 for data in parsed_data.values() if not np.isnan(data.get('train_acc', np.nan))) # Conta chunks com dados válidos
    print("\n" + "="*70); print("Overall Performance Summary (from History File)"); print("="*70)
    print(f"Avg Train Accuracy: {avg_train_acc:.4f} +/- {std_train_acc:.4f} (over {count} chunks with data)")
    print(f"Avg Test Accuracy:  {avg_test_acc:.4f} +/- {std_test_acc:.4f} (over {count} chunks with data)")
    print(f"Avg Test F1:        {avg_test_f1:.4f} +/- {std_test_f1:.4f} (over {count} chunks with data)")
    print("="*70 + "\n")


# --- Main Execution ---
def main():
    # ... (código como antes) ...
    parser = argparse.ArgumentParser(description="Analyze rule changes between chunks from a RulesHistory*.txt file.")
    parser.add_argument("history_file", help="Path to the RulesHistory_*.txt file.")
    parser.add_argument( "-t", "--threshold", type=float, default=0.35, help="Normalized Levenshtein distance threshold for modified rules (0.0 to 1.0, lower means more similar). Default: 0.35" )
    parser.add_argument( "-o", "--output", help="Optional base path to save outputs (e.g., 'analysis_run1'). Suffixes _report.txt, _matrix.csv, _matrix.png added." )
    parser.add_argument( "-v", "--verbose", action="store_true", help="Enable debug logging." )
    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO;
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler); logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)-8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if not 0.0 <= args.threshold <= 1.0: logging.error("Similarity threshold must be between 0.0 and 1.0."); return

    logging.info(f"Analyzing rule history file: {args.history_file}"); logging.info(f"Using similarity threshold: {args.threshold}")
    parsed_data = parse_rules_history(args.history_file)
    if parsed_data is None or not isinstance(parsed_data, dict): logging.error("Failed to parse history file."); return
    chunk_indices = sorted(parsed_data.keys())
    if len(chunk_indices) < 2: logging.warning("Need at least two chunks to compare."); return

    all_reports = []; all_counts = []
    report_header = [f"Rule Diff Analysis Report", f"Source File: {os.path.basename(args.history_file)}", f"Similarity Threshold: {args.threshold}"]
    all_reports.extend(report_header)

    for i in range(len(chunk_indices) - 1):
        chunk_idx_i = chunk_indices[i]; chunk_idx_i_plus_1 = chunk_indices[i+1]
        if chunk_idx_i not in parsed_data or chunk_idx_i_plus_1 not in parsed_data: logging.warning(f"Missing data for chunk {chunk_idx_i} or {chunk_idx_i_plus_1}. Skipping."); continue
        logging.info(f"Comparing Chunk {chunk_idx_i} vs Chunk {chunk_idx_i_plus_1}...")
        diff = compare_chunk_rules( parsed_data[chunk_idx_i], parsed_data[chunk_idx_i_plus_1], similarity_threshold=args.threshold )
        if diff:
            report = generate_diff_report( diff, parsed_data[chunk_idx_i], parsed_data[chunk_idx_i_plus_1], chunk_idx_i )
            print(report)
            all_reports.append(report)
            counts = calculate_diff_counts(diff) # Usa 'remain' atualizado
            all_counts.append(counts)
        else: logging.warning(f"Comparison failed for chunks {chunk_idx_i} -> {chunk_idx_i_plus_1}")

    df_matrix = generate_evolution_matrix_table(all_counts) # Usa label 'Remain Rules...'
    print_performance_summary(parsed_data)

    if args.output:
        output_base_path = args.output; output_dir = os.path.dirname(output_base_path);
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        report_path = output_base_path + "_report.txt"
        try: # Salva relatório texto
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(all_reports))
                # Adiciona sumário de performance ao final
                train_accs = [data.get('train_acc', np.nan) for data in parsed_data.values()]; test_accs = [data.get('test_acc_next', np.nan) for data in parsed_data.values()]; test_f1s = [data.get('test_f1_next', np.nan) for data in parsed_data.values()]
                avg_train_acc, std_train_acc = (np.nanmean(train_accs), np.nanstd(train_accs)) if not np.all(np.isnan(train_accs)) else (np.nan, np.nan); avg_test_acc, std_test_acc = (np.nanmean(test_accs), np.nanstd(test_accs)) if not np.all(np.isnan(test_accs)) else (np.nan, np.nan); avg_test_f1, std_test_f1 = (np.nanmean(test_f1s), np.nanstd(test_f1s)) if not np.all(np.isnan(test_f1s)) else (np.nan, np.nan); count = sum(1 for data in parsed_data.values() if not np.isnan(data.get('train_acc', np.nan)))
                f.write("\n\n" + "="*70 + "\nOverall Performance Summary (from History File)\n" + "="*70 + "\n"); f.write(f"Avg Train Accuracy: {avg_train_acc:.4f} +/- {std_train_acc:.4f} (over {count} chunks)\n"); f.write(f"Avg Test Accuracy:  {avg_test_acc:.4f} +/- {std_test_acc:.4f} (over {count} chunks)\n"); f.write(f"Avg Test F1:        {avg_test_f1:.4f} +/- {std_test_f1:.4f} (over {count} chunks)\n"); f.write("="*70 + "\n")
            logging.info(f"Combined diff report saved to: {report_path}")
        except Exception as e: logging.error(f"Failed to save text report: {e}")
        if df_matrix is not None: # Salva CSV e Plot da matriz
            matrix_csv_path = output_base_path + "_matrix.csv"; matrix_plot_path = output_base_path + "_matrix.png"
            try: df_matrix.to_csv(matrix_csv_path); logging.info(f"Evolution matrix table saved to: {matrix_csv_path}")
            except Exception as e: logging.error(f"Failed to save matrix CSV: {e}")
            try: plot_evolution_matrix(df_matrix, os.path.basename(args.history_file), save_path=matrix_plot_path)
            except Exception as e: logging.error(f"Failed to save matrix plot: {e}")

if __name__ == "__main__":
    main()