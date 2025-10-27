# chunk_transition_analyzer.py

import os
import re
from collections import defaultdict
import Levenshtein # Para o pareamento inicial de regras modificadas
import math

# --- Importações das Partes Anteriores ---
try:
    from rule_condition_parser import parse_rule_condition
    from literal_comparison import calculate_atomic_condition_change_severity
    # calculate_rule_modification_severity está em rule_similarity_analyzer.py
    from rule_similarity_analyzer import calculate_rule_modification_severity
except ImportError as e:
    print(f"Erro de importação: {e}")
    print("Certifique-se de que os scripts 'rule_condition_parser_v12.py', 'literal_comparison.py', e 'rule_similarity_analyzer.py' estão acessíveis.")
    exit()

# --- Pesos para STT (a serem ajustados) ---
W_INSTABILITY = 0.6
W_MODIFICATION_IMPACT = 0.4

# --- Função de Parsing de Histórico (assumindo que está correta e como na v12 da discussão anterior) ---
def parse_rules_history_to_asts(file_path):
    """
    Parseia o arquivo de histórico de regras e retorna um dicionário com
    chunks como chaves e um dicionário contendo metadados, 'rules_asts' e
    'rules_raw_strings' (ambos são dicts class_label -> lista) como valores.
    """
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo de histórico não encontrado: {file_path}")
        return None

    chunks_data = {}
    current_chunk_idx = -1
    parsing_state = "LOOKING_FOR_CHUNK"
    current_rules_raw_strings_by_class = defaultdict(list)
    current_chunk_metadata = {}
    accumulated_condition_lines = []

    chunk_header_re = re.compile(r"^-+ Chunk (\d+) \(Trained\)")
    rule_start_re = re.compile(r"^\s*\d+:\s*IF (.*)")
    rule_end_re = re.compile(r"(.*?) THEN Class\s+(\S+)$")
    train_perf_re = re.compile(r"Train Perf \(Chunk \d+\): TrainAcc=([-\d.]+)")
    test_perf_re = re.compile(r"Test Perf \(Chunk \d+\): TestAcc=([-\d.]+),\s*TestF1=([-\d.]+)")
    fitness_re = re.compile(r"^\s*Fitness:\s*([-\d.]+)", re.IGNORECASE)
    default_class_re = re.compile(r"^\s*Default Class:\s*(\S+)", re.IGNORECASE)
    class_header_re = re.compile(r"^\s*Class\s+(\S+):")
    rules_total_re = re.compile(r"^\s*Rules \((\d+) total\):")

    # print(f"Parseando arquivo de histórico para ASTs: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, raw_line in enumerate(f):
                line = raw_line.strip()

                chunk_match = chunk_header_re.match(raw_line)
                if chunk_match:
                    if current_chunk_idx != -1:
                        if accumulated_condition_lines:
                            print(f"Aviso L{line_num+1}: Chunk {current_chunk_idx} terminou com regra multi-linha incompleta.")
                            accumulated_condition_lines = []
                        
                        asts_by_class = defaultdict(list)
                        for cls_label, raw_conds in current_rules_raw_strings_by_class.items():
                            for cond_str in raw_conds:
                                ast = parse_rule_condition(cond_str)
                                if ast:
                                    asts_by_class[cls_label].append(ast)
                        current_chunk_metadata['rules_asts'] = dict(asts_by_class)
                        current_chunk_metadata['rules_raw_strings'] = dict(current_rules_raw_strings_by_class)
                        chunks_data[current_chunk_idx] = current_chunk_metadata
                        # print(f"  Chunk {current_chunk_idx} salvo com {sum(len(v) for v in asts_by_class.values())} ASTs.")

                    current_chunk_idx = int(chunk_match.group(1))
                    current_rules_raw_strings_by_class = defaultdict(list)
                    current_chunk_metadata = {'chunk_id': current_chunk_idx, 'rules_asts': {}, 'rules_raw_strings': {}}
                    parsing_state = "IN_CHUNK_HEADER"
                    
                    test_match = test_perf_re.search(raw_line)
                    if test_match:
                        try:
                            current_chunk_metadata['test_acc_next'] = float(test_match.group(1))
                            current_chunk_metadata['test_f1_next'] = float(test_match.group(2))
                        except (ValueError, IndexError): pass
                    continue

                if current_chunk_idx == -1: continue

                if parsing_state == "IN_CHUNK_HEADER":
                    train_match = train_perf_re.search(line)
                    if train_match:
                        try: current_chunk_metadata['train_acc'] = float(train_match.group(1))
                        except (ValueError, IndexError): pass
                        continue # Ainda esperando '---'
                    elif line.startswith("---"):
                        parsing_state = "IN_CHUNK_META"
                        continue
                    elif "Fitness:" in line or "Default Class:" in line or "Class " in line or "Rules (" in line or rule_start_re.match(line):
                        parsing_state = "IN_CHUNK_META"
                        # Reprocessar linha
                
                if parsing_state == "IN_CHUNK_META":
                    fit_match = fitness_re.match(line)
                    if fit_match: # ... (extração de fitness)
                        try: current_chunk_metadata['fitness'] = float(fit_match.group(1))
                        except ValueError: pass; continue
                    dc_match = default_class_re.match(line)
                    if dc_match: # ... (extração de default class)
                        dc_str = dc_match.group(1)
                        try: current_chunk_metadata['default_class'] = int(dc_str)
                        except ValueError:
                            try: current_chunk_metadata['default_class'] = float(dc_str)
                            except ValueError: current_chunk_metadata['default_class'] = dc_str
                        continue
                    rules_total_match = rules_total_re.match(line)
                    if rules_total_match: # ... (extração de total de regras reportado)
                        try: current_chunk_metadata['rules_total_reported'] = int(rules_total_match.group(1))
                        except ValueError: pass; continue
                    class_header_match = class_header_re.match(line)
                    if class_header_match: continue

                    rule_start_match = rule_start_re.match(line)
                    if rule_start_match:
                        content_after_if = rule_start_match.group(1).strip()
                        rule_end_on_same_line_match = rule_end_re.search(content_after_if)
                        if rule_end_on_same_line_match:
                            condition_only_str = rule_end_on_same_line_match.group(1).strip()
                            class_label_str = rule_end_on_same_line_match.group(2).strip()
                            try: class_label = int(class_label_str)
                            except ValueError:
                                try: class_label = float(class_label_str)
                                except ValueError: class_label = class_label_str
                            current_rules_raw_strings_by_class[class_label].append(condition_only_str)
                        else:
                            accumulated_condition_lines = [content_after_if]
                            parsing_state = "PARSING_RULE"
                        continue
                    if line: # Linha não reconhecida em IN_CHUNK_META
                        # print(f"Aviso L{line_num+1}: Linha não reconhecida em IN_CHUNK_META: '{line[:50]}...'")
                        pass
                    continue

                if parsing_state == "PARSING_RULE":
                    # Adiciona linha atual ao acumulado ANTES de verificar se é o fim
                    if line: # Evita adicionar linhas vazias que podem ter sido permitidas
                        accumulated_condition_lines.append(line)
                    
                    # Tenta encontrar "THEN Class" no conteúdo acumulado
                    # Isso é mais robusto para condições que contêm a palavra "THEN" ou "Class"
                    full_accumulated_str = " ".join(accumulated_condition_lines)
                    rule_end_match = rule_end_re.search(full_accumulated_str)

                    if rule_end_match:
                        condition_only_str = rule_end_match.group(1).strip()
                        class_label_str = rule_end_match.group(2).strip()
                        condition_only_str = re.sub(r'\s+', ' ', condition_only_str) # Normaliza espaços

                        try: class_label = int(class_label_str)
                        except ValueError:
                            try: class_label = float(class_label_str)
                            except ValueError: class_label = class_label_str
                        
                        current_rules_raw_strings_by_class[class_label].append(condition_only_str)
                        accumulated_condition_lines = []
                        parsing_state = "IN_CHUNK_META"
                    # Se não for o fim da regra, e a linha atual não é um novo header/meta, continua acumulando.
                    # Se for um novo header/meta, pode indicar uma regra malformada.
                    elif chunk_header_re.match(raw_line) or (line.startswith("---") and not accumulated_condition_lines) or \
                         ("Fitness:" in line and not accumulated_condition_lines) or \
                         ("Default Class:" in line and not accumulated_condition_lines) or \
                         (class_header_re.match(line) and not accumulated_condition_lines) or \
                         (rules_total_re.match(line) and not accumulated_condition_lines):
                        if accumulated_condition_lines: # Só se realmente tinha algo acumulado
                           print(f"Aviso L{line_num+1}: Nova seção ({line[:30]}) iniciada com regra multi-linha incompleta. Descartando: {' '.join(accumulated_condition_lines)}")
                        accumulated_condition_lines = []
                        parsing_state = "IN_CHUNK_META" # Volta para meta e a linha atual será reprocessada no próximo loop
                        # Para reprocessar a linha atual no novo estado
                        # Precisamos de uma forma de não avançar o iterador do arquivo, o que é complexo
                        # Uma maneira mais simples é não dar continue e deixar o topo do loop lidar com ela
                        # Mas isso pode ser complicado. Por enquanto, a perda de uma linha de meta é menor.
                        # Se a linha era um novo chunk_header, o if no topo pega.
                        # Se era "---" ou outra meta, ela será reavaliada em IN_CHUNK_META.
                        # Se for reprocessar, precisamos fazer algo aqui ou o continue abaixo pula.
                        # Melhor é ter certeza que a linha não era continuação e então mudar estado e reprocessar
                        # Essa lógica de reprocessamento é complexa com loop for.
                        # O mais seguro é que a linha que quebra PARSING_RULE seja um separador claro.
                        # Por enquanto, vamos assumir que a linha que não é rule_end_match é continuação.
                        # Se for um novo chunk, o if no topo cuidará disso e salvará o anterior.

            if current_chunk_idx != -1:
                if accumulated_condition_lines:
                    print(f"Aviso: Fim do arquivo com regra multi-linha incompleta no Chunk {current_chunk_idx}.")
                
                asts_by_class = defaultdict(list)
                for cls_label, raw_conds in current_rules_raw_strings_by_class.items():
                    for cond_str in raw_conds:
                        ast = parse_rule_condition(cond_str)
                        if ast:
                            asts_by_class[cls_label].append(ast)
                current_chunk_metadata['rules_asts'] = dict(asts_by_class)
                current_chunk_metadata['rules_raw_strings'] = dict(current_rules_raw_strings_by_class)
                chunks_data[current_chunk_idx] = current_chunk_metadata
                # print(f"  Chunk {current_chunk_idx} (final) salvo com {sum(len(v) for v in asts_by_class.values())} ASTs.")

        if not chunks_data:
            print(f"Nenhum chunk parseado com sucesso do arquivo: {file_path}")
            return None
        
        total_rules_parsed = sum(sum(len(rules_list) for class_label, rules_list in chunk_data['rules_asts'].items()) for chunk_idx, chunk_data in chunks_data.items())
        # print(f"Parseados {len(chunks_data)} chunks do arquivo {file_path}, total de {total_rules_parsed} ASTs de regras geradas.")
        return chunks_data

    except Exception as e:
        print(f"Erro EXCEPCIONAL ao parsear arquivo de histórico {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_chunk_transition(chunk_data_i, chunk_data_j, 
                             levenshtein_similarity_threshold=0.7, 
                             sm_threshold_for_modified=0.9):
    """
    Compara dois chunks, identifica mudanças nas regras e calcula a severidade.
    Usa ASTs para comparação de regras modificadas e strings brutas para Levenshtein.
    sm_threshold_for_modified: se SM < threshold, é considerada modificada.
    levenshtein_similarity_threshold: 1 - (dist / max_len). Se similaridade > threshold, é candidato.
    """
    results = {
        "unchanged_count": 0, "modified_count": 0, "new_count": 0, "deleted_count": 0,
        "total_rules_i": 0, "total_rules_j": 0,
        "modified_pairs_details": [], # Lista de {'old_raw', 'new_raw', 'sm_score'}
        "MI": 0.0, "SMM": 0.0, "STT": 0.0,
    }

    # Preparar listas de regras com AST, raw_string e classe
    rules_i_list = []
    for class_label, ast_list in chunk_data_i.get('rules_asts', {}).items():
        raw_list = chunk_data_i.get('rules_raw_strings', {}).get(class_label, [])
        for idx, ast in enumerate(ast_list):
            rules_i_list.append({'ast': ast, 'raw': raw_list[idx] if idx < len(raw_list) else str(ast), 'class': class_label, 'matched_to_j': -1})
    
    rules_j_list = []
    for class_label, ast_list in chunk_data_j.get('rules_asts', {}).items():
        raw_list = chunk_data_j.get('rules_raw_strings', {}).get(class_label, [])
        for idx, ast in enumerate(ast_list):
            rules_j_list.append({'ast': ast, 'raw': raw_list[idx] if idx < len(raw_list) else str(ast), 'class': class_label, 'matched_from_i': -1})

    results["total_rules_i"] = len(rules_i_list)
    results["total_rules_j"] = len(rules_j_list)

    # 1. Identificar Inalteradas (AST idêntica E classe idêntica)
    for i_idx, r_i in enumerate(rules_i_list):
        if r_i['matched_to_j'] != -1: continue # Já pareada
        for j_idx, r_j in enumerate(rules_j_list):
            if r_j['matched_from_i'] != -1: continue # Já pareada
            if r_i['class'] == r_j['class'] and r_i['ast'] == r_j['ast']:
                results["unchanged_count"] += 1
                r_i['matched_to_j'] = j_idx
                r_j['matched_from_i'] = i_idx
                break
    
    # 2. Identificar Modificadas (entre as restantes, mesma classe)
    # Lista de tuplas (sm_score, i_idx, j_idx) para ordenação
    potential_modifications = [] 
    for i_idx, r_i in enumerate(rules_i_list):
        if r_i['matched_to_j'] != -1: continue # Já é inalterada
        if not r_i['raw']: continue # Precisa da string bruta

        for j_idx, r_j in enumerate(rules_j_list):
            if r_j['matched_from_i'] != -1: continue # Já é inalterada ou pareada com outra modificada
            if not r_j['raw']: continue
            
            if r_i['class'] == r_j['class']: # Apenas compara regras da mesma classe para modificação
                # Filtro Levenshtein
                dist = Levenshtein.distance(r_i['raw'], r_j['raw'])
                max_len = max(len(r_i['raw']), len(r_j['raw']), 1) # Evita divisão por zero
                norm_dist = dist / max_len
                lev_similarity = 1.0 - norm_dist
                
                if lev_similarity >= levenshtein_similarity_threshold: # Ex: 0.3 ou mais de similaridade Levenshtein
                    # Calcular severidade detalhada (SM)
                    sm_score = calculate_rule_modification_severity(r_i['ast'], r_j['ast'])
                    if sm_score < sm_threshold_for_modified: # Ex: SM < 0.9 (não é uma mudança total)
                        potential_modifications.append({'sm': sm_score, 'lev_sim': lev_similarity, 'i_idx': i_idx, 'j_idx': j_idx})

    # Ordenar os candidatos a modificados pelo melhor SM (menor severidade), depois por Levenshtein (maior similaridade)
    potential_modifications.sort(key=lambda x: (x['sm'], -x['lev_sim']))

    modified_pairs_severities_list = []
    for mod_candidate in potential_modifications:
        i_idx, j_idx = mod_candidate['i_idx'], mod_candidate['j_idx']
        # Verifica se algum dos dois já foi pareado (porque um r_i pode ter múltiplos r_j candidatos, e vice-versa)
        if rules_i_list[i_idx]['matched_to_j'] == -1 and rules_j_list[j_idx]['matched_from_i'] == -1:
            rules_i_list[i_idx]['matched_to_j'] = j_idx
            rules_j_list[j_idx]['matched_from_i'] = i_idx
            results["modified_count"] += 1
            modified_pairs_severities_list.append(mod_candidate['sm'])
            results["modified_pairs_details"].append({
                'old_raw': rules_i_list[i_idx]['raw'],
                'new_raw': rules_j_list[j_idx]['raw'],
                'sm_score': mod_candidate['sm'],
                'lev_similarity': mod_candidate['lev_sim']
            })

    # 3. Contar Novas e Deletadas
    results["new_count"] = sum(1 for r_j in rules_j_list if r_j['matched_from_i'] == -1)
    results["deleted_count"] = sum(1 for r_i in rules_i_list if r_i['matched_to_j'] == -1)

    # 4. Calcular SMM, MI, STT
    if results["modified_count"] > 0:
        results["SMM"] = sum(modified_pairs_severities_list) / results["modified_count"]
    
    total_involved_in_transition = results["total_rules_i"] + results["total_rules_j"]
    if total_involved_in_transition > 0:
        # MI baseada no número de regras que não são inalteradas nem modificadas (ou seja, puramente novas + puramente deletadas)
        # Ou, mais simples, (novas + deletadas) em relação ao "espaço total"
        results["MI"] = (results["new_count"] + results["deleted_count"]) / total_involved_in_transition
    else:
        results["MI"] = 0.0

    prop_modified = 0.0
    if results["total_rules_j"] > 0:
        prop_modified = results["modified_count"] / results["total_rules_j"]

    results["STT"] = (W_INSTABILITY * results["MI"] +
                      W_MODIFICATION_IMPACT * prop_modified * results["SMM"])
    results["STT"] = min(max(results["STT"], 0.0), 1.0)
    
    # Remover a lista completa de detalhes dos pares do resultado principal para não poluir o log de resumo
    # Poderia ser retornado separadamente se necessário para análise detalhada
    final_results_summary = {k:v for k,v in results.items() if k != "modified_pairs_details"}
    final_results_summary["modified_pairs_severities_scores"] = modified_pairs_severities_list


    return final_results_summary, results["modified_pairs_details"] # Retorna o resumo e os detalhes


# --- Coleta de Quantitativos (sem alteração) ---
def collect_ast_quantitatives(ast_node):
    quantitatives = {"and_count": 0, "or_count": 0, "atomic_condition_count": 0, "threshold_values": []}
    if not isinstance(ast_node, list): return quantitatives
    if len(ast_node) == 3 and isinstance(ast_node[0], str) and ast_node[0] not in ("AND", "OR") and \
       isinstance(ast_node[1], str) and ast_node[1] not in ("AND", "OR"):
        quantitatives["atomic_condition_count"] = 1
        if isinstance(ast_node[2], (int, float)):
            quantitatives["threshold_values"].append(ast_node[2])
        return quantitatives
    if len(ast_node) == 3 and ast_node[0] in ("AND", "OR"):
        op = ast_node[0]
        if op == "AND": quantitatives["and_count"] = 1
        elif op == "OR": quantitatives["or_count"] = 1
        left_quants = collect_ast_quantitatives(ast_node[1])
        right_quants = collect_ast_quantitatives(ast_node[2])
        quantitatives["and_count"] += left_quants["and_count"] + right_quants["and_count"]
        quantitatives["or_count"] += left_quants["or_count"] + right_quants["or_count"]
        quantitatives["atomic_condition_count"] += left_quants["atomic_condition_count"] + right_quants["atomic_condition_count"]
        quantitatives["threshold_values"].extend(left_quants["threshold_values"]); quantitatives["threshold_values"].extend(right_quants["threshold_values"])
        return quantitatives
    return quantitatives

# --- Main ---
if __name__ == "__main__":
    print("Iniciando Parte 4: Cálculo STT e Quantitativos para TODAS as Transições (com pareamento de modificadas)...\n")
    
    history_file_path = r"G:\Outros computadores\Meu laptop\Downloads\DSL-AG\drift_experiment_results_big\RBF_c1_c3move_c4add_Abrupt\run_1\RulesHistory_RBF_c1_c3move_c4add_Abrupt_Run1.txt"
    # history_file_path = r"C:\Users\EAI.001\Downloads\DSL-AG\drift_experiment_results_big\AGRAWAL_f1f6f9_Abrupt\run_1\RulesHistory_AGRAWAL_f1f6f9_Abrupt_Run1.txt"

    # Limiares para pareamento de regras modificadas
    LEVENSHTEIN_PRE_FILTER = 0.5  # Similaridade Levenshtein mínima para ser candidato (0.0 a 1.0)
    SM_THRESHOLD_MODIFIED = 0.8   # Severidade detalhada (SM) MÁXIMA para ser considerado modificado (0.0 a 1.0)
                                  # SM < threshold -> é modificada
   
    if not os.path.exists(history_file_path):
        print(f"ERRO: Arquivo de histórico de teste não encontrado: {history_file_path}")
    else:
        parsed_chunks = parse_rules_history_to_asts(history_file_path)

        if parsed_chunks and len(parsed_chunks) >= 2:
            chunk_indices = sorted(parsed_chunks.keys())
            all_transition_data_for_graph = []

            for i in range(len(chunk_indices) - 1):
                idx_i = chunk_indices[i]; idx_j = chunk_indices[i+1]
                print(f"\n\n--- Analisando Transição: Chunk {idx_i} -> Chunk {idx_j} ---")
                
                chunk_i_data = parsed_chunks[idx_i]; chunk_j_data = parsed_chunks[idx_j]
                
                transition_summary, modified_details = analyze_chunk_transition(
                    chunk_i_data, chunk_j_data,
                    levenshtein_similarity_threshold=LEVENSHTEIN_PRE_FILTER,
                    sm_threshold_for_modified=SM_THRESHOLD_MODIFIED
                )
                print("\n  Resultados da Transição (Resumo):")
                for key, value in transition_summary.items():
                    if key == "modified_pairs_severities_scores":
                         print(f"    {key}: {len(value)} scores, Média SMM calculada: {transition_summary['SMM']:.4f}")
                         # print(f"    Scores individuais: {[f'{s:.4f}' for s in value]}") # Descomente para ver scores
                    elif isinstance(value, float): print(f"    {key}: {value:.4f}")
                    else: print(f"    {key}: {value}")
                
                # if modified_details:
                #     print("\n  Detalhes das Regras Modificadas:")
                #     for detail_idx, detail in enumerate(modified_details):
                #         print(f"    Par {detail_idx+1}: SM={detail['sm_score']:.4f}, LevSim={detail['lev_similarity']:.4f}")
                #         print(f"      Antiga: {detail['old_raw']}")
                #         print(f"      Nova  : {detail['new_raw']}")


                # Coleta de quantitativos para o chunk de destino
                # ... (código de coleta de quantitativos como antes) ...
                print(f"\n  Quantitativos para as Regras do Chunk {idx_j} (Destino da Transição):")
                total_quants_j = { "and_count": 0, "or_count": 0, "atomic_condition_count": 0, "threshold_values": []}
                if 'rules_asts' in chunk_j_data:
                    for class_label, ast_list in chunk_j_data['rules_asts'].items():
                        for ast_rule in ast_list: # Renomeado para evitar conflito com módulo ast
                            quants = collect_ast_quantitatives(ast_rule)
                            total_quants_j["and_count"] += quants["and_count"]
                            total_quants_j["or_count"] += quants["or_count"]
                            total_quants_j["atomic_condition_count"] += quants["atomic_condition_count"]
                            total_quants_j["threshold_values"].extend(quants["threshold_values"])
                print(f"    Total de Operadores AND: {total_quants_j['and_count']}")
                print(f"    Total de Operadores OR: {total_quants_j['or_count']}")
                print(f"    Total de Condições Atômicas: {total_quants_j['atomic_condition_count']}")
                avg_threshold, min_threshold, max_threshold = None, None, None
                if total_quants_j['threshold_values']:
                    avg_threshold = sum(total_quants_j['threshold_values']) / len(total_quants_j['threshold_values'])
                    min_threshold = min(total_quants_j['threshold_values'])
                    max_threshold = max(total_quants_j['threshold_values'])
                    print(f"    Thresholds: Média={avg_threshold:.4f}, Min={min_threshold:.4f}, Max={max_threshold:.4f} (de {len(total_quants_j['threshold_values'])} valores)")
                else:
                    print("    Nenhum threshold numérico encontrado.")
                
                # Armazenar dados para o gráfico futuro
                data_for_graph = {
                    "transition": f"{idx_i}->{idx_j}",
                    "STT": transition_summary["STT"], "MI": transition_summary["MI"], "SMM": transition_summary["SMM"],
                    "unchanged": transition_summary["unchanged_count"], "modified": transition_summary["modified_count"],
                    "new": transition_summary["new_count"], "deleted": transition_summary["deleted_count"],
                    "total_rules_j": transition_summary["total_rules_j"],
                    "ands_j": total_quants_j["and_count"], "ors_j": total_quants_j["or_count"],
                    "atomics_j": total_quants_j["atomic_condition_count"], "avg_thresh_j": avg_threshold
                }
                all_transition_data_for_graph.append(data_for_graph)

            print("\n\n--- Resumo de Todas as Transições para Gráfico ---")
            import pandas as pd
            df_transitions = pd.DataFrame(all_transition_data_for_graph)
            print(df_transitions.to_string())

        else:
            print("Não foi possível parsear o arquivo de histórico ou não há chunks suficientes para comparação.")