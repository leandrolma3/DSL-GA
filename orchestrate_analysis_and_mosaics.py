# orchestrate_analysis_and_mosaics.py

import os
import subprocess
import glob
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns # <<< ADICIONE ESTA LINHA

# --- Configurações Globais ---
# Altere estes caminhos conforme necessário
# ROOT_PATH = r"C:\Users\EAI.001\Downloads\DSL-AG\drift_experiment_results_big"
# PYTHON_EXECUTABLE = "python" # ou o caminho completo para o seu executável python, ex: "C:/Python39/python.exe"
# RULE_ANALYZER_SCRIPT_PATH = r"C:\caminho\completo\para\rule_diff_analyzer.py" # CAMINHO ABSOLUTO RECOMENDADO
COMPOSITE_MOSAIC_OUTPUT_FOLDER = "composite_mosaics_final"

RUN_FOLDER_NAME = "run_1"

# Nomes de arquivo esperados (prefixos ou nomes completos se fixos)
# A função de busca usará glob, então '*' pode ser usado.
IMAGE_PATTERNS = {
    "main_accuracy": "Plot_AccuracyPeriodic_*.png", # Em run_1/plots/
    "heatmap": "Plot_RuleComponents_Heatmap_*.png", # Em run_1/plots/
    "attribute_usage": "Plot_AttributeUsage_*.png", # Em run_1/plots/
    "evolution_matrix_pattern": "*_matrix.png",   # Gerado por rule_analyzer em run_1/ (o nome base será run_1)
    "feature_drift": "FeatureDrift_*.png"           # Em run_1/
}

# def find_target_subdirectories(root_path, folders_to_ignore=None):
#     """
#     Encontra subdiretórios no root_path que contêm '_' no nome,
#     ignorando pastas especificadas.
#     """
#     target_dirs = []
#     if folders_to_ignore is None:
#         folders_to_ignore = set()

#     if not os.path.isdir(root_path):
#         print(f"Erro: O caminho raiz '{root_path}' não é um diretório válido.")
#         return target_dirs

#     for item in os.listdir(root_path):
#         item_path = os.path.join(root_path, item)
#         if os.path.isdir(item_path) and '_' in item and item not in folders_to_ignore:
#             target_dirs.append(item_path)
#     return target_dirs

def find_target_subdirectories(root_path, folders_to_ignore=None):
    """
    Finds all subdirectories in the root_path,
    optionally ignoring specified folders.
    """
    target_dirs = []
    if folders_to_ignore is None:
        folders_to_ignore = set()

    if not os.path.isdir(root_path):
        print(f"Error: Root path '{root_path}' is not a valid directory.")
        return target_dirs

    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        # Agora verifica apenas se é um diretório e se não está na lista de ignorados
        if os.path.isdir(item_path) and item not in folders_to_ignore:
            target_dirs.append(item_path)
    return target_dirs


def find_first_matching_file(directory, pattern):
    """
    Encontra o primeiro arquivo em um diretório que corresponde ao padrão glob.
    Retorna o caminho completo do arquivo ou None se não encontrar.
    """
    if not os.path.isdir(directory):
        # print(f"Aviso: Diretório para busca de imagem não encontrado: {directory}")
        return None
    
    search_pattern = os.path.join(directory, pattern)
    files_found = glob.glob(search_pattern)
    if files_found:
        # print(f"  Encontrado para padrão '{pattern}' em '{directory}': {files_found[0]}")
        return files_found[0] # Retorna o primeiro encontrado
    # print(f"  Nenhum arquivo encontrado para padrão '{pattern}' em '{directory}'")
    return None

def run_rule_analyzer_for_subdir(sub_dir_path, python_executable, rule_analyzer_script):
    """
    Executa o script rule_diff_analyzer.py para um subdiretório.
    Retorna o caminho para a imagem da matriz de evolução gerada se bem-sucedido, caso contrário None.
    """
    sub_dir_name = os.path.basename(sub_dir_path) # <<< ADICIONE ESTA LINHA
    print(f"  Executando rule_diff_analyzer para: {sub_dir_name}") # Agora sub_dir_name está definido
    run_1_path = os.path.join(sub_dir_path, "run_1")
    
    if not os.path.isdir(run_1_path):
        print(f"    Erro: Pasta 'run_1' não encontrada em {sub_dir_path}")
        return None

    rules_history_file = find_first_matching_file(run_1_path, "RulesHistory_*.txt")
    if not rules_history_file:
        print(f"    Erro: Arquivo RulesHistory_*.txt não encontrado em {run_1_path}")
        return None
    
    output_base_name_for_analyzer = "rule_evolution_analysis"
    output_path_for_analyzer = os.path.join(run_1_path, output_base_name_for_analyzer)

    command = [
        python_executable,
        rule_analyzer_script,
        rules_history_file,
        "-o", output_path_for_analyzer
    ]

    try:
        print(f"    Comando: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=False, shell=False)
        if result.returncode == 0:
            print(f"    rule_diff_analyzer.py executado com sucesso para {sub_dir_name}.") # sub_dir_name já está definido
            evolution_matrix_image_path = output_path_for_analyzer + "_matrix.png"
            if os.path.exists(evolution_matrix_image_path):
                return evolution_matrix_image_path
            else:
                print(f"    Erro: Imagem da matriz de evolução esperada '{evolution_matrix_image_path}' não encontrada após execução.")
                # print(f"    Stdout: {result.stdout}") 
                # print(f"    Stderr: {result.stderr}")
                return None
        else:
            # sub_dir_name já está definido para esta mensagem de erro
            print(f"    Erro ao executar rule_diff_analyzer.py para {sub_dir_name}. Código de retorno: {result.returncode}")
            print(f"    Stderr: {result.stderr}")
            return None
    except FileNotFoundError:
        print(f"    Erro: Executável Python '{python_executable}' ou script '{rule_analyzer_script}' não encontrado.")
        return None
    except Exception as e:
        print(f"    Exceção ao executar rule_diff_analyzer.py: {e}")
        return None

def create_composite_mosaic(main_img_path, small_img_paths, output_mosaic_file):
    """
    Cria um mosaico composto.
    Layout: Imagem principal à esquerda, 4 imagens menores em uma grade 2x2 à direita.
    small_img_paths: lista de 4 caminhos de imagens menores.
    """
    print(f"  Criando mosaico composto: {output_mosaic_file}")
    try:
        main_img_orig = Image.open(main_img_path)
        small_imgs_orig = [Image.open(p) for p in small_img_paths if p and os.path.exists(p)]

        if len(small_imgs_orig) != 4:
            print(f"    Aviso: Esperadas 4 imagens menores, encontradas {len(small_imgs_orig)}. Pulando mosaico.")
            main_img_orig.close()
            for img in small_imgs_orig: img.close()
            return

        # Definir dimensões do layout (exemplo, pode precisar de ajuste)
        # Dimensões da imagem principal no mosaico
        main_w, main_h = 800, 600 # Exemplo
        # Dimensões de cada imagem pequena no mosaico
        small_w, small_h = main_w // 2, main_h // 2 # Fazendo com que a grade 2x2 de pequenas tenha o mesmo tamanho da principal

        # Redimensionar imagens
        main_img = main_img_orig.resize((main_w, main_h), Image.Resampling.LANCZOS)
        small_imgs = [img.resize((small_w, small_h), Image.Resampling.LANCZOS) for img in small_imgs_orig]

        # Largura total: principal + grade de pequenas
        # Altura total: determinada pela imagem principal (ou a maior altura entre principal e grade de pequenas)
        total_width = main_w + (small_w * 2) # Imagem principal + duas colunas de pequenas
        total_height = main_h                 # Altura da imagem principal

        # Alternativa de layout: Principal no topo, 2x2 de pequenas abaixo
        # total_width = main_w 
        # total_height = main_h + (small_h * 2)
        # if main_w < (small_w * 2): # Ajusta largura total se a grade de pequenas for mais larga
        #     total_width = small_w * 2


        mosaic = Image.new('RGB', (total_width, total_height), (255, 255, 255))

        # Colar imagem principal (Layout: principal à esquerda)
        mosaic.paste(main_img, (0, 0))

        # Colar imagens pequenas em grade 2x2 à direita da principal
        mosaic.paste(small_imgs[0], (main_w, 0))
        mosaic.paste(small_imgs[1], (main_w + small_w, 0))
        mosaic.paste(small_imgs[2], (main_w, small_h))
        mosaic.paste(small_imgs[3], (main_w + small_w, small_h))

        # Layout alternativo: Principal no topo, 2x2 abaixo
        # mosaic.paste(main_img, ( (total_width - main_w) // 2 , 0) ) # Centraliza a principal se for mais estreita
        # mosaic.paste(small_imgs[0], ( (total_width - small_w*2)//2 , main_h))
        # mosaic.paste(small_imgs[1], ( (total_width - small_w*2)//2 + small_w, main_h))
        # mosaic.paste(small_imgs[2], ( (total_width - small_w*2)//2, main_h + small_h))
        # mosaic.paste(small_imgs[3], ( (total_width - small_w*2)//2 + small_w, main_h + small_h))


        output_dir = os.path.dirname(output_mosaic_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        mosaic.save(output_mosaic_file)
        print(f"    Mosaico composto salvo em: {output_mosaic_file}")

    except FileNotFoundError as e:
        print(f"    Erro: Arquivo de imagem não encontrado para o mosaico - {e}")
    except Exception as e:
        print(f"    Erro ao criar mosaico composto: {e}")
    finally:
        if 'main_img_orig' in locals() and hasattr(main_img_orig, 'fp') and main_img_orig.fp: main_img_orig.close()
        if 'main_img' in locals() and main_img is not main_img_orig and hasattr(main_img, 'fp') and main_img.fp : main_img.close() # type: ignore
        if 'small_imgs_orig' in locals():
            for img in small_imgs_orig:
                if hasattr(img, 'fp') and img.fp: img.close()
        if 'small_imgs' in locals():
            for i, img in enumerate(small_imgs): # type: ignore
                # Fecha apenas se for uma nova instância (diferente da original)
                is_original = False
                if i < len(small_imgs_orig):
                    if img is small_imgs_orig[i]:
                        is_original = True
                if not is_original and hasattr(img, 'fp') and img.fp : img.close()
        if 'mosaic' in locals() and hasattr(mosaic, 'fp') and mosaic.fp : mosaic.close() # type: ignore


def main_orchestrator(root_path, python_exe, analyzer_script_path):
    """
    Função principal para orquestrar a análise e criação de mosaicos.
    """
    print(f"Iniciando orquestração no diretório raiz: {root_path}")
    
    # Cria diretório de saída para os mosaicos compostos, se não existir
    composite_output_dir_abs = os.path.join(root_path, COMPOSITE_MOSAIC_OUTPUT_FOLDER)
    if not os.path.exists(composite_output_dir_abs):
        os.makedirs(composite_output_dir_abs)
        print(f"Diretório de saída para mosaicos compostos criado: {composite_output_dir_abs}")

    # Ignorar a própria pasta de saída dos mosaicos ao buscar subdiretórios
    folders_to_ignore = {COMPOSITE_MOSAIC_OUTPUT_FOLDER, "global_mosaics_output_paginated", "mosaics_output"} 
    
    target_subdirs = find_target_subdirectories(root_path, folders_to_ignore)

    if not target_subdirs:
        print("Nenhum subdiretório alvo encontrado.")
        return

    for sub_dir_path in target_subdirs:
        sub_dir_name = os.path.basename(sub_dir_path)
        print(f"\nProcessando subdiretório: {sub_dir_name}")

        # --- Etapa 1: Executar rule_diff_analyzer.py ---
        evolution_matrix_img_path = run_rule_analyzer_for_subdir(sub_dir_path, python_exe, analyzer_script_path)
        
        if not evolution_matrix_img_path:
            print(f"  Falha ao gerar Rule Evolution Matrix para {sub_dir_name}. Pulando criação de mosaico composto.")
            continue

        # --- Etapa 2: Coletar imagens e criar mosaico composto ---
        run_1_path = os.path.join(sub_dir_path, "run_1")
        plots_path = os.path.join(run_1_path, "plots")

        # Localizar as imagens
        main_accuracy_img = find_first_matching_file(plots_path, IMAGE_PATTERNS["main_accuracy"])
        heatmap_img = find_first_matching_file(plots_path, IMAGE_PATTERNS["heatmap"])
        attribute_usage_img = find_first_matching_file(plots_path, IMAGE_PATTERNS["attribute_usage"])
        feature_drift_img = find_first_matching_file(run_1_path, IMAGE_PATTERNS["feature_drift"])
        # evolution_matrix_img_path já foi obtido

        # Verificar se todas as imagens foram encontradas
        required_images = {
            "Principal (Accuracy)": main_accuracy_img,
            "Heatmap Regras": heatmap_img,
            "Uso Atributos": attribute_usage_img,
            "Matriz Evolução": evolution_matrix_img_path, # Já é o caminho completo
            "Drift Features": feature_drift_img
        }

        missing_files = [name for name, path in required_images.items() if not path or not os.path.exists(path)]
        
        if missing_files:
            print(f"  Imagens faltando para {sub_dir_name}: {', '.join(missing_files)}. Pulando mosaico composto.")
            for name, path in required_images.items():
                 print(f"    {name}: {'Encontrado' if path and os.path.exists(path) else 'NÃO ENCONTRADO'} ({path})")
            continue

        small_images_list = [
            required_images["Heatmap Regras"],
            required_images["Uso Atributos"],
            required_images["Matriz Evolução"],
            required_images["Drift Features"]
        ]
        
        output_mosaic_filename = os.path.join(composite_output_dir_abs, f"{sub_dir_name}_composite_mosaic.png")
        create_composite_mosaic(main_accuracy_img, small_images_list, output_mosaic_filename) # type: ignore

    print("\nOrquestração concluída.")


def main(root_path_arg, python_exe_arg, analyzer_script_path_arg):
    """
    Função principal para orquestrar a análise e criação de mosaicos.
    """
    print(f"Starting analysis in root directory: {root_path_arg}")
    
    # --- LINHA DE DEBUG ADICIONADA DENTRO DE MAIN ---
    print(f"DEBUG: Value of analyzer_script_path_arg INSIDE main: '{analyzer_script_path_arg}'")
    # --- FIM DA LINHA DE DEBUG ---

    main_output_abs_path = os.path.join(root_path_arg, COMPOSITE_MOSAIC_OUTPUT_FOLDER)
    if not os.path.exists(main_output_abs_path):
        os.makedirs(main_output_abs_path)
        print(f"Main output folder created: {main_output_abs_path}")

    # Ignorar a própria pasta de saída ao buscar subdiretórios
    folders_to_ignore = {COMPOSITE_MOSAIC_OUTPUT_FOLDER, "global_mosaics_output_paginated", "mosaics_output", "global_mosaics_output"} 
    
    target_subdirs = find_target_subdirectories(root_path_arg, folders_to_ignore)

    if not target_subdirs:
        print(f"No target subdirectories found in '{root_path_arg}'.")
        print("Ensure subfolders start with an uppercase letter, contain '_', and are not output folders.")
        return

    print(f"Found {len(target_subdirs)} target experiment folder(s).")

    # ----- PARA TESTE INICIAL COM POUCAS PASTAS (OPCIONAL) -----
    # subdirs_to_process = target_subdirs[:2] # Processa apenas as duas primeiras
    # print(f"\n--- INITIATING TEST RUN: Processing first {len(subdirs_to_process)} experiment folder(s) ---")
    # ----- FIM DO BLOCO DE TESTE -----
    subdirs_to_process = target_subdirs # Para processar todas

    for sub_dir_path in subdirs_to_process:
        sub_dir_name = os.path.basename(sub_dir_path)
        print(f"\nProcessing subdirectory: {sub_dir_name}")

        # --- Etapa 1: Executar rule_diff_analyzer.py ---
        # Passa python_exe_arg e analyzer_script_path_arg
        evolution_matrix_img_path = run_rule_analyzer_for_subdir(sub_dir_path, python_exe_arg, analyzer_script_path_arg)
        
        if not evolution_matrix_img_path:
            print(f"  Failed to generate Rule Evolution Matrix for {sub_dir_name}. Skipping composite mosaic creation.")
            continue

        # --- Etapa 2: Coletar imagens e criar mosaico composto ---
        run_1_path = os.path.join(sub_dir_path, RUN_FOLDER_NAME) # Usando constante global
        plots_path = os.path.join(run_1_path, "plots") # Assumindo que "plots" é o nome da pasta

        # Localizar as imagens
        main_accuracy_img = find_first_matching_file(plots_path, IMAGE_PATTERNS["main_accuracy"])
        heatmap_img = find_first_matching_file(plots_path, IMAGE_PATTERNS["heatmap"])
        attribute_usage_img = find_first_matching_file(plots_path, IMAGE_PATTERNS["attribute_usage"])
        # O nome do arquivo da matriz de evolução agora é baseado no output_base_name_for_analyzer
        # Se output_base_name_for_analyzer = "rule_evolution_analysis", então o arquivo é "rule_evolution_analysis_matrix.png"
        # evolution_matrix_img_path já é o caminho correto retornado por run_rule_analyzer_for_subdir

        feature_drift_img = find_first_matching_file(run_1_path, IMAGE_PATTERNS["feature_drift"])
        
        required_images_dict = {
            "Main Accuracy Plot": main_accuracy_img,
            "Rule Components Heatmap": heatmap_img,
            "Attribute Usage Plot": attribute_usage_img,
            "Rule Evolution Matrix": evolution_matrix_img_path, # Este já é o caminho validado
            "Feature Drift Plot": feature_drift_img
        }

        all_images_found = True
        small_images_path_list = []
        
        # Verifica se a imagem principal existe
        if not main_accuracy_img or not os.path.exists(main_accuracy_img):
            print(f"  Missing MAIN image: Main Accuracy Plot ({IMAGE_PATTERNS['main_accuracy']}) in {plots_path}")
            all_images_found = False
        
        # Verifica as imagens menores
        temp_small_paths = [ # Ordem desejada no mosaico
            required_images_dict["Rule Components Heatmap"],
            required_images_dict["Attribute Usage Plot"],
            required_images_dict["Rule Evolution Matrix"],
            required_images_dict["Feature Drift Plot"]
        ]

        for name, path in [
            ("Rule Components Heatmap", temp_small_paths[0]),
            ("Attribute Usage Plot", temp_small_paths[1]),
            ("Rule Evolution Matrix", temp_small_paths[2]),
            ("Feature Drift Plot", temp_small_paths[3])
        ]:
            if not path or not os.path.exists(path):
                print(f"  Missing SMALL image: {name} (Pattern: {IMAGE_PATTERNS.get(name.lower().replace(' ', '_'), 'N/A')}) expected in its respective folder.")
                all_images_found = False
            else:
                small_images_path_list.append(path)

        if not all_images_found or len(small_images_path_list) != 4:
            print(f"  Not all required images found for {sub_dir_name}. Skipping composite mosaic creation.")
            # Imprime o status de cada imagem para depuração
            for name, path_val in required_images_dict.items():
                status = "Found" if path_val and os.path.exists(path_val) else "NOT FOUND"
                print(f"    - {name}: {status} (Path tried: {path_val})")
            continue
        
        output_mosaic_filename = os.path.join(main_output_abs_path, sub_dir_name, f"{sub_dir_name}_composite_mosaic.png")
        
        # Garante que o diretório específico do experimento para o mosaico exista
        os.makedirs(os.path.join(main_output_abs_path, sub_dir_name), exist_ok=True)

        create_composite_mosaic(main_accuracy_img, small_images_path_list, output_mosaic_filename)

    print("\nOrchestration complete.")

if __name__ == "__main__":
    sns.set_theme(style="whitegrid") # Configura o tema do Seaborn
    
    # --- DEFINA ESTES CAMINHOS ---
    ROOT_PATH_CONFIG = r"G:\Outros computadores\Meu laptop\Downloads\DSL-AG\debug_standard_experiment_results"
    
    # Certifique-se que este é o caminho correto para o seu executável python
    # Pode ser apenas "python" se estiver no PATH, ou o caminho completo.
    PYTHON_EXECUTABLE_CONFIG = "python" 
    # Exemplo Windows: PYTHON_EXECUTABLE_CONFIG = r"C:\Program Files\Python39\python.exe"
    # Exemplo Linux/macOS: PYTHON_EXECUTABLE_CONFIG = "/usr/bin/python3"

    # Caminho completo e absoluto para o script rule_diff_analyzer.py
    RULE_ANALYZER_SCRIPT_PATH_CONFIG = r"G:\Outros computadores\Meu laptop\Downloads\DSL-AG\rule_diff_analyzer.py" # **COLOQUE O CAMINHO CORRETO AQUI**
    # ---------------------------------

    # --- LINHAS DE DEBUG ADICIONADAS ANTES DA VERIFICAÇÃO DE EXISTÊNCIA ---
    print(f"DEBUG: Value of RULE_ANALYZER_SCRIPT_PATH_CONFIG BEFORE check: '{RULE_ANALYZER_SCRIPT_PATH_CONFIG}'")
    print(f"DEBUG: Type of RULE_ANALYZER_SCRIPT_PATH_CONFIG: {type(RULE_ANALYZER_SCRIPT_PATH_CONFIG)}")
    # --- FIM DAS LINHAS DE DEBUG ---

    if not os.path.exists(RULE_ANALYZER_SCRIPT_PATH_CONFIG):
        print(f"CRITICAL ERROR: The script rule_diff_analyzer.py was not found at '{RULE_ANALYZER_SCRIPT_PATH_CONFIG}'.") # Imprime o valor que ele usou
        print("Please configure the 'RULE_ANALYZER_SCRIPT_PATH_CONFIG' variable correctly.")
    elif not os.path.isdir(ROOT_PATH_CONFIG):
        print(f"CRITICAL ERROR: The root directory '{ROOT_PATH_CONFIG}' does not exist.")
        print("Please configure the 'ROOT_PATH_CONFIG' variable correctly.")
    else:
        # Passando as configurações para a função main
        main(ROOT_PATH_CONFIG, PYTHON_EXECUTABLE_CONFIG, RULE_ANALYZER_SCRIPT_PATH_CONFIG)