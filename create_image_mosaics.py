# create_image_mosaics_paginated.py

import os
from PIL import Image
import math # Adicionado para math.ceil

def find_target_subdirectories(root_path, folders_to_ignore):
    """
    Encontra subdiretórios no root_path que contêm '_' no nome,
    ignorando pastas especificadas.
    Retorna uma lista de caminhos completos para esses subdiretórios.
    """
    target_dirs = []
    if not os.path.isdir(root_path):
        print(f"Erro: O caminho raiz '{root_path}' não é um diretório válido.")
        return target_dirs

    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        if os.path.isdir(item_path) and '_' in item and item not in folders_to_ignore:
            target_dirs.append(item_path)
    return target_dirs

def find_images_by_prefix(folder_path, prefix, debug_prefix=None):
    """
    Encontra arquivos de imagem em folder_path que começam com o prefixo especificado.
    Retorna uma lista ordenada de caminhos de imagem.
    """
    images = []
    if not os.path.isdir(folder_path):
        if debug_prefix:
            print(f"    DEBUG ({debug_prefix}): Pasta de origem '{folder_path}' não é um diretório ou não existe.")
        return images

    if debug_prefix: # Mantido para depuração, pode ser removido ou simplificado
        # print(f"    DEBUG ({debug_prefix}): Listando conteúdo de '{folder_path}':")
        # try:
        #     all_files_in_folder = os.listdir(folder_path)
        #     if not all_files_in_folder:
        #         print(f"      -> Pasta '{folder_path}' está vazia.")
        # except Exception as e:
        #     print(f"      Erro ao listar arquivos em '{folder_path}': {e}")
        pass


    for fname in sorted(os.listdir(folder_path)):
        if fname.startswith(prefix) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            if debug_prefix: # Mantido para depuração
                # print(f"      DEBUG ({debug_prefix}): ENCONTRADO! Arquivo '{fname}' corresponde ao prefixo '{prefix}'.")
                pass
            images.append(os.path.join(folder_path, fname))

    # if debug_prefix and not images: # Mantido para depuração
        # print(f"    DEBUG ({debug_prefix}): Nenhuma imagem correspondente ao prefixo '{prefix}' encontrada em '{folder_path}' após varredura.")
    return images

def create_mosaic(image_paths_chunk, mosaic_dim, output_filename): # Renomeado image_paths para image_paths_chunk
    """
    Cria um mosaico a partir de uma lista (chunk) de caminhos de imagem com as dimensões especificadas.
    Salva o mosaico no output_filename.
    """
    if not image_paths_chunk: # Verificando o chunk
        print(f"Nenhum chunk de imagem fornecido para criar o mosaico {output_filename}.")
        return

    # Interpretação das dimensões do mosaico
    if isinstance(mosaic_dim, int): # (N) -> 1 linha, N colunas
        rows = 1
        cols = mosaic_dim
    elif len(mosaic_dim) == 1: # (N,) -> 1 linha, N colunas
        rows = 1
        cols = mosaic_dim[0]
    elif len(mosaic_dim) == 2: # (L, C) -> L linhas, C colunas
        rows, cols = mosaic_dim
    else:
        print(f"Erro: Dimensão do mosaico inválida {mosaic_dim} para {output_filename}. Use (linhas, colunas) ou (N,).")
        return

    if rows <= 0 or cols <= 0:
        print(f"Erro: Dimensões do mosaico devem ser positivas: {mosaic_dim} para {output_filename}")
        return

    # A função agora espera que image_paths_chunk seja o conjunto exato de imagens para este mosaico específico.
    # O número de slots é apenas para cálculo do layout da grade.
    num_slots_in_mosaic_layout = rows * cols
    
    # O número de imagens no chunk pode ser menor que num_slots_in_mosaic_layout (para o último mosaico)
    # A lógica abaixo já lida com isso, pois itera sobre image_paths_chunk.

    print(f"Tentando criar mosaico {output_filename} com {len(image_paths_chunk)} imagens em uma grade {rows}x{cols}.")
    if len(image_paths_chunk) < num_slots_in_mosaic_layout:
        print(f"  Aviso: Este mosaico terá {num_slots_in_mosaic_layout - len(image_paths_chunk)} espaços vazios (último chunk).")

    pil_images_opened = []
    processed_pil_images = []
    
    try:
        for path in image_paths_chunk: # Usando o chunk diretamente
            pil_images_opened.append(Image.open(path))

        if not pil_images_opened:
            return

        ref_width, ref_height = pil_images_opened[0].size
        if ref_width == 0 or ref_height == 0:
            print(f"Erro: A primeira imagem {image_paths_chunk[0]} tem dimensões inválidas ({ref_width}x{ref_height}).")
            return

        for i, img_pil in enumerate(pil_images_opened):
            if img_pil.size != (ref_width, ref_height):
                try:
                    resized_img = img_pil.resize((ref_width, ref_height), Image.Resampling.LANCZOS)
                    processed_pil_images.append(resized_img)
                except Exception as e:
                    print(f"  Erro ao redimensionar {image_paths_chunk[i]}: {e}. Pulando esta imagem.")
                    continue
            else:
                processed_pil_images.append(img_pil)

        if not processed_pil_images:
            print(f"Nenhuma imagem válida restante após tentativa de redimensionamento para {output_filename}.")
            return

        cell_width, cell_height = ref_width, ref_height
        mosaic_total_width = cols * cell_width
        mosaic_total_height = rows * cell_height

        if mosaic_total_width == 0 or mosaic_total_height == 0:
            print(f"Erro: Dimensões calculadas para o mosaico {output_filename} são zero.")
            return

        mosaic_image = Image.new('RGB', (mosaic_total_width, mosaic_total_height), (255, 255, 255))

        for i, img_to_paste in enumerate(processed_pil_images):
            current_row = i // cols
            current_col = i % cols
            x_offset = current_col * cell_width
            y_offset = current_row * cell_height
            mosaic_image.paste(img_to_paste, (x_offset, y_offset))

        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"Erro ao criar diretório de saída {output_dir}: {e}")
                return

        mosaic_image.save(output_filename)
        print(f"Mosaico salvo em: {output_filename}")

    except Exception as e:
        print(f"Erro durante a criação ou salvamento do mosaico {output_filename}: {e}")
    finally:
        for img in pil_images_opened:
            if hasattr(img, 'fp') and img.fp:
                try: img.close()
                except Exception: pass
        
        for proc_img in processed_pil_images:
            is_original_flag = False
            for orig_img in pil_images_opened:
                if proc_img is orig_img:
                    is_original_flag = True
                    break
            if not is_original_flag:
                if hasattr(proc_img, 'fp') and proc_img.fp:
                    try: proc_img.close()
                    except Exception: pass
        
        if 'mosaic_image' in locals() and hasattr(mosaic_image, 'fp') and mosaic_image.fp:
            try: mosaic_image.close()
            except Exception: pass


def process_for_global_mosaics(root_path, mosaic_layouts, output_base_folder="global_mosaics_output_paginated"):
    """
    Coleta imagens de todos os subdiretórios e cria mosaicos globais para cada prefixo.
    Se houver mais imagens do que o layout comporta, cria múltiplos mosaicos (paginação).
    """
    folders_to_ignore_set = {output_base_folder, "mosaics_output", "global_mosaics_output"}

    base_output_path = os.path.join(root_path, output_base_folder)
    if not os.path.exists(base_output_path):
        try:
            os.makedirs(base_output_path)
            print(f"Criado diretório base para mosaicos globais: {base_output_path}")
        except OSError as e:
            print(f"Erro ao criar diretório base de saída {base_output_path}: {e}")
            return

    subdirectories = find_target_subdirectories(root_path, folders_to_ignore_set)
    if not subdirectories:
        print(f"Nenhum subdiretório com '_' (e que não seja de output) encontrado em '{root_path}'.")
        return

    all_images_by_prefix = {prefix: [] for prefix in mosaic_layouts.keys()}

    print("\n--- Fase 1: Coletando todos os arquivos de imagem ---")
    for sub_dir_path in subdirectories:
        sub_dir_name = os.path.basename(sub_dir_path)
        print(f"  Analisando subdiretório: {sub_dir_name}")
        run1_path = os.path.join(sub_dir_path, "run_1")

        for prefix_key in mosaic_layouts.keys():
            current_images_found = []
            source_folder_for_prefix = ""
            # debug_this_prefix = None # Descomente para depuração específica
            
            if prefix_key == "FeatureDrift_":
                source_folder_for_prefix = run1_path # Corrigido na v4 para buscar em run_1
                # debug_this_prefix = "FeatureDrift"
                if os.path.isdir(source_folder_for_prefix):
                    current_images_found = find_images_by_prefix(source_folder_for_prefix, prefix_key, debug_prefix=None) # Removido debug_prefix daqui para não poluir
                # else: # Log já é feito por find_images_by_prefix se debug_prefix for passado
                    # print(f"    DEBUG (FeatureDrift): Pasta '{source_folder_for_prefix}' não encontrada.")
            else:
                plot_path = os.path.join(run1_path, "plots")
                source_folder_for_prefix = plot_path
                if not os.path.isdir(plot_path):
                    continue
                current_images_found = find_images_by_prefix(plot_path, prefix_key, debug_prefix=None)
            
            if current_images_found:
                # print(f"    -> Encontradas {len(current_images_found)} imagens para '{prefix_key}' em '{source_folder_for_prefix}'")
                all_images_by_prefix[prefix_key].extend(current_images_found)

    print("\n--- Fase 2: Criando Mosaicos Globais Paginados ---")
    for prefix_key, dims in mosaic_layouts.items():
        images_for_this_prefix = all_images_by_prefix[prefix_key]
        
        if not images_for_this_prefix:
            print(f"Nenhuma imagem coletada globalmente para o prefixo '{prefix_key}'. Pulando mosaico.")
            continue
            
        total_images_for_prefix = len(images_for_this_prefix)
        print(f"\nTotal de {total_images_for_prefix} imagens coletadas globalmente para o prefixo '{prefix_key}'.")

        # Determina slots por mosaico a partir das dimensões
        if isinstance(dims, int):
            rows, cols = 1, dims
        elif len(dims) == 1:
            rows, cols = 1, dims[0]
        else:
            rows, cols = dims
        
        slots_per_mosaic = rows * cols
        if slots_per_mosaic <= 0:
            print(f"  Erro: Layout inválido {dims} (slots <=0) para prefixo '{prefix_key}'. Pulando.")
            continue

        num_mosaics_to_create = math.ceil(total_images_for_prefix / slots_per_mosaic)
        print(f"  Layout do mosaico: {rows}x{cols} ({slots_per_mosaic} imagens por mosaico).")
        print(f"  Serão criados {num_mosaics_to_create} mosaico(s) para este prefixo.")

        # Define o diretório de saída para este tipo de mosaico
        clean_prefix_for_folder = prefix_key.replace("Plot_", "").strip('_')
        mosaic_type_folder_name = f"{clean_prefix_for_folder}_global_mosaics"
        specific_output_dir_for_mosaic_type = os.path.join(base_output_path, mosaic_type_folder_name)
        # O diretório será criado dentro de create_mosaic se não existir

        for i in range(num_mosaics_to_create):
            start_index = i * slots_per_mosaic
            end_index = start_index + slots_per_mosaic
            current_image_chunk = images_for_this_prefix[start_index:end_index]

            if not current_image_chunk: # Segurança, não deve acontecer se num_mosaics_to_create > 0
                continue

            # Define o nome do arquivo de mosaico, incluindo o número da parte se houver mais de um
            base_mosaic_filename = f"GLOBAL_{prefix_key.strip('_')}_mosaic"
            if num_mosaics_to_create > 1:
                output_mosaic_filename_final = os.path.join(specific_output_dir_for_mosaic_type, f"{base_mosaic_filename}_part{i+1}.png")
            else:
                output_mosaic_filename_final = os.path.join(specific_output_dir_for_mosaic_type, f"{base_mosaic_filename}.png")
            
            create_mosaic(current_image_chunk, dims, output_mosaic_filename_final)


if __name__ == "__main__":
    raiz = r"C:\Users\EAI.001\Downloads\DSL-AG\drift_experiment_results_big"
    
    # ATENÇÃO: As dimensões em layouts agora definem o tamanho de CADA PÁGINA do mosaico.
    # O script calculará quantas páginas são necessárias.
    layouts = {
        "Plot_AccuracyPeriodic_": (2, 3),      # Cada mosaico de AccuracyPeriodic terá 2x2 = 4 imagens.
        "FeatureDrift_": (2, 3),               # Cada mosaico de FeatureDrift terá 1x3 = 3 imagens.
        "Plot_RuleComponents_Heatmap_": (2, 3) # Cada mosaico de RuleComponents terá 2x3 = 6 imagens.
    }
    # Exemplo: Se houver 10 imagens "Plot_AccuracyPeriodic_", com layout (2,2):
    # - mosaico_part1.png terá 4 imagens
    # - mosaico_part2.png terá 4 imagens
    # - mosaico_part3.png terá 2 imagens (e 2 espaços vazios)
    
    if not os.path.isdir(raiz):
        print(f"ERRO CRÍTICO: O diretório raiz '{raiz}' não existe. Por favor, configure corretamente.")
    else:
        print(f"Iniciando processamento GLOBAL PAGINADO no diretório raiz: {raiz}")
        process_for_global_mosaics(raiz, layouts)
        print("\nProcessamento GLOBAL PAGINADO concluído.")