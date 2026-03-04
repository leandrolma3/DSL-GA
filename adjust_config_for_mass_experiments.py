#!/usr/bin/env python3
"""
adjust_config_for_mass_experiments.py

Ajusta config.yaml para experimentos massivos com 6 chunks:
- Reduz population_size de 120 → 80
- Configura num_chunks = 6, max_instances = 36000
- Rescala concept_sequence de todas as 41 drift simulations proporcionalmente
- Ajusta gradual_drift_width_chunks proporcionalmente
- Gera config_6chunks.yaml para uso nos experimentos
- Gera lista categorizada de experimentos

Autor: Claude Code
Data: 2025-10-28
"""

import yaml
import math
import os
from typing import Dict, List, Any
from collections import defaultdict

TARGET_CHUNKS = 6
TARGET_POPULATION = 80
TARGET_MAX_INSTANCES = 36000  # 6 chunks × 6000 instances


def calculate_rescale_ratio(original_total_chunks: int, target_chunks: int = TARGET_CHUNKS) -> float:
    """Calcula a razão de rescaling para ajustar duration_chunks."""
    if original_total_chunks <= 0:
        raise ValueError(f"Original total chunks inválido: {original_total_chunks}")
    return target_chunks / original_total_chunks


def rescale_concept_sequence(
    concept_sequence: List[Dict[str, Any]],
    target_total_chunks: int = TARGET_CHUNKS
) -> List[Dict[str, Any]]:
    """
    Rescala concept_sequence para somar exatamente target_total_chunks.

    Estratégia:
    1. Calcula total original
    2. Aplica scaling proporcional
    3. Arredonda para inteiros
    4. Ajusta resíduo para somar exatamente target
    """
    if not concept_sequence:
        raise ValueError("concept_sequence vazio")

    # Calcula total original
    original_total = sum(stage.get('duration_chunks', 0) for stage in concept_sequence)

    if original_total <= 0:
        raise ValueError(f"Total original de chunks inválido: {original_total}")

    # Calcula durations escalados (float)
    scale_ratio = target_total_chunks / original_total
    scaled_durations_float = [
        stage.get('duration_chunks', 0) * scale_ratio
        for stage in concept_sequence
    ]

    # Arredonda para inteiros (floor primeiro)
    scaled_durations_int = [math.floor(d) for d in scaled_durations_float]

    # Calcula resíduo
    current_sum = sum(scaled_durations_int)
    residual = target_total_chunks - current_sum

    # Distribui resíduo baseado nas partes fracionárias (maior fração recebe primeiro)
    if residual != 0:
        fractional_parts = [
            (scaled_durations_float[i] - scaled_durations_int[i], i)
            for i in range(len(scaled_durations_float))
        ]
        fractional_parts.sort(reverse=True)  # Maior fração primeiro

        for i in range(abs(residual)):
            if i < len(fractional_parts):
                idx = fractional_parts[i][1]
                scaled_durations_int[idx] += 1 if residual > 0 else -1

    # Garante mínimo de 1 chunk por conceito (exceto blips que podem ser 2)
    # Para blips, manter mínimo de 2 chunks se estava >= 1 no original
    for i, stage in enumerate(concept_sequence):
        if scaled_durations_int[i] < 1:
            scaled_durations_int[i] = 1

    # Reajusta se ultrapassamos o target após garantir mínimos
    current_sum = sum(scaled_durations_int)
    if current_sum > target_total_chunks:
        # Remove do maior
        max_idx = scaled_durations_int.index(max(scaled_durations_int))
        scaled_durations_int[max_idx] -= (current_sum - target_total_chunks)

    # Cria novo concept_sequence
    new_sequence = []
    for i, stage in enumerate(concept_sequence):
        new_stage = stage.copy()
        new_stage['duration_chunks'] = scaled_durations_int[i]
        new_sequence.append(new_stage)

    # Validação final
    final_sum = sum(s['duration_chunks'] for s in new_sequence)
    if final_sum != target_total_chunks:
        raise ValueError(
            f"Rescaling falhou: soma={final_sum}, esperado={target_total_chunks}. "
            f"Original: {[s.get('duration_chunks') for s in concept_sequence]} → "
            f"Novo: {[s['duration_chunks'] for s in new_sequence]}"
        )

    return new_sequence


def adjust_gradual_width(
    gradual_width: int,
    original_total_chunks: int,
    target_total_chunks: int = TARGET_CHUNKS
) -> int:
    """
    Ajusta gradual_drift_width_chunks proporcionalmente.
    Garante que width < duration do primeiro conceito após drift.
    """
    if gradual_width <= 0:
        return 0

    # Scaling proporcional
    scale_ratio = target_total_chunks / original_total_chunks
    new_width = max(1, round(gradual_width * scale_ratio))

    # Limita a no máximo metade do target (para não ser maior que conceitos)
    new_width = min(new_width, target_total_chunks // 2)

    return new_width


def categorize_stream(stream_name: str, stream_config: Dict[str, Any]) -> str:
    """Categoriza stream por tipo de drift."""
    if 'concept_sequence' not in stream_config:
        # Real dataset ou stationary
        dataset_type = stream_config.get('dataset_type', 'UNKNOWN')
        if dataset_type in ['ELECTRICITY', 'SHUTTLE', 'COVERTYPE', 'POKER', 'INTELLABSENSORS']:
            return 'Real Datasets'
        else:
            return 'Stationary'

    drift_type = stream_config.get('drift_type', 'abrupt')
    has_noise = 'noise_config' in stream_config and stream_config['noise_config'].get('enabled', False)
    concept_sequence = stream_config['concept_sequence']

    # Detecta recurring (conceito repetido)
    concept_ids = [stage['concept_id'] for stage in concept_sequence]
    is_recurring = len(concept_ids) != len(set(concept_ids))

    # Detecta blip (conceito curto no meio)
    is_blip = False
    if len(concept_sequence) == 3:
        durations = [stage.get('duration_chunks', 0) for stage in concept_sequence]
        if durations[1] <= 2 and durations[0] > 2 and durations[2] > 2:
            is_blip = True

    # Categorização
    if has_noise:
        return 'Noise'
    elif is_blip:
        return 'Blips'
    elif is_recurring:
        return 'Recurring'
    elif drift_type == 'gradual':
        return 'Gradual'
    elif drift_type == 'abrupt':
        return 'Abrupt'
    else:
        return 'Mixed'


def adjust_stream_definition(
    stream_name: str,
    stream_config: Dict[str, Any],
    target_chunks: int = TARGET_CHUNKS
) -> Dict[str, Any]:
    """Ajusta definição de um stream para target_chunks."""

    # Se não tem concept_sequence, não precisa ajustar (real dataset ou stationary)
    if 'concept_sequence' not in stream_config:
        return stream_config.copy()

    adjusted_config = stream_config.copy()

    # Calcula total original
    concept_sequence = stream_config['concept_sequence']
    original_total = sum(stage.get('duration_chunks', 0) for stage in concept_sequence)

    # Rescala concept_sequence
    try:
        new_sequence = rescale_concept_sequence(concept_sequence, target_chunks)
        adjusted_config['concept_sequence'] = new_sequence
    except Exception as e:
        print(f"  ⚠️  ERRO ao rescalar {stream_name}: {e}")
        print(f"      Original: {concept_sequence}")
        raise

    # Ajusta gradual_drift_width_chunks se presente
    if 'gradual_drift_width_chunks' in stream_config:
        original_width = stream_config['gradual_drift_width_chunks']
        new_width = adjust_gradual_width(original_width, original_total, target_chunks)
        adjusted_config['gradual_drift_width_chunks'] = new_width

        # Validação: width deve ser < duration do conceito seguinte
        if new_width > 0 and len(new_sequence) > 1:
            second_concept_duration = new_sequence[1]['duration_chunks']
            if new_width >= second_concept_duration:
                new_width = max(1, second_concept_duration - 1)
                adjusted_config['gradual_drift_width_chunks'] = new_width
                print(f"  ⚙️  {stream_name}: Ajustado gradual_width para {new_width} "
                      f"(< {second_concept_duration})")

    return adjusted_config


def adjust_global_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ajusta parâmetros globais do config."""
    adjusted_config = config.copy()

    # Ajusta data_params
    if 'data_params' not in adjusted_config:
        adjusted_config['data_params'] = {}

    adjusted_config['data_params']['num_chunks'] = TARGET_CHUNKS
    adjusted_config['data_params']['max_instances'] = TARGET_MAX_INSTANCES

    # Ajusta ga_params
    if 'ga_params' not in adjusted_config:
        adjusted_config['ga_params'] = {}

    adjusted_config['ga_params']['population_size'] = TARGET_POPULATION

    return adjusted_config


def generate_experiment_list(
    experimental_streams: Dict[str, Any],
    output_path: str = "EXPERIMENT_LIST_6CHUNKS.md"
) -> None:
    """Gera lista categorizada de experimentos."""

    categories = defaultdict(list)

    for stream_name, stream_config in experimental_streams.items():
        category = categorize_stream(stream_name, stream_config)

        # Informações do stream
        info = {
            'name': stream_name,
            'dataset_type': stream_config.get('dataset_type', 'N/A'),
            'drift_type': stream_config.get('drift_type', 'N/A'),
        }

        if 'concept_sequence' in stream_config:
            seq = stream_config['concept_sequence']
            info['num_concepts'] = len(set(stage['concept_id'] for stage in seq))
            info['total_chunks'] = sum(stage['duration_chunks'] for stage in seq)
            info['sequence'] = ' → '.join([
                f"{stage['concept_id']}({stage['duration_chunks']})"
                for stage in seq
            ])
        else:
            info['num_concepts'] = 1
            info['total_chunks'] = TARGET_CHUNKS
            info['sequence'] = 'Stationary/Real'

        categories[category].append(info)

    # Gera markdown
    lines = [
        "# LISTA DE EXPERIMENTOS - 6 CHUNKS",
        "",
        f"**Total de streams**: {len(experimental_streams)}",
        f"**Configuracao**: {TARGET_CHUNKS} chunks x 6000 instances = {TARGET_MAX_INSTANCES} instances",
        f"**Populacao**: {TARGET_POPULATION} individuos",
        "",
        "---",
        ""
    ]

    # Ordena categorias
    category_order = ['Abrupt', 'Gradual', 'Recurring', 'Blips', 'Noise', 'Mixed', 'Stationary', 'Real Datasets']

    for category in category_order:
        if category not in categories:
            continue

        streams = categories[category]
        lines.append(f"## Categoria: {category}")
        lines.append(f"**Total**: {len(streams)} streams")
        lines.append("")

        # Tabela
        lines.append("| # | Stream Name | Dataset Type | Drift Type | Concepts | Sequence |")
        lines.append("|---|-------------|--------------|------------|----------|----------|")

        for i, info in enumerate(sorted(streams, key=lambda x: x['name']), 1):
            lines.append(
                f"| {i} | `{info['name']}` | {info['dataset_type']} | "
                f"{info['drift_type']} | {info['num_concepts']} | "
                f"{info['sequence']} |"
            )

        lines.append("")
        lines.append("---")
        lines.append("")

    # Estatísticas
    lines.append("## Estatisticas")
    lines.append("")
    total_streams = len(experimental_streams)
    drift_simulations = sum(
        1 for s in experimental_streams.values()
        if 'concept_sequence' in s
    )

    lines.append(f"- **Total de streams**: {total_streams}")
    lines.append(f"- **Drift simulations**: {drift_simulations}")
    lines.append(f"- **Real/Stationary**: {total_streams - drift_simulations}")
    lines.append("")

    for category in category_order:
        if category in categories:
            count = len(categories[category])
            percentage = (count / total_streams) * 100
            lines.append(f"- **{category}**: {count} ({percentage:.1f}%)")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"**Gerado por**: adjust_config_for_mass_experiments.py")
    lines.append(f"**Data**: 2025-10-28")

    # Salva arquivo
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n[OK] Lista de experimentos salva em: {output_path}")


def main():
    """Função principal."""
    print("=" * 70)
    print("AJUSTE DE CONFIG PARA EXPERIMENTOS MASSIVOS - 6 CHUNKS")
    print("=" * 70)
    print()

    # Carrega config original
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"ERRO: {config_path} não encontrado!")
        return

    print(f"[*] Carregando {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"[OK] Config carregado com sucesso")
    print()

    # Ajusta parâmetros globais
    print("[*] Ajustando parâmetros globais...")
    adjusted_config = adjust_global_params(config)
    print(f"   - num_chunks: {config.get('data_params', {}).get('num_chunks', 'N/A')} -> {TARGET_CHUNKS}")
    print(f"   - population_size: {config.get('ga_params', {}).get('population_size', 'N/A')} -> {TARGET_POPULATION}")
    print(f"   - max_instances: {config.get('data_params', {}).get('max_instances', 'N/A')} -> {TARGET_MAX_INSTANCES}")
    print()

    # Processa experimental_streams
    if 'experimental_streams' not in config:
        print("ERRO: 'experimental_streams' não encontrado no config!")
        return

    experimental_streams = config['experimental_streams']
    print(f"[*] Encontrados {len(experimental_streams)} streams no config")
    print()

    # Identifica drift simulations
    drift_simulations = {
        name: cfg for name, cfg in experimental_streams.items()
        if 'concept_sequence' in cfg
    }

    print(f"[*] Identificados {len(drift_simulations)} drift simulations para ajustar")
    print()

    # Ajusta cada drift simulation
    print("[*] Ajustando drift simulations...")
    print()

    adjusted_streams = {}
    errors = []

    for stream_name, stream_config in experimental_streams.items():
        if 'concept_sequence' not in stream_config:
            # Copia sem alterações (real dataset ou stationary)
            adjusted_streams[stream_name] = stream_config.copy()
            continue

        try:
            # Informações originais
            original_seq = stream_config['concept_sequence']
            original_total = sum(s.get('duration_chunks', 0) for s in original_seq)
            original_width = stream_config.get('gradual_drift_width_chunks', 0)

            # Ajusta
            adjusted_stream = adjust_stream_definition(stream_name, stream_config, TARGET_CHUNKS)
            adjusted_streams[stream_name] = adjusted_stream

            # Informações ajustadas
            new_seq = adjusted_stream['concept_sequence']
            new_total = sum(s['duration_chunks'] for s in new_seq)
            new_width = adjusted_stream.get('gradual_drift_width_chunks', 0)

            # Exibe mudança
            original_str = '-'.join(str(s.get('duration_chunks')) for s in original_seq)
            new_str = '-'.join(str(s['duration_chunks']) for s in new_seq)

            print(f"  [OK] {stream_name}",  flush=True)
            print(f"     Chunks: {original_str} (Sum={original_total}) => {new_str} (Sum={new_total})", flush=True)
            if original_width > 0:
                print(f"     Gradual width: {original_width} => {new_width}", flush=True)
            print(flush=True)

        except Exception as e:
            errors.append((stream_name, str(e)))
            print(f"  [ERROR] em {stream_name}: {e}")
            print()

    # Atualiza config
    adjusted_config['experimental_streams'] = adjusted_streams

    # Salva config ajustado
    output_path = 'config_6chunks.yaml'
    print(f"[*] Salvando config ajustado em: {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(adjusted_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"[OK] Config ajustado salvo com sucesso!")
    print()

    # Gera lista de experimentos
    print("[*] Gerando lista categorizada de experimentos...")
    generate_experiment_list(adjusted_streams)

    # Resumo final
    print()
    print("=" * 70)
    print("RESUMO")
    print("=" * 70)
    print(f"[OK] Streams processados: {len(experimental_streams)}")
    print(f"[OK] Drift simulations ajustados: {len(drift_simulations)}")
    print(f"[OK] Real/Stationary mantidos: {len(experimental_streams) - len(drift_simulations)}")

    if errors:
        print(f"[WARNING] Erros encontrados: {len(errors)}")
        for stream_name, error in errors:
            print(f"   - {stream_name}: {error}")
    else:
        print(f"[OK] Sem erros!")

    print()
    print(f"Arquivos gerados:")
    print(f"   - {output_path}")
    print(f"   - EXPERIMENT_LIST_6CHUNKS.md")
    print()
    print("Proximo passo: Validar config_6chunks.yaml e executar experimentos!")
    print("=" * 70)


if __name__ == '__main__':
    main()
