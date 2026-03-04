#!/usr/bin/env python3
"""
Generate stream_definitions.json from batch config YAML files.

Parses configs/config_batch_*.yaml to extract experimental_streams
and builds a {dataset_name: stream_definition} lookup dictionary.

Output: paper_data/stream_definitions.json

Author: Automated Analysis
Date: 2026-02-25
"""

import os
import sys
import json
import logging
from pathlib import Path
from glob import glob

import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIGS_DIR = Path("configs")
OUTPUT_DIR = Path("paper_data")
OUTPUT_FILE = OUTPUT_DIR / "stream_definitions.json"

# Config file patterns to scan (ordered by priority: batch_N first)
CONFIG_PATTERNS = [
    "config_batch_[0-9]*.yaml",
    "config_balanced_batch_*.yaml",
    "config_chunk2000_batch_*.yaml",
]


def extract_stream_definitions():
    """Extract all stream definitions from batch config files."""
    all_definitions = {}
    files_processed = 0

    # Collect all config files matching patterns
    config_files = set()
    for pattern in CONFIG_PATTERNS:
        matched = sorted(CONFIGS_DIR.glob(pattern))
        for f in matched:
            # Skip backup files
            if '.backup' in f.name:
                continue
            config_files.add(f)

    config_files = sorted(config_files)
    logger.info(f"Found {len(config_files)} config files to scan")

    for config_path in config_files:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to parse {config_path}: {e}")
            continue

        if not config or 'experimental_streams' not in config:
            continue

        streams = config['experimental_streams']
        files_processed += 1
        new_count = 0

        for stream_name, stream_def in streams.items():
            if stream_name in all_definitions:
                continue  # already captured from an earlier file

            # Build the definition dict with relevant fields
            definition = {
                'dataset_type': stream_def.get('dataset_type', ''),
            }

            # Only include drift-related fields if present
            if 'drift_type' in stream_def:
                definition['drift_type'] = stream_def['drift_type']
            if 'gradual_drift_width_chunks' in stream_def:
                definition['gradual_drift_width_chunks'] = stream_def['gradual_drift_width_chunks']
            if 'concept_sequence' in stream_def:
                definition['concept_sequence'] = stream_def['concept_sequence']
            if 'noise_config' in stream_def:
                definition['noise_config'] = stream_def['noise_config']
            if 'params_override' in stream_def:
                definition['params_override'] = stream_def['params_override']

            all_definitions[stream_name] = definition
            new_count += 1

        if new_count > 0:
            logger.info(f"  {config_path.name}: +{new_count} new streams (total: {len(all_definitions)})")

    logger.info(f"Processed {files_processed} config files, extracted {len(all_definitions)} stream definitions")
    return all_definitions


def main():
    logger.info("=" * 60)
    logger.info("GENERATING STREAM DEFINITIONS LOOKUP")
    logger.info("=" * 60)

    definitions = extract_stream_definitions()

    if not definitions:
        logger.error("No stream definitions found. Check configs/ directory.")
        return 1

    # Separate into categories for summary
    with_drift = [k for k, v in definitions.items() if 'concept_sequence' in v]
    stationary = [k for k, v in definitions.items()
                  if 'concept_sequence' not in v and not v.get('dataset_type', '').startswith(('ELECTRICITY', 'SHUTTLE', 'COVERTYPE', 'POKER', 'INTEL', 'ASSETNEG'))]
    real_world = [k for k, v in definitions.items()
                  if v.get('dataset_type', '') in ('ELECTRICITY', 'SHUTTLE', 'COVERTYPE', 'POKER', 'INTELLABSENSORS', 'ASSETNEGOTIATION')]

    logger.info(f"\nSummary:")
    logger.info(f"  With concept drift: {len(with_drift)}")
    logger.info(f"  Stationary: {len(stationary)}")
    logger.info(f"  Real-world / other: {len(real_world)}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(definitions, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved: {OUTPUT_FILE} ({len(definitions)} definitions)")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
