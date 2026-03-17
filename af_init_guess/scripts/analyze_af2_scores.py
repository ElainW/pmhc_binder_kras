#!/usr/bin/env python3
"""
Merge and analyze AF2 initial guess score files.

Usage:
    python analyze_af2_scores.py \
        --score_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/af2_kras/ \
        --output_csv /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/af2_kras_all_scores.csv \
        --passing_csv /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/af2_kras_passing.csv \
        --pae_cutoff 10.0 \
        --plddt_cutoff 70.0 \
        --rmsd_cutoff 2.0
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster use
import matplotlib.pyplot as plt


# --- Score file parsing ---

def parse_score_file(score_file):
    """Parse a Rosetta-format .sc score file into a DataFrame."""
    with open(score_file) as f:
        lines = f.readlines()

    header = None
    data_lines = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == 'SCORE:' and 'description' in parts:
            header = parts[1:]
        elif parts[0] == 'SCORE:' and header is not None:
            data_lines.append(parts[1:])

    if header is None or not data_lines:
        return pd.DataFrame()

    df = pd.DataFrame(data_lines, columns=header)

    for col in df.columns:
        if col != 'description':
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

    return df


def merge_score_files(score_dir):
    """Merge all .sc files in a directory into a single DataFrame."""
    score_files = sorted(glob.glob(os.path.join(score_dir, '*.sc')))

    if not score_files:
        raise FileNotFoundError(f"No .sc files found in {score_dir}")

    print(f"Found {len(score_files)} score files in {score_dir}")

    dfs = []
    for sf in score_files:
        df = parse_score_file(sf)
        if not df.empty:
            dfs.append(df)
        else:
            print(f"  Warning: empty or unparseable — {os.path.basename(sf)}")

    if not dfs:
        raise ValueError("No valid score data found across all .sc files")

    merged = pd.concat(dfs, ignore_index=True)
    print(f"Total designs scored: {len(merged)}")
    return merged


# --- Analysis and filtering ---

def analyze_and_filter(df, args):
    """Print score statistics, plot distributions, apply filters."""

    key_cols = ['pae_interaction', 'binder_plddt', 'binder_rmsd']
    available = [c for c in key_cols if c in df.columns]

    if not available:
        raise ValueError(
            f"None of the expected score columns found. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"\n{'='*55}")
    print(f"Score statistics (n={len(df)})")
    print(f"{'='*55}")
    print(df[available].describe().round(3).to_string())

    # --- Passage rates at different thresholds ---
    print(f"\n{'='*55}")
    print(f"Passage rates")
    print(f"{'='*55}")

    if 'pae_interaction' in df.columns:
        for t in [5, 7, 10]:
            n = (df['pae_interaction'] < t).sum()
            print(f"  pae_in