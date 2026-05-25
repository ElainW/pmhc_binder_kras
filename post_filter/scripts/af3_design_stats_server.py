"""
af3_design_stats_server.py
==========================
Wrapper around af3_design_stats.py that overrides AF3 file discovery
functions to match the AF3 server output directory structure:

  {af3_out_dir}/{design_sanitized}/
    fold_{design_sanitized}_model_0.cif
    fold_{design_sanitized}_summary_confidences_0.json
    fold_{design_sanitized}_full_data_0.json

The AF3 server sanitizes design names: replaces '.' with '' and '__' with '_'.
This wrapper handles the name mapping transparently.

Usage:
    python af3_design_stats_server.py \
        --af3_out_dir     $DIR/outputs/r2.5/af3_nomsa/ \
        --relaxed_pdb_dir $DIR/outputs/r2.5/fastrelax_af3_nomsa/relaxed_pdbs/ \
        --out_dir         $DIR/outputs/r2.5/af3_nomsa/ \
        --targets_csv     $DIR/inputs/r2.5/design_epitopes.csv \
        --fastrelax_tsv   $DIR/outputs/r2.5/fastrelax_af3_nomsa/fastrelax_af3_scores.tsv
"""

from __future__ import annotations

import os
import glob
import json

import numpy as np

import af3_design_stats
from af3_design_stats import *   # noqa: F401, F403

MODEL_IDX = 0


# ── Override: file discovery ──────────────────────────────────────────────────

def find_model_cif(af3_out_dir: str, design: str) -> str | None:
    """
    Find fold_{design}_model_{MODEL_IDX}.cif.
    af3_out_dir is already the per-design subdir when called from process_design.
    """
    path = os.path.join(af3_out_dir, f'fold_{design}_model_{MODEL_IDX}.cif')
    if os.path.exists(path):
        return path
    matches = sorted(glob.glob(os.path.join(af3_out_dir, f'fold_*_model_{MODEL_IDX}.cif')))
    return matches[0] if matches else None


def find_top_seed_confidences(af3_out_dir: str) -> str | None:
    """
    Return path to fold_*_full_data_{MODEL_IDX}.json (PAE + token_chain_ids).
    af3_out_dir is the per-design subdir.
    """
    matches = sorted(glob.glob(
        os.path.join(af3_out_dir, f'fold_*_full_data_{MODEL_IDX}.json')
    ))
    return matches[0] if matches else None


def get_top_level_confidences(af3_out_dir: str, design: str) -> dict:
    """Read fold_{design}_summary_confidences_{MODEL_IDX}.json."""
    path = os.path.join(af3_out_dir, f'fold_{design}_summary_confidences_{MODEL_IDX}.json')
    if not os.path.exists(path):
        matches = sorted(glob.glob(
            os.path.join(af3_out_dir, f'fold_*_summary_confidences_{MODEL_IDX}.json')
        ))
        if not matches:
            return {}
        path = matches[0]
    try:
        with open(path) as f:
            d = json.load(f)
        return {
            'af3_iptm':                d.get('iptm',                np.nan),
            'af3_ptm':                 d.get('ptm',                 np.nan),
            'af3_fraction_disordered': d.get('fraction_disordered', np.nan),
        }
    except Exception:
        return {}


def get_top_ranking_score(af3_out_dir: str) -> dict:
    """Read ranking_score from fold_*_summary_confidences_{MODEL_IDX}.json."""
    matches = sorted(glob.glob(
        os.path.join(af3_out_dir, f'fold_*_summary_confidences_{MODEL_IDX}.json')
    ))
    if not matches:
        return {}
    try:
        with open(matches[0]) as f:
            d = json.load(f)
        return {'af3_ranking_score': float(d.get('ranking_score', np.nan))}
    except Exception:
        return {}


# ── Save original before patching ────────────────────────────────────────────
_original_process_design = af3_design_stats.process_design

# Patch module-level discovery functions
af3_design_stats.find_model_cif            = find_model_cif
af3_design_stats.find_top_seed_confidences = find_top_seed_confidences
af3_design_stats.get_top_level_confidences = get_top_level_confidences
af3_design_stats.get_top_ranking_score     = get_top_ranking_score


# ── Override: process_design ──────────────────────────────────────────────────

def process_design(
    design: str,
    af3_out_dir: str,
    relaxed_pdb_dir: str,
    target_info: dict,
) -> tuple[dict, dict[str, list]]:
    """
    Resolves the per-design subdirectory ({af3_out_dir}/{design}/) and calls
    the original process_design with that path.
    Design names in the CSV are already sanitized to match the server directory names.
    """

    return _original_process_design(
        design          = design,
        af3_out_dir     = af3_out_dir,
        relaxed_pdb_dir = relaxed_pdb_dir,
        target_info     = target_info,
    )


# Patch process_design last
af3_design_stats.process_design = process_design


if __name__ == '__main__':
    af3_design_stats.main()