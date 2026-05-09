"""
af3_design_stats_server.py
==========================
Wrapper around af3_design_stats.py that overrides AF3 file discovery
functions to match the AF3 server output directory structure:

  {af3_out_dir}/{design}/
    fold_{design}_model_0.cif
    fold_{design}_summary_confidences_0.json   <- iptm/ptm/ranking_score
    fold_{design}_full_data_0.json             <- PAE, token_chain_ids

Usage (same as af3_design_stats.py):
    python af3_design_stats_server.py \
        --af3_out_dir     /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2.5/af3_nomsa/ \
        --relaxed_pdb_dir /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2.5/fastrelax_af3/relaxed_pdbs/ \
        --out_dir         /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2.5/ \
        --targets_csv     /n/groups/marks/users/aaron/pmhc_cp/post_filter/inputs/r2.5/design_targets.csv \
        --fastrelax_tsv   /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2.5/fastrelax_af3/fastrelax_af3_scores.tsv
"""

from __future__ import annotations

import os
import glob
import json

import numpy as np

# Import everything from af3_design_stats — then override discovery functions
import af3_design_stats
from af3_design_stats import *   # noqa: F401, F403  re-exports all public names

MODEL_IDX = 0   # AF3 server outputs 0-4; always use index 0


# ── Override: file discovery ──────────────────────────────────────────────────

def find_model_cif(af3_out_dir: str, design: str) -> str | None:
    """Find fold_{design}_model_{MODEL_IDX}.cif in {af3_out_dir}/{design}/."""
    path = os.path.join(af3_out_dir, design,
                        f'fold_{design}_model_{MODEL_IDX}.cif')
    return path if os.path.exists(path) else None


def find_top_seed_confidences(af3_out_dir: str) -> str | None:
    """
    Return path to fold_*_full_data_{MODEL_IDX}.json (contains PAE + token_chain_ids).
    """
    matches = sorted(glob.glob(
        os.path.join(af3_out_dir, f'fold_*_full_data_{MODEL_IDX}.json')
    ))
    return matches[0] if matches else None


def get_top_level_confidences(af3_out_dir: str, design: str) -> dict:
    """Read fold_{design}_summary_confidences_{MODEL_IDX}.json."""
    path = os.path.join(af3_out_dir, design,
                        f'fold_{design}_summary_confidences_{MODEL_IDX}.json')
    if not os.path.exists(path):
        return {}
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


# Save original process_design BEFORE patching to avoid infinite recursion
_original_process_design = af3_design_stats.process_design

# Patch module-level discovery functions (safe — no circular reference)
af3_design_stats.find_model_cif            = find_model_cif
af3_design_stats.find_top_seed_confidences = find_top_seed_confidences
af3_design_stats.get_top_level_confidences = get_top_level_confidences
af3_design_stats.get_top_ranking_score     = get_top_ranking_score


# ── Override: process_design — resolve per-design subdir for server layout ────

def process_design(
    design: str,
    af3_out_dir: str,
    relaxed_pdb_dir: str,
    target_info: dict,
) -> tuple[dict, dict[str, list]]:
    """
    Resolves the per-design subdirectory ({af3_out_dir}/{design}/) and calls
    the original af3_design_stats.process_design with that path, so the
    overridden discovery functions glob within the correct directory.
    """
    design_dir = os.path.join(af3_out_dir, design)
    if not os.path.isdir(design_dir):
        print(f"    WARNING: design directory not found: {design_dir}")
        design_dir = af3_out_dir   # fallback

    return _original_process_design(
        design          = design,
        af3_out_dir     = design_dir,
        relaxed_pdb_dir = relaxed_pdb_dir,
        target_info     = target_info,
    )


# ── Patch process_design last (after _original_process_design is saved) ───────
af3_design_stats.process_design = process_design


if __name__ == '__main__':
    af3_design_stats.main()