"""
af3_design_stats.py
===================
Computes four analysis metrics on AF3-predicted pMHC-minibinder structures:

  1. CMS    — ContactMolecularSurface for hotspot peptide residues + total
              (binder vs each peptide residue, on relaxed structure)
  2. ipSAE  — binder<->peptide ipSAE from AF3 PAE matrix
              (top-ranked seed's confidences.json — coordinate-independent)
  3. DSSP   — secondary structure of BINDER CHAIN ONLY
              (% helix / sheet / loop, counts, full SS string, on relaxed structure)
  4. Contacts — all / polar / hydrophobic / H-bond contacts between binder
                and peptide, imported directly from contact_map.py

contact_map.py must be on the Python path (same directory or PYTHONPATH).
Imported functions: compute_contacts, compute_hbonds,
                    get_residues_from_complex, THREE_TO_ONE

Contact detail TSVs are written to {out_dir}/contacts/.
Per-position contact counts are added to the main af3_design_stats.tsv.

Pipeline per design:
  AF3 CIF -> FastRelax -> temp PDB -> contacts (BioPython) + CMS + DSSP
  ranking_scores.csv -> top seed -> seed-{s}_sample-{n}/confidences.json -> ipSAE

Design targets CSV (--targets_csv):
  Required columns: design, peptide_len, cms_hotspot_positions
  cms_hotspot_positions: comma-separated 1-based peptide positions, e.g. "5,7,8"
  See design_targets.csv (Liu et al. Science 2025)

AF3 directory convention:
  {af3_out_dir}/{design}/
    {design}_model.cif              <- top-ranked model
    ranking_scores.csv              <- seed, sample, ranking_score
    seed-{seed}_sample-{sample}/
      confidences.json              <- PAE, token_chain_ids, plddts

Chain convention in AF3 CIF:
  A = binder,  B = MHC alpha,  C = peptide,  [D = B2M]

Usage:
    python af3_design_stats.py \
        --af3_out_dir     /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3/ \
        --relaxed_pdb_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/fastrelax_af3/relaxed_pdbs/ \
        --out_dir         /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/ \
        --targets_csv     /n/groups/marks/users/aaron/pmhc/post_filter/inputs/design_epitopes.csv

    # Specific designs:
    python af3_design_stats.py ... --designs ctnnb1-15 gp100-3 mart1-3

    # Inspect chain layout only:
    python af3_design_stats.py ... --inspect ctnnb1-15
"""

from __future__ import annotations

import os
import re
import glob
import json
import tempfile
import argparse

import numpy as np
import pandas as pd

import pyrosetta
from pyrosetta import Pose
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
from pyrosetta.rosetta.core.import_pose import FileType
from pyrosetta.rosetta.core.select.residue_selector import (
    ChainSelector,
    ResidueIndexSelector,
)
from pyrosetta.rosetta.protocols.simple_filters import ContactMolecularSurfaceFilter
from pyrosetta.rosetta.protocols.moves import DsspMover

# contact_map.py must be on the path (same directory or PYTHONPATH)
from contact_map import (
    compute_contacts,
    compute_hbonds,
    get_residues_from_complex,
    THREE_TO_ONE,
    PEPTIDE_LEN as CM_PEPTIDE_LEN,   # used only as a sanity reference
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SS_MAP = {
    'H': 'helix', 'G': 'helix', 'I': 'helix',
    'E': 'sheet', 'B': 'sheet',
    'T': 'loop',  'S': 'loop',  'L': 'loop',
    ' ': 'loop',
}

DEFAULT_PEPTIDE_LEN = 9


# ─────────────────────────────────────────────────────────────────────────────
# Design targets table
# ─────────────────────────────────────────────────────────────────────────────

def load_targets(csv_path: str) -> dict:
    """
    Load design_targets.csv.
    Returns {design: {'peptide_len': int, 'hotspot_positions': [int]}}
    """
    df = pd.read_csv(csv_path)
    missing = {'design', 'peptide_len', 'cms_hotspot_positions'} - set(df.columns)
    if missing:
        raise ValueError(f"design_targets.csv missing columns: {missing}")
    out = {}
    for _, row in df.iterrows():
        design   = str(row['design']).strip()
        plen     = int(row['peptide_len'])
        hotspots = [int(x.strip())
                    for x in str(row['cms_hotspot_positions']).split(',')
                    if x.strip()]
        out[design] = {'peptide_len': plen, 'hotspot_positions': hotspots}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PyRosetta init
# ─────────────────────────────────────────────────────────────────────────────

def init_pyrosetta():
    pyrosetta.init(
        '-mute all -ex1 -ex2aro -use_input_sc '
        '-ignore_unrecognized_res true '
        '-ignore_zero_occupancy false '
        '-load_PDB_components false',
        silent=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# AF3 file discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_designs(af3_out_dir: str) -> list[str]:
    return sorted(
        d for d in os.listdir(af3_out_dir)
        if os.path.isdir(os.path.join(af3_out_dir, d))
    )


def find_top_seed_confidences(af3_out_dir: str) -> str | None:
    ranking = os.path.join(af3_out_dir, 'ranking_scores.csv')
    if not os.path.exists(ranking):
        return None
    df  = pd.read_csv(ranking)
    row = df.sort_values('ranking_score', ascending=False).iloc[0]
    p   = os.path.join(af3_out_dir,
                       f'seed-{int(row["seed"])}_sample-{int(row["sample"])}',
                       'confidences.json')
    return p if os.path.exists(p) else None


def get_top_level_confidences(af3_out_dir: str, design: str) -> dict:
    path = os.path.join(af3_out_dir, f'{design}_confidences.json')
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


def get_top_ranking_score(af3_out_dir: str, design: str) -> dict:
    path = os.path.join(af3_out_dir, 'ranking_scores.csv')
    if not os.path.exists(path):
        return {}
    try:
        df  = pd.read_csv(path)
        row = df.sort_values('ranking_score', ascending=False).iloc[0]
        return {
            'af3_top_seed':      int(row['seed']),
            'af3_top_sample':    int(row['sample']),
            'af3_ranking_score': float(row['ranking_score']),
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# CIF / Pose utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_cif(cif_path: str) -> Pose:
    pose = Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_file(
        pose, cif_path, False, FileType.CIF_file
    )
    return pose


def get_chain_ids(pose: Pose) -> list[str]:
    chains, seen = [], set()
    for i in range(1, pose.total_residue() + 1):
        ch = pose.pdb_info().chain(i)
        if ch not in seen:
            seen.add(ch)
            chains.append(ch)
    return chains


def detect_peptide_chain(chain_ids: list[str]) -> str:
    """3rd chain = peptide (A=binder, B=MHC, C=peptide, [D=B2M])."""
    if len(chain_ids) < 3:
        raise ValueError(f"Expected >= 3 chains, got: {chain_ids}")
    return chain_ids[2]


def get_peptide_pose_indices(pose: Pose, pep_chain: str, pep_len: int) -> list[int]:
    res = [i for i in range(1, pose.total_residue() + 1)
           if pose.pdb_info().chain(i) == pep_chain]
    if len(res) < pep_len:
        raise ValueError(
            f"Chain {pep_chain} has {len(res)} residues, expected >= {pep_len}"
        )
    return res[-pep_len:]


def build_interface_string(chain_ids: list[str]) -> str:
    others = [c for c in chain_ids if c != 'A']
    return f"A_{''.join(others)}"


# ─────────────────────────────────────────────────────────────────────────────
# FastRelax
# ─────────────────────────────────────────────────────────────────────────────

def make_scorefxn(with_cst: bool = False):
    sfxn = ScoreFunctionFactory.create_score_function('ref2015')
    if with_cst:
        sfxn.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint, 1.0
        )
    return sfxn


def run_fastrelax(pose: Pose, scorefxn, n_repeats: int = 3) -> Pose:
    fr = FastRelax(scorefxn_in=scorefxn, standard_repeats=n_repeats)
    fr.constrain_relax_to_start_coords(True)
    fr.apply(pose)
    return pose


# ─────────────────────────────────────────────────────────────────────────────
# DSSP — binder chain only
# ─────────────────────────────────────────────────────────────────────────────

def compute_dssp_binder(pose: Pose, binder_chain: str = 'A') -> dict:
    """
    DsspMover applied once to full pose; fractions computed for binder only.
    H/G/I -> helix, E/B -> sheet, T/S/L/' ' -> loop.
    """
    DsspMover().apply(pose)
    counts   = {'helix': 0, 'sheet': 0, 'loop': 0}
    ss_chars = []
    for i in range(1, pose.total_residue() + 1):
        if pose.pdb_info().chain(i) != binder_chain:
            continue
        ch = pose.secstruct(i)
        counts[SS_MAP.get(ch, 'loop')] += 1
        ss_chars.append(ch)
    n = sum(counts.values())
    if n == 0:
        return {'n_binder_res': 0,
                'ss_helix_pct': np.nan, 'ss_sheet_pct': np.nan,
                'ss_loop_pct':  np.nan,
                'n_helix': 0, 'n_sheet': 0, 'n_loop': 0,
                'dssp_string': ''}
    return {
        'n_binder_res':  n,
        'ss_helix_pct':  round(100.0 * counts['helix'] / n, 2),
        'ss_sheet_pct':  round(100.0 * counts['sheet'] / n, 2),
        'ss_loop_pct':   round(100.0 * counts['loop']  / n, 2),
        'n_helix':       counts['helix'],
        'n_sheet':       counts['sheet'],
        'n_loop':        counts['loop'],
        'dssp_string':   ''.join(ss_chars),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CMS scoring
# ─────────────────────────────────────────────────────────────────────────────

def _cms_single_residue(pose: Pose, binder_chain: str,
                         pep_res_idx: int, dist_w: float = 0.5) -> float:
    cms = ContactMolecularSurfaceFilter()
    cms.distance_weight(dist_w)
    cms.selector1(ChainSelector(binder_chain))
    cms.selector2(ResidueIndexSelector(str(pep_res_idx)))
    cms.verbose(False)
    return cms.report_sm(pose)


def compute_cms(pose: Pose, binder_chain: str, pep_chain: str,
                pep_len: int, hotspot_positions: list[int],
                dist_w: float = 0.5) -> dict:
    """
    cms_p{1..N}, cms_peptide_total,
    cms_hotspot_p{pos} for each hotspot, cms_hotspot_total.
    """
    pep_indices = get_peptide_pose_indices(pose, pep_chain, pep_len)
    result, cms_values = {}, []
    for pos, res_idx in enumerate(pep_indices, start=1):
        val = _cms_single_residue(pose, binder_chain, res_idx, dist_w)
        result[f'cms_p{pos}'] = round(val, 4)
        cms_values.append(val)
    result['cms_peptide_total'] = round(sum(cms_values), 4)
    hotspot_vals = []
    for pos in hotspot_positions:
        if 1 <= pos <= len(cms_values):
            val = cms_values[pos - 1]
            result[f'cms_hotspot_p{pos}'] = round(val, 4)
            hotspot_vals.append(val)
    result['cms_hotspot_total'] = round(sum(hotspot_vals), 4)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ipSAE from AF3 PAE matrix
# ─────────────────────────────────────────────────────────────────────────────

def load_af3_pae(conf_path: str) -> tuple[np.ndarray | None, list[str] | None]:
    with open(conf_path) as f:
        d = json.load(f)
    if 'pae' not in d or 'token_chain_ids' not in d:
        return None, None
    pae = np.array(d['pae'], dtype=np.float32)
    tok = d['token_chain_ids']
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"PAE unexpected shape: {pae.shape}")
    if len(tok) != pae.shape[0]:
        raise ValueError(f"token_chain_ids len {len(tok)} != PAE dim {pae.shape[0]}")
    return pae, tok


def _chain_idx(token_chain_ids: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, ch in enumerate(token_chain_ids):
        out.setdefault(ch, []).append(i)
    return out


def _ipsae_dir(pae_sub: np.ndarray, cutoff: float = 12.0,
               min_d0: float = 0.5) -> tuple[float, int, float]:
    flat = pae_sub.flatten()
    ct   = flat[flat < cutoff]
    n    = len(ct)
    if n == 0:
        return 0.0, 0, min_d0
    d0    = max(1.24 * max(n - 15, 1) ** (1.0 / 3.0) - 1.8, min_d0)
    score = float(np.mean(1.0 / (1.0 + (ct / d0) ** 2)))
    return score, n, d0


def compute_ipsae(pae: np.ndarray, token_chain_ids: list[str],
                  binder_chain: str = 'A', peptide_chain: str = 'C',
                  cutoff: float = 12.0) -> dict:
    """Schaeffer & Dunbrack (2025) binder<->peptide ipSAE."""
    cidx = _chain_idx(token_chain_ids)
    for ch, label in [(binder_chain, 'binder'), (peptide_chain, 'peptide')]:
        if ch not in cidx:
            raise ValueError(f"{label} chain '{ch}' not in PAE: {list(cidx)}")
    b, p   = cidx[binder_chain], cidx[peptide_chain]
    pae_bp = pae[np.ix_(b, p)]
    pae_pb = pae[np.ix_(p, b)]
    s_bp, n_bp, d0_bp = _ipsae_dir(pae_bp, cutoff)
    s_pb, n_pb, d0_pb = _ipsae_dir(pae_pb, cutoff)
    if s_bp >= s_pb:
        best, n_best, d0_best, pae_best = s_bp, n_bp, d0_bp, pae_bp
    else:
        best, n_best, d0_best, pae_best = s_pb, n_pb, d0_pb, pae_pb
    ct           = pae_best.flatten()[pae_best.flatten() < cutoff]
    mean_contact = float(np.mean(ct)) if len(ct) else np.nan
    return {
        'ipsae_binder_peptide': round(best,                     4),
        'ipsae_bp':             round(s_bp,                     4),
        'ipsae_pb':             round(s_pb,                     4),
        'ipsae_n_contacts':     n_best,
        'ipsae_d0':             round(d0_best,                  3),
        'mean_pae_bp':          round(float(np.mean(pae_bp)),   3),
        'mean_pae_contact':     round(mean_contact,             3),
        'af3_chains_in_pae':    ','.join(sorted(set(token_chain_ids))),
        'n_binder_tokens':      len(b),
        'n_peptide_tokens':     len(p),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Contact analysis (via contact_map.py import)
# ─────────────────────────────────────────────────────────────────────────────

def _contacts_summary(
    binder_res: list,
    pep_res: list,
    all_cts: list,
    polar_cts: list,
    hydro_cts: list,
    hbond_cts: list,
    has_H: bool,
    peptide_len: int,
    hotspot_positions: list[int],
) -> dict:
    """
    Build the contacts summary dict from pre-computed contact lists.
    Shared by compute_contacts_from_pose (pose path) and process_design
    (direct relaxed PDB path) to avoid duplicating logic.
    """
    SALT_BRIDGE_AA = {'R', 'K'}
    n_pep = len(pep_res)

    def _per_pos(contacts):
        counts = [0] * n_pep
        for item in contacts:
            pi = item['pi'] if isinstance(item, dict) else item[1]
            if 0 <= pi < n_pep:
                counts[pi] += 1
        return counts

    all_pp   = _per_pos(all_cts)
    polar_pp = _per_pos(polar_cts)
    hydro_pp = _per_pos(hydro_cts)
    hbond_pp = _per_pos(hbond_cts)

    summary: dict = {
        'n_all_contacts':         len(all_cts),
        'n_polar_contacts':       len(polar_cts),
        'n_hydrophobic_contacts': len(hydro_cts),
        'n_hbonds':               len(hbond_cts),
        'hbond_available':        has_H,
        'binder_res_in_contact':  len(set(
            item[0] if not isinstance(item, dict) else item['bi']
            for item in all_cts
        )),
    }

    for i, (a, po, h, hb) in enumerate(
            zip(all_pp, polar_pp, hydro_pp, hbond_pp), start=1):
        summary[f'contacts_all_p{i}']   = a
        summary[f'contacts_polar_p{i}'] = po
        summary[f'contacts_hydro_p{i}'] = h
        summary[f'contacts_hbond_p{i}'] = hb

    hs_total_all = hs_total_polar = hs_total_hydro = hs_total_hbond = 0

    for pos in hotspot_positions:
        pi = pos - 1
        if not (0 <= pi < n_pep):
            continue
        n_all   = all_pp[pi]
        n_polar = polar_pp[pi]
        n_hydro = hydro_pp[pi]
        n_hbond = hbond_pp[pi]

        hs_total_all   += n_all
        hs_total_polar += n_polar
        hs_total_hydro += n_hydro
        hs_total_hbond += n_hbond

        summary[f'n_contacts_hotspot_p{pos}_all']   = n_all
        summary[f'n_contacts_hotspot_p{pos}_polar']  = n_polar
        summary[f'n_contacts_hotspot_p{pos}_hydro']  = n_hydro
        summary[f'n_contacts_hotspot_p{pos}_hbond']  = n_hbond

        # Salt bridge: polar contact at this hotspot where binder AA is R or K
        # polar_cts tuple: (bi, pi, dist, b_atom, p_atom, b_aa, p_aa)
        sb = [(bi, b_aa) for (bi, ppi, dist, b_at, p_at, b_aa, p_aa) in polar_cts
              if ppi == pi and b_aa in SALT_BRIDGE_AA]
        summary[f'n_saltbridge_hotspot_p{pos}']              = len(sb)
        summary[f'saltbridge_hotspot_p{pos}_binder_resnums'] = str(
            sorted(set(binder_res[bi].id[1] for bi, _ in sb
                       if bi < len(binder_res)))
        )
        summary[f'saltbridge_hotspot_p{pos}_binder_aas']     = str(
            [b_aa for _, b_aa in sb]
        )

    summary['n_contacts_hotspot_total_all']   = hs_total_all
    summary['n_contacts_hotspot_total_polar']  = hs_total_polar
    summary['n_contacts_hotspot_total_hydro']  = hs_total_hydro
    summary['n_contacts_hotspot_total_hbond']  = hs_total_hbond

    return summary


def compute_contacts_from_pose(
    pose: Pose,
    peptide_len: int,
    hotspot_positions: list[int],
    binder_chain: str = 'A',
    pmhc_chain:   str = 'B',
) -> tuple[dict, list, list, list, list]:
    """
    Dump pose to a temp PDB, load with BioPython, compute contacts.
    Used when a pre-relaxed PDB is not available.
    Returns (summary_dict, all_cts, polar_cts, hydro_cts, hbond_cts).
    """
    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pose.dump_pdb(tmp_path)
        binder_res, pmhc_res = get_residues_from_complex(
            tmp_path, binder_chain=binder_chain, pmhc_chain=pmhc_chain
        )
    finally:
        os.unlink(tmp_path)

    pep_res = pmhc_res[-peptide_len:]

    if not binder_res or not pep_res:
        return {'n_all_contacts': 0, 'n_polar_contacts': 0,
                'n_hydrophobic_contacts': 0, 'n_hbonds': 0,
                'hbond_available': False}, [], [], [], []

    all_cts, polar_cts, hydro_cts = compute_contacts(binder_res, pep_res)
    has_H     = any(a.element == 'H'
                    for res in binder_res for a in res.get_atoms())
    hbond_cts = compute_hbonds(binder_res, pep_res) if has_H else []

    summary = _contacts_summary(
        binder_res, pep_res,
        all_cts, polar_cts, hydro_cts, hbond_cts,
        has_H, peptide_len, hotspot_positions,
    )
    return summary, all_cts, polar_cts, hydro_cts, hbond_cts


def contacts_to_rows(design: str, all_cts, polar_cts, hydro_cts, hbond_cts,
                     binder_res_list, pep_res_list,
                     hotspot_positions: list[int] = None) -> dict[str, list]:
    """
    Convert raw contact tuples/dicts to TSV-ready row lists.
    Adds is_hotspot column (True/False) to every row when hotspot_positions given.
    """
    hotspot_set  = set(hotspot_positions) if hotspot_positions else set()
    all_rows, polar_rows, hydro_rows, hbond_rows = [], [], [], []

    for (bi, pi, dist, b_aa, p_aa) in all_cts:
        all_rows.append({
            'design': design, 'binder_pos_0idx': bi,
            'binder_resnum': binder_res_list[bi].id[1] if bi < len(binder_res_list) else '',
            'binder_aa': b_aa, 'pep_pos_1idx': pi + 1, 'pep_aa': p_aa,
            'min_heavy_dist': dist,
            'is_hotspot': (pi + 1) in hotspot_set,
        })
    for (bi, pi, dist, b_at, p_at, b_aa, p_aa) in polar_cts:
        polar_rows.append({
            'design': design, 'binder_pos_0idx': bi,
            'binder_resnum': binder_res_list[bi].id[1] if bi < len(binder_res_list) else '',
            'binder_aa': b_aa, 'binder_atom': b_at,
            'pep_pos_1idx': pi + 1, 'pep_aa': p_aa, 'pep_atom': p_at,
            'polar_dist': dist,
            'is_hotspot': (pi + 1) in hotspot_set,
        })
    for (bi, pi, dist, b_aa, p_aa) in hydro_cts:
        hydro_rows.append({
            'design': design, 'binder_pos_0idx': bi,
            'binder_resnum': binder_res_list[bi].id[1] if bi < len(binder_res_list) else '',
            'binder_aa': b_aa, 'pep_pos_1idx': pi + 1, 'pep_aa': p_aa,
            'pep_resname': pep_res_list[pi].resname if pi < len(pep_res_list) else '',
            'min_dist': dist,
            'is_hotspot': (pi + 1) in hotspot_set,
        })
    for hb in hbond_cts:
        hbond_rows.append({
            'design': design,
            'binder_pos_0idx': hb['bi'],
            'binder_resnum': binder_res_list[hb['bi']].id[1]
                             if hb['bi'] < len(binder_res_list) else '',
            'binder_aa':   hb['b_aa'],  'binder_atom': hb['b_atom'],
            'pep_pos_1idx': hb['pi'] + 1, 'pep_aa': hb['p_aa'],
            'pep_atom':    hb['p_atom'], 'direction': hb['direction'],
            'da_dist':     hb['da_dist'], 'dha_angle': hb['dha_angle'],
            'is_hotspot': (hb['pi'] + 1) in hotspot_set,
        })
    return {'all': all_rows, 'polar': polar_rows,
            'hydro': hydro_rows, 'hbond': hbond_rows}


# ─────────────────────────────────────────────────────────────────────────────
# Per-design orchestration
# ─────────────────────────────────────────────────────────────────────────────

def process_design(
    design: str,
    af3_out_dir: str,
    relaxed_pdb_dir: str,
    target_info: dict,
) -> tuple[dict, dict[str, list]]:
    """
    Returns (record_dict, contact_rows_dict).
    contact_rows_dict keys: 'all', 'polar', 'hydro', 'hbond'

    CMS and DSSP run on the pre-relaxed PDB loaded into PyRosetta.
    Contacts run on the same pre-relaxed PDB loaded via BioPython (contact_map).
    ipSAE comes from AF3 confidences.json (coordinate-independent).
    """
    peptide_len       = target_info['peptide_len']
    hotspot_positions = target_info['hotspot_positions']
    empty_rows        = {'all': [], 'polar': [], 'hydro': [], 'hbond': []}

    record = {
        'design':            design,
        'peptide_len':       peptide_len,
        'hotspot_positions': ','.join(str(p) for p in hotspot_positions),
    }

    af3_out_dir_full = "Empty"
    # accommodate alternative af3_out_dir
    for out_dir in glob.glob(f"{af3_out_dir}/{design}*"):
        if os.path.join(out_dir, f"{design}_confidences.json"):
            af3_out_dir_full = out_dir

    # AF3 summary metrics
    record.update(get_top_level_confidences(af3_out_dir_full, design))
    record.update(get_top_ranking_score(af3_out_dir_full, design))

    # ipSAE — PAE-based, coordinate-independent
    conf_path = find_top_seed_confidences(af3_out_dir_full)
    if conf_path:
        try:
            pae, tok = load_af3_pae(conf_path)
            if pae is not None:
                unique_ch    = list(dict.fromkeys(tok))
                pep_ch_pae   = unique_ch[2] if len(unique_ch) >= 3 else unique_ch[-1]
                ipsae_scores = compute_ipsae(pae, tok,
                                             binder_chain='A',
                                             peptide_chain=pep_ch_pae)
                record.update(ipsae_scores)
                print(f"    ipSAE: {ipsae_scores['ipsae_binder_peptide']:.4f}  "
                      f"(pep_chain={pep_ch_pae}, "
                      f"n_contacts={ipsae_scores['ipsae_n_contacts']})")
            else:
                print(f"    WARNING: PAE/token_chain_ids missing in {conf_path}")
        except Exception as e:
            print(f"    WARNING: ipSAE failed: {e}")
    else:
        print(f"    WARNING: top-seed confidences.json not found at {conf_path}")

    # Load pre-relaxed PDB
    relaxed_pdb = os.path.join(relaxed_pdb_dir, f'{design}_relaxed.pdb')
    if not os.path.exists(relaxed_pdb):
        print(f"    ERROR: relaxed PDB not found: {relaxed_pdb}")
        return record, empty_rows
    record['relaxed_pdb'] = relaxed_pdb

    # Load into PyRosetta for CMS + DSSP
    try:
        pose = pyrosetta.pose_from_pdb(relaxed_pdb)
    except Exception as e:
        print(f"    ERROR loading relaxed PDB into PyRosetta: {e}")
        return record, empty_rows

    chain_ids = get_chain_ids(pose)
    interface = build_interface_string(chain_ids)
    record['chains']    = ','.join(chain_ids)
    record['interface'] = interface

    try:
        pep_chain = detect_peptide_chain(chain_ids)
    except ValueError as e:
        print(f"    ERROR: {e}")
        return record, empty_rows

    print(f"    chains: {chain_ids}  iface: {interface}  "
          f"pep_chain: {pep_chain}  pep_len: {peptide_len}")

    # DSSP — binder chain only; single DsspMover call
    try:
        dssp = compute_dssp_binder(pose, binder_chain='A')
        record.update(dssp)
        print(f"    DSSP(binder): H={dssp['ss_helix_pct']}%  "
              f"E={dssp['ss_sheet_pct']}%  "
              f"L={dssp['ss_loop_pct']}%  "
              f"n={dssp['n_binder_res']}")
    except Exception as e:
        print(f"    WARNING: DSSP failed: {e}")

    # CMS — all positions + hotspots
    try:
        cms = compute_cms(pose, binder_chain='A', pep_chain=pep_chain,
                          pep_len=peptide_len, hotspot_positions=hotspot_positions)
        record.update(cms)
        print(f"    CMS: total={cms['cms_peptide_total']:.2f}  "
              f"hotspot={cms['cms_hotspot_total']:.2f}  "
              f"pos={hotspot_positions}")
    except Exception as e:
        print(f"    WARNING: CMS failed: {e}")

    # Contacts — via contact_map.py, reads relaxed PDB directly with BioPython
    # relaxed PDB is 2-chain: A=binder, B=MHC+peptide
    contact_rows = empty_rows
    try:
        binder_res_list, pmhc_res_list = get_residues_from_complex(
            relaxed_pdb, binder_chain='A', pmhc_chain='B'
        )
        pep_res_list = pmhc_res_list[-peptide_len:]

        all_cts, polar_cts, hydro_cts = compute_contacts(binder_res_list, pep_res_list)
        has_H     = any(a.element == 'H'
                        for res in binder_res_list for a in res.get_atoms())
        hbond_cts = compute_hbonds(binder_res_list, pep_res_list) if has_H else []

        # Build summary directly from the loaded residue lists
        # (mirrors compute_contacts_from_pose logic without the temp PDB write)
        ct_summary = _contacts_summary(
            binder_res_list, pep_res_list,
            all_cts, polar_cts, hydro_cts, hbond_cts,
            has_H, peptide_len, hotspot_positions,
        )
        record.update(ct_summary)

        contact_rows = contacts_to_rows(
            design, all_cts, polar_cts, hydro_cts, hbond_cts,
            binder_res_list, pep_res_list,
            hotspot_positions=hotspot_positions,
        )
        hs_all   = ct_summary.get('n_contacts_hotspot_total_all',  0)
        hs_polar = ct_summary.get('n_contacts_hotspot_total_polar', 0)
        print(f"    Contacts: all={ct_summary['n_all_contacts']}  "
              f"polar={ct_summary['n_polar_contacts']}  "
              f"hydro={ct_summary['n_hydrophobic_contacts']}  "
              f"hbonds={ct_summary['n_hbonds']}  "
              f"(H={ct_summary['hbond_available']})  "
              f"hotspot_all={hs_all}  hotspot_polar={hs_polar}")
    except Exception as e:
        print(f"    WARNING: contact_map failed: {e}")

    return record, contact_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--af3_out_dir',     required=True)
    parser.add_argument('--relaxed_pdb_dir', required=True,
        help='Directory containing {design}_relaxed.pdb files from FastRelax '
             '(e.g. .../fastrelax_af3/relaxed_pdbs/)')
    parser.add_argument('--out_dir',         required=True)
    parser.add_argument('--targets_csv',     required=True,
        help='CSV: design, peptide_len, cms_hotspot_positions')
    parser.add_argument('--designs',         nargs='*')
    parser.add_argument('--inspect',         metavar='DESIGN')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    contacts_dir = os.path.join(args.out_dir, 'contacts')
    os.makedirs(contacts_dir, exist_ok=True)

    targets = load_targets(args.targets_csv)
    print(f"Loaded {len(targets)} targets from {args.targets_csv}")

    init_pyrosetta()

    # Inspect mode
    if args.inspect:
        d           = args.inspect
        relaxed_pdb = os.path.join(args.relaxed_pdb_dir, f'{d}_relaxed.pdb')
        if not os.path.exists(relaxed_pdb):
            print(f"ERROR: relaxed PDB not found: {relaxed_pdb}"); return
        pose   = pyrosetta.pose_from_pdb(relaxed_pdb)
        chains = get_chain_ids(pose)
        conf   = find_top_seed_confidences(args.af3_out_dir, d)
        info   = targets.get(d, {})
        print(f"\nDesign:      {d}")
        print(f"Relaxed PDB: {relaxed_pdb}")
        print(f"Chains:      {chains}  ->  {build_interface_string(chains)}")
        for ch in chains:
            sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(ch)
            rv  = pyrosetta.rosetta.core.select.residue_selector.ResidueVector(
                      sel.apply(pose))
            print(f"  Chain {ch}: {len(rv)} residues")
        if info:
            print(f"peptide_len:  {info['peptide_len']}")
            print(f"hotspots:     {info['hotspot_positions']}")
        if conf:
            pae, tok = load_af3_pae(conf)
            if pae is not None:
                cc = {}
                for c in tok: cc[c] = cc.get(c, 0) + 1
                print(f"PAE: {pae.shape[0]}x{pae.shape[1]}  chains={cc}")
        return

    # Discover designs from relaxed PDB dir if not specified
    if args.designs:
        designs = args.designs
    else:
        pdbs    = glob.glob(os.path.join(args.relaxed_pdb_dir, '*_relaxed.pdb'))
        designs = sorted(os.path.basename(p).replace('_relaxed.pdb', '') for p in pdbs)
        unknown = [d for d in designs if d not in targets]
        if unknown:
            print(f"WARNING: {len(unknown)} relaxed PDBs not in targets CSV "
                  f"(fallback pep_len={DEFAULT_PEPTIDE_LEN}): {unknown}")

    print(f"Processing {len(designs)} designs\n")

    records       = []
    all_ct_rows   = []
    polar_ct_rows = []
    hydro_ct_rows = []
    hbond_ct_rows = []

    for idx, design in enumerate(designs):
        print(f"[{idx+1}/{len(designs)}] {design}")
        info = targets.get(design, {
            'peptide_len':       DEFAULT_PEPTIDE_LEN,
            'hotspot_positions': [],
        })
        rec, ct_rows = process_design(
            design          = design,
            af3_out_dir     = args.af3_out_dir,
            relaxed_pdb_dir = args.relaxed_pdb_dir,
            target_info     = info,
        )
        records.append(rec)
        all_ct_rows   .extend(ct_rows['all'])
        polar_ct_rows .extend(ct_rows['polar'])
        hydro_ct_rows .extend(ct_rows['hydro'])
        hbond_ct_rows .extend(ct_rows['hbond'])
        print()

    # Save contact TSVs
    pd.DataFrame(all_ct_rows)  .to_csv(os.path.join(contacts_dir, 'all_contacts.tsv'),   sep='\t', index=False)
    pd.DataFrame(polar_ct_rows).to_csv(os.path.join(contacts_dir, 'polar_contacts.tsv'), sep='\t', index=False)
    pd.DataFrame(hydro_ct_rows).to_csv(os.path.join(contacts_dir, 'hydrophobic_contacts.tsv'), sep='\t', index=False)
    pd.DataFrame(hbond_ct_rows).to_csv(os.path.join(contacts_dir, 'hbond_contacts.tsv'), sep='\t', index=False)
    print(f"Contact TSVs -> {contacts_dir}/")

    # Save main TSV
    df = pd.DataFrame(records)

    id_cols    = ['design', 'peptide_len', 'hotspot_positions',
                  'cif_path', 'chains', 'interface', 'relaxed_pdb']
    af3_cols   = ['af3_iptm', 'af3_ptm', 'af3_ranking_score',
                  'af3_fraction_disordered', 'af3_top_seed', 'af3_top_sample']
    ipsae_cols = ['ipsae_binder_peptide', 'ipsae_bp', 'ipsae_pb',
                  'ipsae_n_contacts', 'ipsae_d0',
                  'mean_pae_bp', 'mean_pae_contact',
                  'af3_chains_in_pae', 'n_binder_tokens', 'n_peptide_tokens']
    dssp_cols  = ['n_binder_res', 'ss_helix_pct', 'ss_sheet_pct', 'ss_loop_pct',
                  'n_helix', 'n_sheet', 'n_loop', 'dssp_string']
    max_plen   = int(df['peptide_len'].max()) if 'peptide_len' in df.columns else 10
    cms_cols   = ([f'cms_p{i}' for i in range(1, max_plen + 1)]
                  + ['cms_peptide_total', 'cms_hotspot_total']
                  + sorted([c for c in df.columns if c.startswith('cms_hotspot_p')],
                            key=lambda x: int(x.replace('cms_hotspot_p', ''))))
    ct_sum_cols = ['n_all_contacts', 'n_polar_contacts', 'n_hydrophobic_contacts',
                   'n_hbonds', 'hbond_available', 'binder_res_in_contact',
                   'n_contacts_hotspot_total_all', 'n_contacts_hotspot_total_polar',
                   'n_contacts_hotspot_total_hydro', 'n_contacts_hotspot_total_hbond']
    # Per-hotspot position summary columns (sorted by position number)
    ct_hotspot_cols = sorted(
        [c for c in df.columns if 'hotspot_p' in c and not c.startswith('contacts_')],
        key=lambda x: int(re.search(r'_p(\d+)_', x).group(1))
        if re.search(r'_p(\d+)_', x) else 0
    )
    ct_pos_cols = sorted(
        [c for c in df.columns
         if c.startswith('contacts_') and '_p' in c],
        key=lambda x: (x.split('_p')[0], int(x.split('_p')[1]))
    )
    ros_cols   = ['rosetta_score_before', 'rosetta_score_after',
                  'rosetta_score_delta']

    ordered = (id_cols + af3_cols + ipsae_cols + dssp_cols
               + cms_cols + ct_sum_cols + ct_hotspot_cols + ct_pos_cols + ros_cols)
    present = [c for c in ordered if c in df.columns]
    extras  = [c for c in df.columns if c not in present]
    df      = df[present + extras]

    out_path = os.path.join(args.out_dir, 'af3_design_stats.tsv')
    df.to_csv(out_path, sep='\t', index=False)
    print(f"Saved -> {out_path}\n")

    # Summary
    key_cols = ['af3_iptm', 'af3_ranking_score', 'ipsae_binder_peptide',
                'mean_pae_bp', 'ss_helix_pct', 'ss_sheet_pct', 'ss_loop_pct',
                'cms_peptide_total', 'cms_hotspot_total',
                'n_all_contacts', 'n_polar_contacts', 'n_hbonds',
                'n_contacts_hotspot_total_all', 'n_contacts_hotspot_total_polar',
                'n_contacts_hotspot_total_hbond']
    anchor = 'ipsae_binder_peptide'
    valid  = df.dropna(subset=[anchor]) if anchor in df.columns else df
    print(f"-- Summary ({len(valid)} designs with ipSAE) --")
    for col in key_cols:
        if col in valid.columns and valid[col].notna().any():
            v = valid[col].dropna()
            print(f"  {col:30s}  mean={v.mean():.3f}  "
                  f"min={v.min():.3f}  max={v.max():.3f}  "
                  f"std={v.std():.3f}")


if __name__ == '__main__':
    main()