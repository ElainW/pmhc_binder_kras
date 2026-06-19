"""
compare_msa_nomsa.py
====================
Compares AF3 structure predictions with and without minibinder MSA.

Imports core scoring functions from af3_design_stats.py — no code duplication.

Outputs (all written to --out_dir):
  1. msa_nomsa_raw.tsv      — raw stats for both conditions, one row per design×condition
  2. msa_nomsa_delta.tsv    — delta (nomsa − msa) for each metric, one row per design
  3. msa_nomsa_rmsd.tsv     — Ca RMSD per design:
                               - binder RMSD  (after aligning on MHC chain B Ca)
                               - peptide RMSD (after aligning on MHC chain B Ca)
                               - MHC RMSD     (direct chain B Ca RMSD, sanity check)

Metrics compared:
  ipsae_binder_peptide, ipsae_n_contacts,
  cms_hotspot_total, cms_peptide_total,
  n_contacts_hotspot_total_all, n_contacts_hotspot_total_polar,
  n_all_contacts, n_polar_contacts, n_hbonds,
  ss_helix_pct, ss_loop_pct,
  n_binder_res (sanity check)

Ca RMSD strategy:
  1. Load MSA relaxed PDB and no-MSA relaxed PDB
  2. Extract MHC chain B Ca coords from both
  3. Kabsch-align no-MSA onto MSA using MHC Ca (common reference frame)
  4. Apply same transform to binder (chain A) and peptide (chain C) Ca
  5. Report RMSD for binder Ca and peptide Ca in the MHC-aligned frame

Directory conventions:
  MSA:    {af3_msa_dir}/{design}_*/        (e.g. ctnnb1-15_20260412_181420)
  no-MSA: {af3_nomsa_dir}/{design}_*/      (e.g. ctnnb1-15_20260421_HHMMSS)
  MSA relaxed PDBs:    {relaxed_msa_dir}/{design}_relaxed.pdb
  no-MSA relaxed PDBs: {relaxed_nomsa_dir}/{design}_relaxed.pdb

Usage:
    python compare_msa_nomsa.py \
        --af3_msa_dir    ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/af3/ \
        --af3_nomsa_dir  ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/af3_nomsa/ \
        --relaxed_msa_dir   ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/fastrelax_af3/relaxed_pdbs/ \
        --relaxed_nomsa_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/fastrelax_af3_nomsa/relaxed_pdbs/ \
        --targets_csv    ~/Desktop/pmhc_binder_kras/post_filter/inputs/design_epitopes.csv \
        --out_dir        ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/msa_comparison/ \
        --designs gp100-3 hiv-10 hiv-9 mage-282 mage-4 mage-513 phox2b-11 prame-2 prame-9 sars-6 wt1-5 wt1-8 yfv-2
"""

from __future__ import annotations

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

# Import core functions from af3_design_stats.py — must be on PYTHONPATH
# or in the same directory
try:
    from af3_design_stats import (
        load_targets,
        init_pyrosetta,
        find_top_seed_confidences,
        get_top_level_confidences,
        get_top_ranking_score,
        load_af3_pae,
        compute_ipsae,
        get_chain_ids,
        detect_peptide_chain,
        compute_dssp_binder,
        compute_cms,
        get_peptide_pose_indices,
        _contacts_summary,
        contacts_to_rows,
    )
    from contact_map import (
        compute_contacts,
        compute_hbonds,
        get_residues,
    )
    import pyrosetta
except ImportError as e:
    print(f"ERROR: could not import from af3_design_stats.py or contact_map.py: {e}")
    print("Ensure both files are in the same directory or on PYTHONPATH.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# AF3 directory discovery (handles {design}_{timestamp} naming)
# ─────────────────────────────────────────────────────────────────────────────

def find_af3_design_dir(af3_root: str, design: str) -> str | None:
    """
    Find the AF3 output directory for a design.
    Matches {af3_root}/{design}_*/ (timestamped) or {af3_root}/{design}/.
    Returns the first match sorted alphabetically (most recent timestamp last).
    """
    # Timestamped pattern: design_YYYYMMDD_HHMMSS
    matches = sorted(glob.glob(os.path.join(af3_root, f'{design}_*/')))
    if matches:
        return matches[-1].rstrip('/')   # most recent
    # Exact match
    exact = os.path.join(af3_root, design)
    if os.path.isdir(exact):
        return exact
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Kabsch Ca RMSD (same as align_af2_monomer_to_af3.py)
# ─────────────────────────────────────────────────────────────────────────────

def _get_ca_coords(pdb_path: str, chain_id: str) -> np.ndarray:
    """Extract Ca coordinates for a given chain from a PDB file."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('s', pdb_path)
    coords = []
    for model in struct:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if res.id[0] != ' ':
                    continue
                if 'CA' in res:
                    coords.append(res['CA'].get_vector().get_array())
        break
    return np.array(coords, dtype=np.float64) if coords else np.empty((0, 3))


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Kabsch superposition of P (mobile) onto Q (reference).
    Returns (rmsd, R, t) — P_aligned = P @ R.T + t.
    """
    assert P.shape == Q.shape
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H  = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d  = np.linalg.det(Vt.T @ U.T)
    R  = Vt.T @ np.diag([1., 1., d]) @ U.T
    rmsd = float(np.sqrt(((Pc @ R.T - Qc) ** 2).sum() / len(P)))
    t    = Q.mean(axis=0) - P.mean(axis=0) @ R.T
    return rmsd, R, t


def compute_ca_rmsds(
    msa_pdb: str,
    nomsa_pdb: str,
    mhc_chain:    str = 'B',
    binder_chain: str = 'A',
    pep_chain:    str = 'C',
) -> dict:
    """
    Align no-MSA structure onto MSA structure using MHC chain Ca,
    then report binder Ca RMSD and peptide Ca RMSD in that common frame.

    Steps:
      1. Extract MHC Ca from both structures
      2. Kabsch-align no-MSA MHC onto MSA MHC → get R, t
      3. Transform no-MSA binder and peptide Ca using same R, t
      4. Compute RMSD vs MSA binder and peptide Ca
    """
    result = {
        'mhc_ca_rmsd':    np.nan,
        'binder_ca_rmsd': np.nan,
        'pep_ca_rmsd':    np.nan,
        'n_mhc_ca':       0,
        'n_binder_ca':    0,
        'n_pep_ca':       0,
        'rmsd_notes':     '',
    }

    # MHC Ca — reference frame
    msa_mhc   = _get_ca_coords(msa_pdb,   mhc_chain)
    nomsa_mhc = _get_ca_coords(nomsa_pdb, mhc_chain)

    if len(msa_mhc) == 0 or len(nomsa_mhc) == 0:
        result['rmsd_notes'] = f'MHC chain {mhc_chain} Ca not found'
        return result

    n_mhc = min(len(msa_mhc), len(nomsa_mhc))
    if len(msa_mhc) != len(nomsa_mhc):
        result['rmsd_notes'] = (f'MHC Ca length mismatch: msa={len(msa_mhc)} '
                                f'nomsa={len(nomsa_mhc)}, aligning over {n_mhc}')
    result['n_mhc_ca'] = n_mhc

    mhc_rmsd, R, t = kabsch(nomsa_mhc[:n_mhc], msa_mhc[:n_mhc])
    result['mhc_ca_rmsd'] = round(mhc_rmsd, 4)

    # Binder Ca RMSD in MHC-aligned frame
    msa_bind   = _get_ca_coords(msa_pdb,   binder_chain)
    nomsa_bind = _get_ca_coords(nomsa_pdb, binder_chain)
    if len(msa_bind) > 0 and len(nomsa_bind) > 0:
        n_b = min(len(msa_bind), len(nomsa_bind))
        result['n_binder_ca'] = n_b
        nomsa_bind_aligned = nomsa_bind[:n_b] @ R.T + t
        diff = nomsa_bind_aligned - msa_bind[:n_b]
        result['binder_ca_rmsd'] = round(
            float(np.sqrt((diff ** 2).sum() / n_b)), 4
        )
    else:
        result['rmsd_notes'] += f' | binder chain {binder_chain} Ca not found'

    # Peptide Ca RMSD in MHC-aligned frame
    msa_pep   = _get_ca_coords(msa_pdb,   pep_chain)
    nomsa_pep = _get_ca_coords(nomsa_pdb, pep_chain)
    if len(msa_pep) > 0 and len(nomsa_pep) > 0:
        n_p = min(len(msa_pep), len(nomsa_pep))
        result['n_pep_ca'] = n_p
        nomsa_pep_aligned = nomsa_pep[:n_p] @ R.T + t
        diff = nomsa_pep_aligned - msa_pep[:n_p]
        result['pep_ca_rmsd'] = round(
            float(np.sqrt((diff ** 2).sum() / n_p)), 4
        )
    else:
        result['rmsd_notes'] += f' | peptide chain {pep_chain} Ca not found'

    result['rmsd_notes'] = result['rmsd_notes'].strip(' |')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Score one condition (msa or nomsa) for one design
# ─────────────────────────────────────────────────────────────────────────────

METRICS = [
    'ipsae_binder_peptide', 'ipsae_n_contacts',
    'cms_hotspot_total', 'cms_peptide_total',
    'n_contacts_hotspot_total_all', 'n_contacts_hotspot_total_polar',
    'n_all_contacts', 'n_polar_contacts', 'n_hbonds',
    'ss_helix_pct', 'ss_loop_pct', 'n_binder_res',
]


def score_condition(
    design: str,
    af3_dir: str,
    relaxed_pdb: str,
    target_info: dict,
    condition: str,          # 'msa' or 'nomsa'
) -> dict:
    """
    Compute all comparison metrics for one design under one MSA condition.
    Returns a flat dict with condition prefix on each metric key.
    """
    peptide_len       = target_info['peptide_len']
    hotspot_positions = target_info['hotspot_positions']
    prefix = f'{condition}_'

    record = {
        'design':    design,
        'condition': condition,
    }

    # ── ipSAE from PAE ──────────────────────────────────────────────────────
    conf_path = find_top_seed_confidences(af3_dir)
    if conf_path:
        try:
            pae, tok = load_af3_pae(conf_path)
            if pae is not None:
                unique_ch  = list(dict.fromkeys(tok))
                pep_ch_pae = unique_ch[2] if len(unique_ch) >= 3 else unique_ch[-1]
                ipsae      = compute_ipsae(pae, tok,
                                           binder_chain='A',
                                           peptide_chain=pep_ch_pae)
                record[f'{prefix}ipsae_binder_peptide'] = ipsae['ipsae_binder_peptide']
                record[f'{prefix}ipsae_n_contacts']     = ipsae['ipsae_n_contacts']
                record[f'{prefix}mean_pae_bp']          = ipsae['mean_pae_bp']
        except Exception as e:
            print(f"    WARNING [{condition}] ipSAE failed: {e}")

    if not os.path.exists(relaxed_pdb):
        print(f"    WARNING [{condition}] relaxed PDB not found: {relaxed_pdb}")
        return record

    # ── Load pose ────────────────────────────────────────────────────────────
    try:
        pose = pyrosetta.pose_from_pdb(relaxed_pdb)
    except Exception as e:
        print(f"    WARNING [{condition}] pose load failed: {e}")
        return record

    chain_ids = get_chain_ids(pose)
    try:
        pep_chain = detect_peptide_chain(chain_ids)
    except ValueError as e:
        print(f"    WARNING [{condition}] chain detection: {e}")
        return record

    # ── DSSP ─────────────────────────────────────────────────────────────────
    try:
        dssp = compute_dssp_binder(pose, binder_chain='A')
        record[f'{prefix}ss_helix_pct'] = dssp['ss_helix_pct']
        record[f'{prefix}ss_loop_pct']  = dssp['ss_loop_pct']
        record[f'{prefix}n_binder_res'] = dssp['n_binder_res']
    except Exception as e:
        print(f"    WARNING [{condition}] DSSP failed: {e}")

    # ── CMS ──────────────────────────────────────────────────────────────────
    try:
        cms = compute_cms(pose, binder_chain='A', pep_chain=pep_chain,
                          pep_len=peptide_len, hotspot_positions=hotspot_positions)
        record[f'{prefix}cms_hotspot_total'] = cms['cms_hotspot_total']
        record[f'{prefix}cms_peptide_total'] = cms['cms_peptide_total']
    except Exception as e:
        print(f"    WARNING [{condition}] CMS failed: {e}")

    # ── Contacts ─────────────────────────────────────────────────────────────
    try:
        binder_res_list = get_residues(relaxed_pdb, 'A')
        pep_res_list    = get_residues(relaxed_pdb, pep_chain)
        all_cts, polar_cts, hydro_cts = compute_contacts(binder_res_list, pep_res_list)
        has_H     = any(a.element == 'H'
                        for res in binder_res_list for a in res.get_atoms())
        hbond_cts = compute_hbonds(binder_res_list, pep_res_list) if has_H else []
        ct_summary = _contacts_summary(
            binder_res_list, pep_res_list,
            all_cts, polar_cts, hydro_cts, hbond_cts,
            has_H, peptide_len, hotspot_positions,
        )
        record[f'{prefix}n_all_contacts']                = ct_summary['n_all_contacts']
        record[f'{prefix}n_polar_contacts']              = ct_summary['n_polar_contacts']
        record[f'{prefix}n_hbonds']                      = ct_summary['n_hbonds']
        record[f'{prefix}n_contacts_hotspot_total_all']  = ct_summary['n_contacts_hotspot_total_all']
        record[f'{prefix}n_contacts_hotspot_total_polar'] = ct_summary['n_contacts_hotspot_total_polar']
    except Exception as e:
        print(f"    WARNING [{condition}] contacts failed: {e}")

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--af3_msa_dir',      required=True,
        help='AF3 output root dir for MSA runs (e.g. .../af3/)')
    parser.add_argument('--af3_nomsa_dir',    required=True,
        help='AF3 output root dir for no-MSA runs (e.g. .../af3_nomsa/)')
    parser.add_argument('--relaxed_msa_dir',  required=True,
        help='FastRelax relaxed PDB dir for MSA runs')
    parser.add_argument('--relaxed_nomsa_dir', required=True,
        help='FastRelax relaxed PDB dir for no-MSA runs')
    parser.add_argument('--targets_csv',      required=True,
        help='design_epitopes.csv with peptide_len and cms_hotspot_positions')
    parser.add_argument('--out_dir',          required=True)
    parser.add_argument('--designs',          nargs='*',
        help='Designs to compare (default: all with no-MSA relaxed PDB)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    targets = load_targets(args.targets_csv)
    init_pyrosetta()

    # Discover designs
    if args.designs:
        designs = args.designs
    else:
        pdbs    = glob.glob(os.path.join(args.relaxed_nomsa_dir, '*_relaxed.pdb'))
        designs = sorted(
            os.path.basename(p).replace('_relaxed.pdb', '') for p in pdbs
        )
    print(f"Comparing {len(designs)} designs (MSA vs no-MSA)\n")

    raw_records  = []
    rmsd_records = []

    for idx, design in enumerate(designs):
        print(f"[{idx+1}/{len(designs)}] {design}")
        info = targets.get(design, {
            'peptide_len':       9,
            'hotspot_positions': [],
        })

        # Find AF3 dirs
        msa_af3_dir   = find_af3_design_dir(args.af3_msa_dir,   design)
        nomsa_af3_dir = find_af3_design_dir(args.af3_nomsa_dir, design)

        if not msa_af3_dir:
            print(f"  WARNING: MSA AF3 dir not found for {design}")
        if not nomsa_af3_dir:
            print(f"  WARNING: no-MSA AF3 dir not found for {design}")

        msa_pdb   = os.path.join(args.relaxed_msa_dir,   f'{design}_relaxed.pdb')
        nomsa_pdb = os.path.join(args.relaxed_nomsa_dir, f'{design}_relaxed.pdb')

        # Score both conditions
        for condition, af3_dir, relaxed_pdb in [
            ('msa',   msa_af3_dir,   msa_pdb),
            ('nomsa', nomsa_af3_dir, nomsa_pdb),
        ]:
            if af3_dir is None:
                raw_records.append({'design': design, 'condition': condition})
                continue
            print(f"  [{condition}] af3_dir: {os.path.basename(af3_dir)}")
            rec = score_condition(design, af3_dir, relaxed_pdb, info, condition)
            raw_records.append(rec)

        # Ca RMSD — MHC-aligned
        print(f"  [rmsd] aligning no-MSA onto MSA via MHC Ca...")
        if os.path.exists(msa_pdb) and os.path.exists(nomsa_pdb):
            try:
                rmsd = compute_ca_rmsds(msa_pdb, nomsa_pdb)
                rmsd['design'] = design
                rmsd_records.append(rmsd)
                print(f"    MHC={rmsd['mhc_ca_rmsd']:.3f} A  "
                      f"binder={rmsd['binder_ca_rmsd']:.3f} A  "
                      f"pep={rmsd['pep_ca_rmsd']:.3f} A")
            except Exception as e:
                print(f"    WARNING: RMSD failed: {e}")
                rmsd_records.append({'design': design, 'rmsd_notes': str(e)})
        else:
            missing = []
            if not os.path.exists(msa_pdb):   missing.append('MSA PDB')
            if not os.path.exists(nomsa_pdb): missing.append('no-MSA PDB')
            rmsd_records.append({'design': design,
                                  'rmsd_notes': f'missing: {", ".join(missing)}'})
        print()

    # ── Save raw TSV ──────────────────────────────────────────────────────────
    df_raw = pd.DataFrame(raw_records)
    raw_path = os.path.join(args.out_dir, 'msa_nomsa_raw.tsv')
    df_raw.to_csv(raw_path, sep='\t', index=False)
    print(f"Saved -> {raw_path}")

    # ── Save RMSD TSV ─────────────────────────────────────────────────────────
    df_rmsd = pd.DataFrame(rmsd_records)
    col_order = ['design', 'mhc_ca_rmsd', 'binder_ca_rmsd', 'pep_ca_rmsd',
                 'n_mhc_ca', 'n_binder_ca', 'n_pep_ca', 'rmsd_notes']
    df_rmsd = df_rmsd[[c for c in col_order if c in df_rmsd.columns]]
    rmsd_path = os.path.join(args.out_dir, 'msa_nomsa_rmsd.tsv')
    df_rmsd.to_csv(rmsd_path, sep='\t', index=False)
    print(f"Saved -> {rmsd_path}")

    # ── Compute delta TSV (nomsa − msa) ──────────────────────────────────────
    metric_cols = [
        'ipsae_binder_peptide', 'ipsae_n_contacts', 'mean_pae_bp',
        'cms_hotspot_total', 'cms_peptide_total',
        'n_contacts_hotspot_total_all', 'n_contacts_hotspot_total_polar',
        'n_all_contacts', 'n_polar_contacts', 'n_hbonds',
        'ss_helix_pct', 'ss_loop_pct', 'n_binder_res',
    ]

    delta_records = []
    for design in designs:
        msa_row   = df_raw[(df_raw['design'] == design) &
                           (df_raw['condition'] == 'msa')]
        nomsa_row = df_raw[(df_raw['design'] == design) &
                           (df_raw['condition'] == 'nomsa')]
        if msa_row.empty or nomsa_row.empty:
            delta_records.append({'design': design})
            continue

        delta = {'design': design}
        for m in metric_cols:
            msa_col   = f'msa_{m}'
            nomsa_col = f'nomsa_{m}'
            if msa_col in df_raw.columns and nomsa_col in df_raw.columns:
                msa_val   = msa_row[msa_col].values[0]
                nomsa_val = nomsa_row[nomsa_col].values[0]
                delta[f'msa_{m}']   = msa_val
                delta[f'nomsa_{m}'] = nomsa_val
                try:
                    delta[f'delta_{m}'] = round(
                        float(nomsa_val) - float(msa_val), 4
                    )
                except (TypeError, ValueError):
                    delta[f'delta_{m}'] = np.nan
        delta_records.append(delta)

    df_delta = pd.DataFrame(delta_records)

    # Column order: design, then per metric: msa_X, nomsa_X, delta_X
    ordered_cols = ['design']
    for m in metric_cols:
        for prefix in ['msa_', 'nomsa_', 'delta_']:
            col = f'{prefix}{m}'
            if col in df_delta.columns:
                ordered_cols.append(col)
    df_delta = df_delta[[c for c in ordered_cols if c in df_delta.columns]]

    delta_path = os.path.join(args.out_dir, 'msa_nomsa_delta.tsv')
    df_delta.to_csv(delta_path, sep='\t', index=False)
    print(f"Saved -> {delta_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n-- Delta summary (nomsa - msa, positive = nomsa better) --")
    print(f"{'Design':<20} {'dipsae':>8} {'dn_hs_all':>10} "
          f"{'dcms_hs':>9} {'dbinder_rmsd':>13} {'dpep_rmsd':>10}")
    print('-' * 75)

    rmsd_lookup = df_rmsd.set_index('design').to_dict('index') \
                  if not df_rmsd.empty else {}

    for _, row in df_delta.iterrows():
        d          = row['design']
        d_ipsae    = row.get('delta_ipsae_binder_peptide', np.nan)
        d_hs_all   = row.get('delta_n_contacts_hotspot_total_all', np.nan)
        d_cms_hs   = row.get('delta_cms_hotspot_total', np.nan)
        rmsd_info  = rmsd_lookup.get(d, {})
        b_rmsd     = rmsd_info.get('binder_ca_rmsd', np.nan)
        p_rmsd     = rmsd_info.get('pep_ca_rmsd', np.nan)

        def _fmt(v, fmt='.3f'):
            return f'{v:{fmt}}' if pd.notna(v) else 'N/A'

        print(f"{d:<20} {_fmt(d_ipsae):>8} {_fmt(d_hs_all, '.1f'):>10} "
              f"{_fmt(d_cms_hs, '.1f'):>9} {_fmt(b_rmsd):>13} {_fmt(p_rmsd):>10}")


if __name__ == '__main__':
    main()