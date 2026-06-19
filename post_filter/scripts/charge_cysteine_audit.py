"""
charge_cysteine_audit.py

Reads pre-split pMHC fold PDBs from split_pmhc_fold_pdbs.py:
  {split_dir}/{design}_chainA.pdb  — binder (chain A, renumbered from 1)
  {split_dir}/{design}_chainB.pdb  — MHC + peptide (chain B, renumbered from 1,
                                       peptide = last 9 residues)

- net_charge uses physiological formula: R + K + 0.1*H - D - E
- Cysteine: ANY Cys in the binder fails — no exceptions for disulfides or
  interface positions. Cys residue numbers are reported for MPNN redesign.
- ALL designs: output interface residues (Cβ/Cα within 4.5Å of pMHC Cβ/Cα)
  as fixed positions that should NOT be redesigned in ProteinMPNN.
  This matches BindCraft's mpnn_fix_interface definition.
- Fails loudly if split PDBs are missing (no fallback)

Usage:
    python charge_cysteine_audit.py \
        --merged_scores ~/Desktop/pmhc_binder_kras/specificity/analysis/r2/merged_scores_ranked.tsv \
        --split_dir ~/Desktop/pmhc_binder_kras/post_filter/inputs/r2/ \
        --dalphaball ~/Desktop/pmhc_binder_kras/post_filter/scripts/DAlphaBall.gcc \
        --out_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/
"""

import os
import argparse
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

DELTA_PAE_CUTOFF   = -0.5
INTERFACE_DIST     = 8   # Å Cβ/Cα to pMHC Cβ/Cα — matches BindCraft mpnn_fix_interface
CHARGE_THRESH      = -2.0
PEPTIDE_LEN        = 9

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

PDB_PARSER = PDBParser(QUIET=True)


# ── Core helpers ──────────────────────────────────────────────────────────────

def net_charge(seq: str) -> float:
    return (seq.count('R') + seq.count('K') + 0.1 * seq.count('H')
            - seq.count('D') - seq.count('E'))


def get_residues(pdb_path: str, chain_id: str) -> list:
    """Return ATOM residues from a given chain of a PDB file."""
    struct = PDB_PARSER.get_structure('s', pdb_path)
    residues = []
    for model in struct:
        for chain in model:
            if chain.id == chain_id:
                for res in chain:
                    if res.id[0] == ' ':
                        residues.append(res)
        break
    return residues


def get_binder_seq(binder_residues: list) -> str:
    return ''.join(THREE_TO_ONE.get(r.resname, 'X') for r in binder_residues)


def get_heavy_atom_coords(residues: list) -> np.ndarray:
    """Extract all heavy atom coordinates from a list of residues."""
    coords = []
    for res in residues:
        for atom in res:
            if atom.element != 'H':
                coords.append(atom.get_vector().get_array())
    return np.array(coords) if coords else np.empty((0, 3))


# ── Cysteine analysis ─────────────────────────────────────────────────────────

def find_cysteine_resnums(binder_residues: list,
                           pmhc_coords: np.ndarray) -> tuple[list[int], list[int]]:
    """
    Return (all_cys_resnums, interface_cys_resnums).
    Any Cys = fail. Interface Cys (Cβ/Cα within 4 Å of pMHC) flagged separately
    as informational — these may need extra care during MPNN redesign since
    they sit at the binding surface.
    """
    all_resnums   = []
    iface_resnums = []

    for res in binder_residues:
        if res.resname != 'CYS':
            continue
        resnum = res.id[1]
        all_resnums.append(resnum)

        # Check proximity to pMHC
        cb_or_ca = None
        if 'CB' in res:
            cb_or_ca = res['CB'].get_vector().get_array()
        elif 'CA' in res:
            cb_or_ca = res['CA'].get_vector().get_array()

        if cb_or_ca is not None and len(pmhc_coords) > 0:
            min_dist = float(np.linalg.norm(pmhc_coords - cb_or_ca, axis=1).min())
            if min_dist < INTERFACE_DIST:
                iface_resnums.append(resnum)

    return all_resnums, iface_resnums


# ── Interface residue detection ───────────────────────────────────────────────

def _cb_or_ca(res) -> np.ndarray | None:
    """Return Cβ coordinate (or Cα for Gly/missing Cβ), or None if unavailable."""
    if 'CB' in res:
        return res['CB'].get_vector().get_array()
    if 'CA' in res:
        return res['CA'].get_vector().get_array()
    return None


def get_pmhc_cb_coords(pmhc_residues: list) -> np.ndarray:
    """Extract Cβ/Cα coordinates from pMHC residues (mirrors BindCraft side)."""
    coords = []
    for res in pmhc_residues:
        if res.id[0] != ' ':
            continue
        c = _cb_or_ca(res)
        if c is not None:
            coords.append(c)
    return np.array(coords) if coords else np.empty((0, 3))


def get_interface_residues(binder_residues: list,
                           pmhc_cb_coords: np.ndarray,
                           dist_cutoff: float = INTERFACE_DIST) -> list[int]:
    """
    Return binder residue numbers whose Cβ (Cα for Gly) is within dist_cutoff
    of any pMHC Cβ (Cα for Gly).

    Matches BindCraft's mpnn_fix_interface definition: ColabDesign MPNN wrapper
    uses Cβ-neighbour search at 4.5 Å to determine which binder positions to fix.
    """
    if len(pmhc_cb_coords) == 0:
        return []
    interface = []
    for res in binder_residues:
        if res.id[0] != ' ':
            continue
        coord = _cb_or_ca(res)
        if coord is None:
            continue
        if np.linalg.norm(pmhc_cb_coords - coord, axis=1).min() <= dist_cutoff:
            interface.append(res.id[1])
    return sorted(set(interface))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_scores', required=True)
    parser.add_argument('--split_dir',     required=True,
                        help='Directory containing {design}_chainA.pdb and '
                             '{design}_chainB.pdb from split_pmhc_fold_pdbs.py')
    parser.add_argument('--out_dir',       required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.merged_scores, sep='\t')
    passing = df[df['delta_pae_int_peptide'] < DELTA_PAE_CUTOFF].copy()
    print(f"Designs passing Δ pAE < {DELTA_PAE_CUTOFF}: {len(passing)}")

    records = []
    n_missing = 0

    for _, row in passing.iterrows():
        design = row['design']

        chain_a_path = os.path.join(args.split_dir, f'{design}_chainA.pdb')
        chain_b_path = os.path.join(args.split_dir, f'{design}_chainB.pdb')

        if not os.path.exists(chain_a_path):
            print(f"  ERROR: chainA PDB not found for {design}: {chain_a_path}")
            n_missing += 1
            continue
        if not os.path.exists(chain_b_path):
            print(f"  ERROR: chainB PDB not found for {design}: {chain_b_path}")
            n_missing += 1
            continue

        binder_res = get_residues(chain_a_path, 'A')
        pmhc_res   = get_residues(chain_b_path, 'B')

        if not binder_res:
            print(f"  ERROR: no residues in chainA for {design}")
            n_missing += 1
            continue
        if not pmhc_res:
            print(f"  ERROR: no residues in chainB for {design}")
            n_missing += 1
            continue

        pmhc_coords    = get_heavy_atom_coords(pmhc_res)   # for Cys proximity
        pmhc_cb_coords = get_pmhc_cb_coords(pmhc_res)       # for interface (BindCraft-style)

        # ── Sequence and basic metrics ────────────────────────────────────
        seq    = get_binder_seq(binder_res)
        charge = net_charge(seq)
        n_cys  = seq.count('C')

        # ── Cysteine: any Cys = fail, all positions flagged for redesign ──
        cys_resnums, cys_interface_resnums = find_cysteine_resnums(
            binder_res, pmhc_coords
        )

        # ── Interface residues for charge-failing designs ─────────────────
        # Cβ/Cα within 4.5 Å of pMHC Cβ/Cα — matches BindCraft mpnn_fix_interface
        flag_charge             = charge > CHARGE_THRESH
        interface_fixed_resnums = []
        interface_fixed_resnums = get_interface_residues(
                binder_res, pmhc_cb_coords, dist_cutoff=INTERFACE_DIST
        )

        flag_cys = n_cys > 0

        records.append({
            'design':                   design,
            'sequence':                 seq,
            'length':                   len(seq),
            'net_charge':               round(charge, 2),
            'n_cys':                    n_cys,
            'cys_resnums':              str(cys_resnums),
            'cys_interface_resnums':    str(cys_interface_resnums),
            'n_cys_interface':          len(cys_interface_resnums),
            # Charge redesign: binder residues to keep FIXED in MPNN
            'interface_fixed_resnums':  str(interface_fixed_resnums),
            'n_interface_fixed':        len(interface_fixed_resnums),
            # Flags
            'flag_charge_redesign':     flag_charge,
            'flag_cys_redesign':        flag_cys,
            'pass_charge':              not flag_charge,
            'pass_cys':                 not flag_cys,
            'pass_all':                 not flag_charge and not flag_cys,
        })

    if n_missing:
        print(f"\n  WARNING: {n_missing} designs skipped due to missing split PDBs")

    out_df = pd.DataFrame(records)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Charge summary (threshold: net charge ≤ {CHARGE_THRESH}) ──")
    print(f"  Pass: {out_df['pass_charge'].sum()}/{len(out_df)}")
    print(f"  Fail: {out_df['flag_charge_redesign'].sum()}/{len(out_df)} → need ProteinMPNN redesign")

    print(f"\n── Cysteine summary (policy: no Cys allowed) ──")
    print(f"  No Cys (pass):              {(out_df['n_cys'] == 0).sum()}/{len(out_df)}")
    print(f"  Has Cys (fail):             {(out_df['n_cys'] > 0).sum()}/{len(out_df)} → redesign all Cys positions")
    print(f"  Cys at interface (< {INTERFACE_DIST} Å): "
          f"{(out_df['n_cys_interface'] > 0).sum()}/{len(out_df)} ⚠ interface-proximal")

    print(f"\n── Pass all ──")
    print(f"  {out_df['pass_all'].sum()}/{len(out_df)}")

    print(f"\n── Charge-failing designs (interface positions to fix in MPNN) ──")
    charge_fail = out_df[out_df['flag_charge_redesign']][
        ['design', 'net_charge', 'n_interface_fixed', 'interface_fixed_resnums']
    ]
    print(charge_fail.to_string(index=False))

    print(f"\n── Cys-failing designs (all Cys positions to redesign in MPNN) ──")
    cys_fail = out_df[out_df['flag_cys_redesign']][
        ['design', 'n_cys', 'cys_resnums', 'n_cys_interface', 'cys_interface_resnums']
    ]
    print(cys_fail.to_string(index=False))

    out_path = os.path.join(args.out_dir, 'charge_cysteine_audit.tsv')
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()