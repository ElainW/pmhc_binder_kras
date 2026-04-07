"""
charge_cysteine_audit.py

Reads pre-split pMHC fold PDBs from split_pmhc_fold_pdbs.py:
  {split_dir}/{design}_chainA.pdb  — binder (chain A, renumbered from 1)
  {split_dir}/{design}_chainB.pdb  — MHC + peptide (chain B, renumbered from 1,
                                       peptide = last 9 residues)

- net_charge uses physiological formula: R + K + 0.1*H - D - E
- Unpaired Cys: flag for redesign if NOT within 4Å of pMHC interface
- Charge-failing designs: output interface residues (within 4Å of pMHC)
  as fixed positions that should NOT be redesigned in ProteinMPNN
- Fails loudly if split PDBs are missing (no fallback)

Usage:
    python charge_cysteine_audit.py \
        --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
        --split_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/ \
        --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/
"""

import os
import argparse
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

DELTA_PAE_CUTOFF   = -0.5
DISULFIDE_DIST     = 2.3   # Å SG–SG
INTERFACE_DIST     = 4.0   # Å Cβ (or Cα) to any pMHC heavy atom
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

def analyze_cysteines(binder_residues: list,
                      pmhc_coords: np.ndarray) -> list[dict]:
    """
    For each CYS in the binder, determine:
      - paired_disulfide  : SG–SG < 2.3 Å  → keep
      - unpaired_interface: unpaired but Cβ within 4 Å of pMHC → keep
      - unpaired_non_interface: not in interface → flag for redesign
    """
    cys_info = {}
    for res in binder_residues:
        if res.resname != 'CYS':
            continue
        sg       = res['SG'].get_vector().get_array() if 'SG' in res else None
        cb       = res['CB'].get_vector().get_array() if 'CB' in res else None
        ca       = res['CA'].get_vector().get_array() if 'CA' in res else None
        cb_or_ca = cb if cb is not None else ca
        cys_info[res.id[1]] = {'sg': sg, 'cb': cb_or_ca}

    if not cys_info:
        return []

    resnums = list(cys_info.keys())
    paired  = set()
    for i, r1 in enumerate(resnums):
        sg1 = cys_info[r1]['sg']
        if sg1 is None:
            continue
        for r2 in resnums[i + 1:]:
            sg2 = cys_info[r2]['sg']
            if sg2 is None:
                continue
            if np.linalg.norm(sg1 - sg2) < DISULFIDE_DIST:
                paired.add(r1)
                paired.add(r2)

    results = []
    for resnum, cys in cys_info.items():
        is_paired        = resnum in paired
        is_interface     = False
        min_dist_to_pmhc = np.nan

        if not is_paired and cys['cb'] is not None and len(pmhc_coords) > 0:
            dists            = np.linalg.norm(pmhc_coords - cys['cb'], axis=1)
            min_dist_to_pmhc = float(dists.min())
            is_interface     = min_dist_to_pmhc < INTERFACE_DIST

        if is_paired:
            status, action = 'paired_disulfide', 'keep'
        elif is_interface:
            status, action = 'unpaired_interface', 'keep'
        else:
            status = ('unpaired_non_interface' if not np.isnan(min_dist_to_pmhc)
                      else 'unpaired_pmhc_unknown')
            action = 'redesign_position'

        results.append({
            'resnum':           resnum,
            'status':           status,
            'action':           action,
            'min_dist_to_pmhc': (round(min_dist_to_pmhc, 3)
                                 if not np.isnan(min_dist_to_pmhc) else 'na'),
        })
    return results


# ── Interface residue detection ───────────────────────────────────────────────

def get_interface_residues(binder_residues: list,
                           pmhc_coords: np.ndarray,
                           dist_cutoff: float = INTERFACE_DIST) -> list[int]:
    """Return binder residue numbers (1-indexed) within dist_cutoff of any pMHC heavy atom."""
    if len(pmhc_coords) == 0:
        return []
    interface = []
    for res in binder_residues:
        if res.id[0] != ' ':
            continue
        for atom in res:
            if atom.element == 'H':
                continue
            coord = atom.get_vector().get_array()
            if np.linalg.norm(pmhc_coords - coord, axis=1).min() <= dist_cutoff:
                interface.append(res.id[1])
                break
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

        pmhc_coords = get_heavy_atom_coords(pmhc_res)

        # ── Sequence and basic metrics ────────────────────────────────────
        seq    = get_binder_seq(binder_res)
        charge = net_charge(seq)
        n_cys  = seq.count('C')

        # ── Cysteine analysis ─────────────────────────────────────────────
        cys_analysis         = analyze_cysteines(binder_res, pmhc_coords)
        n_paired             = sum(1 for c in cys_analysis if c['status'] == 'paired_disulfide')
        n_unpaired_interface = sum(1 for c in cys_analysis if c['status'] == 'unpaired_interface')
        n_unpaired_non_iface = sum(1 for c in cys_analysis if c['action'] == 'redesign_position')
        cys_redesign_resnums = [c['resnum'] for c in cys_analysis
                                if c['action'] == 'redesign_position']

        # ── Interface residues for charge-failing designs ─────────────────
        # Binder positions within 4 Å of pMHC — must be FIXED during MPNN
        # redesign to preserve the binding interface geometry
        flag_charge             = charge > CHARGE_THRESH
        interface_fixed_resnums = []
        if flag_charge:
            interface_fixed_resnums = get_interface_residues(
                binder_res, pmhc_coords, dist_cutoff=INTERFACE_DIST
            )

        flag_cys = n_unpaired_non_iface > 0

        records.append({
            'design':                   design,
            'sequence':                 seq,
            'length':                   len(seq),
            'net_charge':               round(charge, 2),
            'n_cys_total':              n_cys,
            'n_cys_paired':             n_paired,
            'n_cys_unpaired_interface': n_unpaired_interface,
            'n_cys_unpaired_non_iface': n_unpaired_non_iface,
            'cys_redesign_resnums':     str(cys_redesign_resnums),
            'cys_detail':               str(cys_analysis),
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

    print(f"\n── Cysteine summary ──")
    print(f"  No Cys:                    {(out_df['n_cys_total'] == 0).sum()}/{len(out_df)}")
    print(f"  Paired only:               "
          f"{((out_df['n_cys_paired'] > 0) & (out_df['n_cys_unpaired_non_iface'] == 0)).sum()}"
          f"/{len(out_df)}")
    print(f"  Unpaired interface (keep): {(out_df['n_cys_unpaired_interface'] > 0).sum()}/{len(out_df)}")
    print(f"  Unpaired non-interface:    {out_df['flag_cys_redesign'].sum()}/{len(out_df)} → redesign")

    print(f"\n── Pass all ──")
    print(f"  {out_df['pass_all'].sum()}/{len(out_df)}")

    print(f"\n── Charge-failing designs (interface positions to fix in MPNN) ──")
    charge_fail = out_df[out_df['flag_charge_redesign']][
        ['design', 'net_charge', 'n_interface_fixed', 'interface_fixed_resnums']
    ]
    print(charge_fail.to_string(index=False))

    print(f"\n── Designs needing Cys position redesign ──")
    cys_fail = out_df[out_df['flag_cys_redesign']][
        ['design', 'n_cys_total', 'n_cys_unpaired_non_iface', 'cys_redesign_resnums']
    ]
    print(cys_fail.to_string(index=False))

    out_path = os.path.join(args.out_dir, 'charge_cysteine_audit.tsv')
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()