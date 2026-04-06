"""
charge_cysteine_audit.py  (updated)

Net charge uses physiological formula: R + K + 0.1*H - D - E
Unpaired Cys: flag for redesign only if NOT within 4Å of pMHC interface
              (interface = any atom of chain B in the pMHC fold PDB)

Usage:
    python charge_cysteine_audit.py \
        --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
        --pdb_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
        --pmhc_fold_dir /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/outputs/r2/pmhc_fold_on_runs \
        --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

DELTA_PAE_CUTOFF     = -0.5
DISULFIDE_DIST       = 2.3   # Å — SG-SG distance for disulfide
INTERFACE_DIST       = 4.0   # Å — Cβ to any pMHC atom for "interface Cys"
CHARGE_THRESH        = -2.0  # net charge threshold


def net_charge(seq: str) -> float:
    """Physiological net charge at pH 7.4."""
    return (seq.count('R') + seq.count('K') + 0.1 * seq.count('H')
            - seq.count('D') - seq.count('E'))


def get_binder_seq(pdb_path: str, chain_id: str = 'A') -> str:
    three_to_one = {
        'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
        'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
        'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
        'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
    }
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('s', pdb_path)
    seq = ''
    for model in struct:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if res.id[0] == ' ':
                    seq += three_to_one.get(res.resname, 'X')
    return seq


def find_pmhc_fold_pdb(design: str, pmhc_fold_dir: str) -> str | None:
    pattern = os.path.join(
        pmhc_fold_dir, 'sample_df_chunk_*',
        f'{design}_A1101_VVGADGVGK_unrelaxed_model_1.pdb'
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def get_all_heavy_atom_coords(struct, chain_id: str) -> np.ndarray:
    """Return (N, 3) array of all non-H atom coordinates from a chain."""
    coords = []
    for model in struct:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if res.id[0] != ' ':
                    continue
                for atom in res:
                    if atom.element != 'H':
                        coords.append(atom.get_vector().get_array())
    return np.array(coords) if coords else np.empty((0, 3))


def analyze_cysteines(af2_pdb: str, fold_pdb: str | None) -> list[dict]:
    """
    For each Cys in chain A of the AF2 init guess PDB:
      1. Check if paired (disulfide): SG-SG < 2.3 Å with any other Cys
      2. Check if interface: Cβ < 4 Å from any chain B atom in pMHC fold PDB
      3. Classify: paired | interface_unpaired | non_interface_unpaired

    Returns list of per-Cys dicts.
    """
    parser = PDBParser(QUIET=True)
    struct_af2 = parser.get_structure('af2', af2_pdb)

    # Collect all Cys in chain A with their SG and CB coords
    cys_residues = {}  # resnum -> {'sg': coords, 'cb': coords}
    for model in struct_af2:
        for chain in model:
            if chain.id != 'A':
                continue
            for res in chain:
                if res.id[0] != ' ' or res.resname != 'CYS':
                    continue
                sg = res['SG'].get_vector().get_array() if 'SG' in res else None
                cb = res['CB'].get_vector().get_array() if 'CB' in res else None
                ca = res['CA'].get_vector().get_array() if 'CA' in res else None
                cys_residues[res.id[1]] = {'sg': sg, 'cb': cb if cb is not None else ca}

    if not cys_residues:
        return []

    # Check disulfide pairing
    resnums = list(cys_residues.keys())
    paired = set()
    for i, r1 in enumerate(resnums):
        sg1 = cys_residues[r1]['sg']
        if sg1 is None:
            continue
        for r2 in resnums[i+1:]:
            sg2 = cys_residues[r2]['sg']
            if sg2 is None:
                continue
            if np.linalg.norm(sg1 - sg2) < DISULFIDE_DIST:
                paired.add(r1)
                paired.add(r2)

    # Get pMHC chain B heavy atom coords from fold PDB
    pmhc_coords = np.empty((0, 3))
    if fold_pdb and os.path.exists(fold_pdb):
        struct_fold = parser.get_structure('fold', fold_pdb)
        pmhc_coords = get_all_heavy_atom_coords(struct_fold, 'B')

    results = []
    for resnum, cys in cys_residues.items():
        is_paired = resnum in paired

        # Check interface proximity using Cβ
        is_interface = False
        min_dist_to_pmhc = np.nan
        if not is_paired and cys['cb'] is not None and len(pmhc_coords) > 0:
            dists = np.linalg.norm(pmhc_coords - cys['cb'], axis=1)
            min_dist_to_pmhc = float(dists.min())
            is_interface = min_dist_to_pmhc < INTERFACE_DIST

        if is_paired:
            status = 'paired_disulfide'
            action = 'keep'
        elif is_interface:
            status = 'unpaired_interface'
            action = 'keep'   # may be making contacts — keep for now
        else:
            status = 'unpaired_non_interface'
            action = 'redesign_position'  # flag for ProteinMPNN redesign

        results.append({
            'resnum': resnum,
            'status': status,
            'action': action,
            'min_dist_to_pmhc': round(min_dist_to_pmhc, 3) if not np.isnan(min_dist_to_pmhc) else np.nan,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_scores',  required=True)
    parser.add_argument('--pdb_dir',         required=True)
    parser.add_argument('--pmhc_fold_dir',   required=True)
    parser.add_argument('--out_dir',          required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.merged_scores, sep='\t')
    passing = df[df['delta_pae_int_peptide'] < DELTA_PAE_CUTOFF].copy()
    print(f"Designs passing Δ pAE < {DELTA_PAE_CUTOFF}: {len(passing)}")

    records = []
    for _, row in passing.iterrows():
        design   = row['design']
        af2_path = os.path.join(args.pdb_dir, design + '.pdb')
        fold_path = find_pmhc_fold_pdb(design, args.pmhc_fold_dir)

        if not os.path.exists(af2_path):
            print(f"  WARNING: AF2 PDB not found for {design}")
            continue

        seq     = get_binder_seq(af2_path)
        charge  = net_charge(seq)
        n_met   = seq.count('M')
        n_cys   = seq.count('C')

        # Per-Cys analysis
        cys_analysis = analyze_cysteines(af2_path, fold_path)
        n_paired               = sum(1 for c in cys_analysis if c['status'] == 'paired_disulfide')
        n_unpaired_interface   = sum(1 for c in cys_analysis if c['status'] == 'unpaired_interface')
        n_unpaired_non_iface   = sum(1 for c in cys_analysis if c['status'] == 'unpaired_non_interface')
        cys_redesign_resnums   = [c['resnum'] for c in cys_analysis if c['action'] == 'redesign_position']

        # Flags
        flag_charge    = charge > CHARGE_THRESH
        flag_cys       = n_unpaired_non_iface > 0

        records.append({
            'design':                   design,
            'sequence':                 seq,
            'length':                   len(seq),
            'net_charge':               round(charge, 2),
            'n_met':                    n_met,
            'n_cys_total':              n_cys,
            'n_cys_paired':             n_paired,
            'n_cys_unpaired_interface': n_unpaired_interface,
            'n_cys_unpaired_non_iface': n_unpaired_non_iface,
            'cys_redesign_resnums':     str(cys_redesign_resnums),
            'cys_detail':               str(cys_analysis),
            'flag_charge_redesign':     flag_charge,
            'flag_cys_redesign':        flag_cys,
            'pass_charge':              not flag_charge,
            'pass_cys':                 not flag_cys,
            'pass_all':                 not flag_charge and not flag_cys,
        })

    out_df = pd.DataFrame(records)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Charge summary (threshold: net charge ≤ {CHARGE_THRESH}) ──")
    print(f"  Pass:  {out_df['pass_charge'].sum()}/{len(out_df)}")
    print(f"  Fail:  {out_df['flag_charge_redesign'].sum()}/{len(out_df)} "
          f"→ need ProteinMPNN redesign")

    print(f"\n── Cysteine summary ──")
    print(f"  No Cys:                       {(out_df['n_cys_total'] == 0).sum()}/{len(out_df)}")
    print(f"  Paired only (disulfide):      {((out_df['n_cys_paired'] > 0) & (out_df['n_cys_unpaired_non_iface'] == 0)).sum()}/{len(out_df)}")
    print(f"  Unpaired interface (keep):    {(out_df['n_cys_unpaired_interface'] > 0).sum()}/{len(out_df)}")
    print(f"  Unpaired non-interface:       {out_df['flag_cys_redesign'].sum()}/{len(out_df)} "
          f"→ need redesign")

    print(f"\n── Methionine ──")
    print(f"  Designs with Met:             {(out_df['n_met'] > 0).sum()}/{len(out_df)}")

    print(f"\n── Pass all (no redesign needed) ──")
    print(f"  {out_df['pass_all'].sum()}/{len(out_df)}")

    # Designs needing charge redesign
    print(f"\n── Designs needing charge redesign ──")
    charge_fail = out_df[out_df['flag_charge_redesign']][
        ['design', 'net_charge', 'sequence']]
    print(charge_fail.to_string(index=False))

    # Designs needing Cys redesign
    print(f"\n── Designs needing Cys position redesign ──")
    cys_fail = out_df[out_df['flag_cys_redesign']][
        ['design', 'n_cys_total', 'n_cys_unpaired_non_iface', 'cys_redesign_resnums']]
    print(cys_fail.to_string(index=False))

    out_path = os.path.join(args.out_dir, 'charge_cysteine_audit.tsv')
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()