"""
compute_rmsd.py

Computes Cα RMSD between AF2 initial guess PDB and pMHC fold _unrelaxed_model_1.pdb
for each of the 49 passing designs. Reports RMSD_total, RMSD_A (binder), RMSD_B (pMHC).

Superimposition is performed on chain B Cα atoms (MHC groove anchor),
then RMSD is computed on chain A and chain B separately.

Usage:
    python compute_rmsd.py \
        --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
        --af2_pdb_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
        --pmhc_fold_dir /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/outputs/r2/pmhc_fold_on_runs \
        --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer

DELTA_PAE_CUTOFF = -0.5
PEPTIDE_LEN = 9

def find_pmhc_fold_pdb(design: str, pmhc_fold_dir: str) -> str | None:
    pattern = os.path.join(
        pmhc_fold_dir,
        'sample_df_chunk_*',
        f'{design}_A1101_VVGADGVGK_unrelaxed_model_1.pdb'
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def get_ca_atoms(struct, chain_id: str, skip_hetatm: bool = True):
    """Return ordered list of Cα atoms from a chain."""
    atoms = []
    for model in struct:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if skip_hetatm and res.id[0] != ' ':
                    continue
                if 'CA' in res:
                    atoms.append(res['CA'])
    return atoms

def compute_rmsd_after_superimpose(fixed_atoms, moving_atoms):
    """Superimpose moving onto fixed using provided atom lists, return RMSD."""
    if len(fixed_atoms) == 0 or len(moving_atoms) == 0:
        return np.nan
    n = min(len(fixed_atoms), len(moving_atoms))
    fixed_atoms  = fixed_atoms[:n]
    moving_atoms = moving_atoms[:n]
    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)
    return float(sup.rms)

def compute_rmsd_given_rotation(fixed_atoms, moving_atoms, rotran):
    """Apply a precomputed rotation/translation and compute RMSD."""
    rot, tran = rotran
    if len(fixed_atoms) == 0 or len(moving_atoms) == 0:
        return np.nan
    n = min(len(fixed_atoms), len(moving_atoms))
    fixed_coords  = np.array([a.get_vector().get_array() for a in fixed_atoms[:n]])
    moving_coords = np.array([a.get_vector().get_array() for a in moving_atoms[:n]])
    # Apply rotation and translation
    moved = moving_coords @ rot.T + tran
    diff = fixed_coords - moved
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_scores',  required=True)
    parser.add_argument('--af2_pdb_dir',    required=True)
    parser.add_argument('--pmhc_fold_dir',  required=True)
    parser.add_argument('--out_dir',         required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.merged_scores, sep='\t')
    passing = df[df['delta_pae_int_peptide'] < DELTA_PAE_CUTOFF].copy()
    designs = passing['design'].tolist()
    print(f"Computing RMSD for {len(designs)} designs...")

    pdb_parser = PDBParser(QUIET=True)
    records = []

    for design in designs:
        af2_path   = os.path.join(args.af2_pdb_dir, design + '.pdb')
        fold_path  = find_pmhc_fold_pdb(design, args.pmhc_fold_dir)

        if not os.path.exists(af2_path):
            print(f"  WARNING: AF2 init guess PDB not found for {design}")
            continue
        if fold_path is None:
            print(f"  WARNING: pMHC fold PDB not found for {design}")
            continue

        # Load both structures
        af2_struct  = pdb_parser.get_structure('af2',  af2_path)
        fold_struct = pdb_parser.get_structure('fold', fold_path)

        # Get Cα atoms per chain
        # AF2 init guess: chain A = binder, chain B = MHC+peptide (merged)
        # pMHC fold:      chain A = binder, chain B = MHC+peptide (merged)
        af2_ca_A  = get_ca_atoms(af2_struct,  'A')
        af2_ca_B  = get_ca_atoms(af2_struct,  'B')
        fold_ca_A = get_ca_atoms(fold_struct, 'A')
        fold_ca_B = get_ca_atoms(fold_struct, 'B')

        n_A = min(len(af2_ca_A), len(fold_ca_A))
        n_B = min(len(af2_ca_B), len(fold_ca_B))

        if n_A == 0 or n_B == 0:
            print(f"  WARNING: empty chain for {design}")
            continue

        # Step 1: Superimpose on chain B (MHC anchor) — this is the fixed reference
        sup_B = Superimposer()
        sup_B.set_atoms(af2_ca_B[:n_B], fold_ca_B[:n_B])
        rmsd_B = float(sup_B.rms)
        rotran = (sup_B.rotran[0], sup_B.rotran[1])

        # Step 2: Apply same rotation to chain A and compute RMSD
        rot, tran = rotran
        af2_A_coords  = np.array([a.get_vector().get_array() for a in af2_ca_A[:n_A]])
        fold_A_coords = np.array([a.get_vector().get_array() for a in fold_ca_A[:n_A]])
        fold_A_moved  = fold_A_coords @ rot.T + tran
        diff_A = af2_A_coords - fold_A_moved
        rmsd_A = float(np.sqrt((diff_A ** 2).sum(axis=1).mean()))

        # Step 3: Total RMSD — all Cα atoms combined after chain B superimposition
        af2_all_coords  = np.vstack([af2_A_coords,
                                      np.array([a.get_vector().get_array()
                                                for a in af2_ca_B[:n_B]])])
        fold_A_moved_full = fold_A_moved
        fold_B_coords = np.array([a.get_vector().get_array() for a in fold_ca_B[:n_B]])
        fold_B_moved  = fold_B_coords @ rot.T + tran
        fold_all_moved = np.vstack([fold_A_moved_full, fold_B_moved])
        diff_total = af2_all_coords - fold_all_moved
        rmsd_total = float(np.sqrt((diff_total ** 2).sum(axis=1).mean()))

        records.append({
            'design':       design,
            'n_ca_A':       n_A,
            'n_ca_B':       n_B,
            'RMSD_total':   round(rmsd_total, 3),
            'RMSD_A':       round(rmsd_A, 3),   # binder RMSD after MHC superimposition
            'RMSD_B':       round(rmsd_B, 3),   # MHC RMSD (superimposition anchor)
            'af2_pdb':      af2_path,
            'fold_pdb':     fold_path,
        })

    out_df = pd.DataFrame(records)
    out_path = os.path.join(args.out_dir, 'rmsd_af2_vs_pmhcfold.tsv')
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved RMSD table to {out_path}")

    print(f"\n── RMSD summary ({len(out_df)} designs) ──")
    for col in ['RMSD_total', 'RMSD_A', 'RMSD_B']:
        print(f"  {col}: mean={out_df[col].mean():.3f}, "
              f"median={out_df[col].median():.3f}, "
              f"max={out_df[col].max():.3f}")

    # Flag designs where binder moved a lot relative to MHC
    high_rmsd_A = out_df[out_df['RMSD_A'] > 2.0]
    if len(high_rmsd_A):
        print(f"\n── Designs with RMSD_A > 2.0 Å (binder shifted relative to MHC) ──")
        print(high_rmsd_A[['design', 'RMSD_total', 'RMSD_A', 'RMSD_B']].to_string(index=False))

if __name__ == '__main__':
    main()
