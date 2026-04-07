"""
compute_rmsd.py

Computes Cα RMSD between AF2 initial guess PDB and pMHC fold complex PDB
for each of the passing designs.

Reads pre-split PDBs from split_pmhc_fold_pdbs.py:
  {split_dir}/{design}_complex.pdb  — full complex, chain A = binder, chain B = MHC+peptide

Superimposition is performed on chain B Cα atoms (MHC groove anchor),
then RMSD is computed on chain A (binder) and chain B (MHC) separately.

Reports: RMSD_total, RMSD_A (binder), RMSD_B (MHC+peptide, superimposition anchor).

Usage:
    python compute_rmsd.py \
        --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
        --af2_pdb_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
        --split_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/ \
        --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2
"""

import os
import argparse
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer

DELTA_PAE_CUTOFF = -0.5
PEPTIDE_LEN      = 9


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_ca_atoms(pdb_path: str, chain_id: str) -> list:
    """Return ordered list of Cα atoms from a named chain."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('s', pdb_path)
    atoms  = []
    for model in struct:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if res.id[0] != ' ':
                    continue
                if 'CA' in res:
                    atoms.append(res['CA'])
        break
    return atoms


def apply_rotation_and_rmsd(fixed_atoms: list, moving_atoms: list,
                              rot: np.ndarray, tran: np.ndarray) -> float:
    """Apply a precomputed rotation/translation and compute RMSD."""
    if not fixed_atoms or not moving_atoms:
        return np.nan
    n             = min(len(fixed_atoms), len(moving_atoms))
    fixed_coords  = np.array([a.get_vector().get_array() for a in fixed_atoms[:n]])
    moving_coords = np.array([a.get_vector().get_array() for a in moving_atoms[:n]])
    moved         = moving_coords @ rot.T + tran
    diff          = fixed_coords - moved
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_scores', required=True)
    parser.add_argument('--af2_pdb_dir',   required=True)
    parser.add_argument('--split_dir',     required=True,
                        help='Directory containing {design}_complex.pdb from '
                             'split_pmhc_fold_pdbs.py')
    parser.add_argument('--out_dir',       required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.merged_scores, sep='\t')
    passing = df[df['delta_pae_int_peptide'] < DELTA_PAE_CUTOFF].copy()
    designs = passing['design'].tolist()
    print(f"Computing RMSD for {len(designs)} designs...")

    records = []
    for design in designs:
        af2_path     = os.path.join(args.af2_pdb_dir, design + '.pdb')
        complex_path = os.path.join(args.split_dir, f'{design}_complex.pdb')

        if not os.path.exists(af2_path):
            print(f"  ERROR: AF2 init guess PDB not found for {design}: {af2_path}")
            continue
        if not os.path.exists(complex_path):
            print(f"  ERROR: complex PDB not found for {design}: {complex_path}")
            continue

        # AF2 init guess: chain A = binder, chain B = MHC+peptide (merged)
        # pMHC fold complex: chain A = binder, chain B = MHC+peptide (merged)
        # Both use the same chain convention — direct comparison is valid
        af2_ca_A  = get_ca_atoms(af2_path,     'A')
        af2_ca_B  = get_ca_atoms(af2_path,     'B')
        fold_ca_A = get_ca_atoms(complex_path, 'A')
        fold_ca_B = get_ca_atoms(complex_path, 'B')

        n_A = min(len(af2_ca_A), len(fold_ca_A))
        n_B = min(len(af2_ca_B), len(fold_ca_B))

        if n_A == 0 or n_B == 0:
            print(f"  ERROR: empty chain for {design} "
                  f"(n_A={n_A}, n_B={n_B}) — skipping")
            continue

        # Step 1: Superimpose on chain B (MHC anchor) — fixed = AF2, moving = fold
        sup_B = Superimposer()
        sup_B.set_atoms(af2_ca_B[:n_B], fold_ca_B[:n_B])
        rmsd_B = float(sup_B.rms)
        rot    = sup_B.rotran[0]   # (3, 3) rotation matrix
        tran   = sup_B.rotran[1]   # (3,)   translation vector

        # Step 2: Apply the same MHC-anchored rotation to chain A and compute RMSD
        rmsd_A = apply_rotation_and_rmsd(af2_ca_A[:n_A], fold_ca_A[:n_A], rot, tran)

        # Step 3: Total RMSD — all Cα atoms combined under the same chain B superimposition
        af2_A_coords  = np.array([a.get_vector().get_array() for a in af2_ca_A[:n_A]])
        af2_B_coords  = np.array([a.get_vector().get_array() for a in af2_ca_B[:n_B]])
        fold_A_coords = np.array([a.get_vector().get_array() for a in fold_ca_A[:n_A]])
        fold_B_coords = np.array([a.get_vector().get_array() for a in fold_ca_B[:n_B]])

        fold_A_moved  = fold_A_coords @ rot.T + tran
        fold_B_moved  = fold_B_coords @ rot.T + tran

        af2_all  = np.vstack([af2_A_coords,  af2_B_coords])
        fold_all = np.vstack([fold_A_moved,  fold_B_moved])
        diff     = af2_all - fold_all
        rmsd_total = float(np.sqrt((diff ** 2).sum(axis=1).mean()))

        records.append({
            'design':    design,
            'n_ca_A':    n_A,
            'n_ca_B':    n_B,
            'RMSD_total': round(rmsd_total, 3),
            'RMSD_A':    round(rmsd_A, 3),   # binder RMSD after MHC superimposition
            'RMSD_B':    round(rmsd_B, 3),   # MHC RMSD (superimposition anchor)
            'af2_pdb':   af2_path,
            'fold_complex_pdb': complex_path,
        })

    out_df   = pd.DataFrame(records)
    out_path = os.path.join(args.out_dir, 'rmsd_af2_vs_pmhcfold.tsv')
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved RMSD table to {out_path}")

    print(f"\n── RMSD summary ({len(out_df)} designs) ──")
    for col in ['RMSD_total', 'RMSD_A', 'RMSD_B']:
        print(f"  {col}: mean={out_df[col].mean():.3f}, "
              f"median={out_df[col].median():.3f}, "
              f"max={out_df[col].max():.3f}")

    # Flag designs where binder shifted substantially relative to MHC
    high_rmsd_A = out_df[out_df['RMSD_A'] > 2.0]
    if len(high_rmsd_A):
        print(f"\n── Designs with RMSD_A > 2.0 Å (binder shifted relative to MHC) ──")
        print(high_rmsd_A[['design', 'RMSD_total', 'RMSD_A', 'RMSD_B']].to_string(index=False))


if __name__ == '__main__':
    main()