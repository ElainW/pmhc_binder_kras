"""
compute_rmsd.py

Computes Cα RMSD between AF2 initial guess PDB and pMHC fold complex PDB
for each of the passing designs.

Reads pre-split PDBs from split_pmhc_fold_pdbs.py:
  {split_dir}/{design}_complex.pdb  — chain A = binder, chain B = MHC+peptide

Superimposition strategy (Kabsch SVD):
  - Fixed (reference): AF2 initial guess, chain B (MHC+peptide Cα)
  - Mobile:            pMHC fold complex, chain B (MHC+peptide Cα)
  - Same rotation applied to chain A (binder) to get RMSD_A

Output PDB per design (in {out_dir}/superposed_pdbs/):
  Chain A : AF2 init guess binder     (reference frame, original coordinates)
  Chain B : AF2 init guess MHC+pep    (reference frame, original coordinates)
  Chain Z : pMHC fold binder          (mobile, transformed into AF2 init guess frame)

This lets you open the PDB in PyMOL and immediately see whether the
pMHC fold binder (chain Z) overlays the AF2 init guess binder (chain A).

Usage:
    python compute_rmsd.py \
        --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
        --af2_pdb_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
        --split_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/ \
        --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2
"""

from __future__ import annotations
import io
import os
import argparse

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select

DELTA_PAE_CUTOFF = -0.5


# ── Structure helpers ─────────────────────────────────────────────────────────

def load_structure(pdb_path: str, sid: str = 's'):
    parser = PDBParser(QUIET=True)
    return parser.get_structure(sid, pdb_path)


def get_ca_atoms(pdb_path: str, chain_id: str) -> list:
    """Return ordered list of Cα atoms from a named chain."""
    struct = load_structure(pdb_path)
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


def get_ca_coords(atoms: list) -> np.ndarray:
    return np.array([a.get_vector().get_array() for a in atoms], dtype=np.float64)


# ── Kabsch SVD superposition ──────────────────────────────────────────────────

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Superpose P (mobile) onto Q (fixed/reference) via Kabsch SVD.
    Returns (rmsd, R, t) such that P_aligned = P @ R.T + t.
    Handles reflection via determinant sign correction.
    """
    assert P.shape == Q.shape, f"Shape mismatch: {P.shape} vs {Q.shape}"
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)
    H   = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    d   = np.linalg.det(Vt.T @ U.T)          # reflection correction
    R   = Vt.T @ np.diag([1., 1., d]) @ U.T
    rmsd = float(np.sqrt(((P_c @ R.T - Q_c) ** 2).sum() / len(P)))
    t    = Q.mean(axis=0) - P.mean(axis=0) @ R.T
    return rmsd, R, t


def apply_rmsd_only(P: np.ndarray, Q: np.ndarray,
                     R: np.ndarray, t: np.ndarray) -> float:
    """Apply precomputed R, t to P and compute RMSD against Q."""
    if len(P) == 0 or len(Q) == 0:
        return np.nan
    n    = min(len(P), len(Q))
    moved = P[:n] @ R.T + t
    diff  = Q[:n] - moved
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


# ── Transform all atoms of a loaded structure in-place ────────────────────────

def transform_structure(struct, R: np.ndarray, t: np.ndarray):
    """Apply R, t to every atom in struct: x_new = x @ R.T + t"""
    for model in struct:
        for chain in model:
            for res in chain:
                for atom in res:
                    old = atom.get_vector().get_array()
                    atom.set_coord(old @ R.T + t)


# ── Write combined PDB ────────────────────────────────────────────────────────

class _AcceptAll(Select):
    def accept_residue(self, r): return 1


def write_combined_pdb(af2_struct, fold_struct_transformed, out_path: str):
    """
    Write a single PDB:
      Chains A, B : AF2 init guess (reference frame, original coordinates)
      Chain Z     : pMHC fold binder (transformed into AF2 frame)

    fold_struct_transformed must already have had transform_structure() applied.
    Only chain A of the fold complex is written (the binder; chain B is MHC
    which by construction is already aligned to AF2 chain B).
    """
    io_obj = PDBIO()

    def _to_lines(struct) -> list[str]:
        buf = io.StringIO()
        io_obj.set_structure(struct)
        io_obj.save(buf, _AcceptAll())
        return buf.getvalue().splitlines()

    af2_lines  = _to_lines(af2_struct)
    fold_lines = _to_lines(fold_struct_transformed)

    # AF2 init guess: keep all ATOM/HETATM/TER lines as-is (chains A, B)
    af2_atom_lines = [l for l in af2_lines
                      if l.startswith('ATOM') or l.startswith('HETATM')
                      or l.startswith('TER')]

    # pMHC fold: keep only chain A (binder), relabel to chain Z
    fold_z_lines = []
    for line in fold_lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            if len(line) > 21 and line[21] == 'A':
                line = line[:21] + 'Z' + line[22:]
                fold_z_lines.append(line)
        elif line.startswith('TER') and len(line) > 21 and line[21] == 'A':
            fold_z_lines.append(line[:21] + 'Z' + line[22:])

    with open(out_path, 'w') as f:
        f.write("REMARK  AF2 init guess (chains A+B) + pMHC fold binder superposed (chain Z)\n")
        f.write("REMARK  Chain A: AF2 init guess binder (reference frame)\n")
        f.write("REMARK  Chain B: AF2 init guess MHC+peptide (reference frame)\n")
        f.write("REMARK  Chain Z: pMHC fold binder (transformed into AF2 init guess frame)\n")
        for line in af2_atom_lines:
            f.write(line + '\n')
        for line in fold_z_lines:
            f.write(line + '\n')
        f.write("END\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_scores', required=True)
    parser.add_argument('--af2_pdb_dir',   required=True)
    parser.add_argument('--split_dir',     required=True,
                        help='Directory containing {design}_complex.pdb')
    parser.add_argument('--out_dir',       required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    superposed_dir = os.path.join(args.out_dir, 'superposed_pdb')
    os.makedirs(superposed_dir, exist_ok=True)

    df      = pd.read_csv(args.merged_scores, sep='\t')
    passing = df[df['delta_pae_int_peptide'] < DELTA_PAE_CUTOFF].copy()
    designs = passing['design'].tolist()
    print(f"Computing RMSD for {len(designs)} designs...")

    records = []
    for design in designs:
        af2_path     = os.path.join(args.af2_pdb_dir, design + '.pdb')
        complex_path = os.path.join(args.split_dir, f'{design}_complex.pdb')

        if not os.path.exists(af2_path):
            print(f"  ERROR: AF2 PDB not found: {af2_path}")
            continue
        if not os.path.exists(complex_path):
            print(f"  ERROR: complex PDB not found: {complex_path}")
            continue

        af2_ca_A  = get_ca_atoms(af2_path,     'A')
        af2_ca_B  = get_ca_atoms(af2_path,     'B')
        fold_ca_A = get_ca_atoms(complex_path, 'A')
        fold_ca_B = get_ca_atoms(complex_path, 'B')

        n_A = min(len(af2_ca_A), len(fold_ca_A))
        n_B = min(len(af2_ca_B), len(fold_ca_B))

        if n_A == 0 or n_B == 0:
            print(f"  ERROR: empty chain for {design} (n_A={n_A}, n_B={n_B})")
            continue

        notes = 'ok'
        if len(af2_ca_B) != len(fold_ca_B):
            notes = (f'chain B length mismatch: AF2={len(af2_ca_B)} '
                     f'fold={len(fold_ca_B)}; aligned over {n_B} residues')
            print(f"  WARNING {design}: {notes}")

        # ── Kabsch: superpose fold chain B (mobile) onto AF2 chain B (fixed) ──
        af2_B_coords  = get_ca_coords(af2_ca_B[:n_B])
        fold_B_coords = get_ca_coords(fold_ca_B[:n_B])
        rmsd_B, R, t  = kabsch_rmsd(fold_B_coords, af2_B_coords)

        # ── RMSD_A: apply same transform to fold chain A, compare with AF2 A ──
        af2_A_coords  = get_ca_coords(af2_ca_A[:n_A])
        fold_A_coords = get_ca_coords(fold_ca_A[:n_A])
        rmsd_A = apply_rmsd_only(fold_A_coords, af2_A_coords, R, t)

        # ── Total RMSD over all Cα ─────────────────────────────────────────────
        fold_A_moved  = fold_A_coords @ R.T + t
        fold_B_moved  = fold_B_coords @ R.T + t
        af2_all       = np.vstack([af2_A_coords,  af2_B_coords])
        fold_all_moved= np.vstack([fold_A_moved,  fold_B_moved])
        rmsd_total    = float(np.sqrt(
            ((af2_all - fold_all_moved) ** 2).sum(axis=1).mean()
        ))

        print(f"  {design}: RMSD_B={rmsd_B:.3f} Å  RMSD_A={rmsd_A:.3f} Å  "
              f"RMSD_total={rmsd_total:.3f} Å")

        # ── Write superposed PDB ───────────────────────────────────────────────
        superposed_pdb = os.path.join(superposed_dir,
                                       f'{design}_superposed.pdb')
        try:
            af2_struct  = load_structure(af2_path,     sid='af2')
            fold_struct = load_structure(complex_path, sid='fold')
            transform_structure(fold_struct, R, t)
            write_combined_pdb(af2_struct, fold_struct, superposed_pdb)
        except Exception as e:
            print(f"    WARNING: could not write superposed PDB: {e}")
            superposed_pdb = f'ERROR: {e}'

        records.append({
            'design':           design,
            'n_ca_A':           n_A,
            'n_ca_B':           n_B,
            'RMSD_total':       round(rmsd_total, 3),
            'RMSD_A':           round(rmsd_A,     3),
            'RMSD_B':           round(rmsd_B,     3),
            'af2_pdb':          af2_path,
            'fold_complex_pdb': complex_path,
            'superposed_pdb':   superposed_pdb,
            'notes':            notes,
        })

    out_df   = pd.DataFrame(records)
    out_path = os.path.join(args.out_dir, 'rmsd_af2_vs_pmhcfold.tsv')
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved TSV  → {out_path}")
    print(f"Saved PDBs → {superposed_dir}/")

    print(f"\n── RMSD summary ({len(out_df)} designs) ──")
    for col in ['RMSD_total', 'RMSD_A', 'RMSD_B']:
        if col in out_df and out_df[col].notna().any():
            print(f"  {col}: mean={out_df[col].mean():.3f}  "
                  f"median={out_df[col].median():.3f}  "
                  f"max={out_df[col].max():.3f}")

    high_rmsd_A = out_df[out_df['RMSD_A'] > 2.0]
    if len(high_rmsd_A):
        print(f"\n── RMSD_A > 2.0 Å (binder shifted relative to MHC) ──")
        print(high_rmsd_A[['design', 'RMSD_total', 'RMSD_A',
                            'RMSD_B']].to_string(index=False))


if __name__ == '__main__':
    main()