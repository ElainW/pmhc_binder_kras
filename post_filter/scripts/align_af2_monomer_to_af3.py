"""
align_af2_monomer_to_af3.py
===========================
Aligns AF2 monomer binder predictions to the AF3 complex binder chain (chain A)
and reports Ca RMSD per design.

AF2 monomer path convention:
  /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer/author_{design}/
  The best (lowest pLDDT rank) PDB is selected automatically by scanning
  *_model_1.pdb, *_unrelaxed_rank_001*.pdb, *.pdb (first sorted match).

AF3 complex path convention:
  {af3_out_dir}/{design}/{design}_model.cif
  Chain A = binder.

Alignment strategy:
  - Extract binder chain A Ca atoms from AF3 CIF (reference)
  - Extract all Ca atoms from AF2 monomer PDB (mobile)
  - Sequence-length check: warn if lengths differ (no alignment attempted)
  - Superpose mobile onto reference via SVD (Kabsch algorithm)
  - Report Ca RMSD over all aligned residues

No PyRosetta required — uses Biopython only.

Output TSV columns:
  design, af3_binder_len, af2_monomer_len, rmsd_ca,
  af3_cif, af2_pdb, notes

Usage:
    python align_af2_monomer_to_af3.py \
        --af3_out_dir  /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3/ \
        --af2_mono_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer/ \
        --out_tsv      /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af2_af3_rmsd.tsv \
        --save_superposed \
        --superposed_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af2_af3_superimposed_pdbs/

    # Specific designs:
    python align_af2_monomer_to_af3.py \
        --af3_out_dir ... --af2_mono_dir ... --out_tsv ... \
        --designs ctnnb1-15 gp100-3 mart1-3

    # Save superposed AF2 PDBs (binder only, in AF3 frame):
    python align_af2_monomer_to_af3.py \
        --af3_out_dir ... --af2_mono_dir ... --out_tsv ... \
        --save_superposed --superposed_dir /path/to/superposed/
"""

from __future__ import annotations

import os
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from Bio import PDB as bpdb
    from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


# ─────────────────────────────────────────────────────────────────────────────
# Structure loading
# ─────────────────────────────────────────────────────────────────────────────

def load_structure(path: str, structure_id: str = 'struct'):
    """Load PDB or mmCIF into a BioPython Structure object."""
    path = str(path)
    if path.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(structure_id, path)


def get_ca_coords(structure, chain_id: str | None = None) -> tuple[np.ndarray, list]:
    """
    Extract Ca coordinates from a BioPython Structure.
    If chain_id is given, restrict to that chain.
    Returns (coords array [N,3], list of residue labels for debugging).
    """
    coords = []
    labels = []
    model  = structure[0]  # first model

    chains = [model[chain_id]] if chain_id and chain_id in model \
             else list(model.get_chains())

    for chain in chains:
        for res in chain.get_residues():
            if res.id[0] != ' ':   # skip HETATM / water
                continue
            if 'CA' in res:
                coords.append(res['CA'].get_vector().get_array())
                labels.append(f"{chain.id}:{res.resname}{res.id[1]}")

    return np.array(coords, dtype=np.float64), labels


# ─────────────────────────────────────────────────────────────────────────────
# Kabsch RMSD
# ─────────────────────────────────────────────────────────────────────────────

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Kabsch superposition of P (mobile) onto Q (reference).
    Both arrays shape (N, 3).
    Returns (rmsd, rotation_matrix R, translation t)
    such that P_aligned = P @ R.T + t
    """
    assert P.shape == Q.shape, f"Shape mismatch: {P.shape} vs {Q.shape}"

    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)

    H   = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    P_rot = P_c @ R.T
    diff  = P_rot - Q_c
    rmsd  = float(np.sqrt((diff ** 2).sum() / len(P)))

    t = Q.mean(axis=0) - P.mean(axis=0) @ R.T
    return rmsd, R, t


def apply_transform(
    coords: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Apply rotation R and translation t to coordinate array."""
    return coords @ R.T + t


# ─────────────────────────────────────────────────────────────────────────────
# AF3 / AF2 file discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_af3_cif(af3_out_dir: str, design: str) -> str | None:
    """
    Find AF3 top-ranked model CIF.
    Primary:  {af3_out_dir}/{design}/{design}_model.cif
    Fallback: ranking_scores.csv -> top seed -> seed-{s}_sample-{n}/model.cif
    """
    design_dir = os.path.join(af3_out_dir, design)
    top_cif    = os.path.join(design_dir, f'{design}_model.cif')
    if os.path.exists(top_cif):
        return top_cif

    ranking = os.path.join(design_dir, 'ranking_scores.csv')
    if os.path.exists(ranking):
        df  = pd.read_csv(ranking)
        top = df.sort_values('ranking_score', ascending=False).iloc[0]
        seed, sample = int(top['seed']), int(top['sample'])
        seed_cif = os.path.join(
            design_dir, f'seed-{seed}_sample-{sample}', 'model.cif'
        )
        if os.path.exists(seed_cif):
            return seed_cif
    return None


def find_af2_monomer_pdb(af2_mono_dir: str, design: str) -> str | None:
    """
    Find best AF2 monomer PDB for a design.
    Directory: {af2_mono_dir}/author_{design}/
    Priority:
      1. *_unrelaxed_rank_001*.pdb   (AF2 standard best model)
      2. *_model_1.pdb
      3. first *.pdb sorted alphabetically
    """
    design_dir = os.path.join(af2_mono_dir, f'author_{design}')
    if not os.path.isdir(design_dir):
        # Also try without 'author_' prefix
        design_dir = os.path.join(af2_mono_dir, design)
        if not os.path.isdir(design_dir):
            # Also try without the dates at the end
            design_dir = os.path.join(af2_mono_dir, f"author_{design.split('_')[0]}")
            if not os.path.isdir(design_dir):
                return None

    for pattern in [
        '*_unrelaxed_rank_001*.pdb',
        '*_unrelaxed_rank_1*.pdb',
        '*_model_1.pdb',
        '*.pdb',
    ]:
        matches = sorted(glob.glob(os.path.join(design_dir, pattern)))
        if matches:
            return matches[0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Optional: save superposed PDB
# ─────────────────────────────────────────────────────────────────────────────

class AllSelect(Select):
    def accept_residue(self, residue):
        return 1


def save_superposed_pdb(
    af2_struct,
    R: np.ndarray,
    t: np.ndarray,
    out_path: str,
):
    """
    Apply Kabsch transform to all atoms of af2_struct and write to PDB.
    Modifies coordinates in-place (on a copy is not done — caller should
    reload structure if original coords needed afterwards).
    """
    model = af2_struct[0]
    for chain in model.get_chains():
        for res in chain.get_residues():
            for atom in res.get_atoms():
                old = atom.get_vector().get_array()
                new = old @ R.T + t
                atom.set_coord(new)

    io = PDBIO()
    io.set_structure(af2_struct)
    io.save(out_path, AllSelect())


# ─────────────────────────────────────────────────────────────────────────────
# Per-design alignment
# ─────────────────────────────────────────────────────────────────────────────

def align_design(
    design: str,
    af3_out_dir: str,
    af2_mono_dir: str,
    save_superposed: bool = False,
    superposed_dir: str  = '',
) -> dict:
    record = {'design': design}

    # Find files
    af3_cif = find_af3_cif(af3_out_dir, design)
    af2_pdb = find_af2_monomer_pdb(af2_mono_dir, design)

    record['af3_cif'] = af3_cif or 'NOT FOUND'
    record['af2_pdb'] = af2_pdb or 'NOT FOUND'

    if not af3_cif:
        record['notes'] = 'AF3 CIF not found'
        print(f"    ERROR: AF3 CIF not found for {design}")
        return record
    if not af2_pdb:
        record['notes'] = 'AF2 monomer PDB not found'
        print(f"    ERROR: AF2 monomer PDB not found for {design}")
        return record

    # Load structures
    try:
        af3_struct = load_structure(af3_cif,  structure_id=f'{design}_af3')
        af2_struct = load_structure(af2_pdb,  structure_id=f'{design}_af2')
    except Exception as e:
        record['notes'] = f'Load error: {e}'
        print(f"    ERROR loading structures: {e}")
        return record

    # Extract Ca coords
    # AF3: chain A = binder only
    # AF2 monomer: single chain (all Ca)
    try:
        ref_coords, ref_labels = get_ca_coords(af3_struct, chain_id='A')
        mob_coords, mob_labels = get_ca_coords(af2_struct, chain_id=None)
    except Exception as e:
        record['notes'] = f'Ca extraction error: {e}'
        print(f"    ERROR extracting Ca: {e}")
        return record

    af3_len = len(ref_coords)
    af2_len = len(mob_coords)
    record['af3_binder_len'] = af3_len
    record['af2_monomer_len'] = af2_len

    if af3_len == 0:
        record['notes'] = 'No Ca found in AF3 chain A'
        print(f"    ERROR: No Ca in AF3 chain A")
        return record
    if af2_len == 0:
        record['notes'] = 'No Ca found in AF2 monomer'
        print(f"    ERROR: No Ca in AF2 monomer")
        return record

    # Length check
    if af3_len != af2_len:
        note = (f'Length mismatch: AF3={af3_len} AF2={af2_len} — '
                f'aligning over min({af3_len},{af2_len}) N-terminal residues')
        record['notes'] = note
        print(f"    WARNING: {note}")
        n   = min(af3_len, af2_len)
        ref_coords = ref_coords[:n]
        mob_coords = mob_coords[:n]
    else:
        record['notes'] = 'ok'

    # Kabsch superposition
    try:
        rmsd, R, t = kabsch_rmsd(mob_coords, ref_coords)
        record['rmsd_ca'] = round(rmsd, 4)
        print(f"    Ca RMSD: {rmsd:.3f} A  "
              f"(AF3 binder len={af3_len}, AF2 monomer len={af2_len})")
    except Exception as e:
        record['notes'] = f'RMSD error: {e}'
        print(f"    ERROR computing RMSD: {e}")
        return record

    # Optionally save superposed AF2 PDB
    if save_superposed and superposed_dir:
        try:
            out_pdb = os.path.join(superposed_dir, f'{design}_af2mono_on_af3.pdb')
            save_superposed_pdb(af2_struct, R, t, out_pdb)
            record['superposed_pdb'] = out_pdb
            print(f"    Saved superposed PDB -> {out_pdb}")
        except Exception as e:
            print(f"    WARNING: could not save superposed PDB: {e}")

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--af3_out_dir',  required=True,
        help='Root AF3 output dir; each design has its own subdir')
    parser.add_argument('--af2_mono_dir', required=True,
        help='Root AF2 monomer dir; design subdirs named author_{design}/')
    parser.add_argument('--out_tsv',      required=True,
        help='Output TSV path')
    parser.add_argument('--designs', nargs='*',
        help='Design names (default: all subdirs in af3_out_dir)')
    parser.add_argument('--save_superposed', action='store_true',
        help='Save superposed AF2 monomer PDBs (in AF3 binder frame)')
    parser.add_argument('--superposed_dir', default='',
        help='Directory for superposed PDBs (required if --save_superposed)')
    args = parser.parse_args()

    if not HAS_BIOPYTHON:
        raise ImportError(
            "Biopython is required: pip install biopython --break-system-packages"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_tsv)), exist_ok=True)

    if args.save_superposed:
        if not args.superposed_dir:
            parser.error('--superposed_dir required when --save_superposed is set')
        os.makedirs(args.superposed_dir, exist_ok=True)

    # Discover designs
    if args.designs:
        designs = args.designs
    else:
        designs = sorted(
            d for d in os.listdir(args.af3_out_dir)
            if os.path.isdir(os.path.join(args.af3_out_dir, d))
        )

    print(f"Aligning {len(designs)} designs: AF2 monomer -> AF3 binder (chain A)\n")

    records = []
    for idx, design in enumerate(designs):
        print(f"[{idx+1}/{len(designs)}] {design}")
        rec = align_design(
            design          = design,
            af3_out_dir     = args.af3_out_dir,
            af2_mono_dir    = args.af2_mono_dir,
            save_superposed = args.save_superposed,
            superposed_dir  = args.superposed_dir,
        )
        records.append(rec)

    df = pd.DataFrame(records)

    col_order = ['design', 'af3_binder_len', 'af2_monomer_len',
                 'rmsd_ca', 'af3_cif', 'af2_pdb', 'notes']
    if 'superposed_pdb' in df.columns:
        col_order.append('superposed_pdb')
    present = [c for c in col_order if c in df.columns]
    df      = df[present]

    df.to_csv(args.out_tsv, sep='\t', index=False)
    print(f"\nSaved -> {args.out_tsv}")

    # Summary
    if 'rmsd_ca' in df.columns and df['rmsd_ca'].notna().any():
        v = df['rmsd_ca'].dropna()
        print(f"\n-- RMSD summary ({len(v)}/{len(df)} designs) --")
        print(f"  mean : {v.mean():.3f} A")
        print(f"  median: {v.median():.3f} A")
        print(f"  min  : {v.min():.3f} A  ({df.loc[v.idxmin(), 'design']})")
        print(f"  max  : {v.max():.3f} A  ({df.loc[v.idxmax(), 'design']})")
        print(f"  std  : {v.std():.3f} A")
        print(f"\n  Per design:")
        for _, row in df.sort_values('rmsd_ca').iterrows():
            rmsd = row.get('rmsd_ca', np.nan)
            note = row.get('notes', '')
            if pd.notna(rmsd):
                print(f"    {row['design']:20s}  {rmsd:.3f} A")
            else:
                print(f"    {row['design']:20s}  FAILED  ({note})")


if __name__ == '__main__':
    main()