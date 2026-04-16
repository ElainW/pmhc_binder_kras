"""
align_af2_monomer_to_af3.py
===========================
Aligns AF2 monomer binder predictions to the AF3 complex binder chain (chain A)
and saves one combined PDB per design containing:

  Chain A : AF3 binder          (reference, original coordinates)
  Chain B : AF3 MHC alpha       (reference, original coordinates)
  Chain C : AF3 peptide         (reference, original coordinates)
  Chain D : AF3 B2M             (reference, if present)
  Chain Z : AF2 monomer binder  (mobile, transformed into AF3 frame)

Open {design}_superposed.pdb in PyMOL to see the AF2 monomer (chain Z)
overlaid on the full AF3 complex (chains A-C/D) in a single file.

AF2 monomer path convention:
  {af2_mono_dir}/author_{design}/
  Best model selected by priority:
    1. *_unrelaxed_rank_001*.pdb
    2. *_unrelaxed_rank_1*.pdb
    3. *_model_1.pdb
    4. first *.pdb alphabetically

AF3 complex path convention:
  {af3_out_dir}/{design}/{design}_model.cif
  Chain A = binder, B = MHC, C = peptide, [D = B2M]

Alignment strategy:
  - Reference: Ca atoms of AF3 chain A (binder)
  - Mobile:    Ca atoms of AF2 monomer (all residues)
  - Kabsch SVD superposition
  - Length mismatch: align over N-terminal min(af3_len, af2_len) residues + warn

No PyRosetta required — Biopython only.

Output TSV columns:
  design, af3_binder_len, af2_monomer_len, rmsd_ca, superposed_pdb,
  af3_cif, af2_pdb, notes

Usage:
    python align_af2_monomer_to_af3.py \
        --af3_out_dir  /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3/ \
        --af2_mono_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer/ \
        --out_tsv      /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af2_af3_rmsd.tsv \
        --superposed_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af2_af3_superimposed_pdb/

    # Specific designs only:
    python align_af2_monomer_to_af3.py \
        --af3_out_dir ... --af2_mono_dir ... --out_tsv ... --superposed_dir ... \
        --designs ctnnb1-15 gp100-3 mart1-3
"""

from __future__ import annotations

import os
import io
import glob
import argparse

import numpy as np
import pandas as pd

try:
    from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select, Structure, Model, Chain, Residue
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


# ─────────────────────────────────────────────────────────────────────────────
# Structure loading
# ─────────────────────────────────────────────────────────────────────────────

def load_structure(path: str, structure_id: str = 'struct'):
    """Load PDB or mmCIF into a BioPython Structure."""
    if path.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(structure_id, path)


def get_ca_coords(structure, chain_id: str | None = None) -> tuple[np.ndarray, list]:
    """
    Extract Ca coordinates. If chain_id given, restrict to that chain.
    Returns (coords [N,3], labels list).
    Skips HETATM records (res.id[0] != ' ').
    """
    coords, labels = [], []
    model  = structure[0]
    chains = ([model[chain_id]] if (chain_id and chain_id in model)
              else list(model.get_chains()))
    for chain in chains:
        for res in chain.get_residues():
            if res.id[0] != ' ':
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
    Superpose P (mobile) onto Q (reference) via Kabsch/SVD.
    Returns (rmsd, R, t) where P_aligned = P @ R.T + t.
    """
    assert P.shape == Q.shape, f"Shape mismatch: {P.shape} vs {Q.shape}"
    P_c  = P - P.mean(axis=0)
    Q_c  = Q - Q.mean(axis=0)
    H    = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    d    = np.linalg.det(Vt.T @ U.T)          # reflection correction
    R    = Vt.T @ np.diag([1., 1., d]) @ U.T
    rmsd = float(np.sqrt(((P_c @ R.T - Q_c) ** 2).sum() / len(P)))
    t    = Q.mean(axis=0) - P.mean(axis=0) @ R.T
    return rmsd, R, t


# ─────────────────────────────────────────────────────────────────────────────
# AF3 / AF2 file discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_af3_cif(af3_out_dir: str, design: str) -> str | None:
    design_dir = os.path.join(af3_out_dir, design)
    top = os.path.join(design_dir, f'{design}_model.cif')
    if os.path.exists(top):
        return top
    ranking = os.path.join(design_dir, 'ranking_scores.csv')
    if os.path.exists(ranking):
        df  = pd.read_csv(ranking)
        row = df.sort_values('ranking_score', ascending=False).iloc[0]
        p   = os.path.join(design_dir,
                           f'seed-{int(row["seed"])}_sample-{int(row["sample"])}',
                           'model.cif')
        if os.path.exists(p):
            return p
    return None


def find_af2_monomer_pdb(af2_mono_dir: str, design: str) -> str | None:
    design_dir = os.path.join(af2_mono_dir, f'author_{design}')
    if not os.path.isdir(design_dir):
        # Also try without 'author_' prefix
        design_dir = os.path.join(af2_mono_dir, design)
        if not os.path.isdir(design_dir):
            # Also try without the dates at the end
            design_dir = os.path.join(af2_mono_dir, f"author_{design.split('_')[0]}")
            if not os.path.isdir(design_dir):
                return None

    for pattern in ['*_unrelaxed_rank_001*.pdb', '*_unrelaxed_rank_1*.pdb',
                    '*_model_1.pdb', '*.pdb']:
        matches = sorted(glob.glob(os.path.join(design_dir, pattern)))
        if matches:
            return matches[0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Apply transform in-place to all atoms of a structure
# ─────────────────────────────────────────────────────────────────────────────

def transform_structure(structure, R: np.ndarray, t: np.ndarray):
    """Rotate + translate all atoms of structure in-place: x_new = x @ R.T + t"""
    for model in structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    old = atom.get_vector().get_array()
                    atom.set_coord(old @ R.T + t)


# ─────────────────────────────────────────────────────────────────────────────
# Write combined PDB: full AF3 complex + transformed AF2 monomer as chain Z
# ─────────────────────────────────────────────────────────────────────────────

class _AcceptAll(Select):
    def accept_residue(self, r): return 1


def write_combined_pdb(
    af3_struct,
    af2_struct_transformed,
    out_path: str,
):
    """
    Write a single PDB containing:
      - All chains from AF3 complex (A=binder, B=MHC, C=peptide, [D=B2M])
        with their original coordinates (reference frame).
      - AF2 monomer (already transformed into AF3 frame) as chain Z.

    Both structures must already carry the correct coordinates before calling.

    Strategy: write each source to an in-memory PDB string, strip END/ENDMDL
    lines, relabel the AF2 monomer chain to Z, then concatenate + write.
    """
    io_obj = PDBIO()

    def _struct_to_pdb_lines(struct) -> list[str]:
        buf = io.StringIO()
        io_obj.set_structure(struct)
        io_obj.save(buf, _AcceptAll())
        return buf.getvalue().splitlines()

    af3_lines = _struct_to_pdb_lines(af3_struct)
    af2_lines = _struct_to_pdb_lines(af2_struct_transformed)

    # Strip terminal records from AF3 block (we'll add one at the very end)
    af3_atom_lines = [l for l in af3_lines
                      if l.startswith('ATOM') or l.startswith('HETATM')
                      or l.startswith('TER')]

    # Relabel AF2 chain to Z in ATOM/HETATM/TER records
    # PDB format: column 22 (0-indexed 21) = chain ID
    af2_atom_lines = []
    for line in af2_lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            line = line[:21] + 'Z' + line[22:]
        elif line.startswith('TER') and len(line) > 21:
            line = line[:21] + 'Z' + line[22:]
        elif line.startswith('END'):
            continue   # strip END/ENDMDL — we add our own
        af2_atom_lines.append(line)

    with open(out_path, 'w') as f:
        f.write("REMARK  AF3 complex (chains A-D) + AF2 monomer superposed (chain Z)\n")
        f.write("REMARK  Chain A: AF3 binder (reference)\n")
        f.write("REMARK  Chain B: AF3 MHC alpha (reference)\n")
        f.write("REMARK  Chain C: AF3 peptide (reference)\n")
        f.write("REMARK  Chain D: AF3 B2M if present (reference)\n")
        f.write("REMARK  Chain Z: AF2 monomer binder (transformed into AF3 frame)\n")
        for line in af3_atom_lines:
            f.write(line + '\n')
        for line in af2_atom_lines:
            f.write(line + '\n')
        f.write("END\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-design alignment
# ─────────────────────────────────────────────────────────────────────────────

def align_design(
    design: str,
    af3_out_dir: str,
    af2_mono_dir: str,
    superposed_dir: str,
) -> dict:
    record = {'design': design}

    af3_cif = find_af3_cif(af3_out_dir, design)
    af2_pdb = find_af2_monomer_pdb(af2_mono_dir, design)
    record['af3_cif'] = af3_cif or 'NOT FOUND'
    record['af2_pdb'] = af2_pdb or 'NOT FOUND'

    if not af3_cif:
        record['notes'] = 'AF3 CIF not found'
        print(f"    ERROR: AF3 CIF not found")
        return record
    if not af2_pdb:
        record['notes'] = 'AF2 monomer PDB not found'
        print(f"    ERROR: AF2 monomer PDB not found")
        return record

    # Load
    try:
        af3_struct = load_structure(af3_cif, structure_id=f'{design}_af3')
        af2_struct = load_structure(af2_pdb, structure_id=f'{design}_af2')
    except Exception as e:
        record['notes'] = f'Load error: {e}'
        print(f"    ERROR: {e}")
        return record

    # Ca coords: AF3 chain A (reference), AF2 all chains (mobile)
    try:
        ref_coords, _ = get_ca_coords(af3_struct, chain_id='A')
        mob_coords, _ = get_ca_coords(af2_struct, chain_id=None)
    except Exception as e:
        record['notes'] = f'Ca extraction error: {e}'
        print(f"    ERROR: {e}")
        return record

    af3_len = len(ref_coords)
    af2_len = len(mob_coords)
    record['af3_binder_len'] = af3_len
    record['af2_monomer_len'] = af2_len

    if af3_len == 0 or af2_len == 0:
        record['notes'] = f'Empty Ca: af3={af3_len} af2={af2_len}'
        print(f"    ERROR: empty Ca coords")
        return record

    # Handle length mismatch — align over shared N-terminal residues
    if af3_len != af2_len:
        n = min(af3_len, af2_len)
        note = (f'Length mismatch AF3={af3_len} AF2={af2_len}; '
                f'aligned over first {n} residues')
        record['notes'] = note
        print(f"    WARNING: {note}")
        ref_aln = ref_coords[:n]
        mob_aln = mob_coords[:n]
    else:
        record['notes'] = 'ok'
        ref_aln = ref_coords
        mob_aln = mob_coords

    # Kabsch
    try:
        rmsd, R, t = kabsch_rmsd(mob_aln, ref_aln)
        record['rmsd_ca'] = round(rmsd, 4)
        print(f"    Ca RMSD: {rmsd:.3f} A  "
              f"(AF3 binder={af3_len} res, AF2 monomer={af2_len} res)")
    except Exception as e:
        record['notes'] = f'RMSD error: {e}'
        print(f"    ERROR: {e}")
        return record

    # Apply transform to AF2 monomer (all atoms)
    transform_structure(af2_struct, R, t)

    # Write combined PDB: full AF3 complex + transformed AF2 monomer as chain Z
    try:
        out_pdb = os.path.join(superposed_dir, f'{design}_superposed.pdb')
        write_combined_pdb(af3_struct, af2_struct, out_pdb)
        record['superposed_pdb'] = out_pdb
        print(f"    Saved -> {out_pdb}")
    except Exception as e:
        record['notes'] += f'; save error: {e}'
        print(f"    WARNING: could not save combined PDB: {e}")

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--af3_out_dir',   required=True,
        help='Root AF3 output dir; each design in its own subdir')
    parser.add_argument('--af2_mono_dir',  required=True,
        help='Root AF2 monomer dir; design dirs named author_{design}/')
    parser.add_argument('--out_tsv',       required=True,
        help='Output TSV path for RMSD table')
    parser.add_argument('--superposed_dir', required=True,
        help='Output dir for {design}_superposed.pdb files')
    parser.add_argument('--designs', nargs='*',
        help='Design names (default: all subdirs in af3_out_dir)')
    args = parser.parse_args()

    if not HAS_BIOPYTHON:
        raise ImportError(
            "Biopython required: pip install biopython --break-system-packages"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_tsv)), exist_ok=True)
    os.makedirs(args.superposed_dir, exist_ok=True)

    if args.designs:
        designs = args.designs
    else:
        designs = sorted(
            d for d in os.listdir(args.af3_out_dir)
            if os.path.isdir(os.path.join(args.af3_out_dir, d))
        )

    print(f"Aligning {len(designs)} designs\n"
          f"  Reference: AF3 full complex (chains A-D)\n"
          f"  Mobile:    AF2 monomer -> chain Z in output PDB\n")

    records = []
    for idx, design in enumerate(designs):
        print(f"[{idx+1}/{len(designs)}] {design}")
        rec = align_design(
            design        = design,
            af3_out_dir   = args.af3_out_dir,
            af2_mono_dir  = args.af2_mono_dir,
            superposed_dir = args.superposed_dir,
        )
        records.append(rec)

    df = pd.DataFrame(records)
    col_order = ['design', 'af3_binder_len', 'af2_monomer_len',
                 'rmsd_ca', 'superposed_pdb', 'af3_cif', 'af2_pdb', 'notes']
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(args.out_tsv, sep='\t', index=False)
    print(f"\nSaved TSV -> {args.out_tsv}")

    # Summary
    if 'rmsd_ca' in df.columns and df['rmsd_ca'].notna().any():
        v = df['rmsd_ca'].dropna()
        print(f"\n-- RMSD summary ({len(v)}/{len(df)} designs) --")
        print(f"  mean   : {v.mean():.3f} A")
        print(f"  median : {v.median():.3f} A")
        print(f"  min    : {v.min():.3f} A  ({df.loc[v.idxmin(), 'design']})")
        print(f"  max    : {v.max():.3f} A  ({df.loc[v.idxmax(), 'design']})")
        print(f"  std    : {v.std():.3f} A")
        print(f"\n  Per design (sorted by RMSD):")
        for _, row in df.sort_values('rmsd_ca').iterrows():
            rmsd = row.get('rmsd_ca', np.nan)
            note = row.get('notes', '')
            if pd.notna(rmsd):
                flag = '  ** length mismatch' if 'mismatch' in str(note) else ''
                print(f"    {row['design']:20s}  {rmsd:.3f} A{flag}")
            else:
                print(f"    {row['design']:20s}  FAILED  ({note})")


if __name__ == '__main__':
    main()