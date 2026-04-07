"""
split_pmhc_fold_pdbs.py

Splits pMHC fold _unrelaxed_model_1.pdb files (single chain output from AF2)
into proper chain A (binder) and chain B (MHC+peptide) PDB files.

Chain assignment is determined by residue count from the AF2 init guess PDB:
  - First binder_len residues  → chain A
  - Remaining residues         → chain B (MHC + peptide)

Output structure per design:
  {out_dir}/{design}_chainA.pdb   ← binder
  {out_dir}/{design}_chainB.pdb   ← MHC + peptide (merged, sequential numbering)
  {out_dir}/{design}_complex.pdb  ← full complex with correct chain labels

Usage:
    python split_pmhc_fold_pdbs.py \
        --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
        --af2_pdb_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
        --pmhc_fold_dir /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/outputs/r2/pmhc_fold_on_runs \
        --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select

DELTA_PAE_CUTOFF = -0.5
PEPTIDE_LEN      = 9
PDB_PARSER       = PDBParser(QUIET=True)


# ── Selectors for PDBIO ───────────────────────────────────────────────────────

class ResidueListSelect(Select):
    """Write only residues in a given set of (chain_id, res_serial_index) tuples."""
    def __init__(self, residue_serials: set, new_chain_id: str):
        self.residue_serials = residue_serials  # set of serial indices (0-indexed position)
        self.new_chain_id    = new_chain_id
        self._counter        = {}               # original chain → running count

    def accept_residue(self, residue):
        return id(residue) in self.residue_serials

    def accept_chain(self, chain):
        return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_pmhc_fold_pdb(design: str, pmhc_fold_dir: str) -> str | None:
    pattern = os.path.join(
        pmhc_fold_dir, 'sample_df_chunk_*',
        f'{design}_A1101_VVGADGVGK_unrelaxed_model_1.pdb'
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def get_chain_residue_count(pdb_path: str, chain_id: str) -> int:
    """Count ATOM residues in a given chain of a PDB file."""
    struct = PDB_PARSER.get_structure('s', pdb_path)
    count = 0
    for model in struct:
        for chain in model:
            if chain.id == chain_id:
                for res in chain:
                    if res.id[0] == ' ':
                        count += 1
        break
    return count


def write_chain_pdb(residues: list, chain_id: str, out_path: str):
    """
    Write a list of BioPython residue objects to a PDB file,
    assigning them all to chain_id and renumbering from 1.
    """
    serial = [0]

    def fmt_atom(atom, chain_id: str, resnum: int, resname: str) -> str:
        serial[0] += 1
        name     = atom.get_name()
        altloc   = atom.get_altloc() or ' '
        x, y, z  = atom.get_vector()
        occ      = atom.get_occupancy() or 1.0
        bfac     = atom.get_bfactor() or 0.0
        element  = (atom.element or '').strip().rjust(2)
        # Atom name formatting: 4-char field
        if len(name) < 4:
            name_fmt = f' {name:<3}'
        else:
            name_fmt = name[:4]
        return (
            f"ATOM  {serial[0]:5d} {name_fmt}{altloc}"
            f"{resname:3s} {chain_id}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occ:6.2f}{bfac:6.2f}          "
            f"{element:>2s}\n"
        )

    with open(out_path, 'w') as f:
        for new_resnum, res in enumerate(residues, start=1):
            resname = res.resname
            for atom in res.get_atoms():
                # Skip hydrogens and alternate locations beyond 'A'
                altloc = atom.get_altloc()
                if altloc not in (' ', 'A', ''):
                    continue
                f.write(fmt_atom(atom, chain_id, new_resnum, resname))
        f.write('TER\nEND\n')


def write_complex_pdb(binder_res: list, pmhc_res: list, out_path: str):
    """
    Write full complex with chain A (binder) and chain B (MHC+peptide),
    renumbered sequentially within each chain.
    """
    serial = [0]

    def fmt_atom(atom, chain_id: str, resnum: int, resname: str) -> str:
        serial[0] += 1
        name    = atom.get_name()
        altloc  = atom.get_altloc() or ' '
        x, y, z = atom.get_vector()
        occ     = atom.get_occupancy() or 1.0
        bfac    = atom.get_bfactor() or 0.0
        element = (atom.element or '').strip().rjust(2)
        if len(name) < 4:
            name_fmt = f' {name:<3}'
        else:
            name_fmt = name[:4]
        return (
            f"ATOM  {serial[0]:5d} {name_fmt}{altloc}"
            f"{resname:3s} {chain_id}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occ:6.2f}{bfac:6.2f}          "
            f"{element:>2s}\n"
        )

    with open(out_path, 'w') as f:
        # Chain A — binder
        for new_resnum, res in enumerate(binder_res, start=1):
            for atom in res.get_atoms():
                if atom.get_altloc() not in (' ', 'A', ''):
                    continue
                f.write(fmt_atom(atom, 'A', new_resnum, res.resname))
        f.write('TER\n')
        # Chain B — MHC + peptide
        for new_resnum, res in enumerate(pmhc_res, start=1):
            for atom in res.get_atoms():
                if atom.get_altloc() not in (' ', 'A', ''):
                    continue
                f.write(fmt_atom(atom, 'B', new_resnum, res.resname))
        f.write('TER\nEND\n')


def split_fold_pdb(fold_pdb: str,
                   binder_len: int,
                   mhc_len: int,
                   peptide_len: int = PEPTIDE_LEN) -> dict | None:
    """
    Load pMHC fold PDB (single chain) and split into components.
    Returns dict with residue lists, or None on failure.
    """
    struct = PDB_PARSER.get_structure('fold', fold_pdb)

    all_residues = []
    for model in struct:
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':
                    all_residues.append(res)
        break  # first model only

    total    = len(all_residues)
    expected = binder_len + mhc_len + peptide_len

    if total != expected:
        print(f"  WARNING: {os.path.basename(fold_pdb)} has {total} residues, "
              f"expected {binder_len}+{mhc_len}+{peptide_len}={expected} "
              f"— using first {binder_len} as binder, rest as pMHC")

    binder_res  = all_residues[:binder_len]
    pmhc_res    = all_residues[binder_len:]   # MHC + peptide
    mhc_res     = all_residues[binder_len:binder_len + mhc_len]
    peptide_res = all_residues[-peptide_len:]

    return {
        'binder':  binder_res,
        'pmhc':    pmhc_res,
        'mhc':     mhc_res,
        'peptide': peptide_res,
        'all':     all_residues,
        'n_total': total,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

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
    print(f"Splitting {len(designs)} pMHC fold PDBs...")

    summary_rows = []
    n_ok = 0

    for design in designs:
        af2_path  = os.path.join(args.af2_pdb_dir, design + '.pdb')
        fold_path = find_pmhc_fold_pdb(design, args.pmhc_fold_dir)

        if not os.path.exists(af2_path):
            print(f"  WARNING: AF2 init guess PDB not found: {design}")
            continue
        if fold_path is None or not os.path.exists(fold_path):
            print(f"  WARNING: pMHC fold PDB not found: {design}")
            continue

        # Get lengths from AF2 init guess (ground truth chain lengths)
        binder_len = get_chain_residue_count(af2_path, 'A')
        mhc_len    = get_chain_residue_count(af2_path, 'B') - PEPTIDE_LEN

        if binder_len == 0 or mhc_len <= 0:
            print(f"  WARNING: unexpected chain lengths for {design}: "
                  f"binder={binder_len}, mhc={mhc_len}")
            continue

        # Split the fold PDB
        split = split_fold_pdb(fold_path, binder_len, mhc_len, PEPTIDE_LEN)
        if split is None:
            continue

        # Verify peptide sequence
        three_to_one = {
            'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
            'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
            'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
            'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
        }
        pep_seq = ''.join(
            three_to_one.get(r.resname, 'X') for r in split['peptide']
        )
        if pep_seq != 'VVGADGVGK':
            print(f"  WARNING: {design} peptide = {pep_seq} (expected VVGADGVGK)")

        # Write output files
        chain_a_path  = os.path.join(args.out_dir, f'{design}_chainA.pdb')
        chain_b_path  = os.path.join(args.out_dir, f'{design}_chainB.pdb')
        complex_path  = os.path.join(args.out_dir, f'{design}_complex.pdb')

        write_chain_pdb(split['binder'],  'A', chain_a_path)
        write_chain_pdb(split['pmhc'],    'B', chain_b_path)
        write_complex_pdb(split['binder'], split['pmhc'], complex_path)

        summary_rows.append({
            'design':          design,
            'binder_len':      binder_len,
            'mhc_len':         mhc_len,
            'peptide_len':     PEPTIDE_LEN,
            'total_residues':  split['n_total'],
            'peptide_seq':     pep_seq,
            'chain_a_pdb':     chain_a_path,
            'chain_b_pdb':     chain_b_path,
            'complex_pdb':     complex_path,
            'fold_pdb_source': fold_path,
        })
        n_ok += 1

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, 'split_pdb_summary.tsv')
    summary_df.to_csv(summary_path, sep='\t', index=False)

    print(f"\nDone: {n_ok}/{len(designs)} PDBs split successfully")
    print(f"Outputs in: {args.out_dir}")
    print(f"  *_chainA.pdb  — binder (chain A, renumbered from 1)")
    print(f"  *_chainB.pdb  — MHC+peptide (chain B, renumbered from 1)")
    print(f"  *_complex.pdb — full complex with correct chain labels")
    print(f"Summary: {summary_path}")


if __name__ == '__main__':
    main()