#!/usr/bin/env python3
"""
prepare_monomer_stubs.py

Reads a FASTA file of binder sequences and writes one minimal stub PDB per
design into a flat output directory. The stub PDBs are single-residue GLY
placeholders — predict.py with -no_initial_guess ignores the coordinates
entirely and folds from sequence only.

Also writes a design list TSV (design_index, design_name, stub_pdb) used
by the SLURM array script to map SLURM_ARRAY_TASK_ID → design.

Usage:
    python prepare_monomer_stubs.py \
        --fasta  /n/groups/marks/users/aaron/pmhc_cp/post_filter/inputs/r2.5/af2_monomer/af2_monomer_inputs.fasta \
        --out_dir /n/groups/marks/users/aaron/pmhc_cp/post_filter/inputs/r2.5/af2_monomer/stubs/
"""

from __future__ import annotations
import os
import argparse

# Minimal single-residue GLY PDB template
# predict.py -no_initial_guess ignores coordinates; only sequence matters
_GLY_PDB = """\
ATOM      1  N   GLY A   1       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       2.000   1.000   1.000  1.00  0.00           C
ATOM      3  C   GLY A   1       3.000   1.000   1.000  1.00  0.00           C
ATOM      4  O   GLY A   1       3.500   2.000   1.000  1.00  0.00           O
END
"""


def parse_fasta(fasta_path: str) -> list[tuple[str, str]]:
    """Parse FASTA → [(header, sequence), ...]"""
    records = []
    header, lines = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    records.append((header, ''.join(lines)))
                header = line[1:].strip()
                lines = []
            else:
                lines.append(line)
    if header is not None:
        records.append((header, ''.join(lines)))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta',   required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = parse_fasta(args.fasta)
    print(f"Found {len(records)} sequences in {args.fasta}")

    index_path = os.path.join(args.out_dir, 'design_index.tsv')
    with open(index_path, 'w') as idx:
        idx.write('array_idx\tdesign\tsequence\tstub_dir\n')
        for i, (header, seq) in enumerate(records):
            name     = header.removesuffix('_af2pred')
            stub_dir = os.path.join(args.out_dir, name)
            os.makedirs(stub_dir, exist_ok=True)
            stub_pdb = os.path.join(stub_dir, f'{name}.pdb')
            with open(stub_pdb, 'w') as f:
                f.write(_GLY_PDB)
            idx.write(f'{i}\t{name}\t{seq}\t{stub_dir}\n')

    print(f"Written {len(records)} stub PDBs → {args.out_dir}")
    print(f"Design index → {index_path}")
    print(f"\nSubmit SLURM array with:")
    print(f"  sbatch --array=0-{len(records)-1} run_af2_monomer_r2.5_array.sh")


if __name__ == '__main__':
    main()