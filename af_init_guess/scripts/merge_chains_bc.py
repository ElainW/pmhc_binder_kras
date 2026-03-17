#!/usr/bin/env python3
"""
Merge chains B and C into a single chain B in threaded PDBs.
Required because dl_binder_design af2_initial_guess only supports 2 chains.

Usage:
    python merge_chains_bc.py \
        --input_dir ../inputs/threaded_pdbs/ \
        --output_dir ../inputs/threaded_pdbs_merged/
"""

import os
import glob
import argparse


def merge_chains_bc(input_pdb, output_pdb):
    """
    Rename chain C to chain B in a PDB file.
    Keeps chain A (binder) unchanged.
    Keeps chain B (MHC) unchanged.
    Renames chain C (peptide) to chain B so B+C become one chain.
    """
    with open(input_pdb) as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            chain = line[21]
            if chain == 'C':
                # Rename chain C to B
                line = line[:21] + 'B' + line[22:]
        elif line.startswith('TER') and len(line) > 21:
            chain = line[21]
            if chain == 'C':
                line = line[:21] + 'B' + line[22:]
        new_lines.append(line)

    with open(output_pdb, 'w') as f:
        f.writelines(new_lines)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    pdb_files = sorted(glob.glob(os.path.join(args.input_dir, '*.pdb')))
    print(f"Found {len(pdb_files)} PDB files")

    success = 0
    failed = 0

    for pdb in pdb_files:
        basename = os.path.basename(pdb)
        out_path = os.path.join(args.output_dir, basename)
        try:
            merge_chains_bc(pdb, out_path)
            success += 1
        except Exception as e:
            print(f"  Failed: {basename}: {e}")
            failed += 1

    print(f"\nDone: {success} merged, {failed} failed")
    print(f"Output: {args.output_dir}")

    # Verify a sample
    sample = sorted(glob.glob(os.path.join(args.output_dir, '*.pdb')))[0]
    chains = set()
    with open(sample) as f:
        for line in f:
            if line.startswith('ATOM'):
                chains.add(line[21])
    print(f"\nVerification on {os.path.basename(sample)}:")
    print(f"  Chains present: {sorted(chains)}")
    print(f"  Expected: ['A', 'B']")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge chains B and C into single chain B for "
                    "dl_binder_design af2_initial_guess compatibility"
    )
    parser.add_argument(
        '--input_dir', required=True,
        help='Directory containing threaded PDB files (3 chains: A, B, C)'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory for merged PDB files (2 chains: A, B)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)