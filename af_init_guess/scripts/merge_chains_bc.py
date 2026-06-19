#!/usr/bin/env python3
"""
Merge chains B and C into a single chain B with consecutive residue
numbering to prevent PyRosetta from detecting internal chain breaks.

Usage:
    python merge_chains_bc.py \
        --input_dir ~/Desktop/pmhc_binder_kras/af_init_guess/inputs/threaded_pdbs/ \
        --output_dir ~/Desktop/pmhc_binder_kras/af_init_guess/inputs/threaded_pdbs_merged/
"""

import os
import glob
import argparse


def merge_chains_bc(input_pdb, output_pdb):
    """
    Rename chain C to chain B and renumber all chain B+C residues
    consecutively. Remove all TER records and rewrite only two clean
    TER records — one after chain A and one at the end of chain B.
    """
    with open(input_pdb) as f:
        lines = f.readlines()

    # --- Pass 1: collect chain A residue IDs ---
    chain_a_resnums = []
    seen_a = set()
    for line in lines:
        if (line.startswith('ATOM') or line.startswith('HETATM')) \
                and line[21] == 'A':
            res_id = line[22:26].strip()
            if res_id not in seen_a:
                seen_a.add(res_id)
                chain_a_resnums.append(int(res_id))

    # Chain B starts after chain A with gap of 200 for AF2 chain break signal
    chain_b_start = max(chain_a_resnums) + 201 if chain_a_resnums else 201

    # --- Pass 2: collect ordered unique residues from chains B and C ---
    bc_residues = []
    seen_bc = set()
    for line in lines:
        if (line.startswith('ATOM') or line.startswith('HETATM')) \
                and line[21] in ('B', 'C'):
            chain = line[21]
            res_num = line[22:26].strip()
            icode = line[26]
            res_id = (chain, res_num, icode)
            if res_id not in seen_bc:
                seen_bc.add(res_id)
                bc_residues.append(res_id)

    # Consecutive renumbering for chain B
    renumber_map = {}
    for new_num, res_id in enumerate(bc_residues, start=chain_b_start):
        renumber_map[res_id] = new_num

    # --- Pass 3: write output ---
    # Strategy:
    #   - Keep all ATOM/HETATM records, renaming C->B and renumbering
    #   - Strip ALL existing TER records
    #   - Write exactly two TER records:
    #       1. After the last chain A ATOM record
    #       2. After the last chain B ATOM record (at the very end)

    atom_lines = []
    last_chain_a_idx = -1
    last_chain_b_idx = -1

    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            chain = line[21]
            if chain == 'A':
                atom_lines.append(line)
                last_chain_a_idx = len(atom_lines) - 1
            elif chain in ('B', 'C'):
                res_num = line[22:26].strip()
                icode = line[26]
                res_id = (chain, res_num, icode)
                new_num = renumber_map[res_id]
                line = line[:21] + 'B' + f'{new_num:4d}' + ' ' + line[27:]
                atom_lines.append(line)
                last_chain_b_idx = len(atom_lines) - 1
        # Skip all TER, END, ENDMDL records — we will rewrite them cleanly

    # Build final output: insert TER after chain A, TER after chain B, END
    new_lines = []
    for i, line in enumerate(atom_lines):
        new_lines.append(line)
        if i == last_chain_a_idx:
            new_lines.append('TER\n')
        if i == last_chain_b_idx:
            new_lines.append('TER\n')

    new_lines.append('END\n')

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

    # Verify with PyRosetta
    sample = sorted(glob.glob(os.path.join(args.output_dir, '*.pdb')))[0]
    print(f"\nVerifying {os.path.basename(sample)} with PyRosetta...")
    try:
        from pyrosetta import init, pose_from_pdb
        init('-mute all')
        pose = pose_from_pdb(sample)
        n_splits = len(pose.split_by_chain())
        print(f"  num_chains():     {pose.num_chains()}")
        print(f"  split_by_chain(): {n_splits}")
        print(f"  total_residue():  {pose.total_residue()}")
        breaks = [i for i in range(1, pose.total_residue())
                  if pose.chain(i) != pose.chain(i + 1)]
        print(f"  chain breaks at:  {breaks}")
        if n_splits == 2:
            print("  PASSED — ready for AF2 initial guess")
        else:
            print("  FAILED — still seeing more than 2 chains")
    except ImportError:
        print("  PyRosetta not available — check manually")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge chains B+C into single consecutive chain B"
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