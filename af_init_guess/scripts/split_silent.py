#!/usr/bin/env python3
"""
Split a silent file into chunks for parallel AF2 jobs.

Usage:
    python split_silent.py \
        --silent threaded_designs.silent \
        --output_dir af2_initial_guess/chunks/ \
        --chunk_size 50
"""

import os
import argparse


def split_silent(silent_file, output_dir, chunk_size):
    """
    Split a Rosetta silent file into chunks of chunk_size entries.
    Preserves the header line in each chunk.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read entire silent file
    with open(silent_file) as f:
        lines = f.readlines()

    # Separate header (SEQUENCE/SCORE header lines at top)
    # and entry blocks
    header_lines = []
    entry_blocks = []
    current_block = []

    for line in lines:
        if line.startswith('SEQUENCE:') or \
           (line.startswith('SCORE:') and 'description' in line):
            header_lines.append(line)
        elif line.startswith('SCORE:'):
            # Start of a new entry
            if current_block:
                entry_blocks.append(current_block)
            current_block = [line]
        else:
            if current_block:
                current_block.append(line)

    # Don't forget the last block
    if current_block:
        entry_blocks.append(current_block)

    total_entries = len(entry_blocks)
    n_chunks = (total_entries + chunk_size - 1) // chunk_size

    print(f"Total entries:  {total_entries}")
    print(f"Chunk size:     {chunk_size}")
    print(f"Total chunks:   {n_chunks}")

    # Write chunks
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_entries)
        chunk_entries = entry_blocks[start:end]

        chunk_path = os.path.join(
            output_dir, f"chunk_{chunk_idx}.silent"
        )
        with open(chunk_path, 'w') as f:
            # Write header
            f.writelines(header_lines)
            # Write entries
            for block in chunk_entries:
                f.writelines(block)

        print(f"  Chunk {chunk_idx:3d}: "
              f"{len(chunk_entries):4d} entries → {chunk_path}")

    return n_chunks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a Rosetta silent file into chunks for "
                    "parallel AF2 initial guess jobs"
    )
    parser.add_argument(
        '--silent', required=True,
        help='Input silent file'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory for chunk files'
    )
    parser.add_argument(
        '--chunk_size', type=int, default=50,
        help='Number of designs per chunk (default: 50)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    n_chunks = split_silent(args.silent, args.output_dir, args.chunk_size)
    print(f"\nTo submit as SLURM array job:")
    print(f"  sbatch --array=0-{n_chunks-1} run_af2_initial_guess_array.sh")