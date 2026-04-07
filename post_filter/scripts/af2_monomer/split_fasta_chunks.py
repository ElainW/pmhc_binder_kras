"""
split_fasta_chunks.py

Splits a FASTA file into fixed-size chunks for SLURM array job submission.
Each chunk is written as a separate .fasta file.

Usage:
    python split_fasta_chunks.py \
        --fasta /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer_inputs.fasta \
        --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/monomer_chunks/ \
        --chunk_size 5
"""

import os
import argparse


def parse_fasta(fasta_path: str) -> list[tuple[str, str]]:
    """Return list of (header, sequence) pairs."""
    records = []
    header, seq_lines = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    records.append((header, ''.join(seq_lines)))
                header = line[1:]   # strip leading '>'
                seq_lines = []
            else:
                seq_lines.append(line)
    if header is not None:
        records.append((header, ''.join(seq_lines)))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta',      required=True,
                        help='Input FASTA (from prepare_af2_monomer_inputs.py)')
    parser.add_argument('--out_dir',    required=True,
                        help='Directory to write chunk_N.fasta files')
    parser.add_argument('--chunk_size', type=int, default=5,
                        help='Sequences per chunk (default: 5)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = parse_fasta(args.fasta)
    total   = len(records)
    n_chunks = (total + args.chunk_size - 1) // args.chunk_size

    print(f"Total sequences:  {total}")
    print(f"Chunk size:       {args.chunk_size}")
    print(f"Total chunks:     {n_chunks}")

    for chunk_idx in range(n_chunks):
        start  = chunk_idx * args.chunk_size
        end    = min(start + args.chunk_size, total)
        chunk  = records[start:end]

        out_path = os.path.join(args.out_dir, f'chunk_{chunk_idx}.fasta')
        with open(out_path, 'w') as f:
            for header, seq in chunk:
                f.write(f'>{header}\n{seq}\n')

        print(f"  chunk_{chunk_idx}.fasta — {len(chunk)} sequences "
              f"({', '.join(h.split('|')[0] for h, _ in chunk[:2])}"
              f"{'...' if len(chunk) > 2 else ''})")

    print(f"\nSubmit with:")
    print(f"  sbatch --array=0-{n_chunks - 1} run_af2_monomer_array.sh")


if __name__ == '__main__':
    main()