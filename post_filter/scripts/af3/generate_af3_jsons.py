#!/usr/bin/env python3
"""
Usage:
python generate_af3_jsons.py \
    --fasta /n/groups/marks/users/aaron/pmhc/post_filter/inputs/author_design_stats/author_binder_pMHC.fasta \
    --output_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/af3

Generate AlphaFold3 JSON input files from a multi-chain FASTA file.
Each unique design (e.g. author_mage-282) becomes one JSON file with chains A, B, C.
All three chains (binder, MHC, peptide) are typed as "protein".
"""

import json
import os
import sys
import re
from collections import defaultdict
import argparse


def parse_fasta(fasta):
    """Parse multi-chain fasta → {design_name: {chain_id: sequence}}."""
    designs = defaultdict(dict)
    current_header = None
    current_seq_lines = []

    with open(fasta, 'r') as fasta_text:
        for line in fasta_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    name, chain = parse_header(current_header)
                    designs[name][chain] = "".join(current_seq_lines)
                current_header = line[1:]
                current_seq_lines = []
            else:
                current_seq_lines.append(line)

    # Save last entry
    if current_header is not None:
        name, chain = parse_header(current_header)
        designs[name][chain] = "".join(current_seq_lines)

    return designs


def parse_header(header):
    """Split 'author_mage-282_A' → ('author_mage-282', 'A')."""
    match = re.match(r"^(.+)_([A-Z])$", header)
    if not match:
        raise ValueError(f"Cannot parse header: {header}")
    return match.group(1), match.group(2)


def build_json(name, chains):
    """Build the AF3 JSON dict for one design."""
    sequences = []
    for chain_id in ["A", "B", "C"]:
        seq = chains.get(chain_id)
        if seq is None:
            raise ValueError(f"Missing chain {chain_id} for design '{name}'")
        sequences.append({
            "protein": {
                "id": chain_id,
                "sequence": seq
            }
        })

    return {
        "name": name,
        "modelSeeds": MODEL_SEEDS,
        "dialect": "alphafold3",
        "version": 1,
        "sequences": sequences
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True)
    parser.add_argument('--output_dir',       required=True)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    FASTA = args.fasta
    MODEL_SEEDS = [1, 2, 3, 4, 5]

    designs = parse_fasta(FASTA)
    print(f"Found {len(designs)} designs:\n  " + "\n  ".join(sorted(designs.keys())))
    print()

    for name, chains in sorted(designs.items()):
        data = build_json(name, chains)
        out_path = os.path.join(args.output_dir, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        # Quick sanity: print chain lengths
        lens = {c: len(s) for c, s in chains.items()}
        print(f"  {name}.json  (A={lens['A']}, B={lens['B']}, C={lens['C']})")

    print(f"\nDone. {len(designs)} JSON files written to {args.output_dir}")


if __name__ == "__main__":
    main()
