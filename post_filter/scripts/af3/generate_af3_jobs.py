#!/usr/bin/env python3
"""
Usage:
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

python generate_af3_jobs.py \
    --fasta /n/groups/marks/users/aaron/pmhc/post_filter/inputs/author_design_stats/author_binder_pMHC.fasta \
    --output_json_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/author_design_stats/af3 \
    --output_prediction_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3/ \
    --output_slurm_dir /n/groups/marks/users/aaron/pmhc/post_filter/slurm/

Generate AlphaFold3 JSON input files from a multi-chain FASTA file.
Each unique design (e.g. author_mage-282) becomes one JSON file with chains A, B, C.
All three chains (binder, MHC, peptide) are typed as "protein".

Sbatch AlphaFold3 sbatch jobs: split into two parts: MSA generation and inference
"""

import json
import os
import sys
import re
from collections import defaultdict
import argparse
import subprocess

from cmd_runner import *


def parse_fasta(fasta):
    """Parse multi-chain fasta → {design_name: {chain_id: sequence}}."""
    designs = defaultdict(dict)
    current_header = None
    current_seq_lines = []

    with open(fasta, 'r') as fasta_text:
        for line in fasta_text:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    name_tmp, chain = parse_header(current_header)
                    name = name_tmp.removeprefix("author_")
                    designs[name][chain] = "".join(current_seq_lines)
                current_header = line[1:]
                current_seq_lines = []
            else:
                current_seq_lines.append(line)

    # Save last entry
    if current_header is not None:
        name_tmp, chain = parse_header(current_header)
        name = name_tmp.removeprefix("author_")
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
        if seq != 'C':
            sequences.append({
                "protein": {
                    "id": chain_id,
                    "sequence": seq
                }
            })
        else: # skip HMM search for peptide
            sequences.append({
                "protein": {
                    "id": chain_id,
                    "sequence": seq,
                    "templates": []
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
    parser.add_argument('--fasta', required=True, help="Fasta file containing all the sequence input to AF3. Each complex should be separated into chain A: minibinder, chain B: truncated MHC, chain C: peptide")
    parser.add_argument('--output_json_dir', required=True, help="output directory of json file, which is required as input to AF3 run")
    parser.add_argument('--output_prediction_dir', required=True, help="output directory of all AF3 output. A subdirectory will be created with the sample name")
    parser.add_argument('--output_slurm_dir', required=True, help="directory to write slurm .out and .err files")
    parser.add_argument('--max_template_date', default='2026-04-07', help="maximum template date for AF3 to look up similar sequences in AFDB")
    parser.add_argument('--skip', nargs='*', help="sample names to skip. e.g. mage-282")

    args = parser.parse_args()

    os.makedirs(args.output_json_dir, exist_ok=True)
    os.makedirs(args.output_prediction_dir, exist_ok=True)
    os.makedirs(args.output_slurm_dir, exist_ok=True)
    output_json_dir = args.output_json_dir
    output_prediction_dir = args.output_prediction_dir
    slurm_dir = args.output_slurm_dir
    max_template_date = args.max_template_date
    FASTA = args.fasta
    sample_to_skip = args.skip
    global MODEL_SEEDS
    MODEL_SEEDS = [1, 2, 3, 4, 5]

    designs = parse_fasta(FASTA)
    print(f"Found {len(designs)} designs:\n  " + "\n  ".join(sorted(designs.keys())))
    print()

    for name, chains in sorted(designs.items()):
        data = build_json(name, chains)
        out_path = os.path.join(output_json_dir, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        # Quick sanity: print chain lengths
        lens = {c: len(s) for c, s in chains.items()}
        print(f"  {name}.json  (A={lens['A']}, B={lens['B']}, C={lens['C']})")

    print(f"\nDone. {len(designs)} JSON files written to {output_json_dir}")

    for name, _ in sorted(designs.items()):
        if name not in sample_to_skip:
            cmd1 = f"sbatch -o {slurm_dir}/{name}-AF3-p1-%j.out -e {slurm_dir}/{name}-AF3-p1-%j.err -J {name}_AF3_p1 AF3_part1.sh {name} {output_prediction_dir} {output_json_dir} {max_template_date}"
            result = subprocess.run(cmd1, capture_output=True, text=True, check=True, shell=True)
            job_id = result.stdout.strip().split(' ')[-1]

            print(f"AF3 MSA job submitted for {name}. Job ID: {job_id}")

#             cmd2 = f"sbatch -o {slurm_dir}/{name}-AF3-p2-%j.out -e {slurm_dir}/{name}-AF3-p2-%j.err -J {name}_AF3_p2 AF3_part2.sh {name} {output_prediction_dir}"
#             print(f"AF3 inference job submitted for {name}")
            cmd2 = f"sbatch --dependency=afterok:{job_id} -o {slurm_dir}/{name}-AF3-p2-%j.out -e {slurm_dir}/{name}-AF3-p2-%j.err -J {name}_AF3_p2 AF3_part2.sh {name} {output_prediction_dir}"
            run_cmd_small_output(cmd2)

            print(f"AF3 inference job submitted for {name} with dependency on {job_id}")
        else:
            print(f"Skipping {name}")


if __name__ == "__main__":
    main()
