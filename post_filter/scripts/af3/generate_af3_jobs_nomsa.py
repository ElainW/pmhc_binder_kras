#!/usr/bin/env python3
"""
Usage:
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/dl_binder_design

python generate_af3_jobs_nomsa.py \
    --fasta ~/Desktop/pmhc_binder_kras/post_filter/inputs/author_design_stats/author_binder_pMHC.fasta \
    --output_json_dir ~/Desktop/pmhc_binder_kras/post_filter/inputs/author_design_stats/af3 \
    --output_prediction_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/af3/ \
    --output_slurm_dir ~/Desktop/pmhc_binder_kras/post_filter/slurm/ \
    --peptide_chain C \
    --skip ctnnb1-15 gp100-3 hiv-9 hiv-10 mage-4 mage-282 mage-513 mart1-3 mart1-43 pap-116 phox2b-5 phox2b-11 sars-6 wt1-5 wt1-8 yfv-2

python generate_af3_jobs_nomsa.py \
    --fasta ~/Desktop/pmhc_binder_kras/post_filter/inputs/author_design_stats/author_binder_pMHC_full.fasta \
    --output_json_dir ~/Desktop/pmhc_binder_kras/post_filter/inputs/author_design_stats/af3_nomsa \
    --output_prediction_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/af3_nomsa/ \
    --output_slurm_dir ~/Desktop/pmhc_binder_kras/post_filter/slurm/ \
    --peptide_chain C \
    --skip ctnnb1-15 mart1-3 mart1-43 pap-116 phox2b-5

python generate_af3_jobs_nomsa.py \
    --fasta ~/Desktop/pmhc_binder_kras/post_filter/inputs/r2/binder_pMHC_full.fasta \
    --output_json_dir ~/Desktop/pmhc_binder_kras/post_filter/inputs/r2/af3_nomsa \
    --output_prediction_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/af3_nomsa/ \
    --output_slurm_dir ~/Desktop/pmhc_binder_kras/post_filter/slurm/ \
    --peptide_chain C \
    --skip orient_158_pt12__64_sample01 orient_158_pt12__76_sample04 orient_158_pt12__76_sample07 orient_252_pt20__31_sample01 orient_252_pt20__31_sample04 orient_252_pt20__31_sample06 orient_255_pt12__0_sample02 orient_255_pt12__11_sample07 orient_255_pt12__14_sample06 orient_255_pt12__15_sample01 orient_255_pt12__17_sample03 orient_255_pt12__18_sample01 orient_255_pt12__21_sample01 orient_255_pt12__21_sample07 orient_255_pt12__27_sample02 orient_255_pt12__27_sample03 orient_255_pt12__27_sample04

Generate AlphaFold3 JSON input files from a multi-chain FASTA file.
Each unique design becomes one JSON file. Chain layout is flexible:

  Truncated MHC (3-chain):   A = binder, B = MHC alpha, C = peptide
  Full MHC + B2M (4-chain):  A = binder, B = MHC alpha, C = peptide, D = B2M

The peptide chain (default: C) always gets "templates": [] to skip HMM search,
which would crash on short sequences. All other chains are standard protein entries.

Chains are written in sorted order (A, B, C, D, ...) regardless of FASTA order.
"""

import json
import os
import re
import subprocess
from collections import defaultdict
import argparse

from cmd_runner import run_cmd_small_output


# ── FASTA parsing ─────────────────────────────────────────────────────────────

def parse_fasta(fasta_path: str) -> dict[str, dict[str, str]]:
    """
    Parse a multi-chain FASTA into {design_name: {chain_id: sequence}}.

    Header format: >design_name_CHAINID
    e.g. >author_mage-282_A  or  >author_sars-6_D

    The 'author_' prefix is stripped from design names.
    """
    designs: dict[str, dict[str, str]] = defaultdict(dict)
    current_header = None
    current_seq_lines: list[str] = []

    def _flush():
        if current_header is None:
            return
        name_raw, chain = _parse_header(current_header)
        name = name_raw.removeprefix("author_")
        designs[name][chain] = "".join(current_seq_lines)

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                current_header = line[1:]
                current_seq_lines = []
            else:
                current_seq_lines.append(line)
    _flush()

    return dict(designs)


def _parse_header(header: str) -> tuple[str, str]:
    """
    Split 'author_mage-282_A' → ('author_mage-282', 'A').
    Chain ID is the last underscore-delimited token and must be a single letter.
    """
    match = re.match(r"^(.+)_([A-Z])$", header)
    if not match:
        raise ValueError(
            f"Cannot parse FASTA header: '{header}'\n"
            f"Expected format: <design_name>_<CHAIN_LETTER>  e.g. author_mage-282_A"
        )
    return match.group(1), match.group(2)


# ── JSON building ─────────────────────────────────────────────────────────────

def build_json(name: str,
               chains: dict[str, str],
               peptide_chain: str,
               binder_chain: str,
               model_seeds: list[int]) -> dict:
    """
    Build the AF3 input JSON for one design.

    chains:        {chain_id: sequence}  — any subset of A/B/C/D/...
    peptide_chain: chain ID that carries the short peptide (default 'C').
                   This chain gets "templates": [] to skip HMM profile search,
                   which crashes on sequences < ~15 residues.
    binder_chain:  chain ID that carries the minibinder (default 'A').
                   This chain gets "pairedMsa": "", "unpairedMsa": ""
                   to skip MSA generation.
    """
    chain_ids = sorted(chains.keys())   # deterministic order: A, B, C, D, ...

    missing = [c for c in chain_ids if chains.get(c) is None]
    if missing:
        raise ValueError(f"Design '{name}': missing sequences for chains {missing}")

    sequences = []
    for chain_id in chain_ids:
        seq = chains[chain_id]
        entry: dict = {"id": chain_id, "sequence": seq}

        # Skip template HMM search for the peptide chain — short sequences
        # cause hmmbuild/hmmsearch to produce empty Stockholm output, crashing
        # AF3's parsers.convert_stockholm_to_a3m with StopIteration.
        if chain_id == peptide_chain:
            entry["templates"] = []
        elif chain_id == binder_chain:
            entry["pairedMsa"] = ""
            entry["unpairedMsa"] = ""

        sequences.append({"protein": entry})

    return {
        "name":      name,
        "modelSeeds": model_seeds,
        "dialect":   "alphafold3",
        "version":   1,
        "sequences": sequences,
    }


# ── Validation ────────────────────────────────────────────────────────────────

def validate_designs(designs: dict[str, dict[str, str]],
                     peptide_chain: str) -> None:
    """Warn about unexpected chain layouts."""
    for name, chains in sorted(designs.items()):
        chain_ids = sorted(chains.keys())

        if peptide_chain not in chains:
            print(f"  WARNING: {name} — peptide chain '{peptide_chain}' not found "
                  f"(chains present: {chain_ids})")

        for chain_id, seq in chains.items():
            if not seq:
                print(f"  WARNING: {name} chain {chain_id} — empty sequence")

        # Flag unexpectedly short non-peptide chains
        for chain_id, seq in chains.items():
            if chain_id != peptide_chain and len(seq) < 20:
                print(f"  WARNING: {name} chain {chain_id} — sequence very short "
                      f"({len(seq)} aa); consider adding to --peptide_chain or "
                      f"checking FASTA")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--fasta',                required=True,
                        help='Multi-chain FASTA. Headers: <design_name>_<CHAIN>')
    parser.add_argument('--output_json_dir',      required=True,
                        help='Directory for AF3 input JSON files')
    parser.add_argument('--output_prediction_dir', required=True,
                        help='Root output directory for AF3 predictions')
    parser.add_argument('--output_slurm_dir',     required=True,
                        help='Directory for SLURM .out/.err files')
    parser.add_argument('--peptide_chain',         default='C',
                        help='Chain ID of the peptide (default: C). '
                             'This chain gets "templates": [] to skip HMM search.')
    parser.add_argument('--binder_chain',         default='A',
                        help='Chain ID of the minibinder (default: A). '
                             'This chain gets "pairedMsa": "", "unpairedMsa": "" to skip MSA.')
    parser.add_argument('--max_template_date',    default='2026-04-07',
                        help='Maximum template date for AF3 (default: 2026-04-07)')
    parser.add_argument('--model_seeds',           nargs='+', type=int,
                        default=[1, 2, 3, 4, 5],
                        help='AF3 model seeds (default: 1 2 3 4 5)')
    parser.add_argument('--skip',                  nargs='*', default=[],
                        help='Design names to skip (without "author_" prefix)')
    args = parser.parse_args()

    os.makedirs(args.output_json_dir,       exist_ok=True)
    os.makedirs(args.output_prediction_dir, exist_ok=True)
    os.makedirs(args.output_slurm_dir,      exist_ok=True)

    # ── Parse FASTA ───────────────────────────────────────────────────────────
    designs = parse_fasta(args.fasta)
    print(f"Found {len(designs)} designs:")
    for name, chains in sorted(designs.items()):
        lens = {c: len(s) for c, s in sorted(chains.items())}
        lens_str = ', '.join(f"{c}={l}" for c, l in lens.items())
        print(f"  {name}  ({lens_str})")
    print()

    validate_designs(designs, args.peptide_chain)

    # ── Write JSONs ───────────────────────────────────────────────────────────
    for name, chains in sorted(designs.items()):
        data     = build_json(name, chains, args.peptide_chain, args.binder_chain, args.model_seeds)
        out_path = os.path.join(args.output_json_dir, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"Written {len(designs)} JSON files → {args.output_json_dir}\n")

#     # ── Submit SLURM jobs ─────────────────────────────────────────────────────
    skip_set = {s.lower() for s in (args.skip or [])}

    for name in sorted(designs.keys()):
        if name in skip_set:
            print(f"Skipping {name}")
            continue

        slurm_dir   = args.output_slurm_dir
        out_pred    = args.output_prediction_dir
        out_json    = args.output_json_dir
        max_tmpl    = args.max_template_date

        # skip finished samples
        if os.path.isfile(f"{out_pred}/{name}/{name}_data.json"):
            print(f"{name} has been run. Skipping...")
            continue

        # Part 1: MSA + template search (CPU only, --norun_inference)
        cmd1 = (
            f"sbatch "
            f"-o {slurm_dir}/{name}-AF3-p1-nomsa-%j.out "
            f"-e {slurm_dir}/{name}-AF3-p1-nomsa-%j.err "
            f"-J {name}_AF3_p1 "
            f"AF3_part1.sh {name} {out_pred} {out_json} {max_tmpl}"
        )
        result = subprocess.run(
            cmd1, capture_output=True, text=True, check=True, shell=True
        )
        job_id = result.stdout.strip().split()[-1]
        print(f"AF3 MSA job submitted for {name}  (job {job_id})")

        # Part 2: Inference (GPU), depends on Part 1 completing successfully
        cmd2 = (
            f"sbatch "
            f"--dependency=afterok:{job_id} "
            f"-o {slurm_dir}/{name}-AF3-p2-nomsa-%j.out "
            f"-e {slurm_dir}/{name}-AF3-p2-nomsa-%j.err "
            f"-J {name}_AF3_p2 "
            f"AF3_part2.sh {name} {out_pred}"
        )
        run_cmd_small_output(cmd2)
        print(f"AF3 inference job submitted for {name}  (depends on {job_id})")

#         cmd2 = (
#             f"bash "
#             f"AF3_part2.sh {name} {out_pred}"
#         )
#         run_cmd_small_output(cmd2)


if __name__ == "__main__":
    main()
