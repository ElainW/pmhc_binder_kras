#!/usr/bin/env python3
"""
Usage:
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/dl_binder_design

python generate_af3_json_nomsa_server.py \
    --fasta ~/Desktop/pmhc_binder_kras_cp/post_filter/inputs/r2.5/binder_pMHC_full.fasta \
    --output_json_dir ~/Desktop/pmhc_binder_kras_cp/post_filter/inputs/r2.5/af3_nomsa \
    --peptide_chain C

Generate AlphaFold3 server JSON input files from a multi-chain FASTA file.
Each unique design becomes one JSON file containing a single-element list
(the AF3 server accepts a list of job dicts).

Chain layout is flexible:
  Truncated MHC (3-chain):   A = binder, B = MHC alpha, C = peptide
  Full MHC + B2M (4-chain):  A = binder, B = MHC alpha, C = peptide, D = B2M

The peptide chain (default: C) gets useStructureTemplate=false to skip HMM
search, which crashes on short sequences. The binder chain (default: A) gets
unpairedMsa="" to skip MSA generation.

Chains are written in sorted order (A, B, C, D, ...) regardless of FASTA order.
"""

import json
import os
import re
from collections import defaultdict
import argparse


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
        name = name_raw.removeprefix("author_").replace(".", "")
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
    Build one AF3 server job dict for one design.

    chains:        {chain_id: sequence}
    peptide_chain: chain ID carrying the short peptide.
                   Gets useStructureTemplate=false (short sequences crash HMM search).
    binder_chain:  chain ID carrying the minibinder.
                   Gets unpairedMsa="" to skip MSA generation.
    All chains get maxTemplateDate=max_template_date.
    """
    chain_ids = sorted(chains.keys())
    missing = [c for c in chain_ids if chains.get(c) is None]
    if missing:
        raise ValueError(f"Design '{name}': missing sequences for chains {missing}")

    sequences = []
    for chain_id in chain_ids:
        seq = chains[chain_id]
        entry: dict = {
            "sequence":        seq,
            "count":           1,
        }

        if chain_id == peptide_chain:
            entry["useStructureTemplate"] = False
        elif chain_id == binder_chain:
            entry["unpairedMsa"] = f">query\n{seq}"

        sequences.append({"proteinChain": entry})

    return {
        "name":       name,
        "modelSeeds": model_seeds,
        "sequences":  sequences,
        "dialect":    "alphafoldserver",
        "version":    1,
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
    parser.add_argument('--fasta',             required=True,
                        help='Multi-chain FASTA. Headers: <design_name>_<CHAIN>')
    parser.add_argument('--output_json_dir',   required=True,
                        help='Output directory; one JSON file is written containing all designs')
    parser.add_argument('--peptide_chain',     default='C',
                        help='Chain ID of the peptide (default: C). '
                             'Gets useStructureTemplate=false to skip HMM search.')
    parser.add_argument('--binder_chain',      default='A',
                        help='Chain ID of the minibinder (default: A). '
                             'Gets unpairedMsa="" to skip MSA.')
    parser.add_argument('--model_seeds',       nargs='+', type=int,
                        default=[1, 2, 3, 4, 5],
                        help='Pool of AF3 model seeds — one seed assigned per design, cycling through the list (default: 1 2 3 4 5)')
    parser.add_argument('--skip',              nargs='*', default=[],
                        help='Design names to skip (without "author_" prefix)')
    args = parser.parse_args()

    os.makedirs(args.output_json_dir, exist_ok=True)

    # ── Parse FASTA ───────────────────────────────────────────────────────────
    designs = parse_fasta(args.fasta)
    print(f"Found {len(designs)} designs:")
    for name, chains in sorted(designs.items()):
        lens_str = ', '.join(f"{c}={len(s)}" for c, s in sorted(chains.items()))
        print(f"  {name}  ({lens_str})")
    print()

    validate_designs(designs, args.peptide_chain)

    # ── Write all designs into one JSON file (one seed per design) ───────────
    # Cycle through model_seeds, assigning one seed per design in order.
    jobs      = []
    n_skipped = 0
    design_list = sorted(designs.items())
    for i, (name, chains) in enumerate(design_list):
        if name in args.skip:
            print(f"  Skipping {name}")
            n_skipped += 1
            continue

        seed = args.model_seeds[i % len(args.model_seeds)]
        jobs.append(build_json(
            name=name,
            chains=chains,
            peptide_chain=args.peptide_chain,
            binder_chain=args.binder_chain,
            model_seeds=[seed],
        ))

    fasta_stem = os.path.splitext(os.path.basename(args.fasta))[0]
    out_path   = os.path.join(args.output_json_dir, f"{fasta_stem}.json")
    with open(out_path, "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"\nWritten {len(jobs)} jobs (one seed per design) → {out_path}")
    if n_skipped:
        print(f"Skipped {n_skipped} designs")


if __name__ == "__main__":
    main()