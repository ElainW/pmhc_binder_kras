"""
author_designs_noMSA_MHC_af3_serve.py
======================================
Converts author-design AF3 input JSONs (alphafold3 dialect, no MSA) to
AlphaFold Server format (alphafoldserver dialect) with:

  Chain A (binder)  : unpairedMsa = ">query\\n<sequence>"
  Chain B (MHC)     : unpairedMsa = ">query\\n<sequence>"
  Chain C (peptide) : useStructureTemplate = false
  Chain D (B2M)     : unpairedMsa = ">query\\n<sequence>"

Model seeds rotate: job 1 -> seed 1, job 2 -> seed 2, etc.

Usage:
    python author_designs_noMSA_MHC_af3_serve.py \
        --input_dir  <path>/inputs/author_design_stats/af3_nomsa/ \
        --output     <path>/inputs/author_design_stats/binder_pMHC_full_noMSA_MHC.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def convert(input_dir: str, output_path: str) -> None:
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    output = []
    for seed, fpath in enumerate(files, start=1):
        with open(fpath) as f:
            data = json.load(f)

        chains = {p["protein"]["id"]: p["protein"]["sequence"]
                  for p in data["sequences"]}

        seq_a = chains["A"]
        seq_b = chains["B"]
        seq_c = chains["C"]
        seq_d = chains["D"]

        entry = {
            "name": data["name"],
            "modelSeeds": [seed],
            "sequences": [
                {"proteinChain": {"sequence": seq_a, "count": 1,
                                  "unpairedMsa": f">query\n{seq_a}"}},
                {"proteinChain": {"sequence": seq_b, "count": 1,
                                  "unpairedMsa": f">query\n{seq_b}"}},
                {"proteinChain": {"sequence": seq_c, "count": 1,
                                  "useStructureTemplate": False}},
                {"proteinChain": {"sequence": seq_d, "count": 1,
                                  "unpairedMsa": f">query\n{seq_d}"}},
            ],
            "dialect": "alphafoldserver",
            "version": 1,
        }
        output.append(entry)
        print(f"  [{seed:2d}] {data['name']}")

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(output)} entries -> {output_path}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(here, "..", "inputs", "author_design_stats")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", default=os.path.join(base, "af3_nomsa"),
                        help="Directory containing per-design alphafold3-dialect JSONs")
    parser.add_argument("--output", default=os.path.join(base, "binder_pMHC_full_noMSA_MHC.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    convert(args.input_dir, args.output)


if __name__ == "__main__":
    main()
