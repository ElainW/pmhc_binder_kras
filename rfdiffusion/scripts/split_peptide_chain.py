#!/usr/bin/env python3
"""
Split the last 9 residues of chain B into a new chain C.

For each input PDB:
  - Chain A: binder (unchanged)
  - Chain B: MHC only (all chain B residues except the last 9)
  - Chain C: peptide (last 9 residues of chain B, renumbered 1-9)

Outputs a new PDB to a specified directory, which is used as
the RFdiffusion input. The original PDBs are not modified.

Usage:
    python split_peptide_chain.py \
        --input_dir  /path/to/filtered \
        --output_dir /path/to/split \
        --backbones  orient_252_pT20__31 orient_255_pT12__33 ...
"""

import argparse
import os

PEPTIDE_LEN = 9

def split_pdb(input_path: str, output_path: str) -> None:
    """Read a PDB, split last 9 residues of chain B into chain C."""

    # Collect unique residue numbers for chain B (preserving order)
    chain_b_resnums = []
    seen = set()
    with open(input_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and len(line) >= 27:
                chain = line[21]
                resnum = int(line[22:26].strip())
                if chain == "B" and resnum not in seen:
                    chain_b_resnums.append(resnum)
                    seen.add(resnum)

    if len(chain_b_resnums) < PEPTIDE_LEN:
        raise ValueError(
            f"{input_path}: chain B has only {len(chain_b_resnums)} residues, "
            f"expected at least {PEPTIDE_LEN}"
        )

    peptide_resnums = set(chain_b_resnums[-PEPTIDE_LEN:])
    # Map old peptide resnum -> new chain C resnum (1-indexed)
    peptide_resnum_map = {
        old: new
        for new, old in enumerate(sorted(peptide_resnums), start=1)
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path) as fh, open(output_path, "w") as out:
        ter_written = {"B": False}
        for line in fh:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 27:
                chain = line[21]
                resnum = int(line[22:26].strip())

                if chain == "B" and resnum in peptide_resnums:
                    # Write TER for chain B before first peptide residue
                    if not ter_written["B"]:
                        out.write("TER\n")
                        ter_written["B"] = True
                    # Remap to chain C, renumber 1-9
                    new_resnum = peptide_resnum_map[resnum]
                    new_line = (
                        line[:21]
                        + "C"
                        + f"{new_resnum:4d}"
                        + line[26:]
                    )
                    out.write(new_line)
                else:
                    out.write(line)

            elif line.startswith("TER"):
                # Skip original TER lines; we insert our own above
                continue
            elif line.startswith("END"):
                out.write("TER\n")
                out.write("END\n")
                break
            else:
                out.write(line)

    print(f"  Written: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir",  required=True,
                        help="Directory containing original *_relaxed.pdb files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write split PDBs into")
    parser.add_argument("--backbones",  nargs="+", required=True,
                        help="Backbone names (without .pdb extension)")
    args = parser.parse_args()

    for bb in args.backbones:
        input_path  = os.path.join(args.input_dir,  f"{bb}.pdb")
        output_path = os.path.join(args.output_dir, f"{bb}.pdb")

        if not os.path.exists(input_path):
            print(f"  [WARN] Not found, skipping: {input_path}")
            continue

        print(f"Processing {bb} ...")
        split_pdb(input_path, output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()