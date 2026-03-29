# reorder_for_af2.py
# Converts A=MHC, P=peptide, Q=minibinder
# to A=minibinder, B=MHC+peptide for af2_initial_guess predict.py

import sys
from pathlib import Path

def reorder_for_af2(input_pdb: str, output_pdb: str) -> None:
    in_path  = Path(input_pdb)
    out_path = Path(output_pdb)

    # Collect ATOM lines per chain
    chains: dict[str, list[str]] = {"A": [], "P": [], "Q": []}

    with open(in_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain = line[21]
            if chain not in chains:
                print(f"WARNING: unexpected chain '{chain}', skipping", file=sys.stderr)
                continue
            chains[chain].append(line)

    # Verify expected sizes
    print(f"Input chain sizes:")
    print(f"  Chain A (MHC)      : {len(chains['A'])} ATOM lines")
    print(f"  Chain P (peptide)  : {len(chains['P'])} ATOM lines")
    print(f"  Chain Q (binder)   : {len(chains['Q'])} ATOM lines")

    # Build output: binder first (A), then MHC+peptide merged (B)
    # Order: Q -> new A, then A+P -> new B
    out_lines = []
    atom_serial = 1

    def write_chain(source_lines, new_chain_id):
        nonlocal atom_serial, out_lines
        for line in source_lines:
            new_line = (
                line[:6]
                + f"{atom_serial:5d}"
                + line[11:21]
                + new_chain_id
                + line[22:]
            )
            out_lines.append(new_line)
            atom_serial += 1
        out_lines.append("TER\n")

    # Chain A of output = minibinder (was Q)
    write_chain(chains["Q"], "A")

    # Chain B of output = MHC (was A) + peptide (was P), merged
    # Write MHC portion
    for line in chains["A"]:
        new_line = (
            line[:6]
            + f"{atom_serial:5d}"
            + line[11:21]
            + "B"
            + line[22:]
        )
        out_lines.append(new_line)
        atom_serial += 1
    # Append peptide directly — no TER between them
    for line in chains["P"]:
        new_line = (
            line[:6]
            + f"{atom_serial:5d}"
            + line[11:21]
            + "B"
            + line[22:]
        )
        out_lines.append(new_line)
        atom_serial += 1
    out_lines.append("TER\n")
    out_lines.append("END\n")

    with open(out_path, "w") as fh:
        fh.writelines(out_lines)

    # Verify output
    out_a = sum(1 for l in out_lines if l.startswith("ATOM") and l[21] == "A")
    out_b = sum(1 for l in out_lines if l.startswith("ATOM") and l[21] == "B")
    print(f"\nOutput chain sizes:")
    print(f"  Chain A (binder)        : {out_a} ATOM lines  [was Q]")
    print(f"  Chain B (MHC + peptide) : {out_b} ATOM lines  [was A + P]")
    print(f"\nWritten: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python reorder_for_af2.py input.pdb output.pdb")
        sys.exit(1)
    reorder_for_af2(sys.argv[1], sys.argv[2])