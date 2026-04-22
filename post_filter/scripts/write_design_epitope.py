import os
import sys
import argparse


def _parse_header(header: str) -> tuple[str, str]:
    """
    Split 'orient_158_pt12__64_sample01_A' → ('orient_158_pt12__64_sample01', 'A').
    Chain ID is the last underscore-delimited token and must be a single letter.
    """
    match = re.match(r"^(.+)_([A-Z])$", header)
    if not match:
        raise ValueError(
            f"Cannot parse FASTA header: '{header}'\n"
            f"Expected format: <design_name>_<CHAIN_LETTER>  e.g. orient_158_pt12__64_sample01_A"
        )
    return match.group(1), match.group(2)


def parse_fasta(fasta_path: str) -> dict[str, dict[str, str]]:
    """
    Parse a multi-chain FASTA into {design_name: {chain_id: sequence}}.

    Header format: >design_name_CHAINID
    e.g. >orient_158_pt12__64_sample01_A  or  >orient_158_pt12__76_sample07_D
    """
    designs: dict[str, dict[str, str]] = defaultdict(dict)
    current_header = None
    current_seq_lines: list[str] = []

    def _flush():
        if current_header is None:
            return
        name_raw, chain = _parse_header(current_header)
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--fasta',                required=True,
                        help='Multi-chain FASTA. Headers: <design_name>_<CHAIN>')
    parser.add_argument('--output_csv',            required=True,
                        help='.csv output to run af3_design_stats.py and plot_author_stats.py')
    parser.add_argument('--peptide_chain',         type=str,
                        default='C',
                        help='Chain ID of the peptide (default: C). '
                             'Take the peptide sequence from this chain')
    parser.add_argument('--epitope_resum',         type=int,
                        default='5',
                        help='The 1-based index of the residue of interest for each design')
    parser.add_argument('--epitope_notes',         type=str,
                        default='G12D so 5th residue in the peptide',
                        help='Epitope notes for each design')
    parser.add_argument('--target_name',           type=str,
                        default='Name of the protein each peptide is derived from',
                        help='Content of the target_name column')
    parser.add_argument('--hla',                   type=str,
                        default='A*11:01',
                        help='Name of the HLA allele')
    args = parser.parse_args()

    # ── Parse FASTA ───────────────────────────────────────────────────────────
    designs = parse_fasta(args.fasta)
    print(designs)
    print(f"Found {len(designs)} designs:")

    validate_designs(designs, args.peptide_chain)

    with open(args.output_csv, 'w') as f_out:
        f_out.write(','.join(['design','hla','peptide','peptide_len',
                              'target_name','cms_hotspot_positions',
                              'cms_hotspot_notes\n']))
        for name, chains in sorted(designs.items()):
            if chains == args.peptide_chain:
                f_out.write(",".join(
                    [
                        name,
                        args.hla,
                        designs[name][chains],
                        str(len(designs[name][chains])),
                        args.target_name,
                        str(args.epitope_resum),
                        args.epitope_notes
                    ]
                )
                           )



if __name__ == "__main__":
    main()
