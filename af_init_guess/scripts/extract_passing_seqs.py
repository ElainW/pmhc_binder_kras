#!/usr/bin/env python3
"""
Extract sequences from a ProteinMPNN FASTA whose AF2 ipSAE score passes a threshold.

Usage:
    python extract_passing_seqs.py \
        --scores  /path/to/af2_ipae_ipsae_scores.csv \
        --fasta   /path/to/all_sequences.fa \
        --output  /path/to/passing_seqs.fa \
        --ipsae_cutoff 0.8

The CSV label column uses the threaded-PDB naming convention:
    orient_252_pt20__31_pt12__0_a3_t0.1_s3_af2pred  (lowercase, _sN, _af2pred)

The FASTA header token (before first comma) uses the original MPNN convention:
    orient_252_pT20__31_pT12__0_a3_t0.1_T=0.1       (mixed case, _T=val)

Both are normalised to a common key before matching:
    orient_252_pt20__31_pt12__0_a3_t0.1_s3
"""

import re
import argparse
import pandas as pd


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise_label(label: str) -> str:
    """
    Normalise a CSV label to a match key.
    orient_252_pt20__31_pt12__0_a3_t0.1_s3_af2pred
    -> orient_252_pt20__31_pt12__0_a3_t0.1_s3
    """
    return re.sub(r'_af2pred$', '', label.strip(), flags=re.IGNORECASE)


def normalise_fasta_header(header: str) -> str:
    """
    Normalise a FASTA header token to a match key.
    orient_252_pT20__31_pT12__0_a3_t0.1_T=0.1, sample=3, ...
    -> orient_252_pt20__31_pt12__0_a3_t0.1_s3
    """
    token = header.lstrip('>').split(',')[0].strip()

    # Strip _T=<value> anywhere
    token = re.sub(r'_T=[\d.]+', '', token)

    # Lowercase trailing _T<value> -> _t<value> (old-format headers)
    token = re.sub(r'_T([\d.]+)$', lambda m: f'_t{m.group(1)}', token)

    # Extract sample number and append as _sN
    sample_match = re.search(r'\bsample=(\d+)', header)
    if sample_match:
        token = f"{token}_s{sample_match.group(1)}"

    return token.lower()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # Load scores and filter
    scores = pd.read_csv(args.scores)
    passing = scores[scores['ipsae_binder_peptide'] >= args.ipsae_cutoff].copy()
    passing['key'] = passing['label'].apply(normalise_label)
    passing_keys = set(passing['key'])

    print(f"Total scored designs:  {len(scores)}")
    print(f"Passing ipsae >= {args.ipsae_cutoff}: {len(passing)}")

    # Parse FASTA and match
    matched   = 0
    unmatched = 0
    out_lines = []

    with open(args.fasta) as f:
        header = None
        sequence = None
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                # Write previous entry if it passed
                if header is not None and sequence is not None:
                    key = normalise_fasta_header(header)
                    if key in passing_keys:
                        out_lines.append(header)
                        out_lines.append(sequence)
                        matched += 1
                    else:
                        unmatched += 1
                header = line
                sequence = None
            else:
                sequence = line

        # Handle last entry
        if header is not None and sequence is not None:
            key = normalise_fasta_header(header)
            if key in passing_keys:
                out_lines.append(header)
                out_lines.append(sequence)
                matched += 1
            else:
                unmatched += 1

    # Check for passing keys not found in FASTA
    found_keys = set()
    with open(args.fasta) as f:
        for line in f:
            if line.startswith('>'):
                found_keys.add(normalise_fasta_header(line.rstrip()))

    missing = passing_keys - found_keys
    if missing:
        print(f"\nWARNING: {len(missing)} passing label(s) not found in FASTA:")
        for k in sorted(missing)[:10]:
            print(f"  {k}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")

    # Write output
    with open(args.output, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')

    print(f"\nExtracted:  {matched} sequences -> {args.output}")
    if matched != len(passing):
        print(f"WARNING: expected {len(passing)} but extracted {matched} "
              f"({len(passing)-matched} not found in FASTA)")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract FASTA sequences passing an ipSAE threshold',
    )
    parser.add_argument('--scores',       required=True,
                        help='CSV with label and ipsae_binder_peptide columns')
    parser.add_argument('--fasta',        required=True,
                        help='ProteinMPNN output FASTA (all_sequences.fa)')
    parser.add_argument('--output',       required=True,
                        help='Output FASTA of passing sequences')
    parser.add_argument('--ipsae_cutoff', type=float, default=0.8,
                        help='Minimum ipsae_binder_peptide score (default: 0.8)')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())