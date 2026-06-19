#!/usr/bin/env python3
"""
generate_full_complex_fasta.py

Reads minibinder sequences from af2_monomer_sequences.csv (source==your_designs)
and writes a 4-chain FASTA for each design:
  A = minibinder
  B = full-length HLA-A*11:01 alpha chain
  C = KRAS G12D peptide (VVGADGVGK)
  D = B2M

Output: ~/Desktop/pmhc_binder_kras/post_filter/inputs/r2/binder_pMHC_full.fasta

Usage:
    python ~/Desktop/pmhc_binder_kras/post_filter/scripts/af3/generate_full_complex_fasta.py \
        --csv ~/Desktop/pmhc_binder_kras/post_filter/inputs/r2/monomer_fasta/af2_monomer_sequences.csv \
        --out ~/Desktop/pmhc_binder_kras/post_filter/inputs/r2/binder_pMHC_full.fasta

    python ~/Desktop/pmhc_binder_kras_cp/post_filter/scripts/af3/generate_full_complex_fasta.py \
        --csv ~/Desktop/pmhc_binder_kras_cp/post_filter/inputs/r2.5/af2_monomer/af2_monomer_sequences.csv \
        --out ~/Desktop/pmhc_binder_kras_cp/post_filter/inputs/r2.5/binder_pMHC_full.fasta

    python ~/Desktop/pmhc_binder_kras_cp/post_filter/scripts/af3/generate_full_complex_fasta.py \
        --csv ~/Desktop/pmhc_binder_kras_cp/post_filter/inputs/r3/monomer_fasta/af2_monomer_sequences.csv \
        --out ~/Desktop/pmhc_binder_kras_cp/post_filter/inputs/r3/binder_pMHC_full.fasta
"""

import argparse
import os
import pandas as pd

# ── Fixed sequences ───────────────────────────────────────────────────────────

MHC_ALPHA = (
    "GSHSMRYFYTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMLEPRAPWIEQEGPEYWDQETRNVKAQ"
    "SQTDRVDLGTLRGYYNQSEDGSHTIQIMYGCDVGPDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQ"
    "ITKRKWEAAHAAEQQRAYLEGRCVEWLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEI"
    "TLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWE"
)

PEPTIDE = "VVGADGVGK"

B2M = (
    "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEF"
    "TPTEKDEYACRVNHVTLSQPKIVKWDRDM"
)

# Strip design suffix for FASTA header — drop trailing _af2pred
def header_name(design: str) -> str:
    return design.removesuffix('_af2pred').lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True,
                        help='Path to af2_monomer_sequences.csv')
    parser.add_argument('--out', required=True,
                        help='Output FASTA path')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    your_designs = df[df['source'] == 'your_designs'].copy()
    print(f"Found {len(your_designs)} designs with source==your_designs")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.out, 'w') as f:
        for _, row in your_designs.iterrows():
            name = header_name(row['name'])
            binder = row['sequence'].strip()
            f.write(f'>{name}_A\n{binder}\n')
            f.write(f'>{name}_B\n{MHC_ALPHA}\n')
            f.write(f'>{name}_C\n{PEPTIDE}\n')
            f.write(f'>{name}_D\n{B2M}\n')

    print(f"Written {len(your_designs)} 4-chain entries → {args.out}")


if __name__ == '__main__':
    main()
