"""
prepare_af2_monomer_inputs.py

Extracts binder sequences from:
  1. Your 49 passing designs (from AF2 init guess PDBs)
  2. Author-provided sequences from science_adv0185_data_s1.xlsx

Writes a combined FASTA file for AF2 monomer prediction.

Usage:
    python prepare_af2_monomer_inputs.py \
        --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
        --pdb_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
        --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/monomer_fasta

    python prepare_af2_monomer_inputs.py \
        --merged_scores /n/groups/marks/users/aaron/pmhc_cp/specificity/analysis/r3/merged_scores_ranked.tsv \
        --pdb_dir /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/outputs/r3/af2_kras/top_pdbs/ \
        --out_dir /n/groups/marks/users/aaron/pmhc_cp/post_filter/inputs/r3/monomer_fasta
"""

import os
import argparse
import pandas as pd
from Bio.PDB import PDBParser

DELTA_PAE_CUTOFF = -0.5

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

def get_binder_seq(pdb_path: str, chain_id: str = 'A') -> str:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('s', pdb_path)
    seq = ''
    for model in struct:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if res.id[0] == ' ':
                    seq += THREE_TO_ONE.get(res.resname, 'X')
    return seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_scores', required=True)
    parser.add_argument('--pdb_dir',       required=True)
    parser.add_argument('--out_dir',     required=True)

    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Your x designs ───────────────────────────────────────────────────
    df = pd.read_csv(args.merged_scores, sep='\t')
    passing = df[df['delta_pae_int_peptide'] < DELTA_PAE_CUTOFF].copy()
    print(f"Loading sequences for {len(passing)} passing designs...")

    your_records = []
    for _, row in passing.iterrows():
        design = row['design']
        pdb_path = os.path.join(args.pdb_dir, design + '.pdb')
        if not os.path.exists(pdb_path):
            print(f"  WARNING: PDB not found for {design}")
            continue
        seq = get_binder_seq(pdb_path)
        your_records.append({'name': design, 'sequence': seq, 'source': 'your_designs'})

#     # ── 2. Author sequences from xlsx ────────────────────────────────────────
#     print("Loading author sequences from csv...")
#     data = pd.read_csv(args.author_csv)
#     print(data)

#     name_col = 'Binder name'
#     seq_col  = 'Amino acid sequence'

#     author_records = []
#     for _, row in data.iterrows():
#         name = str(row[name_col]).strip()
#         seq  = str(row[seq_col]).strip()
#         if name and seq and seq != 'nan':
#             author_records.append({
#                 'name': f'author_{name}',
#                 'sequence': seq,
#                 'source': 'author_liu2025'
#             })

#     print(f"  {len(author_records)} author sequences loaded")

    # ── 3. Combine and write FASTA ────────────────────────────────────────────
    total = len(your_records)

    fasta_path = os.path.join(args.out_dir, 'af2_monomer_inputs.fasta')
    with open(fasta_path, 'w') as f:
        for rec in your_records:
            f.write(f">{rec['name']}\n{rec['sequence']}\n")

    print(f"\nWritten {total} sequences to {fasta_path}")
    print(f"  Your designs: {len(your_records)}")

    # Also write a CSV for reference
    seq_df = pd.DataFrame(your_records)
    seq_df.to_csv(os.path.join(args.out_dir, 'af2_monomer_sequences.csv'), index=False)

#     # Print author sequences for verification
#     print("\n── Author sequences ──")
#     for rec in author_records:
#         print(f"  {rec['name']}: {rec['sequence'][:30]}... (len={len(rec['sequence'])})")

if __name__ == '__main__':
    main()
