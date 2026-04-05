"""
generate_mpnn_pepscan_csv.py
Generates the input CSV for zero_shot_mutation_effect_prediction.py
for KRAS G12D VVGADGVGK alanine scan across all designs.

Usage:
    python generate_mpnn_pepscan_csv.py \
        --pdb_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
        --out_csv /n/groups/marks/users/aaron/pmhc/mpnn_spec/pepscan/pepscan_kras.csv
"""

import os
import argparse
import pandas as pd

PEPTIDE_SEQ = 'VVGADGVGK'  # p1-p9, D is at index 4 (p5)

def get_peptide_resnums(pdb_path: str, peptide_len: int = 9) -> list[tuple[int, str]]:
    """
    Returns list of (resnum, resname_1letter) for the last peptide_len
    residues of chain B in the PDB.
    """
    three_to_one = {
        'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
        'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
        'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
        'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
    }
    seen = {}  # resnum -> resname, order-of-appearance
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            chain = line[21]
            if chain != 'B':
                continue
            resnum = int(line[22:26].strip())
            resname = line[17:20].strip()
            if resnum not in seen:
                seen[resnum] = three_to_one.get(resname, 'X')

    resnums = list(seen.keys())
    aa1s = list(seen.values())

    # last peptide_len residues
    pep_resnums = resnums[-peptide_len:]
    pep_aas = aa1s[-peptide_len:]

    return list(zip(pep_resnums, pep_aas))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir', required=True)
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--peptide_len', type=int, default=9)
    args = parser.parse_args()

    pdb_files = sorted([
        f for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')
    ])
    print(f"Found {len(pdb_files)} PDB files")

    # Verify peptide sequence is consistent across all PDBs using first file
    first_pdb = os.path.join(args.pdb_dir, pdb_files[0])
    pep_residues = get_peptide_resnums(first_pdb, args.peptide_len)
    pep_seq_found = ''.join(aa for _, aa in pep_residues)
    print(f"Peptide residues from {pdb_files[0]}:")
    for resnum, aa in pep_residues:
        print(f"  resnum={resnum}, aa={aa}")
    print(f"Peptide sequence: {pep_seq_found}")
    assert pep_seq_found == PEPTIDE_SEQ, \
        f"Expected {PEPTIDE_SEQ}, got {pep_seq_found}"

    rows = []
    for pdb_file in pdb_files:
        pdb_name = pdb_file.replace('.pdb', '')
        pdb_path = os.path.join(args.pdb_dir, pdb_file)
        pep_residues = get_peptide_resnums(pdb_path, args.peptide_len)

        for resnum, aa_wt in pep_residues:
            aa_mt = 'A'
            rows.append({
                'mutant': f'{aa_wt}{resnum}{aa_mt}',
                'is_WT': False,
                'wt_pdb': pdb_name,
                'mutant_chain': 'B'
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Written {len(df)} rows to {args.out_csv}")
    print(f"({len(pdb_files)} designs × {args.peptide_len} positions)")

if __name__ == '__main__':
    main()
