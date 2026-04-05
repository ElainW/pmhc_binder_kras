"""
generate_pmhc_fold_csvs.py

Generates:
1. On-target CSV  — uses existing AF2 initial guess PDBs as-is
2. Off-target CSV — creates merged PDBs (chain A from design + chain B from 8I5E_chainB_wt)
                    and generates corresponding CSV rows

Usage:
    python generate_pmhc_fold_csvs.py \
        --pdb_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
        --wt_chainB /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/inputs/extracted_8I5E.pdb \
        --off_target_pdb_dir /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/inputs/r2/off_target_pdbs/ \
        --out_dir /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/inputs/r2/
"""

import os
import argparse
import pandas as pd
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Select

# ── Constants ────────────────────────────────────────────────────────────────
ALLELE = "A*11:01"
ALLELE_MSA = "A*11:01"  # used in allele_msa_file_names column
ON_TARGET_PEP  = "VVGADGVGK"
OFF_TARGET_PEP = "VVGAGGVGK"
ALLELE_SEQ = (
    "HSMRYFYTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDQETRNVKAQSQ"
    "TDRVDLGTLRGYYNQSEDGSHTIQIMYGCDVGPDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQI"
    "TKRKWEAAHAAEQQRAYLEGRCVEWLRRYLENGKETLQ"
).replace(" ", "")

# ── Selectors ────────────────────────────────────────────────────────────────
class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_binder_seq(pdb_path: str) -> str:
    """Extract chain A sequence from a design PDB."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("s", pdb_path)
    three_to_one = {
        'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
        'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
        'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
        'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
    }
    seq = ''
    for model in struct:
        for chain in model:
            if chain.id == 'A':
                for res in chain:
                    if res.id[0] == ' ':
                        seq += three_to_one.get(res.resname, 'X')
    return seq


def write_chain(structure, chain_id, out_path):
    """Write a single chain from a structure to a PDB file."""
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, ChainSelect(chain_id))


def merge_chains_into_pdb(chain_a_pdb: str, chain_b_pdb: str, out_path: str):
    """
    Merge chain A from chain_a_pdb and chain B from chain_b_pdb
    into a single PDB with chains A and B.
    """
    parser = PDBParser(QUIET=True)
    struct_a = parser.get_structure("a", chain_a_pdb)
    struct_b = parser.get_structure("b", chain_b_pdb)

    # Write chain A atoms
    with open(out_path, 'w') as out:
        for model in struct_a:
            for chain in model:
                if chain.id == 'A':
                    for res in chain:
                        for atom in res:
                            out.write(atom_to_pdb_line(chain.id, res, atom))
        # Write chain B atoms
        for model in struct_b:
            for chain in model:
                if chain.id == 'B':
                    for res in chain:
                        for atom in res:
                            out.write(atom_to_pdb_line('B', res, atom))
        out.write("END\n")


def atom_to_pdb_line(chain_id: str, res, atom) -> str:
    """Format a BioPython atom as a PDB ATOM line."""
    res_id = res.id[1]
    res_name = res.resname
    atom_name = atom.get_name()
    x, y, z = atom.get_vector()
    bfactor = atom.get_bfactor()
    occupancy = atom.get_occupancy() or 1.0
    element = atom.element if atom.element else ' '

    # PDB format: columns are fixed-width
    atom_name_fmt = f" {atom_name:<3}" if len(atom_name) < 4 else atom_name
    line = (
        f"ATOM  {atom.serial_number:5d} {atom_name_fmt} {res_name:3s} "
        f"{chain_id}{res_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}          "
        f"{element:>2s}\n"
    )
    return line


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir',           required=True)
    parser.add_argument('--wt_chainB',         required=True)
    parser.add_argument('--off_target_pdb_dir', required=True)
    parser.add_argument('--out_dir',           required=True)
    args = parser.parse_args()

    os.makedirs(args.off_target_pdb_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    pdb_files = sorted([
        f for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')
    ])
    print(f"Found {len(pdb_files)} design PDBs")

    on_rows  = []
    off_rows = []

    for pdb_file in pdb_files:
        pdb_name = pdb_file.replace('.pdb', '')
        pdb_path = os.path.join(args.pdb_dir, pdb_file)

        binder_seq = get_binder_seq(pdb_path)

        # ── On-target row ──────────────────────────────────────────────────
        on_rows.append({
            'entry_index':         0,
            'peptide_seq':         ON_TARGET_PEP,
            'allele_sequence':     ALLELE_SEQ,
            'target_chainseq':     f"{ALLELE_SEQ}/{binder_seq}/{ON_TARGET_PEP}",
            'targetid':            f"{ALLELE_MSA}_{ON_TARGET_PEP}",
            'allele_msa_file_names': ALLELE_MSA,
            'description':         f"{pdb_name}_{ALLELE_MSA}_{ON_TARGET_PEP}",
            'pdb_name':            pdb_name,
        })

        # ── Off-target: merge chain A from design + chain B from 8I5E ─────
        off_pdb_name = f"{pdb_name}_wt"
        off_pdb_path = os.path.join(args.off_target_pdb_dir, off_pdb_name + '.pdb')

        if not os.path.exists(off_pdb_path):
            merge_chains_into_pdb(pdb_path, args.wt_chainB, off_pdb_path)

        off_rows.append({
            'entry_index':         0,
            'peptide_seq':         OFF_TARGET_PEP,
            'allele_sequence':     ALLELE_SEQ,
            'target_chainseq':     f"{ALLELE_SEQ}/{binder_seq}/{OFF_TARGET_PEP}",
            'targetid':            f"{ALLELE_MSA}_{OFF_TARGET_PEP}",
            'allele_msa_file_names': ALLELE_MSA,
            'description':         f"{pdb_name}_{ALLELE_MSA}_{OFF_TARGET_PEP}",
            'pdb_name':            off_pdb_name,
        })

    pd.DataFrame(on_rows).to_csv(
        os.path.join(args.out_dir, 'pmhc_fold_on_target.csv'), index=False)
    pd.DataFrame(off_rows).to_csv(
        os.path.join(args.out_dir, 'pmhc_fold_off_target.csv'), index=False)

    print(f"Written {len(on_rows)} on-target rows")
    print(f"Written {len(off_rows)} off-target rows")
    print(f"Off-target merged PDBs in {args.off_target_pdb_dir}")


if __name__ == '__main__':
    main()
