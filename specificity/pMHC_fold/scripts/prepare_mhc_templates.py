"""
prepare_mhc_templates.py
Prepares HLA-A*11:01 pMHC template PDBs for pmhc_fold.py.
Output: chain A = MHC (residues 3-180), chain B = peptide, renumbered sequentially.

Usage:
    python prepare_mhc_templates.py
"""

from Bio.PDB import PDBParser, PDBIO, Select
import os

TEMPLATE_DIR = "/n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/inputs/pmhc_pdbs/"

# (pdb_id, mhc_chain, mhc_resi_start, mhc_resi_end, peptide_chain)
STRUCTURES = [
    ("6jtp", "A", 1, 181, "C"),
    ("8i5c", "A", 1, 181, "C"),
    ("8i5d", "H", 1, 181, "P"),
    ("8i5e", "H", 1, 181, "P"),
]

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

def get_residues(structure, chain_id, resi_start=None, resi_end=None):
    """Return list of residues from a chain, optionally filtered by residue range."""
    residues = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if res.id[0] != ' ':  # skip HETATM
                    continue
                resnum = res.id[1]
                if resi_start is not None and resnum < resi_start:
                    continue
                if resi_end is not None and resnum > resi_end:
                    continue
                residues.append(res)
    return residues


def write_merged_pdb(mhc_residues, pep_residues, out_path):
    """
    Write merged PDB with:
      chain A = mhc_residues (renumbered from 1)
      chain B = pep_residues (renumbered from 1)
    """
    serial = [0]

    def format_atom_line(atom, chain_id, res_name, res_seq):
        serial[0] += 1
        atom_name = atom.get_name()
        atom_name_fmt = f" {atom_name:<3}" if len(atom_name) < 4 else atom_name
        x, y, z = atom.get_vector()
        occupancy = atom.get_occupancy() or 1.0
        bfactor = atom.get_bfactor()
        element = atom.element.strip() if atom.element else ''
        return (
            f"ATOM  {serial[0]:5d} {atom_name_fmt} {res_name:3s} "
            f"{chain_id}{res_seq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occupancy:6.2f}{bfactor:6.2f}          "
            f"{element:>2s}\n"
        )

    with open(out_path, 'w') as f:
        # Chain A — MHC
        for new_resnum, res in enumerate(mhc_residues, start=1):
            for atom in res:
                f.write(format_atom_line(atom, 'A', res.resname, new_resnum))

        # Chain B — peptide
        for new_resnum, res in enumerate(pep_residues, start=1):
            for atom in res:
                f.write(format_atom_line(atom, 'B', res.resname, new_resnum))

        f.write("END\n")


def verify_output(out_path, peptide_len):
    """Print chain lengths and peptide sequence for verification."""
    chain_residues = {'A': [], 'B': []}
    seen = {'A': {}, 'B': {}}
    with open(out_path) as fh:
        for line in fh:
            if not line.startswith('ATOM'):
                continue
            chain = line[21]
            resnum = int(line[22:26].strip())
            resname = line[17:20].strip()
            if chain in seen and resnum not in seen[chain]:
                seen[chain][resnum] = resname
    for ch in ['A', 'B']:
        resnums = sorted(seen[ch].keys())
        seq = ''.join(THREE_TO_ONE.get(seen[ch][r], 'X') for r in resnums)
        print(f"  Chain {ch}: {len(resnums)} residues — {seq}")


def main():
    parser = PDBParser(QUIET=True)

    for pdb_id, mhc_chain, resi_start, resi_end, pep_chain in STRUCTURES:
        in_path  = os.path.join(TEMPLATE_DIR, f"{pdb_id}.pdb")
        out_path = os.path.join(TEMPLATE_DIR, f"{pdb_id}_prep.pdb")

        print(f"\nProcessing {pdb_id}...")
        structure = parser.get_structure(pdb_id, in_path)

        mhc_residues = get_residues(structure, mhc_chain, resi_start, resi_end)
        pep_residues = get_residues(structure, pep_chain)

        print(f"  MHC residues (chain {mhc_chain}, {resi_start}-{resi_end}): {len(mhc_residues)}")
        print(f"  Peptide residues (chain {pep_chain}): {len(pep_residues)}")

        write_merged_pdb(mhc_residues, pep_residues, out_path)

        print(f"  Written to {out_path}")
        print(f"  Verification:")
        verify_output(out_path, len(pep_residues))

    print("\nDone. Check peptide sequences above match expected HLA-A*11:01 peptides.")


if __name__ == "__main__":
    main()