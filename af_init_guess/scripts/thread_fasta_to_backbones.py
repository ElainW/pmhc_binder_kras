#!/usr/bin/env python3
"""
Thread ProteinMPNN designed sequences onto RFdiffusion backbone PDBs.

Usage:
    python thread_mpnn_to_backbone.py \
        --fasta_dir path/to/mpnn/seqs/ \
        --backbone_dir path/to/backbone/pdbs/ \
        --output_dir path/to/threaded_pdbs/

    # Or with a single combined FASTA file:
    python thread_mpnn_to_backbone.py \
        --fasta_file path/to/all_seqs.fa \
        --backbone_dir path/to/backbone/pdbs/ \
        --output_dir path/to/threaded_pdbs/
"""

import os
import re
import glob
import argparse
from collections import defaultdict


# --- Constants ---

AA1TO3 = {
    'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS',
    'Q':'GLN','E':'GLU','G':'GLY','H':'HIS','I':'ILE',
    'L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO',
    'S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'
}

AA3TO1 = {v: k for k, v in AA1TO3.items()}

BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}


# --- Parsing ---

def parse_mpnn_fasta(fasta_file):
    """
    Parse a ProteinMPNN output FASTA file.

    Handles both:
      - One file per backbone (e.g. kras_orient_0.fa)
      - Single combined file with all backbones

    Returns:
        dict: {backbone_name: [{'score': float, 'sequence': str}, ...]}
              Native sequence (first entry per backbone) is skipped.
    """
    designs = defaultdict(list)
    current_backbone = None
    current_score = None
    is_native = False

    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if line.startswith('>T='):
                    # Designed sample
                    score_match = re.search(r'score=([\d.]+)', line)
                    current_score = float(score_match.group(1)) \
                        if score_match else None
                    is_native = False
                else:
                    # Native sequence header — extract backbone name
                    backbone_name = line.lstrip('>').split(',')[0].strip()
                    current_backbone = backbone_name
                    current_score = None
                    is_native = True

            else:
                # Sequence line
                if current_backbone and not is_native:
                    designs[current_backbone].append({
                        'score': current_score,
                        'sequence': line
                    })

    return designs


def parse_fasta_directory(fasta_dir):
    """
    Parse all .fa files in a directory and merge into one designs dict.
    """
    all_designs = {}
    fasta_files = sorted(glob.glob(os.path.join(fasta_dir, '*.fa')))

    if not fasta_files:
        raise FileNotFoundError(
            f"No .fa files found in {fasta_dir}"
        )

    print(f"Found {len(fasta_files)} FASTA files in {fasta_dir}")

    for fa in fasta_files:
        designs = parse_mpnn_fasta(fa)
        all_designs.update(designs)

    return all_designs


# --- Threading ---

def get_chain_a_residues(pdb_lines):
    """
    Extract ordered list of unique chain A residue IDs from PDB lines.
    Returns list of (chain, resnum, icode) tuples.
    """
    residues = []
    seen = set()
    for line in pdb_lines:
        if line.startswith('ATOM') and line[21] == 'A':
            res_id = (line[21], line[22:26].strip(), line[26].strip())
            if res_id not in seen:
                seen.add(res_id)
                residues.append(res_id)
    return residues


def thread_sequence_onto_backbone(backbone_pdb, new_sequence, output_pdb):
    """
    Thread a designed sequence onto a backbone PDB.

    - Chain A: residue names replaced with designed sequence,
               sidechain atoms removed (backbone only: N, CA, C, O)
    - Chains B, C: kept exactly as-is (MHC and peptide)

    Args:
        backbone_pdb:  Path to input backbone PDB
        new_sequence:  Designed amino acid sequence string (1-letter)
        output_pdb:    Path to write threaded PDB

    Raises:
        ValueError: if sequence length doesn't match chain A residue count
                    or if unknown amino acid encountered
    """
    with open(backbone_pdb) as f:
        lines = f.readlines()

    # Get chain A residues and validate length
    chain_a_residues = get_chain_a_residues(lines)

    if len(chain_a_residues) != len(new_sequence):
        raise ValueError(
            f"Sequence length mismatch in {os.path.basename(backbone_pdb)}: "
            f"{len(chain_a_residues)} residues in chain A, "
            f"{len(new_sequence)} in designed sequence"
        )

    # Validate all amino acids
    for i, aa in enumerate(new_sequence):
        if aa not in AA1TO3:
            raise ValueError(
                f"Unknown amino acid '{aa}' at position {i+1} "
                f"in sequence for {os.path.basename(backbone_pdb)}"
            )

    # Build residue ID → new 3-letter code mapping
    res_to_aa3 = {
        res_id: AA1TO3[aa]
        for res_id, aa in zip(chain_a_residues, new_sequence)
    }

    # Write threaded PDB
    new_lines = []
    for line in lines:
        if not line.startswith('ATOM'):
            new_lines.append(line)
            continue

        chain = line[21]
        res_id = (line[21], line[22:26].strip(), line[26].strip())
        atom_name = line[12:16].strip()

        if chain == 'A':
            if atom_name not in BACKBONE_ATOMS:
                continue  # remove sidechain atoms
            new_resname = res_to_aa3[res_id]
            line = line[:17] + new_resname + line[20:]

        new_lines.append(line)

    with open(output_pdb, 'w') as f:
        f.writelines(new_lines)


# --- Verification ---

def verify_threaded_pdb(output_pdb, expected_sequence):
    """
    Read back a threaded PDB and confirm chain A sequence matches expected.
    Returns True if match, False otherwise.
    """
    chain_a_seq = []
    seen = set()

    with open(output_pdb) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == 'A':
                res_id = line[22:26].strip()
                resname = line[17:20].strip()
                if res_id not in seen:
                    seen.add(res_id)
                    aa1 = AA3TO1.get(resname, '?')
                    chain_a_seq.append(aa1)

    return ''.join(chain_a_seq) == expected_sequence


# --- Main ---

def main(args):

    # --- Load designs ---
    if args.fasta_file:
        print(f"Parsing single FASTA file: {args.fasta_file}")
        all_designs = parse_mpnn_fasta(args.fasta_file)
    else:
        print(f"Parsing FASTA directory: {args.fasta_dir}")
        all_designs = parse_fasta_directory(args.fasta_dir)

    print(f"Backbones loaded:   {len(all_designs)}")
    print(f"Total sequences:    {sum(len(v) for v in all_designs.values())}")

    # --- Create output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Thread sequences ---
    success = 0
    failed = 0
    skipped = 0
    verify_failed = 0

    for backbone_name, sequences in sorted(all_designs.items()):

        backbone_pdb = os.path.join(
            args.backbone_dir, f"{backbone_name}.pdb"
        )

        if not os.path.exists(backbone_pdb):
            print(f"  [SKIP] Backbone not found: {backbone_name}.pdb")
            skipped += 1
            continue

        for i, design in enumerate(sequences):
            sample_num = i + 1
            out_name = f"{backbone_name}_sample{sample_num:02d}.pdb"
            out_path = os.path.join(args.output_dir, out_name)

            try:
                thread_sequence_onto_backbone(
                    backbone_pdb,
                    design['sequence'],
                    out_path
                )

                # Optional verification
                if args.verify:
                    if not verify_threaded_pdb(out_path, design['sequence']):
                        print(f"  [VERIFY FAIL] {out_name}")
                        verify_failed += 1

                success += 1

            except ValueError as e:
                print(f"  [FAIL] {backbone_name} sample {sample_num}: {e}")
                failed += 1
            except Exception as e:
                print(f"  [ERROR] {backbone_name} sample {sample_num}: "
                      f"{type(e).__name__}: {e}")
                failed += 1

        if args.verbose:
            print(f"  {backbone_name}: {len(sequences)} sequences threaded")

    # --- Summary ---
    total_pdbs = len(glob.glob(os.path.join(args.output_dir, '*.pdb')))
    print(f"\n{'='*50}")
    print(f"Threading complete")
    print(f"{'='*50}")
    print(f"  Successful:       {success}")
    print(f"  Failed:           {failed}")
    print(f"  Skipped:          {skipped}")
    if args.verify:
        print(f"  Verify failed:    {verify_failed}")
    print(f"  PDBs in output:   {total_pdbs}")
    print(f"  Output directory: {args.output_dir}")

    if failed > 0:
        print(f"\nWarning: {failed} sequences failed — check output above")
    if skipped > 0:
        print(f"Warning: {skipped} backbones not found in {args.backbone_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Thread ProteinMPNN sequences onto RFdiffusion backbone PDBs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One .fa file per backbone in a directory
  python thread_mpnn_to_backbone.py \\
      --fasta_dir outputs/kras_v3/seqs/ \\
      --backbone_dir rfdiffusion/outputs/rep/ \\
      --output_dir threaded_pdbs/

  # Single combined FASTA file
  python thread_mpnn_to_backbone.py \\
      --fasta_file outputs/all_seqs.fa \\
      --backbone_dir rfdiffusion/outputs/rep/ \\
      --output_dir threaded_pdbs/ \\
      --verify --verbose
        """
    )

    # Input — mutually exclusive: fasta_file vs fasta_dir
    fasta_group = parser.add_mutually_exclusive_group(required=True)
    fasta_group.add_argument(
        '--fasta_file', type=str,
        help='Path to a single combined ProteinMPNN output FASTA file'
    )
    fasta_group.add_argument(
        '--fasta_dir', type=str,
        help='Path to directory containing per-backbone .fa files'
    )

    parser.add_argument(
        '--backbone_dir', type=str, required=True,
        help='Path to directory containing backbone PDB files'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Path to output directory for threaded PDB files'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify each threaded PDB by reading back chain A sequence'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print progress for each backbone'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
