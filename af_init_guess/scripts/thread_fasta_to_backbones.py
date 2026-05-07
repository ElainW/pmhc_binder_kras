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

Two header formats are supported and auto-detected:

  OLD format (Rounds 1 / 2):
    >orient_158_pT12__0, score=2.8433, fixed_chains=['B'], ...   ← native, skipped
    >T=0.1, sample=1, score=1.0047, ...                          ← designed
    SEQUENCE...
    Backbone name taken from the last native header seen.
    Output PDB: {backbone}_sample{N:02d}.pdb

  NEW format (Round 2.5):
    >orient_255_pT12__33_sample06_a1_T0.1_T=0.1, sample=2, score=0.8468, ...
    SEQUENCE...
    All entries treated as designed (none skipped).
    Output PDB: {normalised_stem}.pdb
      where normalised_stem strips trailing _T=<val> and lowercases _T<val>→_t<val>
    Backbone: strip _sample\d+_a\d+_t[\d.]+ suffix from stem
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

# Matches the trailing _sample\d+_a\d+_t[\d.]+ to recover backbone stem.
# e.g. orient_255_pT12__33_sample06_a1_t0.1 → orient_255_pT12__33
_SUFFIX_RE = re.compile(r'_sample\d+_a\d+_t[\d.]+$', re.IGNORECASE)


# --- Header parsing ---

def _normalise_output_stem(raw_token: str) -> str:
    """
    Normalise the header token (everything before the first comma) into
    the output PDB stem.

    Steps:
      1. Strip whitespace.
      2. Remove a trailing _T=<value> segment  (e.g. _T=0.1).
      3. Lower-case any remaining T=<value> segment to t<value>
         (e.g. _T0.1 → _t0.1).

    Example:
      'orient_255_pT12__33_sample06_a1_T0.1_T=0.1'
      → strip _T=0.1  → 'orient_255_pT12__33_sample06_a1_T0.1'
      → lower T0.1    → 'orient_255_pT12__33_sample06_a1_t0.1'
    """
    stem = raw_token.strip()
    # Remove trailing _T=<value>
    stem = re.sub(r'_T=[\d.]+$', '', stem)
    # Lower-case _T<value> → _t<value>
    stem = re.sub(r'_T([\d.]+)', lambda m: f'_t{m.group(1)}', stem)
    return stem


def _backbone_from_stem(stem: str) -> str:
    """
    Derive backbone PDB stem from the output PDB stem by stripping
    the _sample\d+_a\d+_t[\d.]+ suffix.

    e.g. orient_255_pT12__33_sample06_a1_t0.1 → orient_255_pT12__33
    """
    return _SUFFIX_RE.sub('', stem)


# --- Format detection ---

# Old-format designed header starts with >T=
_OLD_DESIGNED_RE = re.compile(r'^>T=[\d.]+')

# New-format header: backbone name embeds _sample\d+_a\d+_[Tt] (before first comma)
_NEW_FORMAT_RE   = re.compile(r'_sample\d+_a\d+_[Tt][\d.]+', re.IGNORECASE)


def _is_new_format_header(raw_header: str) -> bool:
    """Return True if this is a Round-2.5 style header."""
    token = raw_header.lstrip('>').split(',')[0]
    return bool(_NEW_FORMAT_RE.search(token))


# --- Parsing ---

def parse_mpnn_fasta(fasta_file: str) -> dict:
    """
    Parse a ProteinMPNN output FASTA file. Auto-detects old vs new format.

    OLD format (Rounds 1/2):
        >orient_158_pT12__0, score=..., fixed_chains=[...]   ← native, skipped
        >T=0.1, sample=1, score=...                          ← designed
        SEQUENCE
        Output stem: {backbone}_sample{N:02d}
        Backbone:    token before first comma of native header

    NEW format (Round 2.5):
        >orient_255_pT12__33_sample06_a1_T0.1_T=0.1, sample=2, score=...
        SEQUENCE
        Output stem: normalised token before first comma
        Backbone:    output stem with _sample\d+_a\d+_t[\d.]+ stripped
        All entries treated as designed (none skipped).

    Returns:
        dict mapping output_pdb_stem → list of
            {'backbone': str, 'score': float|None, 'sequence': str}
    """
    designs          = defaultdict(list)
    current_stem     = None
    current_backbone = None
    current_score    = None
    old_sample_count = 0   # per-backbone counter for old format
    in_old_native    = False

    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                score_match   = re.search(r'\bscore=([\d.eE+\-]+)', line)
                current_score = float(score_match.group(1)) if score_match else None

                if _is_new_format_header(line):
                    # ── New format ──────────────────────────────────────────
                    raw_token        = line.lstrip('>').split(',')[0]
                    current_stem     = _normalise_output_stem(raw_token)
                    current_backbone = _backbone_from_stem(current_stem)
                    sample_match     = re.search(r'\bsample=(\d+)', line)
                    current_sample   = int(sample_match.group(1)) if sample_match else None
                    in_old_native    = False

                elif _OLD_DESIGNED_RE.match(line):
                    # ── Old format: designed entry (>T=0.1, sample=N, ...) ─
                    # current_backbone already set by preceding native header
                    old_sample_count += 1
                    current_stem   = (f"{current_backbone}"
                                      f"_sample{old_sample_count:02d}")
                    current_sample = old_sample_count
                    in_old_native  = False

                else:
                    # ── Old format: native header ───────────────────────────
                    raw_token        = line.lstrip('>').split(',')[0].strip()
                    current_backbone = raw_token
                    current_stem     = None
                    current_sample   = None
                    old_sample_count = 0
                    in_old_native    = True

            else:
                # Sequence line
                if current_stem and not in_old_native and line:
                    designs[current_stem].append({
                        'backbone': current_backbone,
                        'sample':   current_sample,
                        'score':    current_score,
                        'sequence': line,
                    })

    return designs


def parse_fasta_directory(fasta_dir: str) -> dict:
    """
    Parse all .fa files in a directory and merge into one designs dict.
    """
    all_designs = {}
    fasta_files = sorted(glob.glob(os.path.join(fasta_dir, '*.fa')))

    if not fasta_files:
        raise FileNotFoundError(f"No .fa files found in {fasta_dir}")

    print(f"Found {len(fasta_files)} FASTA files in {fasta_dir}")

    for fa in fasta_files:
        designs = parse_mpnn_fasta(fa)
        all_designs.update(designs)

    return all_designs


# --- Threading ---

def get_chain_a_residues(pdb_lines: list) -> list:
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


def thread_sequence_onto_backbone(backbone_pdb: str,
                                   new_sequence: str,
                                   output_pdb: str):
    """
    Thread a designed sequence onto a backbone PDB.

    - Chain A: residue names replaced with designed sequence,
               sidechain atoms removed (backbone only: N, CA, C, O)
    - Chains B, C: kept exactly as-is (MHC and peptide)

    Raises:
        ValueError: sequence length mismatch or unknown amino acid
    """
    with open(backbone_pdb) as f:
        lines = f.readlines()

    chain_a_residues = get_chain_a_residues(lines)

    if len(chain_a_residues) != len(new_sequence):
        raise ValueError(
            f"Sequence length mismatch in {os.path.basename(backbone_pdb)}: "
            f"{len(chain_a_residues)} residues in chain A, "
            f"{len(new_sequence)} in designed sequence"
        )

    for i, aa in enumerate(new_sequence):
        if aa not in AA1TO3:
            raise ValueError(
                f"Unknown amino acid '{aa}' at position {i+1} "
                f"in sequence for {os.path.basename(backbone_pdb)}"
            )

    res_to_aa3 = {
        res_id: AA1TO3[aa]
        for res_id, aa in zip(chain_a_residues, new_sequence)
    }

    new_lines = []
    for line in lines:
        if not line.startswith('ATOM'):
            new_lines.append(line)
            continue

        chain    = line[21]
        res_id   = (line[21], line[22:26].strip(), line[26].strip())
        atom_name = line[12:16].strip()

        if chain == 'A':
            if atom_name not in BACKBONE_ATOMS:
                continue  # strip sidechains
            new_resname = res_to_aa3[res_id]
            line = line[:17] + new_resname + line[20:]

        new_lines.append(line)

    with open(output_pdb, 'w') as f:
        f.writelines(new_lines)


# --- Verification ---

def verify_threaded_pdb(output_pdb: str, expected_sequence: str) -> bool:
    """
    Read back a threaded PDB and confirm chain A sequence matches expected.
    """
    chain_a_seq = []
    seen = set()

    with open(output_pdb) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == 'A':
                res_id  = line[22:26].strip()
                resname = line[17:20].strip()
                if res_id not in seen:
                    seen.add(res_id)
                    chain_a_seq.append(AA3TO1.get(resname, '?'))

    return ''.join(chain_a_seq) == expected_sequence


# --- Main ---

def main(args):

    # Load designs
    if args.fasta_file:
        print(f"Parsing single FASTA file: {args.fasta_file}")
        all_designs = parse_mpnn_fasta(args.fasta_file)
    else:
        print(f"Parsing FASTA directory: {args.fasta_dir}")
        all_designs = parse_fasta_directory(args.fasta_dir)

    print(f"Output PDB stems loaded: {len(all_designs)}")
    print(f"Total sequences:         {sum(len(v) for v in all_designs.values())}")

    os.makedirs(args.output_dir, exist_ok=True)

    success      = 0
    failed       = 0
    skipped      = 0
    verify_failed = 0

    for output_stem, sequences in sorted(all_designs.items()):

        # Each entry carries its own backbone name (all should be identical
        # within a stem, but we read it per-entry for safety)
        for design in sequences:
            backbone_name = design['backbone'].replace("pt", "pT")
            backbone_pdb  = os.path.join(args.backbone_dir,
                                         f"{backbone_name}.pdb")

            if not os.path.exists(backbone_pdb):
                if args.verbose:
                    print(f"  [SKIP] Backbone not found: {backbone_name}.pdb")
                skipped += 1
                continue

            out_name = (f"{output_stem}_s{design['sample']}.pdb"
                        if design['sample'] is not None
                        else f"{output_stem}.pdb")
            out_path = os.path.join(args.output_dir, out_name)

            try:
                thread_sequence_onto_backbone(
                    backbone_pdb,
                    design['sequence'],
                    out_path
                )

                if args.verify:
                    if not verify_threaded_pdb(out_path, design['sequence']):
                        print(f"  [VERIFY FAIL] {out_name}")
                        verify_failed += 1

                success += 1

                if args.verbose:
                    score_str = (f"score={design['score']:.4f}"
                                 if design['score'] is not None else 'score=N/A')
                    print(f"  {out_name}  ({score_str})")

            except ValueError as e:
                print(f"  [FAIL] {output_stem}: {e}")
                failed += 1
            except Exception as e:
                print(f"  [ERROR] {output_stem}: {type(e).__name__}: {e}")
                failed += 1

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
        print(f"Warning: {skipped} backbone PDBs not found in {args.backbone_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Thread ProteinMPNN sequences onto RFdiffusion backbone PDBs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python thread_mpnn_to_backbone.py \
      --fasta_dir outputs/kras_r25/seqs/ \
      --backbone_dir rfdiffusion/outputs/rep/ \
      --output_dir threaded_pdbs/

  python thread_mpnn_to_backbone.py \
      --fasta_file outputs/all_seqs.fa \
      --backbone_dir rfdiffusion/outputs/rep/ \
      --output_dir threaded_pdbs/ \
      --verify --verbose
        """
    )

    fasta_group = parser.add_mutually_exclusive_group(required=True)
    fasta_group.add_argument(
        '--fasta_file', type=str,
        help='Path to a single combined ProteinMPNN output FASTA file'
    )
    fasta_group.add_argument(
        '--fasta_dir', type=str,
        help='Path to directory containing per-backbone .fa files'
    )
    parser.add_argument('--backbone_dir', type=str, required=True,
                        help='Directory containing backbone PDB files')
    parser.add_argument('--output_dir',  type=str, required=True,
                        help='Output directory for threaded PDB files')
    parser.add_argument('--verify',  action='store_true',
                        help='Verify each threaded PDB by reading back chain A')
    parser.add_argument('--verbose', action='store_true',
                        help='Print one line per threaded PDB')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)