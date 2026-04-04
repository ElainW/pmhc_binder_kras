"""
run_mpnn_with_charge_filter.py
================================
Wrapper around standalone ProteinMPNN (protein_mpnn_run.py) that enforces
a minimum net charge threshold via rejection sampling. For each backbone,
keeps re-sampling with different seeds until the requested number of
sequences meeting the charge criterion are collected, or until
max_attempts is reached.

Usage
-----
python run_mpnn_with_charge_filter.py \
    --pdb_dir       /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/filtered \
    --out_dir       /n/groups/marks/users/aaron/pmhc/proteinmpnn/outputs/r3 \
    --mpnn_path     /path/to/python \
    --mpnn_script   /path/to/ProteinMPNN/protein_mpnn_run.py \
    --n_seqs        8 \
    --temperature   0.1 \
    --omit_AAs      C \
    --design_chains A \
    --min_charge    -2.0 \
    --max_attempts  5
"""

import os
import re
import sys
import glob
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path


# ── Net charge ────────────────────────────────────────────────────────────────

def net_charge(seq: str, ph: float = 7.4) -> float:
    """Approximate net charge at pH 7.4 (Henderson-Hasselbalch, standard pKa)."""
    pos = seq.count('R') + seq.count('K') + seq.count('H') * 0.1
    neg = seq.count('D') + seq.count('E')
    return pos - neg


# ── FASTA parsing ─────────────────────────────────────────────────────────────

def parse_fasta(fasta_path: str) -> list[dict]:
    """Parse a ProteinMPNN FASTA output into a list of {header, seq} dicts."""
    records = []
    header, lines = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if header is not None:
                    records.append({'header': header, 'seq': ''.join(lines)})
                header, lines = line[1:], []
            else:
                lines.append(line)
    if header is not None:
        records.append({'header': header, 'seq': ''.join(lines)})
    return records


def write_fasta(records: list[dict], path: str):
    with open(path, 'w') as f:
        for r in records:
            f.write(f">{r['header']}\n{r['seq']}\n")


# ── MPNN runner ───────────────────────────────────────────────────────────────

def run_mpnn(python_bin: str,
             mpnn_script: str,
             pdb_path: str,
             out_dir: str,
             n_seqs: int,
             temperature: float,
             omit_AAs: str,
             design_chains: str,
             seed: int) -> str | None:
    """
    Run standalone ProteinMPNN (protein_mpnn_run.py) on a single PDB and
    return the path to the output FASTA, or None if the run fails.

    Key standalone ProteinMPNN flags:
      --pdb_path_chains  chains to DESIGN (all others are fixed)
                         e.g. "A" designs chain A, fixes chain B
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        python_bin, mpnn_script,
        '--pdb_path',           pdb_path,
        '--out_folder',         out_dir,
        '--num_seq_per_target', str(n_seqs),
        '--sampling_temp',      str(temperature),
        '--omit_AAs',           omit_AAs,
        '--pdb_path_chains',    design_chains,
        '--seed',               str(seed),
        '--batch_size',         '1',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] MPNN failed on {pdb_path}:\n{result.stderr[:300]}")
        return None

    # standalone ProteinMPNN writes FASTA to out_folder/seqs/<pdb_stem>.fa
    stem = Path(pdb_path).stem
    fasta = os.path.join(out_dir, 'seqs', f'{stem}.fa')
    return fasta if os.path.exists(fasta) else None


# ── Per-backbone rejection sampling ──────────────────────────────────────────

def collect_sequences_with_charge_filter(
        python_bin: str,
        mpnn_script: str,
        pdb_path: str,
        out_dir: str,
        n_seqs: int,
        temperature: float,
        omit_AAs: str,
        design_chains: str,
        min_charge: float,
        max_attempts: int,
        base_seed: int = 42) -> list[dict]:
    """
    Collect exactly `n_seqs` sequences per backbone that satisfy net_charge <= min_charge.
    Re-runs MPNN with a different seed each attempt if needed.
    Returns a list of {header, seq, charge, attempt} dicts.
    """
    collected = []
    seen_seqs = set()
    stem = Path(pdb_path).stem

    for attempt in range(1, max_attempts + 1):
        if len(collected) >= n_seqs:
            break

        seed = base_seed + attempt * 1000
        tmp_dir = tempfile.mkdtemp(prefix=f'mpnn_{stem}_attempt{attempt}_')

        try:
            fasta_path = run_mpnn(
                python_bin, mpnn_script, pdb_path,
                tmp_dir, n_seqs, temperature, omit_AAs,
                design_chains, seed
            )
            if fasta_path is None:
                continue

            records = parse_fasta(fasta_path)
            # skip the poly-G reference sequence (first record)
            seqs = [r for r in records if not re.match(r'^G+$', r['seq'])]

            for r in seqs:
                if len(collected) >= n_seqs:
                    break
                seq = r['seq']
                charge = net_charge(seq)
                if seq in seen_seqs:
                    continue
                if charge <= min_charge:
                    seen_seqs.add(seq)
                    collected.append({
                        'header':  f"{stem}_a{attempt}_{r['header']}",
                        'seq':     seq,
                        'charge':  charge,
                        'attempt': attempt,
                    })

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if len(collected) < n_seqs:
        print(f"  [WARN] {stem}: only {len(collected)}/{n_seqs} sequences passed "
              f"charge <= {min_charge} after {max_attempts} attempts")

    return collected


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ProteinMPNN wrapper with net charge rejection sampling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--pdb_dir',      required=True,
                   help='Directory of input PDB files.')
    p.add_argument('--out_dir',      required=True,
                   help='Output directory for filtered FASTAs and log.')
    p.add_argument('--mpnn_path',    required=True,
                   help='Path to Python binary for MPNN environment.')
    p.add_argument('--mpnn_script',  required=True,
                   help='Path to mpnn_fr.py or protein_mpnn_run.py.')
    p.add_argument('--n_seqs',       type=int,   default=8,
                   help='Number of passing sequences to collect per backbone (default: 8).')
    p.add_argument('--temperature',  type=float, default=0.1,
                   help='Sampling temperature (default: 0.1).')
    p.add_argument('--omit_AAs',     default='C',
                   help='Amino acids to omit (default: C).')
    p.add_argument('--design_chains', default='A',
                   help='Chains to design in standalone ProteinMPNN (all others '
                        'are fixed). Default: A (designs binder, fixes MHC+peptide).')
    p.add_argument('--min_charge',   type=float, default=-2.0,
                   help='Maximum net charge allowed (sequences with charge > this are rejected). '
                        'Default: -2 (i.e. charge must be <= -2).')
    p.add_argument('--max_attempts', type=int,   default=5,
                   help='Max MPNN re-runs per backbone before giving up (default: 5).')
    p.add_argument('--base_seed',    type=int,   default=42,
                   help='Base random seed (incremented per attempt) (default: 42).')
    p.add_argument('--pattern',      default='*.pdb',
                   help='Glob pattern for PDB files (default: *.pdb).')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pdb_files = sorted(glob.glob(os.path.join(args.pdb_dir, args.pattern)))
    if not pdb_files:
        raise ValueError(f"No PDB files found in {args.pdb_dir} with pattern {args.pattern}")

    print(f"Processing {len(pdb_files)} backbones")
    print(f"  n_seqs={args.n_seqs}, T={args.temperature}, "
          f"omit={args.omit_AAs}, design_chains={args.design_chains}, "
          f"min_charge<={args.min_charge}, max_attempts={args.max_attempts}")

    all_results = []
    combined_fasta_path = os.path.join(args.out_dir, 'all_sequences.fa')

    with open(combined_fasta_path, 'w') as combined_fa:
        for i, pdb_path in enumerate(pdb_files):
            stem = Path(pdb_path).stem
            print(f"\n[{i+1}/{len(pdb_files)}] {stem}")

            collected = collect_sequences_with_charge_filter(
                python_bin    = args.mpnn_path,
                mpnn_script   = args.mpnn_script,
                pdb_path      = pdb_path,
                out_dir       = args.out_dir,
                n_seqs        = args.n_seqs,
                temperature   = args.temperature,
                omit_AAs      = args.omit_AAs,
                design_chains = args.design_chains,
                min_charge    = args.min_charge,
                max_attempts  = args.max_attempts,
                base_seed     = args.base_seed,
            )

            for r in collected:
                print(f"  seq {r['attempt']}: charge={r['charge']:.1f}  "
                      f"{r['seq'][:30]}...")
                combined_fa.write(f">{r['header']}\n{r['seq']}\n")
                all_results.append({
                    'backbone': stem,
                    'header':   r['header'],
                    'seq':      r['seq'],
                    'charge':   r['charge'],
                    'attempt':  r['attempt'],
                    'n_C':      r['seq'].count('C'),
                    'n_M':      r['seq'].count('M'),
                    'length':   len(r['seq']),
                })

    # write log
    import pandas as pd
    log = pd.DataFrame(all_results)
    log_path = os.path.join(args.out_dir, 'mpnn_charge_filtered_log.csv')
    log.to_csv(log_path, index=False)

    print(f"\n{'='*55}")
    print(f"  Total backbones processed: {len(pdb_files)}")
    print(f"  Total sequences collected: {len(all_results)}")
    print(f"  Expected (n_seqs × n_pdbs): {args.n_seqs * len(pdb_files)}")
    backbones_full = (log.groupby('backbone').size() == args.n_seqs).sum()
    print(f"  Backbones with full {args.n_seqs} seqs: {backbones_full}/{len(pdb_files)}")
    print(f"  Net charge — mean: {log['charge'].mean():.1f}  "
          f"min: {log['charge'].min():.1f}  max: {log['charge'].max():.1f}")
    print(f"  Sequences needing >1 attempt: "
          f"{(log['attempt'] > 1).sum()} ({100*(log['attempt']>1).mean():.1f}%)")
    print(f"  Combined FASTA: {combined_fasta_path}")
    print(f"  Log CSV:        {log_path}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
