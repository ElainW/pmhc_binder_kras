#!/usr/bin/env python3
"""
redesign_w_charge_residue_filter.py

Wrapper around standalone ProteinMPNN (protein_mpnn_run.py) that:
  1. Enforces net charge <= min_charge via rejection sampling
  2. Omits cysteines via --omit_AAs C
  3. Optionally fixes interface residues so the binding geometry is preserved

Two modes:

  Mode A — new backbone PDBs (original behaviour):
    Pass --pdb_dir. No masking, all of chain A is designed.

  Mode B — redesign from charge_cysteine_audit.tsv:
    Pass --audit_tsv + --split_dir.
    For each failing design, the chainA PDB is used as backbone and
    interface_fixed_resnums from the audit are passed to ProteinMPNN as
    fixed positions (equivalent to BindCraft's mpnn_fix_interface=True).
    Only non-interface positions are redesigned to fix charge/Cys, so the
    binding interface is preserved.

    Select which designs to redesign with one of:
      --redesign_failing_charge   designs where pass_charge == False
      --redesign_failing_cys      designs where pass_cys == False
      --redesign_failing_any      designs where pass_all == False  (default)
      --redesign_all              all 49 designs regardless of flags

Fixed positions mechanism:
  make_fixed_positions_dict.py is called to write a tmp fixed_positions.jsonl:
    {"<pdb_stem>": {"A": [res1, res2, ...]}}
  ProteinMPNN reads this and holds those residues fixed (keeps their sequence
  from the input PDB) while freely designing the rest of chain A.
  Interface residues that are Cys are NOT fixed — they are included in the
  redesign since the policy is zero Cys regardless of location.

Usage — Mode A:
    python redesign_w_charge_residue_filter.py \
        --pdb_dir /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r3/ \
        --out_dir /n/groups/marks/users/aaron/pmhc/proteinmpnn/outputs/r3/ \
        --mpnn_path /path/to/proteinmpnn/python \
        --mpnn_script /path/to/ProteinMPNN/protein_mpnn_run.py \
        --n_seqs 8 --min_charge -3 --max_attempts 5

Usage — Mode B:
    python redesign_w_charge_residue_filter.py \
        --audit_tsv /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/charge_cysteine_audit.tsv \
        --split_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/ \
        --out_dir   /n/groups/marks/users/aaron/pmhc/proteinmpnn/outputs/r2_redesign/ \
        --mpnn_path /path/to/proteinmpnn/python \
        --mpnn_script /path/to/ProteinMPNN/protein_mpnn_run.py \
        --n_seqs 8 --min_charge -3 --max_attempts 5 \
        --redesign_failing_any
"""

from __future__ import annotations

import os
import re
import ast
import glob
import json
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


# ── Net charge ────────────────────────────────────────────────────────────────

def net_charge(seq: str) -> float:
    """Physiological net charge: R + K + 0.1*H - D - E"""
    return (seq.count('R') + seq.count('K') + 0.1 * seq.count('H')
            - seq.count('D') - seq.count('E'))


# ── FASTA helpers ─────────────────────────────────────────────────────────────

def parse_fasta(fasta_path: str) -> list[dict]:
    """Parse ProteinMPNN FASTA → [{header, seq}]."""
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


# ── Fixed positions JSONL ─────────────────────────────────────────────────────

def write_fixed_positions_jsonl(pdb_stem: str,
                                 fixed_resnums: list[int],
                                 out_path: str):
    """
    Write a ProteinMPNN fixed_positions_jsonl for one PDB.
    Format: {"<pdb_stem>": {"A": [resnum1, resnum2, ...]}}

    Residue numbers must match the PDB exactly (1-indexed, as in the
    chainA PDB written by split_pmhc_fold_pdbs.py).
    If fixed_resnums is empty, writes an empty chain A list which tells
    MPNN to design all of chain A freely.
    """
    data = {pdb_stem: {'A': sorted(fixed_resnums)}}
    with open(out_path, 'w') as f:
        f.write(json.dumps(data) + '\n')


def parse_resnums_from_audit(cell: str) -> list[int]:
    """
    Parse a cell from charge_cysteine_audit.tsv that looks like:
      '[]'  or  '[12, 34, 56]'
    Returns a list of ints.
    """
    try:
        val = ast.literal_eval(str(cell))
        return [int(x) for x in val] if val else []
    except Exception:
        return []


# ── Core MPNN runner ──────────────────────────────────────────────────────────

def run_mpnn(python_bin: str,
             mpnn_script: str,
             pdb_path: str,
             out_dir: str,
             n_seqs: int,
             temperature: float,
             omit_AAs: str,
             design_chains: str,
             seed: int,
             fixed_positions_jsonl: str | None = None) -> str | None:
    """
    Run standalone protein_mpnn_run.py on a single PDB.
    Returns path to output FASTA, or None on failure.

    fixed_positions_jsonl: path to a JSONL produced by write_fixed_positions_jsonl().
    When provided, those residues on chain A are held fixed (their sequence
    from the input PDB is preserved). All other chain A residues are designed.
    This is the same mechanism as BindCraft's mpnn_fix_interface.
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
    if fixed_positions_jsonl:
        cmd += ['--fixed_positions_jsonl', fixed_positions_jsonl]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [WARN] MPNN failed: {result.stderr[:300]}")
        return None

    stem   = Path(pdb_path).stem
    fasta  = os.path.join(out_dir, 'seqs', f'{stem}.fa')
    return fasta if os.path.exists(fasta) else None


# ── Rejection sampling loop ───────────────────────────────────────────────────

def collect_sequences(python_bin: str,
                       mpnn_script: str,
                       pdb_path: str,
                       out_dir: str,
                       n_seqs: int,
                       temperature: float,
                       omit_AAs: str,
                       design_chains: str,
                       min_charge: float,
                       max_attempts: int,
                       base_seed: int = 42,
                       fixed_resnums: list[int] | None = None) -> list[dict]:
    """
    Collect up to n_seqs sequences with net_charge <= min_charge.
    Re-runs MPNN with a different seed each attempt if needed.

    fixed_resnums: residue numbers on chain A to fix (interface positions).
      These are passed via --fixed_positions_jsonl. Cys positions are NOT
      added to fixed_resnums even if they are interface residues, because
      the policy is zero Cys regardless.
    """
    collected = []
    seen_seqs = set()
    stem      = Path(pdb_path).stem
    fixed_resnums = fixed_resnums or []

    for attempt in range(1, max_attempts + 1):
        if len(collected) >= n_seqs:
            break

        seed    = base_seed + attempt * 1000
        tmp_dir = tempfile.mkdtemp(prefix=f'mpnn_{stem}_a{attempt}_')

        try:
            # Write fixed positions JSONL into tmp dir
            fixed_jsonl = None
            if fixed_resnums:
                fixed_jsonl = os.path.join(tmp_dir, 'fixed_positions.jsonl')
                write_fixed_positions_jsonl(stem, fixed_resnums, fixed_jsonl)

            fasta_path = run_mpnn(
                python_bin, mpnn_script, pdb_path,
                tmp_dir, n_seqs, temperature, omit_AAs,
                design_chains, seed, fixed_jsonl
            )
            if fasta_path is None:
                continue

            records = parse_fasta(fasta_path)
            # Skip poly-G/poly-A reference (first record per backbone)
            seqs = [r for r in records
                    if not re.match(r'^[GA]+$', r['seq'])]

            for r in seqs:
                if len(collected) >= n_seqs:
                    break
                seq    = r['seq']
                charge = net_charge(seq)
                if seq in seen_seqs:
                    continue
                if charge <= min_charge:
                    seen_seqs.add(seq)
                    collected.append({
                        'header':  f"{stem}_a{attempt}_{r['header']}",
                        'seq':     seq,
                        'charge':  round(charge, 2),
                        'attempt': attempt,
                        'n_cys':   seq.count('C'),
                        'n_fixed': len(fixed_resnums),
                    })

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if len(collected) < n_seqs:
        print(f"    [WARN] {stem}: {len(collected)}/{n_seqs} seqs passed "
              f"charge <= {min_charge} after {max_attempts} attempts")

    return collected


# ── Mode A: process a directory of PDB files ──────────────────────────────────

def run_mode_a(args):
    pdb_files = sorted(glob.glob(os.path.join(args.pdb_dir, args.pattern)))
    if not pdb_files:
        raise ValueError(f"No PDB files in {args.pdb_dir} matching {args.pattern}")
    print(f"Mode A: designing {len(pdb_files)} backbone PDBs (no interface masking)")
    return [(pdb, []) for pdb in pdb_files]


# ── Mode B: redesign from audit TSV ──────────────────────────────────────────

def run_mode_b(args) -> list[tuple[str, list[int]]]:
    """
    Returns list of (chainA_pdb_path, fixed_resnums) for designs selected
    by the redesign flag. Cys positions are excluded from fixed_resnums
    even if they are interface residues (zero-Cys policy).
    """
    audit = pd.read_csv(args.audit_tsv, sep='\t')

    # Select designs to redesign
    if args.redesign_all:
        selected = audit
    elif args.redesign_failing_charge:
        selected = audit[audit['flag_charge_redesign'] == True]
    elif args.redesign_failing_cys:
        selected = audit[audit['flag_cys_redesign'] == True]
    else:  # default: redesign_failing_any
        selected = audit[audit['pass_all'] == False]

    print(f"Mode B: redesigning {len(selected)} designs from audit TSV")
    print(f"  (charge fail: {audit['flag_charge_redesign'].sum()}, "
          f"cys fail: {audit['flag_cys_redesign'].sum()}, "
          f"either fail: {(audit['pass_all']==False).sum()})")

    tasks = []
    for _, row in selected.iterrows():
        design = row['design']
        chain_a_pdb = os.path.join(args.split_dir, f'{design}_chainA.pdb')

        if not os.path.exists(chain_a_pdb):
            print(f"  [WARN] chainA PDB not found for {design}: {chain_a_pdb}")
            continue

        # Interface residues to fix — these preserve the binding geometry
        interface_resnums = parse_resnums_from_audit(row.get('interface_fixed_resnums', '[]'))

        # Cys positions — excluded from fixed_resnums regardless of location
        # (zero-Cys policy: Cys must be redesigned even if at interface)
        cys_resnums = parse_resnums_from_audit(row.get('cys_resnums', '[]'))
        cys_set     = set(cys_resnums)

        # Final fixed set: interface residues minus any Cys
        fixed_resnums = [r for r in interface_resnums if r not in cys_set]

        if cys_set & set(interface_resnums):
            cys_at_iface = sorted(cys_set & set(interface_resnums))
            print(f"  [INFO] {design}: {len(cys_at_iface)} Cys at interface "
                  f"({cys_at_iface}) — excluded from fixed set, will be redesigned")

        tasks.append((chain_a_pdb, fixed_resnums))

    return tasks


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Mode selection
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pdb_dir',   help='Mode A: directory of backbone PDB files')
    mode.add_argument('--audit_tsv', help='Mode B: charge_cysteine_audit.tsv')

    # Mode B options
    p.add_argument('--split_dir', help='Mode B: directory with {design}_chainA.pdb files')
    redesign = p.add_mutually_exclusive_group()
    redesign.add_argument('--redesign_failing_any',    action='store_true', default=True,
                          help='Redesign designs where pass_all==False (default)')
    redesign.add_argument('--redesign_failing_charge', action='store_true',
                          help='Redesign designs where pass_charge==False only')
    redesign.add_argument('--redesign_failing_cys',    action='store_true',
                          help='Redesign designs where pass_cys==False only')
    redesign.add_argument('--redesign_all',            action='store_true',
                          help='Redesign all designs regardless of flags')

    # MPNN settings
    p.add_argument('--out_dir',       required=True)
    p.add_argument('--mpnn_path',     required=True,
                   help='Python binary for ProteinMPNN environment')
    p.add_argument('--mpnn_script',   required=True,
                   help='Path to protein_mpnn_run.py')
    p.add_argument('--n_seqs',        type=int,   default=8)
    p.add_argument('--temperature',   type=float, default=0.1)
    p.add_argument('--omit_AAs',      default='C',
                   help='AAs to omit (default: C — no cysteines)')
    p.add_argument('--design_chains', default='A',
                   help='Chains to design; others are fixed context (default: A)')
    p.add_argument('--min_charge',    type=float, default=-2.0,
                   help='Max allowed net charge (default: -2.0)')
    p.add_argument('--max_attempts',  type=int,   default=5)
    p.add_argument('--base_seed',     type=int,   default=42)
    p.add_argument('--pattern',       default='*.pdb',
                   help='Mode A: glob pattern for PDB files (default: *.pdb)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Build task list: [(pdb_path, fixed_resnums), ...]
    if args.pdb_dir:
        tasks = run_mode_a(args)
    else:
        if not args.split_dir:
            raise ValueError('--split_dir required with --audit_tsv')
        tasks = run_mode_b(args)

    print(f"\nProteinMPNN settings:")
    print(f"  n_seqs={args.n_seqs}, T={args.temperature}, omit={args.omit_AAs}, "
          f"design_chains={args.design_chains}, min_charge<={args.min_charge}, "
          f"max_attempts={args.max_attempts}")

    all_results   = []
    combined_path = os.path.join(args.out_dir, 'all_sequences.fa')

    with open(combined_path, 'w') as combined_fa:
        for i, (pdb_path, fixed_resnums) in enumerate(tasks):
            stem = Path(pdb_path).stem
            n_fixed = len(fixed_resnums)
            print(f"\n[{i+1}/{len(tasks)}] {stem}  "
                  f"(fixed interface residues: {n_fixed})")

            collected = collect_sequences(
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
                fixed_resnums = fixed_resnums,
            )

            for r in collected:
                print(f"  attempt {r['attempt']}: charge={r['charge']:+.1f}  "
                      f"Cys={r['n_cys']}  {r['seq'][:30]}...")
                combined_fa.write(f">{r['header']}\n{r['seq']}\n")
                all_results.append({
                    'backbone':    stem,
                    'pdb_path':    pdb_path,
                    'header':      r['header'],
                    'seq':         r['seq'],
                    'length':      len(r['seq']),
                    'charge':      r['charge'],
                    'n_cys':       r['n_cys'],
                    'n_met':       r['seq'].count('M'),
                    'attempt':     r['attempt'],
                    'n_fixed':     r['n_fixed'],
                })

    log     = pd.DataFrame(all_results)
    log_path = os.path.join(args.out_dir, 'mpnn_charge_filtered_log.tsv')
    log.to_csv(log_path, sep='\t', index=False)

    print(f"\n{'='*60}")
    print(f"  Backbones processed:          {len(tasks)}")
    print(f"  Total sequences collected:    {len(all_results)}")
    if len(all_results):
        n_full = (log.groupby('backbone').size() == args.n_seqs).sum()
        print(f"  Backbones with full {args.n_seqs} seqs:   {n_full}/{len(tasks)}")
        print(f"  Net charge — mean={log['charge'].mean():.1f}  "
              f"min={log['charge'].min():.1f}  max={log['charge'].max():.1f}")
        print(f"  Sequences needing >1 attempt: "
              f"{(log['attempt']>1).sum()} ({100*(log['attempt']>1).mean():.0f}%)")
        if log['n_cys'].sum() > 0:
            print(f"  WARNING: {(log['n_cys']>0).sum()} sequences still contain Cys — "
                  f"increase max_attempts or check omit_AAs setting")
    print(f"  Combined FASTA: {combined_path}")
    print(f"  Log TSV:        {log_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()