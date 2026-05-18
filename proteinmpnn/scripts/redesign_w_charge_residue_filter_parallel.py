#!/usr/bin/env python3
"""
redesign_w_charge_residue_filter_parallel.py

SLURM array wrapper around redesign_w_charge_residue_filter.py.

All core logic (net_charge, sequence_passes_filters, run_mpnn, collect_sequences,
run_mode_a, run_mode_b, etc.) is imported directly from
redesign_w_charge_residue_filter.py — this file adds only:

  - --design_list / --write_design_list : explicit path list for reproducible
    array indexing (one absolute PDB/CIF path per line)
  - --chunk_size : number of PDBs per array task (default 50, ~4h/job)
  - --design_idx : 0-based chunk index, set to $SLURM_ARRAY_TASK_ID
  - per_design/ output layout : each task writes {stem}.fa + {stem}.tsv so
    jobs are fully independent with no write conflicts
  - --merge : combine per_design/ outputs into all_sequences.fa +
    mpnn_redesign_log.tsv after all array tasks finish

Both redesign_w_charge_residue_filter.py and this script must be in the same
directory (or on sys.path).

Usage -- write design list and get chunk count:

    N=$(python redesign_w_charge_residue_filter_parallel.py \
        --pdb_dir  /path/to/pdbs \
        --out_dir  /path/to/outputs \
        --mpnn_path ... --mpnn_script ... \
        --chunk_size 50 --write_design_list | tail -1)
    # prints: chunk_size=50  n_chunks=7  array=0-6
    #         7

Usage -- SLURM array (in batch script):

    python redesign_w_charge_residue_filter_parallel.py \
        --pdb_dir     /path/to/pdbs \
        --out_dir     /path/to/outputs \
        --mpnn_path   ... --mpnn_script ... \
        --design_list /path/to/outputs/design_list.txt \
        --design_idx  $SLURM_ARRAY_TASK_ID \
        --chunk_size  50

Usage -- merge after all tasks complete:

    python redesign_w_charge_residue_filter_parallel.py \
        --pdb_dir  /path/to/pdbs \
        --out_dir  /path/to/outputs \
        --mpnn_path ... --mpnn_script ... \
        --merge
"""

from __future__ import annotations

import math
import os
import glob
import argparse
import pandas as pd
from pathlib import Path

# -- Import all shared logic from the base script -----------------------------
# Both files must be in the same directory or on sys.path.
from redesign_w_charge_residue_filter import (
    # sequence scoring / filtering
    net_charge,
    sequence_passes_filters,
    # parsing helpers
    parse_fasta,
    parse_resnums_from_audit,
    parse_redesign_reasons,
    summarize_reasons,
    # structure helpers
    cif_to_pdb,
    strip_to_abc,
    design_stem,
    KEEP_CHAINS,
    # MPNN pipeline
    run_subprocess,
    run_mpnn,
    TEMP_SCHEDULE,
    _run_attempts,
    collect_sequences,
    # task-list builders
    run_mode_a,
    run_mode_b,
)


# -- Per-design output helpers ------------------------------------------------

def per_design_paths(out_dir: str, stem: str):
    """Return (fasta_path, tsv_path) for per-design output files."""
    d = os.path.join(out_dir, 'per_design')
    os.makedirs(d, exist_ok=True)
    return (os.path.join(d, f'{stem}.fa'),
            os.path.join(d, f'{stem}.tsv'))


def write_per_design(out_dir, stem, collected, reasons_str,
                     surface_hyd_resnums, pdb_path):
    """Write per-design FASTA and TSV for one backbone.
    If collected is empty, skips writing so merge does not see empty files."""
    if not collected:
        print(f"    [WARN] {stem}: no sequences collected, skipping output files")
        return []
    fa_path, tsv_path = per_design_paths(out_dir, stem)
    rows = []
    with open(fa_path, 'w') as fa:
        for r in collected:
            fa.write(f">{r['header']}\n{r['seq']}\n")
            rows.append({
                'backbone':                    stem,
                'pdb_path':                    pdb_path,
                'header':                      r['header'],
                'seq':                         r['seq'],
                'length':                      len(r['seq']),
                'net_charge':                  r['charge'],
                'n_cys':                       r['n_cys'],
                'n_met':                       r['n_met'],
                'attempt':                     r['attempt'],
                'temperature':                 r['temperature'],
                'n_fixed':                     r['n_fixed'],
                'redesign_reasons':            reasons_str,
                'surface_hydrophobic_resnums': str(surface_hyd_resnums),
                'reference_seq':               r['reference_seq'],
            })
    pd.DataFrame(rows).to_csv(tsv_path, sep='\t', index=False)
    return rows


def merge_outputs(out_dir: str, n_seqs: int):
    """
    Merge all per_design/*.tsv and per_design/*.fa into
    out_dir/all_sequences.fa and out_dir/mpnn_redesign_log.tsv.
    Called once after all array jobs complete.
    """
    per_dir   = os.path.join(out_dir, 'per_design')
    tsv_files = sorted(glob.glob(os.path.join(per_dir, '*.tsv')))
    fa_files  = sorted(glob.glob(os.path.join(per_dir, '*.fa')))

    if not tsv_files:
        print(f"[WARN] No per-design TSV files found in {per_dir}")
        return

    dfs, empty = [], []
    for f in tsv_files:
        try:
            df = pd.read_csv(f, sep='\t')
            if df.empty:
                empty.append(f)
            else:
                dfs.append(df)
        except pd.errors.EmptyDataError:
            empty.append(f)

    if empty:
        print(f"[WARN] {len(empty)} empty TSV(s) skipped (0 sequences passed filters):")
        for f in empty:
            print(f"  {os.path.basename(f)}")

    if not dfs:
        print("[WARN] All TSV files were empty -- nothing to merge.")
        return

    log = pd.concat(dfs, ignore_index=True)
    log_path = os.path.join(out_dir, 'mpnn_redesign_log.tsv')
    log.to_csv(log_path, sep='\t', index=False)

    combined_path = os.path.join(out_dir, 'all_sequences.fa')
    with open(combined_path, 'w') as out:
        for fa in fa_files:
            out.write(open(fa).read())

    n_backbones = log['backbone'].nunique()
    n_full      = (log.groupby('backbone').size() == n_seqs).sum()
    print(f"\n{'='*60}")
    print(f"  Merged {len(tsv_files)} per-design files")
    print(f"  Backbones:                {n_backbones}")
    print(f"  Total sequences:          {len(log)}")
    print(f"  Backbones with full {n_seqs}: {n_full}/{n_backbones}")
    print(f"  Net charge -- mean={log['net_charge'].mean():.1f}  "
          f"min={log['net_charge'].min():.1f}  "
          f"max={log['net_charge'].max():.1f}")
    print(f"  Combined FASTA: {combined_path}")
    print(f"  Log TSV:        {log_path}")
    print(f"{'='*60}")


# -- Argument parsing ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Inherited mode/input args (passed through to run_mode_a / run_mode_b)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pdb_dir',   help='Mode A: directory of backbone PDB files')
    mode.add_argument('--audit_tsv', help='Mode B: redesign_list.tsv')

    p.add_argument('--af3_out_dir',
                   help='Mode B: AF3 output root with {design}_YYYYMMDD_HHMMSS/ subdirs')

    redesign = p.add_mutually_exclusive_group()
    redesign.add_argument('--redesign_failing_any',    action='store_true', default=True)
    redesign.add_argument('--redesign_failing_charge', action='store_true')
    redesign.add_argument('--redesign_failing_cys',    action='store_true')
    redesign.add_argument('--redesign_all',            action='store_true')

    p.add_argument('--out_dir',       required=True)
    p.add_argument('--mpnn_path',     required=True)
    p.add_argument('--mpnn_script',   required=True)
    p.add_argument('--n_seqs',        type=int,   default=8)
    p.add_argument('--omit_AAs',      default='C')
    p.add_argument('--design_chains', default='A')
    p.add_argument('--max_charge',    type=float, default=-3.0)
    p.add_argument('--max_attempts',  type=int,   default=10)
    p.add_argument('--base_seed',     type=int,   default=42)
    p.add_argument('--pattern',       default='*.pdb')

    # Parallel-specific args
    p.add_argument('--design_list', default=None,
                   help='Path to design_list.txt (one absolute PDB/CIF path per line). '
                        'Written by --write_design_list; read by array jobs via --design_idx.')
    p.add_argument('--chunk_size', type=int, default=50,
                   help='PDBs per array task (default: 50, ~4h/job for 349 PDBs -> 7 jobs).')
    p.add_argument('--design_idx', type=int, default=None,
                   help='0-based chunk index into --design_list. '
                        'Set to $SLURM_ARRAY_TASK_ID in the batch script.')
    p.add_argument('--write_design_list', action='store_true',
                   help='Write design_list.txt and print n_chunks (ceil(n/chunk_size)); '
                        'use that number as the SLURM --array upper bound.')
    p.add_argument('--merge', action='store_true',
                   help='Merge per_design/ outputs into all_sequences.fa + '
                        'mpnn_redesign_log.tsv, then exit.')

    return p.parse_args()


# -- Main ---------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # -- Merge mode -----------------------------------------------------------
    if args.merge:
        merge_outputs(args.out_dir, args.n_seqs)
        return

    # -- Array task mode: read chunk from design_list.txt --------------------
    if args.design_idx is not None:
        if not args.design_list:
            raise ValueError('--design_list is required when using --design_idx')
        with open(args.design_list) as f:
            all_paths = [l.strip() for l in f if l.strip()]
        n_chunks = math.ceil(len(all_paths) / args.chunk_size)
        if args.design_idx >= n_chunks:
            raise ValueError(
                f"--design_idx {args.design_idx} out of range "
                f"(design_list has {len(all_paths)} PDBs, "
                f"{n_chunks} chunks of {args.chunk_size}; "
                f"valid indices 0-{n_chunks-1})"
            )
        start = args.design_idx * args.chunk_size
        chunk = all_paths[start : start + args.chunk_size]
        tasks = [(p, [], None, 'none', []) for p in chunk]

        print(f"\nArray task {args.design_idx}/{n_chunks-1}: "
              f"{len(chunk)} PDBs (list indices {start}-{start+len(chunk)-1})")

    else:
        # -- Build full task list --------------------------------------------
        if args.pdb_dir:
            tasks = run_mode_a(args)
        else:
            if not args.af3_out_dir:
                raise ValueError('--af3_out_dir required with --audit_tsv')
            tasks = run_mode_b(args)

        # -- write_design_list: write file, print n_chunks, exit -------------
        if args.write_design_list:
            list_path = os.path.join(args.out_dir, 'design_list.txt')
            with open(list_path, 'w') as f:
                for (pdb_path, *_) in tasks:
                    f.write(pdb_path + '\n')
            n_chunks = math.ceil(len(tasks) / args.chunk_size)
            print(f"Wrote {len(tasks)} paths -> {list_path}")
            print(f"chunk_size={args.chunk_size}  n_chunks={n_chunks}  "
                  f"array=0-{n_chunks-1}")
            print(n_chunks)  # last line: captured by shell for sbatch --array
            return

    print(f"ProteinMPNN settings:")
    print(f"  n_seqs={args.n_seqs}  omit_AAs={args.omit_AAs}  "
          f"design_chains={args.design_chains}")
    print(f"  max_charge<={args.max_charge}  max_attempts={args.max_attempts} per tier")
    print(f"  temperature schedule: {TEMP_SCHEDULE}\n")

    for i, (pdb_path, fixed_resnums, ref_seq, reasons_str, surface_hyd_resnums) in enumerate(tasks):
        stem     = design_stem(pdb_path)
        hyd_note = f"  surface_hyd={len(surface_hyd_resnums)}" if surface_hyd_resnums else ""
        print(f"[{i+1}/{len(tasks)}] {stem}  "
              f"(fixed={len(fixed_resnums)} res  reasons=[{reasons_str}]{hyd_note})")

        # Skip if already done — safe to resubmit failed chunks
        fa_path, _ = per_design_paths(args.out_dir, stem)
        if os.path.exists(fa_path):
            print(f"  [SKIP] already exists: {fa_path}")
            continue

        collected = collect_sequences(
            python_bin    = args.mpnn_path,
            mpnn_script   = args.mpnn_script,
            pdb_path      = pdb_path,
            out_dir       = args.out_dir,
            n_seqs        = args.n_seqs,
            omit_AAs      = args.omit_AAs,
            design_chains = args.design_chains,
            max_charge    = args.max_charge,
            max_attempts  = args.max_attempts,
            base_seed     = args.base_seed,
            fixed_resnums = fixed_resnums,
            reference_seq = ref_seq,
        )

        for r in collected:
            print(f"  attempt {r['attempt']}: charge={r['charge']:+.1f}  "
                  f"T={r['temperature']}  Cys={r['n_cys']}  "
                  f"{r['seq'][:40]}...")

        write_per_design(args.out_dir, stem, collected,
                         reasons_str, surface_hyd_resnums, pdb_path)


if __name__ == '__main__':
    main()