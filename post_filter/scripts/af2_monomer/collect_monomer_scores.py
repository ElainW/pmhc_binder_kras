"""
collect_monomer_scores.py

Merges per-chunk monomer_scores.sc files (from predict.py -force_monomer)
into a single ranked TSV and prints a summary.

The .sc format from dl_binder_design predict.py -force_monomer contains:
  plddt_binder  — mean pLDDT of the monomer (whole structure in -force_monomer mode)
  description   — design name

Usage:
    python collect_monomer_scores.py \
        --monomer_out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer/ \
        --out_dir         /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/
"""

import os
import glob
import argparse
import pandas as pd

PLDDT_CUTOFF = 80.0


def parse_sc_file(sc_path: str) -> pd.DataFrame:
    """
    Parse a Rosetta-style .sc score file.
    Header line: 'SCORE: total_score ...'
    Data lines:  'SCORE: <values>'
    """
    rows    = []
    headers = None
    with open(sc_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('SCORE:'):
                continue
            parts = line.split()
            if parts[1] == 'total_score':
                headers = parts[1:]
            else:
                if headers is None:
                    continue
                values = parts[1:]
                if len(values) == len(headers):
                    rows.append(dict(zip(headers, values)))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col != 'description':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monomer_out_dir', required=True,
                        help='Dir containing chunk_N/ subdirs from run_af2_monomer_array.sh')
    parser.add_argument('--out_dir',         required=True,
                        help='Where to write merged TSVs')
    parser.add_argument('--plddt_cutoff',    type=float, default=PLDDT_CUTOFF,
                        help=f'Monomer pLDDT pass threshold (default: {PLDDT_CUTOFF})')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pattern  = os.path.join(args.monomer_out_dir, 'chunk_*', 'monomer_scores.sc')
    sc_files = sorted(glob.glob(pattern))

    if not sc_files:
        print(f"ERROR: no monomer_scores.sc files found under {args.monomer_out_dir}")
        return

    print(f"Found {len(sc_files)} chunk score files")

    dfs = []
    for sc_path in sc_files:
        df_chunk = parse_sc_file(sc_path)
        if df_chunk.empty:
            print(f"  WARNING: no scores parsed from {sc_path}")
        else:
            dfs.append(df_chunk)

    if not dfs:
        print("ERROR: no valid score data found")
        return

    df = pd.concat(dfs, ignore_index=True)

    # plddt_binder = whole-structure pLDDT in -force_monomer mode
    plddt_col = 'plddt_binder' if 'plddt_binder' in df.columns else 'plddt'
    df = df.rename(columns={plddt_col: 'monomer_plddt'})
    df = df.sort_values('monomer_plddt', ascending=False).reset_index(drop=True)

    author_mask  = df['description'].str.startswith('author_')
    your_designs = df[~author_mask].copy()
    author_seqs  = df[author_mask].copy()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── All sequences ({len(df)} total) ──")
    print(f"  mean monomer pLDDT: {df['monomer_plddt'].mean():.2f}")
    print(f"  min:  {df['monomer_plddt'].min():.2f}")
    print(f"  max:  {df['monomer_plddt'].max():.2f}")

    print(f"\n── Your designs ({len(your_designs)}) ──")
    n_pass = (your_designs['monomer_plddt'] >= args.plddt_cutoff).sum()
    print(f"  Pass monomer pLDDT >= {args.plddt_cutoff}: {n_pass}/{len(your_designs)}")
    print(f"  mean: {your_designs['monomer_plddt'].mean():.2f}")
    print(your_designs[['description', 'monomer_plddt']].to_string(index=False))

    print(f"\n── Author controls ({len(author_seqs)}) ──")
    if len(author_seqs):
        print(f"  mean: {author_seqs['monomer_plddt'].mean():.2f}")
        print(author_seqs[['description', 'monomer_plddt']].to_string(index=False))

    # ── Save ─────────────────────────────────────────────────────────────────
    all_path  = os.path.join(args.out_dir, 'monomer_scores_all.tsv')
    df.to_csv(all_path, sep='\t', index=False)
    print(f"\nSaved all     -> {all_path}")

    your_path = os.path.join(args.out_dir, 'monomer_scores_your_designs.tsv')
    your_designs.to_csv(your_path, sep='\t', index=False)
    print(f"Saved yours   -> {your_path}")

    passing   = your_designs[your_designs['monomer_plddt'] >= args.plddt_cutoff]
    pass_path = os.path.join(args.out_dir, 'monomer_scores_passing.tsv')
    passing.to_csv(pass_path, sep='\t', index=False)
    print(f"Saved passing -> {pass_path}  ({len(passing)} designs)")


if __name__ == '__main__':
    main()