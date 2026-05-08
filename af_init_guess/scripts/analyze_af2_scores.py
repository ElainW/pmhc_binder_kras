#!/usr/bin/env python3
"""
analyze_af2_scores.py  —  merge, filter, plot, extract, and view AF2 initial guess results

Usage:
    python analyze_af2_scores.py \
        --score_dir /path/to/sc_files/ \
        --silent_dir /path/to/silent_files/ \
        --output_csv  af2_kras_all_scores.csv \
        --passing_csv af2_kras_passing.csv \
        --pae_cutoff 10.0 \
        --plddt_cutoff 70.0 \
        --binder_rmsd_cutoff 2.0 \
        --target_rmsd_cutoff 5.0 \
        --top_n 20

    # To also view top hits in a Jupyter notebook, import and call:
    #   from analyze_af2_scores import run_pipeline, view_candidates
    #   passing = run_pipeline(args)
    #   view_candidates(passing, silent_dir="...", top_n=12)
"""

import os
import sys
import re
import glob
import shutil
import argparse
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


# ── Score file parsing ────────────────────────────────────────────────────────

def parse_score_file(score_file):
    """Parse a Rosetta-format .sc score file into a DataFrame."""
    with open(score_file) as f:
        lines = f.readlines()

    header = None
    data_lines = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == 'SCORE:' and 'description' in parts:
            header = parts[1:]
        elif parts[0] == 'SCORE:' and header is not None:
            data_lines.append(parts[1:])

    if header is None or not data_lines:
        return pd.DataFrame()

    df = pd.DataFrame(data_lines, columns=header)
    for col in df.columns:
        if col != 'description':
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
    return df


def merge_score_files(score_dir):
    """Merge all .sc files in a directory into a single DataFrame."""
    score_files = sorted(glob.glob(os.path.join(score_dir, '*.sc')))
    if not score_files:
        raise FileNotFoundError(f"No .sc files found in {score_dir}")

    print(f"Found {len(score_files)} score file(s) in {score_dir}")

    dfs = []
    for sf in score_files:
        df = parse_score_file(sf)
        if not df.empty:
            dfs.append(df)
        else:
            print(f"  Warning: empty or unparseable — {os.path.basename(sf)}")

    if not dfs:
        raise ValueError("No valid score data found across all .sc files")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset='description', keep='first')
    print(f"Total unique designs scored: {len(merged)}")
    return merged


# ── Statistics + plots ────────────────────────────────────────────────────────

def print_statistics(df):
    key_cols = ['pae_interaction', 'plddt_binder', 'binder_aligned_rmsd', 'target_aligned_rmsd']
    available = [c for c in key_cols if c in df.columns]

    print(f"\n{'='*55}")
    print(f"Score statistics (n={len(df)})")
    print(f"{'='*55}")
    print(df[available].describe().round(3).to_string())

    print(f"\n{'='*55}")
    print(f"Passage rates")
    print(f"{'='*55}")

    if 'pae_interaction' in df.columns:
        for t in [5, 7, 10, 15, 20]:
            n = (df['pae_interaction'] < t).sum()
            print(f"  pae_interaction < {t:2d}: {n:5d}/{len(df)} ({100*n/len(df):.1f}%)")

    if 'plddt_binder' in df.columns:
        for t in [70, 80, 90]:
            n = (df['plddt_binder'] > t).sum()
            print(f"  plddt_binder    > {t:2d}: {n:5d}/{len(df)} ({100*n/len(df):.1f}%)")

    if 'binder_aligned_rmsd' in df.columns:
        for t in [1.0, 1.5, 2.0]:
            n = (df['binder_aligned_rmsd'] < t).sum()
            print(f"  binder_rmsd     < {t:.1f}: {n:5d}/{len(df)} ({100*n/len(df):.1f}%)")

    if 'target_aligned_rmsd' in df.columns:
        for t in [1.0, 1.5, 2.0, 5.0]:
            n = (df['target_aligned_rmsd'] < t).sum()
            print(f"  target_rmsd     < {t:.1f}: {n:5d}/{len(df)} ({100*n/len(df):.1f}%)")


def plot_distributions(df, passing, output_dir, pae_cutoff, plddt_cutoff,
                       binder_rmsd_cutoff, target_rmsd_cutoff):
    """Plot score distributions for all designs, highlighting passing ones."""
    plot_cols = [
        ('pae_interaction',     pae_cutoff,          'max', 'lower is better', 'steelblue'),
        ('plddt_binder',        plddt_cutoff,         'min', 'higher is better', 'coral'),
        ('binder_aligned_rmsd', binder_rmsd_cutoff,   'max', 'lower is better', 'mediumseagreen'),
        ('target_aligned_rmsd', target_rmsd_cutoff,   'max', 'lower is better', 'mediumpurple'),
    ]
    available = [(col, cut, d, lab, clr)
                 for col, cut, d, lab, clr in plot_cols if col in df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, (col, cutoff, direction, label, color) in zip(axes, available):
        all_data  = df[col].dropna()
        pass_data = passing[col].dropna() if col in passing.columns else pd.Series([], dtype=float)

        ax.hist(all_data,  bins=50, color=color,   alpha=0.5, label='all',     edgecolor='none')
        ax.hist(pass_data, bins=50, color='black',  alpha=0.6, label='passing', edgecolor='none')
        ax.axvline(x=cutoff, color='red', linestyle='--', linewidth=1.5,
                   label=f'cutoff = {cutoff}')
        ax.set_xlabel(col.replace('_', ' '))
        ax.set_ylabel('Count')
        ax.set_title(f"{col.replace('_', ' ')}\n({label})")
        ax.legend(fontsize=8)

    plt.suptitle(
        f'AF2 Initial Guess — KRAS (n={len(df)} total, {len(passing)} passing)',
        y=1.02, fontsize=12
    )
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'af2_kras_score_distributions.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nDistribution plot saved: {plot_path}")
    return plot_path


# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_candidates(df, pae_cutoff, plddt_cutoff,
                      binder_rmsd_cutoff, target_rmsd_cutoff):
    print(f"\n{'='*55}")
    print(f"Applying filters")
    print(f"{'='*55}")

    mask = pd.Series([True] * len(df), index=df.index)

    checks = [
        ('pae_interaction',     'max', pae_cutoff),
        ('plddt_binder',        'min', plddt_cutoff),
        ('binder_aligned_rmsd', 'max', binder_rmsd_cutoff),
        ('target_aligned_rmsd', 'max', target_rmsd_cutoff),
    ]
    for col, direction, cutoff in checks:
        if col not in df.columns:
            print(f"  SKIP  {col} — column not found")
            continue
        before = mask.sum()
        if direction == 'max':
            mask &= df[col] <= cutoff
            print(f"  {col:<28} ≤ {cutoff:<8} {before:5d} → {mask.sum():5d}")
        else:
            mask &= df[col] >= cutoff
            print(f"  {col:<28} ≥ {cutoff:<8} {before:5d} → {mask.sum():5d}")

    passing = df[mask].copy()
    print(f"\n  Passing all filters: {len(passing)}/{len(df)} "
          f"({100*len(passing)/len(df):.1f}%)")
    return passing


# ── Backbone summary ──────────────────────────────────────────────────────────

def backbone_summary(df, passing):
    if 'description' not in passing.columns or passing.empty:
        return passing

    for frame in [df, passing]:
        frame['backbone'] = frame['description'].str.replace(
            r'_sample\d+_af2pred$', '', regex=True
        )

    n_total   = df['backbone'].nunique()
    n_passing = passing['backbone'].nunique()

    print(f"\n{'='*55}")
    print(f"Backbone summary")
    print(f"{'='*55}")
    print(f"  Total unique backbones:          {n_total}")
    print(f"  Backbones with ≥1 passing seq:   {n_passing} "
          f"({100*n_passing/n_total:.1f}%)")

    counts = passing['backbone'].value_counts()
    print(f"\n  Distribution of passing seqs per backbone:")
    for n_seqs, count in counts.value_counts().sort_index().items():
        print(f"    {n_seqs} passing seq(s): {count} backbone(s)")

    print(f"\n  Top 10 backbones by passing sequence count:")
    print(counts.head(10).to_string())

    print(f"\n{'='*55}")
    print(f"Next steps")
    print(f"{'='*55}")
    if n_passing >= 50:
        print(f"  {n_passing} backbones pass — proceed to deep MPNN mining")
        print(f"  (32–50 seq/backbone, T=0.1 + 0.2)")
    elif n_passing >= 20:
        print(f"  {n_passing} backbones pass — proceed to deep MPNN mining")
        print(f"  + consider one more RFdiffusion round")
    else:
        print(f"  Only {n_passing} backbones pass — consider revisiting")
        print(f"  backbone generation (partial diffusion)")

    return passing


# ── Rank and select top N ─────────────────────────────────────────────────────

def rank_and_select(passing, top_n):
    """
    Composite rank: normalise each metric, apply weights, sort ascending.
    plddt_binder is negated (higher = better).
    """
    if passing.empty:
        return passing

    rank_weights = {
        'pae_interaction':     2.0,
        'plddt_binder':       -1.5,
        'pae_binder':          1.0,
        'binder_aligned_rmsd': 0.5,
    }

    score = pd.Series(0.0, index=passing.index)
    for col, w in rank_weights.items():
        if col not in passing.columns:
            continue
        rng = passing[col].max() - passing[col].min()
        if rng == 0:
            continue
        normed = (passing[col] - passing[col].min()) / rng
        score += w * normed

    passing = passing.copy()
    passing['rank_score'] = score
    passing = passing.sort_values('rank_score').reset_index(drop=True)

    top = passing.head(top_n)
    print(f"\n{'='*55}")
    print(f"Top {len(top)} candidates (ranked)")
    print(f"{'='*55}")

    show_cols = [c for c in [
        'description', 'pae_interaction', 'plddt_binder',
        'pae_binder', 'binder_aligned_rmsd', 'target_aligned_rmsd', 'rank_score'
    ] if c in top.columns]

    display = top[show_cols].copy()
    if 'description' in display.columns:
        display['description'] = display['description'].str.replace(
            '_af2pred', '', regex=False
        )
    for col in display.select_dtypes(include='number').columns:
        display[col] = display[col].round(3)
    print(display.to_string(index=True))

    return passing, top


# ── PDB extraction ────────────────────────────────────────────────────────────

def extract_top_pdbs(top, silent_dir, out_dir):
    """
    Extract PDBs for top candidates from .silent files using silentextract.
    Falls back to silent_tools if silentextract is not on PATH.
    """
    if not silent_dir:
        print("\nNo --silent_dir provided — skipping PDB extraction.")
        return []

    os.makedirs(out_dir, exist_ok=True)
    tags = top['description'].tolist()
    print(len(tags))

    silent_files = sorted(glob.glob(os.path.join(silent_dir, '*.silent')))
    if not silent_files:
        print(f"\nNo .silent files found in {silent_dir} — skipping extraction.")
        return []

    # Find which silent file contains each tag
    print(f"\n{'='*55}")
    print(f"Extracting {len(tags)} PDBs from silent files")
    print(f"{'='*55}")

    # Build a tag → silent file map by scanning SCORE lines
    tag_to_silent = {}
    for sf in silent_files:
        with open(sf, errors='replace') as fh:
            for line in fh:
                if line.startswith('SCORE:'):
                    parts = line.split()
                    if parts[-1] != 'description' and parts[-1] in tags:
                        tag_to_silent[parts[-1]] = sf

    missing = [t for t in tags if t not in tag_to_silent]
    if missing:
        print(f"  WARNING: {len(missing)} tag(s) not found in any silent file: {missing[:5]}")

    # Group tags by silent file and extract
    silent_to_tags = defaultdict(list)
    for tag, sf in tag_to_silent.items():
        silent_to_tags[sf].append(tag)

    tool = shutil.which('silentextractspecific')
    if not tool:
        print("  WARNING: silentextractspecific not found on PATH.")
        print("    Make sure silent_tools is on PATH before running this script.")
        return []

    extracted = []
    for sf, sf_tags in silent_to_tags.items():
        print(f"  {os.path.basename(sf)}: extracting {len(sf_tags)} tag(s)")

        # Write tags to a temp file to avoid shell argument length limits
        tag_file = os.path.join(out_dir, '_extract_tags.txt')
        with open(tag_file, 'w') as fh:
            fh.write('\n'.join(sf_tags) + '\n')

        # silentextractspecific <silent_file> <tag1> <tag2> ...
        cmd = [tool, sf] + sf_tags
        my_env = os.environ.copy()
        my_env["PATH"] = f"{Path(sys.executable).parent}:" + my_env["PATH"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=out_dir, env=my_env)
        if result.returncode != 0:
            print(f"    WARNING: silentextractspecific error: {result.stderr}")

        os.remove(tag_file)

    # Collect extracted PDBs — silentextractspecific writes <tag>.pdb in cwd
    for tag in tags:
        full_path = os.path.join(out_dir, tag + '.pdb')
        if os.path.exists(full_path):
            extracted.append(full_path)

    print(f"  Extracted {len(extracted)}/{len(tags)} PDBs → {out_dir}")
    return extracted


# ── py3Dmol viewer ────────────────────────────────────────────────────────────

def view_candidates(pdb_paths, top_df=None, cols=4, width=300, height=300):
    """
    View extracted PDB files in a py3Dmol grid (Jupyter / Colab).

    Parameters
    ----------
    pdb_paths : list of str    paths to extracted .pdb files
    top_df    : pd.DataFrame   optional — adds pae_interaction label to each panel
    cols      : int            grid columns
    """
    try:
        import py3Dmol
    except ImportError:
        raise ImportError("pip install py3Dmol")

    if not pdb_paths:
        print("No PDB files to display.")
        return

    n = len(pdb_paths)
    rows = (n + cols - 1) // cols

    view = py3Dmol.view(
        viewergrid=(rows, cols),
        width=width * cols,
        height=height * rows,
    )

    for i, pdb_path in enumerate(pdb_paths):
        row, col = divmod(i, cols)
        with open(pdb_path) as f:
            pdb_data = f.read()
        view.addModel(pdb_data, 'pdb', viewer=(row, col))
        view.setStyle({'cartoon': {'color': 'spectrum'}}, viewer=(row, col))

        tag = os.path.basename(pdb_path).replace('.pdb', '')
        short = tag.replace('_af2pred', '').replace('kras_', '')

        label = short
        if top_df is not None and 'description' in top_df.columns:
            row_data = top_df[top_df['description'] == tag]
            if not row_data.empty and 'pae_interaction' in row_data.columns:
                val = row_data['pae_interaction'].values[0]
                label = f"{short}\npae={val:.2f}"

        view.addLabel(
            label,
            {'fontSize': 9, 'fontColor': 'white',
             'backgroundOpacity': 0.55, 'backgroundColor': 'black',
             'inFront': True},
            viewer=(row, col),
        )
        view.zoomTo(viewer=(row, col))

    print(f"Viewing {n} structures  ({rows} rows × {cols} cols)")
    return view.show()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    # 1. Merge
    df = merge_score_files(args.score_dir)
    df.to_csv(args.output_csv, index=False)
    print(f"All scores saved: {args.output_csv}")

    # 2. Statistics
    print_statistics(df)

    # 3. Filter
    passing = filter_candidates(
        df,
        pae_cutoff=args.pae_cutoff,
        plddt_cutoff=args.plddt_cutoff,
        binder_rmsd_cutoff=args.binder_rmsd_cutoff,
        target_rmsd_cutoff=args.target_rmsd_cutoff,
    )

    # 4. Backbone summary
    passing = backbone_summary(df, passing)

    # 5. Rank + select top N — returns (all_passing_ranked, top_n_df)
    passing_ranked, top = rank_and_select(passing, top_n=args.top_n)
    passing_ranked.to_csv(args.passing_csv, index=False)
    print(f"\nPassing designs saved: {args.passing_csv}")

    # 6. Plot distributions (all vs passing highlighted)
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    plot_distributions(
        df, passing_ranked, out_dir,
        pae_cutoff=args.pae_cutoff,
        plddt_cutoff=args.plddt_cutoff,
        binder_rmsd_cutoff=args.binder_rmsd_cutoff,
        target_rmsd_cutoff=args.target_rmsd_cutoff,
    )

    # 7. Extract PDBs for top N
    pdb_out = os.path.join(args.score_dir, 'top_pdbs')
    pdb_paths = extract_top_pdbs(top, args.silent_dir, pdb_out)

    return passing_ranked, top, pdb_paths


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge, filter, plot, and extract top AF2 initial guess results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python analyze_af2_scores.py \\
        --score_dir  /path/to/sc_files/ \\
        --silent_dir /path/to/silent_files/ \\
        --output_csv  af2_kras_all_scores.csv \\
        --passing_csv af2_kras_passing.csv \\
        --pae_cutoff 10.0 \\
        --plddt_cutoff 70.0 \\
        --binder_rmsd_cutoff 2.0 \\
        --target_rmsd_cutoff 5.0 \\
        --top_n 50
        """
    )
    parser.add_argument('--score_dir',  required=True,
                        help='Directory containing .sc score files')
    parser.add_argument('--silent_dir', default=None,
                        help='Directory containing .silent files for PDB extraction '
                             '(optional — skip extraction if not provided)')
    parser.add_argument('--output_csv',  required=True,
                        help='Output CSV for all scored designs')
    parser.add_argument('--passing_csv', required=True,
                        help='Output CSV for designs passing all filters')
    parser.add_argument('--pae_cutoff',          type=float, default=10.0)
    parser.add_argument('--plddt_cutoff',         type=float, default=70.0)
    parser.add_argument('--binder_rmsd_cutoff',   type=float, default=2.0)
    parser.add_argument('--target_rmsd_cutoff',   type=float, default=5.0)
    parser.add_argument('--top_n', type=int, default=50,
                        help='Number of top candidates to extract PDBs for (default 20)')
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    passing_ranked, top, pdb_paths = run_pipeline(args)

    if pdb_paths:
        print(f"\nTo view in a notebook:")
        print(f"  from analyze_af2_scores import view_candidates")
        print(f"  import glob, pandas as pd")
        print(f"  top = pd.read_csv('{args.passing_csv}').head({args.top_n})")
        print(f"  pdbs = sorted(glob.glob('{os.path.join(args.score_dir, 'top_pdbs')}/*.pdb'))")
        print(f"  view_candidates(pdbs, top_df=top, cols=4)")


if __name__ == '__main__':
    main()