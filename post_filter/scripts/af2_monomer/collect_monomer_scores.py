"""
collect_monomer_scores.py

Collects per-residue pLDDT from dl_binder_design predict.py -force_monomer
output JSON files and produces ranked summary TSVs.

Directory structure expected:
  {monomer_out_dir}/{design}/confidence_model_1_pred_0.json

JSON format:
  {"residueNumber": [1, 2, ...], "confidenceScore": [80.2, 91.45, ...], ...}

Outputs:
  monomer_scores_all.tsv          — all designs, ranked by mean pLDDT
  monomer_scores_your_designs.tsv — your designs only (non-author_ prefix)
  monomer_scores_passing.tsv      — your designs passing --plddt_cutoff
  monomer_scores_authors.tsv      — author_ controls
  monomer_plddt_per_residue.tsv   — per-residue pLDDT for all designs
                                    (use for loop pLDDT analysis at interface)

Usage:
    python collect_monomer_scores.py \
        --monomer_out_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/af2_monomer/ \
        --out_dir         ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/af2_monomer/stats/
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd

PLDDT_CUTOFF = 80.0


def find_json_files(monomer_out_dir: str) -> dict[str, str]:
    """
    Glob for all confidence JSONs under monomer_out_dir.
    Returns {design_name: json_path}.
    Expected: {monomer_out_dir}/{design}/confidence_model_1_pred_0.json
    Falls back to any confidence_*.json in that subdir.
    """
    # Primary pattern
    primary = glob.glob(
        os.path.join(monomer_out_dir, '*', 'confidence_model_1_pred_0.json')
    )
    found = {}
    for path in primary:
        design = os.path.basename(os.path.dirname(path))
        found[design] = path

    # Fallback: any confidence_*.json not already captured
    fallback = glob.glob(
        os.path.join(monomer_out_dir, '*', 'confidence_*.json')
    )
    for path in fallback:
        design = os.path.basename(os.path.dirname(path))
        if design not in found:
            found[design] = path

    return found


def parse_confidence_json(json_path: str) -> dict | None:
    """
    Parse a dl_binder_design confidence JSON file.
    Returns dict with:
      residue_numbers — list[int]
      plddt_per_res   — list[float]  (confidenceScore)
      mean_plddt      — float
    Returns None if unparseable.
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
        scores  = data.get('confidenceScore', [])
        resnums = data.get('residueNumber',
                           list(range(1, len(scores) + 1)))
        if not scores:
            print(f"  WARNING: empty confidenceScore in {json_path}")
            return None
        return {
            'residue_numbers': resnums,
            'plddt_per_res':   [float(s) for s in scores],
            'mean_plddt':      float(np.mean(scores)),
        }
    except Exception as e:
        print(f"  WARNING: could not parse {json_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monomer_out_dir', required=True,
                        help='Root dir containing {design}/ subdirs, '
                             'each with confidence_model_1_pred_0.json')
    parser.add_argument('--out_dir',      required=True,
                        help='Where to write output TSVs')
    parser.add_argument('--plddt_cutoff', type=float, default=PLDDT_CUTOFF,
                        help=f'Mean pLDDT pass threshold (default: {PLDDT_CUTOFF})')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Discover JSON files ───────────────────────────────────────────────────
    design_to_json = find_json_files(args.monomer_out_dir)
    if not design_to_json:
        print(f"ERROR: no confidence JSON files found under {args.monomer_out_dir}")
        return
    print(f"Found {len(design_to_json)} designs with confidence JSON files")

    # ── Parse all JSONs ───────────────────────────────────────────────────────
    summary_rows = []
    per_res_rows = []
    n_failed     = 0

    for design, json_path in sorted(design_to_json.items()):
        result = parse_confidence_json(json_path)
        if result is None:
            n_failed += 1
            continue

        summary_rows.append({
            'description':   design,
            'monomer_plddt': result['mean_plddt'],
            'n_residues':    len(result['plddt_per_res']),
            'plddt_min':     float(np.min(result['plddt_per_res'])),
            'plddt_max':     float(np.max(result['plddt_per_res'])),
            'json_path':     json_path,
        })

        for resnum, plddt in zip(result['residue_numbers'],
                                  result['plddt_per_res']):
            per_res_rows.append({
                'design':  design,
                'resnum':  resnum,
                'plddt':   plddt,
            })

    if not summary_rows:
        print("ERROR: no valid JSON data found")
        return

    if n_failed:
        print(f"WARNING: {n_failed} JSON files failed to parse")

    df = pd.DataFrame(summary_rows)
    df = df.sort_values('monomer_plddt', ascending=False).reset_index(drop=True)

    # ── Split author controls vs your designs ─────────────────────────────────
    author_mask  = df['description'].str.startswith('author_')
    your_designs = df[~author_mask].copy()
    author_seqs  = df[author_mask].copy()

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n── All sequences ({len(df)} total) ──")
    print(f"  mean pLDDT: {df['monomer_plddt'].mean():.2f}")
    print(f"  min:        {df['monomer_plddt'].min():.2f}")
    print(f"  max:        {df['monomer_plddt'].max():.2f}")

    print(f"\n── Your designs ({len(your_designs)}) ──")
    n_pass = (your_designs['monomer_plddt'] >= args.plddt_cutoff).sum()
    print(f"  Pass pLDDT >= {args.plddt_cutoff}: {n_pass}/{len(your_designs)}")
    print(f"  mean: {your_designs['monomer_plddt'].mean():.2f}")
    print(your_designs[['description', 'monomer_plddt',
                         'n_residues', 'plddt_min',
                         'plddt_max']].to_string(index=False))

    print(f"\n── Author controls ({len(author_seqs)}) ──")
    if len(author_seqs):
        print(f"  mean: {author_seqs['monomer_plddt'].mean():.2f}")
        print(author_seqs[['description', 'monomer_plddt',
                            'n_residues', 'plddt_min',
                            'plddt_max']].to_string(index=False))

    # ── Save outputs ──────────────────────────────────────────────────────────
    all_path = os.path.join(args.out_dir, 'monomer_scores_all.tsv')
    df.to_csv(all_path, sep='\t', index=False)
    print(f"\nSaved all          → {all_path}")

    your_path = os.path.join(args.out_dir, 'monomer_scores_your_designs.tsv')
    your_designs.to_csv(your_path, sep='\t', index=False)
    print(f"Saved yours        → {your_path}")

    passing   = your_designs[your_designs['monomer_plddt'] >= args.plddt_cutoff]
    pass_path = os.path.join(args.out_dir, 'monomer_scores_passing.tsv')
    passing.to_csv(pass_path, sep='\t', index=False)
    print(f"Saved passing      → {pass_path}  ({len(passing)} designs)")

    if len(author_seqs):
        auth_path = os.path.join(args.out_dir, 'monomer_scores_authors.tsv')
        author_seqs.to_csv(auth_path, sep='\t', index=False)
        print(f"Saved authors      → {auth_path}")

    per_res_path = os.path.join(args.out_dir, 'monomer_plddt_per_residue.tsv')
    pd.DataFrame(per_res_rows).to_csv(per_res_path, sep='\t', index=False)
    print(f"Saved per-residue  → {per_res_path}")
    print(f"  (filter by design + resnum to check pLDDT at interface residues)")


if __name__ == '__main__':
    main()