"""
fastrelax_designs.py

Runs Rosetta FastRelax on pMHC fold complex PDBs for the passing designs.

Interface scores are extracted from pose.scores dict after
InterfaceAnalyzerMover.apply() — IAM deposits all scores into the pose's
DataCache, which is then accessible as a flat Python dict via pose.scores.
This is build-agnostic and captures the full score suite without relying
on specific getter method names.

Protocol:
 - ref2015 score function
 - Harmonic Cα coordinate constraints (weight=1.0) to limit backbone drift
 - FastRelax (standard_repeats=3 by default)
 - InterfaceAnalyzerMover on 'A_B' for dG, dSASA, shape complementarity

Scores captured (side1/side2 normalized/score excluded):
  complex_normalized, dG_cross, dG_cross/dSASAx100,
  dG_separated, dG_separated/dSASAx100,
  dSASA_hphobic, dSASA_int, dSASA_polar,
  delta_unsatHbonds, hbond_E_fraction, hbonds_int,
  nres_all, nres_int, packstat, per_residue_energy_int, sc_value

Usage:
    python fastrelax_designs.py \
        --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
        --split_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/ \
        --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/fastrelax/ \
        --n_repeats 3
"""

import os
import argparse
import numpy as np
import pandas as pd
import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory

DELTA_PAE_CUTOFF = -0.5

# Scores to skip from pose.scores (redundant or unwanted)
SKIP_COLS = {
    'side1_normalized', 'side1_score',
    'side2_normalized', 'side2_score',
}


# ── Rosetta helpers ───────────────────────────────────────────────────────────

def run_fastrelax(pose, scorefxn, n_repeats: int = 3):
    """
    Run FastRelax with backbone coordinate constraints.
    fr.constrain_relax_to_start_coords(True) adds harmonic Cα restraints
    internally — no VRT anchor needed, stable across PyRosetta versions.
    """
    fr = FastRelax(scorefxn_in=scorefxn, standard_repeats=n_repeats)
    fr.constrain_relax_to_start_coords(True)
    fr.apply(pose)
    return pose


def compute_interface_scores(pose, interface: str = 'A_B') -> dict:
    """
    Run InterfaceAnalyzerMover and extract scores from pose.scores dict.

    IAM.apply() deposits all interface metrics into the pose DataCache,
    accessible via pose.scores — a flat {name: value} Python dict.
    This is more reliable than individual getter methods whose availability
    varies across PyRosetta build versions.
    """
    iam = InterfaceAnalyzerMover(interface)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_input(False)
    iam.set_pack_separated(False)
    iam.apply(pose)

    scores = {}
    for key, val in pose.scores.items():
        if key in SKIP_COLS:
            continue
        try:
            scores[key] = float(val)
        except (TypeError, ValueError):
            scores[key] = val

    return scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_scores', required=True)
    parser.add_argument('--split_dir',     required=True,
                        help='Directory containing {design}_complex.pdb from '
                             'split_pmhc_fold_pdbs.py')
    parser.add_argument('--out_dir',       required=True)
    parser.add_argument('--n_repeats',     type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    relaxed_pdb_dir = os.path.join(args.out_dir, 'relaxed_pdbs')
    os.makedirs(relaxed_pdb_dir, exist_ok=True)

    pyrosetta.init(
        '-mute all '
        '-ex1 -ex2aro '
        '-use_input_sc '
        '-ignore_unrecognized_res true '
        '-ignore_zero_occupancy false '
        '-load_PDB_components false',
        silent=True,
    )

    scorefxn = ScoreFunctionFactory.create_score_function('ref2015')
    scorefxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint, 1.0
    )

    df      = pd.read_csv(args.merged_scores, sep='\t')
    passing = df[df['delta_pae_int_peptide'] < DELTA_PAE_CUTOFF].copy()
    designs = passing['design'].tolist()
    print(f"Running FastRelax on {len(designs)} designs ({args.n_repeats} repeats each)...")

    records = []

    for idx, design in enumerate(designs):
        complex_path = os.path.join(args.split_dir, f'{design}_complex.pdb')

        if not os.path.exists(complex_path):
            print(f"  ERROR: complex PDB not found: {complex_path}")
            records.append({'design': design})
            continue

        print(f"  [{idx+1}/{len(designs)}] {design}")

        try:
            pose = pose_from_pdb(complex_path)

            # Score before relaxation (constraint term active)
            score_before = scorefxn(pose)

            # Relax
            pose = run_fastrelax(pose, scorefxn, n_repeats=args.n_repeats)

            # Score after relaxation without constraint term
            scorefxn_no_cst = ScoreFunctionFactory.create_score_function('ref2015')
            score_after     = scorefxn_no_cst(pose)

            # Save relaxed PDB
            relaxed_pdb = os.path.join(relaxed_pdb_dir, f'{design}_relaxed.pdb')
            pose.dump_pdb(relaxed_pdb)

            # Interface analysis — scores deposited into pose.scores by IAM
            iface = compute_interface_scores(pose, interface='A_B')

            if not iface:
                print(f"    WARNING: no IAM scores found in pose.scores for {design}")
            else:
                print(f"    IAM scores captured: {sorted(iface.keys())}")

            record = {
                'design':               design,
                'rosetta_score_before': round(score_before, 3),
                'rosetta_score_after':  round(score_after,  3),
                'rosetta_score_delta':  round(score_after - score_before, 3),
                'complex_pdb':          complex_path,
                'relaxed_pdb':          relaxed_pdb,
            }
            record.update(iface)
            records.append(record)

        except Exception as e:
            print(f"    ERROR on {design}: {e}")
            records.append({'design': design})

    out_df   = pd.DataFrame(records)
    out_path = os.path.join(args.out_dir, 'fastrelax_scores.tsv')
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved FastRelax scores → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    key_cols = [
        'dG_separated', 'dG_separated/dSASAx100', 'dSASA_int',
        'delta_unsatHbonds', 'hbonds_int', 'packstat',
        'sc_value', 'per_residue_energy_int',
    ]
    anchor = 'dG_separated'
    valid  = out_df.dropna(subset=[anchor]) if anchor in out_df.columns else out_df
    print(f"\n── FastRelax summary ({len(valid)} designs) ──")
    for col in key_cols:
        if col in valid.columns and valid[col].notna().any():
            print(f"  {col}: mean={valid[col].mean():.3f}, "
                  f"min={valid[col].min():.3f}, "
                  f"max={valid[col].max():.3f}")


if __name__ == '__main__':
    main()