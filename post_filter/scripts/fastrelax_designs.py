"""
fastrelax_designs.py

Runs Rosetta FastRelax on pMHC fold complex PDBs for the passing designs.
Computes interface energy (dG) and per-design Rosetta scores.

Reads pre-split PDBs from split_pmhc_fold_pdbs.py:
  {split_dir}/{design}_complex.pdb  — full complex, chain A = binder, chain B = MHC+peptide

Protocol:
  - ref2015 score function
  - Harmonic Cα coordinate constraints (weight=1.0) to limit backbone drift
  - FastRelax (standard_repeats=3 by default)
  - InterfaceAnalyzerMover on 'A_B' for dG, dSASA, shape complementarity

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
from pyrosetta.rosetta.core.scoring.constraints import (
    CoordinateConstraint,
    ConstraintSet,
)
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc

DELTA_PAE_CUTOFF = -0.5


# ── Rosetta helpers ───────────────────────────────────────────────────────────

def add_coordinate_constraints(pose, coord_cst_weight: float = 1.0):
    """
    Add harmonic coordinate constraints on all Cα atoms to prevent
    large backbone deviations during FastRelax.
    Weight=1.0 matches BindCraft default.
    """
    cst_set  = ConstraintSet()
    ref_pose = pose.clone()

    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if not res.has('CA'):
            continue
        ca_id  = AtomID(res.atom_index('CA'), i)
        vrt_id = AtomID(
            pose.residue(pose.fold_tree().root()).atom_index('ORIG'),
            pose.fold_tree().root()
        )
        ca_xyz = ref_pose.residue(i).xyz('CA')
        target = xyzVector_double_t(ca_xyz.x, ca_xyz.y, ca_xyz.z)
        func   = HarmonicFunc(0.0, coord_cst_weight)
        cst    = CoordinateConstraint(ca_id, vrt_id, target, func)
        cst_set.add_constraint(cst)

    pose.add_constraints(cst_set)


def run_fastrelax(pose, scorefxn, n_repeats: int = 3):
    """Run FastRelax with coordinate constraints active."""
    fr = FastRelax(scorefxn_in=scorefxn, standard_repeats=n_repeats)
    fr.constrain_relax_to_start_coords(True)
    fr.apply(pose)
    return pose


def compute_interface_scores(pose, interface: str = 'A_B') -> dict:
    """
    Run InterfaceAnalyzerMover to get dG, dSASA, shape complementarity.
    'A_B': chain A (binder) vs chain B (MHC+peptide).
    """
    iam = InterfaceAnalyzerMover(interface)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_input(False)
    iam.set_pack_separated(False)
    iam.apply(pose)

    n_iface_res = 0
    try:
        n_iface_res = iam.get_interface_residues().size()
    except Exception:
        pass

    return {
        'dG':                   iam.get_interface_dG(),
        'dSASA':                iam.get_interface_delta_sasa(),
        'dG_per_dSASA':         iam.get_interface_dG() / max(iam.get_interface_delta_sasa(), 1e-6),
        'n_interface_residues': n_iface_res,
        'packstat':             iam.get_interface_packstat(),
        'shape_complementarity': iam.get_shape_complementarity(),
        'n_hbonds':             iam.get_interface_Hbond_num(),
    }


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
    os.makedirs(os.path.join(args.out_dir, 'relaxed_pdbs'), exist_ok=True)

    # Initialize PyRosetta
    pyrosetta.init(
        '-mute all '
        '-ex1 -ex2aro '
        '-use_input_sc '
        '-ignore_unrecognized_res true '
        '-ignore_zero_occupancy false '
        '-load_PDB_components false',
        silent=True
    )

    scorefxn = ScoreFunctionFactory.create_score_function('ref2015')
    scorefxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint, 1.0
    )

    df = pd.read_csv(args.merged_scores, sep='\t')
    passing = df[df['delta_pae_int_peptide'] < DELTA_PAE_CUTOFF].copy()
    designs = passing['design'].tolist()
    print(f"Running FastRelax on {len(designs)} designs ({args.n_repeats} repeats each)...")

    records = []
    for idx, design in enumerate(designs):
        complex_path = os.path.join(args.split_dir, f'{design}_complex.pdb')

        if not os.path.exists(complex_path):
            print(f"  ERROR: complex PDB not found for {design}: {complex_path}")
            records.append({'design': design, 'dG': np.nan})
            continue

        print(f"  [{idx+1}/{len(designs)}] {design}")

        try:
            # Load complex with proper chain A/B labels
            pose = pose_from_pdb(complex_path)

            # Score before relaxation
            score_before = scorefxn(pose)

            # Add coordinate constraints and relax
            add_coordinate_constraints(pose, coord_cst_weight=1.0)
            pose = run_fastrelax(pose, scorefxn, n_repeats=args.n_repeats)

            # Score after relaxation without constraint term
            scorefxn_no_cst = ScoreFunctionFactory.create_score_function('ref2015')
            score_after = scorefxn_no_cst(pose)

            # Interface analysis: chain A (binder) vs chain B (MHC+peptide)
            interface_scores = compute_interface_scores(pose, interface='A_B')

            # Save relaxed PDB
            relaxed_pdb = os.path.join(
                args.out_dir, 'relaxed_pdbs', f'{design}_relaxed.pdb'
            )
            pose.dump_pdb(relaxed_pdb)

            records.append({
                'design':                design,
                'rosetta_score_before':  round(score_before, 3),
                'rosetta_score_after':   round(score_after, 3),
                'rosetta_score_delta':   round(score_after - score_before, 3),
                'dG':                    round(interface_scores['dG'], 3),
                'dSASA':                 round(interface_scores['dSASA'], 3),
                'dG_per_dSASA':          round(interface_scores['dG_per_dSASA'], 6),
                'packstat':              round(interface_scores['packstat'], 3),
                'shape_complementarity': round(interface_scores['shape_complementarity'], 3),
                'n_hbonds':              interface_scores['n_hbonds'],
                'n_interface_residues':  interface_scores['n_interface_residues'],
                'complex_pdb':           complex_path,
                'relaxed_pdb':           relaxed_pdb,
            })

        except Exception as e:
            print(f"    ERROR on {design}: {e}")
            records.append({'design': design, 'dG': np.nan})

    out_df   = pd.DataFrame(records)
    out_path = os.path.join(args.out_dir, 'fastrelax_scores.tsv')
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved FastRelax scores to {out_path}")

    valid = out_df.dropna(subset=['dG'])
    print(f"\n── FastRelax summary ({len(valid)} designs) ──")
    for col in ['dG', 'dSASA', 'packstat', 'shape_complementarity', 'n_hbonds']:
        if col in valid.columns:
            print(f"  {col}: mean={valid[col].mean():.3f}, "
                  f"min={valid[col].min():.3f}, "
                  f"max={valid[col].max():.3f}")


if __name__ == '__main__':
    main()