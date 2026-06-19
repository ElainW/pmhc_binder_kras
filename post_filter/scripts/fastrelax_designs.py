"""
fastrelax_designs.py

Runs Rosetta FastRelax on pMHC fold complex PDBs for the passing designs.

Interface scores are extracted from pose.scores dict after
InterfaceAnalyzerMover.apply() — IAM deposits all scores into the pose's
DataCache, which is then accessible as a flat Python dict via pose.scores.
This is build-agnostic and captures the full score suite without relying
on specific getter method names.

Scores captured (side1/side2 normalized/score excluded):
  complex_normalized, dG_cross, dG_cross/dSASAx100,
  dG_separated, dG_separated/dSASAx100,
  dSASA_hphobic, dSASA_int, dSASA_polar,
  delta_unsatHbonds, hbond_E_fraction, hbonds_int,
  nres_all, nres_int, packstat, per_residue_energy_int, sc_value

Additional BuriedUnsatHbonds score (BindCraft-equivalent):
  buns_delta_unsat  — delta buried unsatisfied H-bonds across the interface,
                      computed with BuriedUnsatHbonds filter using:
                        report_all_heavy_atom_unsats="true"
                        use_ddG_style="true"   (delta: complex minus separated)
                        dalphaball_sasa="1"    (DAlphaBall burial detection)
                        probe_radius="1.1"
                        burial_cutoff_apo="0.2"
                        ignore_surface_res="false"
                      Threshold in BindCraft default_filters: < 4 (models 1+2+avg)

Additional surface hydrophobicity score (BindCraft-equivalent):
  surface_hydrophobicity — fraction of solvent-exposed binder residues that are
                           hydrophobic (apolar + PHE/TRP/TYR), computed via
                           PyRosetta LayerSelector on chain A subpose.
                           Threshold in BindCraft default_filters: <= 0.35

Usage:
    python fastrelax_designs.py \
        --merged_scores ~/Desktop/pmhc_binder_kras/specificity/analysis/r2/merged_scores_ranked.tsv \
        --split_dir ~/Desktop/pmhc_binder_kras/post_filter/inputs/r2/ \
        --out_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/fastrelax/ \
        --dalphaball ~/Desktop/pmhc_binder_kras/post_filter/scripts/DAlphaBall.gcc \
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
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector, LayerSelector

DELTA_PAE_CUTOFF  = -0.5
BUNS_THRESHOLD    = 4     # BindCraft default_filters threshold for n_InterfaceUnsatHbonds

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
    iam.set_pack_input(True)
    iam.set_pack_separated(True)
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


def compute_buns(pose) -> float | None:
    """
    Compute delta buried unsatisfied H-bonds using BuriedUnsatHbonds filter,
    matching BindCraft's pyrosetta_utils.py score_interface() exactly:

      report_all_heavy_atom_unsats="true"  — count all heavy atom unsats
      use_ddG_style="true"                 — delta (complex minus apo partners)
      dalphaball_sasa="1"                  — DAlphaBall SASA (requires binary)
      probe_radius="1.1"
      burial_cutoff_apo="0.2"
      ignore_surface_res="false"           — count surface unsats too

    Returns float count, or None on failure.
    BindCraft default filter threshold: < 4 (reject if >= 4).
    """
    try:
        buns_filter = XmlObjects.static_get_filter(
            '<BuriedUnsatHbonds '
            'report_all_heavy_atom_unsats="true" '
            'scorefxn="scorefxn" '
            'ignore_surface_res="false" '
            'use_ddG_style="true" '
            'dalphaball_sasa="1" '
            'probe_radius="1.1" '
            'burial_cutoff_apo="0.2" '
            'confidence="0" />'
        )
        return float(buns_filter.report_sm(pose))
    except Exception as e:
        print(f"    WARNING: BuriedUnsatHbonds failed: {e}")
        return None


def compute_surface_hydrophobicity(pose, binder_chain: str = 'A') -> float | None:
    """
    Fraction of solvent-exposed binder residues that are hydrophobic.

    Matches BindCraft's pyrosetta_utils.py score_interface() exactly:
      - Extract binder chain subpose
      - Apply LayerSelector (surface layer only: pick_core=False, pick_boundary=False)
      - Count apolar residues + PHE/TRP/TYR as hydrophobic
      - Return hydrophobic_count / total_surface_count

    BindCraft threshold: <= 0.35 (Surface_Hydrophobicity filter).
    High surface hydrophobicity → aggregation risk.
    """
    try:
        # Split pose to binder chain subpose
        chain_sel    = ChainSelector(binder_chain)
        chain_subset = chain_sel.apply(pose)
        chain_indices = pyrosetta.rosetta.core.select.get_residues_from_subset(chain_subset)

        binder_pose = pyrosetta.rosetta.core.pose.Pose()
        pyrosetta.rosetta.core.pose.pdbslice(binder_pose, pose, chain_indices)

        # Apply LayerSelector: surface residues only
        layer_sel = LayerSelector()
        layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
        surface_res = layer_sel.apply(binder_pose)

        # Hydrophobic AAs: apolar residues + aromatic PHE/TRP/TYR
        # Matches BindCraft: res.is_apolar() or name in {PHE, TRP, TYR}
        HYDROPHOBIC = {'PHE', 'TRP', 'TYR', 'ALA', 'VAL', 'ILE', 'LEU',
                       'MET', 'PRO', 'GLY'}

        hydrophobic_count = 0
        total_count       = 0

        for i in range(1, binder_pose.total_residue() + 1):
            if not surface_res[i]:
                continue
            res = binder_pose.residue(i)
            if not res.is_protein():
                continue
            total_count += 1
            if res.is_apolar() or res.name3() in {'PHE', 'TRP', 'TYR'}:
                hydrophobic_count += 1

        if total_count == 0:
            return None
        return round(hydrophobic_count / total_count, 4)

    except Exception as e:
        print(f"    WARNING: surface_hydrophobicity failed: {e}")
        return None


def compute_interface_aa_counts(pose, binder_chain: str = 'A',
                                 aas: tuple = ('K', 'M'),
                                 dist_cutoff: float = 8.0) -> dict:
    """
    Count occurrences of specific amino acids among binder interface residues.

    Interface defined as binder Cβ/Cα within dist_cutoff of any pMHC Cβ/Cα
    (same 8 Å definition as charge_cysteine_audit.py).

    BindCraft default_filters thresholds: K <= 3, M <= 3 at interface.

    Returns dict: {'interface_n_K': int, 'interface_n_M': int, ...}
    """
    THREE_TO_ONE = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    }

    counts = {f'interface_n_{aa}': 0 for aa in aas}

    try:
        # Collect pMHC Cβ/Cα coords
        pmhc_cb = []
        for i in range(1, pose.total_residue() + 1):
            if pose.pdb_info().chain(i) == binder_chain:
                continue
            res = pose.residue(i)
            if not res.is_protein():
                continue
            if res.has('CB'):
                pmhc_cb.append(res.xyz('CB'))
            elif res.has('CA'):
                pmhc_cb.append(res.xyz('CA'))

        if not pmhc_cb:
            return counts

        import numpy as np
        pmhc_arr = np.array([[v.x, v.y, v.z] for v in pmhc_cb])

        for i in range(1, pose.total_residue() + 1):
            if pose.pdb_info().chain(i) != binder_chain:
                continue
            res = pose.residue(i)
            if not res.is_protein():
                continue

            # Cβ/Cα coordinate
            if res.has('CB'):
                xyz = res.xyz('CB')
            elif res.has('CA'):
                xyz = res.xyz('CA')
            else:
                continue

            coord = np.array([xyz.x, xyz.y, xyz.z])
            if np.linalg.norm(pmhc_arr - coord, axis=1).min() <= dist_cutoff:
                one_letter = THREE_TO_ONE.get(res.name3())
                if one_letter in aas:
                    counts[f'interface_n_{one_letter}'] += 1

    except Exception as e:
        print(f"    WARNING: interface_aa_counts failed: {e}")

    return counts
# Thresholds from default_filters.json (average + models 1+2 level)
# Applied as informational columns — all designs kept regardless of pass/fail.

BINDCRAFT_FILTERS = {
    # (column_in_tsv,          threshold, higher_is_better)
    'dG_separated':            (0,        False),   # dG < 0
    'dSASA_int':               (1,        True),    # dSASA >= 1 Å²
    'hbonds_int':              (3,        True),    # n_InterfaceHbonds >= 3
    'nres_int':                (7,        True),    # n_InterfaceResidues >= 7
    'buns_delta_unsat':        (4,        False),   # n_InterfaceUnsatHbonds < 4
    'sc_value':                (0.6,      True),    # ShapeComplementarity >= 0.6
    'surface_hydrophobicity':  (0.35,     False),   # Surface_Hydrophobicity <= 0.35
    'interface_n_K':           (3,        False),   # InterfaceAAs K <= 3
    'interface_n_M':           (3,        False),   # InterfaceAAs M <= 3
}

def apply_bindcraft_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean pass column for each BindCraft filter that has a threshold,
    plus a combined pass_all_bindcraft column. All designs are retained.
    """
    df = df.copy()
    pass_cols = []

    for col, (threshold, higher) in BINDCRAFT_FILTERS.items():
        if threshold is None:
            continue
        if col not in df.columns:
            continue
        pass_col = f'bindcraft_pass_{col}'
        if higher:
            df[pass_col] = df[col] >= threshold
        else:
            df[pass_col] = df[col] < threshold
        pass_cols.append(pass_col)

    if pass_cols:
        df['pass_all_bindcraft'] = df[pass_cols].all(axis=1)

    return df, pass_cols

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_scores', required=True)
    parser.add_argument('--split_dir',     required=True,
                        help='Directory containing {design}_complex.pdb from '
                             'split_pmhc_fold_pdbs.py')
    parser.add_argument('--out_dir',       required=True)
    parser.add_argument('--n_repeats',     type=int, default=3)
    parser.add_argument('--dalphaball',
                        default='~/Desktop/pmhc_binder_kras/post_filter/scripts/dalphaball.gcc',
                        help='Path to DAlphaBall binary for BuriedUnsatHbonds SASA calculation '
                             '(default: ~/Desktop/pmhc_binder_kras/post_filter/scripts/dalphaball.gcc)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    relaxed_pdb_dir = os.path.join(args.out_dir, 'relaxed_pdbs')
    os.makedirs(relaxed_pdb_dir, exist_ok=True)

    if not os.path.exists(args.dalphaball):
        raise FileNotFoundError(
            f"DAlphaBall binary not found: {args.dalphaball}\n"
            f"Download from https://github.com/martinpacesa/BindCraft and "
            f"place at the path above, or pass --dalphaball <path>."
        )

    pyrosetta.init(
        f'-mute all '
        f'-ex1 -ex2aro '
        f'-use_input_sc '
        f'-ignore_unrecognized_res true '
        f'-ignore_zero_occupancy false '
        f'-load_PDB_components false '
        f'-holes:dalphaball {args.dalphaball}',
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

            # BuriedUnsatHbonds — BindCraft-equivalent delta unsat H-bonds
            buns = compute_buns(pose)
            buns_pass = (buns < BUNS_THRESHOLD) if buns is not None else None
            if buns is not None:
                print(f"    buns_delta_unsat={buns:.1f}  "
                      f"({'PASS' if buns_pass else 'FAIL'} < {BUNS_THRESHOLD})")

            # Surface hydrophobicity — fraction of exposed binder residues that are hydrophobic
            surf_hydrophob = compute_surface_hydrophobicity(pose, binder_chain='A')
            if surf_hydrophob is not None:
                print(f"    surface_hydrophobicity={surf_hydrophob:.3f}  "
                      f"({'PASS' if surf_hydrophob <= 0.35 else 'FAIL'} <= 0.35)")

            # Interface AA counts for K and M — BindCraft threshold <= 3 each
            iface_aa = compute_interface_aa_counts(pose, binder_chain='A', aas=('K', 'M'))
            print(f"    interface_n_K={iface_aa['interface_n_K']}  "
                  f"interface_n_M={iface_aa['interface_n_M']}")

            record = {
                'design':               design,
                'rosetta_score_before': round(score_before, 3),
                'rosetta_score_after':  round(score_after,  3),
                'rosetta_score_delta':  round(score_after - score_before, 3),
                'complex_pdb':          complex_path,
                'relaxed_pdb':          relaxed_pdb,
                'buns_delta_unsat':     buns,
                'buns_pass':            buns_pass,
                'surface_hydrophobicity': surf_hydrophob,
            }
            record.update(iface_aa)
            record.update(iface)
            records.append(record)

        except Exception as e:
            print(f"    ERROR on {design}: {e}")
            records.append({'design': design})

    out_df, pass_cols = apply_bindcraft_filters(pd.DataFrame(records))

    out_path = os.path.join(args.out_dir, 'fastrelax_scores.tsv')
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved FastRelax scores → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    key_cols = [
        'dG_separated', 'dG_separated/dSASAx100', 'dSASA_int',
        'delta_unsatHbonds', 'hbonds_int', 'packstat',
        'sc_value', 'per_residue_energy_int', 'buns_delta_unsat',
        'surface_hydrophobicity', 'interface_n_K', 'interface_n_M',
    ]
    anchor = 'dG_separated'
    valid  = out_df.dropna(subset=[anchor]) if anchor in out_df.columns else out_df
    print(f"\n── FastRelax summary ({len(valid)} designs) ──")
    for col in key_cols:
        if col in valid.columns and valid[col].notna().any():
            print(f"  {col}: mean={valid[col].mean():.3f}, "
                  f"min={valid[col].min():.3f}, "
                  f"max={valid[col].max():.3f}")

    print(f"\n── BindCraft filters (all designs retained) ──")
    for pcol in pass_cols:
        if pcol in out_df.columns:
            n = out_df[pcol].sum()
            print(f"  {pcol}: {int(n)}/{len(valid)} pass")
    if 'pass_all_bindcraft' in out_df.columns:
        n_all = out_df['pass_all_bindcraft'].sum()
        print(f"  pass_all_bindcraft: {int(n_all)}/{len(valid)}")

    if 'buns_pass' in out_df.columns and out_df['buns_pass'].notna().any():
        n_pass = out_df['buns_pass'].sum()
        print(f"\n── BuriedUnsatHbonds filter (< {BUNS_THRESHOLD}, BindCraft default) ──")
        print(f"  Pass: {int(n_pass)}/{len(valid)}")
        print(f"  Fail: {int(len(valid) - n_pass)}/{len(valid)}")


if __name__ == '__main__':
    main()