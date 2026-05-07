"""
fastrelax_designs_af3.py

Runs Rosetta FastRelax on AF3 top-ranked model CIF files for a set of designs,
then computes the full InterfaceAnalyzer score suite.

AF3 output convention:
  {af3_out_dir}/{design}_{timestamp}/{design}_model.cif   <- top-ranked model

Chain convention (matching your AF3 JSON input):
  A = binder
  B = MHC alpha
  C = peptide
  D = B2M (if present -- 4-chain complex)

Interface analysis is run on A_BC (binder vs MHC+peptide) for 3-chain inputs,
or A_BCD (binder vs full pMHC+B2M) for 4-chain inputs. Chain layout is
auto-detected from the CIF.

Scores captured (side1/side2 normalized/score excluded):
  complex_normalized, dG_cross, dG_cross/dSASAx100,
  dG_separated, dG_separated/dSASAx100,
  dSASA_hphobic, dSASA_int, dSASA_polar,
  delta_unsatHbonds, hbond_E_fraction, hbonds_int,
  nres_all, nres_int, packstat, per_residue_energy_int, sc_value

Additional BuriedUnsatHbonds score (BindCraft-equivalent):
  buns_delta_unsat  -- delta buried unsatisfied H-bonds across the interface,
                       computed with BuriedUnsatHbonds filter using:
                         report_all_heavy_atom_unsats="true"
                         use_ddG_style="true"   (delta: complex minus separated)
                         dalphaball_sasa="1"    (DAlphaBall burial detection)
                         probe_radius="1.1"
                         burial_cutoff_apo="0.2"
                         ignore_surface_res="false"
                       Threshold in BindCraft default_filters: < 4 (models 1+2+avg)

Additional surface hydrophobicity score (BindCraft-equivalent):
  surface_hydrophobicity -- fraction of solvent-exposed binder residues that are
                            hydrophobic (apolar + PHE/TRP/TYR), computed via
                            PyRosetta LayerSelector on chain A subpose.
                            Threshold in BindCraft default_filters: <= 0.35

Usage:
    python fastrelax_af3.py \
        --af3_out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3/ \
        --out_dir     /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/fastrelax_af3/ \
        --dalphaball  /n/groups/marks/users/aaron/pmhc/post_filter/scripts/dalphaball.gcc \
        --n_repeats   3

    python fastrelax_af3.py \
        --af3_out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3/ \
        --out_dir     /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/fastrelax_af3/ \
        --dalphaball  /n/groups/marks/users/aaron/pmhc/post_filter/scripts/dalphaball.gcc \
        --n_repeats   3

    # Run on a specific list of designs:
    python fastrelax_af3.py \
        --af3_out_dir ... \
        --out_dir     ... \
        --designs ctnnb1-15 gp100-3 mart1-3

    # Inspect chain layout of one CIF before running:
    python fastrelax_af3.py --af3_out_dir ... --out_dir ... --inspect ctnnb1-15
"""

from __future__ import annotations

import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
import pyrosetta
from pyrosetta import Pose
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
from pyrosetta.rosetta.core.import_pose import FileType
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector, LayerSelector

SKIP_COLS = {
    'side1_normalized', 'side1_score',
    'side2_normalized', 'side2_score',
}

BUNS_THRESHOLD = 4   # BindCraft default_filters: n_InterfaceUnsatHbonds < 4


# -- CIF discovery ------------------------------------------------------------

def find_model_cif(af3_out_dir: str, design: str) -> str | None:
    """
    Find {design}_model.cif inside any timestamped subdirectory.
    Pattern: {af3_out_dir}/{design}_*/{design}_model.cif
    Returns path of the first match, or None.
    """
    pattern = os.path.join(af3_out_dir, f'{design}_*', f'{design}_model.cif')
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def find_ranking_scores(af3_out_dir: str, design: str) -> dict:
    """Read ranking_scores.csv; return top seed info or empty dict."""
    pattern = os.path.join(af3_out_dir, f'{design}_*', 'ranking_scores.csv')
    matches = sorted(glob.glob(pattern))
    if not matches:
        return {}
    try:
        df  = pd.read_csv(matches[0])
        top = df.sort_values('ranking_score', ascending=False).iloc[0]
        return {
            'top_seed':          int(top.get('seed', -1)),
            'top_sample':        int(top.get('sample', -1)),
            'af3_ranking_score': float(top.get('ranking_score', np.nan)),
        }
    except Exception:
        return {}


def get_confidences(af3_out_dir: str, design: str) -> dict:
    """Read {design}_confidences.json; return iptm/ptm/fraction_disordered."""
    pattern = os.path.join(
        af3_out_dir, f'{design}_*', f'{design}_confidences.json'
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        return {}
    try:
        with open(matches[0]) as f:
            data = json.load(f)
        return {
            'af3_iptm':                data.get('iptm',               np.nan),
            'af3_ptm':                 data.get('ptm',                np.nan),
            'af3_fraction_disordered': data.get('fraction_disordered', np.nan),
        }
    except Exception:
        return {}


# -- CIF loading --------------------------------------------------------------

def load_cif(cif_path: str) -> Pose:
    """Load an mmCIF file into a PyRosetta Pose."""
    pose = Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_file(
        pose, cif_path, False, FileType.CIF_file
    )
    return pose


def get_chain_ids(pose: Pose) -> list[str]:
    """Return ordered unique chain IDs present in the pose."""
    chains, seen = [], set()
    for i in range(1, pose.total_residue() + 1):
        ch = pose.pdb_info().chain(i)
        if ch not in seen:
            seen.add(ch)
            chains.append(ch)
    return chains


def build_interface_string(chain_ids: list[str]) -> str:
    """
    Build InterfaceAnalyzerMover interface string.
    Binder = chain A. Target = all remaining chains joined.

    3-chain (A, B, C):    -> 'A_BC'
    4-chain (A, B, C, D): -> 'A_BCD'
    """
    binder = 'A'
    target_chains = [c for c in chain_ids if c != binder]
    if not target_chains:
        raise ValueError(f"No non-binder chains found: {chain_ids}")
    return f"{binder}_{''.join(target_chains)}"


# -- Rosetta helpers ----------------------------------------------------------

def run_fastrelax(pose: Pose, scorefxn, n_repeats: int = 3) -> Pose:
    fr = FastRelax(scorefxn_in=scorefxn, standard_repeats=n_repeats)
    fr.constrain_relax_to_start_coords(True)
    fr.apply(pose)
    return pose


def compute_interface_scores(pose: Pose, interface: str) -> dict:
    """Run InterfaceAnalyzerMover and extract scores from pose.scores dict."""
    iam = InterfaceAnalyzerMover(interface)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_input(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    scores = {}
    for key, val in pose.cache.items():
        if key in SKIP_COLS:
            continue
        try:
            scores[key] = float(val)
        except (TypeError, ValueError):
            scores[key] = val
    return scores


def compute_buns(pose: Pose) -> float | None:
    """
    Compute delta buried unsatisfied H-bonds using BuriedUnsatHbonds filter.

    use_ddG_style requires <= 3 chains (Rosetta limitation). For 4-chain poses
    (binder + MHC + peptide + B2M), chain D (B2M) is stripped before the filter
    is applied so the calculation runs on the 3-chain binder/MHC/peptide subpose.
    The stripped pose is a local copy — the original is not modified.

    Returns float count, or None on failure.
    BindCraft default filter threshold: < 4 (reject if >= 4).
    """
    try:
        if pose.num_chains() >= 4:
            # Extract chains A, B, C only — drop chain D (B2M).
            # use_ddG_style requires <= 3 chains (Rosetta limitation).
            chain_ids = []
            seen = set()
            for i in range(1, pose.total_residue() + 1):
                ch = pose.pdb_info().chain(i)
                if ch not in seen:
                    seen.add(ch)
                    chain_ids.append(ch)
            keep_chains = chain_ids[:3]
            selector = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(
                ','.join(keep_chains)
            )
            subset    = selector.apply(pose)
            indices   = pyrosetta.rosetta.core.select.get_residues_from_subset(subset)
            buns_pose = Pose()
            pyrosetta.rosetta.core.pose.pdbslice(buns_pose, pose, indices)
        else:
            buns_pose = pose

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
        return float(buns_filter.report_sm(buns_pose))
    except Exception as e:
        print(f"    WARNING: BuriedUnsatHbonds failed: {e}")
        return None


def compute_surface_hydrophobicity(pose: Pose, binder_chain: str = 'A') -> float | None:
    """
    Fraction of solvent-exposed binder residues that are hydrophobic.

    Matches BindCraft's pyrosetta_utils.py score_interface() exactly:
      - Extract binder chain subpose
      - Apply LayerSelector (surface layer only: pick_core=False, pick_boundary=False)
      - Count apolar residues + PHE/TRP/TYR as hydrophobic
      - Return hydrophobic_count / total_surface_count

    BindCraft threshold: <= 0.35 (Surface_Hydrophobicity filter).
    """
    try:
        chain_sel     = ChainSelector(binder_chain)
        chain_subset  = chain_sel.apply(pose)
        chain_indices = pyrosetta.rosetta.core.select.get_residues_from_subset(chain_subset)

        binder_pose = pyrosetta.rosetta.core.pose.Pose()
        pyrosetta.rosetta.core.pose.pdbslice(binder_pose, pose, chain_indices)

        layer_sel = LayerSelector()
        layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
        surface_res = layer_sel.apply(binder_pose)

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


def compute_interface_aa_counts(pose: Pose, binder_chain: str = 'A',
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

        pmhc_arr = np.array([[v.x, v.y, v.z] for v in pmhc_cb])

        for i in range(1, pose.total_residue() + 1):
            if pose.pdb_info().chain(i) != binder_chain:
                continue
            res = pose.residue(i)
            if not res.is_protein():
                continue
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--af3_out_dir', required=True,
                        help='Root AF3 output directory containing '
                             '{design}_{timestamp}/ subdirs')
    parser.add_argument('--out_dir',     required=True,
                        help='Output directory for relaxed PDBs and scores TSV')
    parser.add_argument('--designs',     nargs='*',
                        help='Design names to process (default: all found in af3_out_dir)')
    parser.add_argument('--n_repeats',   type=int, default=3)
    parser.add_argument('--inspect',     metavar='DESIGN',
                        help='Inspect chain layout of one design and exit')
    parser.add_argument('--dalphaball',
                        default='./DAlphaBall.gcc',
                        help='Path to DAlphaBall binary for BuriedUnsatHbonds SASA calculation')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    relaxed_dir = os.path.join(args.out_dir, 'relaxed_pdbs')
    os.makedirs(relaxed_dir, exist_ok=True)

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
        f'-holes:dalphaball {args.dalphaball} '
        f'-detect_disulf false',
        silent=True,
    )

    # -- Inspect mode ---------------------------------------------------------
    if args.inspect:
        cif = find_model_cif(args.af3_out_dir, args.inspect)
        if not cif:
            print(f"ERROR: no model CIF found for '{args.inspect}'")
            return
        pose   = load_cif(cif)
        chains = get_chain_ids(pose)
        iface  = build_interface_string(chains)
        print(f"Design:     {args.inspect}")
        print(f"CIF:        {cif}")
        print(f"Chains:     {chains}")
        print(f"Interface:  {iface}")
        print(f"N residues: {pose.total_residue()}")
        for ch in chains:
            sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(ch)
            rv  = pyrosetta.rosetta.core.select.residue_selector.ResidueVector(
                      sel.apply(pose))
            print(f"  Chain {ch}: {len(rv)} residues")
        return

    # -- Discover designs -----------------------------------------------------
    if args.designs:
        designs = args.designs
    else:
        subdirs = glob.glob(os.path.join(args.af3_out_dir, '*_*/'))
        names   = set()
        for d in subdirs:
            basename = os.path.basename(d.rstrip('/'))
            match = re.match(r'^(.+)_\d{8}_\d{6}$', basename)
            if match:
                names.add(match.group(1))
        designs = sorted(names)

    print(f"Processing {len(designs)} designs...")

    scorefxn = ScoreFunctionFactory.create_score_function('ref2015')
    scorefxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint, 1.0
    )

    records = []

    for idx, design in enumerate(designs):
        cif_path = find_model_cif(args.af3_out_dir, design)
        if not cif_path:
            print(f"  ERROR: model CIF not found for '{design}'")
            records.append({'design': design})
            continue

        print(f"  [{idx+1}/{len(designs)}] {design}")
        print(f"    CIF: {cif_path}")

        try:
            pose = load_cif(cif_path)

            chain_ids = get_chain_ids(pose)
            interface = build_interface_string(chain_ids)
            print(f"    Chains: {chain_ids}  ->  interface: {interface}")

            score_before = scorefxn(pose)
            pose         = run_fastrelax(pose, scorefxn, n_repeats=args.n_repeats)

            scorefxn_no_cst = ScoreFunctionFactory.create_score_function('ref2015')
            score_after     = scorefxn_no_cst(pose)

            relaxed_pdb = os.path.join(relaxed_dir, f'{design}_relaxed.pdb')
            pose.dump_pdb(relaxed_pdb)

            iface_scores = compute_interface_scores(pose, interface)
            if not iface_scores:
                print(f"    WARNING: no IAM scores in pose.scores for {design}")

            # BuriedUnsatHbonds -- BindCraft-equivalent delta unsat H-bonds
            buns      = compute_buns(pose)
            buns_pass = (buns < BUNS_THRESHOLD) if buns is not None else None
            if buns is not None:
                print(f"    buns_delta_unsat={buns:.1f}  "
                      f"({'PASS' if buns_pass else 'FAIL'} < {BUNS_THRESHOLD})")

            surf_hydrophob = compute_surface_hydrophobicity(pose, binder_chain='A')
            if surf_hydrophob is not None:
                print(f"    surface_hydrophobicity={surf_hydrophob:.3f}  "
                      f"({'PASS' if surf_hydrophob <= 0.35 else 'FAIL'} <= 0.35)")

            # Interface AA counts for K and M -- BindCraft threshold <= 3 each
            iface_aa = compute_interface_aa_counts(pose, binder_chain='A', aas=('K', 'M'))
            print(f"    interface_n_K={iface_aa['interface_n_K']}  "
                  f"interface_n_M={iface_aa['interface_n_M']}")

            confidences  = get_confidences(args.af3_out_dir, design)
            ranking_info = find_ranking_scores(args.af3_out_dir, design)

            record = {
                'design':                 design,
                'cif_path':               cif_path,
                'chains':                 ','.join(chain_ids),
                'interface':              interface,
                'rosetta_score_before':   round(score_before, 3),
                'rosetta_score_after':    round(score_after,  3),
                'rosetta_score_delta':    round(score_after - score_before, 3),
                'relaxed_pdb':            relaxed_pdb,
                'buns_delta_unsat':       buns,
                'buns_pass':              buns_pass,
                'surface_hydrophobicity': surf_hydrophob,
            }
            record.update(iface_aa)
            record.update(confidences)   # af3_iptm, af3_ptm, af3_fraction_disordered
            record.update(ranking_info)  # top_seed, top_sample, af3_ranking_score
            record.update(iface_scores)  # all IAM metrics
            records.append(record)

        except Exception as e:
            print(f"    ERROR on {design}: {e}")
            records.append({'design': design, 'cif_path': cif_path})

    out_df   = pd.DataFrame(records)
    out_path = os.path.join(args.out_dir, 'fastrelax_af3_scores.tsv')
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved -> {out_path}")

    # -- Summary --------------------------------------------------------------
    key_cols = [
        'af3_iptm', 'af3_ranking_score',
        'dG_separated', 'dG_separated/dSASAx100', 'dSASA_int',
        'delta_unsatHbonds', 'hbonds_int', 'packstat',
        'sc_value', 'per_residue_energy_int', 'buns_delta_unsat',
        'surface_hydrophobicity', 'interface_n_K', 'interface_n_M',
    ]
    anchor = 'dG_separated'
    valid  = out_df.dropna(subset=[anchor]) if anchor in out_df.columns else out_df
    print(f"\n-- Summary ({len(valid)} designs) --")
    for col in key_cols:
        if col in valid.columns and valid[col].notna().any():
            print(f"  {col}: mean={valid[col].mean():.3f}, "
                  f"min={valid[col].min():.3f}, "
                  f"max={valid[col].max():.3f}")

    if 'buns_pass' in out_df.columns and out_df['buns_pass'].notna().any():
        n_pass = out_df['buns_pass'].sum()
        print(f"\n-- BuriedUnsatHbonds filter (< {BUNS_THRESHOLD}, BindCraft default) --")
        print(f"  Pass: {int(n_pass)}/{len(valid)}")
        print(f"  Fail: {int(len(valid) - n_pass)}/{len(valid)}")


if __name__ == '__main__':
    main()