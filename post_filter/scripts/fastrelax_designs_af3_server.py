"""
fastrelax_designs_af3_server.py

Runs Rosetta FastRelax on AF3 server output CIF files for a set of designs,
then computes the full InterfaceAnalyzer score suite.

AF3 server output convention (model index 0 used for all processing):
  {af3_out_dir}/{design}/fold_{design}_model_0.cif
  {af3_out_dir}/{design}/fold_{design}_summary_confidences_0.json
  {af3_out_dir}/{design}/fold_{design}_full_data_0.json

Chain convention (matching your AF3 JSON input):
  A = binder
  B = MHC alpha
  C = peptide
  D = B2M (if present -- 4-chain complex)

Interface analysis is run on A_BC (binder vs MHC+peptide) for 3-chain inputs,
or A_BCD (binder vs full pMHC+B2M) for 4-chain inputs. Chain layout is
auto-detected from the CIF.

Usage:
    python fastrelax_designs_af3_server.py \
        --af3_out_dir ~/Desktop/pmhc_binder_kras_cp/post_filter/outputs/r2.5/af3_nomsa/ \
        --out_dir     ~/Desktop/pmhc_binder_kras_cp/post_filter/outputs/r2.5/fastrelax_af3/ \
        --dalphaball  ~/Desktop/pmhc_binder_kras/post_filter/scripts/dalphaball.gcc \
        --n_repeats   3

    # Run on specific designs:
    python fastrelax_designs_af3_server.py \
        --af3_out_dir ... --out_dir ... \
        --designs orient_252_pt20_31_sample06_a11_t015_s7

    # Inspect chain layout of one CIF before running:
    python fastrelax_designs_af3_server.py \
        --af3_out_dir ... --out_dir ... \
        --inspect orient_252_pt20_31_sample06_a11_t015_s7
"""

from __future__ import annotations

import os
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

MODEL_IDX   = 0   # AF3 server outputs 0-4; always use index 0
BUNS_THRESHOLD = 4


# ── AF3 server file discovery ─────────────────────────────────────────────────

def find_model_cif(af3_out_dir: str, design: str) -> str | None:
    """
    Find fold_{design}_model_{MODEL_IDX}.cif inside {af3_out_dir}/{design}/.
    Returns path or None.
    """
    path = os.path.join(
        af3_out_dir, design,
        f'fold_{design}_model_{MODEL_IDX}.cif'
    )
    return path if os.path.exists(path) else None


def get_confidences(af3_out_dir: str, design: str) -> dict:
    """
    Read fold_{design}_summary_confidences_{MODEL_IDX}.json.
    Returns iptm/ptm/fraction_disordered or empty dict.
    """
    path = os.path.join(
        af3_out_dir, design,
        f'fold_{design}_summary_confidences_{MODEL_IDX}.json'
    )
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {
            'af3_iptm':                data.get('iptm',                np.nan),
            'af3_ptm':                 data.get('ptm',                 np.nan),
            'af3_fraction_disordered': data.get('fraction_disordered', np.nan),
            'af3_ranking_score':       data.get('ranking_score',       np.nan),
        }
    except Exception:
        return {}


def discover_designs(af3_out_dir: str) -> list[str]:
    """
    Return sorted list of design names — subdirectory names directly under
    af3_out_dir that contain a fold_*_model_0.cif file.
    """
    designs = []
    for entry in sorted(os.scandir(af3_out_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        cif = os.path.join(entry.path, f'fold_{entry.name}_model_{MODEL_IDX}.cif')
        if os.path.exists(cif):
            designs.append(entry.name)
    return designs


# ── CIF loading ───────────────────────────────────────────────────────────────

def load_cif(cif_path: str) -> Pose:
    pose = Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_file(
        pose, cif_path, False, FileType.CIF_file
    )
    return pose


def get_chain_ids(pose: Pose) -> list[str]:
    chains, seen = [], set()
    for i in range(1, pose.total_residue() + 1):
        ch = pose.pdb_info().chain(i)
        if ch not in seen:
            seen.add(ch)
            chains.append(ch)
    return chains


def build_interface_string(chain_ids: list[str]) -> str:
    binder = 'A'
    target_chains = [c for c in chain_ids if c != binder]
    if not target_chains:
        raise ValueError(f"No non-binder chains found: {chain_ids}")
    return f"{binder}_{''.join(target_chains)}"


# ── Rosetta helpers ───────────────────────────────────────────────────────────

def run_fastrelax(pose: Pose, scorefxn, n_repeats: int = 3) -> Pose:
    fr = FastRelax(scorefxn_in=scorefxn, standard_repeats=n_repeats)
    fr.constrain_relax_to_start_coords(True)
    fr.apply(pose)
    return pose


def compute_interface_scores(pose: Pose, interface: str) -> dict:
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
    try:
        if pose.num_chains() >= 4:
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
            xyz = res.xyz('CB') if res.has('CB') else (res.xyz('CA') if res.has('CA') else None)
            if xyz is None:
                continue
            coord = np.array([xyz.x, xyz.y, xyz.z])
            if np.linalg.norm(pmhc_arr - coord, axis=1).min() <= dist_cutoff:
                one_letter = THREE_TO_ONE.get(res.name3())
                if one_letter in aas:
                    counts[f'interface_n_{one_letter}'] += 1
    except Exception as e:
        print(f"    WARNING: interface_aa_counts failed: {e}")
    return counts


def init_pyrosetta(dalphaball: str) -> None:
    pyrosetta.init(
        f'-mute all '
        f'-ex1 -ex2aro '
        f'-use_input_sc '
        f'-ignore_unrecognized_res true '
        f'-ignore_zero_occupancy false '
        f'-load_PDB_components false '
        f'-holes:dalphaball {dalphaball} '
        f'-detect_disulf false',
        silent=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--af3_out_dir', required=True,
        help='Root AF3 server output directory containing {design}/ subdirs')
    parser.add_argument('--out_dir',     required=True,
        help='Output directory for relaxed PDBs and scores TSV')
    parser.add_argument('--designs',     nargs='*',
        help='Design names to process (default: all discovered in af3_out_dir)')
    parser.add_argument('--n_repeats',   type=int, default=3)
    parser.add_argument('--inspect',     metavar='DESIGN',
        help='Inspect chain layout of one design and exit')
    parser.add_argument('--dalphaball',
        default='~/Desktop/pmhc_binder_kras/post_filter/scripts/dalphaball.gcc',
        help='Path to DAlphaBall binary')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    relaxed_dir = os.path.join(args.out_dir, 'relaxed_pdbs')
    os.makedirs(relaxed_dir, exist_ok=True)

    if not os.path.exists(args.dalphaball):
        raise FileNotFoundError(
            f"DAlphaBall binary not found: {args.dalphaball}\n"
            f"Pass --dalphaball <path>."
        )

    init_pyrosetta(args.dalphaball)

    # ── Inspect mode ─────────────────────────────────────────────────────────
    if args.inspect:
        cif = find_model_cif(args.af3_out_dir, args.inspect)
        if not cif:
            print(f"ERROR: model CIF not found for '{args.inspect}'")
            print(f"  Expected: {os.path.join(args.af3_out_dir, args.inspect, f'fold_{args.inspect}_model_{MODEL_IDX}.cif')}")
            return
        pose   = load_cif(cif)
        chains = get_chain_ids(pose)
        iface  = build_interface_string(chains)
        print(f"Design:     {args.inspect}")
        print(f"CIF:        {cif}")
        print(f"Chains:     {chains}  ->  interface: {iface}")
        print(f"N residues: {pose.total_residue()}")
        for ch in chains:
            sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(ch)
            rv  = pyrosetta.rosetta.core.select.residue_selector.ResidueVector(sel.apply(pose))
            print(f"  Chain {ch}: {len(rv)} residues")
        return

    # ── Discover designs ──────────────────────────────────────────────────────
    designs = args.designs if args.designs else discover_designs(args.af3_out_dir)
    print(f"Processing {len(designs)} designs  (model index {MODEL_IDX})...")

    scorefxn = ScoreFunctionFactory.create_score_function('ref2015')
    scorefxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint, 1.0
    )

    records = []

    for idx, design in enumerate(designs):
        cif_path = find_model_cif(args.af3_out_dir, design)
        if not cif_path:
            print(f"  [{idx+1}/{len(designs)}] ERROR: model CIF not found for '{design}'")
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
                print(f"    WARNING: no IAM scores for {design}")

            buns      = compute_buns(pose)
            buns_pass = (buns < BUNS_THRESHOLD) if buns is not None else None
            if buns is not None:
                print(f"    buns_delta_unsat={buns:.1f}  "
                      f"({'PASS' if buns_pass else 'FAIL'} < {BUNS_THRESHOLD})")

            surf_hydrophob = compute_surface_hydrophobicity(pose, binder_chain='A')
            if surf_hydrophob is not None:
                print(f"    surface_hydrophobicity={surf_hydrophob:.3f}  "
                      f"({'PASS' if surf_hydrophob <= 0.35 else 'FAIL'} <= 0.35)")

            iface_aa = compute_interface_aa_counts(pose, binder_chain='A', aas=('K', 'M'))
            print(f"    interface_n_K={iface_aa['interface_n_K']}  "
                  f"interface_n_M={iface_aa['interface_n_M']}")

            confidences = get_confidences(args.af3_out_dir, design)

            record = {
                'design':                 design,
                'cif_path':               cif_path,
                'model_idx':              MODEL_IDX,
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
            record.update(confidences)   # af3_iptm, af3_ptm, af3_fraction_disordered, af3_ranking_score
            record.update(iface_scores)  # all IAM metrics
            records.append(record)

        except Exception as e:
            print(f"    ERROR on {design}: {e}")
            records.append({'design': design, 'cif_path': cif_path})

    out_df   = pd.DataFrame(records)
    out_path = os.path.join(args.out_dir, 'fastrelax_af3_scores.tsv')
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved -> {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
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
        print(f"\n-- BuriedUnsatHbonds filter (< {BUNS_THRESHOLD}) --")
        print(f"  Pass: {int(n_pass)}/{len(valid)}")
        print(f"  Fail: {int(len(valid) - n_pass)}/{len(valid)}")


if __name__ == '__main__':
    main()