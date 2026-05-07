"""
surface_hydrophobicity_resnums.py
==================================
Reads relaxed PDBs and identifies surface-exposed hydrophobic residues on the
binder chain using PyRosetta's LayerSelector.

Imports compute_surface_hydrophobicity directly from fastrelax_designs_af3.py
to guarantee identical computation. Extends it to also return the residue
numbers and amino acid identities.

Adds two columns to an existing TSV (or outputs a new one):
  surface_hydrophobic_resnums  — PDB residue numbers of surface-exposed
                                  hydrophobic binder residues
  surface_hydrophobic_aas      — corresponding single-letter amino acid codes
  surface_hydrophobicity_check — recomputed fraction (cross-check vs stored)

Usage:
    # Add columns to any existing TSV:
    python surface_hydrophobicity_resnums.py \
        --relaxed_pdb_dir /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2/fastrelax_af3_nomsa/relaxed_pdbs/ \
        --in_tsv          /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2/prioritization/redesign_list.tmp \
        --out_tsv         /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2/prioritization/redesign_list.tsv

    python surface_hydrophobicity_resnums.py \
        --relaxed_pdb_dir /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2/fastrelax_af3_nomsa/relaxed_pdbs/ \
        --in_tsv          /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2/prioritization/redesign_list_clean.tsv \
        --out_tsv         /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2/prioritization/redesign_list_clean_hydrophobic.tsv

    # Specific designs only:
    python surface_hydrophobicity_resnums.py \
        --relaxed_pdb_dir ... --in_tsv ... --out_tsv ... \
        --designs orient_255_pt12__33_sample06 orient_255_pt12__63_sample06
"""

import os
import sys
import argparse

import pandas as pd
import pyrosetta

# # Import from fastrelax_designs_af3.py — must be on PYTHONPATH or same dir
# _IMPORT_ERROR = None
# for _mod in ('fastrelax_designs_af3'):
#     try:
#         import importlib
#         _m = importlib.import_module(_mod)
#         compute_surface_hydrophobicity = _m.compute_surface_hydrophobicity
#         break
#     except ImportError as e:
#         _IMPORT_ERROR = e
# else:
#     print(f"ERROR: could not import from fastrelax_designs_af3.py: "
#           f"{_IMPORT_ERROR}")
#     print("Ensure fastrelax_designs_af3.py is in the same directory or on PYTHONPATH.")
#     sys.exit(1)

from fastrelax_designs_af3 import compute_surface_hydrophobicity

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}



# ─────────────────────────────────────────────────────────────────────────────
# Surface hydrophobic residue detection
# ─────────────────────────────────────────────────────────────────────────────

def get_surface_hydrophobic_residues(
    pose: pyrosetta.Pose,
    binder_chain: str = 'A',
) -> tuple[list[int], list[str], float]:
    """
    Calls imported compute_surface_hydrophobicity() for the fraction value,
    then re-runs the same pdbslice + LayerSelector logic to collect residue
    numbers and amino acid identities of the surface-exposed hydrophobic residues.

    Returns (resnums, aas, fraction):
      resnums  — PDB residue numbers (original numbering from pose.pdb_info())
      aas      — single-letter amino acid codes
      fraction — matches compute_surface_hydrophobicity() exactly
    """
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector, LayerSelector

    # Fraction via imported function (identical to fastrelax_designs_af3.py)
    fraction = compute_surface_hydrophobicity(pose, binder_chain=binder_chain)

    # Residue-level detail — same pdbslice + LayerSelector logic
    chain_sel     = ChainSelector(binder_chain)
    chain_subset  = chain_sel.apply(pose)
    chain_indices = pyrosetta.rosetta.core.select.get_residues_from_subset(chain_subset)

    # PDB resnum mapping before slicing
    pose_to_pdb = {i: pose.pdb_info().number(i)
                   for i in range(1, pose.total_residue() + 1)
                   if pose.pdb_info().chain(i) == binder_chain}

    binder_pose = pyrosetta.rosetta.core.pose.Pose()
    pyrosetta.rosetta.core.pose.pdbslice(binder_pose, pose, chain_indices)

    layer_sel = LayerSelector()
    layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
    surface_res = layer_sel.apply(binder_pose)

    idx_list = list(chain_indices)
    resnums  = []
    aas      = []

    for sub_i in range(1, binder_pose.total_residue() + 1):
        if not surface_res[sub_i]:
            continue
        res = binder_pose.residue(sub_i)
        if not res.is_protein():
            continue
        if res.is_apolar() or res.name3() in ('PHE', 'TRP', 'TYR'):
            orig_pose_idx = idx_list[sub_i - 1]
            pdb_resnum    = pose_to_pdb.get(orig_pose_idx, sub_i)
            aa            = THREE_TO_ONE.get(res.name3().strip(), 'X')
            resnums.append(pdb_resnum)
            aas.append(aa)

    return resnums, aas, fraction if fraction is not None else float('nan')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--relaxed_pdb_dir', required=True,
        help='Directory containing {design}_relaxed.pdb files')
    parser.add_argument('--in_tsv',          required=True,
        help='Input TSV to merge results into (e.g. fastrelax_af3_scores.tsv)')
    parser.add_argument('--out_tsv',         required=True,
        help='Output TSV path (input TSV + two new columns)')
    parser.add_argument('--designs',         nargs='*',
        help='Specific designs to process (default: all in --in_tsv)')
    parser.add_argument('--binder_chain',    default='A',
        help='Binder chain ID in relaxed PDB (default: A)')
    parser.add_argument('--dalphaball',
        default='./DAlphaBall.gcc',
        help='Path to DAlphaBall binary for BuriedUnsatHbonds SASA calculation')
    args = parser.parse_args()

    df = pd.read_csv(args.in_tsv, sep='\t')
    print(f"Loaded {len(df)} designs from {args.in_tsv}")

    designs = args.designs if args.designs else df['design'].tolist()
    print(f"Processing {len(designs)} designs\n")

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

    records = []
    for idx, design in enumerate(designs):
        pdb = os.path.join(args.relaxed_pdb_dir, f'{design}_relaxed.pdb')
        if not os.path.exists(pdb):
            print(f"  [{idx+1}/{len(designs)}] {design}: PDB not found, skipping")
            records.append({
                'design':                       design,
                'surface_hydrophobic_resnums':  '[]',
                'surface_hydrophobic_aas':      '[]',
                'surface_hydrophobicity_check': float('nan'),
            })
            continue

        try:
            pose     = pyrosetta.pose_from_pdb(pdb)
            resnums, aas, frac = get_surface_hydrophobic_residues(
                pose, binder_chain=args.binder_chain
            )
            records.append({
                'design':                       design,
                'surface_hydrophobic_resnums':  str(resnums),
                'surface_hydrophobic_aas':      str(aas),
                'surface_hydrophobicity_check': frac,
            })
            print(f"  [{idx+1}/{len(designs)}] {design}: "
                  f"frac={frac:.3f}  n={len(resnums)}  "
                  f"resnums={resnums}  aas={aas}")
        except Exception as e:
            print(f"  [{idx+1}/{len(designs)}] {design}: ERROR {e}")
            records.append({
                'design':                       design,
                'surface_hydrophobic_resnums':  '[]',
                'surface_hydrophobic_aas':      '[]',
                'surface_hydrophobicity_check': float('nan'),
            })

    df_new = pd.DataFrame(records)

    # Cross-check fraction against existing surface_hydrophobicity column
    if 'surface_hydrophobicity' in df.columns:
        df_new = df_new.merge(
            df[['design', 'surface_hydrophobicity']], on='design', how='left'
        )
        discrepant = df_new[
            (df_new['surface_hydrophobicity_check'].notna()) &
            (df_new['surface_hydrophobicity'].notna()) &
            (abs(df_new['surface_hydrophobicity_check'] -
                 df_new['surface_hydrophobicity']) > 0.05)
        ]
        if len(discrepant):
            print(f"\nWARNING: {len(discrepant)} designs with >0.05 discrepancy "
                  f"between recomputed and stored surface_hydrophobicity:")
            for _, r in discrepant.iterrows():
                print(f"  {r['design']}: stored={r['surface_hydrophobicity']:.3f}  "
                      f"recomputed={r['surface_hydrophobicity_check']:.3f}")
        df_new = df_new.drop(columns=['surface_hydrophobicity'])

    # Merge into input TSV
    merge_cols = ['surface_hydrophobic_resnums', 'surface_hydrophobic_aas',
                  'surface_hydrophobicity_check']
    # Drop existing columns if present (rerun case)
    df = df.drop(columns=[c for c in merge_cols if c in df.columns])
    df_out = df.merge(df_new[['design'] + merge_cols], on='design', how='left')

    df_out.to_csv(args.out_tsv, sep='\t', index=False)
    print(f"\nSaved -> {args.out_tsv}")

    # Summary of high-hydrophobicity designs
    thresh = 0.35  # BindCraft threshold
    high = df_out[df_out.get('surface_hydrophobicity', pd.Series(dtype=float)) > thresh] \
           if 'surface_hydrophobicity' in df_out.columns else \
           df_out[df_out['surface_hydrophobicity_check'] > thresh]
    if len(high):
        print(f"\nDesigns with surface_hydrophobicity > {thresh} "
              f"({len(high)}/{len(df_out)}):")
        for _, r in high.sort_values('surface_hydrophobicity_check',
                                     ascending=False).iterrows():
            print(f"  {r['design']:<45}  "
                  f"frac={r['surface_hydrophobicity_check']:.3f}  "
                  f"resnums={r['surface_hydrophobic_resnums']}  "
                  f"aas={r['surface_hydrophobic_aas']}")


if __name__ == '__main__':
    main()
