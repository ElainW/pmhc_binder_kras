"""
compute_binder_metrics.py

Compute per-design structural metrics from post-filter pMHC fold PDBs.
Chain layout: A = MHC (groove), B = MHC (beta2m / non-interacting),
              C = peptide, D = binder

Metrics computed:
  - binder Rg (Cα only)
  - number of helix segments (DSSP on binder subpose)
  - helix fraction
  - binder–peptide orientation angle (SVD principal axes)
  - binder COM lateral and vertical displacement from peptide COM
    (decomposed along MHC chain A principal axis)

Output: TSV with one row per design.

Usage:
    python compute_binder_metrics.py \\
        --input_dir ~/Desktop/pmhc_binder_kras/post_filter/inputs/r2/ \\
        --output_tsv ~/Desktop/pmhc_binder_kras/post_filter/analysis/r2/binder_metrics.tsv \\
        [--pattern "*.pdb"] \\
        [--min_helix_length 4]
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.protocols.moves import DsspMover

init("-mute all")


# ── Chain utilities ───────────────────────────────────────────────────────────

def get_chain_index_by_letter(pose, chain_letter):
    """
    Return the PyRosetta internal chain index (1-based) for a given PDB chain
    letter, using pdb_info to look up the actual letter rather than assuming
    alphabetical order.  Raises ValueError if the chain is not found.

    This is necessary because PyRosetta numbers chains by their order of
    appearance in the PDB file, which may differ from alphabetical order
    (e.g. chain D first, then A, B, C).
    """
    pdb_info = pose.pdb_info()
    for chain_idx in range(1, pose.num_chains() + 1):
        # any residue in this chain will do; use the first one
        res_idx = pose.chain_begin(chain_idx)
        if pdb_info.chain(res_idx) == chain_letter:
            return chain_idx
    raise ValueError(
        f"Chain '{chain_letter}' not found in pose. "
        f"Available chains: "
        + ", ".join(
            pose.pdb_info().chain(pose.chain_begin(i))
            for i in range(1, pose.num_chains() + 1)
        )
    )


def get_chain_pose(pose, chain="A"):
    """Extract a single chain subpose by PDB chain letter."""
    chain_idx = get_chain_index_by_letter(pose, chain)
    return pose.split_by_chain(chain_idx)


def get_ca_coords(pose, chain):
    """Return (N, 3) array of Cα coordinates for a given PDB chain letter."""
    subpose = get_chain_pose(pose, chain)
    coords = []
    for i in range(1, subpose.total_residue() + 1):
        res = subpose.residue(i)
        if res.has("CA"):
            ca = res.xyz("CA")
            coords.append([ca.x, ca.y, ca.z])
    return np.array(coords)


# ── Metric functions ──────────────────────────────────────────────────────────

def calc_rg(coords):
    """Radius of gyration from a (N, 3) Cα coordinate array."""
    centroid = coords.mean(axis=0)
    rg = np.sqrt(((coords - centroid) ** 2).sum(axis=1).mean())
    return float(rg)


def calc_helix_segments(pose, binder_chain="D", min_helix_length=7):
    """
    Run DSSP on the binder subpose only and return helix metrics.

    Returns:
        ss_string      : DSSP string for the binder
        n_helix_segs   : number of helix segments >= min_helix_length
        helix_lengths  : list of individual helix lengths
        helix_fraction : fraction of binder residues in helix
    """
    binder_pose = get_chain_pose(pose, binder_chain)
    DsspMover().apply(binder_pose)
    ss_string = binder_pose.secstruct()

    segments = []
    count = 0
    for c in ss_string:
        if c == 'H':
            count += 1
        else:
            if count >= min_helix_length:
                segments.append(count)
            count = 0
    if count >= min_helix_length:
        segments.append(count)

    helix_fraction = ss_string.count('H') / len(ss_string) if ss_string else 0.0

    return ss_string, len(segments), segments, round(helix_fraction, 3)


def principal_axis(coords):
    centered = coords - coords.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered)
    return Vt[0]


def calc_orientation_angle(binder_coords, peptide_coords):
    """
    Angle (degrees) between binder and peptide principal axes.
    Uses abs(dot) to handle SVD sign ambiguity.
    """
    b_axis = principal_axis(binder_coords)
    p_axis = principal_axis(peptide_coords)
    cos_a  = np.clip(np.abs(np.dot(b_axis, p_axis)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def calc_com_displacement(binder_coords, peptide_coords, mhc_coords):
    """
    Decompose binder–peptide COM displacement into lateral and vertical
    components using MHC chain A principal axis as the groove direction.

    lateral  : displacement parallel to MHC principal axis (along groove)
    vertical : displacement perpendicular to MHC principal axis (above groove)
    """
    binder_com  = binder_coords.mean(axis=0)
    peptide_com = peptide_coords.mean(axis=0)
    mhc_axis    = principal_axis(mhc_coords)

    disp         = binder_com - peptide_com
    vertical     = float(np.abs(np.dot(disp, mhc_axis)))
    lateral      = float(np.linalg.norm(disp - np.dot(disp, mhc_axis) * mhc_axis))
    return lateral, vertical


# ── Main loop ─────────────────────────────────────────────────────────────────

def compute_metrics(input_dir, output_tsv,
                    binder_chain="D", peptide_chain="C", mhc_chain="A",
                    min_helix_length=4, pattern="*.pdb"):

    pdb_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    print(f"Found {len(pdb_files)} PDB files in {input_dir}")

    rows = []
    for pdb_path in pdb_files:
        design = os.path.splitext(os.path.basename(pdb_path))[0]
        try:
            pose = pose_from_pdb(pdb_path)

            binder_coords  = get_ca_coords(pose, binder_chain)
            peptide_coords = get_ca_coords(pose, peptide_chain)
            mhc_coords     = get_ca_coords(pose, mhc_chain)

            rg = calc_rg(binder_coords)

            ss_string, n_helix_segs, helix_lengths, helix_fraction = \
                calc_helix_segments(pose, binder_chain, min_helix_length)

            orientation_angle = calc_orientation_angle(binder_coords, peptide_coords)

            lateral_dist, vertical_dist = calc_com_displacement(
                binder_coords, peptide_coords, mhc_coords
            )

            rows.append({
                "design":             design,
                "pdb_path":           pdb_path,
                "n_binder_res":       len(binder_coords),
                "rg":                 round(rg, 3),
                "n_helix_segs":       n_helix_segs,
                "helix_fraction":     helix_fraction,
                "helix_lengths":      str(helix_lengths),
                "ss_string":          ss_string,
                "orientation_angle":  round(orientation_angle, 2),
                "lateral_dist":       round(lateral_dist, 2),
                "vertical_dist":      round(vertical_dist, 2),
            })

        except Exception as e:
            print(f"  WARNING: failed {pdb_path}: {e}")
            rows.append({"design": design, "pdb_path": pdb_path, "error": str(e)})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
    df.to_csv(output_tsv, sep='\t', index=False)
    print(f"\nWrote {len(df)} rows to {output_tsv}")

    # summary stats for numeric columns
    metric_cols = ["rg", "n_helix_segs", "helix_fraction",
                   "orientation_angle", "lateral_dist", "vertical_dist"]
    present = [c for c in metric_cols if c in df.columns]
    print("\nMetric summary:")
    print(df[present].describe(percentiles=[0.25, 0.5, 0.75]).round(3).to_string())

    return df


# ── Argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Rg, helix segments, orientation angle, and COM "
                    "displacement for pMHC fold PDBs (chains A=MHC, B=MHC-beta2m, "
                    "C=peptide, D=binder)."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing per-design PDB files."
    )
    parser.add_argument(
        "--output_tsv", required=True,
        help="Path for output TSV file."
    )
    parser.add_argument(
        "--binder_chain", default="D",
        help="Chain letter of the binder (default: D)."
    )
    parser.add_argument(
        "--peptide_chain", default="C",
        help="Chain letter of the peptide (default: C)."
    )
    parser.add_argument(
        "--mhc_chain", default="A",
        help="Chain letter of the groove-forming MHC chain (default: A)."
    )
    parser.add_argument(
        "--min_helix_length", type=int, default=7,
        help="Minimum consecutive helix residues to count as a segment (default: 7)."
    )
    parser.add_argument(
        "--pattern", default="*.pdb",
        help="Glob pattern for PDB files (default: *.pdb)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_metrics(
        input_dir       = args.input_dir,
        output_tsv      = args.output_tsv,
        binder_chain    = args.binder_chain,
        peptide_chain   = args.peptide_chain,
        mhc_chain       = args.mhc_chain,
        min_helix_length= args.min_helix_length,
        pattern         = args.pattern,
    )