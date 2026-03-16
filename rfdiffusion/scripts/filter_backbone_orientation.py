import os
import glob
import argparse
import numpy as np
import pandas as pd
from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.core.pose import Pose

init("-mute all")


# ── Shared utilities ──────────────────────────────────────────────────────────

def get_chain_pose(pose, chain="A"):
    """Extract a single chain from a pose by letter."""
    chain_id = ord(chain) - ord('A') + 1
    return pose.split_by_chain(chain_id)


def get_ca_coords_by_letter(pose, chain="A"):
    """Extract Cα coordinates for a given chain letter."""
    chain_pose = get_chain_pose(pose, chain)
    coords = []
    for i in range(1, chain_pose.total_residue() + 1):
        residue = chain_pose.residue(i)
        if residue.has("CA"):
            ca = residue.xyz("CA")
            coords.append([ca.x, ca.y, ca.z])
    return np.array(coords)


def inspect_chains_detailed(pdb_path):
    """
    Print chain letters, residue counts, and residue index ranges.
    Run on one PDB first to confirm chain letters before filtering.
    """
    pose = pose_from_pdb(pdb_path)
    print(f"\n{pdb_path}")
    print(f"  Total residues: {pose.total_residue()}")
    for chain_idx in range(1, pose.num_chains() + 1):
        chain_letter = chr(ord('A') + chain_idx - 1)
        start     = pose.chain_begin(chain_idx)
        end       = pose.chain_end(chain_idx)
        n_res     = end - start + 1
        first_res = pose.residue(start).name3()
        last_res  = pose.residue(end).name3()
        print(f"  Chain {chain_letter}: residues {start}-{end} "
              f"({n_res} res) [{first_res}...{last_res}]")


# ── Orientation filters ───────────────────────────────────────────────────────

def calc_orientation_angle(pose, binder_chain="A", peptide_chain="C"):
    """
    Calculate the angle between the binder's principal axis and the
    peptide's principal axis via SVD.

    A well-docked binder sitting over the groove should be roughly
    parallel to the peptide (low angle). A perpendicular binder like
    the one docking against the MHC helices will have a high angle.

    Returns:
        angle (float): angle in degrees between binder and peptide axes.
                       Suggest filtering out > 60°.
    """
    def principal_axis(coords):
        centered = coords - coords.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered)
        return Vt[0]  # first right singular vector = principal axis

    binder_coords  = get_ca_coords_by_letter(pose, binder_chain)
    peptide_coords = get_ca_coords_by_letter(pose, peptide_chain)

    binder_axis  = principal_axis(binder_coords)
    peptide_axis = principal_axis(peptide_coords)

    # abs: axis direction is arbitrary (SVD can flip sign)
    cos_angle = np.clip(np.dot(binder_axis, peptide_axis), -1.0, 1.0)
    angle     = np.degrees(np.arccos(np.abs(cos_angle)))
    return angle


def calc_binder_com_displacement(pose, binder_chain="A", peptide_chain="C",
                                  mhc_chain="B"):
    """
    Compute displacement of the binder COM relative to the peptide COM,
    decomposed into lateral (in-plane) and vertical (normal) components
    using the MHC principal axis as the groove direction reference.

    A well-positioned binder should have:
      - low lateral displacement (COM is above the peptide, not off to the side)
      - moderate vertical displacement (binder is hovering above, not clashing)

    Returns:
        lateral_dist  (float): displacement in Å parallel to MHC groove axis
        vertical_dist (float): displacement in Å perpendicular to MHC groove axis
    """
    def get_com(coords):
        return coords.mean(axis=0)

    binder_com  = get_com(get_ca_coords_by_letter(pose, binder_chain))
    peptide_com = get_com(get_ca_coords_by_letter(pose, peptide_chain))
    mhc_coords  = get_ca_coords_by_letter(pose, mhc_chain)

    # MHC principal axis ≈ groove direction
    centered   = mhc_coords - mhc_coords.mean(axis=0)
    _, _, Vt   = np.linalg.svd(centered)
    mhc_normal = Vt[0]

    displacement  = binder_com - peptide_com
    vertical_dist = np.abs(np.dot(displacement, mhc_normal))
    lateral_dist  = np.linalg.norm(
        displacement - np.dot(displacement, mhc_normal) * mhc_normal
    )
    return lateral_dist, vertical_dist


def calc_peptide_vs_mhc_proximity(pose, binder_chain="A",
                                   peptide_chain="C", mhc_chain="B"):
    """
    For each binder residue, determine whether it is closer to the
    peptide or to the MHC. A well-docked binder should have a
    substantial fraction of residues closer to the peptide.

    Returns:
        peptide_fraction (float): fraction of binder residues whose
                                  nearest neighbour is on the peptide
                                  rather than the MHC.
    """
    binder_coords  = get_ca_coords_by_letter(pose, binder_chain)
    peptide_coords = get_ca_coords_by_letter(pose, peptide_chain)
    mhc_coords     = get_ca_coords_by_letter(pose, mhc_chain)

    dist_to_peptide = np.linalg.norm(
        binder_coords[:, None, :] - peptide_coords[None, :, :], axis=-1
    ).min(axis=1)

    dist_to_mhc = np.linalg.norm(
        binder_coords[:, None, :] - mhc_coords[None, :, :], axis=-1
    ).min(axis=1)

    peptide_fraction = (dist_to_peptide < dist_to_mhc).sum() / len(binder_coords)
    return float(peptide_fraction)


def check_binder_orientation(pose,
                              binder_chain="A",
                              peptide_chain="C",
                              mhc_chain="B",
                              max_angle=60.0,
                              max_lateral_dist=20.0,
                              min_peptide_fraction=0.3):
    """
    Combined orientation filter. Runs all three checks and returns a
    results dict with individual metrics and pass/fail flags.

    Args:
        max_angle:             Maximum allowed angle (°) between binder and
                               peptide principal axes (default: 60)
        max_lateral_dist:      Maximum allowed lateral COM displacement in Å
                               from peptide COM (default: 20)
        min_peptide_fraction:  Minimum fraction of binder residues that must
                               be closer to peptide than MHC (default: 0.3)
    """
    angle                       = calc_orientation_angle(pose, binder_chain, peptide_chain)
    lateral_dist, vertical_dist = calc_binder_com_displacement(pose, binder_chain, peptide_chain, mhc_chain)
    peptide_fraction            = calc_peptide_vs_mhc_proximity(pose, binder_chain, peptide_chain, mhc_chain)

    passes_angle    = angle          <  max_angle
    passes_lateral  = lateral_dist   <  max_lateral_dist
    passes_fraction = peptide_fraction >= min_peptide_fraction
    passes_all      = passes_angle and passes_lateral and passes_fraction

    return {
        "orientation_angle":  round(angle, 2),
        "lateral_dist":       round(lateral_dist, 2),
        "vertical_dist":      round(vertical_dist, 2),
        "peptide_fraction":   round(peptide_fraction, 3),
        "passes_angle":       passes_angle,
        "passes_lateral":     passes_lateral,
        "passes_fraction":    passes_fraction,
        "passes_orientation": passes_all,
    }


# ── Main filter loop ──────────────────────────────────────────────────────────

def filter_orientation(
    input_dirs,
    output_dir,
    binder_chain="A",
    peptide_chain="C",
    mhc_chain="B",
    max_angle=60.0,
    max_lateral_dist=20.0,
    min_peptide_fraction=0.3,
    pattern="*.pdb",
):
    """
    Filter PDB backbones by binder orientation relative to the pMHCI complex.

    Intended to be run AFTER filter.py (geometric pre-filters) on the
    already-filtered output directory.

    Args:
        input_dirs:            List of directories containing PDB files
        output_dir:            Directory to write passing structures and log
        binder_chain:          Chain letter of the designed binder (default: A)
        peptide_chain:         Chain letter of the peptide (default: C)
        mhc_chain:             Chain letter of the MHC (default: B)
        max_angle:             Max angle in ° between binder and peptide axes
        max_lateral_dist:      Max lateral COM displacement in Å from peptide
        min_peptide_fraction:  Min fraction of binder residues closer to peptide
        pattern:               Glob pattern for PDB files
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    passed  = 0

    for input_dir in input_dirs:
        pdb_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
        print(f"Found {len(pdb_files)} structures in {input_dir}")

        for pdb_path in pdb_files:
            try:
                pose   = pose_from_pdb(pdb_path)
                orient = check_binder_orientation(
                    pose,
                    binder_chain          = binder_chain,
                    peptide_chain         = peptide_chain,
                    mhc_chain             = mhc_chain,
                    max_angle             = max_angle,
                    max_lateral_dist      = max_lateral_dist,
                    min_peptide_fraction  = min_peptide_fraction,
                )

                result = {
                    "filename_old": pdb_path,
                    "filename_new": "NA",
                    **orient,
                }
                results.append(result)

                if orient["passes_orientation"]:
                    filename_new = os.path.join(output_dir, f"kras_orient_{passed}.pdb")
                    pose.dump_pdb(filename_new)
                    passed += 1
                    results[-1]["filename_new"] = filename_new

            except Exception as e:
                print(f"  Warning: failed to process {pdb_path}: {e}")

    # summary
    if results:
        df = pd.DataFrame(results)
        metric_cols = ["orientation_angle", "lateral_dist", "vertical_dist", "peptide_fraction"]

        print(f"\nResults: {passed}/{len(results)} structures passed orientation filters")
        print(f"  Failed angle:             {sum(not r['passes_angle']    for r in results)}")
        print(f"  Failed lateral dist:      {sum(not r['passes_lateral']  for r in results)}")
        print(f"  Failed peptide fraction:  {sum(not r['passes_fraction'] for r in results)}")
        print("\nMetric statistics:")
        print(df[metric_cols].describe(percentiles=[0.25, 0.5, 0.75]).round(3).to_string())
    else:
        print("No structures were processed.")
    log_path = os.path.join(output_dir, "orientation_filter_log.csv")
    pd.DataFrame(results).to_csv(log_path, index=False)
    print(f"Log written to {log_path}")

    return results


# ── Argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter RFdiffusion backbone PDBs by binder orientation relative "
            "to the pMHCI complex. Run after filter.py geometric pre-filters."
        )
    )

    # inspect mode
    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        metavar="PDB_PATH",
        help="Run inspect_chains_detailed() on a single PDB and exit. "
             "Use this to confirm chain letters before filtering."
    )

    # input / output
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        required=False,
        metavar="DIR",
        help="One or more directories containing PDB files to filter."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        metavar="DIR",
        help="Directory to write passing structures and orientation_filter_log.csv."
    )

    # chain letters
    parser.add_argument(
        "--binder_chain",
        type=str,
        default="A",
        help="Chain letter of the designed binder (default: A)."
    )
    parser.add_argument(
        "--peptide_chain",
        type=str,
        default="C",
        help="Chain letter of the peptide (default: C)."
    )
    parser.add_argument(
        "--mhc_chain",
        type=str,
        default="B",
        help="Chain letter of the MHC (default: B)."
    )

    # orientation filter thresholds
    parser.add_argument(
        "--max_angle",
        type=float,
        default=60.0,
        help="Maximum angle in degrees between binder and peptide principal "
             "axes (default: 60). Increase to be more permissive."
    )
    parser.add_argument(
        "--max_lateral_dist",
        type=float,
        default=20.0,
        help="Maximum lateral displacement in Å of binder COM from peptide "
             "COM (default: 20). Increase to be more permissive."
    )
    parser.add_argument(
        "--min_peptide_fraction",
        type=float,
        default=0.3,
        help="Minimum fraction of binder residues that must be closer to the "
             "peptide than to the MHC (default: 0.3). Inspect distribution "
             "before setting a hard cutoff."
    )

    # glob pattern
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdb",
        help="Glob pattern for PDB files (default: *.pdb)."
    )

    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # inspect mode — print chain info and exit
    if args.inspect:
        inspect_chains_detailed(args.inspect)
        exit(0)

    # validate required args for filter mode
    missing = []
    if not args.input_dirs:
        missing.append("--input_dirs")
    if not args.output_dir:
        missing.append("--output_dir")
    if missing:
        raise ValueError(
            f"The following arguments are required for filtering: {', '.join(missing)}\n"
            "Tip: run with --inspect <pdb_path> first to confirm chain letters."
        )

    filter_orientation(
        input_dirs           = args.input_dirs,
        output_dir           = args.output_dir,
        binder_chain         = args.binder_chain,
        peptide_chain        = args.peptide_chain,
        mhc_chain            = args.mhc_chain,
        max_angle            = args.max_angle,
        max_lateral_dist     = args.max_lateral_dist,
        min_peptide_fraction = args.min_peptide_fraction,
        pattern              = args.pattern,
    )