import os
import glob
import pandas as pd
import numpy as np
import argparse
from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.core.pose import Pose, append_pose_to_pose

# Initialize PyRosetta (silent mode to reduce output)
init("-mute all")

def inspect_chains_detailed(pdb_path):
    """
    Print chain letters, residue counts, and residue index ranges —
    useful for identifying the peptide range within chain B.
    """
    pose = pose_from_pdb(pdb_path)
    print(f"\n{pdb_path}")
    print(f"  Total residues: {pose.total_residue()}")
    for chain_idx in range(1, pose.num_chains() + 1):
        chain_letter = chr(ord('A') + chain_idx - 1)
        start = pose.chain_begin(chain_idx)
        end   = pose.chain_end(chain_idx)
        n_res = end - start + 1
        # show first and last residue names to help identify peptide vs MHC
        first_res = pose.residue(start).name3()
        last_res  = pose.residue(end).name3()
        print(f"  Chain {chain_letter}: residues {start}-{end} "
              f"({n_res} res) [{first_res}...{last_res}]")

# relabel peptide chain from appended to mhc (chain B) to chain C
def relabel_peptide_chain(pose, peptide_start, peptide_end,
                          target_chain="B", new_chain="C"):
    """
    Extracts peptide residues into a separate chain using pose slicing.
    Uses conformation-safe splitting via pdb_info + renaming after
    physically separating the peptide into its own subpose.
    """
    chain_id    = ord(target_chain) - ord('A') + 1
    chain_start = pose.chain_begin(chain_id)
    chain_end   = pose.chain_end(chain_id)

    abs_pep_start = chain_start + peptide_start - 1
    abs_pep_end   = chain_start + peptide_end   - 1

    if abs_pep_start < chain_start or abs_pep_end > chain_end:
        raise ValueError(
            f"Peptide range {peptide_start}-{peptide_end} is out of bounds "
            f"for chain {target_chain} "
            f"(chain has {chain_end - chain_start + 1} residues)"
        )

    # clone pose and extract peptide as its own subpose
    mhc_pose     = pose.clone()
    peptide_pose = Pose()

    # extract peptide residues into peptide_pose
    for i, res_idx in enumerate(range(abs_pep_start, abs_pep_end + 1)):
        if i == 0:
            peptide_pose.append_residue_by_jump(
                mhc_pose.residue(res_idx).clone(), 1
            )
        else:
            peptide_pose.append_residue_by_bond(
                mhc_pose.residue(res_idx).clone()
            )

    # delete peptide residues from mhc_pose (delete in reverse to preserve indices)
    for res_idx in range(abs_pep_end, abs_pep_start - 1, -1):
        mhc_pose.delete_residue_range_slow(res_idx, res_idx)

    # rejoin: mhc_pose (now without peptide) + peptide_pose as new chain
    append_pose_to_pose(mhc_pose, peptide_pose, new_chain=True)

    # now relabel chain letters to match expected A, B, C layout
    pdb_info = mhc_pose.pdb_info()
    for res_idx in range(1, mhc_pose.total_residue() + 1):
        chain_num    = mhc_pose.chain(res_idx)
        chain_letter = chr(ord('A') + chain_num - 1)
        pdb_info.chain(res_idx, chain_letter)
    mhc_pose.pdb_info(pdb_info)

    return mhc_pose


def verify_relabeling(pdb_path, peptide_start, peptide_end):
    """Quick sanity check — print chains before and after relabeling."""
    pose = pose_from_pdb(pdb_path)

    print("BEFORE:")
    inspect_chains_detailed(pdb_path)

    relabeled = relabel_peptide_chain(pose, peptide_start, peptide_end)

    print("\nAFTER:")
    for chain_idx in range(1, relabeled.num_chains() + 1):
        chain_letter = chr(ord('A') + chain_idx - 1)
        start  = relabeled.chain_begin(chain_idx)
        end    = relabeled.chain_end(chain_idx)
        print(f"  Chain {chain_letter}: {end - start + 1} residues")


# shared utils
def get_chain_pose(pose, chain="A"):
    """Extract a single chain from a pose."""
    chain_id = ord(chain) - ord('A') + 1  # convert letter to PyRosetta chain index
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


# individual filters
def calc_rg_binder_only(pose, binder_chain="A"):
    binder_pose = get_chain_pose(pose, binder_chain)
    coords = get_ca_coords_by_letter(pose, binder_chain)
    centroid = coords.mean(axis=0)
    rg = np.sqrt(((coords - centroid) ** 2).sum(axis=1).mean())
    return rg


def check_n_residues(pose, binder_chain="A", expected_min=60, expected_max=100):
    """Check binder chain has a reasonable number of residues."""
    binder_pose = get_chain_pose(pose, binder_chain)
    n = binder_pose.total_residue()
    passes = expected_min <= n <= expected_max
    return n, passes


def check_clashes(pose, binder_chain="A", clash_dist=4, max_clashes=0):
    """
    Count Cα-Cα clashes between binder and all target chains.
    A clash is defined as Cα-Cα distance below clash_dist (Å).
    """
    binder_coords = get_ca_coords_by_letter(pose, binder_chain)

    # collect all non-binder chain coords
    target_coords = []
    for chain_idx in range(1, pose.num_chains() + 1):
        chain_letter = chr(ord('A') + chain_idx - 1)
        if chain_letter == binder_chain:
            continue
        target_coords.append(get_ca_coords_by_letter(pose, chain_letter))
    target_coords = np.vstack(target_coords)

    # pairwise distances: (n_binder, n_target)
    dists = np.linalg.norm(
        binder_coords[:, None, :] - target_coords[None, :, :], axis=-1
    )
    n_clashes = int((dists < clash_dist).sum())
    passes = n_clashes <= max_clashes
    return n_clashes, passes


def check_peptide_contacts(pose, binder_chain="A", peptide_chain="C",
                           contact_dist=8.0, required_peptide_positions=None):
    """
    Check that the binder has at least one Cα within contact_dist (Å) of
    each specified peptide residue position.

    Args:
        pose:                       PyRosetta pose
        binder_chain:               Chain letter of the binder
        peptide_chain:              Chain letter of the peptide
        contact_dist:               Cα-Cα distance cutoff in Å
        required_peptide_positions: List of 1-based residue positions within
                                    the peptide chain that must be contacted.
                                    e.g. [4, 5, 7] for a MART-1 binder
                                    targeting L4, G5, I7.
                                    If None, falls back to requiring at least
                                    one binder residue contacts any peptide residue.

    Returns:
        contacts_per_position: dict mapping each required position to the
                               number of binder residues contacting it
        passes:                True if all required positions are contacted
    """
    binder_coords  = get_ca_coords_by_letter(pose, binder_chain)
    peptide_pose   = get_chain_pose(pose, peptide_chain)

    # fallback: any contact with any peptide residue
    if required_peptide_positions is None:
        peptide_coords = get_ca_coords_by_letter(pose, peptide_chain)
        dists = np.linalg.norm(
            binder_coords[:, None, :] - peptide_coords[None, :, :], axis=-1
        )
        n_contacts = int((dists.min(axis=1) < contact_dist).sum())
        passes = n_contacts >= 1
        return {"any": n_contacts}, passes

    # per-position contact check
    contacts_per_position = {}
    for pos in required_peptide_positions:
        if pos < 1 or pos > peptide_pose.total_residue():
            raise ValueError(
                f"Peptide position {pos} out of range "
                f"(peptide has {peptide_pose.total_residue()} residues)"
            )
        residue = peptide_pose.residue(pos)
        if not residue.has("CA"):
            raise ValueError(f"Peptide residue at position {pos} has no Cα atom")

        ca = residue.xyz("CA")
        pep_coord = np.array([ca.x, ca.y, ca.z])

        # distances from all binder Cαs to this peptide residue's Cα
        dists = np.linalg.norm(binder_coords - pep_coord, axis=-1)
        n_contacts = int((dists < contact_dist).sum())
        contacts_per_position[pos] = n_contacts

    passes = all(n > 0 for n in contacts_per_position.values())
    return contacts_per_position, passes


# combined filter
def filter_backbones(
    input_dirs,
    output_dir,
    binder_chain="A",
    peptide_chain="C",      # confirm with inspect_chains() before running
    peptide_start=None,        # 1-based start position within chain B
    peptide_end=None,          # 1-based end position within chain B
    mhc_peptide_chain="B",     # chain containing MHC + peptide from RFdiffusion
    rg_min=None,
    rg_max=None,
    expected_min=60,
    expected_max=100,
    clash_dist=4.0,
    max_clashes=0,
    contact_dist=8.0,
    required_peptide_positions=None,   # e.g. [4, 5, 7] for outward-facing residues
    pattern="*.pdb"
):
    """
    Filter RFdiffusion backbone PDBs by:
      1. Radius of gyration (binder Cα only)
      2. Number of residues (binder only)
      3. Cα clashes between binder and target
      4. Peptide contact count

    Args:
        input_dirs:   List of directories containing RFdiffusion output PDBs
        output_dir:   Directory to write passing structures and log
        binder_chain: Chain letter of the designed binder (default "A")
        peptide_chain:Chain letter of the target peptide — confirm with
                      inspect_chains() before running at scale
        rg_min:       Minimum binder Rg in Å
        rg_max:       Maximum binder Rg in Å
        expected_min: Minimum number of binder residues
        expected_max: Maximum number of binder residues
        clash_dist:   Cα-Cα distance (Å) below which a clash is counted
        max_clashes:  Maximum tolerated clashes
        contact_dist: Cα-Cα distance (Å) cutoff for peptide contact
        min_contacts: Minimum number of binder residues contacting peptide
        pattern:      Glob pattern for PDB files
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    passed = 0

    # validate peptide range is specified
    if peptide_start is None or peptide_end is None:
        raise ValueError(
            "peptide_start and peptide_end must be specified. "
            "Run inspect_chains_detailed() on a sample PDB first."
        )

    for input_dir in input_dirs:
        pdb_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
        print(f"Found {len(pdb_files)} structures in {input_dir}")

        for pdb_path in pdb_files:
            try:
                pose = pose_from_pdb(pdb_path)

                # relabel peptide residues from mhc_peptide_chain to peptide_chain
                pose = relabel_peptide_chain(
                                                pose,
                                                peptide_start = peptide_start,
                                                peptide_end   = peptide_end,
                                                target_chain  = mhc_peptide_chain,
                                                new_chain     = peptide_chain,
                                            )

                rg                            = calc_rg_binder_only(pose, binder_chain)
                n_res, passes_nres            = check_n_residues(pose, binder_chain, expected_min, expected_max)
                n_clashes, passes_clashes     = check_clashes(pose, binder_chain, clash_dist, max_clashes)
                contacts_per_pos, passes_cont = check_peptide_contacts(
                    pose, binder_chain, peptide_chain,
                    contact_dist, required_peptide_positions
                )

                passes_rg  = (rg_min is None or rg >= rg_min) and (rg_max is None or rg <= rg_max)
                passes_all = passes_rg and passes_nres and passes_clashes and passes_cont

                results.append({
                    "filename_old":          pdb_path,
                    "filename_new":          'NA',
                    "rg":                    round(rg, 3),
                    "n_residues":            n_res,
                    "n_clashes":             n_clashes,
                    "contacts_per_position": contacts_per_pos,   # e.g. {4: 3, 5: 1, 7: 2}
                    "passes_rg":             passes_rg,
                    "passes_nres":           passes_nres,
                    "passes_clashes":        passes_clashes,
                    "passes_contacts":       passes_cont,
                    "passes_all":            passes_all,
                })

                if passes_all:
                    filename_new = os.path.join(output_dir, f"kras_{passed}.pdb")
                    pose.dump_pdb(filename_new)
                    passed += 1
                    # update the new filename
                    results[-1]['filename_new'] = filename_new

            except Exception as e:
                print(f"  Warning: failed to process {pdb_path}: {e}")

    # summary
    if results:
        rg_values = [r["rg"] for r in results]
        print(f"\nResults: {passed}/{len(results)} structures passed all filters")
        print(f"  Rg range:          {min(rg_values):.2f} – {max(rg_values):.2f} Å")
        print(f"  Failed rg:         {sum(not r['passes_rg']       for r in results)}")
        print(f"  Failed n_residues: {sum(not r['passes_nres']      for r in results)}")
        print(f"  Failed clashes:    {sum(not r['passes_clashes']   for r in results)}")
        print(f"  Failed contacts:   {sum(not r['passes_contacts']  for r in results)}")
    else:
        print("No structures were processed.")

    # write log using pandas since it's already imported
    log_path = os.path.join(output_dir, "filter_log.csv")
    pd.DataFrame(results).to_csv(log_path, index=False)
    print(f"Log written to {log_path}")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter RFdiffusion backbone PDBs by Rg, clashes, and peptide contacts."
    )

    # mode: inspect or filter
    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        metavar="PDB_PATH",
        help="Run inspect_chains_detailed() on a single PDB and exit. "
             "Use this before filtering to confirm chain letters and peptide range."
    )

    # input/output
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        required=False,
        metavar="DIR",
        help="One or more directories containing RFdiffusion output PDBs."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        metavar="DIR",
        help="Directory to write passing structures and filter_log.csv."
    )

    # chain letters
    parser.add_argument(
        "--binder_chain",
        type=str,
        default="A",
        help="Chain letter of the designed binder (default: A)."
    )
    parser.add_argument(
        "--mhc_peptide_chain",
        type=str,
        default="B",
        help="Chain letter containing MHC + peptide in RFdiffusion output (default: B)."
    )
    parser.add_argument(
        "--peptide_chain",
        type=str,
        default="C",
        help="Chain letter to assign to peptide after relabeling (default: C)."
    )

    # peptide range within mhc_peptide_chain
    parser.add_argument(
        "--peptide_start",
        type=int,
        required=False,
        help="1-based start position of peptide within mhc_peptide_chain."
    )
    parser.add_argument(
        "--peptide_end",
        type=int,
        required=False,
        help="1-based end position of peptide within mhc_peptide_chain."
    )

    # rg filter
    parser.add_argument(
        "--rg_min",
        type=float,
        default=None,
        help="Minimum binder Rg in Å (default: no lower bound)."
    )
    parser.add_argument(
        "--rg_max",
        type=float,
        default=None,
        help="Maximum binder Rg in Å (default: no upper bound)."
    )

    # residue count filter
    parser.add_argument(
        "--expected_min",
        type=int,
        default=50,
        help="Minimum number of binder residues (default: 60)."
    )
    parser.add_argument(
        "--expected_max",
        type=int,
        default=100,
        help="Maximum number of binder residues (default: 100)."
    )

    # clash filter
    parser.add_argument(
        "--clash_dist",
        type=float,
        default=4.0,
        help="Cα-Cα distance in Å below which a clash is counted (default: 4.0)."
    )
    parser.add_argument(
        "--max_clashes",
        type=int,
        default=0,
        help="Maximum number of tolerated clashes (default: 0)."
    )

    # contact filter
    parser.add_argument(
        "--contact_dist",
        type=float,
        default=8.0,
        help="Cα-Cα distance cutoff in Å for peptide contact (default: 8.0)."
    )
    parser.add_argument(
        "--required_peptide_positions",
        type=int,
        nargs="+",
        default=None,
        metavar="POS",
        help="1-based peptide positions that must be contacted by the binder. "
             "e.g. --required_peptide_positions 4 5 7"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # inspect mode — run and exit without filtering
    if args.inspect:
        inspect_chains_detailed(args.inspect)
        verify_relabeling(args.inspect, peptide_start=args.peptide_start, peptide_end=args.peptide_end)
        exit(0)

    # validate required args for filter mode
    missing = []
    if not args.input_dirs:
        missing.append("--input_dirs")
    if not args.output_dir:
        missing.append("--output_dir")
    if args.peptide_start is None:
        missing.append("--peptide_start")
    if args.peptide_end is None:
        missing.append("--peptide_end")
    if missing:
        raise ValueError(
            f"The following arguments are required for filtering: {', '.join(missing)}\n"
            "Tip: run with --inspect <pdb_path> first to confirm chain letters and peptide range."
        )

    filter_backbones(
        input_dirs                = args.input_dirs,
        output_dir                = args.output_dir,
        binder_chain              = args.binder_chain,
        peptide_chain             = args.peptide_chain,
        peptide_start             = args.peptide_start,
        peptide_end               = args.peptide_end,
        mhc_peptide_chain         = args.mhc_peptide_chain,
        rg_min                    = args.rg_min,
        rg_max                    = args.rg_max,
        expected_min              = args.expected_min,
        expected_max              = args.expected_max,
        clash_dist                = args.clash_dist,
        max_clashes               = args.max_clashes,
        contact_dist              = args.contact_dist,
        required_peptide_positions= args.required_peptide_positions,
    )
