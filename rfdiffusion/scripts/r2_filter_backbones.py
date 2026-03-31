"""
filter_backbones_partial.py
============================
Backbone filter for RFdiffusion PARTIAL DIFFUSION outputs.

Key differences from the original filter_backbones.py:
  - Partial diffusion outputs have TWO chains only:
      Chain A = binder (poly-G, variable length)
      Chain B = MHC + peptide (merged, no separate chain C)
  - Peptide position within chain B is auto-detected as the last
    PEPTIDE_LEN residues (default 9 for KRAS VVGADGVGK).
  - Only two filters are applied:
      1. Clash detection (Cα-Cα between binder and MHC+peptide)
      2. Peptide contact (binder must contact specified peptide positions)
  - All geometric filters (Rg, n_residues, principal axis, CoM, etc.)
    are dropped — they are redundant for partial diffusion outputs which
    are perturbations of already-validated scaffolds.

Usage
-----
# Inspect a single PDB to verify chain layout and peptide range:
python filter_backbones_partial.py \
    --inspect /path/to/orient_247_pT12_0.pdb \
    --peptide_len 9

# Filter all tier 1 outputs:
python r2_filter_backbones.py \
    --input_dirs \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier1/orient_247 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier1/orient_34 \
    --output_dir /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier1_filtered \
    --peptide_len 9 \
    --clash_dist 4.0 \
    --max_clashes 0 \
    --contact_dist 8.0 \
    --required_peptide_positions 6 7
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.core.pose import Pose, append_pose_to_pose

init("-mute all")


# ── Chain inspection ──────────────────────────────────────────────────────────

def inspect_chains(pdb_path, peptide_len=9):
    """
    Print chain layout and inferred peptide range within chain B.
    Run this on a sample PDB before filtering at scale.
    """
    pose = pose_from_pdb(pdb_path)
    print(f"\n{pdb_path}")
    print(f"  Total residues: {pose.total_residue()}")
    for chain_idx in range(1, pose.num_chains() + 1):
        chain_letter = chr(ord('A') + chain_idx - 1)
        start  = pose.chain_begin(chain_idx)
        end    = pose.chain_end(chain_idx)
        n_res  = end - start + 1
        first  = pose.residue(start).name3()
        last   = pose.residue(end).name3()
        print(f"  Chain {chain_letter}: residues {start}–{end} "
              f"({n_res} res) [{first}...{last}]")

    # infer peptide range within chain B
    chain_b_id    = 2  # chain B is always index 2
    chain_b_len   = pose.chain_end(chain_b_id) - pose.chain_begin(chain_b_id) + 1
    pep_start_abs = chain_b_len - peptide_len + 1
    pep_end_abs   = chain_b_len
    print(f"\n  Inferred peptide (last {peptide_len} res of chain B): "
          f"positions {pep_start_abs}–{pep_end_abs} within chain B")

    # show peptide sequence
    chain_b_begin = pose.chain_begin(chain_b_id)
    pep_seq = "".join(
        pose.residue(chain_b_begin + i - 1).name1()
        for i in range(pep_start_abs, pep_end_abs + 1)
    )
    print(f"  Peptide sequence: {pep_seq}")


# ── Chain utilities ───────────────────────────────────────────────────────────

def get_ca_coords(pose, chain_letter):
    """Extract Cα coordinates for a chain given its letter."""
    chain_id = ord(chain_letter) - ord('A') + 1
    chain_pose = pose.split_by_chain(chain_id)
    coords = []
    for i in range(1, chain_pose.total_residue() + 1):
        res = chain_pose.residue(i)
        if res.has("CA"):
            ca = res.xyz("CA")
            coords.append([ca.x, ca.y, ca.z])
    return np.array(coords)


def split_chain_b(pose, peptide_len=9):
    """
    Split chain B (MHC + peptide) into:
      - MHC coords:     first (chain_B_len - peptide_len) residues
      - Peptide coords: last peptide_len residues

    Returns two numpy arrays: (mhc_coords, peptide_coords)
    """
    chain_b_id    = 2
    chain_b_begin = pose.chain_begin(chain_b_id)
    chain_b_end   = pose.chain_end(chain_b_id)
    chain_b_len   = chain_b_end - chain_b_begin + 1

    if chain_b_len <= peptide_len:
        raise ValueError(
            f"Chain B has only {chain_b_len} residues, "
            f"cannot extract {peptide_len}-residue peptide."
        )

    mhc_coords = []
    pep_coords = []

    for abs_idx in range(chain_b_begin, chain_b_end + 1):
        res = pose.residue(abs_idx)
        if not res.has("CA"):
            continue
        ca    = res.xyz("CA")
        coord = [ca.x, ca.y, ca.z]
        pos_in_chain = abs_idx - chain_b_begin  # 0-based
        if pos_in_chain < chain_b_len - peptide_len:
            mhc_coords.append(coord)
        else:
            pep_coords.append(coord)

    return np.array(mhc_coords), np.array(pep_coords)


# ── Filters ───────────────────────────────────────────────────────────────────

def check_clashes(binder_coords, target_coords, clash_dist=4.0, max_clashes=0):
    """
    Count Cα-Cα clashes between binder and target (MHC+peptide combined).
    Returns (n_clashes, passes).
    """
    dists = np.linalg.norm(
        binder_coords[:, None, :] - target_coords[None, :, :], axis=-1
    )
    n_clashes = int((dists < clash_dist).sum())
    return n_clashes, n_clashes <= max_clashes


def check_peptide_contacts(binder_coords, pep_coords,
                           contact_dist=8.0,
                           required_positions=None):
    """
    Check that the binder contacts specified peptide residue positions.

    Args:
        binder_coords:       (N, 3) Cα coords of binder
        pep_coords:          (M, 3) Cα coords of peptide (in order)
        contact_dist:        Cα-Cα cutoff in Å
        required_positions:  list of 1-based peptide positions that must be
                             contacted; if None, requires at least one contact
                             with any peptide residue.

    Returns:
        contacts_per_pos:    dict of {position: n_binder_residues_contacting}
        passes:              True if all required positions are contacted
    """
    if required_positions is None:
        dists     = np.linalg.norm(
            binder_coords[:, None, :] - pep_coords[None, :, :], axis=-1
        )
        n_any = int((dists.min(axis=1) < contact_dist).sum())
        return {"any": n_any}, n_any >= 1

    contacts_per_pos = {}
    for pos in required_positions:
        if pos < 1 or pos > len(pep_coords):
            raise ValueError(
                f"Peptide position {pos} out of range "
                f"(peptide has {len(pep_coords)} residues)"
            )
        pep_coord = pep_coords[pos - 1]                           # 0-based
        dists     = np.linalg.norm(binder_coords - pep_coord, axis=-1)
        contacts_per_pos[pos] = int((dists < contact_dist).sum())

    passes = all(n > 0 for n in contacts_per_pos.values())
    return contacts_per_pos, passes


# ── Main filter loop ──────────────────────────────────────────────────────────

def filter_backbones(
    input_dirs,
    output_dir,
    peptide_len=9,
    clash_dist=4.0,
    max_clashes=0,
    contact_dist=8.0,
    required_peptide_positions=None,
    pattern="*.pdb",
):
    """
    Filter partial diffusion backbone PDBs by:
      1. Clash detection (binder vs MHC+peptide Cα-Cα)
      2. Peptide contact (binder must reach required peptide positions)

    Peptide is auto-detected as the last `peptide_len` residues of chain B.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    passed  = 0

    for input_dir in input_dirs:
        pdb_files = sorted(glob.glob(os.path.join(input_dir, "**", pattern),
                                     recursive=True))
        if not pdb_files:
            pdb_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
        print(f"Found {len(pdb_files)} structures in {input_dir}")

        for pdb_path in pdb_files:
            try:
                pose = pose_from_pdb(pdb_path)

                # extract coords
                binder_coords          = get_ca_coords(pose, "A")
                mhc_coords, pep_coords = split_chain_b(pose, peptide_len)
                target_coords          = np.vstack([mhc_coords, pep_coords])

                # filter 1: clashes
                n_clashes, passes_clashes = check_clashes(
                    binder_coords, target_coords, clash_dist, max_clashes
                )

                # filter 2: peptide contacts
                contacts_per_pos, passes_cont = check_peptide_contacts(
                    binder_coords, pep_coords,
                    contact_dist, required_peptide_positions
                )

                passes_all = passes_clashes and passes_cont

                results.append({
                    "filename_old":          pdb_path,
                    "filename_new":          "NA",
                    "n_clashes":             n_clashes,
                    "contacts_per_position": contacts_per_pos,
                    "passes_clashes":        passes_clashes,
                    "passes_contacts":       passes_cont,
                    "passes_all":            passes_all,
                })

                if passes_all:
                    # preserve original backbone name for traceability
                    base        = os.path.splitext(os.path.basename(pdb_path))[0]
                    filename_new = os.path.join(output_dir, f"{base}_filtered.pdb")
                    pose.dump_pdb(filename_new)
                    passed += 1
                    results[-1]["filename_new"] = filename_new

            except Exception as e:
                print(f"  [WARN] Failed to process {pdb_path}: {e}")
                results.append({
                    "filename_old":          pdb_path,
                    "filename_new":          "ERROR",
                    "n_clashes":             None,
                    "contacts_per_position": None,
                    "passes_clashes":        False,
                    "passes_contacts":       False,
                    "passes_all":            False,
                })

    # summary
    total = len(results)
    print(f"\n{'='*50}")
    print(f"  Results: {passed}/{total} structures passed all filters")
    if total > 0:
        print(f"  Failed clashes:  {sum(not r['passes_clashes']  for r in results)}")
        print(f"  Failed contacts: {sum(not r['passes_contacts'] for r in results)}")
    print(f"{'='*50}")

    log_path = os.path.join(output_dir, "filter_log.csv")
    pd.DataFrame(results).to_csv(log_path, index=False)
    print(f"Log: {log_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter RFdiffusion PARTIAL DIFFUSION backbone PDBs.\n"
            "Applies clash detection and peptide contact filters only.\n"
            "Peptide is auto-detected as the last --peptide_len residues of chain B."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        metavar="PDB_PATH",
        help="Inspect a single PDB and print chain/peptide info, then exit."
    )
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        metavar="DIR",
        help="Directories containing partial diffusion output PDBs (searched recursively)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        metavar="DIR",
        help="Directory to write passing structures and filter_log.csv."
    )
    parser.add_argument(
        "--peptide_len",
        type=int,
        default=9,
        help="Number of residues in the target peptide (default: 9 for KRAS VVGADGVGK)."
    )
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
        help="Maximum tolerated clashes between binder and target (default: 0)."
    )
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
        help=(
            "1-based peptide positions that must be contacted by the binder. "
            "e.g. --required_peptide_positions 6 7  (for KRAS hotspots). "
            "If not specified, requires at least one contact with any peptide residue."
        )
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdb",
        help="Glob pattern for PDB files (default: *.pdb)."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.inspect:
        inspect_chains(args.inspect, peptide_len=args.peptide_len)
        raise SystemExit(0)

    # validate required args
    missing = []
    if not args.input_dirs:
        missing.append("--input_dirs")
    if not args.output_dir:
        missing.append("--output_dir")
    if missing:
        raise ValueError(
            f"Required arguments missing: {', '.join(missing)}\n"
            "Tip: run with --inspect <pdb_path> first to verify chain layout."
        )

    filter_backbones(
        input_dirs                = args.input_dirs,
        output_dir                = args.output_dir,
        peptide_len               = args.peptide_len,
        clash_dist                = args.clash_dist,
        max_clashes               = args.max_clashes,
        contact_dist              = args.contact_dist,
        required_peptide_positions= args.required_peptide_positions,
        pattern                   = args.pattern,
    )