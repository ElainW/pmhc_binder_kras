#!/usr/bin/env python3
"""
dssp_analysis.py

Runs PyRosetta DSSP on each downloaded experimental PDB, extracts per-residue
secondary structure assignments, and computes per-structure summary statistics.

Secondary structure codes (PyRosetta DSSP):
    H = alpha-helix
    E = beta-strand
    L = loop/coil/other

Usage:
    python dssp_analysis.py \
        --pdb_dir ./pdbs \
        --manifest ./synthetic_manifest.tsv \
        --out_per_residue ./dssp_per_residue.tsv \
        --out_summary ./dssp_summary.tsv
"""

import argparse
import sys
import os
from pathlib import Path

import pyrosetta
from pyrosetta.rosetta.protocols.moves import DsspMover
from pyrosetta.rosetta.core.pose import Pose


def init_pyrosetta():
    pyrosetta.init(
        options="-ignore_unrecognized_res true -ignore_zero_occupancy false "
                "-load_PDB_components false -no_fconfig",
        silent=True,
        set_logging_handler=False
    )


def dssp_from_pose(pose: Pose) -> str:
    """Return DSSP string (H/E/L per residue) for the full pose."""
    dssp_mover = DsspMover()
    dssp_mover.apply(pose)
    ss = pose.secstruct()
    # Remap DSSP codes to H/E/L (PyRosetta uses H/E/L already, but map anything
    # that is not H or E to L for clarity)
    mapped = ""
    for c in ss:
        if c == "H":
            mapped += "H"
        elif c == "E":
            mapped += "E"
        else:
            mapped += "L"
    return mapped


def ss_composition(ss_string: str) -> dict:
    """Compute fractional composition of H, E, L from DSSP string."""
    n = len(ss_string)
    if n == 0:
        return {"n_residues": 0, "n_helix": 0, "n_strand": 0, "n_loop": 0,
                "frac_helix": 0.0, "frac_strand": 0.0, "frac_loop": 0.0}
    n_H = ss_string.count("H")
    n_E = ss_string.count("E")
    n_L = ss_string.count("L")
    return {
        "n_residues": n,
        "n_helix": n_H,
        "n_strand": n_E,
        "n_loop": n_L,
        "frac_helix": n_H / n,
        "frac_strand": n_E / n,
        "frac_loop": n_L / n,
    }


def count_ss_segments(ss_string: str) -> dict:
    """Count number of distinct contiguous secondary structure segments."""
    if not ss_string:
        return {"n_helix_segments": 0, "n_strand_segments": 0, "n_loop_segments": 0}

    counts = {"H": 0, "E": 0, "L": 0}
    prev = None
    for c in ss_string:
        if c != prev:
            counts[c] += 1
            prev = c
    return {
        "n_helix_segments": counts["H"],
        "n_strand_segments": counts["E"],
        "n_loop_segments": counts["L"],
    }


def load_chain(pdb_path: Path, chain_id: str) -> Pose | None:
    """Load a single chain from a PDB file into a PyRosetta Pose."""
    try:
        full_pose = pyrosetta.pose_from_file(str(pdb_path))
    except Exception as e:
        print(f"  WARNING: Could not load {pdb_path.name}: {e}", file=sys.stderr)
        return None

    # Extract chain if multi-chain
    chain_ids_in_pose = []
    for i in range(1, full_pose.num_chains() + 1):
        begin = full_pose.chain_begin(i)
        chain_letter = full_pose.pdb_info().chain(begin)
        chain_ids_in_pose.append(chain_letter)

    if chain_id not in chain_ids_in_pose:
        # Fall back to chain A or first chain
        chain_id = chain_ids_in_pose[0] if chain_ids_in_pose else "A"

    try:
        chain_idx = chain_ids_in_pose.index(chain_id) + 1
        chain_pose = pyrosetta.rosetta.core.pose.Pose()
        pyrosetta.rosetta.core.pose.append_subpose_to_pose(
            chain_pose,
            full_pose,
            full_pose.chain_begin(chain_idx),
            full_pose.chain_end(chain_idx),
            True
        )
        return chain_pose
    except Exception as e:
        print(f"  WARNING: Could not extract chain {chain_id} from {pdb_path.name}: {e}",
              file=sys.stderr)
        return full_pose  # return full pose as fallback


def main():
    parser = argparse.ArgumentParser(description="DSSP analysis of synthetic proteins")
    parser.add_argument("--pdb_dir", default="./pdbs", help="Directory containing PDB files")
    parser.add_argument("--manifest", default="./synthetic_manifest.tsv",
                        help="TSV with pdb_id, chain_id, length")
    parser.add_argument("--out_per_residue", default="./dssp_per_residue.tsv",
                        help="Output: per-residue DSSP assignments")
    parser.add_argument("--out_summary", default="./dssp_summary.tsv",
                        help="Output: per-structure summary statistics")
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)

    print("Initializing PyRosetta...")
    init_pyrosetta()

    # Read manifest
    manifest = []
    with open(args.manifest) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pdb_id = parts[0]
                chain_id = parts[1]
                length = int(parts[2]) if len(parts) > 2 else 0
                manifest.append((pdb_id, chain_id, length))

    print(f"Processing {len(manifest)} structures...")

    summary_rows = []
    per_residue_rows = []

    for i, (pdb_id, chain_id, length) in enumerate(manifest):
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            print(f"  SKIP: {pdb_id}.pdb not found", file=sys.stderr)
            continue

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(manifest)}] Processing {pdb_id}...", flush=True)

        pose = load_chain(pdb_path, chain_id)
        if pose is None or pose.total_residue() == 0:
            continue

        ss_string = dssp_from_pose(pose)
        comp = ss_composition(ss_string)
        segs = count_ss_segments(ss_string)

        # Per-residue rows
        for resi, ss_code in enumerate(ss_string, start=1):
            per_residue_rows.append({
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "residue_index": resi,
                "ss_code": ss_code,
            })

        # Summary row
        row = {
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "ss_string": ss_string,
            **comp,
            **segs,
        }
        summary_rows.append(row)

    print(f"Processed {len(summary_rows)} structures successfully.")

    # Write per-residue TSV
    print(f"Writing per-residue output to {args.out_per_residue}...")
    with open(args.out_per_residue, "w") as f:
        f.write("pdb_id\tchain_id\tresidue_index\tss_code\n")
        for row in per_residue_rows:
            f.write(f"{row['pdb_id']}\t{row['chain_id']}\t{row['residue_index']}\t{row['ss_code']}\n")

    # Write summary TSV
    print(f"Writing summary to {args.out_summary}...")
    summary_cols = [
        "pdb_id", "chain_id", "n_residues",
        "n_helix", "n_strand", "n_loop",
        "frac_helix", "frac_strand", "frac_loop",
        "n_helix_segments", "n_strand_segments", "n_loop_segments",
        "ss_string"
    ]
    with open(args.out_summary, "w") as f:
        f.write("\t".join(summary_cols) + "\n")
        for row in summary_rows:
            vals = [str(row.get(c, "")) for c in summary_cols]
            f.write("\t".join(vals) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
