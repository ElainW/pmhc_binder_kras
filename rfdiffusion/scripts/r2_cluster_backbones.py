"""
cluster_backbones_partial.py
=============================
Cross-backbone clustering for partial diffusion filtered outputs.

Strategy:
  1. Parse the parent backbone ID from each filename
     (e.g. orient_247 from orient_247_pT12_3_filtered.pdb)
  2. Pick one representative per backbone (first trajectory by filename sort)
  3. Cluster those representatives by binder Cα RMSD using hierarchical
     clustering (average linkage) at a given RMSD cutoff
  4. For each cluster, keep the medoid — backbone with lowest mean RMSD
     to all others in the cluster
  5. Copy all filtered structures from surviving backbones to output_dir

Usage
-----
# Dry run first to see clustering without copying files:
python r2_cluster_backbones.py \
    --input_dir /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/filtered \
    --output_dir /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/clustered \
    --rmsd_cutoff 2.0 \
    --dry_run

# Full run:
python r2_cluster_backbones.py \
    --input_dir /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/filtered \
    --output_dir /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/clustered \
    --rmsd_cutoff 2.0
"""

import os
import re
import shutil
import argparse
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio import PDB
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster


# ── PDB parsing ───────────────────────────────────────────────────────────────

def get_binder_ca_coords(pdb_path, binder_chain="A"):
    """Extract Cα coordinates for the binder chain."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    coords = []
    for model in structure:
        for chain in model:
            if chain.id == binder_chain:
                for residue in chain:
                    if "CA" in residue:
                        coords.append(residue["CA"].get_vector().get_array())
    if not coords:
        raise ValueError(f"No Cα atoms found in chain {binder_chain} of {pdb_path}")
    return np.array(coords)


def parse_backbone_id(filename):
    """
    Extract parent backbone ID from a partial diffusion output filename.
    e.g. orient_247_pT12_3_filtered.pdb → orient_247
    """
    basename = os.path.basename(filename)
    match = re.match(r"(orient_\d+)", basename)
    if not match:
        raise ValueError(f"Cannot parse backbone ID from: {basename}")
    return match.group(1)


# ── RMSD (Kabsch) ─────────────────────────────────────────────────────────────

def calc_rmsd(coords_a, coords_b):
    """
    Superpose coords_b onto coords_a (Kabsch algorithm) and return Cα RMSD.
    Returns np.inf if lengths differ.
    """
    if len(coords_a) != len(coords_b):
        return np.inf

    ca = coords_a - coords_a.mean(axis=0)
    cb = coords_b - coords_b.mean(axis=0)

    H          = cb.T @ ca
    U, S, Vt   = np.linalg.svd(H)
    d          = np.linalg.det(Vt.T @ U.T)
    R          = Vt.T @ np.diag([1, 1, d]) @ U.T
    cb_rot     = cb @ R.T

    return float(np.sqrt(((ca - cb_rot) ** 2).sum(axis=1).mean()))


# ── Backbone grouping ─────────────────────────────────────────────────────────

def group_by_backbone(pdb_files):
    """
    Group PDB paths by parent backbone ID.
    Returns:
        backbone_to_rep: {backbone_id: representative_pdb_path}  (first by sort)
        backbone_to_all: {backbone_id: [all pdb paths]}
    """
    backbone_to_all = defaultdict(list)
    for pdb in sorted(pdb_files):
        bb_id = parse_backbone_id(pdb)
        backbone_to_all[bb_id].append(pdb)

    backbone_to_rep = {
        bb_id: sorted(paths)[0]
        for bb_id, paths in backbone_to_all.items()
    }
    return backbone_to_rep, dict(backbone_to_all)


# ── RMSD matrix ───────────────────────────────────────────────────────────────

def build_rmsd_matrix(backbone_to_rep, binder_chain="A"):
    """
    Compute pairwise Cα RMSD between all backbone representatives.
    Backbones with different binder lengths get RMSD=inf (never merged).
    Returns (rmsd_matrix, backbone_ids, coords_list).
    """
    backbone_ids = sorted(backbone_to_rep.keys())
    n            = len(backbone_ids)
    coords_list  = []

    print(f"Loading {n} backbone representatives...")
    for bb_id in backbone_ids:
        try:
            coords = get_binder_ca_coords(backbone_to_rep[bb_id], binder_chain)
        except Exception as e:
            print(f"  [WARN] {bb_id}: {e}")
            coords = None
        coords_list.append(coords)

    rmsd_matrix = np.full((n, n), np.inf)
    np.fill_diagonal(rmsd_matrix, 0.0)

    for i in range(n):
        for j in range(i + 1, n):
            if coords_list[i] is None or coords_list[j] is None:
                continue
            rmsd = calc_rmsd(coords_list[i], coords_list[j])
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

    return rmsd_matrix, backbone_ids, coords_list


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_and_pick_medoids(rmsd_matrix, backbone_ids, rmsd_cutoff=2.0):
    """
    Average-linkage hierarchical clustering on the RMSD matrix.
    Picks the medoid (lowest mean intra-cluster RMSD) per cluster.
    Returns (cluster_labels, medoids dict {cluster_id: backbone_id}).
    """
    finite_matrix = np.where(np.isinf(rmsd_matrix), 1e6, rmsd_matrix)
    condensed     = squareform(finite_matrix, checks=False)
    Z             = linkage(condensed, method="average")
    labels        = fcluster(Z, t=rmsd_cutoff, criterion="distance")

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    medoids = {}
    for cluster_id, members in clusters.items():
        if len(members) == 1:
            medoids[cluster_id] = backbone_ids[members[0]]
        else:
            sub        = rmsd_matrix[np.ix_(members, members)]
            sub_finite = np.where(np.isinf(sub), 1e6, sub)
            best_local = int(sub_finite.mean(axis=1).argmin())
            medoids[cluster_id] = backbone_ids[members[best_local]]

    return labels, medoids


# ── Main ──────────────────────────────────────────────────────────────────────

def run(input_dirs, output_dir, rmsd_cutoff=2.0, binder_chain="A",
        pattern="*.pdb", dry_run=False):

    pdb_files = []
    for input_dir in input_dirs:
        pdb_files.extend(sorted(glob.glob(os.path.join(input_dir, pattern))))
    if not pdb_files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {input_dir}")
    print(f"Found {len(pdb_files)} filtered structures")

    # step 1: group by backbone
    backbone_to_rep, backbone_to_all = group_by_backbone(pdb_files)
    n_backbones = len(backbone_to_rep)
    print(f"Unique parent backbones: {n_backbones}")
    for bb_id, paths in sorted(backbone_to_all.items()):
        print(f"  {bb_id}: {len(paths)} structures")

    # step 2: pairwise RMSD between representatives
    rmsd_matrix, backbone_ids, coords_list = build_rmsd_matrix(
        backbone_to_rep, binder_chain
    )

    # print RMSD summary
    finite_vals = rmsd_matrix[~np.isinf(rmsd_matrix) & (rmsd_matrix > 0)]
    if len(finite_vals):
        print(f"\nPairwise RMSD (between-backbone): "
              f"min={finite_vals.min():.2f} "
              f"mean={finite_vals.mean():.2f} "
              f"max={finite_vals.max():.2f} Å")

    # step 3: cluster
    cluster_labels, medoids = cluster_and_pick_medoids(
        rmsd_matrix, backbone_ids, rmsd_cutoff
    )
    n_clusters          = len(medoids)
    surviving_backbones = list(medoids.values())

    print(f"\nClustering at {rmsd_cutoff} Å → "
          f"{n_backbones} backbones → {n_clusters} clusters")
    print(f"\nClusters:")
    for cluster_id, medoid in sorted(medoids.items()):
        members = [backbone_ids[i]
                   for i, l in enumerate(cluster_labels) if l == cluster_id]
        merged  = [m for m in members if m != medoid]
        if merged:
            print(f"  [{cluster_id}] {medoid} (medoid) ← merged: {', '.join(merged)}")
        else:
            print(f"  [{cluster_id}] {medoid}")

    # step 4: count surviving structures
    total_structures = sum(
        len(backbone_to_all[bb]) for bb in surviving_backbones
    )
    print(f"\nStructures from surviving backbones: {total_structures} "
          f"(from {n_clusters} backbones)")

    # step 5: copy structures
    if dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        os.makedirs(output_dir, exist_ok=True)
        copied = 0
        for bb_id in surviving_backbones:
            for pdb_path in backbone_to_all[bb_id]:
                dst = os.path.join(output_dir, os.path.basename(pdb_path))
                shutil.copy2(pdb_path, dst)
                copied += 1
        print(f"\nCopied {copied} structures → {output_dir}")

    # step 6: CSV log
    rows = []
    for idx, bb_id in enumerate(backbone_ids):
        cid = cluster_labels[idx]
        rows.append({
            "backbone_id":        bb_id,
            "representative_pdb": backbone_to_rep[bb_id],
            "binder_len":         len(coords_list[idx]) if coords_list[idx] is not None else None,
            "cluster_id":         cid,
            "medoid":             medoids[cid],
            "is_medoid":          bb_id == medoids[cid],
            "n_structures":       len(backbone_to_all[bb_id]),
        })

    df = pd.DataFrame(rows).sort_values(
        ["cluster_id", "is_medoid"], ascending=[True, False]
    )

    if not dry_run:
        log_path = os.path.join(output_dir, "r2_clustering_log.csv")
        df.to_csv(log_path, index=False)
        print(f"Log: {log_path}")

    print(f"\nFinal: {n_backbones} backbones → {n_clusters} clusters "
          f"→ {total_structures} structures ready for ProteinMPNN")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-backbone clustering for partial diffusion outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dirs", type=str, nargs="+", required=False, metavar="DIR",
        help="One or more directories containing RFdiffusion output PDBs."
    )
    parser.add_argument("--output_dir", type=str, required=False, metavar="DIR",
        help="Directory to write passing structures and filter_log.csv."
    )
    parser.add_argument("--rmsd_cutoff",  type=float, default=2.0,
                        help="Cα RMSD cutoff in Å (default: 2.0).")
    parser.add_argument("--binder_chain", type=str,   default="A",
                        help="Binder chain letter (default: A).")
    parser.add_argument("--pattern",      type=str,   default="*_filtered.pdb",
                        help="Glob pattern for input files.")
    parser.add_argument("--dry_run",      action="store_true",
                        help="Show clustering results without copying files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_dirs    = args.input_dirs,
        output_dir   = args.output_dir,
        rmsd_cutoff  = args.rmsd_cutoff,
        binder_chain = args.binder_chain,
        pattern      = args.pattern,
        dry_run      = args.dry_run,
    )
