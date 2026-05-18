"""
partial_diffusion_backbone_RMSD.py
=============================
Cross-backbone clustering for partial diffusion filtered outputs.

Strategy:
  1. Parse the parent backbone ID from each filename
     (e.g. orient_247_pT20__31 from orient_247_pT20__31_pT12__31.pdb)
  2. Pick one representative per backbone (first trajectory by filename sort)
  3. Cluster those representatives by binder Cα RMSD using hierarchical
     clustering (average linkage) at a given RMSD cutoff
  4. For each cluster, keep the medoid — backbone with lowest mean RMSD
     to all others in the cluster
  5. Copy all filtered structures from surviving backbones to output_dir (not doing this yet)

New features:
  --visualize_orientations
      For each orientation group (e.g. orient_252_pT20__31), compute the
      intra-group pairwise Cα RMSD matrix among all backbone representatives
      and save a heatmap PNG to output_dir/orientation_heatmaps/.

  --superpose_orientations
      For each orientation group, superpose all backbones onto the group
      medoid (lowest mean intra-group RMSD) and write a multi-model PDB
      to output_dir/orientation_superposed/.  Only binder chain A is
      written; the superposition is computed on binder Cα atoms.

Usage
-----
# Dry run with heatmaps + superposed PDBs:
python partial_diffusion_backbone_RMSD.py \
    --input_dirs /path/to/filtered \
    --output_dir /path/to/clustered \
    --rmsd_cutoff 2.0 \
    --visualize_orientations \
    --superpose_orientations \
    --dry_run

# Full run:
python partial_diffusion_backbone_RMSD.py \
    --input_dirs /path/to/filtered \
    --output_dir /path/to/clustered \
    --rmsd_cutoff 2.0 \
    --visualize_orientations \
    --superpose_orientations
"""

import os
import re
import shutil
import argparse
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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


def get_full_chain_atoms(pdb_path, chain_id="A"):
    """
    Return list of (atom_name, resnum, resname, coords) for all atoms
    in a given chain. Used for writing superposed PDBs.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    atoms = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    resnum  = residue.get_id()[1]
                    resname = residue.get_resname()
                    for atom in residue:
                        atoms.append({
                            "name":    atom.get_name(),
                            "resnum":  resnum,
                            "resname": resname,
                            "coords":  atom.get_vector().get_array().copy(),
                        })
    return atoms


def parse_backbone_id(filename):
    """
    Extract parent backbone ID from a partial diffusion output filename.
    Handles both r2 and r3 naming conventions:
      orient_252_pT20__31_pT12_3_filtered.pdb  → orient_252_pT20__31
      orient_255_pT12__33_pT12_0_filtered.pdb  → orient_255_pT12__33
    """
    basename = os.path.basename(filename)
    # Match orient_<digits>_<tag>__<digits>  (e.g. orient_252_pT20__31)
    match = re.match(r"(orient_\d+_[^_]+__\d+)", basename)
    if match:
        return match.group(1)
    # Fallback: just orient_<digits>
    match = re.match(r"(orient_\d+)", basename)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot parse backbone ID from: {basename}")


def parse_orientation_id(backbone_id):
    """
    Extract the orientation group from a backbone ID.
    orient_252_pT20__31  → orient_252_pT20__31   (it IS the orientation)

    For r3 backbones produced by partial diffusion the backbone id already
    encodes the orientation, so we return it unchanged.  If your filenames
    have an additional sample suffix you can extend this regex.
    """
    return backbone_id


# ── RMSD (Kabsch) ─────────────────────────────────────────────────────────────

def kabsch_rotation(coords_a, coords_b):
    """
    Return rotation matrix R such that coords_b @ R.T best superposes onto
    coords_a (both assumed already centred).
    """
    H        = coords_b.T @ coords_a
    U, S, Vt = np.linalg.svd(H)
    d        = np.linalg.det(Vt.T @ U.T)
    R        = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R


def calc_rmsd(coords_a, coords_b):
    """
    Superpose coords_b onto coords_a (Kabsch) and return Cα RMSD.
    Returns np.inf if lengths differ.
    """
    if len(coords_a) != len(coords_b):
        return np.inf
    ca = coords_a - coords_a.mean(axis=0)
    cb = coords_b - coords_b.mean(axis=0)
    R  = kabsch_rotation(ca, cb)
    cb_rot = cb @ R.T
    return float(np.sqrt(((ca - cb_rot) ** 2).sum(axis=1).mean()))


def superpose_coords(mobile_coords, ref_coords):
    """
    Superpose mobile_coords onto ref_coords.
    Returns the transformed mobile_coords (same shape).
    """
    ref_centre    = ref_coords.mean(axis=0)
    mobile_centre = mobile_coords.mean(axis=0)
    ca = ref_coords    - ref_centre
    cb = mobile_coords - mobile_centre
    R  = kabsch_rotation(ca, cb)
    return (mobile_coords - mobile_centre) @ R.T + ref_centre


# ── Backbone grouping ─────────────────────────────────────────────────────────

def group_by_backbone(pdb_files):
    """
    Group PDB paths by parent backbone ID.
    Returns:
        backbone_to_rep: {backbone_id: representative_pdb_path}
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


def group_by_orientation(backbone_to_rep):
    """
    Group backbone IDs by their orientation.
    Returns {orientation_id: [backbone_id, ...]}
    """
    orientation_to_backbones = defaultdict(list)
    for bb_id in backbone_to_rep:
        oid = parse_orientation_id(bb_id)
        orientation_to_backbones[oid].append(bb_id)
    return dict(orientation_to_backbones)


# ── RMSD matrix ───────────────────────────────────────────────────────────────

def build_rmsd_matrix(backbone_to_rep, binder_chain="A"):
    """
    Compute pairwise Cα RMSD between all backbone representatives.
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


def build_orientation_rmsd_matrix(backbone_ids, backbone_to_rep,
                                  coords_by_id, binder_chain="A"):
    """
    Build pairwise RMSD matrix for a single orientation group.
    coords_by_id: {backbone_id: np.ndarray | None}
    Returns (rmsd_matrix, labels) or (None, None) if <2 valid backbones.
    """
    valid = [(bb_id, coords_by_id[bb_id])
             for bb_id in backbone_ids
             if coords_by_id.get(bb_id) is not None]
    if len(valid) < 2:
        return None, None

    labels = [bb_id for bb_id, _ in valid]
    n      = len(labels)
    mat    = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            rmsd        = calc_rmsd(valid[i][1], valid[j][1])
            mat[i, j]   = rmsd if not np.isinf(rmsd) else np.nan
            mat[j, i]   = mat[i, j]

    return mat, labels


# ── Orientation medoid ────────────────────────────────────────────────────────

def find_orientation_medoid(rmsd_matrix, labels):
    """
    Given an intra-orientation RMSD matrix and labels, return the medoid
    backbone_id (lowest mean RMSD to all others, ignoring NaN).
    """
    mean_rmsds = np.nanmean(rmsd_matrix, axis=1)
    return labels[int(np.argmin(mean_rmsds))]


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_orientation_heatmap(rmsd_matrix, labels, orientation_id, output_dir):
    """
    Save a seaborn heatmap of the intra-orientation pairwise RMSD matrix.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Shorten labels for readability: keep the part after the orientation prefix
    prefix  = orientation_id + "_"
    short   = [l.replace(prefix, "") if l.startswith(prefix) else l
               for l in labels]

    n       = len(labels)
    figsize = max(4, n * 0.5 + 1)

    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))
    sns.heatmap(
        rmsd_matrix,
        xticklabels=short,
        yticklabels=short,
        annot=(n <= 20),          # annotate cells only when matrix is small
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.3,
        ax=ax,
        cbar_kws={"label": "Cα RMSD (Å)"},
        vmin=0,
    )
    ax.set_title(f"{orientation_id}\nintra-group pairwise Cα RMSD", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    plt.tight_layout()

    safe_name = orientation_id.replace("/", "_")
    out_path  = os.path.join(output_dir, f"{safe_name}_rmsd_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved: {out_path}")


# ── Multi-model superposed PDB ────────────────────────────────────────────────

def write_superposed_multimodel_pdb(backbone_ids, backbone_to_rep,
                                    medoid_id, coords_by_id,
                                    orientation_id, output_dir,
                                    binder_chain="A"):
    """
    Superpose all backbones in an orientation group onto the medoid (binder
    chain Cα), then write a multi-model PDB of binder chain A only.

    Each MODEL in the output corresponds to one backbone; the medoid is
    MODEL 1.  A REMARK line records the backbone ID and RMSD to medoid.
    """
    os.makedirs(output_dir, exist_ok=True)

    ref_coords = coords_by_id[medoid_id]
    if ref_coords is None:
        print(f"  [WARN] Medoid {medoid_id} has no coords — skipping superposed PDB")
        return

    # Medoid first, then the rest sorted
    ordered = [medoid_id] + sorted(bb for bb in backbone_ids if bb != medoid_id)

    safe_name = orientation_id.replace("/", "_")
    out_path  = os.path.join(output_dir, f"{safe_name}_superposed.pdb")

    with open(out_path, "w") as fh:
        fh.write(f"REMARK  Orientation: {orientation_id}\n")
        fh.write(f"REMARK  Medoid: {medoid_id}\n")
        fh.write(f"REMARK  {len(ordered)} models, superposed on medoid binder Cα\n")

        for model_num, bb_id in enumerate(ordered, start=1):
            mobile_ca = coords_by_id.get(bb_id)
            if mobile_ca is None:
                continue

            # Compute superposition on Cα, apply to all atoms
            if bb_id == medoid_id:
                transform = None
                rmsd_val  = 0.0
            else:
                rmsd_val = calc_rmsd(ref_coords, mobile_ca)
                # Compute transformation parameters
                ref_centre    = ref_coords.mean(axis=0)
                mobile_centre = mobile_ca.mean(axis=0)
                R = kabsch_rotation(
                    ref_coords - ref_centre,
                    mobile_ca  - mobile_centre,
                )
                transform = (mobile_centre, ref_centre, R)

            all_atoms = get_full_chain_atoms(backbone_to_rep[bb_id], binder_chain)

            fh.write(f"MODEL     {model_num:4d}\n")
            fh.write(f"REMARK  backbone={bb_id}  RMSD_to_medoid={rmsd_val:.3f}\n")

            atom_serial = 1
            for atom in all_atoms:
                coords = atom["coords"].copy()
                if transform is not None:
                    mobile_centre, ref_centre, R = transform
                    coords = (coords - mobile_centre) @ R.T + ref_centre

                x, y, z   = coords
                name      = atom["name"]
                resname   = atom["resname"]
                resnum    = atom["resnum"]
                # PDB ATOM record format
                atom_field = f" {name:<3s}" if len(name) < 4 else name
                line = (
                    f"ATOM  {atom_serial:5d} {atom_field:<4s} {resname:<3s} "
                    f"{binder_chain}{resnum:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00           {name[0]:>2s}\n"
                )
                fh.write(line)
                atom_serial += 1

            fh.write("ENDMDL\n")

        fh.write("END\n")

    print(f"  Superposed PDB: {out_path}  ({len(ordered)} models)")


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
        pattern="*.pdb", dry_run=False,
        visualize_orientations=False, superpose_orientations=False):

    pdb_files = []
    for input_dir in input_dirs:
        pdb_files.extend(sorted(glob.glob(os.path.join(input_dir, pattern))))
    if not pdb_files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in {input_dirs}"
        )
    print(f"Found {len(pdb_files)} filtered structures")

    # ── step 1: group by backbone ─────────────────────────────────────────────
    backbone_to_rep, backbone_to_all = group_by_backbone(pdb_files)
    n_backbones = len(backbone_to_rep)
    print(f"Unique parent backbones: {n_backbones}")
    for bb_id, paths in sorted(backbone_to_all.items()):
        print(f"  {bb_id}: {len(paths)} structures")

    # ── step 2: pairwise RMSD between representatives ─────────────────────────
    rmsd_matrix, backbone_ids, coords_list = build_rmsd_matrix(
        backbone_to_rep, binder_chain
    )
    coords_by_id = {bb: c for bb, c in zip(backbone_ids, coords_list)}

    finite_vals = rmsd_matrix[~np.isinf(rmsd_matrix) & (rmsd_matrix > 0)]
    if len(finite_vals):
        print(f"\nPairwise RMSD (between-backbone): "
              f"min={finite_vals.min():.2f} "
              f"mean={finite_vals.mean():.2f} "
              f"max={finite_vals.max():.2f} Å")

    # ── step 3: intra-orientation analysis ───────────────────────────────────
    orientation_groups = group_by_orientation(backbone_to_rep)
    orientation_medoids = {}  # {orientation_id: medoid_backbone_id}

    if visualize_orientations or superpose_orientations:
        heatmap_dir   = os.path.join(output_dir, "orientation_heatmaps")
        superpose_dir = os.path.join(output_dir, "orientation_superposed")

        print(f"\n── Intra-orientation analysis ──────────────────────────────")
        for oid, bb_ids in sorted(orientation_groups.items()):
            print(f"\n  Orientation: {oid}  ({len(bb_ids)} backbones)")

            mat, labels = build_orientation_rmsd_matrix(
                bb_ids, backbone_to_rep, coords_by_id, binder_chain
            )
            if mat is None:
                print(f"    Skipped (fewer than 2 valid backbones)")
                continue

            # RMSD summary for this orientation
            upper = mat[np.triu_indices(len(labels), k=1)]
            valid = upper[~np.isnan(upper)]
            if len(valid):
                print(f"    RMSD: min={valid.min():.2f}  "
                      f"mean={valid.mean():.2f}  "
                      f"max={valid.max():.2f} Å")

            medoid = find_orientation_medoid(mat, labels)
            orientation_medoids[oid] = medoid
            print(f"    Medoid: {medoid}")

            if not dry_run:
                if visualize_orientations:
                    plot_orientation_heatmap(mat, labels, oid, heatmap_dir)

                if superpose_orientations:
                    write_superposed_multimodel_pdb(
                        bb_ids, backbone_to_rep, medoid,
                        coords_by_id, oid, superpose_dir, binder_chain
                    )
            else:
                print(f"    [DRY RUN] Would write heatmap and superposed PDB")

    # ── step 4: cross-backbone clustering ────────────────────────────────────
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

    total_structures = sum(
        len(backbone_to_all[bb]) for bb in surviving_backbones
    )
    print(f"\nStructures from surviving backbones: {total_structures} "
          f"(from {n_clusters} backbones)")

    # ── step 5: copy structures ───────────────────────────────────────────────
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

    # ── step 6: CSV log ───────────────────────────────────────────────────────
    rows = []
    for idx, bb_id in enumerate(backbone_ids):
        cid = cluster_labels[idx]
        oid = parse_orientation_id(bb_id)
        rows.append({
            "backbone_id":          bb_id,
            "orientation_id":       oid,
            "representative_pdb":   backbone_to_rep[bb_id],
            "binder_len":           len(coords_list[idx]) if coords_list[idx] is not None else None,
            "cluster_id":           cid,
            "cluster_medoid":       medoids[cid],
            "is_cluster_medoid":    bb_id == medoids[cid],
            "orientation_medoid":   orientation_medoids.get(oid, ""),
            "is_orientation_medoid": bb_id == orientation_medoids.get(oid, ""),
            "n_structures":         len(backbone_to_all[bb_id]),
        })

    df = pd.DataFrame(rows).sort_values(
        ["cluster_id", "is_cluster_medoid"], ascending=[True, False]
    )

    if not dry_run:
        log_path = os.path.join(output_dir, "clustering_log.csv")
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
    parser.add_argument("--input_dirs", type=str, nargs="+", required=False,
        metavar="DIR",
        help="One or more directories containing RFdiffusion output PDBs."
    )
    parser.add_argument("--output_dir", type=str, required=False,
        metavar="DIR",
        help="Directory to write passing structures and clustering_log.csv."
    )
    parser.add_argument("--rmsd_cutoff",  type=float, default=2.0,
        help="Cα RMSD cutoff in Å for cross-backbone clustering (default: 2.0).")
    parser.add_argument("--binder_chain", type=str,   default="A",
        help="Binder chain letter (default: A).")
    parser.add_argument("--pattern",      type=str,   default="*.pdb",
        help="Glob pattern for input files.")
    parser.add_argument("--dry_run",      action="store_true",
        help="Show clustering results without copying files or writing plots.")
    parser.add_argument("--visualize_orientations", action="store_true",
        help="Save intra-orientation pairwise RMSD heatmap PNGs to "
             "output_dir/orientation_heatmaps/.")
    parser.add_argument("--superpose_orientations", action="store_true",
        help="Write medoid-superposed multi-model PDBs (binder chain only) to "
             "output_dir/orientation_superposed/.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_dirs              = args.input_dirs,
        output_dir              = args.output_dir,
        rmsd_cutoff             = args.rmsd_cutoff,
        binder_chain            = args.binder_chain,
        pattern                 = args.pattern,
        dry_run                 = args.dry_run,
        visualize_orientations  = args.visualize_orientations,
        superpose_orientations  = args.superpose_orientations,
    )