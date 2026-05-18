"""
partial_diffusion_backbone_iRMSD.py
=============================
Cross-backbone clustering for partial diffusion filtered outputs.

Clustering metric: interface RMSD (iRMSD)
  Global Cα RMSD is a poor metric for binder-peptide systems because two
  backbones can superpose well globally while contacting the peptide
  completely differently.  iRMSD restricts the RMSD calculation to binder
  residues whose Cα is within a distance cutoff of any peptide Cα
  in either structure.  This focuses the metric on the interface geometry
  that actually matters.

Chain layout assumed:
  Chain A = binder
  Chain B = MHC (residues 1..N-9) + peptide (last 9 residues)
  The peptide is extracted automatically from the last 9 residues of chain B.
  No chain-splitting preprocessing is required.

Strategy:
  1. Parse the FULL trajectory backbone ID from each filename — each pT__N
     trajectory (e.g. orient_252_pT20__31_pT12__0, ..._pT12__99) is treated
     as its own independent backbone. They are NOT pre-collapsed.
  2. Pick one PDB per backbone ID as its representative (first by filename sort).
  3. Compute pairwise iRMSD between all backbone representatives.
  4. Cluster by iRMSD using hierarchical clustering (average linkage) at a
     given iRMSD cutoff.
  5. For each cluster, keep the medoid — backbone with lowest mean iRMSD
     to all others in the cluster.
  6. Copy all filtered structures from surviving backbones to output_dir.

Visualizations:
  --visualize_orientations
      For each orientation group (e.g. orient_252_pT20__31):
        (a) Intra-group pairwise iRMSD heatmap
        (b) Per-peptide-position min-distance heatmap: rows = backbones,
            cols = peptide residues C1-C9.  Shows which backbone contacts
            which peptide position, making contact diversity visible at a
            glance even when iRMSD is similar.

  --superpose_orientations
      For each orientation group, superpose all backbones onto the group
      medoid (lowest mean intra-group iRMSD) and write a multi-model PDB
      to output_dir/orientation_superposed/.  Only binder chain A is
      written; superposition is computed on interface Cα atoms.

Usage
-----
python partial_diffusion_backbone_iRMSD.py \
    --input_dirs  /path/to/r3/tier1 /path/to/r3/tier2 \
    --output_dir  /path/to/clustered \
    --peptide_chain C \
    --contact_cutoff 8.0 \
    --irmsd_cutoff 2.0 \
    --visualize_orientations \
    --superpose_orientations \
    --dry_run
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

# ── Chain B peptide extraction ────────────────────────────────────────────────
# The last PEPTIDE_LEN residues of chain B are the peptide; everything before
# is MHC (ignored).  Chain A is the binder.
PEPTIDE_LEN = 9


def _split_chain_b(pdb_path):
    """
    Parse chain B and split into (mhc_residues, peptide_residues) by position.
    peptide_residues = last PEPTIDE_LEN unique resnums of chain B, sorted.
    Returns (peptide_residues_list, all_chain_b_residues_list).
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    residues = []
    for model in structure:
        for chain in model:
            if chain.id == "B":
                for residue in chain:
                    residues.append(residue)
        break
    if not residues:
        raise ValueError(f"No chain B in {pdb_path}")
    # Deduplicate by resnum, preserve order
    seen, ordered = set(), []
    for r in residues:
        rn = r.get_id()[1]
        if rn not in seen:
            seen.add(rn)
            ordered.append(r)
    peptide_residues = ordered[-PEPTIDE_LEN:]
    return peptide_residues, ordered


def get_chain_ca_coords(pdb_path, chain_id):
    """Return Cα coords (N,3) for all residues in chain_id.
    chain_id='A' → binder; chain_id='B' → full chain B (MHC+peptide, rarely needed)."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    coords = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if "CA" in residue:
                        coords.append(residue["CA"].get_vector().get_array())
        break
    if not coords:
        raise ValueError(f"No Cα found in chain {chain_id} of {pdb_path}")
    return np.array(coords)


def get_peptide_ca_coords(pdb_path):
    """
    Return Cα coords (P,3) for the peptide = last PEPTIDE_LEN residues
    of chain B.  Used for the interface contact mask in iRMSD.
    """
    pep_residues, _ = _split_chain_b(pdb_path)
    coords = []
    for residue in pep_residues:
        if "CA" in residue:
            coords.append(residue["CA"].get_vector().get_array())
    if not coords:
        raise ValueError(f"No peptide Cα atoms in {pdb_path}")
    return np.array(coords)


def get_peptide_ca_by_resnum(pdb_path, chain_id=None):
    """
    Return {position_index: ca_coord} for the peptide residues (1-indexed,
    1 = first peptide residue).  chain_id is ignored — always uses last
    PEPTIDE_LEN residues of chain B.
    """
    pep_residues, _ = _split_chain_b(pdb_path)
    result = {}
    for pos, residue in enumerate(pep_residues, start=1):
        if "CA" in residue:
            result[pos] = residue["CA"].get_vector().get_array()
    return result


def get_full_chain_atoms(pdb_path, chain_id):
    """Return list of atom dicts for all atoms in chain_id (for PDB writing)."""
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
        break
    return atoms


# ── Filename parsing ──────────────────────────────────────────────────────────

def parse_backbone_id(filename):
    """
    Extract full trajectory backbone ID. Each pT__N trajectory is its own
    backbone — NOT collapsed into its parent orientation.

    orient_252_pT20__31_pT12__0.pdb  → orient_252_pT20__31_pT12__0
    orient_255_pT12__33_pT12__5.pdb  → orient_255_pT12__33_pT12__5
    orient_252_pT20__31_filtered.pdb → orient_252_pT20__31  (r2 style)
    """
    basename = os.path.basename(filename)
    m = re.match(r"(orient_\d+_[^_]+__\d+_pT\d+__\d+)", basename)
    if m:
        return m.group(1)
    m = re.match(r"(orient_\d+_[^_]+__\d+)", basename)
    if m:
        return m.group(1)
    m = re.match(r"(orient_\d+)", basename)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse backbone ID from: {basename}")


def parse_orientation_id(backbone_id):
    """
    Extract the parent orientation from a full trajectory backbone ID.
    Used ONLY for visualization grouping — never for clustering.

    orient_252_pT20__31_pT12__0  → orient_252_pT20__31
    orient_252_pT20__31          → orient_252_pT20__31
    """
    m = re.match(r"(orient_\d+_[^_]+__\d+)_pT\d+__\d+", backbone_id)
    return m.group(1) if m else backbone_id


# ── Kabsch superposition ──────────────────────────────────────────────────────

def kabsch_rotation(coords_a, coords_b):
    """
    Kabsch rotation R s.t. coords_b @ R.T best superposes onto coords_a.
    Both arrays must already be centred.
    """
    H        = coords_b.T @ coords_a
    U, S, Vt = np.linalg.svd(H)
    d        = np.linalg.det(Vt.T @ U.T)
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def calc_rmsd_aligned(coords_a, coords_b):
    """RMSD after Kabsch superposition of equal-length Cα arrays."""
    assert len(coords_a) == len(coords_b)
    ca = coords_a - coords_a.mean(axis=0)
    cb = coords_b - coords_b.mean(axis=0)
    R  = kabsch_rotation(ca, cb)
    return float(np.sqrt(((ca - cb @ R.T) ** 2).sum(axis=1).mean()))


def apply_kabsch_transform(mobile, ref_centre, mobile_centre, R):
    """Apply a pre-computed Kabsch transform to arbitrary coordinates."""
    return (mobile - mobile_centre) @ R.T + ref_centre


# ── Interface residue selection ───────────────────────────────────────────────

def interface_mask(binder_ca, pep_ca, contact_cutoff):
    """
    Boolean mask (N,) over binder Cα: True if within contact_cutoff Å of
    any peptide Cα.

    Uses vectorised distance: (N, P) distance matrix via broadcasting.
    """
    diff  = binder_ca[:, None, :] - pep_ca[None, :, :]  # (N, P, 3)
    dists = np.linalg.norm(diff, axis=-1)                # (N, P)
    return dists.min(axis=1) < contact_cutoff            # (N,)


# ── iRMSD ─────────────────────────────────────────────────────────────────────

def calc_irmsd(binder_ca_a, pep_ca_a,
               binder_ca_b, pep_ca_b,
               contact_cutoff=8.0):
    """
    Interface RMSD between two binder backbones using Cα throughout.

    1. Find interface residues in each structure: binder Cα within
       contact_cutoff of any peptide Cα.
    2. Take the union (include a residue if in contact in EITHER structure).
    3. Superpose those binder Cα and return RMSD.

    Falls back to global RMSD if fewer than 3 interface residues in union.
    Returns np.inf if binder lengths differ.
    """
    if len(binder_ca_a) != len(binder_ca_b):
        return np.inf

    mask_a = interface_mask(binder_ca_a, pep_ca_a, contact_cutoff)
    mask_b = interface_mask(binder_ca_b, pep_ca_b, contact_cutoff)
    union  = mask_a | mask_b

    if union.sum() < 3:
        return calc_rmsd_aligned(binder_ca_a, binder_ca_b)

    return calc_rmsd_aligned(binder_ca_a[union], binder_ca_b[union])


# ── Per-peptide contact distances ─────────────────────────────────────────────

def contact_distance_vector(binder_ca, pep_ca_by_resnum):
    """
    For each peptide residue (sorted by resnum), compute the minimum Cα–Cα
    distance to any binder Cα.

    Returns (resnums_list, distances_array).
    """
    resnums = sorted(pep_ca_by_resnum.keys())
    dists   = [
        np.linalg.norm(binder_ca - pep_ca_by_resnum[rn], axis=-1).min()
        for rn in resnums
    ]
    return resnums, np.array(dists)


# ── Backbone grouping ─────────────────────────────────────────────────────────

def group_by_backbone(pdb_files):
    """
    Map each backbone ID to its single PDB file.
    Each backbone is expected to have exactly one associated PDB.
    Warns if duplicates are found (e.g. from overlapping input_dirs).
    """
    backbone_to_pdb = {}
    for pdb in sorted(pdb_files):
        bb = parse_backbone_id(pdb)
        if bb in backbone_to_pdb:
            print(f"  [WARN] Duplicate backbone {bb} — "
                  f"keeping {backbone_to_pdb[bb]}, ignoring {pdb}")
        else:
            backbone_to_pdb[bb] = pdb
    return backbone_to_pdb


def group_by_orientation(backbone_to_pdb):
    oid_to_bbs = defaultdict(list)
    for bb in backbone_to_pdb:
        oid_to_bbs[parse_orientation_id(bb)].append(bb)
    return dict(oid_to_bbs)


# ── Load all coordinate data ──────────────────────────────────────────────────

def load_backbone_data(backbone_to_pdb, binder_chain):
    """
    Load binder Cα, peptide Cα (for contact mask), and peptide Cα-by-resnum for every
    backbone representative.

    Returns three dicts keyed by backbone_id — value is None on load failure.
    """
    backbone_ids     = sorted(backbone_to_pdb.keys())
    binder_ca_by_id  = {}
    pep_ca_by_id  = {}
    pep_ca_res_by_id = {}

    print(f"Loading {len(backbone_ids)} backbone representatives...")
    for bb in backbone_ids:
        pdb = backbone_to_pdb[bb]
        try:
            binder_ca_by_id[bb]  = get_chain_ca_coords(pdb, binder_chain)
            pep_ca_by_id[bb]  = get_peptide_ca_coords(pdb)
            pep_ca_res_by_id[bb] = get_peptide_ca_by_resnum(pdb)
        except Exception as e:
            print(f"  [WARN] {bb}: {e}")
            binder_ca_by_id[bb]  = None
            pep_ca_by_id[bb]  = None
            pep_ca_res_by_id[bb] = None

    return binder_ca_by_id, pep_ca_by_id, pep_ca_res_by_id


# ── iRMSD matrix ─────────────────────────────────────────────────────────────

def build_irmsd_matrix(backbone_ids, binder_ca_by_id, pep_ca_by_id,
                       contact_cutoff):
    n   = len(backbone_ids)
    mat = np.full((n, n), np.inf)
    np.fill_diagonal(mat, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            bca_i = binder_ca_by_id[backbone_ids[i]]
            bca_j = binder_ca_by_id[backbone_ids[j]]
            pca_i = pep_ca_by_id[backbone_ids[i]]
            pca_j = pep_ca_by_id[backbone_ids[j]]
            if any(x is None for x in [bca_i, bca_j, pca_i, pca_j]):
                continue
            v = calc_irmsd(bca_i, pca_i, bca_j, pca_j, contact_cutoff)
            mat[i, j] = mat[j, i] = v
    return mat


# ── Medoid selection ──────────────────────────────────────────────────────────

def find_medoid(mat, labels):
    """Return label with lowest mean iRMSD (ignoring inf)."""
    safe  = np.where(np.isinf(mat), np.nan, mat)
    means = np.nanmean(safe, axis=1)
    return labels[int(np.argmin(means))]


# ── Visualization ─────────────────────────────────────────────────────────────

def _short(labels, orientation_id):
    prefix = orientation_id + "_"
    return [l.replace(prefix, "") if l.startswith(prefix) else l
            for l in labels]


def plot_irmsd_heatmap(mat, labels, orientation_id, output_dir):
    """Save intra-orientation pairwise iRMSD heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    short   = _short(labels, orientation_id)
    n       = len(labels)
    fs      = max(4, n * 0.5 + 1)
    fig, ax = plt.subplots(figsize=(fs, fs * 0.85))
    disp    = np.where(np.isinf(mat), np.nan, mat)
    sns.heatmap(disp, xticklabels=short, yticklabels=short,
                annot=(n <= 20), fmt=".2f", cmap="YlOrRd",
                linewidths=0.3, ax=ax, vmin=0,
                cbar_kws={"label": "interface Cα RMSD (Å)"})
    ax.set_title(f"{orientation_id}\nintra-group pairwise iRMSD", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    plt.tight_layout()
    safe = orientation_id.replace("/", "_")
    path = os.path.join(output_dir, f"{safe}_irmsd_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    iRMSD heatmap: {path}")


def plot_contact_distance_heatmap(labels, pep_ca_res_by_id, binder_ca_by_id,
                                  orientation_id, output_dir, medoid_id=None):
    """
    Per-peptide-position min Cα–Cα distance heatmap.

    Rows  = backbones (medoid first, marked with *)
    Cols  = peptide residue positions sorted by resnum (C1 … C9)
    Color = min distance from that peptide residue to any binder Cα:
              red  (close) → green (far)
            vmin=3.5 Å  vmax=20 Å

    This reveals contact diversity that iRMSD can miss: two backbones with
    similar iRMSD can still differ in which peptide positions they contact.
    """
    os.makedirs(output_dir, exist_ok=True)

    valid = [(bb, pep_ca_res_by_id[bb], binder_ca_by_id[bb])
             for bb in labels
             if pep_ca_res_by_id.get(bb) is not None
             and binder_ca_by_id.get(bb) is not None]
    if not valid:
        return

    all_resnums = sorted(valid[0][1].keys())

    # Medoid first, then sorted
    ordered = ([e for e in valid if e[0] == medoid_id] +
               [e for e in valid if e[0] != medoid_id]) if medoid_id else valid

    prefix     = orientation_id + "_"
    row_labels = []
    matrix     = []
    for bb, pca_res, bca in ordered:
        _, dists = contact_distance_vector(bca, pca_res)
        matrix.append(dists)
        short = bb.replace(prefix, "") if bb.startswith(prefix) else bb
        row_labels.append(f"* {short}" if bb == medoid_id else short)

    matrix     = np.array(matrix)
    col_labels = [f"C{rn}" for rn in all_resnums]

    fs      = (max(5, len(col_labels) * 0.7 + 1),
               max(3, len(row_labels) * 0.45 + 1))
    fig, ax = plt.subplots(figsize=fs)
    sns.heatmap(matrix, xticklabels=col_labels, yticklabels=row_labels,
                annot=True, fmt=".1f", cmap="RdYlGn_r",
                linewidths=0.3, ax=ax, vmin=3.5, vmax=20.0,
                cbar_kws={"label": "min Cα–Cα distance to binder (Å)"})
    ax.set_title(
        f"{orientation_id}\n"
        f"min binder Cα distance per peptide position  (* = iRMSD medoid)",
        fontsize=9,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    plt.tight_layout()
    safe = orientation_id.replace("/", "_")
    path = os.path.join(output_dir, f"{safe}_contact_distances.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Contact distance heatmap: {path}")


# ── Multi-model superposed PDB ────────────────────────────────────────────────

def write_superposed_multimodel_pdb(backbone_ids, backbone_to_pdb,
                                    medoid_id, binder_ca_by_id,
                                    orientation_id, output_dir,
                                    binder_chain="A"):
    """
    Superpose all backbones in the group onto the medoid using all binder Cα,
    then write a multi-model PDB (binder chain only).
    Medoid is MODEL 1; REMARK records backbone ID and iRMSD to medoid.
    """
    os.makedirs(output_dir, exist_ok=True)
    ref_ca = binder_ca_by_id.get(medoid_id)
    if ref_ca is None:
        print(f"  [WARN] Medoid {medoid_id} has no coords — skipping")
        return

    ordered    = [medoid_id] + sorted(bb for bb in backbone_ids
                                      if bb != medoid_id)
    ref_centre = ref_ca.mean(axis=0)
    safe       = orientation_id.replace("/", "_")
    out_path   = os.path.join(output_dir, f"{safe}_superposed.pdb")

    with open(out_path, "w") as fh:
        fh.write(f"REMARK  Orientation: {orientation_id}\n")
        fh.write(f"REMARK  Medoid (MODEL 1): {medoid_id}\n")
        fh.write(f"REMARK  {len(ordered)} models superposed on medoid binder Ca\n")

        for model_num, bb in enumerate(ordered, start=1):
            mobile_ca = binder_ca_by_id.get(bb)
            if mobile_ca is None:
                continue

            if bb == medoid_id:
                irmsd_val, transform = 0.0, None
            else:
                irmsd_val     = calc_rmsd_aligned(ref_ca, mobile_ca)
                mobile_centre = mobile_ca.mean(axis=0)
                R             = kabsch_rotation(ref_ca     - ref_centre,
                                                mobile_ca  - mobile_centre)
                transform = (mobile_centre, ref_centre, R)

            fh.write(f"MODEL     {model_num:4d}\n")
            fh.write(f"REMARK  backbone={bb}  iRMSD_to_medoid={irmsd_val:.3f}\n")

            for serial, atom in enumerate(
                    get_full_chain_atoms(backbone_to_pdb[bb], binder_chain),
                    start=1):
                coords = atom["coords"].copy()
                if transform is not None:
                    coords = apply_kabsch_transform(coords, *transform)
                x, y, z    = coords
                name       = atom["name"]
                atom_field = f" {name:<3s}" if len(name) < 4 else name
                fh.write(
                    f"ATOM  {serial:5d} {atom_field:<4s} {atom['resname']:<3s} "
                    f"{binder_chain}{atom['resnum']:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00           {name[0]:>2s}\n"
                )
            fh.write("ENDMDL\n")

        fh.write("END\n")

    print(f"    Superposed PDB: {out_path}  ({len(ordered)} models)")


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_and_pick_medoids(irmsd_matrix, backbone_ids, irmsd_cutoff):
    finite = np.where(np.isinf(irmsd_matrix), 1e6, irmsd_matrix)
    Z      = linkage(squareform(finite, checks=False), method="average")
    labels = fcluster(Z, t=irmsd_cutoff, criterion="distance")

    clusters = defaultdict(list)
    for idx, lbl in enumerate(labels):
        clusters[lbl].append(idx)

    medoids = {}
    for cid, members in clusters.items():
        if len(members) == 1:
            medoids[cid] = backbone_ids[members[0]]
        else:
            sub     = irmsd_matrix[np.ix_(members, members)]
            sub_fin = np.where(np.isinf(sub), 1e6, sub)
            best    = int(sub_fin.mean(axis=1).argmin())
            medoids[cid] = backbone_ids[members[best]]

    return labels, medoids


# ── Main ──────────────────────────────────────────────────────────────────────

def run(input_dirs, output_dir, binder_chain="A",
        contact_cutoff=8.0, irmsd_cutoff=2.0, pattern="*.pdb",
        dry_run=False, visualize_orientations=False,
        superpose_orientations=False):

    # gather files
    pdb_files = []
    for d in input_dirs:
        pdb_files.extend(sorted(glob.glob(os.path.join(d, pattern))))
    if not pdb_files:
        raise FileNotFoundError(f"No '{pattern}' files in {input_dirs}")
    print(f"Found {len(pdb_files)} PDB files")

    # group by backbone (1 PDB per backbone)
    backbone_to_pdb = group_by_backbone(pdb_files)
    backbone_ids    = sorted(backbone_to_pdb.keys())
    n_backbones     = len(backbone_ids)
    print(f"Unique backbones: {n_backbones}")

    # load coordinates
    binder_ca_by_id, pep_ca_by_id, pep_ca_res_by_id = load_backbone_data(
        backbone_to_pdb, binder_chain
    )

    # pairwise iRMSD
    print(f"\nComputing pairwise iRMSD  "
          f"(contact cutoff={contact_cutoff} Å)...")
    irmsd_matrix = build_irmsd_matrix(
        backbone_ids, binder_ca_by_id, pep_ca_by_id, contact_cutoff
    )
    fv = irmsd_matrix[~np.isinf(irmsd_matrix) & (irmsd_matrix > 0)]
    if len(fv):
        print(f"iRMSD: min={fv.min():.2f}  mean={fv.mean():.2f}  "
              f"max={fv.max():.2f} Å")

    # intra-orientation analysis (always runs)
    # each orientation groups e.g. orient_252_pT20__31_pT12__0..99 together
    # without collapsing them — each trajectory remains its own backbone
    orientation_groups  = group_by_orientation(backbone_to_pdb)
    orientation_medoids = {}
    idx_map             = {bb: i for i, bb in enumerate(backbone_ids)}
    heatmap_dir         = os.path.join(output_dir, "orientation_heatmaps")
    superpose_dir       = os.path.join(output_dir, "orientation_superposed")

    print(f"\n── Intra-orientation analysis "
          f"({len(orientation_groups)} orientation groups) ─────────────")

    for oid, bb_ids in sorted(orientation_groups.items()):
        oid_idx = [idx_map[bb] for bb in bb_ids if bb in idx_map]
        n_valid = len(oid_idx)
        print(f"\n  {oid}  ({n_valid} backbones)")

        if n_valid < 2:
            print("    Only 1 backbone — no intra-group comparison possible")
            if n_valid == 1:
                orientation_medoids[oid] = backbone_ids[oid_idx[0]]
            continue

        sub_mat    = irmsd_matrix[np.ix_(oid_idx, oid_idx)]
        sub_labels = [backbone_ids[i] for i in oid_idx]

        upper = sub_mat[np.triu_indices(n_valid, k=1)]
        valid = upper[~np.isinf(upper)]
        if len(valid):
            print(f"    iRMSD: min={valid.min():.2f}  "
                  f"mean={valid.mean():.2f}  "
                  f"max={valid.max():.2f} Å")

        medoid = find_medoid(sub_mat, sub_labels)
        orientation_medoids[oid] = medoid
        print(f"    Medoid: {medoid}")

        if not dry_run:
            if visualize_orientations:
                plot_irmsd_heatmap(
                    sub_mat, sub_labels, oid, heatmap_dir
                )
                plot_contact_distance_heatmap(
                    sub_labels, pep_ca_res_by_id, binder_ca_by_id,
                    oid, heatmap_dir, medoid_id=medoid
                )
            if superpose_orientations:
                write_superposed_multimodel_pdb(
                    bb_ids, backbone_to_pdb, medoid,
                    binder_ca_by_id, oid, superpose_dir, binder_chain
                )
        else:
            extras = []
            if visualize_orientations:
                extras.append("heatmaps")
            if superpose_orientations:
                extras.append("superposed PDB")
            if extras:
                print(f"    [DRY RUN] Would write: {', '.join(extras)}")

    # cross-backbone clustering
    print(f"\nClustering at iRMSD ≤ {irmsd_cutoff} Å...")
    cluster_labels, medoids = cluster_and_pick_medoids(
        irmsd_matrix, backbone_ids, irmsd_cutoff
    )
    n_clusters          = len(medoids)
    surviving_backbones = list(medoids.values())

    print(f"{n_backbones} backbones → {n_clusters} clusters")
    for cid, medoid in sorted(medoids.items()):
        members = [backbone_ids[i]
                   for i, l in enumerate(cluster_labels) if l == cid]
        merged  = [m for m in members if m != medoid]
        tag     = f" ← {', '.join(merged)}" if merged else ""
        print(f"  [{cid}] {medoid} (medoid){tag}")

    print(f"\n{n_clusters} surviving backbones")

    # copy structures (1 PDB per surviving backbone)
    if dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        os.makedirs(output_dir, exist_ok=True)
        for bb in surviving_backbones:
            shutil.copy2(backbone_to_pdb[bb],
                         os.path.join(output_dir,
                                      os.path.basename(backbone_to_pdb[bb])))
        print(f"Copied {len(surviving_backbones)} structures → {output_dir}")

    # CSV log
    rows = []
    for idx, bb in enumerate(backbone_ids):
        cid  = cluster_labels[idx]
        oid  = parse_orientation_id(bb)
        bca  = binder_ca_by_id.get(bb)
        pcr  = pep_ca_res_by_id.get(bb)

        contact_cols = {}
        if bca is not None and pcr is not None:
            resnums, dists = contact_distance_vector(bca, pcr)
            for rn, d in zip(resnums, dists):
                contact_cols[f"min_dist_C{rn}"] = round(float(d), 2)

        rows.append({
            "backbone_id":           bb,
            "orientation_id":        oid,
            "pdb":                   backbone_to_pdb[bb],
            "binder_len":            len(bca) if bca is not None else None,
            "cluster_id":            cid,
            "cluster_medoid":        medoids[cid],
            "is_cluster_medoid":     bb == medoids[cid],
            "orientation_medoid":    orientation_medoids.get(oid, ""),
            "is_orientation_medoid": bb == orientation_medoids.get(oid, ""),
            **contact_cols,
        })

    df = pd.DataFrame(rows).sort_values(
        ["cluster_id", "is_cluster_medoid"], ascending=[True, False]
    )

    if not dry_run:
        log_path = os.path.join(output_dir, "clustering_log.csv")
        df.to_csv(log_path, index=False)
        print(f"Log: {log_path}")

    print(f"\nFinal: {n_backbones} backbones → {n_clusters} clusters ready for ProteinMPNN")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="iRMSD-based backbone clustering for partial diffusion outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_dirs",    nargs="+", required=False, metavar="DIR",
        help="Directories containing RFdiffusion output PDBs.")
    p.add_argument("--output_dir",    required=False, metavar="DIR",
        help="Directory for clustered structures and logs.")
    p.add_argument("--binder_chain",  default="A",
        help="Binder chain letter (default: A).")
    p.add_argument("--contact_cutoff", type=float, default=8.0,
        help="Binder Cα to peptide Cα cutoff in Å for defining "
             "interface residues (default: 8.0).")
    p.add_argument("--irmsd_cutoff",  type=float, default=2.0,
        help="iRMSD cutoff in Å for clustering (default: 2.0).")
    p.add_argument("--pattern",       default="*.pdb",
        help="Glob pattern (default: '*.pdb'). "
             "For r2 use '*_filtered.pdb'.")
    p.add_argument("--dry_run",       action="store_true",
        help="Report clustering without copying files or writing plots.")
    p.add_argument("--visualize_orientations", action="store_true",
        help="Write iRMSD heatmap + contact-distance heatmap PNGs per "
             "orientation to output_dir/orientation_heatmaps/.")
    p.add_argument("--superpose_orientations", action="store_true",
        help="Write medoid-superposed multi-model PDBs per orientation to "
             "output_dir/orientation_superposed/.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_dirs             = args.input_dirs,
        output_dir             = args.output_dir,
        binder_chain           = args.binder_chain,
        contact_cutoff         = args.contact_cutoff,
        irmsd_cutoff           = args.irmsd_cutoff,
        pattern                = args.pattern,
        dry_run                = args.dry_run,
        visualize_orientations = args.visualize_orientations,
        superpose_orientations = args.superpose_orientations,
    )