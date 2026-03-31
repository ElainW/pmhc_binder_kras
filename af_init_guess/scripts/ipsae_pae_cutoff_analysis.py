"""
pae_cutoff_analysis.py
======================
Three complementary analyses for validating the PAE cutoff used in
binder<->peptide ipSAE scoring for pMHCI minibinder designs.

Approach 1: PAE distribution plot
    Visualise the distribution of binder<->peptide PAE values across
    designs to check whether 12 A sits in a natural valley between
    contact and non-contact modes.

Approach 2: Sensitivity / rank-stability analysis
    Compute ipSAE at a sweep of cutoffs and measure Spearman rank
    correlation between them. If rankings are stable across a wide
    range, the exact cutoff value is not critical.

Approach 3: Crystal structure validation (9O5S / mart1-3)
    Use the AF2 prediction of the known crystal complex as ground
    truth. For each cutoff, compute precision / recall / F1 of
    PAE-defined contacts against Cα-distance-defined contacts in
    the crystal PDB.

Usage examples
--------------
# Approach 1: distribution plot for a set of designs
python pae_cutoff_analysis.py distribution \\
    --npy design1_pae.npy design2_pae.npy design3_pae.npy \\
    --pdb design1.pdb design2.pdb design3.pdb \\
    --labels design1 design2 design3 \\
    --out pae_distribution.png

# Approach 2: rank-stability across cutoffs
python pae_cutoff_analysis.py sensitivity \\
    --npy design*.npy \\
    --pdb design*.pdb \\
    --out rank_stability.png

# Approach 3: crystal structure validation
python pae_cutoff_analysis.py crystal \\
    --npy 9O5S_af2pred_pae.npy \\
    --pred-pdb 9O5S_af2pred.pdb \\
    --crystal-pdb 9O5S_crystal.pdb \\
    --binder-chain-crystal A \\
    --peptide-chain-crystal C \\
    --binder 102 --mhc 174 --peptide 10 \\
    --out crystal_validation.png

Dependencies: numpy, matplotlib, scipy, biopython
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── import from pmhci_ipsae ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from calc_ipsae import (
    load_pae_matrix,
    build_chain_slices,
    chain_lengths_from_pdb,
    compute_binder_peptide_ipsae,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_chain_lengths(
    npy_path: str,
    pdb_path: str | None,
    binder: int | None,
    mhc: int | None,
    peptide: int | None,
) -> OrderedDict:
    """Auto-detect from PDB or use manual values, same logic as CLI."""
    if all(v is not None for v in (binder, mhc, peptide)):
        return OrderedDict([("binder", binder), ("MHC", mhc), ("peptide", peptide)])
    if pdb_path:
        cl = chain_lengths_from_pdb(pdb_path)
        if binder  is not None: cl["binder"]  = binder
        if mhc     is not None: cl["MHC"]     = mhc
        if peptide is not None: cl["peptide"] = peptide
        return cl
    raise ValueError(
        f"Cannot resolve chain lengths for {npy_path}: "
        "provide --pdb or all of --binder / --mhc / --peptide."
    )


def load_bp_pae(npy_path: str, chain_lengths: OrderedDict) -> np.ndarray:
    """Return the binder×peptide PAE submatrix (shape: n_binder × n_peptide)."""
    pae    = load_pae_matrix(npy_path)
    slices = build_chain_slices(chain_lengths)
    return pae[slices["binder"], slices["peptide"]]


STYLE = dict(
    marker_cutoff=dict(color="#e63946", linestyle="--", linewidth=1.5,
                       label="cutoff = {c} Å"),
    colors=plt.cm.tab10.colors,
)


# ─────────────────────────────────────────────────────────────────────────────
# Approach 1 — PAE distribution
# ─────────────────────────────────────────────────────────────────────────────

def approach1_distribution(
    npy_paths: list[str],
    pdb_paths: list[str | None],
    labels: list[str],
    cutoff: float = 12.0,
    binder: int | None = None,
    mhc: int | None = None,
    peptide: int | None = None,
    out: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot histogram + CDF of binder<->peptide PAE values.

    What to look for
    ----------------
    - Bimodal or shoulder distribution: one mode below ~10 Å (contacts)
      and one above ~20 Å (non-contacts). Ideal cutoff sits in the valley.
    - Unimodal flat distribution: cutoff matters less.
    - CDF line shows what fraction of all pairs fall below the cutoff.
      For a strong interface this should be well above 0 (see 9O5S: ~100%
      of pairs below 12 Å because mean PAE = 2.92 Å).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_hist, ax_cdf = axes

    all_vals: list[np.ndarray] = []

    for i, (npy, pdb, label) in enumerate(zip(npy_paths, pdb_paths, labels)):
        cl   = resolve_chain_lengths(npy, pdb, binder, mhc, peptide)
        vals = load_bp_pae(npy, cl).flatten()
        all_vals.append(vals)

        color = STYLE["colors"][i % len(STYLE["colors"])]
#         ax_hist.hist(vals, bins=64, alpha=0.5, density=True,
#                      color=color, label=label)
        ax_hist.hist(vals, bins=64, alpha=0.5, density=True,
                     color=color)

        # CDF per design (thin lines)
        sv  = np.sort(vals)
        cdf = np.arange(1, len(sv) + 1) / len(sv)
#         ax_cdf.plot(sv, cdf, color=color, alpha=0.7, linewidth=1.2, label=label)
        ax_cdf.plot(sv, cdf, color=color, alpha=0.7, linewidth=1.2)

    # Pooled CDF (thick)
    pooled = np.concatenate(all_vals)
    sv_p   = np.sort(pooled)
    cdf_p  = np.arange(1, len(sv_p) + 1) / len(sv_p)
    ax_cdf.plot(sv_p, cdf_p, color="black", linewidth=2.5,
                label="pooled", zorder=5)

    frac = float(np.mean(pooled < cutoff))

    for ax in axes:
        ax.axvline(cutoff, color="#e63946", linestyle="--", linewidth=1.8,
                   label=f"cutoff = {cutoff} Å")

    ax_cdf.axhline(frac, color="grey", linestyle=":", linewidth=1.2,
                   label=f"{frac:.1%} of pairs < cutoff")

    ax_hist.set_xlabel("Binder↔peptide PAE (Å)", fontsize=12)
    ax_hist.set_ylabel("Density", fontsize=12)
    ax_hist.set_title("PAE distribution per design", fontsize=13)
    ax_hist.legend(fontsize=7, ncol=2)

    ax_cdf.set_xlabel("Binder↔peptide PAE (Å)", fontsize=12)
    ax_cdf.set_ylabel("Cumulative fraction of pairs", fontsize=12)
    ax_cdf.set_title("CDF — pooled and per design", fontsize=13)
    ax_cdf.legend(fontsize=7, ncol=2)

    fig.suptitle(
        f"Approach 1: Binder↔Peptide PAE Distribution  "
        f"(n={len(npy_paths)} designs, cutoff={cutoff} Å)",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    if show:
        plt.show()
    print(f"  Pooled: {len(pooled):,} pairs, "
          f"{frac:.1%} below {cutoff} Å cutoff, "
          f"mean={pooled.mean():.2f} Å, median={np.median(pooled):.2f} Å")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Approach 2 — Rank stability / sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def approach2_sensitivity(
    npy_paths: list[str],
    pdb_paths: list[str | None],
    labels: list[str],
    nominal_cutoff: float = 12.0,
    cutoff_range: tuple[float, float, float] = (5.0, 25.0, 0.5),
    binder: int | None = None,
    mhc: int | None = None,
    peptide: int | None = None,
    out: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Sweep PAE cutoffs and measure:
    (a) How ipSAE scores change per design.
    (b) Spearman rank correlation of each cutoff vs the nominal cutoff.
    (c) Number of contact pairs as a function of cutoff.

    What to look for
    ----------------
    - Spearman ρ > 0.95 across a wide range: rankings are robust and
      the exact cutoff value is not critical.
    - Score lines that cross at a particular cutoff: designs re-rank
      there; inspect those designs manually.
    - n_contacts elbow: the cutoff where contact count starts rising
      steeply indicates where marginal (high-PAE) pairs enter the score.
      Ideally the nominal cutoff sits before this elbow.
    """
    cutoffs = np.arange(*cutoff_range)

    # score_matrix[design_idx, cutoff_idx]
    score_matrix    = np.zeros((len(npy_paths), len(cutoffs)))
    contacts_matrix = np.zeros((len(npy_paths), len(cutoffs)))

    chain_lengths_list = []
    for i, (npy, pdb) in enumerate(zip(npy_paths, pdb_paths)):
        cl   = resolve_chain_lengths(npy, pdb, binder, mhc, peptide)
        pae  = load_pae_matrix(npy)
        slices = build_chain_slices(cl)
        chain_lengths_list.append(cl)

        for j, c in enumerate(cutoffs):
            r = compute_binder_peptide_ipsae(pae, slices, pae_cutoff=c)
            score_matrix[i, j]    = r["ipsae_binder_peptide"]
            contacts_matrix[i, j] = r["n_contacts"]

    # Spearman rank correlation vs nominal cutoff
    from scipy.stats import spearmanr
    nom_idx = int(np.argmin(np.abs(cutoffs - nominal_cutoff)))
    rho_vs_nominal = np.array([
        spearmanr(score_matrix[:, nom_idx], score_matrix[:, j]).statistic
        if len(npy_paths) > 1 else 1.0
        for j in range(len(cutoffs))
    ])

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax_score   = fig.add_subplot(gs[0, 0])
    ax_rho     = fig.add_subplot(gs[0, 1])
    ax_contacts= fig.add_subplot(gs[1, 0])
    ax_heatmap = fig.add_subplot(gs[1, 1])

    # (a) ipSAE score vs cutoff per design
    for i, label in enumerate(labels):
        ax_score.plot(cutoffs, score_matrix[i, :],
                      color=STYLE["colors"][i % len(STYLE["colors"])],
                      alpha=0.8, linewidth=1.5)
    ax_score.axvline(nominal_cutoff, **{k: v for k, v in STYLE["marker_cutoff"].items()
                                        if k != "label"},
                     label=f"nominal = {nominal_cutoff} Å")
    ax_score.set_xlabel("PAE cutoff (Å)")
    ax_score.set_ylabel("ipSAE binder↔peptide")
    ax_score.set_title("(a) ipSAE vs cutoff")
    ax_score.legend(fontsize=7, ncol=2)
    ax_score.set_ylim(0, 1)

    # (b) Spearman ρ vs nominal cutoff
    ax_rho.plot(cutoffs, rho_vs_nominal, color="#2196f3", linewidth=2)
    ax_rho.axvline(nominal_cutoff, color="#e63946", linestyle="--", linewidth=1.8)
    ax_rho.axhline(0.95, color="grey", linestyle=":", linewidth=1.2,
                   label="ρ = 0.95 threshold")
    ax_rho.fill_between(cutoffs, rho_vs_nominal, 0.95,
                         where=rho_vs_nominal >= 0.95,
                         alpha=0.15, color="#2196f3",
                         label="stable region (ρ ≥ 0.95)")
    ax_rho.set_xlabel("PAE cutoff (Å)")
    ax_rho.set_ylabel("Spearman ρ vs nominal cutoff")
    ax_rho.set_title(f"(b) Rank stability vs nominal cutoff ({nominal_cutoff} Å)")
    ax_rho.set_ylim(-0.1, 1.05)
    ax_rho.legend(fontsize=9)

    # (c) Mean n_contacts vs cutoff
    mean_contacts = contacts_matrix.mean(axis=0)
    ax_contacts.plot(cutoffs, mean_contacts, color="#4caf50", linewidth=2)
    ax_contacts.axvline(nominal_cutoff, color="#e63946", linestyle="--", linewidth=1.8,
                        label=f"nominal = {nominal_cutoff} Å")
    ax_contacts.set_xlabel("PAE cutoff (Å)")
    ax_contacts.set_ylabel("Mean n contact pairs")
    ax_contacts.set_title("(c) Contact pair count vs cutoff")
    ax_contacts.legend(fontsize=9)

    # (d) Score heatmap: designs × cutoffs
    im = ax_heatmap.imshow(
        score_matrix, aspect="auto", cmap="viridis",
        vmin=0, vmax=1,
        extent=[cutoffs[0], cutoffs[-1], len(npy_paths) - 0.5, -0.5],
    )
    ax_heatmap.axvline(nominal_cutoff, color="#e63946", linestyle="--", linewidth=1.8)
    ax_heatmap.set_yticks(range(len(labels)))
#     ax_heatmap.set_yticklabels(labels, fontsize=7)
    ax_heatmap.set_xlabel("PAE cutoff (Å)")
    ax_heatmap.set_title("(d) ipSAE heatmap (designs × cutoffs)")
    plt.colorbar(im, ax=ax_heatmap, label="ipSAE")

    fig.suptitle(
        f"Approach 2: Rank Stability Analysis  "
        f"(n={len(npy_paths)} designs, nominal cutoff={nominal_cutoff} Å)",
        fontsize=14,
    )
    if out:
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    if show:
        plt.show()

    # Print numeric summary
    stable_mask = rho_vs_nominal >= 0.95
    if stable_mask.any():
        stable_range = (cutoffs[stable_mask].min(), cutoffs[stable_mask].max())
        print(f"  Stable range (ρ ≥ 0.95): {stable_range[0]:.1f} – {stable_range[1]:.1f} Å")
    print(f"  ρ at nominal cutoff {nominal_cutoff} Å: 1.000 (reference)")
    for c in [6, 8, 10, 15, 20]:
        if c in cutoffs:
            idx = int(np.argmin(np.abs(cutoffs - c)))
            print(f"  ρ at {c:4.0f} Å: {rho_vs_nominal[idx]:.4f}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Approach 3 — Crystal structure validation
# ─────────────────────────────────────────────────────────────────────────────

def _crystal_ca_coords(
    pdb_path: str,
    chain_id: str,
    slice_last_n: int | None = None,
    slice_first_n: int | None = None,
) -> tuple[list[int], np.ndarray]:
    """
    Extract Cα coordinates for a given chain from a crystal PDB.

    Parameters
    ----------
    pdb_path     : path to PDB file
    chain_id     : chain ID to extract (e.g. "A", "B")
    slice_last_n : if set, return only the LAST n residues of the chain.
                   Use this when the peptide is the C-terminal portion of
                   a merged chain (e.g. chain B = MHC + peptide).
    slice_first_n: if set, return only the FIRST n residues of the chain.

    Returns
    -------
    (residue_numbers, coords_array of shape (N, 3))
    """
    resnums: list[int] = []
    coords:  list[list[float]] = []

    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain_id:
                continue
            if line[12:16].strip() != "CA":
                continue
            # Skip alternate conformers (B, C, ...) but keep blank or 'A'.
            # Do NOT deduplicate by residue number — merged chains (e.g.
            # MHC+peptide in one chain) can legitimately repeat residue
            # numbers, and deduplicating by number would silently drop the
            # peptide residues 1-10 if they clash with MHC residues 1-10.
            altloc = line[16]
            if altloc not in (' ', 'A'):
                continue
            resnum = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            resnums.append(resnum)
            coords.append([x, y, z])

    if slice_last_n is not None:
        resnums = resnums[-slice_last_n:]
        coords  = coords[-slice_last_n:]
    elif slice_first_n is not None:
        resnums = resnums[:slice_first_n]
        coords  = coords[:slice_first_n]

    return resnums, np.array(coords)


def approach3_crystal(
    pred_npy: str,
    crystal_pdb: str,
    binder_chain_crystal: str = "A",
    peptide_chain_crystal: str = "C",
    merged_target_chain: str | None = None,
    binder: int | None = None,
    mhc: int | None = None,
    peptide: int | None = None,
    pred_pdb: str | None = None,
    dist_cutoff: float = 8.0,
    cutoff_range: tuple[float, float, float] = (4.0, 25.0, 0.5),
    nominal_cutoff: float = 12.0,
    out: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Validate PAE cutoff against Cα distances in a crystal structure.

    For each PAE cutoff, a binder↔peptide residue pair is "predicted
    positive" if PAE < cutoff. Ground truth positive = Cα distance
    < dist_cutoff (default 8 Å) in the crystal PDB.

    Compute precision, recall, F1 across the cutoff sweep and report
    the empirically optimal cutoff. Also show the PAE vs distance
    scatter to visualise calibration.

    Recommended usage
    -----------------
    Use 9O5S (mart1-3 crystal) with its AF2 initial guess prediction:
      --pred-npy  9O5S_af2pred_pae.npy
      --pred-pdb  9O5S_af2pred.pdb           (for chain length auto-detect)
      --crystal-pdb 9O5S_crystal.pdb
      --binder-chain-crystal A               (adjust to actual chain IDs in 9O5S)
      --peptide-chain-crystal C

    What to look for
    ----------------
    - F1 peak: the empirically optimal cutoff for this system.
      If it is close to 12 Å, the default is well-calibrated.
    - Precision/recall tradeoff: low cutoff = high precision / low recall
      (only confident contacts included, many missed).
      High cutoff = high recall / low precision (many false positives).
    - PAE vs Cα distance scatter: well-calibrated AF2 should show
      PAE < 8 Å clustering tightly at Cα distances < 8 Å.
    - Diagonal relationship: if PAE correlates with physical distance,
      the cutoff acts as a reliable proxy for structural contact.
    """
    cutoffs = np.arange(*cutoff_range)

    # Load AF2 PAE matrix
    cl  = resolve_chain_lengths(pred_npy, pred_pdb, binder, mhc, peptide)
    pae = load_pae_matrix(pred_npy)
    slices = build_chain_slices(cl)
    pae_bp = pae[slices["binder"], slices["peptide"]]  # (n_binder, n_peptide)

    n_b = cl["binder"]
    n_p = cl["peptide"]
    print(f"PAE submatrix: {n_b} binder × {n_p} peptide = {n_b*n_p} pairs")

    # Load crystal Cα coordinates
    _, binder_coords = _crystal_ca_coords(crystal_pdb, binder_chain_crystal)

    if merged_target_chain is not None:
        # Layout 2: crystal chain B = MHC+peptide; peptide = last n_p residues
        print(f"  Crystal layout: merged target chain '{merged_target_chain}' — "
              f"extracting last {n_p} residues as peptide")
        _, peptide_coords = _crystal_ca_coords(
            crystal_pdb, merged_target_chain, slice_last_n=n_p
        )
    else:
        # Layout 1: peptide has its own chain
        _, peptide_coords = _crystal_ca_coords(crystal_pdb, peptide_chain_crystal)

    if len(binder_coords) != n_b:
        print(
            f"WARNING: crystal binder chain has {len(binder_coords)} Cα atoms "
            f"but PAE binder length is {n_b}. "
            f"Truncating to min({len(binder_coords)}, {n_b})."
        )
        n_use_b = min(len(binder_coords), n_b)
        binder_coords = binder_coords[:n_use_b]
        pae_bp        = pae_bp[:n_use_b, :]
        n_b           = n_use_b

    if len(peptide_coords) != n_p:
        print(
            f"WARNING: crystal peptide chain has {len(peptide_coords)} Cα atoms "
            f"but PAE peptide length is {n_p}. "
            f"Truncating to min({len(peptide_coords)}, {n_p})."
        )
        n_use_p = min(len(peptide_coords), n_p)
        peptide_coords = peptide_coords[:n_use_p]
        pae_bp         = pae_bp[:, :n_use_p]
        n_p            = n_use_p

    # Compute all-pairs Cα distance matrix
    dist_matrix = np.sqrt(
        ((binder_coords[:, None, :] - peptide_coords[None, :, :]) ** 2).sum(axis=2)
    )   # shape (n_b, n_p)

    crystal_contact = dist_matrix < dist_cutoff  # ground truth

    n_true_contacts = int(crystal_contact.sum())
    print(f"Crystal contacts (Cα < {dist_cutoff} Å): {n_true_contacts} / {n_b*n_p} pairs "
          f"({100*n_true_contacts/(n_b*n_p):.1f}%)")

    # Sweep cutoffs
    precision_vals = np.zeros(len(cutoffs))
    recall_vals    = np.zeros(len(cutoffs))
    f1_vals        = np.zeros(len(cutoffs))
    n_pred_vals    = np.zeros(len(cutoffs))

    for j, c in enumerate(cutoffs):
        pred_contact = pae_bp < c
        tp = float(np.sum(pred_contact & crystal_contact))
        fp = float(np.sum(pred_contact & ~crystal_contact))
        fn = float(np.sum(~pred_contact & crystal_contact))
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*p*r/(p+r)   if (p + r)   > 0 else 0.0
        precision_vals[j] = p
        recall_vals[j]    = r
        f1_vals[j]        = f1
        n_pred_vals[j]    = int(pred_contact.sum())

    best_idx = int(np.argmax(f1_vals))
    best_c   = cutoffs[best_idx]
    nom_idx  = int(np.argmin(np.abs(cutoffs - nominal_cutoff)))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax_prf    = fig.add_subplot(gs[0, 0])
    ax_f1     = fig.add_subplot(gs[0, 1])
    ax_scatter= fig.add_subplot(gs[1, 0])
    ax_matrix = fig.add_subplot(gs[1, 1])

    # (a) Precision, Recall, F1 vs cutoff
    ax_prf.plot(cutoffs, precision_vals, color="#2196f3", linewidth=2,
                label="Precision")
    ax_prf.plot(cutoffs, recall_vals,    color="#4caf50", linewidth=2,
                label="Recall")
    ax_prf.plot(cutoffs, f1_vals,        color="#ff9800", linewidth=2.5,
                label="F1")
    ax_prf.axvline(nominal_cutoff, color="#e63946", linestyle="--",
                   linewidth=1.8, label=f"nominal = {nominal_cutoff} Å")
    ax_prf.axvline(best_c, color="#9c27b0", linestyle=":",
                   linewidth=1.8, label=f"best F1 = {best_c:.1f} Å")
    ax_prf.set_xlabel("PAE cutoff (Å)")
    ax_prf.set_ylabel("Score")
    ax_prf.set_title("(a) Precision / Recall / F1 vs cutoff")
    ax_prf.legend(fontsize=9)
    ax_prf.set_ylim(0, 1.05)

    # (b) F1 detail with annotation
    ax_f1.plot(cutoffs, f1_vals, color="#ff9800", linewidth=2.5)
    ax_f1.axvline(nominal_cutoff, color="#e63946", linestyle="--",
                  linewidth=1.8, label=f"nominal {nominal_cutoff} Å  F1={f1_vals[nom_idx]:.3f}")
    ax_f1.axvline(best_c, color="#9c27b0", linestyle=":",
                  linewidth=1.8, label=f"optimal {best_c:.1f} Å  F1={f1_vals[best_idx]:.3f}")
    ax_f1.scatter([best_c], [f1_vals[best_idx]], color="#9c27b0", s=80, zorder=5)
    ax_f1.scatter([nominal_cutoff], [f1_vals[nom_idx]], color="#e63946", s=80, zorder=5)
    ax_f1.set_xlabel("PAE cutoff (Å)")
    ax_f1.set_ylabel("F1 score")
    ax_f1.set_title("(b) F1 detail — optimal vs nominal cutoff")
    ax_f1.legend(fontsize=9)

    # (c) PAE vs Cα distance scatter (coloured by ground truth contact)
    flat_pae  = pae_bp.flatten()
    flat_dist = dist_matrix.flatten()
    flat_gt   = crystal_contact.flatten()

    ax_scatter.scatter(flat_dist[~flat_gt], flat_pae[~flat_gt],
                       c="#90a4ae", alpha=0.3, s=8, label="non-contact (crystal)")
    ax_scatter.scatter(flat_dist[flat_gt],  flat_pae[flat_gt],
                       c="#e63946", alpha=0.6, s=12, label=f"contact (Cα < {dist_cutoff} Å)")
    ax_scatter.axhline(nominal_cutoff, color="#e63946", linestyle="--",
                       linewidth=1.5, label=f"PAE cutoff = {nominal_cutoff} Å")
    ax_scatter.axvline(dist_cutoff, color="#333333", linestyle=":",
                       linewidth=1.2, label=f"Cα cutoff = {dist_cutoff} Å")
    ax_scatter.set_xlabel("Cα distance in crystal (Å)")
    ax_scatter.set_ylabel("Predicted PAE (Å)")
    ax_scatter.set_title("(c) PAE calibration vs crystal Cα distance")
    ax_scatter.legend(fontsize=8)

    # (d) Heatmap: PAE matrix overlaid with crystal contact mask
    vmax = 20
    im = ax_matrix.imshow(pae_bp, cmap="Blues_r", vmin=0, vmax=vmax,
                          aspect="auto", origin="upper")
    # Overlay crystal contacts as green squares
    binder_idx, peptide_idx = np.where(crystal_contact)
    ax_matrix.scatter(peptide_idx, binder_idx, c="#4caf50", s=6,
                      alpha=0.8, label=f"crystal contact (Cα < {dist_cutoff} Å)")
    ax_matrix.set_xlabel("Peptide residue index")
    ax_matrix.set_ylabel("Binder residue index")
    ax_matrix.set_title("(d) PAE heatmap + crystal contacts (green)")
    plt.colorbar(im, ax=ax_matrix, label="PAE (Å)", shrink=0.8)
    ax_matrix.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Approach 3: Crystal Validation — 9O5S (mart1-3)  "
        f"[nominal={nominal_cutoff} Å, dist_cutoff={dist_cutoff} Å]",
        fontsize=14,
    )
    if out:
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    if show:
        plt.show()

    # Print numeric summary
    print(f"\n  === Approach 3 Summary ===")
    print(f"  Optimal cutoff (max F1): {best_c:.1f} Å")
    print(f"    Precision: {precision_vals[best_idx]:.3f}")
    print(f"    Recall:    {recall_vals[best_idx]:.3f}")
    print(f"    F1:        {f1_vals[best_idx]:.3f}")
    print(f"    n_pred:    {int(n_pred_vals[best_idx])}")
    print(f"\n  Nominal cutoff ({nominal_cutoff} Å):")
    print(f"    Precision: {precision_vals[nom_idx]:.3f}")
    print(f"    Recall:    {recall_vals[nom_idx]:.3f}")
    print(f"    F1:        {f1_vals[nom_idx]:.3f}")
    print(f"    n_pred:    {int(n_pred_vals[nom_idx])}")
    delta_f1 = f1_vals[best_idx] - f1_vals[nom_idx]
    if delta_f1 < 0.02:
        print(f"\n  ✓ Nominal cutoff within {delta_f1:.3f} F1 of optimal — well-calibrated.")
    else:
        print(f"\n  ⚠ Nominal cutoff is {delta_f1:.3f} F1 below optimal. "
              f"Consider using {best_c:.1f} Å instead.")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="approach", required=True)

    # ── Common arguments ─────────────────────────────────────────────────────
    def add_common(p):
        p.add_argument("--npy",     nargs="+", required=True,
                       help="PAE .npy file(s)")
        p.add_argument("--pdb",     nargs="+", default=[],
                       help="AF2 output PDB(s) for chain length auto-detection "
                            "(must match --npy order)")
        p.add_argument("--labels",  nargs="+", default=[],
                       help="Design labels (defaults to filenames)")
        p.add_argument("--binder",  type=int, default=None)
        p.add_argument("--mhc",     type=int, default=None)
        p.add_argument("--peptide", type=int, default=None)
        p.add_argument("--cutoff",  type=float, default=12.0,
                       help="Nominal PAE cutoff in Å (default: 12.0)")
        p.add_argument("--out",     type=str,   default=None)

    # ── Approach 1 ───────────────────────────────────────────────────────────
    p1 = sub.add_parser("distribution", help="PAE distribution plot")
    add_common(p1)

    # ── Approach 2 ───────────────────────────────────────────────────────────
    p2 = sub.add_parser("sensitivity", help="Rank stability across cutoffs")
    add_common(p2)

    # ── Approach 3 ───────────────────────────────────────────────────────────
    p3 = sub.add_parser("crystal", help="Crystal structure validation (9O5S)")
    add_common(p3)
    p3.add_argument("--pred-pdb",    type=str, default=None,
                    help="AF2 prediction PDB (for chain length auto-detection)")
    p3.add_argument("--crystal-pdb", type=str, required=True,
                    help="Crystal structure PDB (9O5S)")
    p3.add_argument("--binder-chain-crystal",  type=str, default="A",
                    help="Chain ID of binder in crystal PDB (default: A)")
    p3.add_argument("--peptide-chain-crystal", type=str, default="C",
                    help="Chain ID of peptide in crystal PDB (default: C) "
                    "Ignored if --merged-target-chain is set.")
    p3.add_argument("--merged-target-chain", type=str, default=None,
                    help="Set this to the chain ID that contains MHC + peptide "
                         "concatenated (e.g. 'B' for the 9O5S AF2 output). "
                         "The peptide is extracted as the last --peptide residues "
                         "of that chain. Overrides --peptide-chain-crystal.")
    p3.add_argument("--dist-cutoff", type=float, default=8.0,
                    help="Cα distance threshold for 'contact' in crystal (default: 8.0 Å)")

    args = parser.parse_args()

    # Resolve PDB list (pad with None if shorter than npy list)
    pdb_list = list(args.pdb) + [None] * max(0, len(args.npy) - len(args.pdb))
    labels   = list(args.labels) if args.labels else [Path(n).stem for n in args.npy]

    if args.approach == "distribution":
        out = args.out  # None = don't save unless --out specified
        approach1_distribution(
            args.npy, pdb_list, labels,
            cutoff=args.cutoff,
            binder=args.binder, mhc=args.mhc, peptide=args.peptide,
            out=out, show=False,
        )

    elif args.approach == "sensitivity":
        out = args.out  # None = don't save unless --out specified
        approach2_sensitivity(
            args.npy, pdb_list, labels,
            nominal_cutoff=args.cutoff,
            binder=args.binder, mhc=args.mhc, peptide=args.peptide,
            out=out, show=False,
        )

    elif args.approach == "crystal":
        if len(args.npy) != 1:
            parser.error("crystal approach requires exactly one --npy file")
        out = args.out  # None = don't save unless --out specified
        approach3_crystal(
            pred_npy=args.npy[0],
            crystal_pdb=args.crystal_pdb,
            binder_chain_crystal=args.binder_chain_crystal,
            peptide_chain_crystal=args.peptide_chain_crystal,
            merged_target_chain=args.merged_target_chain,
            binder=args.binder, mhc=args.mhc, peptide=args.peptide,
            pred_pdb=args.pred_pdb or (pdb_list[0]),
            dist_cutoff=args.dist_cutoff,
            nominal_cutoff=args.cutoff,
            out=out, show=False,
        )


if __name__ == "__main__":
    _cli()
