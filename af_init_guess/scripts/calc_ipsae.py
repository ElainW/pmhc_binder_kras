"""
pmhci_ipsae.py
==============
Compute binder<->peptide ipSAE for minibinder + pMHCI complexes.

Restricts inter-chain PAE scoring to binder<->peptide pairs only,
avoiding contamination from binder<->MHC contacts that dominate
the standard ipAE / ipTM metrics.

Assumes a two-chain input to af2_initial_guess:
  Chain 1: designed minibinder
  Chain 2: truncated MHC alpha-chain (peptide-binding domain only,
           no beta-2-microglobulin) + antigen peptide

Requires the full N×N PAE matrix saved from af2_initial_guess predict.py
as a .npy file (see -pae_outdir argument in predict.py).

Dependencies: numpy
    pip install numpy

Reference:
    Schaeffer & Dunbrack (2025) "Res ipSAE loquunt: What's wrong with
    AlphaFold's ipTM score and how to fix it"
    bioRxiv 10.1101/2025.02.10.637595

Usage:
    python pmhci_ipsae.py design_af2pred_pae.npy \\
        --binder 72 --mhc 180 --peptide 9

    # Or import as a library:
    from pmhci_ipsae import score_binder_peptide_ipsae, print_summary
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# PAE loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pae_matrix(npy_path: str | Path) -> np.ndarray:
    """
    Load the N×N PAE matrix from a .npy file saved by predict.py.

    Parameters
    ----------
    npy_path : path to *_pae.npy file written by af2_initial_guess predict.py

    Returns
    -------
    np.ndarray of shape (N, N), values in Angstroms.
    """
    pae = np.load(str(npy_path))

    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(
            f"PAE matrix has unexpected shape {pae.shape}; expected (N, N)."
        )
    return pae


# ─────────────────────────────────────────────────────────────────────────────
# Chain indexing
# ─────────────────────────────────────────────────────────────────────────────

def build_chain_slices(chain_lengths: OrderedDict) -> dict[str, slice]:
    """
    Convert an ordered mapping of {chain_name: n_residues} into residue-index
    slices into the PAE matrix.

    The order MUST match the concatenation order used in af2_initial_guess:
    binder first, then the merged target chain (truncated MHC + peptide).

    Parameters
    ----------
    chain_lengths : OrderedDict
        af2_initial_guess concatenation order::

            OrderedDict([
                ("binder",   72),   # chain 1 - designed minibinder
                ("MHC",     180),   # chain 2 - truncated MHC alpha-chain
                                    #           (peptide-binding domain only,
                                    #            no beta-2-microglobulin)
                ("peptide",   9),   #           antigen peptide (appended to MHC)
            ])

    Returns
    -------
    dict mapping chain name -> slice(start, end)
    """
    slices: dict[str, slice] = {}
    offset = 0
    for name, length in chain_lengths.items():
        slices[name] = slice(offset, offset + length)
        offset += length
    return slices


# ─────────────────────────────────────────────────────────────────────────────
# ipSAE core
# ─────────────────────────────────────────────────────────────────────────────

def _ipsae_one_direction(
    pae_sub: np.ndarray,
    pae_cutoff: float,
    min_d0: float,
) -> tuple[float, int, float]:
    """
    ipSAE for one direction of an inter-chain PAE submatrix.

    Following Schaeffer & Dunbrack (2025):
      1. Flatten submatrix; keep pairs where PAE < pae_cutoff.
      2. Derive d0 from the number of contact pairs (not chain length).
      3. Compute TM-score-like mean over those contact pairs.

    Returns
    -------
    (score, n_contacts, d0)
    """
    flat     = pae_sub.flatten()
    contacts = flat[flat < pae_cutoff]
    n        = len(contacts)

    if n == 0:
        return 0.0, 0, min_d0

    d0    = max(1.24 * max(n - 15, 1) ** (1.0 / 3.0) - 1.8, min_d0)
    score = float(np.mean(1.0 / (1.0 + (contacts / d0) ** 2)))
    return score, n, d0


def compute_binder_peptide_ipsae(
    pae: np.ndarray,
    chain_slices: dict[str, slice],
    pae_cutoff: float = 12.0,
    min_d0: float = 0.5,
) -> dict:
    """
    Compute ipSAE restricted to binder<->peptide residue pairs only.

    Score = max of both directional scores (aligned-on-binder/scored-on-peptide
    and vice versa), following the ipTM convention.

    Parameters
    ----------
    pae          : full N×N PAE matrix (Angstroms)
    chain_slices : from build_chain_slices()
    pae_cutoff   : Angstrom threshold defining a "contact" pair (default 12 A)
    min_d0       : minimum d0 to avoid numerical issues (default 0.5 A)

    Returns
    -------
    dict with keys:
        ipsae_binder_peptide  - primary metric, max of both directions [0, 1]
        ipsae_bp              - score aligned on binder, scored on peptide
        ipsae_pb              - score aligned on peptide, scored on binder
        n_contacts            - contact pairs in the winning direction
        d0                    - d0 value (Angstroms) in the winning direction
        mean_pae_bp           - mean PAE over ALL binder<->peptide pairs (= ipAE)
        mean_pae_contact      - mean PAE restricted to contact pairs only
    """
    sl_b = chain_slices["binder"]
    sl_p = chain_slices["peptide"]

    pae_bp = pae[sl_b, sl_p]   # aligned on binder, scored on peptide
    pae_pb = pae[sl_p, sl_b]   # aligned on peptide, scored on binder

    s_bp, n_bp, d0_bp = _ipsae_one_direction(pae_bp, pae_cutoff, min_d0)
    s_pb, n_pb, d0_pb = _ipsae_one_direction(pae_pb, pae_cutoff, min_d0)

    if s_bp >= s_pb:
        best, n_best, d0_best, pae_best = s_bp, n_bp, d0_bp, pae_bp
    else:
        best, n_best, d0_best, pae_best = s_pb, n_pb, d0_pb, pae_pb

    cv           = pae_best.flatten()[pae_best.flatten() < pae_cutoff]
    mean_contact = float(np.mean(cv)) if len(cv) else float("nan")

    return {
        "ipsae_binder_peptide": round(best, 4),
        "ipsae_bp":             round(s_bp, 4),
        "ipsae_pb":             round(s_pb, 4),
        "n_contacts":           n_best,
        "d0":                   round(d0_best, 3),
        "mean_pae_bp":          round(float(np.mean(pae_bp)), 3),
        "mean_pae_contact":     round(mean_contact, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top-level scoring function
# ─────────────────────────────────────────────────────────────────────────────

def score_binder_peptide_ipsae(
    npy_path: str | Path,
    chain_lengths: OrderedDict,
    pae_cutoff: float = 12.0,
) -> dict:
    """
    Load a PAE .npy file and compute binder<->peptide ipSAE.

    Parameters
    ----------
    npy_path      : path to *_pae.npy saved by predict.py
    chain_lengths : OrderedDict of {chain_name: n_residues} in af2_initial_guess
                    concatenation order: binder, MHC (truncated), peptide.
    pae_cutoff    : contact threshold in Angstroms (default 12 A)

    Returns
    -------
    dict from compute_binder_peptide_ipsae()
    """
    pae    = load_pae_matrix(npy_path)
    slices = build_chain_slices(chain_lengths)

    total_expected = sum(chain_lengths.values())
    if pae.shape[0] != total_expected:
        raise ValueError(
            f"PAE matrix size {pae.shape[0]} does not match sum of chain "
            f"lengths {total_expected}. Check chain_lengths order and values.\n"
            f"  binder={chain_lengths['binder']}, "
            f"MHC={chain_lengths['MHC']}, "
            f"peptide={chain_lengths['peptide']}, "
            f"total={total_expected}"
        )

    return compute_binder_peptide_ipsae(pae, slices, pae_cutoff)


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict, design_name: str = "") -> None:
    """Pretty-print binder<->peptide ipSAE results."""
    header = f"  {design_name}" if design_name else ""
    print(f"\n{'─'*55}")
    print(f"pMHCI Binder<->Peptide ipSAE{header}")
    print(f"{'─'*55}")
    print(f"  ipSAE binder<->peptide : {results['ipsae_binder_peptide']:.4f}  <- primary metric")
    print(f"  ipSAE (bp direction)   : {results['ipsae_bp']:.4f}")
    print(f"  ipSAE (pb direction)   : {results['ipsae_pb']:.4f}")
    print(f"  n contact pairs        : {results['n_contacts']:4d}")
    print(f"  d0                     : {results['d0']:.3f} A")
    print(f"  mean PAE (all bp)      : {results['mean_pae_bp']:.2f} A  (= ipAE binder<->peptide)")
    print(f"  mean PAE (contacts)    : {results['mean_pae_contact']:.2f} A")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Compute binder<->peptide ipSAE for a pMHCI + minibinder complex.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("npy",        help="*_pae.npy file saved by af2_initial_guess predict.py")
    parser.add_argument("--binder",   type=int, required=True, help="Minibinder length (residues)")
    parser.add_argument("--mhc",      type=int, required=True, help="Truncated MHC alpha-chain length (no B2M)")
    parser.add_argument("--peptide",  type=int, required=True, help="Antigen peptide length")
    parser.add_argument("--cutoff",   type=float, default=12.0, help="PAE contact cutoff (Angstroms)")
    parser.add_argument("--name",     type=str,   default="",   help="Design label for output")
    args = parser.parse_args()

    # af2_initial_guess order: binder first, then truncated MHC + peptide
    chain_lengths = OrderedDict([
        ("binder",  args.binder),
        ("MHC",     args.mhc),
        ("peptide", args.peptide),
    ])

    results = score_binder_peptide_ipsae(
        npy_path=args.npy,
        chain_lengths=chain_lengths,
        pae_cutoff=args.cutoff,
    )
    print_summary(results, design_name=args.name or Path(args.npy).stem)


if __name__ == "__main__":
    _cli()