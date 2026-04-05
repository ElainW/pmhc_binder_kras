"""
cms_scoring.py
Computes ContactMolecularSurface (CMS) between each peptide residue and the binder,
for all PDBs in a directory. Outputs a TSV with one CMS value per peptide position
per design.

Chain conventions (truncated MHC input):
  Chain A = binder
  Chain B = truncated MHC (α1α2, ~180 res) + peptide (last 9 res)

Usage:
  python cms_scoring.py \
      --pdb_dir /n/groups/marks/users/aaron/pmhc/round2/af2_filtered/ \
      --out_csv  /n/groups/marks/users/aaron/pmhc/round2/cms_scores.tsv \
      --peptide_len 9 \
      --binder_chain A \
      --pmhc_chain B
"""

import os
import argparse
import pandas as pd
import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.select.residue_selector import (
    ChainSelector,
    ResidueIndexSelector,
)
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.rosetta.core.scoring import ScoreFunction
from pyrosetta.rosetta.core.scoring import get_score_function
from pyrosetta.rosetta.protocols.simple_filters import ContactMolecularSurfaceFilter


def init_pyrosetta():
    pyrosetta.init(
        "-beta_nov16 "                    # match the XML's score function
        "-mute all "                      # suppress Rosetta output noise
        "-ignore_unrecognized_res true "  # tolerate non-standard residues
        "-ignore_zero_occupancy false",
        silent=True
    )


def get_peptide_pose_indices(pose, pmhc_chain: str, peptide_len: int) -> list[int]:
    """
    Returns 1-indexed Rosetta pose residue numbers for the peptide.
    The peptide is the last `peptide_len` residues of pmhc_chain.
    """
    chain_id = ord(pmhc_chain) - ord('A') + 1  # Rosetta chain index (1-based)

    # collect all pose residue indices belonging to pmhc_chain
    chain_residues = []
    for i in range(1, pose.total_residue() + 1):
        if pose.pdb_info().chain(i) == pmhc_chain:
            chain_residues.append(i)

    if len(chain_residues) < peptide_len:
        raise ValueError(
            f"Chain {pmhc_chain} has only {len(chain_residues)} residues, "
            f"fewer than peptide_len={peptide_len}"
        )

    # peptide = last peptide_len residues of that chain
    return chain_residues[-peptide_len:]


def score_cms_for_residue(pose, binder_chain: str, peptide_res_idx: int,
                           distance_weight: float = 0.5) -> float:
    """
    Computes CMS between a single target residue (peptide_res_idx, 1-indexed pose number)
    and the entire binder chain. Mirrors the XML filter exactly.

    The XML does:
      target_selector  = Slice (single residue from chainB)
      binder_selector  = chainA
      distance_weight  = 0.5
    """
    from pyrosetta.rosetta.protocols.simple_filters import ContactMolecularSurfaceFilter

    # selectors
    target_sel = ResidueIndexSelector(str(peptide_res_idx))
    binder_sel = ChainSelector(binder_chain)

    cms_filter = ContactMolecularSurfaceFilter()
    cms_filter.set_distance_weight(distance_weight)
    cms_filter.set_target_selector(target_sel)
    cms_filter.set_binder_selector(binder_sel)
    cms_filter.set_verbose(False)

    # report() returns the float score regardless of threshold (confidence=0 equivalent)
    return cms_filter.report_sm(pose)


def score_pdb(pdb_path: str, binder_chain: str, pmhc_chain: str,
              peptide_len: int, distance_weight: float = 0.5) -> dict:
    """
    Scores a single PDB. Returns a dict with keys:
      description, cms_p1 ... cms_p{peptide_len}, cms_peptide_total
    """
    pose = pose_from_pdb(pdb_path)
    peptide_indices = get_peptide_pose_indices(pose, pmhc_chain, peptide_len)

    result = {"description": os.path.splitext(os.path.basename(pdb_path))[0]}

    cms_values = []
    for pos, res_idx in enumerate(peptide_indices, start=1):
        cms = score_cms_for_residue(pose, binder_chain, res_idx, distance_weight)
        result[f"cms_p{pos}"] = round(cms, 4)
        cms_values.append(cms)

    # total CMS across the whole peptide (useful for overall contact ranking)
    result["cms_peptide_total"] = round(sum(cms_values), 4)

    # also store the P4 score separately for convenience (the neomutant position)
    result["cms_p4_neoepitope"] = result["cms_p4"]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir",      required=True)
    parser.add_argument("--out_csv",      required=True)
    parser.add_argument("--peptide_len",  type=int, default=9)
    parser.add_argument("--binder_chain", default="A")
    parser.add_argument("--pmhc_chain",   default="B")
    parser.add_argument("--distance_weight", type=float, default=0.5)
    args = parser.parse_args()

    init_pyrosetta()

    pdb_files = sorted([
        os.path.join(args.pdb_dir, f)
        for f in os.listdir(args.pdb_dir) if f.endswith(".pdb")
    ])
    print(f"Scoring {len(pdb_files)} PDBs...")

    records = []
    for i, pdb_path in enumerate(pdb_files):
        try:
            rec = score_pdb(
                pdb_path,
                binder_chain=args.binder_chain,
                pmhc_chain=args.pmhc_chain,
                peptide_len=args.peptide_len,
                distance_weight=args.distance_weight,
            )
            records.append(rec)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(pdb_files)} done")
        except Exception as e:
            print(f"  WARNING: failed on {pdb_path}: {e}")

    df = pd.DataFrame(records)
    # column order: description, cms_p1..p9, cms_peptide_total, cms_p4_neoepitope
    cols = (["description"]
            + [f"cms_p{i}" for i in range(1, args.peptide_len + 1)]
            + ["cms_peptide_total", "cms_p4_neoepitope"])
    df = df[cols]
    df.to_csv(args.out_csv, sep="\t", index=False)
    print(f"Written to {args.out_csv}")


if __name__ == "__main__":
    main()
