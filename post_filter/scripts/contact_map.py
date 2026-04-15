"""
contact_map.py

Reads pre-split pMHC fold PDBs from split_pmhc_fold_pdbs.py:
  {split_dir}/{design}_chainA.pdb  — binder (chain A, renumbered from 1)
  {split_dir}/{design}_chainB.pdb  — MHC + peptide (chain B, renumbered from 1,
                                       peptide = last 9 residues)

Contact definitions matching PyMOL conventions:
  all_contacts:        any heavy atom pair < 4.0 Å  (PyMOL mode=0)
  polar_contacts:      N/O donor-acceptor pair < 4 Å  (PyMOL mode=2 approximation)
  hydrophobic_contacts: peptide residue is hydrophobic AND any heavy atom pair < 4.0 Å

Usage:
    python contact_map.py \
        --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
        --split_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/ \
        --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser

DELTA_PAE_CUTOFF         = -0.5
PEPTIDE_SEQ              = 'VVGADGVGK'
PEPTIDE_LEN              = 9
ALL_CONTACT_DIST         = 4.0   # Å — any heavy atom (PyMOL mode=0)
POLAR_CONTACT_DIST       = 4.0   # Å — N/O donor-acceptor (PyMOL mode=2), also include salt bridges
HYDROPHOBIC_CONTACT_DIST = 4.0   # Å — hydrophobic residue contact

# Polar atom elements for H-bond donor/acceptor detection
POLAR_ELEMENTS = {'N', 'O'}

# Standard hydrophobic residue set (3-letter codes)
HYDROPHOBIC_RESIDUES = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO', 'TYR'}

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_residues(pdb_path: str, chain_id: str) -> list:
    """Return ATOM residues from a given chain."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('s', pdb_path)
    residues = []
    for model in struct:
        for chain in model:
            if chain.id == chain_id:
                for res in chain:
                    if res.id[0] == ' ':
                        residues.append(res)
        break
    return residues


# ── Contact computation ───────────────────────────────────────────────────────

def compute_contacts(binder_residues: list, peptide_residues: list):
    """
    Three contact types matching PyMOL selections:

    all_contacts:
        Any heavy atom pair < 4.0 Å (PyMOL mode=0).

    polar_contacts:
        N/O donor-acceptor pair < 4.0 Å (PyMOL mode=2 approximation).
        Reports the closest polar pair per residue pair.

    hydrophobic_contacts:
        Peptide residue is in HYDROPHOBIC_RESIDUES AND any heavy atom
        pair < 4.0 Å (PyMOL: resn ALA+VAL+... within 4.0).

    Returns:
        all_contacts        : list of (bi, pi, min_dist, b_aa, p_aa)
        polar_contacts      : list of (bi, pi, min_dist, b_atom, p_atom, b_aa, p_aa)
        hydrophobic_contacts: list of (bi, pi, min_dist, b_aa, p_aa)
    """
    all_contacts         = []
    polar_contacts       = []
    hydrophobic_contacts = []

    for bi, b_res in enumerate(binder_residues):
        b_aa    = THREE_TO_ONE.get(b_res.resname, 'X')
        b_heavy = [(a, a.get_vector().get_array())
                   for a in b_res.get_atoms()
                   if a.element not in ('H', '') and a.element is not None]
        if not b_heavy:
            continue
        b_coords = np.array([c for _, c in b_heavy])

        for pi, p_res in enumerate(peptide_residues):
            p_aa    = THREE_TO_ONE.get(p_res.resname, 'X')
            p_heavy = [(a, a.get_vector().get_array())
                       for a in p_res.get_atoms()
                       if a.element not in ('H', '') and a.element is not None]
            if not p_heavy:
                continue
            p_coords = np.array([c for _, c in p_heavy])

            # Distance matrix: shape (n_b_atoms, n_p_atoms)
            dists    = np.linalg.norm(b_coords[:, None, :] - p_coords[None, :, :], axis=-1)
            min_dist = float(dists.min())

            # ── All contacts ──────────────────────────────────────────────
            if min_dist <= ALL_CONTACT_DIST:
                all_contacts.append((bi, pi, round(min_dist, 3), b_aa, p_aa))

            # ── Polar contacts ────────────────────────────────────────────
            b_polar = [(a, c) for a, c in b_heavy if a.element in POLAR_ELEMENTS]
            p_polar = [(a, c) for a, c in p_heavy if a.element in POLAR_ELEMENTS]
            if b_polar and p_polar:
                bp_coords   = np.array([c for _, c in b_polar])
                pp_coords   = np.array([c for _, c in p_polar])
                polar_dists = np.linalg.norm(
                    bp_coords[:, None, :] - pp_coords[None, :, :], axis=-1
                )
                min_polar_dist = float(polar_dists.min())
                if min_polar_dist <= POLAR_CONTACT_DIST:
                    min_idx     = np.unravel_index(polar_dists.argmin(), polar_dists.shape)
                    b_atom_name = b_polar[min_idx[0]][0].get_name().strip()
                    p_atom_name = p_polar[min_idx[1]][0].get_name().strip()
                    polar_contacts.append((
                        bi, pi,
                        round(min_polar_dist, 3),
                        b_atom_name, p_atom_name,
                        b_aa, p_aa
                    ))

            # ── Hydrophobic contacts ──────────────────────────────────────
            if p_res.resname in HYDROPHOBIC_RESIDUES and min_dist <= HYDROPHOBIC_CONTACT_DIST:
                hydrophobic_contacts.append((bi, pi, round(min_dist, 3), b_aa, p_aa))

    return all_contacts, polar_contacts, hydrophobic_contacts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_scores', required=True)
    parser.add_argument('--split_dir',     required=True,
                        help='Directory containing {design}_chainA.pdb and '
                             '{design}_chainB.pdb from split_pmhc_fold_pdbs.py')
    parser.add_argument('--out_dir',       required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.merged_scores, sep='\t')
    passing = df[df['delta_pae_int_peptide'] < DELTA_PAE_CUTOFF].copy()
    designs = passing['design'].tolist()
    print(f"Processing {len(designs)} designs...")

    all_contact_rows   = []
    polar_contact_rows = []
    hydro_contact_rows = []   # separate list — not merged with polar
    summary_rows       = []

    # First pass: determine max binder length for frequency matrix sizing
    max_binder_len = 0
    valid_designs  = []
    for design in designs:
        chain_a_path = os.path.join(args.split_dir, f'{design}_chainA.pdb')
        chain_b_path = os.path.join(args.split_dir, f'{design}_chainB.pdb')
        if not os.path.exists(chain_a_path) or not os.path.exists(chain_b_path):
            print(f"  ERROR: split PDBs not found for {design} — skipping")
            continue
        binder_res = get_residues(chain_a_path, 'A')
        if not binder_res:
            print(f"  ERROR: empty chainA for {design} — skipping")
            continue
        max_binder_len = max(max_binder_len, len(binder_res))
        valid_designs.append(design)

    if max_binder_len == 0:
        print("ERROR: no valid designs found")
        return

    # Frequency matrices (rows = binder positions 0-indexed, cols = peptide positions)
    all_freq   = np.zeros((max_binder_len, PEPTIDE_LEN), dtype=float)
    polar_freq = np.zeros((max_binder_len, PEPTIDE_LEN), dtype=float)
    hydro_freq = np.zeros((max_binder_len, PEPTIDE_LEN), dtype=float)
    n_contrib  = np.zeros((max_binder_len, PEPTIDE_LEN), dtype=int)
    n_processed = 0

    for design in valid_designs:
        chain_a_path = os.path.join(args.split_dir, f'{design}_chainA.pdb')
        chain_b_path = os.path.join(args.split_dir, f'{design}_chainB.pdb')

        binder_res = get_residues(chain_a_path, 'A')
        pmhc_res   = get_residues(chain_b_path, 'B')
        pep_res    = pmhc_res[-PEPTIDE_LEN:]   # peptide = last 9 residues of chain B
        n_binder   = len(binder_res)
        binder_seq = ''.join(THREE_TO_ONE.get(r.resname, 'X') for r in binder_res)

        # Sanity-check peptide identity
        pep_seq = ''.join(THREE_TO_ONE.get(r.resname, 'X') for r in pep_res)
        if pep_seq != PEPTIDE_SEQ:
            print(f"  WARNING: {design} pep={pep_seq} (expected {PEPTIDE_SEQ})")

        all_cts, polar_cts, hydro_cts = compute_contacts(binder_res, pep_res)

        # Accumulate frequency matrices
        for (bi, pi, *_) in all_cts:
            if bi < max_binder_len:
                all_freq[bi, pi] += 1
        for (bi, pi, *_) in polar_cts:
            if bi < max_binder_len:
                polar_freq[bi, pi] += 1
        for (bi, pi, *_) in hydro_cts:
            if bi < max_binder_len:
                hydro_freq[bi, pi] += 1
        n_contrib[:n_binder, :] += 1

        # Per-contact detail rows — all contacts
        for (bi, pi, dist, b_aa, p_aa) in all_cts:
            all_contact_rows.append({
                'design':          design,
                'binder_pos_0idx': bi,
                'binder_resnum':   binder_res[bi].id[1],
                'binder_aa':       b_aa,
                'pep_pos_1idx':    pi + 1,
                'pep_aa':          p_aa,
                'min_heavy_dist':  dist,
                'contact_type':    'all',
            })

        # Per-contact detail rows — polar contacts
        for (bi, pi, dist, b_at, p_at, b_aa, p_aa) in polar_cts:
            polar_contact_rows.append({
                'design':          design,
                'binder_pos_0idx': bi,
                'binder_resnum':   binder_res[bi].id[1],
                'binder_aa':       b_aa,
                'binder_atom':     b_at,
                'pep_pos_1idx':    pi + 1,
                'pep_aa':          p_aa,
                'pep_atom':        p_at,
                'polar_dist':      dist,
                'contact_type':    'polar',
            })

        # Per-contact detail rows — hydrophobic contacts (own list)
        for (bi, pi, dist, b_aa, p_aa) in hydro_cts:
            hydro_contact_rows.append({
                'design':          design,
                'binder_pos_0idx': bi,
                'binder_resnum':   binder_res[bi].id[1],
                'binder_aa':       b_aa,
                'pep_pos_1idx':    pi + 1,
                'pep_aa':          p_aa,
                'pep_resname':     pep_res[pi].resname,
                'min_dist':        dist,
                'contact_type':    'hydrophobic',
            })

        # Per-design summary
        contacts_p5_all   = sum(1 for (_, pi, *_) in all_cts   if pi == 4)
        contacts_p5_polar = sum(1 for (_, pi, *_) in polar_cts if pi == 4)
        contacts_p5_hydro = sum(1 for (_, pi, *_) in hydro_cts if pi == 4)

        # Salt bridges at p5: binder Arg or Lys polar-contacting D12 (Asp)
        # These are the strongest G12D-vs-WT specificity determinants because
        # G12 WT (Gly) has no carboxylate to accept this interaction.
        SALT_BRIDGE_AA = {'R', 'K'}
        p5_sb_contacts = [
            (bi, pi, dist, b_at, p_at, b_aa, p_aa)
            for (bi, pi, dist, b_at, p_at, b_aa, p_aa) in polar_cts
            if pi == 4 and b_aa in SALT_BRIDGE_AA
        ]
        n_p5_saltbridge     = len(p5_sb_contacts)
        p5_sb_binder_resnums = sorted(set(binder_res[bi].id[1] for (bi, *_) in p5_sb_contacts))
        p5_sb_binder_aas    = [b_aa for (*_, b_aa, _) in p5_sb_contacts]

        summary_rows.append({
            'design':                        design,
            'binder_len':                    n_binder,
            'binder_seq':                    binder_seq,
            'n_all_contacts':                len(all_cts),
            'n_polar_contacts':              len(polar_cts),
            'n_hydrophobic_contacts':        len(hydro_cts),
            'n_contacts_p5_all':             contacts_p5_all,
            'n_contacts_p5_polar':           contacts_p5_polar,
            'n_contacts_p5_hydro':           contacts_p5_hydro,
            'n_p5_saltbridge':               n_p5_saltbridge,
            'p5_sb_binder_resnums':          str(p5_sb_binder_resnums),
            'p5_sb_binder_aas':              str(p5_sb_binder_aas),
            'binder_res_in_contact':         len(set(bi for (bi, *_) in all_cts)),
            'pep_positions_contacted_all':   sorted(set(pi + 1 for (_, pi, *_) in all_cts)),
            'pep_positions_contacted_polar': sorted(set(pi + 1 for (_, pi, *_) in polar_cts)),
            'pep_positions_contacted_hydro': sorted(set(pi + 1 for (_, pi, *_) in hydro_cts)),
        })
        n_processed += 1

    print(f"Processed {n_processed}/{len(designs)} designs")

    # Normalize by number of contributing designs per matrix cell
    with np.errstate(invalid='ignore'):
        all_freq_norm   = np.where(n_contrib > 0, all_freq   / n_contrib, 0.0)
        polar_freq_norm = np.where(n_contrib > 0, polar_freq / n_contrib, 0.0)
        hydro_freq_norm = np.where(n_contrib > 0, hydro_freq / n_contrib, 0.0)

    # Save tables
    pd.DataFrame(all_contact_rows).to_csv(
        os.path.join(args.out_dir, 'all_contacts.tsv'), sep='\t', index=False)
    pd.DataFrame(polar_contact_rows).to_csv(
        os.path.join(args.out_dir, 'polar_contacts.tsv'), sep='\t', index=False)
    pd.DataFrame(hydro_contact_rows).to_csv(
        os.path.join(args.out_dir, 'hydrophobic_contacts.tsv'), sep='\t', index=False)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.out_dir, 'contact_summary.tsv'), sep='\t', index=False)

    max_occ    = max(r['binder_len'] for r in summary_rows) if summary_rows else max_binder_len
    pep_labels = [f'p{i+1}\n{PEPTIDE_SEQ[i]}' for i in range(PEPTIDE_LEN)]

    # ── Plot 1: Side-by-side contact frequency heatmaps ───────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, max_occ // 6)), sharey=True)

    for ax, freq, title, cmap in zip(
        axes,
        [all_freq_norm[:max_occ], polar_freq_norm[:max_occ], hydro_freq_norm[:max_occ]],
        [f'All contacts (< {ALL_CONTACT_DIST} Å)',
         f'Polar contacts (< {POLAR_CONTACT_DIST} Å, N/O pairs)',
         f'Hydrophobic contacts (< {HYDROPHOBIC_CONTACT_DIST} Å, hydrophobic pep res)'],
        ['YlOrRd', 'Blues', 'Greens']
    ):
        im = ax.imshow(freq, aspect='auto', cmap=cmap,
                       vmin=0, vmax=1, origin='upper', interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label='Contact frequency')
        ax.set_xticks(range(PEPTIDE_LEN))
        ax.set_xticklabels(pep_labels, fontsize=9)
        ax.set_xlabel('Peptide position', fontsize=11)
        ax.set_title(title, fontsize=11)
        # Highlight p5 (D12 — the KRAS G12D neoepitope position)
        ax.axvline(3.5, color='blue', lw=1.5, ls='--', alpha=0.7)
        ax.axvline(4.5, color='blue', lw=1.5, ls='--', alpha=0.7)
        ax.text(4, -1.5, 'D12\n(neo)', ha='center', va='top',
                color='blue', fontsize=8, fontweight='bold')

    axes[0].set_ylabel('Binder residue position (0-indexed)', fontsize=11)
    fig.suptitle(
        f'Binder–Peptide Contact Frequency ({n_processed} designs)',
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'contact_frequency_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 2: Per-peptide-position mean contact frequency bar chart ─────────
    all_per_pos   = all_freq_norm[:max_occ].mean(axis=0)
    polar_per_pos = polar_freq_norm[:max_occ].mean(axis=0)
    hydro_per_pos = hydro_freq_norm[:max_occ].mean(axis=0)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    x     = np.arange(PEPTIDE_LEN)
    width = 0.27
    ax2.bar(x - width, all_per_pos,   width, color='#e74c3c', label='All contacts',        edgecolor='white')
    ax2.bar(x,         polar_per_pos, width, color='#3498db', label='Polar contacts',       edgecolor='white')
    ax2.bar(x + width, hydro_per_pos, width, color='#2ecc71', label='Hydrophobic contacts', edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(pep_labels, fontsize=10)
    ax2.set_ylabel('Mean contact frequency', fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.set_title('Mean Contact Frequency per Peptide Position', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.axvline(3.5, color='blue', lw=1, ls='--', alpha=0.5)
    ax2.axvline(4.5, color='blue', lw=1, ls='--', alpha=0.5)
    plt.tight_layout()
    fig2.savefig(os.path.join(args.out_dir, 'contact_per_pep_position.png'),
                 dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nOutputs written to {args.out_dir}")
    print(f"  all_contacts.tsv, polar_contacts.tsv, hydrophobic_contacts.tsv, contact_summary.tsv")
    print(f"  contact_frequency_heatmap.png, contact_per_pep_position.png")

    print(f"\n── Designs with polar contacts at p5 (D12 neoepitope) ──")
    summary_df = pd.DataFrame(summary_rows)
    p5_polar = summary_df[summary_df['n_contacts_p5_polar'] > 0][
        ['design', 'n_all_contacts', 'n_polar_contacts',
         'n_contacts_p5_all', 'n_contacts_p5_polar', 'n_contacts_p5_hydro']
    ]
    print(p5_polar.to_string(index=False))

    print(f"\n── Designs with Arg/Lys salt bridge at p5 (strongest G12D specificity) ──")
    p5_sb = summary_df[summary_df['n_p5_saltbridge'] > 0][
        ['design', 'n_p5_saltbridge', 'p5_sb_binder_resnums', 'p5_sb_binder_aas',
         'n_contacts_p5_all', 'n_contacts_p5_polar']
    ]
    if len(p5_sb):
        print(p5_sb.to_string(index=False))
    else:
        print("  (none — no Arg/Lys polar contact with D12 found in this set)")


if __name__ == '__main__':
    main()