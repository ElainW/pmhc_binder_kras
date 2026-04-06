"""
contact_map.py  (updated)

Contact definitions matching PyMOL conventions:
  all_contacts:   any heavy atom pair < 4.0 Å  (PyMOL mode=0)
  polar_contacts: N/O donor-acceptor pair < 3.5 Å  (PyMOL mode=2 approximation)

Chain convention in pMHC fold PDBs:
  chain A = binder
  chain B = MHC + peptide (peptide = last 9 residues)

Usage:
    python contact_map.py \
        --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
        --pmhc_fold_dir /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/outputs/r2/pmhc_fold_on_runs \
        --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from Bio.PDB import PDBParser

DELTA_PAE_CUTOFF    = -0.5
PEPTIDE_SEQ         = 'VVGADGVGK'
PEPTIDE_LEN         = 9
ALL_CONTACT_DIST    = 4.0   # Å — any heavy atom (PyMOL mode=0)
POLAR_CONTACT_DIST  = 3.5   # Å — N/O donor-acceptor (PyMOL mode=2)

# Polar atom names (heavy atom donors/acceptors for H-bonds)
POLAR_ELEMENTS = {'N', 'O'}

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}


def find_pmhc_fold_pdb(design: str, pmhc_fold_dir: str) -> str | None:
    pattern = os.path.join(
        pmhc_fold_dir, 'sample_df_chunk_*',
        f'{design}_A1101_VVGADGVGK_unrelaxed_model_1.pdb'
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def get_chain_residues(struct, chain_id: str):
    residues = []
    for model in struct:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if res.id[0] == ' ':
                    residues.append(res)
    return residues


def compute_contacts(binder_residues, peptide_residues):
    """
    Three contact types matching PyMOL selections:

    all_contacts:
        any heavy atom pair < 4.0 Å (PyMOL mode=0)

    polar_contacts:
        N/O donor-acceptor pair < 3.5 Å (PyMOL mode=2 approximation)

    hydrophobic_contacts:
        peptide residue is in HYDROPHOBIC_RESIDUES and any of its atoms
        are within 4.0 Å of any binder atom (PyMOL: resn ALA+VAL+... within 4.0)

    Returns:
        all_contacts:        list of (bi, pi, min_dist, b_aa, p_aa)
        polar_contacts:      list of (bi, pi, min_dist, b_atom, p_atom, b_aa, p_aa)
        hydrophobic_contacts: list of (bi, pi, min_dist, b_aa, p_aa)
    """
    all_contacts         = []
    polar_contacts       = []
    hydrophobic_contacts = []

    for bi, b_res in enumerate(binder_residues):
        b_aa = THREE_TO_ONE.get(b_res.resname, 'X')
        b_heavy = [(a, a.get_vector().get_array())
                   for a in b_res.get_atoms()
                   if a.element not in ('H', '') and a.element is not None]
        if not b_heavy:
            continue
        b_coords = np.array([c for _, c in b_heavy])

        for pi, p_res in enumerate(peptide_residues):
            p_aa = THREE_TO_ONE.get(p_res.resname, 'X')
            p_heavy = [(a, a.get_vector().get_array())
                       for a in p_res.get_atoms()
                       if a.element not in ('H', '') and a.element is not None]
            if not p_heavy:
                continue
            p_coords = np.array([c for _, c in p_heavy])

            # Distance matrix: shape (n_b_atoms, n_p_atoms)
            dists = np.linalg.norm(
                b_coords[:, None, :] - p_coords[None, :, :], axis=-1
            )
            min_dist = float(dists.min())

            # ── All contacts ─────────────────────────────────────────────────
            if min_dist <= ALL_CONTACT_DIST:
                all_contacts.append((bi, pi, round(min_dist, 3), b_aa, p_aa))

            # ── Polar contacts ───────────────────────────────────────────────
            b_polar = [(a, c) for a, c in b_heavy if a.element in POLAR_ELEMENTS]
            p_polar = [(a, c) for a, c in p_heavy if a.element in POLAR_ELEMENTS]
            if b_polar and p_polar:
                bp_coords = np.array([c for _, c in b_polar])
                pp_coords = np.array([c for _, c in p_polar])
                polar_dists = np.linalg.norm(
                    bp_coords[:, None, :] - pp_coords[None, :, :], axis=-1
                )
                min_polar_dist = float(polar_dists.min())
                if min_polar_dist <= POLAR_CONTACT_DIST:
                    min_idx    = np.unravel_index(polar_dists.argmin(), polar_dists.shape)
                    b_atom_name = b_polar[min_idx[0]][0].get_name().strip()
                    p_atom_name = p_polar[min_idx[1]][0].get_name().strip()
                    polar_contacts.append((
                        bi, pi,
                        round(min_polar_dist, 3),
                        b_atom_name, p_atom_name,
                        b_aa, p_aa
                    ))

            # ── Hydrophobic contacts ─────────────────────────────────────────
            # Peptide residue must be hydrophobic by identity
            # Any atom of that residue within 4.0 Å of any binder atom
            if p_res.resname in HYDROPHOBIC_RESIDUES and min_dist <= HYDROPHOBIC_CONTACT_DIST:
                hydrophobic_contacts.append((bi, pi, round(min_dist, 3), b_aa, p_aa))

    return all_contacts, polar_contacts, hydrophobic_contacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_scores',  required=True)
    parser.add_argument('--pmhc_fold_dir',  required=True)
    parser.add_argument('--out_dir',         required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.merged_scores, sep='\t')
    passing = df[df['delta_pae_int_peptide'] < DELTA_PAE_CUTOFF].copy()
    designs = passing['design'].tolist()
    print(f"Processing {len(designs)} designs...")

    pdb_parser = PDBParser(QUIET=True)

    all_contact_rows   = []
    polar_contact_rows = []
    summary_rows       = []

    # First pass for max binder length
    max_binder_len = 0
    fold_pdbs = {}
    for design in designs:
        pdb_path = find_pmhc_fold_pdb(design, args.pmhc_fold_dir)
        fold_pdbs[design] = pdb_path
        if pdb_path:
            struct = pdb_parser.get_structure(design, pdb_path)
            n = len(get_chain_residues(struct, 'A'))
            max_binder_len = max(max_binder_len, n)

    # Frequency matrices
    all_freq   = np.zeros((max_binder_len, PEPTIDE_LEN), dtype=float)
    polar_freq = np.zeros((max_binder_len, PEPTIDE_LEN), dtype=float)
    n_contrib  = np.zeros((max_binder_len, PEPTIDE_LEN), dtype=int)
    hydro_freq = np.zeros((max_binder_len, PEPTIDE_LEN), dtype=float)
    n_processed = 0

    for design in designs:
        fold_path = fold_pdbs.get(design)
        if fold_path is None:
            print(f"  WARNING: not found for {design}")
            continue

        struct = pdb_parser.get_structure(design, fold_path)
        binder_res = get_chain_residues(struct, 'A')
        chain_b    = get_chain_residues(struct, 'B')
        pep_res    = chain_b[-PEPTIDE_LEN:]
        n_binder   = len(binder_res)
        binder_seq = ''.join(THREE_TO_ONE.get(r.resname, 'X') for r in binder_res)

        # Verify peptide
        pep_seq = ''.join(THREE_TO_ONE.get(r.resname, 'X') for r in pep_res)
        if pep_seq != PEPTIDE_SEQ:
            print(f"  WARNING: {design} pep={pep_seq}")

        all_cts, polar_cts, hydro_cts = compute_contacts(binder_res, pep_res)

        # Accumulate frequency matrices
        for (bi, pi, dist, b_aa, p_aa) in all_cts:
            if bi < max_binder_len:
                all_freq[bi, pi] += 1
        for (bi, pi, dist, b_at, p_at, b_aa, p_aa) in polar_cts:
            if bi < max_binder_len:
                polar_freq[bi, pi] += 1
        n_contrib[:n_binder, :] += 1
        # Accumulate hydrophobic frequency
        for (bi, pi, dist, b_aa, p_aa) in hydro_cts:
            if bi < max_binder_len:
                hydro_freq[bi, pi] += 1

        # Per-contact detail rows
        for (bi, pi, dist, b_aa, p_aa) in all_cts:
            all_contact_rows.append({
                'design': design,
                'binder_pos_0idx': bi,
                'binder_resnum': binder_res[bi].id[1],
                'binder_aa': b_aa,
                'pep_pos_1idx': pi + 1,
                'pep_aa': p_aa,
                'min_heavy_dist': dist,
                'contact_type': 'all',
            })
        for (bi, pi, dist, b_at, p_at, b_aa, p_aa) in polar_cts:
            polar_contact_rows.append({
                'design': design,
                'binder_pos_0idx': bi,
                'binder_resnum': binder_res[bi].id[1],
                'binder_aa': b_aa,
                'binder_atom': b_at,
                'pep_pos_1idx': pi + 1,
                'pep_aa': p_aa,
                'pep_atom': p_at,
                'polar_dist': dist,
                'contact_type': 'polar',
            })
        # Save hydrophobic contacts
        for (bi, pi, dist, b_aa, p_aa) in hydro_cts:
            polar_contact_rows.append({   # use a new list: hydro_contact_rows
                'design': design,
                'binder_pos_0idx': bi,
                'binder_resnum': binder_res[bi].id[1],
                'binder_aa': b_aa,
                'pep_pos_1idx': pi + 1,
                'pep_aa': p_aa,
                'pep_resname': pep_res[pi].resname,
                'min_dist': dist,
                'contact_type': 'hydrophobic',
            })

        # Per-design summary
        contacts_p5_all   = sum(1 for (bi, pi, _, _, _) in all_cts   if pi == 4)
        contacts_p5_polar = sum(1 for (bi, pi, _, _, _, _, _) in polar_cts if pi == 4)
        contacts_p5_hydro = sum(1 for (bi, pi, _, _, _) in hydro_cts if pi == 4)
        summary_rows.append({
            'design': design,
            'binder_len': n_binder,
            'binder_seq': binder_seq,
            'n_all_contacts':       len(all_cts),
            'n_polar_contacts':     len(polar_cts),
            'n_contacts_p5_all':    contacts_p5_all,
            'n_contacts_p5_polar':  contacts_p5_polar,
            'n_contacts_p5_hydro': contacts_p5_hydro,
            'pep_positions_contacted_hydro': sorted(set(pi+1 for (_, pi, _, _, _) in hydro_cts)),
            'binder_res_in_contact': len(set(bi for (bi, _, _, _, _) in all_cts)),
            'pep_positions_contacted_all':   sorted(set(pi+1 for (_, pi, _, _, _) in all_cts)),
            'pep_positions_contacted_polar': sorted(set(pi+1 for (_, pi, _, _, _, _, _) in polar_cts)),
        })
        n_processed += 1

    print(f"Processed {n_processed}/{len(designs)} designs")

    # Normalize
    with np.errstate(invalid='ignore'):
        all_freq_norm   = np.where(n_contrib > 0, all_freq   / n_contrib, 0.0)
        polar_freq_norm = np.where(n_contrib > 0, polar_freq / n_contrib, 0.0)
        hydro_freq_norm = np.where(n_contrib > 0, hydro_freq / n_contrib, 0.0)

    # Save tables
    pd.DataFrame(all_contact_rows).to_csv(
        os.path.join(args.out_dir, 'all_contacts.tsv'), sep='\t', index=False)
    pd.DataFrame(polar_contact_rows).to_csv(
        os.path.join(args.out_dir, 'polar_contacts.tsv'), sep='\t', index=False)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.out_dir, 'contact_summary.tsv'), sep='\t', index=False)
    pd.DataFrame(hydro_contact_rows).to_csv(
        os.path.join(args.out_dir, 'hydrophobic_contacts.tsv'), sep='\t', index=False)

    max_occ = max(r['binder_len'] for r in summary_rows) if summary_rows else max_binder_len

    # ── Plot 1: Side-by-side heatmaps ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, max_occ // 6)),
                             sharey=True)
    pep_labels = [f'p{i+1}\n{PEPTIDE_SEQ[i]}' for i in range(PEPTIDE_LEN)]

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
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04,
                     label='Contact frequency')
        ax.set_xticks(range(PEPTIDE_LEN))
        ax.set_xticklabels(pep_labels, fontsize=9)
        ax.set_xlabel('Peptide position', fontsize=11)
        ax.set_title(title, fontsize=11)
        # Highlight p5
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

    # ── Plot 2: Per-position bar chart ────────────────────────────────────────
    all_per_pos   = all_freq_norm[:max_occ].mean(axis=0)
    polar_per_pos = polar_freq_norm[:max_occ].mean(axis=0)
    hydro_per_pos = hydro_freq_norm[:max_occ].mean(axis=0)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    x = np.arange(PEPTIDE_LEN)
    width = 0.27
    colors_all   = ['#e74c3c' if i == 4 else '#3498db' for i in range(PEPTIDE_LEN)]
    colors_polar = ['#c0392b' if i == 4 else '#1a5276' for i in range(PEPTIDE_LEN)]

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
    print(f"  all_contacts.tsv, polar_contacts.tsv, contact_summary.tsv")
    print(f"  contact_frequency_heatmap.png, contact_per_pep_position.png")

    # Print designs with p5 polar contacts
    print(f"\n── Designs with polar contacts at p5 (D12 neoepitope) ──")
    summary_df = pd.DataFrame(summary_rows)
    p5_polar = summary_df[summary_df['n_contacts_p5_polar'] > 0][
        ['design', 'n_all_contacts', 'n_polar_contacts',
         'n_contacts_p5_all', 'n_contacts_p5_polar']]
    print(p5_polar.to_string(index=False))


if __name__ == '__main__':
    main()