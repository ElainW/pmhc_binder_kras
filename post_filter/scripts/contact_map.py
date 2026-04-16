"""
contact_map.py

Reads pre-split pMHC fold PDBs from split_pmhc_fold_pdbs.py:
  {split_dir}/{design}_chainA.pdb  — binder (chain A, renumbered from 1)
  {split_dir}/{design}_chainB.pdb  — MHC + peptide (chain B, last 9 = peptide)

Contact definitions:
  all_contacts:         any heavy atom pair < 4.0 Å
  polar_contacts:       N/O donor-acceptor pair < 4.0 Å
  hydrophobic_contacts: peptide residue is hydrophobic AND any heavy atom < 4.0 Å

Plot 1 (per backbone group):
  Groups designs by backbone ID (the number after 'orient_' in the design name,
  e.g. orient_158_... → group '158'). Produces one figure per group with 3
  heatmap panels (all / polar / hydrophobic). Y-axis = binder residue position
  within that group (padded to group max length). Raw contact counts, not frequency.

Plot 2 (all designs):
  Per-peptide-position bar chart of raw contact counts summed across all designs.
  Figure title shows N designs contributing.

Usage:
    # On unrelaxed pMHC fold structures (default):
    python contact_map.py \
        --merged_scores .../merged_scores_ranked.tsv \
        --split_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/ \
        --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/contact_unrelaxed/

    # On FastRelaxed structures:
    python contact_map.py \
        --merged_scores .../merged_scores_ranked.tsv \
        --split_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/ \
        --relaxed_pdb_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/fastrelax/relaxed_pdbs/ \
        --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/contact_relaxed/

    When --relaxed_pdb_dir is given, residues are read from
    {relaxed_pdb_dir}/{design}_relaxed.pdb (chain A = binder, chain B = MHC+peptide)
    instead of the pre-split chainA/chainB PDBs. The split_dir is still used
    to determine binder lengths for designs missing a relaxed PDB.
"""

import os
import re
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser

DELTA_PAE_CUTOFF         = -0.5
PEPTIDE_SEQ              = 'VVGADGVGK'
PEPTIDE_LEN              = 9
ALL_CONTACT_DIST         = 4.0
POLAR_CONTACT_DIST       = 4.0
HYDROPHOBIC_CONTACT_DIST = 4.0

POLAR_ELEMENTS       = {'N', 'O'}
HYDROPHOBIC_RESIDUES = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO', 'TYR'}

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_residues(pdb_path: str, chain_id: str) -> list:
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


def get_residues_from_complex(pdb_path: str,
                               binder_chain: str = 'A',
                               pmhc_chain: str   = 'B') -> tuple[list, list]:
    """
    Load a single-file complex PDB (e.g. FastRelaxed {design}_relaxed.pdb)
    and return (binder_residues, pmhc_residues) split by chain ID.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('s', pdb_path)
    binder_res, pmhc_res = [], []
    for model in struct:
        for chain in model:
            if chain.id == binder_chain:
                for res in chain:
                    if res.id[0] == ' ':
                        binder_res.append(res)
            elif chain.id == pmhc_chain:
                for res in chain:
                    if res.id[0] == ' ':
                        pmhc_res.append(res)
        break
    return binder_res, pmhc_res


def backbone_group(design: str) -> str:
    """
    Extract backbone group ID from design name.
    e.g. 'orient_158_pT12__64_sample01_af2pred' → '158'
         'orient_252_pT20__31_sample01_af2pred'  → '252'
    Returns 'other' if pattern not matched.
    """
    m = re.match(r'orient_(\d+)_', design)
    return m.group(1) if m else 'other'


def compute_contacts(binder_residues: list, peptide_residues: list):
    all_contacts, polar_contacts, hydrophobic_contacts = [], [], []

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

            dists    = np.linalg.norm(b_coords[:, None, :] - p_coords[None, :, :], axis=-1)
            min_dist = float(dists.min())

            if min_dist <= ALL_CONTACT_DIST:
                all_contacts.append((bi, pi, round(min_dist, 3), b_aa, p_aa))

            b_polar = [(a, c) for a, c in b_heavy if a.element in POLAR_ELEMENTS]
            p_polar = [(a, c) for a, c in p_heavy if a.element in POLAR_ELEMENTS]
            if b_polar and p_polar:
                bp = np.array([c for _, c in b_polar])
                pp = np.array([c for _, c in p_polar])
                pd_ = np.linalg.norm(bp[:, None, :] - pp[None, :, :], axis=-1)
                min_p = float(pd_.min())
                if min_p <= POLAR_CONTACT_DIST:
                    idx = np.unravel_index(pd_.argmin(), pd_.shape)
                    polar_contacts.append((
                        bi, pi, round(min_p, 3),
                        b_polar[idx[0]][0].get_name().strip(),
                        p_polar[idx[1]][0].get_name().strip(),
                        b_aa, p_aa,
                    ))

            if p_res.resname in HYDROPHOBIC_RESIDUES and min_dist <= HYDROPHOBIC_CONTACT_DIST:
                hydrophobic_contacts.append((bi, pi, round(min_dist, 3), b_aa, p_aa))

    return all_contacts, polar_contacts, hydrophobic_contacts


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_backbone_group_heatmaps(group_id: str,
                                  all_counts: np.ndarray,
                                  polar_counts: np.ndarray,
                                  hydro_counts: np.ndarray,
                                  n_designs: int,
                                  out_path: str,
                                  mode_label: str = ''):
    """
    Plot 1: 3-panel raw count heatmap for one backbone group.
    rows = binder residue position (0-indexed, padded to group max length)
    cols = peptide position
    Values = raw counts (number of designs with a contact at that cell).
    """
    max_len    = all_counts.shape[0]
    pep_labels = [f'p{i+1}\n{PEPTIDE_SEQ[i]}' for i in range(PEPTIDE_LEN)]

    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, max_len // 5)), sharey=True)

    vmax_all   = int(all_counts.max())   if all_counts.max()   > 0 else 1
    vmax_polar = int(polar_counts.max()) if polar_counts.max() > 0 else 1
    vmax_hydro = int(hydro_counts.max()) if hydro_counts.max() > 0 else 1

    for ax, counts, title, cmap, vmax in zip(
        axes,
        [all_counts, polar_counts, hydro_counts],
        [f'All contacts (< {ALL_CONTACT_DIST} Å)',
         f'Polar contacts (< {POLAR_CONTACT_DIST} Å, N/O pairs)',
         f'Hydrophobic contacts (< {HYDROPHOBIC_CONTACT_DIST} Å)'],
        ['YlOrRd', 'Blues', 'Greens'],
        [vmax_all, vmax_polar, vmax_hydro],
    ):
        im = ax.imshow(counts, aspect='auto', cmap=cmap,
                       vmin=0, vmax=vmax, origin='upper', interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label='# designs')
        ax.set_xticks(range(PEPTIDE_LEN))
        ax.set_xticklabels(pep_labels, fontsize=9)
        ax.set_xlabel('Peptide position', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.axvline(3.5, color='blue', lw=1.5, ls='--', alpha=0.7)
        ax.axvline(4.5, color='blue', lw=1.5, ls='--', alpha=0.7)
        ax.text(4, -1.5, 'D12\n(neo)', ha='center', va='top',
                color='blue', fontsize=8, fontweight='bold')

    axes[0].set_ylabel('Binder residue position (0-indexed)', fontsize=11)
    mode_str = f' [{mode_label}]' if mode_label else ''
    fig.suptitle(
        f'Backbone group orient_{group_id} — Contact counts '
        f'({n_designs} designs){mode_str}',
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap → {out_path}")


def plot_per_position_counts(all_counts_total: np.ndarray,
                              polar_counts_total: np.ndarray,
                              hydro_counts_total: np.ndarray,
                              n_designs: int,
                              out_path: str):
    """
    Plot 2: Per-peptide-position bar chart of raw contact counts
    summed across all designs. Y-axis = total count, not frequency.
    """
    all_per_pos   = all_counts_total.sum(axis=0)
    polar_per_pos = polar_counts_total.sum(axis=0)
    hydro_per_pos = hydro_counts_total.sum(axis=0)
    pep_labels    = [f'p{i+1}\n{PEPTIDE_SEQ[i]}' for i in range(PEPTIDE_LEN)]

    fig, ax = plt.subplots(figsize=(10, 4))
    x     = np.arange(PEPTIDE_LEN)
    width = 0.27
    ax.bar(x - width, all_per_pos,   width, color='#e74c3c',
           label='All contacts',         edgecolor='white')
    ax.bar(x,         polar_per_pos, width, color='#3498db',
           label='Polar contacts',        edgecolor='white')
    ax.bar(x + width, hydro_per_pos, width, color='#2ecc71',
           label='Hydrophobic contacts',  edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(pep_labels, fontsize=10)
    ax.set_ylabel('Total contacts (# designs × # residues)', fontsize=11)
    ax.set_title(
        f'Total Contact Counts per Peptide Position ({n_designs} designs)',
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.axvline(3.5, color='blue', lw=1, ls='--', alpha=0.5)
    ax.axvline(4.5, color='blue', lw=1, ls='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved bar chart → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_scores',   required=True)
    parser.add_argument('--split_dir',       required=True,
                        help='Pre-split chainA/chainB PDB directory from '
                             'split_pmhc_fold_pdbs.py (always needed for '
                             'length determination; used as structure source '
                             'when --relaxed_pdb_dir is not given)')
    parser.add_argument('--relaxed_pdb_dir', default=None,
                        help='If given, read residues from FastRelaxed '
                             '{design}_relaxed.pdb files in this directory '
                             'instead of the pre-split chainA/chainB PDBs. '
                             'Expected: chain A = binder, chain B = MHC+peptide.')
    parser.add_argument('--out_dir',         required=True)
    args = parser.parse_args()

    use_relaxed = args.relaxed_pdb_dir is not None
    mode_label  = 'FastRelaxed' if use_relaxed else 'pMHC fold (unrelaxed)'
    print(f"Mode: {mode_label}")

    os.makedirs(args.out_dir, exist_ok=True)

    df      = pd.read_csv(args.merged_scores, sep='\t')
    passing = df[df['delta_pae_int_peptide'] < DELTA_PAE_CUTOFF].copy()
    designs = passing['design'].tolist()
    print(f"Processing {len(designs)} designs...")

    # ── First pass: validate files, determine per-group max lengths ───────────
    valid_designs = []
    group_max_len: dict[str, int] = defaultdict(int)

    for design in designs:
        # Always check split PDBs exist (used as fallback for length)
        chain_a_split = os.path.join(args.split_dir, f'{design}_chainA.pdb')
        chain_b_split = os.path.join(args.split_dir, f'{design}_chainB.pdb')

        if use_relaxed:
            relaxed_pdb = os.path.join(args.relaxed_pdb_dir,
                                        f'{design}_relaxed.pdb')
            if not os.path.exists(relaxed_pdb):
                print(f"  WARNING: relaxed PDB not found for {design} — "
                      f"falling back to split PDB for length, skipping contacts")
                # Still need length for matrix sizing — use split PDB
                if os.path.exists(chain_a_split):
                    binder_res = get_residues(chain_a_split, 'A')
                    if binder_res:
                        grp = backbone_group(design)
                        group_max_len[grp] = max(group_max_len[grp], len(binder_res))
                continue
            binder_res, _ = get_residues_from_complex(relaxed_pdb)
        else:
            if not os.path.exists(chain_a_split) or not os.path.exists(chain_b_split):
                print(f"  ERROR: split PDBs not found for {design} — skipping")
                continue
            binder_res = get_residues(chain_a_split, 'A')

        if not binder_res:
            print(f"  ERROR: empty binder for {design} — skipping")
            continue
        grp = backbone_group(design)
        group_max_len[grp] = max(group_max_len[grp], len(binder_res))
        valid_designs.append(design)

    if not valid_designs:
        print("ERROR: no valid designs found")
        return

    global_max_len = max(group_max_len.values())

    # ── Per-group count matrices ──────────────────────────────────────────────
    # group_counts[grp] = {'all': array, 'polar': array, 'hydro': array, 'n': int}
    group_counts: dict[str, dict] = {}
    for grp, max_len in group_max_len.items():
        group_counts[grp] = {
            'all':   np.zeros((max_len, PEPTIDE_LEN), dtype=int),
            'polar': np.zeros((max_len, PEPTIDE_LEN), dtype=int),
            'hydro': np.zeros((max_len, PEPTIDE_LEN), dtype=int),
            'n':     0,
        }

    # Global count matrices for plot 2
    global_all   = np.zeros((global_max_len, PEPTIDE_LEN), dtype=int)
    global_polar = np.zeros((global_max_len, PEPTIDE_LEN), dtype=int)
    global_hydro = np.zeros((global_max_len, PEPTIDE_LEN), dtype=int)

    all_contact_rows   = []
    polar_contact_rows = []
    hydro_contact_rows = []
    summary_rows       = []
    n_processed        = 0

    for design in valid_designs:
        # Load residues from the appropriate source
        if use_relaxed:
            relaxed_pdb = os.path.join(args.relaxed_pdb_dir,
                                        f'{design}_relaxed.pdb')
            binder_res, pmhc_res = get_residues_from_complex(relaxed_pdb)
        else:
            chain_a = os.path.join(args.split_dir, f'{design}_chainA.pdb')
            chain_b = os.path.join(args.split_dir, f'{design}_chainB.pdb')
            binder_res = get_residues(chain_a, 'A')
            pmhc_res   = get_residues(chain_b, 'B')
        pep_res    = pmhc_res[-PEPTIDE_LEN:]
        n_binder   = len(binder_res)
        binder_seq = ''.join(THREE_TO_ONE.get(r.resname, 'X') for r in binder_res)
        grp        = backbone_group(design)

        pep_seq = ''.join(THREE_TO_ONE.get(r.resname, 'X') for r in pep_res)
        if pep_seq != PEPTIDE_SEQ:
            print(f"  WARNING: {design} pep={pep_seq} (expected {PEPTIDE_SEQ})")

        all_cts, polar_cts, hydro_cts = compute_contacts(binder_res, pep_res)

        # Accumulate per-group counts
        gc = group_counts[grp]
        for (bi, pi, *_) in all_cts:
            if bi < gc['all'].shape[0]:
                gc['all'][bi, pi] += 1
        for (bi, pi, *_) in polar_cts:
            if bi < gc['polar'].shape[0]:
                gc['polar'][bi, pi] += 1
        for (bi, pi, *_) in hydro_cts:
            if bi < gc['hydro'].shape[0]:
                gc['hydro'][bi, pi] += 1
        gc['n'] += 1

        # Accumulate global counts for plot 2
        for (bi, pi, *_) in all_cts:
            if bi < global_max_len:
                global_all[bi, pi] += 1
        for (bi, pi, *_) in polar_cts:
            if bi < global_max_len:
                global_polar[bi, pi] += 1
        for (bi, pi, *_) in hydro_cts:
            if bi < global_max_len:
                global_hydro[bi, pi] += 1

        # Per-contact detail rows
        for (bi, pi, dist, b_aa, p_aa) in all_cts:
            all_contact_rows.append({
                'design': design, 'binder_pos_0idx': bi,
                'binder_resnum': binder_res[bi].id[1], 'binder_aa': b_aa,
                'pep_pos_1idx': pi + 1, 'pep_aa': p_aa,
                'min_heavy_dist': dist, 'contact_type': 'all',
            })
        for (bi, pi, dist, b_at, p_at, b_aa, p_aa) in polar_cts:
            polar_contact_rows.append({
                'design': design, 'binder_pos_0idx': bi,
                'binder_resnum': binder_res[bi].id[1], 'binder_aa': b_aa,
                'binder_atom': b_at, 'pep_pos_1idx': pi + 1, 'pep_aa': p_aa,
                'pep_atom': p_at, 'polar_dist': dist, 'contact_type': 'polar',
            })
        for (bi, pi, dist, b_aa, p_aa) in hydro_cts:
            hydro_contact_rows.append({
                'design': design, 'binder_pos_0idx': bi,
                'binder_resnum': binder_res[bi].id[1], 'binder_aa': b_aa,
                'pep_pos_1idx': pi + 1, 'pep_aa': p_aa,
                'pep_resname': pep_res[pi].resname,
                'min_dist': dist, 'contact_type': 'hydrophobic',
            })

        # Salt bridge detection at p5
        SALT_BRIDGE_AA = {'R', 'K'}
        p5_sb = [(bi, pi, d, ba, pa, baa, paa)
                 for (bi, pi, d, ba, pa, baa, paa) in polar_cts
                 if pi == 4 and baa in SALT_BRIDGE_AA]

        summary_rows.append({
            'design':                        design,
            'backbone_group':                grp,
            'binder_len':                    n_binder,
            'binder_seq':                    binder_seq,
            'n_all_contacts':                len(all_cts),
            'n_polar_contacts':              len(polar_cts),
            'n_hydrophobic_contacts':        len(hydro_cts),
            'n_contacts_p5_all':             sum(1 for (_, pi, *_) in all_cts   if pi == 4),
            'n_contacts_p5_polar':           sum(1 for (_, pi, *_) in polar_cts if pi == 4),
            'n_contacts_p5_hydro':           sum(1 for (_, pi, *_) in hydro_cts if pi == 4),
            'n_p5_saltbridge':               len(p5_sb),
            'p5_sb_binder_resnums':          str(sorted(set(binder_res[bi].id[1] for (bi, *_) in p5_sb))),
            'p5_sb_binder_aas':              str([baa for (*_, baa, _) in p5_sb]),
            'binder_res_in_contact':         len(set(bi for (bi, *_) in all_cts)),
            'pep_positions_contacted_all':   sorted(set(pi + 1 for (_, pi, *_) in all_cts)),
            'pep_positions_contacted_polar': sorted(set(pi + 1 for (_, pi, *_) in polar_cts)),
            'pep_positions_contacted_hydro': sorted(set(pi + 1 for (_, pi, *_) in hydro_cts)),
        })
        n_processed += 1

    print(f"Processed {n_processed}/{len(designs)} designs")

    # ── Save TSVs ─────────────────────────────────────────────────────────────
    pd.DataFrame(all_contact_rows).to_csv(
        os.path.join(args.out_dir, 'all_contacts.tsv'), sep='\t', index=False)
    pd.DataFrame(polar_contact_rows).to_csv(
        os.path.join(args.out_dir, 'polar_contacts.tsv'), sep='\t', index=False)
    pd.DataFrame(hydro_contact_rows).to_csv(
        os.path.join(args.out_dir, 'hydrophobic_contacts.tsv'), sep='\t', index=False)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.out_dir, 'contact_summary.tsv'), sep='\t', index=False)

    # ── Plot 1: Per-backbone-group heatmaps (raw counts) ─────────────────────
    print(f"\nGenerating per-backbone-group heatmaps ({len(group_counts)} groups)...")
    for grp in sorted(group_counts.keys()):
        gc       = group_counts[grp]
        out_path = os.path.join(args.out_dir, f'contact_heatmap_orient{grp}.png')
        plot_backbone_group_heatmaps(
            group_id     = grp,
            all_counts   = gc['all'],
            polar_counts = gc['polar'],
            hydro_counts = gc['hydro'],
            n_designs    = gc['n'],
            out_path     = out_path,
            mode_label   = mode_label,
        )

    # ── Plot 2: Global per-position raw counts ────────────────────────────────
    plot_per_position_counts(
        all_counts_total   = global_all,
        polar_counts_total = global_polar,
        hydro_counts_total = global_hydro,
        n_designs          = n_processed,
        out_path           = os.path.join(args.out_dir, 'contact_per_pep_position_counts.png'),
    )

    # ── Console summary ───────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    print(f"\nMode: {mode_label} | {n_processed} designs processed")

    print(f"\n── Designs with polar contacts at p5 (D12) ──")
    p5 = summary_df[summary_df['n_contacts_p5_polar'] > 0][[
        'design', 'backbone_group', 'n_all_contacts', 'n_polar_contacts',
        'n_contacts_p5_all', 'n_contacts_p5_polar',
    ]]
    print(p5.to_string(index=False))

    print(f"\n── Designs with Arg/Lys salt bridge at p5 ──")
    sb = summary_df[summary_df['n_p5_saltbridge'] > 0][[
        'design', 'backbone_group', 'n_p5_saltbridge',
        'p5_sb_binder_resnums', 'p5_sb_binder_aas',
    ]]
    if len(sb):
        print(sb.to_string(index=False))
    else:
        print("  (none)")


if __name__ == '__main__':
    main()