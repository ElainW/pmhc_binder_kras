"""
plot_af3_contacts.py
====================
Produces per-design contact heatmap figures (mirroring contact_map.py style)
from the contact TSVs written by af3_design_stats.py.

One PNG per design with 4 panels:
  - All contacts       (any heavy atom < 4.0 A)
  - Polar — sidechain  (N/O < 4.0 A, peptide SIDE-CHAIN atom)
  - Polar — backbone   (N/O < 4.0 A, peptide BACKBONE atom)
  - Hydrophobic        (hydrophobic peptide residue < 4.0 A)

Polar contacts are split by whether the peptide atom is a side-chain or a
backbone N/O (the pep_atom_is_sidechain column written by af3_design_stats.py).
Two specific polar-interaction types are marked as overlays on the polar panels:
  - Salt bridges (★) — charged side-chain pairs; always on the sidechain panel.
  - H-bonds      (○) — geometric D-H...A (needs explicit H); drawn on whichever
                       panel (sidechain/backbone) the peptide atom belongs to.
Hydrophobic contacts are kept separate and are NOT split.

X-axis: peptide positions labeled p{i}\n{AA} using actual peptide sequence
         from design_epitopes.csv. Hotspot positions highlighted with blue borders.
Y-axis: binder residue position (0-indexed). True binder length read from
         af3_design_stats.tsv (n_binder_res column from DSSP).

Usage:
    python plot_af3_contacts.py \
        --contacts_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/contacts/ \
        --epitopes_csv ~/Desktop/pmhc_binder_kras/post_filter/inputs/design_epitopes.csv \
        --stats_tsv    ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/af3_design_stats.tsv

    # Specific designs only:
    python plot_af3_contacts.py \
        --contacts_dir ... --epitopes_csv ... --stats_tsv ... \
        --designs ctnnb1-15 gp100-3 mart1-3
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


ALL_CONTACT_DIST         = 4.0
POLAR_CONTACT_DIST       = 4.0
HYDROPHOBIC_CONTACT_DIST = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# Load epitopes
# ─────────────────────────────────────────────────────────────────────────────

def load_epitopes(csv_path: str) -> dict:
    """
    Returns {design: {'peptide': str, 'peptide_len': int,
                       'hotspot_positions': [int], 'target_name': str}}
    """
    df = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        design   = str(row['design']).strip()
        hotspots = [int(x.strip())
                    for x in str(row['cms_hotspot_positions']).split(',')
                    if x.strip()]
        out[design] = {
            'peptide':           str(row['peptide']).strip(),
            'peptide_len':       int(row['peptide_len']),
            'hotspot_positions': hotspots,
            'target_name':       str(row['target_name']).strip(),
        }
    return out


def load_binder_lengths(stats_tsv: str) -> dict[str, int]:
    """
    Read true binder lengths from af3_design_stats.tsv (n_binder_res from DSSP).
    Returns {design: n_binder_res}.
    """
    df = pd.read_csv(stats_tsv, sep='\t')
    if 'n_binder_res' not in df.columns:
        print("WARNING: n_binder_res not found in stats TSV — "
              "binder length will be inferred from contacts TSV")
        return {}
    return dict(zip(df['design'].astype(str), df['n_binder_res'].astype(int)))


# ─────────────────────────────────────────────────────────────────────────────
# Build count matrices from TSVs
# ─────────────────────────────────────────────────────────────────────────────

def build_count_matrix(df: pd.DataFrame, n_binder: int,
                       n_pep: int) -> np.ndarray:
    """
    Build (n_binder x n_pep) count matrix from a contact TSV subset.
    Uses binder_pos_0idx (0-based) and pep_pos_1idx (1-based).
    """
    mat = np.zeros((n_binder, n_pep), dtype=int)
    for _, row in df.iterrows():
        bi = int(row['binder_pos_0idx'])
        pi = int(row['pep_pos_1idx']) - 1
        if 0 <= bi < n_binder and 0 <= pi < n_pep:
            mat[bi, pi] += 1
    return mat


def bool_col(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    """
    Return a boolean Series for `col`, robust to TSV round-tripping
    (booleans come back as the strings 'True'/'False'). Missing column →
    a Series filled with `default`.
    """
    if col in df.columns and len(df):
        s = df[col]
        if s.dtype == bool:
            return s
        return s.astype(str).str.strip().str.lower().isin(['true', '1', '1.0', 'yes'])
    return pd.Series([default] * len(df), index=df.index, dtype=bool)


def cells(df: pd.DataFrame, mask: pd.Series) -> set:
    """Set of (binder_pos_0idx, pep_pos_0idx) cells for rows where mask is True."""
    if not len(df):
        return set()
    sub = df[mask]
    return set(zip(sub['binder_pos_0idx'].astype(int),
                   sub['pep_pos_1idx'].astype(int) - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _overlay_markers(ax, kind: str, cell_set: set, n_binder: int, n_pep: int):
    """Scatter salt-bridge (★) or H-bond (○) markers at the given cells."""
    xs = [pi for (bi, pi) in cell_set if 0 <= bi < n_binder and 0 <= pi < n_pep]
    ys = [bi for (bi, pi) in cell_set if 0 <= bi < n_binder and 0 <= pi < n_pep]
    if not xs:
        return
    if kind == 'salt_bridge':
        ax.scatter(xs, ys, marker='*', s=80, facecolor='gold',
                   edgecolor='black', linewidths=0.6, zorder=3)
    else:  # hbond
        ax.scatter(xs, ys, marker='o', s=30, facecolor='none',
                   edgecolor='crimson', linewidths=1.3, zorder=3)


def plot_design_contacts(
    design: str,
    all_mat:      np.ndarray,
    polar_sc_mat: np.ndarray,
    polar_bb_mat: np.ndarray,
    hydro_mat:    np.ndarray,
    saltbridge_cells: set,
    hbond_sc_cells:   set,
    hbond_bb_cells:   set,
    peptide:   str,
    hotspot_positions: list[int],
    target_name: str,
    has_hbond: bool,
    out_path: str,
):
    n_pep    = len(peptide)
    n_binder = all_mat.shape[0]

    # (matrix, title, cmap, [(marker_kind, cell_set), ...])
    panels = [
        (all_mat,      f'All contacts\n(< {ALL_CONTACT_DIST} Å)',          'YlOrRd', []),
        (polar_sc_mat, f'Polar — sidechain\n(< {POLAR_CONTACT_DIST} Å, N/O)', 'Blues',
            [('salt_bridge', saltbridge_cells), ('hbond', hbond_sc_cells)]),
        (polar_bb_mat, f'Polar — backbone\n(< {POLAR_CONTACT_DIST} Å, N/O)', 'Purples',
            [('hbond', hbond_bb_cells)]),
        (hydro_mat,    f'Hydrophobic\n(< {HYDROPHOBIC_CONTACT_DIST} Å)',    'Greens', []),
    ]
    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(3.3 * n_panels, max(3, min(6, n_binder / 4))),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    pep_labels = [f'p{i+1}\n{peptide[i]}' for i in range(n_pep)]

    for ax, (mat, title, cmap, overlays) in zip(axes, panels):
        vmax = int(mat.max()) if mat.max() > 0 else 1
        im   = ax.imshow(mat, aspect='auto', cmap=cmap,
                         vmin=0, vmax=vmax, origin='upper',
                         interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label='contacts')

        ax.set_xticks(range(n_pep))
        ax.set_xticklabels(pep_labels, fontsize=9)
        ax.set_xlabel('Peptide position', fontsize=11)
        ax.set_title(title, fontsize=10)

        # Highlight hotspot columns with blue borders
        for pos in hotspot_positions:
            pi = pos - 1
            if 0 <= pi < n_pep:
                ax.axvline(pi - 0.5, color='royalblue', lw=1.5, alpha=0.8)
                ax.axvline(pi + 0.5, color='royalblue', lw=1.5, alpha=0.8)

        # Salt-bridge / H-bond overlays
        for kind, cell_set in overlays:
            _overlay_markers(ax, kind, cell_set, n_binder, n_pep)

    axes[0].set_ylabel('Binder residue position (0-indexed)', fontsize=11)

    handles = [
        mpatches.Patch(edgecolor='royalblue', facecolor='none',
                       linewidth=1.5, label='Hotspot positions'),
        Line2D([0], [0], marker='*', linestyle='none', markersize=12,
               markerfacecolor='gold', markeredgecolor='black',
               label='Salt bridge (sidechain)'),
        Line2D([0], [0], marker='o', linestyle='none', markersize=9,
               markerfacecolor='none', markeredgecolor='crimson',
               label='H-bond'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=9, framealpha=0.8)

    hb_note = '' if has_hbond else '   (no H atoms → H-bond markers unavailable)'
    fig.suptitle(
        f'{design}  |  {target_name}  |  peptide: {peptide}\n'
        f'hotspots: {hotspot_positions}  |  binder len: {n_binder}{hb_note}',
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--contacts_dir', required=True,
        help='Directory containing all_contacts.tsv, polar_contacts.tsv, '
             'hydrophobic_contacts.tsv, hbond_contacts.tsv')
    parser.add_argument('--epitopes_csv', required=True,
        help='design_epitopes.csv with columns: design, peptide, peptide_len, '
             'cms_hotspot_positions, target_name')
    parser.add_argument('--stats_tsv', required=True,
        help='af3_design_stats.tsv — provides true binder length (n_binder_res)')
    parser.add_argument('--designs', nargs='*',
        help='Designs to plot (default: all in all_contacts.tsv)')
    args = parser.parse_args()

    # Load TSVs
    all_tsv   = os.path.join(args.contacts_dir, 'all_contacts.tsv')
    polar_tsv = os.path.join(args.contacts_dir, 'polar_contacts.tsv')
    hydro_tsv = os.path.join(args.contacts_dir, 'hydrophobic_contacts.tsv')
    hbond_tsv = os.path.join(args.contacts_dir, 'hbond_contacts.tsv')

    df_all   = pd.read_csv(all_tsv,   sep='\t') if os.path.exists(all_tsv)   else pd.DataFrame()
    df_polar = pd.read_csv(polar_tsv, sep='\t') if os.path.exists(polar_tsv) else pd.DataFrame()
    df_hydro = pd.read_csv(hydro_tsv, sep='\t') if os.path.exists(hydro_tsv) else pd.DataFrame()
    df_hbond = pd.read_csv(hbond_tsv, sep='\t') if os.path.exists(hbond_tsv) else pd.DataFrame()

    epitopes       = load_epitopes(args.epitopes_csv)
    binder_lengths = load_binder_lengths(args.stats_tsv)

    # Discover designs
    if args.designs:
        designs = args.designs
    elif not df_all.empty:
        designs = sorted(df_all['design'].unique())
    else:
        print("ERROR: no designs found in all_contacts.tsv")
        return

    print(f"Plotting {len(designs)} designs -> {args.contacts_dir}\n")

    for design in designs:
        info = epitopes.get(design)
        if info is None:
            print(f"  WARNING: {design} not in epitopes CSV — skipping")
            continue

        peptide           = info['peptide']
        n_pep             = info['peptide_len']
        hotspot_positions = info['hotspot_positions']
        target_name       = info['target_name']

        # True binder length: prefer stats TSV, fall back to contacts TSV
        if design in binder_lengths:
            n_binder = binder_lengths[design]
        else:
            sub = df_all[df_all['design'] == design] if not df_all.empty else pd.DataFrame()
            n_binder = int(sub['binder_pos_0idx'].max()) + 1 if not sub.empty else 0
            print(f"  WARNING: {design} not in stats TSV, "
                  f"inferring binder len={n_binder} from contacts")

        # Subset each TSV to this design
        sub_all   = df_all  [df_all  ['design'] == design] if not df_all  .empty else pd.DataFrame()
        sub_polar = df_polar[df_polar['design'] == design] if not df_polar.empty else pd.DataFrame()
        sub_hydro = df_hydro[df_hydro['design'] == design] if not df_hydro.empty else pd.DataFrame()
        sub_hbond = df_hbond[df_hbond['design'] == design] if not df_hbond.empty else pd.DataFrame()

        if sub_all.empty:
            print(f"  WARNING: {design} has no contacts — skipping plot")
            continue

        # Build count matrices using true binder length
        all_mat   = build_count_matrix(sub_all,   n_binder, n_pep)
        hydro_mat = build_count_matrix(sub_hydro, n_binder, n_pep)

        # Split polar contacts by peptide atom: side-chain vs backbone N/O.
        if 'pep_atom_is_sidechain' not in sub_polar.columns and not sub_polar.empty:
            print(f"  WARNING: {design} polar_contacts.tsv lacks "
                  f"'pep_atom_is_sidechain' — regenerate with current "
                  f"af3_design_stats.py. Treating all polar as backbone.")
        polar_sc_mask = bool_col(sub_polar, 'pep_atom_is_sidechain')
        polar_sc_mat  = build_count_matrix(sub_polar[polar_sc_mask],  n_binder, n_pep)
        polar_bb_mat  = build_count_matrix(sub_polar[~polar_sc_mask], n_binder, n_pep)

        # Overlay marker cells: salt bridges (always sidechain) and H-bonds
        # (split by peptide side-chain vs backbone).
        saltbridge_cells = cells(sub_polar, bool_col(sub_polar, 'salt_bridge'))
        hbond_sc_mask    = bool_col(sub_hbond, 'pep_atom_is_sidechain')
        hbond_sc_cells   = cells(sub_hbond, hbond_sc_mask)
        hbond_bb_cells   = cells(sub_hbond, ~hbond_sc_mask)

        has_hbond = not sub_hbond.empty

        out_path = os.path.join(args.contacts_dir, f'{design}_contacts.png')
        print(f"  {design}  (binder={n_binder} res, pep={peptide})")
        plot_design_contacts(
            design            = design,
            all_mat           = all_mat,
            polar_sc_mat      = polar_sc_mat,
            polar_bb_mat      = polar_bb_mat,
            hydro_mat         = hydro_mat,
            saltbridge_cells  = saltbridge_cells,
            hbond_sc_cells    = hbond_sc_cells,
            hbond_bb_cells    = hbond_bb_cells,
            peptide           = peptide,
            hotspot_positions = hotspot_positions,
            target_name       = target_name,
            has_hbond         = has_hbond,
            out_path          = out_path,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()