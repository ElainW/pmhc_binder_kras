"""
plot_design_stats.py
====================
Produces separate PNG figures for my design stats.

Output files:
  01_charge_cysteine.png   — net charge bar + cysteine count scatter
  02_secondary_structure.png — stacked bar of helix/sheet/loop %
  03_fastrelax.png         — 9-panel strip plots for FastRelax metrics
  04_cms_heatmap.png       — 18×10 heatmap of CMS per peptide position
  05_plddt_ipsae.png       — AF2 monomer pLDDT and ipSAE binder-peptide

prame-9 and sars-6 flagged with asterisk (*) in plots where their
metrics are known to be anomalous (ipSAE, and noted in secondary structure).

Usage:
    python plot_design_stats.py \
        --stats_tsv    /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3_design_stats.tsv \
        --fastrelax_tsv /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/fastrelax_af3/fastrelax_af3_scores.tsv \
        --epitopes_csv /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/design_epitopes.csv \
        --sequences /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/monomer_fasta/af2_monomer_sequences.csv \
        --plddt_tsv    /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer/stats/monomer_scores_your_designs.tsv \
        --out_dir      /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3_plots/
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Designs with known anomalous metrics
FLAGGED = {}
FLAG_MARKER = '*'


# Charge formula: R + K + 0.1*H - D - E
def net_charge(seq: str) -> float:
    return (seq.count('R') + seq.count('K') + 0.1 * seq.count('H')
            - seq.count('D') - seq.count('E'))


def design_label(name: str) -> str:
    return f'{name}{FLAG_MARKER}' if name in FLAGGED else name


# ─────────────────────────────────────────────────────────────────────────────
# Plot 01: Net charge + cysteine
# ─────────────────────────────────────────────────────────────────────────────

def plot_charge_cysteine(df_epi: pd.DataFrame, df_seq: pd.DataFrame,
                         out_path: str):
    # Merge sequences from sequences file into epitopes
    df_seq['name'] = df_seq['name'].str.removesuffix("_af2pred")
    df_seq['name'] = df_seq['name'].str.lower()
    xlsx_seq = df_seq[df_seq['source']=='your_designs'].set_index('name')['sequence'].to_dict()
    rows = []
    for _, row in df_epi.iterrows():
        design = row['design']
        seq    = xlsx_seq.get(design, '')
        rows.append({
            'design':     design,
            'net_charge': round(net_charge(seq), 2),
            'n_cys':      seq.count('C'),
            'n_met':      seq.count('M'),
        })
    print()
    df = pd.DataFrame(rows).sort_values('net_charge')
    labels = [design_label(d) for d in df['design']]

    fig, axes = plt.subplots(1, 2, figsize=(10, len(labels)*0.25))

    # Net charge bar
    ax = axes[0]
    colors = ['#e74c3c' if v > -2 else '#3498db' for v in df['net_charge']]
    bars = ax.barh(labels, df['net_charge'], color=colors, edgecolor='white')
    ax.axvline(-2, color='black', lw=1.5, ls='--', alpha=0.7, label='threshold (−2)')
    ax.set_xlabel('Net charge (R+K+0.1H−D−E)', fontsize=11)
    ax.set_title('Net charge', fontsize=12)
    ax.legend(fontsize=9)
    for bar, val in zip(bars, df['net_charge']):
        ax.text(val - 0.2, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', ha='right', fontsize=8)

      # Cys + Met count scatter - stagger offsets to reduce label overlap
    ax = axes[1]
    ax.scatter(df['n_met'], df['n_cys'], s=80, color='#8e44ad', edgecolors='white',
               linewidth=0.5, zorder=3)
    try:
        from adjustText import adjust_text
        texts = [ax.text(r['n_met'], r['n_cys'], design_label(r['design']),
                         fontsize=7) for _, r in df.iterrows()]
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))
    except ImportError:
        for i_pt, (_, r) in enumerate(df.iterrows()):
            dy = 6 if i_pt % 2 == 0 else -11
            dx = 4 if i_pt % 3 != 2 else -44
            ax.annotate(design_label(r['design']),
                        (r['n_met'], r['n_cys']),
                        fontsize=7, ha='left', va='bottom',
                        xytext=(dx, dy), textcoords='offset points')
    ax.set_xlabel('Number of Met residues', fontsize=11)
    ax.set_ylabel('Number of Cys residues', fontsize=11)
    ax.set_title('Cys and Met counts\n(Cys omitted in ProteinMPNN by default)',
                 fontsize=11)
    ax.axhline(0, color='grey', lw=0.5, ls='--')

    if FLAGGED:
        fig.text(0.5, -0.02,
                 f'* flagged designs: {", ".join(sorted(FLAGGED))}',
                 ha='center', fontsize=8, color='grey')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 02: Secondary structure
# ─────────────────────────────────────────────────────────────────────────────

def plot_secondary_structure(df_stats: pd.DataFrame, out_path: str):
    cols = ['design', 'ss_helix_pct', 'ss_sheet_pct', 'ss_loop_pct', 'n_binder_res']
    df = df_stats[cols].dropna(subset=['ss_helix_pct']).copy()
    df = df.sort_values('ss_helix_pct', ascending=True)
    labels = [design_label(d) for d in df['design']]

    fig, ax = plt.subplots(figsize=(8, 7))
    x = np.arange(len(df))
    w = 0.6
    ax.barh(x, df['ss_helix_pct'], w, label='Helix', color='#2980b9')
    ax.barh(x, df['ss_sheet_pct'], w, left=df['ss_helix_pct'],
            label='Sheet', color='#e67e22')
    ax.barh(x, df['ss_loop_pct'],  w,
            left=df['ss_helix_pct'] + df['ss_sheet_pct'],
            label='Loop', color='#95a5a6')

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Percentage of binder residues (%)', fontsize=11)
    ax.set_title('Secondary structure composition (binder chain, post-FastRelax DSSP)',
                 fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 100)

    # Annotate n_binder_res
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(101, i, f'n={int(row["n_binder_res"])}  loop={row["ss_loop_pct"]:.1f}%',
                va='center', fontsize=7, color='grey')

    if FLAGGED:
        fig.text(0.5, -0.02,
                 f'* flagged designs: {", ".join(sorted(FLAGGED))}',
                 ha='center', fontsize=8, color='grey')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 03: FastRelax metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_fastrelax(df_fr: pd.DataFrame, out_path: str):
    metrics = [
        ('dG_separated',          'dG_separated (REU)',           True),
        ('dG_separated/dSASAx100',    'dG_separated/dSASA×100',           True),
        ('dSASA_int',             'dSASA_int (Å²)',                False),
        ('dSASA_polar',           'dSASA_polar (Å²)',              False),
        ('hbonds_int',            'H-bonds at interface',          False),
        ('delta_unsatHbonds',     'ΔUnsatisfied H-bonds',         True),
        ('sc_value',              'Shape complementarity',         False),
        ('packstat',              'Packstat',                      False),
        ('per_residue_energy_int','Per-residue interface energy',  True),
    ]

    # Sort designs consistently by dG_separated
    df = df_fr.dropna(subset=['dG_separated']).copy()
    df = df.sort_values(by='design', ascending=False)
    labels = [design_label(d) for d in df['design']]

    fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharey=True)
    axes = axes.flatten()

    for ax, (col, title, lower_better) in zip(axes, metrics):
        actual_col = col
        if col not in df.columns:
            variants = [c for c in df.columns
                        if c.replace('/','').replace('x','X') ==
                           col.replace('/','').replace('x','X')]
            if variants:
                actual_col = variants[0]
            else:
                ax.set_visible(False)
                continue
        col = actual_col
        vals = df[col].values
        # Color: green = better, red = worse
        if lower_better:
            norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
            colors = plt.cm.RdYlGn_r(norm_vals)
        else:
            norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
            colors = plt.cm.RdYlGn(norm_vals)

        bars = ax.barh(labels, vals, color=colors, edgecolor='white')
        vrange2 = vals.max() - vals.min()
        for bar, val in zip(bars, vals):
            ax.text(val + vrange2 * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', fontsize=6)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='y', labelsize=7)
        ax.axvline(np.median(vals), color='black', lw=1, ls='--',
                   alpha=0.6, label=f'median={np.median(vals):.2f}')
        ax.legend(fontsize=7, loc='lower right')

    if FLAGGED:
        fig.text(0.5, -0.01,
                 f'* flagged designs: {", ".join(sorted(FLAGGED))} '
                 f'(prame-9 includes B2M; sars-6 includes B2M)',
                 ha='center', fontsize=8, color='grey')

    fig.suptitle('FastRelax interface metrics (AF3 structures)', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 04: CMS heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_cms_heatmap(df_stats: pd.DataFrame, df_epi: pd.DataFrame,
                     out_path: str):
    # Build peptide sequence per design for column labels
    pep_map = df_epi.set_index('design')['peptide'].to_dict()
    hotspot_map = {
        row['design']: [int(x) for x in str(row['cms_hotspot_positions']).split(',') if x.strip()]
        for _, row in df_epi.iterrows()
    }

    # Collect CMS columns
    cms_cols_all = sorted(
        [c for c in df_stats.columns if c.startswith('cms_p') and
         c[5:].isdigit()],
        key=lambda x: int(x[5:])
    )
    max_pep = max(int(c[5:]) for c in cms_cols_all)

    # Sort designs by target name for grouping
    df = df_stats.dropna(subset=['cms_p1']).copy()
    df = df.merge(df_epi[['design', 'target_name', 'peptide_len']],
                  on='design', how='left')
    df = df.sort_values(['target_name', 'design'])

    n_designs = len(df)
    fig, ax = plt.subplots(figsize=(max_pep * 0.3 + 3, n_designs * 0.3 + 2))

    mat = np.zeros((n_designs, max_pep))
    for i, (_, row) in enumerate(df.iterrows()):
        for j in range(1, max_pep + 1):
            col = f'cms_p{j}'
            if col in row and not pd.isna(row[col]):
                mat[i, j - 1] = row[col]

    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', origin='upper',
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label='CMS score')

    # Y labels
    ylabels = [design_label(d) for d in df['design']]
    ax.set_yticks(range(n_designs))
    ax.set_yticklabels(ylabels, fontsize=8)

    # X labels: for each position, show p{i}\n{AA} using the design's own peptide
    # Since peptides differ per design, use a generic p{i} label on the axis
    # and annotate hotspots per design with blue dots
    ax.set_xticks(range(max_pep))
    ax.set_xticklabels([f'p{i+1}' for i in range(max_pep)], fontsize=9)
    ax.set_xlabel('Peptide position', fontsize=11)

    # Annotate peptide AA per cell (design-specific)
    for i, (_, row) in enumerate(df.iterrows()):
        design  = row['design']
        peptide = pep_map.get(design, '')
        for j in range(len(peptide)):
            aa  = peptide[j] if j < len(peptide) else ''
            val = mat[i, j]
            txt_color = 'white' if val > mat.max() * 0.6 else 'black'
            ax.text(j, i, aa, ha='center', va='center',
                    fontsize=6, color=txt_color, fontweight='bold')

    # Hotspot columns: blue border per design row
    for i, (_, row) in enumerate(df.iterrows()):
        design   = row['design']
        hotspots = hotspot_map.get(design, [])
        for pos in hotspots:
            j = pos - 1
            if 0 <= j < max_pep:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor='royalblue', lw=1.5
                )
                ax.add_patch(rect)

    ax.set_title('CMS per peptide position (binder vs single residue)\n'
                 'Blue border = hotspot position. AA label = peptide residue.',
                 fontsize=11)

    if FLAGGED:
        fig.text(0.5, -0.02,
                 f'* flagged designs: {", ".join(sorted(FLAGGED))}',
                 ha='center', fontsize=8, color='grey')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 05: pLDDT + ipSAE
# ─────────────────────────────────────────────────────────────────────────────

def plot_plddt_ipsae(df_plddt: pd.DataFrame, df_stats: pd.DataFrame,
                     out_path: str):
    # Normalise design names: strip 'author_' prefix
    df_plddt = df_plddt.copy()
    df_plddt['design'] = df_plddt['description'].str.replace('pT', 'pt',
                                                              regex=True)
    df_plddt['design'] = df_plddt['design'].str.removesuffix('_af2pred')

    # Merge
    df = df_plddt.merge(
        df_stats[['design', 'ipsae_binder_peptide', 'ipsae_n_contacts',
                  'af3_iptm', 'af3_ranking_score']],
        on='design', how='outer'
    )
    df = df.sort_values(by='design', ascending=False)
    labels = [design_label(d) for d in df['design']]

    fig, axes = plt.subplots(1, 3, figsize=(12, 12), sharey=True)

    # AF2 monomer pLDDT
    ax = axes[0]
    colors = ['#e74c3c' if d in FLAGGED else '#2ecc71' for d in df['design']]
    bars = ax.barh(labels, df['monomer_plddt'], color=colors, edgecolor='white')
    ax.set_xlabel('AF2 monomer mean pLDDT', fontsize=11)
    ax.set_title('AF2 monomer pLDDT\n(best model)', fontsize=11)
    ax.axvline(90, color='black', lw=1, ls='--', alpha=0.5, label='pLDDT=90')
    ax.legend(fontsize=8)
    print(df)
    for bar, val in zip(bars, df['monomer_plddt']):
        if not np.isnan(val):
            ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', fontsize=7)

    # ipSAE binder-peptide
    ax = axes[1]
    ipsae_vals = df['ipsae_binder_peptide'].fillna(0)
    colors = ['#e74c3c' if d in FLAGGED else '#3498db' for d in df['design']]
    bars = ax.barh(labels, ipsae_vals, color=colors, edgecolor='white')
    ax.set_xlabel('ipSAE binder–peptide', fontsize=11)
    ax.set_title('ipSAE binder–peptide\n(AF3 top seed PAE)', fontsize=11)
    ax.axvline(0.8, color='black', lw=1, ls='--', alpha=0.5, label='ipSAE=0.8')
    ax.legend(fontsize=8)
    for bar, val, design in zip(bars, ipsae_vals, df['design']):
        label = f'{val:.3f}'
        if design in FLAGGED:
            label += FLAG_MARKER
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=7)

    # ipSAE n_contacts
    ax = axes[2]
    nc_vals = df['ipsae_n_contacts'].fillna(0)
    colors = ['#e74c3c' if d in FLAGGED else '#9b59b6' for d in df['design']]
    bars = ax.barh(labels, nc_vals, color=colors, edgecolor='white')
    for bar, val in zip(bars, nc_vals):
        ax.text(val + nc_vals.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{int(val)}', va='center', fontsize=7)
    ax.set_xlabel('ipSAE n_contacts (PAE < 12 Å)', fontsize=11)
    ax.set_title('ipSAE contact pairs\n(binder–peptide, cutoff 12 Å)', fontsize=11)

    fig.suptitle('AF2 monomer pLDDT and AF3 ipSAE', fontsize=13, y=1.01)

    flag_note = (f'* flagged (anomalous ipSAE — back-face docking or B2M inclusion): '
                 f'{", ".join(sorted(FLAGGED))}')
    fig.text(0.5, -0.02, flag_note, ha='center', fontsize=8, color='grey')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stats_tsv',      required=True,
        help='af3_design_stats.tsv')
    parser.add_argument('--fastrelax_tsv',  required=True,
        help='fastrelax_af3_scores.tsv')
    parser.add_argument('--epitopes_csv',   required=True,
        help='design_epitopes.csv')
    parser.add_argument('--sequences',   required=True,
        help='af2_monomer_sequences.csv')
    parser.add_argument('--plddt_tsv',      required=True,
        help='TSV with columns: description, monomer_plddt')
    parser.add_argument('--out_dir',        required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    df_stats = pd.read_csv(args.stats_tsv,     sep='\t')
    df_fr    = pd.read_csv(args.fastrelax_tsv, sep='\t')
    df_epi   = pd.read_csv(args.epitopes_csv)
    df_seq   = pd.read_csv(args.sequences,     sep=",")
    df_plddt = pd.read_csv(args.plddt_tsv,     sep='\t')

    print(f'Loaded: {len(df_stats)} designs in stats TSV, '
          f'{len(df_fr)} in FastRelax TSV\n')

    plot_charge_cysteine(
        df_epi, df_seq,
        os.path.join(args.out_dir, '01_charge_cysteine.png')
    )
    plot_secondary_structure(
        df_stats,
        os.path.join(args.out_dir, '02_secondary_structure.png')
    )
    plot_fastrelax(
        df_fr,
        os.path.join(args.out_dir, '03_fastrelax.png')
    )
    plot_cms_heatmap(
        df_stats, df_epi,
        os.path.join(args.out_dir, '04_cms_heatmap.png')
    )
    plot_plddt_ipsae(
        df_plddt, df_stats,
        os.path.join(args.out_dir, '05_plddt_ipsae.png')
    )

    print('\nAll done.')


if __name__ == '__main__':
    main()