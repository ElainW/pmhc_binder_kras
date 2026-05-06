"""
plot_design_stats.py
====================
Produces separate PNG figures for my design stats.

Output files:
  01_charge_cysteine.png    — net charge bar + cysteine count scatter
  02_secondary_structure.png — stacked bar of helix/sheet/loop %
  03_fastrelax.png          — 9-panel strip plots for FastRelax metrics
  04_bindcraft.png          — 4-panel strip plots for BindCraft metrics
  05_cms_heatmap.png        — heatmap of CMS per peptide position
  06_plddt_ipsae.png        — 7-panel: AF2 monomer pLDDT, AF2 initial guess pLDDT,
                              pLDDT target, binder RMSD, target RMSD,
                              ipSAE binder-peptide, ipSAE n_contacts

For hbonds_int and delta_unsatHbonds (plot 03) and buns_delta_unsat (plot 04):
  bar length = raw value
  number annotated on bar = value / nres_int (or nres_all for buns)

Usage:
    python plot_design_stats.py \
        --stats_tsv       /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3_design_stats.tsv \
        --epitopes_csv    /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/design_epitopes.csv \
        --sequences       /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/monomer_fasta/af2_monomer_sequences.csv \
        --plddt_tsv       /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer/stats/monomer_scores_your_designs.tsv \
        --specificity_tsv /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
        --out_dir         /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3_plots/
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FLAGGED = {}
FLAG_MARKER = '*'

BINDCRAFT_THRESHOLDS = {
    'buns_delta_unsat':       4,
    'surface_hydrophobicity': 0.35,
    'interface_n_K':          3,
    'interface_n_M':          3,
}

FASTRELAX_PER_RES = {
    'hbonds_int':        'nres_int',
    'delta_unsatHbonds': 'nres_int',
}

BINDCRAFT_PER_RES = {
    'buns_delta_unsat': 'nres_all',
}


def net_charge(seq: str) -> float:
    return (seq.count('R') + seq.count('K') + 0.1 * seq.count('H')
            - seq.count('D') - seq.count('E'))


def design_label(name: str) -> str:
    return f'{name}{FLAG_MARKER}' if name in FLAGGED else name


# ─────────────────────────────────────────────────────────────────────────────
# Shared bar plot helper
# ─────────────────────────────────────────────────────────────────────────────

def _barh_panel(ax, labels, vals, norm_vals_for_color, title,
                lower_better, median_label,
                bar_annotations=None,
                annotation_suffix='',
                threshold=None, threshold_label=None):
    """
    bar length = vals (raw)
    annotation on bar = bar_annotations if provided, else vals
    """
    vrange = vals.max() - vals.min()
    colors = plt.cm.RdYlGn_r(norm_vals_for_color) if lower_better \
             else plt.cm.RdYlGn(norm_vals_for_color)

    bars = ax.barh(labels, vals, color=colors, edgecolor='white')

    annot = bar_annotations if bar_annotations is not None else vals
    for bar, raw_val, ann_val in zip(bars, vals, annot):
        x_pos = raw_val + vrange * 0.01
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{ann_val:.2f}{annotation_suffix}',
                va='center', fontsize=6)

    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='y', labelsize=7)
    ax.axvline(np.median(vals), color='black', lw=1, ls='--',
               alpha=0.6, label=f'median={np.median(vals):.2f}')
    if threshold is not None:
        ax.axvline(threshold, color='royalblue', lw=1, ls=':',
                   alpha=0.8, label=threshold_label or f'thresh={threshold}')
    ax.legend(fontsize=7, loc='lower right')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 01: Net charge + cysteine
# ─────────────────────────────────────────────────────────────────────────────

def plot_charge_cysteine(df_epi: pd.DataFrame, df_seq: pd.DataFrame,
                         out_path: str):
    df_seq = df_seq.copy()
    df_seq['name'] = df_seq['name'].str.removesuffix('_af2pred').str.lower()
    xlsx_seq = df_seq[df_seq['source'] == 'your_designs'].set_index('name')['sequence'].to_dict()

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
    df = pd.DataFrame(rows).sort_values('net_charge')
    labels = [design_label(d) for d in df['design']]

    fig, axes = plt.subplots(1, 2, figsize=(10, max(4, len(labels) * 0.25)))

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

    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.28)))
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

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(101, i, f'n={int(row["n_binder_res"])}  loop={row["ss_loop_pct"]:.1f}%',
                va='center', fontsize=7, color='grey')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 03: FastRelax metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_fastrelax(df_stats: pd.DataFrame, out_path: str):
    metrics = [
        ('dG_separated',           'dG_separated (REU)',                               True),
        ('dG_separated/dSASAx100', 'dG_separated/dSASA×100',                          True),
        ('dSASA_int',              'dSASA_int (Å²)',                                   False),
        ('dSASA_polar',            'dSASA_polar (Å²)',                                 False),
        ('hbonds_int',             'H-bonds at interface\n(bar=raw, label=per iface res)', False),
        ('delta_unsatHbonds',      'ΔUnsatisfied H-bonds\n(bar=raw, label=per iface res)', True),
        ('sc_value',               'Shape complementarity',                            False),
        ('packstat',               'Packstat',                                         False),
        ('per_residue_energy_int', 'Per-residue interface energy',                     True),
    ]

    df = df_stats.dropna(subset=['dG_separated']).copy()
    df = df.sort_values('design', ascending=False)
    labels = [design_label(d) for d in df['design']]

    fig, axes = plt.subplots(3, 3, figsize=(16, max(14, len(df) * 0.55)), sharey=True)
    axes = axes.flatten()

    for ax, (col, title, lower_better) in zip(axes, metrics):
        actual_col = col
        if col not in df.columns:
            variants = [c for c in df.columns
                        if c.replace('/', '').replace('x', 'X') ==
                           col.replace('/', '').replace('x', 'X')]
            if variants:
                actual_col = variants[0]
            else:
                ax.set_visible(False)
                continue

        vals = df[actual_col].fillna(0).values
        vrange = vals.max() - vals.min()
        norm_vals = (vals - vals.min()) / (vrange + 1e-9)

        if actual_col in FASTRELAX_PER_RES:
            norm_col = FASTRELAX_PER_RES[actual_col]
            bar_annotations = (vals / df[norm_col].replace(0, np.nan).values
                               if norm_col in df.columns else None)
        else:
            bar_annotations = None

        _barh_panel(ax, labels, vals, norm_vals, title, lower_better,
                    median_label=f'median={np.median(vals):.2f}',
                    bar_annotations=bar_annotations)

    fig.suptitle('FastRelax interface metrics (AF3 structures)', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 04: BindCraft metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_bindcraft(df_stats: pd.DataFrame, out_path: str):
    metrics = [
        ('buns_delta_unsat',       'Buried unsat H-bonds (complex)\n(bar=raw, label=per residue)', True),
        ('surface_hydrophobicity', 'Fraction exposed hydrophobic',  True),
        ('interface_n_K',          'Lys at interface',              True),
        ('interface_n_M',          'Met at interface',              True),
    ]

    df = df_stats.sort_values('design', ascending=False).copy()
    labels = [design_label(d) for d in df['design']]

    fig, axes = plt.subplots(2, 2, figsize=(10, max(8, len(df) * 0.4)), sharey=True)
    axes = axes.flatten()

    for ax, (col, title, lower_better) in zip(axes, metrics):
        actual_col = col
        if col not in df.columns:
            variants = [c for c in df.columns
                        if c.replace('/', '').replace('x', 'X') ==
                           col.replace('/', '').replace('x', 'X')]
            if variants:
                actual_col = variants[0]
            else:
                ax.set_title(title + '\n[column not found]', fontsize=9, color='red')
                ax.set_visible(False)
                continue

        vals = df[actual_col].fillna(0).values
        vrange = vals.max() - vals.min()
        norm_vals = (vals - vals.min()) / (vrange + 1e-9)

        if actual_col in BINDCRAFT_PER_RES:
            norm_col = BINDCRAFT_PER_RES[actual_col]
            bar_annotations = (vals / df[norm_col].replace(0, np.nan).values
                               if norm_col in df.columns else None)
        else:
            bar_annotations = None

        thresh = BINDCRAFT_THRESHOLDS.get(col)
        _barh_panel(ax, labels, vals, norm_vals, title, lower_better,
                    median_label=f'median={np.median(vals):.2f}',
                    bar_annotations=bar_annotations,
                    threshold=thresh,
                    threshold_label=f'BindCraft thresh={thresh}' if thresh else None)

    fig.suptitle('BindCraft metrics (AF3 structures)', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 05: CMS heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_cms_heatmap(df_stats: pd.DataFrame, df_epi: pd.DataFrame,
                     out_path: str):
    pep_map = df_epi.set_index('design')['peptide'].to_dict()
    hotspot_map = {
        row['design']: [int(x) for x in str(row['cms_hotspot_positions']).split(',')
                        if x.strip()]
        for _, row in df_epi.iterrows()
    }

    cms_cols_all = sorted(
        [c for c in df_stats.columns if c.startswith('cms_p') and c[5:].isdigit()],
        key=lambda x: int(x[5:])
    )
    max_pep = max(int(c[5:]) for c in cms_cols_all)

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

    ylabels = [design_label(d) for d in df['design']]
    ax.set_yticks(range(n_designs))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xticks(range(max_pep))
    ax.set_xticklabels([f'p{i+1}' for i in range(max_pep)], fontsize=9)
    ax.set_xlabel('Peptide position', fontsize=11)

    for i, (_, row) in enumerate(df.iterrows()):
        peptide = pep_map.get(row['design'], '')
        for j in range(len(peptide)):
            val = mat[i, j]
            txt_color = 'white' if val > mat.max() * 0.6 else 'black'
            ax.text(j, i, peptide[j] if j < len(peptide) else '',
                    ha='center', va='center', fontsize=6,
                    color=txt_color, fontweight='bold')

    for i, (_, row) in enumerate(df.iterrows()):
        for pos in hotspot_map.get(row['design'], []):
            j = pos - 1
            if 0 <= j < max_pep:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor='royalblue', lw=1.5
                ))

    ax.set_title('CMS per peptide position (binder vs single residue)\n'
                 'Blue border = hotspot. AA label = peptide residue.', fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 06: pLDDT + ipSAE  (7 panels)
#
# Panel layout (1 row × 7 cols):
#   [0] AF2 monomer pLDDT       (from --plddt_tsv)
#   [1] AF2 initial guess pLDDT (plddt_binder from --specificity_tsv)
#   [2] AF2 initial guess target pLDDT (plddt_target)
#   [3] Binder aligned RMSD     (binder_aligned_rmsd, lower=better)
#   [4] Target aligned RMSD     (target_aligned_rmsd, lower=better)
#   [5] ipSAE binder-peptide
#   [6] ipSAE n_contacts
# ─────────────────────────────────────────────────────────────────────────────

def plot_plddt_ipsae(df_plddt: pd.DataFrame, df_stats: pd.DataFrame,
                     df_spec: pd.DataFrame, out_path: str):
    # Normalise monomer pLDDT design names
    df_plddt = df_plddt.copy()
    df_plddt['design'] = (df_plddt['description']
                          .str.replace('pT', 'pt', regex=True)
                          .str.removesuffix('_af2pred'))

    # Normalise specificity TSV design names
    df_spec = df_spec.copy()
    df_spec['design'] = (df_spec['design']
                         .str.replace('pT', 'pt', regex=True)
                         .str.removesuffix('_af2pred'))

    # Merge everything
    df = df_plddt[['design', 'monomer_plddt']].merge(
        df_stats[['design', 'ipsae_binder_peptide', 'ipsae_n_contacts',
                  'af3_iptm', 'af3_ranking_score']],
        on='design', how='outer'
    ).merge(
        df_spec[['design', 'plddt_binder', 'plddt_target',
                 'binder_aligned_rmsd', 'target_aligned_rmsd']],
        on='design', how='left'
    )
    df = df.sort_values('design', ascending=False)
    labels = [design_label(d) for d in df['design']]
    n = len(df)

    fig, axes = plt.subplots(1, 7, figsize=(28, max(5, n * 0.3)), sharey=True)

    # ── Panel 0: AF2 monomer pLDDT ──────────────────────────────────────────
    ax = axes[0]
    colors = ['#e74c3c' if d in FLAGGED else '#2ecc71' for d in df['design']]
    bars = ax.barh(labels, df['monomer_plddt'].fillna(0), color=colors, edgecolor='white')
    ax.axvline(90, color='black', lw=1, ls='--', alpha=0.5, label='pLDDT=90')
    ax.set_xlabel('pLDDT', fontsize=9)
    ax.set_title('AF2 monomer\npLDDT', fontsize=10)
    ax.legend(fontsize=7)
    for bar, val in zip(bars, df['monomer_plddt']):
        if pd.notna(val) and val > 0:
            ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', fontsize=6)

    # ── Panel 1: AF2 initial guess pLDDT (binder) ───────────────────────────
    ax = axes[1]
    vals = df['plddt_binder'].fillna(0)
    colors = ['#e74c3c' if d in FLAGGED else '#27ae60' for d in df['design']]
    bars = ax.barh(labels, vals, color=colors, edgecolor='white')
    ax.axvline(90, color='black', lw=1, ls='--', alpha=0.5, label='pLDDT=90')
    ax.set_xlabel('pLDDT', fontsize=9)
    ax.set_title('AF2 initial guess\nbinder pLDDT', fontsize=10)
    ax.legend(fontsize=7)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', fontsize=6)

    # ── Panel 2: AF2 initial guess pLDDT (target) ───────────────────────────
    ax = axes[2]
    vals = df['plddt_target'].fillna(0)
    colors = ['#e74c3c' if d in FLAGGED else '#16a085' for d in df['design']]
    bars = ax.barh(labels, vals, color=colors, edgecolor='white')
    ax.axvline(90, color='black', lw=1, ls='--', alpha=0.5, label='pLDDT=90')
    ax.set_xlabel('pLDDT', fontsize=9)
    ax.set_title('AF2 initial guess\ntarget pLDDT', fontsize=10)
    ax.legend(fontsize=7)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', fontsize=6)

    # ── Panel 3: Binder aligned RMSD (lower = better) ───────────────────────
    ax = axes[3]
    vals = df['binder_aligned_rmsd'].fillna(0)
    vmax = vals.max()
    norm_vals = (vals - vals.min()) / (vmax - vals.min() + 1e-9)
    colors = plt.cm.RdYlGn_r(norm_vals)   # lower RMSD = greener
    bars = ax.barh(labels, vals, color=colors, edgecolor='white')
    ax.axvline(np.median(vals[vals > 0]), color='black', lw=1, ls='--',
               alpha=0.5, label=f'median={np.median(vals[vals>0]):.2f}')
    ax.set_xlabel('RMSD (Å)', fontsize=9)
    ax.set_title('AF2 initial guess\nbinder aligned RMSD', fontsize=10)
    ax.legend(fontsize=7)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(val + vmax * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', fontsize=6)

    # ── Panel 4: Target aligned RMSD (lower = better) ───────────────────────
    ax = axes[4]
    vals = df['target_aligned_rmsd'].fillna(0)
    vmax = vals.max()
    norm_vals = (vals - vals.min()) / (vmax - vals.min() + 1e-9)
    colors = plt.cm.RdYlGn_r(norm_vals)
    bars = ax.barh(labels, vals, color=colors, edgecolor='white')
    ax.axvline(np.median(vals[vals > 0]), color='black', lw=1, ls='--',
               alpha=0.5, label=f'median={np.median(vals[vals>0]):.2f}')
    ax.set_xlabel('RMSD (Å)', fontsize=9)
    ax.set_title('AF2 initial guess\ntarget aligned RMSD', fontsize=10)
    ax.legend(fontsize=7)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(val + vmax * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', fontsize=6)

    # ── Panel 5: ipSAE binder-peptide ───────────────────────────────────────
    ax = axes[5]
    ipsae_vals = df['ipsae_binder_peptide'].fillna(0)
    colors = ['#e74c3c' if d in FLAGGED else '#3498db' for d in df['design']]
    bars = ax.barh(labels, ipsae_vals, color=colors, edgecolor='white')
    ax.axvline(0.8, color='black', lw=1, ls='--', alpha=0.5, label='ipSAE=0.8')
    ax.set_xlabel('ipSAE', fontsize=9)
    ax.set_title('ipSAE binder–peptide\n(AF3 top seed PAE)', fontsize=10)
    ax.legend(fontsize=7)
    for bar, val, d in zip(bars, ipsae_vals, df['design']):
        lbl = f'{val:.3f}' + (FLAG_MARKER if d in FLAGGED else '')
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                lbl, va='center', fontsize=6)

    # ── Panel 6: ipSAE n_contacts ────────────────────────────────────────────
    ax = axes[6]
    nc_vals = df['ipsae_n_contacts'].fillna(0)
    colors = ['#e74c3c' if d in FLAGGED else '#9b59b6' for d in df['design']]
    bars = ax.barh(labels, nc_vals, color=colors, edgecolor='white')
    for bar, val in zip(bars, nc_vals):
        ax.text(val + nc_vals.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{int(val)}', va='center', fontsize=6)
    ax.set_xlabel('n contacts', fontsize=9)
    ax.set_title('ipSAE contact pairs\n(binder–peptide, PAE < 12 Å)', fontsize=10)

    fig.suptitle('pLDDT, RMSD (AF2 initial guess) and AF3 ipSAE', fontsize=13, y=1.01)
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
    parser.add_argument('--stats_tsv',       required=True, help='af3_design_stats.tsv')
    parser.add_argument('--epitopes_csv',    required=True, help='design_epitopes.csv')
    parser.add_argument('--sequences',       required=True, help='af2_monomer_sequences.csv')
    parser.add_argument('--plddt_tsv',       required=True,
        help='TSV with columns: description, monomer_plddt')
    parser.add_argument('--specificity_tsv', required=True,
        help='merged_scores_ranked.tsv with plddt_binder, plddt_target, '
             'binder_aligned_rmsd, target_aligned_rmsd')
    parser.add_argument('--out_dir',         required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_stats = pd.read_csv(args.stats_tsv,       sep='\t')
    df_epi   = pd.read_csv(args.epitopes_csv)
    df_seq   = pd.read_csv(args.sequences)
    df_plddt = pd.read_csv(args.plddt_tsv,       sep='\t')
    df_spec  = pd.read_csv(args.specificity_tsv, sep='\t')

    print(f'Loaded: {len(df_stats)} designs\n')

    plot_charge_cysteine(df_epi, df_seq,
        os.path.join(args.out_dir, '01_charge_cysteine.png'))
    plot_secondary_structure(df_stats,
        os.path.join(args.out_dir, '02_secondary_structure.png'))
    plot_fastrelax(df_stats,
        os.path.join(args.out_dir, '03_fastrelax.png'))
    plot_bindcraft(df_stats,
        os.path.join(args.out_dir, '04_bindcraft.png'))
    plot_cms_heatmap(df_stats, df_epi,
        os.path.join(args.out_dir, '05_cms_heatmap.png'))
    plot_plddt_ipsae(df_plddt, df_stats, df_spec,
        os.path.join(args.out_dir, '06_plddt_ipsae.png'))

    print('\nAll done.')


if __name__ == '__main__':
    main()