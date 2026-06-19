"""
compare_contacts.py

Compares relaxed vs unrelaxed contact_summary.tsv files to identify
designs where p5 (D12 neoepitope) contacts are stable, gained, or lost
after FastRelax. Stable contacts = higher confidence hits for Round 3.

Usage:
    python compare_contacts.py \
        --relaxed   ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/contact_relaxed/contact_summary.tsv \
        --unrelaxed ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/contact_unrelaxed/contact_summary.tsv \
        --out_dir   ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--relaxed',   required=True, help='Relaxed contact_summary.tsv')
    parser.add_argument('--unrelaxed', required=True, help='Unrelaxed contact_summary.tsv')
    parser.add_argument('--out_dir',   required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rl = pd.read_csv(args.relaxed,   sep='\t')
    ul = pd.read_csv(args.unrelaxed, sep='\t')

    m = rl.merge(ul, on='design', suffixes=('_rel', '_unrel'))

    # ── Delta columns ─────────────────────────────────────────────────────────
    m['delta_all']        = m['n_all_contacts_rel']      - m['n_all_contacts_unrel']
    m['delta_polar']      = m['n_polar_contacts_rel']    - m['n_polar_contacts_unrel']
    m['delta_p5_all']     = m['n_contacts_p5_all_rel']   - m['n_contacts_p5_all_unrel']
    m['delta_p5_polar']   = m['n_contacts_p5_polar_rel'] - m['n_contacts_p5_polar_unrel']
    m['delta_sb']         = m['n_p5_saltbridge_rel']     - m['n_p5_saltbridge_unrel']

    # ── p5 stability classification ───────────────────────────────────────────
    has_p5_rel   = m['n_contacts_p5_all_rel']   > 0
    has_p5_unrel = m['n_contacts_p5_all_unrel'] > 0
    has_sb_rel   = m['n_p5_saltbridge_rel']      > 0
    has_sb_unrel = m['n_p5_saltbridge_unrel']    > 0

    m['p5_status'] = 'no_contact'
    m.loc[has_p5_rel   & has_p5_unrel,  'p5_status'] = 'stable'
    m.loc[has_p5_rel   & ~has_p5_unrel, 'p5_status'] = 'gained'
    m.loc[~has_p5_rel  & has_p5_unrel,  'p5_status'] = 'lost'

    m['sb_status'] = 'no_sb'
    m.loc[has_sb_rel   & has_sb_unrel,  'sb_status'] = 'stable'
    m.loc[has_sb_rel   & ~has_sb_unrel, 'sb_status'] = 'gained'
    m.loc[~has_sb_rel  & has_sb_unrel,  'sb_status'] = 'lost'

    # ── Use backbone_group from relaxed (both should be same) ─────────────────
    bg_col = 'backbone_group_rel' if 'backbone_group_rel' in m.columns else 'backbone_group'

    out_cols = [
        'design', bg_col,
        'n_all_contacts_unrel',   'n_all_contacts_rel',   'delta_all',
        'n_contacts_p5_all_unrel','n_contacts_p5_all_rel','delta_p5_all',
        'n_contacts_p5_polar_unrel','n_contacts_p5_polar_rel','delta_p5_polar',
        'n_p5_saltbridge_unrel',  'n_p5_saltbridge_rel',  'delta_sb',
        'p5_status', 'sb_status',
    ]
    result = m[out_cols].rename(columns={bg_col: 'backbone_group'})
    result = result.sort_values(
        ['backbone_group', 'n_contacts_p5_polar_rel', 'delta_p5_polar'],
        ascending=[True, False, False]
    )

    # ── Save full comparison table ────────────────────────────────────────────
    out_path = os.path.join(args.out_dir, 'contact_comparison.tsv')
    result.to_csv(out_path, sep='\t', index=False)
    print(f"Saved → {out_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    n = len(result)
    p5_counts = result['p5_status'].value_counts()
    sb_counts = result['sb_status'].value_counts()

    print(f"\n── p5 (D12) contact stability across {n} designs ──")
    print(f"  Stable  (both unrelaxed & relaxed):  {p5_counts.get('stable',  0)}")
    print(f"  Gained  (relaxed only):               {p5_counts.get('gained',  0)}")
    print(f"  Lost    (unrelaxed only):             {p5_counts.get('lost',    0)}")
    print(f"  No contact in either:                 {p5_counts.get('no_contact', 0)}")

    print(f"\n── p5 salt bridge (Arg/Lys↔D12) stability ──")
    print(f"  Stable:  {sb_counts.get('stable', 0)}")
    print(f"  Gained:  {sb_counts.get('gained', 0)}")
    print(f"  Lost:    {sb_counts.get('lost',   0)}")
    print(f"  None:    {sb_counts.get('no_sb',  0)}")

    print(f"\n── Top designs by relaxed p5 polar contacts ──")
    top = result[result['n_contacts_p5_polar_rel'] > 0].sort_values(
        ['n_contacts_p5_polar_rel', 'delta_p5_polar'], ascending=False
    )
    print(top[[
        'design', 'backbone_group',
        'n_contacts_p5_polar_unrel', 'n_contacts_p5_polar_rel',
        'n_p5_saltbridge_unrel', 'n_p5_saltbridge_rel',
        'p5_status', 'sb_status',
    ]].to_string(index=False))

    # ── Plot: scatter delta_all vs delta_p5_polar, colored by p5_status ──────
    color_map = {'stable': '#2ecc71', 'gained': '#3498db',
                 'lost': '#e74c3c', 'no_contact': '#bdc3c7'}
    colors = result['p5_status'].map(color_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: total contacts delta vs p5 polar delta
    ax = axes[0]
    for status, grp in result.groupby('p5_status'):
        ax.scatter(grp['delta_all'], grp['delta_p5_polar'],
                   c=color_map[status], label=status, s=60, alpha=0.8, edgecolors='white')
    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.axvline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('Δ total contacts (relaxed − unrelaxed)', fontsize=11)
    ax.set_ylabel('Δ p5 polar contacts (relaxed − unrelaxed)', fontsize=11)
    ax.set_title('Interface contact changes after FastRelax', fontsize=12)
    ax.legend(title='p5 status', fontsize=9)

    # Right: bar chart of p5 polar contacts in relaxed vs unrelaxed — all 49 designs
    ax2 = axes[1]
    all_sorted = result.sort_values(
        ['n_contacts_p5_polar_rel', 'n_contacts_p5_polar_unrel'],
        ascending=False
    )
    x = np.arange(len(all_sorted))
    w = 0.35
    ax2.bar(x - w/2, all_sorted['n_contacts_p5_polar_unrel'], w,
            color='#95a5a6', label='Unrelaxed', edgecolor='white')
    ax2.bar(x + w/2, all_sorted['n_contacts_p5_polar_rel'],   w,
            color='#e74c3c', label='Relaxed',   edgecolor='white')
    ax2.set_xticks(x)
    short_names = [d.replace('orient_', '').replace('_af2pred', '')
                   for d in all_sorted['design']]
    ax2.set_xticklabels(short_names, rotation=90, ha='right', fontsize=6)
    ax2.set_ylabel('p5 polar contacts', fontsize=11)
    ax2.set_title(f'p5 polar contacts: all {len(all_sorted)} designs', fontsize=11)
    ax2.legend(fontsize=9)

    # Widen figure to accommodate all labels
    fig.set_size_inches(max(14, len(all_sorted) * 0.35), 6)

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, 'contact_comparison.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot → {fig_path}")


if __name__ == '__main__':
    main()