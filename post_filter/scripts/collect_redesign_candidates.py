"""
collect_redesign_candidates.py
===============================
Collects designs that need ProteinMPNN redesign from:
  - tier1.tsv            (Tier 1 designs)
  - redesign_candidate.tsv
  - redesign_audit.tsv   (charge/cys flags)

A design needs redesign if ANY of:
  - flag_charge_redesign == True   (net_charge > -2, fix interface, redesign rest)
  - flag_cys_redesign    == True   (any Cys, all Cys positions to redesign)
  - hard flags in tier (from tier1/redesign_candidate: soft flags not included)

Fixed residue logic:
  interface_fixed_resnums  — positions within 8Å of pMHC (from charge_cysteine_audit)
  saltbridge_hotspot_p5_binder_resnums — salt bridge contacts at p5
  hbond_hotspot_p5_binder_resnums      — hbond contacts at p5

  fixed_resnums = union of all three (positions NOT to redesign)
  Cys resnums (cys_resnums) are NOT added to fixed — they must be redesigned away

Output:
  redesign_candidates.tsv — one row per design needing redesign, with:
    - all flags from both sources
    - fixed_resnums (merged set, sorted)
    - n_fixed_resnums
    - redesign_reasons (list of reasons)

Usage:
    python collect_redesign_candidates.py \
        --tier1_tsv           /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2/prioritization/tier1.tsv \
        --redesign_cand_tsv   /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2/prioritization/redesign_candidate.tsv \
        --audit_tsv           /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2/af3_nomsa/af3_design_stats.tsv \
        --out_path            /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2/prioritization/redesign_list.tmp
"""

import os
import ast
import argparse

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_resnum_list(val) -> list[int]:
    """
    Parse a residue number list stored as a string, e.g. '[12, 15, 16]' or '[]'.
    Returns a sorted list of ints.
    """
    if pd.isna(val) or str(val).strip() in ('', '[]', 'nan'):
        return []
    try:
        parsed = ast.literal_eval(str(val))
        return sorted(int(x) for x in parsed if str(x).strip())
    except Exception:
        return []


def merge_fixed_resnums(*lists: list[int]) -> list[int]:
    """Union of all residue number lists, sorted."""
    result = set()
    for lst in lists:
        result.update(lst)
    return sorted(result)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tier1_tsv',         required=True,
        help='tier1.tsv from prioritize_designs.py')
    parser.add_argument('--redesign_cand_tsv', required=True,
        help='redesign_candidate.tsv from prioritize_designs.py')
    parser.add_argument('--audit_tsv',         required=True,
        help='redesign_audit.tsv from af3_design_stats.py '
             '(columns: design, sequence, net_charge, n_cys, cys_resnums, '
             'interface_fixed_resnums, flag_charge_redesign, flag_cys_redesign, ...)')
    parser.add_argument('--out_path',          required=True,
        help='Output TSV path')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────────────
    df_tier1   = pd.read_csv(args.tier1_tsv,         sep='\t') if os.path.exists(args.tier1_tsv)         else pd.DataFrame()
    df_rcand   = pd.read_csv(args.redesign_cand_tsv, sep='\t') if os.path.exists(args.redesign_cand_tsv) else pd.DataFrame()
    df_audit   = pd.read_csv(args.audit_tsv,         sep='\t')

    # ── Combine tier1 + redesign_candidate ──────────────────────────────────
    df_priority = pd.concat([df_tier1, df_rcand], ignore_index=True).drop_duplicates('design')
    print(f"Tier1: {len(df_tier1)}  Redesign candidates: {len(df_rcand)}  "
          f"Combined unique: {len(df_priority)}")

    # ── Merge audit info ─────────────────────────────────────────────────────
    audit_cols = [
        'design', 'sequence', 'length', 'net_charge',
        'n_cys', 'cys_resnums', 'cys_interface_resnums', 'n_cys_interface',
        'n_met', 'interface_fixed_resnums', 'n_interface_fixed',
        'flag_charge_redesign', 'flag_cys_redesign',
        'pass_charge', 'pass_cys', 'pass_all',
    ]
    audit_present = [c for c in audit_cols if c in df_audit.columns]
    df = df_priority.merge(df_audit[audit_present], on='design', how='left',
                          suffixes=('_priority', '_audit'))

    # Check for conflicts between priority TSV and audit TSV values
    for check_col in ['net_charge', 'pass_all']:
        col_p = f'{check_col}_priority'
        col_a = f'{check_col}_audit'
        if col_p in df.columns and col_a in df.columns:
            # Compare — allow small float tolerance for net_charge
            if check_col == 'net_charge':
                mismatch = df[
                    (df[col_p].notna()) & (df[col_a].notna()) &
                    ((df[col_p] - df[col_a]).abs() > 0.01)
                ]
            else:
                mismatch = df[
                    (df[col_p].notna()) & (df[col_a].notna()) &
                    (df[col_p] != df[col_a])
                ]
            if len(mismatch):
                print(f"\nERROR: {check_col} mismatch between priority TSV and audit TSV "
                      f"for {len(mismatch)} design(s):")
                for _, r in mismatch.iterrows():
                    print(f"  {r['design']}: priority={r[col_p]}  audit={r[col_a]}")
            # Keep the audit version as authoritative (more complete source)
            df[check_col] = df[col_a]
            df = df.drop(columns=[col_p, col_a])
        elif check_col in df.columns:
            pass  # only one source, no conflict possible

    # ── Determine redesign need and reasons ──────────────────────────────────
    def _redesign_reasons(row) -> list[str]:
        reasons = []
        if row.get('flag_charge_redesign', False):
            reasons.append(f"charge>{row.get('net_charge','?')}")
        if row.get('flag_cys_redesign', False):
            reasons.append(f"cys={row.get('cys_resnums','?')}")
        # Hard flags from prioritization
        flags = row.get('hard_flags_triggered', '')
        if isinstance(flags, str) and flags not in ('', '[]', 'nan'):
            try:
                flag_list = ast.literal_eval(flags)
                reasons.extend(flag_list)
            except Exception:
                reasons.append(flags)
        return reasons

    df['redesign_reasons'] = df.apply(_redesign_reasons, axis=1)
    df['needs_redesign']   = df['redesign_reasons'].apply(lambda x: len(x) > 0)

    # ── Build merged fixed residue set ───────────────────────────────────────
    def _fixed_resnums(row) -> list[int]:
        interface_fixed = parse_resnum_list(row.get('interface_fixed_resnums', '[]'))
        sb_resnums      = parse_resnum_list(row.get('saltbridge_hotspot_p5_binder_resnums', '[]'))
        hb_resnums      = parse_resnum_list(row.get('hbond_hotspot_p5_binder_resnums', '[]'))
        # Cys resnums must NOT be fixed — they need to be redesigned away
        cys_resnums     = parse_resnum_list(row.get('cys_resnums', '[]'))
        merged = merge_fixed_resnums(interface_fixed, sb_resnums, hb_resnums)
        # Remove Cys positions from fixed set so MPNN redesigns them
        merged = [r for r in merged if r not in cys_resnums]
        return merged

    df['fixed_resnums']   = df.apply(_fixed_resnums, axis=1)
    df['n_fixed_resnums'] = df['fixed_resnums'].apply(len)
    df['fixed_resnums']   = df['fixed_resnums'].apply(str)

    # ── Filter to only designs needing redesign ───────────────────────────────
    df_redesign = df[df['needs_redesign']].copy()
    df_redesign['redesign_reasons'] = df_redesign['redesign_reasons'].apply(str)

    print(f"\nDesigns needing redesign: {len(df_redesign)}/{len(df)}")
    print(f"  (no redesign needed: {len(df) - len(df_redesign)})")

    # ── Column ordering ───────────────────────────────────────────────────────
    key_cols = [
        'design', 'tier', 'needs_redesign', 'redesign_reasons',
        'fixed_resnums', 'n_fixed_resnums',
        'sequence', 'length', 'net_charge', 'n_cys', 'cys_resnums',
        'cys_interface_resnums', 'n_cys_interface',
        'n_met', 'interface_fixed_resnums', 'n_interface_fixed',
        'saltbridge_hotspot_p5_binder_resnums', 'saltbridge_hotspot_p5_binder_aas',
        'hbond_hotspot_p5_binder_resnums', 'hbond_hotspot_p5_binder_aas',
        'flag_charge_redesign', 'flag_cys_redesign',
        'pass_charge', 'pass_cys', 'pass_all',
        'hard_flags_triggered', 'soft_flags_triggered',
        'n_hard_flags', 'n_soft_flags',
        'cms_hotspot_p5', 'n_contacts_hotspot_p5_polar',
        'n_contacts_hotspot_p5_hbond', 'n_saltbridge_hotspot_p5',
        'ipsae_binder_peptide', 'dG_separated', 'surface_hydrophobicity',
    ]
    present  = [c for c in key_cols if c in df_redesign.columns]
    extras   = [c for c in df_redesign.columns if c not in present]
    df_redesign = df_redesign[present + extras]
    df_redesign = df_redesign.sort_values(
        ['tier', 'n_hard_flags', 'cms_hotspot_p5'],
        ascending=[True, True, False]
    )

    df_redesign.to_csv(args.out_path, sep='\t', index=False)
    print(f"\nSaved -> {args.out_path}")

    # ── Output clean designs (no redesign needed) ─────────────────────────────
    df_clean = df[~df['needs_redesign']].copy()
    # Compute fixed_resnums for clean designs too (needed for MPNN even if no redesign)
    df_clean['fixed_resnums']   = df_clean.apply(_fixed_resnums, axis=1).apply(str)
    df_clean['n_fixed_resnums'] = df_clean['fixed_resnums'].apply(
        lambda x: len(ast.literal_eval(x)) if x not in ('[]','') else 0
    )
    df_clean['redesign_reasons'] = '[]'
    clean_present = [c for c in present + extras if c in df_clean.columns]
    df_clean = df_clean[clean_present].sort_values('cms_hotspot_p5', ascending=False)
    clean_path = args.out_path.replace('.tmp', '_clean.tmp')
    df_clean.to_csv(clean_path, sep='\t', index=False)
    print(f"Clean designs (no redesign) -> {clean_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'Design':<45} {'tier':<22} {'reasons'}")
    print('-' * 90)
    for _, r in df_redesign.iterrows():
        print(f"  {r['design']:<43} {r['tier']:<22} {r['redesign_reasons']}")

    print(f"\nNo redesign needed ({len(df_clean)} designs — ready for experimental validation):")
    for _, r in df_clean.sort_values('cms_hotspot_p5', ascending=False).iterrows():
        sb  = int(r.get('n_saltbridge_hotspot_p5', 0) or 0)
        hb  = int(r.get('n_contacts_hotspot_p5_hbond', 0) or 0)
        cms = r.get('cms_hotspot_p5', float('nan'))
        print(f"  {r['design']:<45} tier={r['tier']:<12} "
              f"cms_p5={cms:.1f}  SB={sb}  hbond_p5={hb}  "
              f"fixed_n={r.get('n_fixed_resnums', 0)}")


if __name__ == '__main__':
    main()