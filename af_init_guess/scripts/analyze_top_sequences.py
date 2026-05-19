#!/usr/bin/env python3
"""
Analyse a passing FASTA for sequence quality metrics.

Usage:
    python analyze_top_sequences.py \
        --fasta /path/to/passing_ipsae0.8.fa \
        --output /path/to/sequence_qc.tsv
"""

import re
import argparse
import pandas as pd

def charge(seq):
    return seq.count('R') + seq.count('K') + 0.1*seq.count('H') \
           - seq.count('D') - seq.count('E')

def parse_fasta(fasta_file):
    entries = []
    header = seq = None
    with open(fasta_file) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if header and seq:
                    entries.append((header, seq))
                header, seq = line, ''
            elif line:
                seq = (seq or '') + line
    if header and seq:
        entries.append((header, seq))
    return entries

def main(args):
    entries = parse_fasta(args.fasta)
    print(f"Total sequences: {len(entries)}")

    rows = []
    for header, seq in entries:
        token = header.lstrip('>').split(',')[0].strip()
        sample_m = re.search(r'\bsample=(\d+)', header)
        sample = int(sample_m.group(1)) if sample_m else None

        # Strip T= and lowercase for consistent name
        name = re.sub(r'_T=[\d.]+', '', token)
        name = re.sub(r'_T([\d.]+)$', lambda m: f'_t{m.group(1)}', name).lower()
        if sample:
            name = f"{name}_s{sample}"

        c    = charge(seq)
        nC   = seq.count('C')
        nM   = seq.count('M')
        nW   = seq.count('W')
        nP   = seq.count('P')
        length = len(seq)

        # Flags
        flags = []
        if c > -3.0:   flags.append('CHARGE_HIGH')
        if nC > 0:     flags.append('CYS')
        if nM >= 3:    flags.append('MET_HIGH')
        if nW > 0:     flags.append('TRP')

        rows.append({
            'name':    name,
            'length':  length,
            'charge':  round(c, 1),
            'n_C':     nC,
            'n_M':     nM,
            'n_W':     nW,
            'n_P':     nP,
            'flags':   '|'.join(flags) if flags else 'OK',
            'sequence': seq,
        })

    df = pd.DataFrame(rows)

    # Summary
    print(f"\n{'='*55}")
    print(f"Sequence QC summary (n={len(df)})")
    print(f"{'='*55}")
    print(f"  Length range:       {df['length'].min()}–{df['length'].max()} aa")
    print(f"  Charge range:       {df['charge'].min():.1f} to {df['charge'].max():.1f}")
    print(f"  Charge mean:        {df['charge'].mean():.1f}")
    print(f"\n  Flag counts:")
    print(f"    Charge > -3.0:    {(df['charge'] > -3.0).sum()}")
    print(f"    Charge -3.0:      {(df['charge'] == -3.0).sum()}  (borderline)")
    print(f"    Any Cys:          {(df['n_C'] > 0).sum()}")
    print(f"    Met >= 3:         {(df['n_M'] >= 3).sum()}")
    print(f"    Met == 2:         {(df['n_M'] == 2).sum()}  (acceptable)")
    print(f"    Met == 1:         {(df['n_M'] == 1).sum()}  (acceptable)")
    print(f"    Any Trp:          {(df['n_W'] > 0).sum()}")
    print(f"    Clean (OK):       {(df['flags'] == 'OK').sum()}")

    print(f"\n{'='*55}")
    print(f"Charge distribution")
    print(f"{'='*55}")
    print(df['charge'].value_counts().sort_index().to_string())

    print(f"\n{'='*55}")
    print(f"Flagged sequences")
    print(f"{'='*55}")
    flagged = df[df['flags'] != 'OK'].sort_values('charge', ascending=False)
    for _, r in flagged.iterrows():
        print(f"  {r['name']:<60} chg={r['charge']:>5.1f}  M={r['n_M']}  W={r['n_W']}  C={r['n_C']}  [{r['flags']}]")

    print(f"\n{'='*55}")
    print(f"Clean sequences (no flags)")
    print(f"{'='*55}")
    clean = df[df['flags'] == 'OK'].sort_values('charge')
    print(f"  {len(clean)} sequences pass all QC filters")
    for _, r in clean.iterrows():
        print(f"  {r['name']:<60} chg={r['charge']:>5.1f}")

    if args.output:
        df.drop(columns=['sequence']).to_csv(args.output, sep='\t', index=False)
        print(f"\nQC table saved: {args.output}")

    return df

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fasta',  required=True)
    p.add_argument('--output', default=None, help='TSV output path (optional)')
    return p.parse_args()

if __name__ == '__main__':
    main(parse_args())