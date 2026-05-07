#!/usr/bin/env python3
"""
redesign_w_charge_residue_filter.py

Wrapper around standalone ProteinMPNN (protein_mpnn_run.py) that:
  1. Enforces net charge <= max_charge via rejection sampling
  2. Omits cysteines via --omit_AAs (default: C)
  3. Fixes interface residues so the binding geometry is preserved

Two modes:

  Mode A — new backbone PDBs (original behaviour):
    Pass --pdb_dir. No interface masking; all of chain A is designed.

  Mode B — redesign from af3_design_stats.py + collect_redesign_candidates.py + surface_hydrophobicity_resnums.py ->  redesign_list.tsv:

    redesign_list.tsv schema (scripts above):
    design   tier    needs_redesign  redesign_reasons
    fixed_resnums   n_fixed_resnums sequence        net_charge
    n_cys   cys_resnums     cys_interface_resnums   n_cys_interface n_met
    interface_fixed_resnums n_interface_fixed        saltbridge_hotspot_p5_binder_resnums    saltbridge_hotspot_p5_binder_aas
    hbond_hotspot_p5_binder_resnums hbond_hotspot_p5_binder_aas        flag_charge_redesign    flag_cys_redesign
    pass_charge     pass_cys        pass_all        hard_flags_triggered    soft_flags_triggered    n_hard_flags
    n_soft_flags    cms_hotspot_p5  n_contacts_hotspot_p5_polar        n_contacts_hotspot_p5_hbond     n_saltbridge_hotspot_p5
    ipsae_binder_peptide    dG_separated    surface_hydrophobicity     has_salt_bridge_p5      flag_cms_hotspot_p5
    ipsae_n_contacts        dG_sep_norm     dSASA_int       hbonds_int hbonds_per_res  delta_unsatHbonds
    delta_unsat_per_res     sc_value        packstat        buns_delta_unsat        interface_n_K   interface_n_M
    n_binder_res    ss_helix_pct    ss_loop_pct     surface_hydrophobic_resnums     surface_hydrophobic_aas
    surface_hydrophobicity_check

    For each selected design:
      - Reads interface_fixed_resnums (binder chain A residues contacting the
        peptide in the AF3 relaxed structure at < 4 Å)
      - Reads cys_resnums (all Cys positions on chain A, regardless of location)
      - Fixed positions = interface_fixed_resnums minus cys_resnums
        (zero-Cys policy: Cys must be redesigned even at the interface)
      - Feeds --complex_dir/{design}_complex.pdb to MPNN with --pdb_path_chains A
        so MPNN sees the full pMHC context (chain B / C) while designing chain A

    --complex_dir should contain {design}_complex.pdb files with the
    3-chain AF3 layout:
      chain A = binder, chain B = MHC (truncated alpha-chain), chain C = peptide
    MPNN designs chain A with chains B and C as fixed structural context.

    Select which designs to redesign:
      --redesign_failing_charge   flag_charge_redesign == True
      --redesign_failing_cys      flag_cys_redesign == True
      --redesign_failing_any      pass_all == False  [default]
      --redesign_all              all designs in the TSV

  The sequence column in redesign_audit.tsv is used only for logging/comparison;
  ProteinMPNN reads the sequence from the input PDB.

Fixed positions mechanism:
  write_fixed_positions_jsonl() writes a tmp JSONL:
    {"<pdb_stem>": {"A": [res1, res2, ...]}}
  ProteinMPNN reads this and holds those residues fixed (preserves the PDB
  sequence at those positions) while freely designing all other chain A residues.
  This is equivalent to BindCraft's mpnn_fix_interface=True.

Usage — Mode A:
    python redesign_w_charge_residue_filter.py \
        --pdb_dir   /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r3/ \
        --out_dir   /n/groups/marks/users/aaron/pmhc/proteinmpnn/outputs/r3/ \
        --mpnn_path /n/groups/marks/projects/marks_lab_and_oatml/ProteinGym2/model_envs/proteinmpnn/bin/python \
        --mpnn_script /n/groups/marks/users/aaron/enzymes/ProteinMPNN/protein_mpnn_run.py \

Usage — Mode B (r2.5 redesign):
    python redesign_w_charge_residue_filter.py \
        --audit_tsv   /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2/prioritization/redesign_list.tsv \
        --af3_out_dir /n/groups/marks/users/aaron/pmhc_cp/post_filter/outputs/r2/af3_nomsa/ \
        --out_dir     /n/groups/marks/users/aaron/pmhc_cp/proteinmpnn/outputs/r2.5/ \
        --mpnn_path /n/groups/marks/projects/marks_lab_and_oatml/ProteinGym2/model_envs/proteinmpnn/bin/python \
        --mpnn_script /n/groups/marks/users/aaron/enzymes/ProteinMPNN/protein_mpnn_run.py \
        --n_seqs 8 --max_charge -3 --max_attempts 10 \
        --redesign_failing_any
"""

from __future__ import annotations

import os
import re
import ast
import glob
import json
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path
import sys
import pandas as pd

sys.path.append('/n/groups/marks/users/aaron/pmhc_cp/post_filter/scripts/')


# ── Net charge ────────────────────────────────────────────────────────────────

def net_charge(seq: str) -> float:
    """Physiological net charge: R + K + 0.1*H - D - E"""
    return (seq.count('R') + seq.count('K') + 0.1 * seq.count('H')
            - seq.count('D') - seq.count('E'))


# ── FASTA helpers ─────────────────────────────────────────────────────────────

def parse_fasta(fasta_path: str) -> list[dict]:
    """Parse ProteinMPNN FASTA → [{header, seq}]."""
    records = []
    header, lines = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if header is not None:
                    records.append({'header': header, 'seq': ''.join(lines)})
                header, lines = line[1:], []
            else:
                lines.append(line)
    if header is not None:
        records.append({'header': header, 'seq': ''.join(lines)})
    return records


# ── AF3 CIF discovery and conversion (imported from fastrelax_designs_af3.py) ─
# find_model_cif and cif_to_pdb are defined in fastrelax_designs_af3.py and
# imported at runtime. fastrelax_designs_af3.py must be on sys.path (pass its
# directory via PYTHONPATH or add it to sys.path before running this script).
try:
    from fastrelax_designs_af3 import find_model_cif, cif_to_pdb
except ImportError as _e:
    raise ImportError(
        "fastrelax_designs_af3.py must be on sys.path. "
        "Add its directory to PYTHONPATH or run with:\n"
        "  PYTHONPATH=/path/to/dir python redesign_w_charge_residue_filter.py ..."
    ) from _e


# ── Fixed positions JSONL ─────────────────────────────────────────────────────

def write_fixed_positions_jsonl(pdb_stem: str,
                                fixed_resnums: list[int],
                                out_path: str):
    """
    Write a ProteinMPNN fixed_positions_jsonl for one PDB.
    Format: {"<pdb_stem>": {"A": [resnum1, resnum2, ...]}}

    Residue numbers must match the PDB exactly (1-indexed, renumbered from 1
    as written by split_pmhc_fold_pdbs.py or af3_design_stats.py).
    An empty list tells MPNN to design all of chain A freely.
    """
    data = {pdb_stem: {'A': sorted(fixed_resnums)}}
    with open(out_path, 'w') as f:
        f.write(json.dumps(data) + '\n')


def parse_resnums_from_audit(cell) -> list[int]:
    """
    Parse a cell from redesign_audit.tsv that looks like:
      '[]'  or  '[12, 34, 56]'
    Returns a list of ints. Handles NaN/None gracefully.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    try:
        val = ast.literal_eval(str(cell).strip())
        return [int(x) for x in val] if val else []
    except Exception:
        return []


def parse_redesign_reasons(cell) -> list[str]:
    """
    Parse the redesign_reasons column from af3_design_stats.py output.
    Cell format: '['charge>-2.0', 'cys=[52]']' or '[]' or NaN.

    Known reason tokens and their meaning:
      'charge>-2.0'          net charge above threshold   → rejection-sample for charge
      'cys=[N,...]'          Cys present                  → omit_AAs C handles this
      'surface_hydrophobicity' high surface hydrophobicity → free redesign of non-fixed residues
      'ipsae_binder_peptide'  low peptide ipSAE            → logged; MPNN cannot fix directly
      'packstat'             poor packing                 → logged; free redesign may help
      'buns_delta_unsat'     buried unsatisfied H-bonds   → logged; free redesign may help

    Returns a list of reason strings (empty list if none or unparseable).
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    try:
        val = ast.literal_eval(str(cell).strip())
        return [str(r) for r in val] if val else []
    except Exception:
        return []


def summarize_reasons(reasons: list[str]) -> str:
    """Return a compact human-readable summary of redesign reasons."""
    if not reasons:
        return 'none'
    tags = []
    for r in reasons:
        if r.startswith('charge'):
            tags.append('charge')
        elif r.startswith('cys'):
            tags.append('cys')
        elif r == 'surface_hydrophobicity':
            tags.append('hydrophob')
        elif r == 'ipsae_binder_peptide':
            tags.append('ipsae')
        elif r == 'packstat':
            tags.append('packstat')
        elif r == 'buns_delta_unsat':
            tags.append('buns')
        else:
            tags.append(r)
    return '+'.join(tags)


# ── Core MPNN runner ──────────────────────────────────────────────────────────

KEEP_CHAINS = {'A', 'B', 'C'}


def strip_to_abc(pdb_path: str, tmp_dir: str) -> str:
    """
    Write a copy of pdb_path retaining only ATOM/HETATM lines for chains A, B, C.
    AF3 outputs may have a 4th chain (e.g. chain D for a second peptide or
    cofactor); MPNN must not see it. The stem is preserved so MPNN's output
    FASTA filename still matches the original design name.
    """
    out_path = os.path.join(tmp_dir, Path(pdb_path).name)
    with open(pdb_path) as fh, open(out_path, 'w') as out:
        for line in fh:
            record = line[:6].strip()
            if record in ('ATOM', 'HETATM'):
                if line[21] not in KEEP_CHAINS:
                    continue
            out.write(line)
    return out_path


def run_mpnn(python_bin: str,
             mpnn_script: str,
             pdb_path: str,
             out_dir: str,
             n_seqs: int,
             temperature: float,
             omit_AAs: str,
             design_chains: str,
             seed: int,
             tmp_dir: str,
             fixed_positions_jsonl: str | None = None) -> str | None:
    """
    Run standalone protein_mpnn_run.py on a single PDB.
    Returns path to output FASTA, or None on failure.

    Chains beyond A/B/C are stripped before passing to MPNN (AF3 structures
    can have a 4th chain that should be ignored). MPNN designs chain A with
    chains B and C as fixed structural context.

    fixed_positions_jsonl: JSONL from write_fixed_positions_jsonl(); those
      chain A residues are held fixed, all others are freely designed.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Strip chain D+ into a temp copy so MPNN only sees ABC
    filtered_pdb = strip_to_abc(pdb_path, tmp_dir)

    cmd = [
        python_bin, mpnn_script,
        '--pdb_path',           filtered_pdb,
        '--out_folder',         out_dir,
        '--num_seq_per_target', str(n_seqs),
        '--sampling_temp',      str(temperature),
        '--omit_AAs',           omit_AAs,
        '--pdb_path_chains',    design_chains,
        '--seed',               str(seed),
        '--batch_size',         '1',
    ]
    if fixed_positions_jsonl:
        cmd += ['--fixed_positions_jsonl', fixed_positions_jsonl]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [WARN] MPNN failed (seed={seed}): {result.stderr[:300]}")
        return None

    # FASTA is keyed by the PDB stem written into tmp_dir by cif_to_pdb,
    # which uses the design name (not the raw CIF stem with '_model' suffix).
    raw_stem = Path(pdb_path).stem
    stem     = raw_stem[:-6] if raw_stem.endswith('_model') else raw_stem
    fasta    = os.path.join(out_dir, 'seqs', f'{stem}.fa')
    return fasta if os.path.exists(fasta) else None


# ── Rejection sampling loop ───────────────────────────────────────────────────

def collect_sequences(python_bin: str,
                      mpnn_script: str,
                      pdb_path: str,
                      out_dir: str,
                      n_seqs: int,
                      temperature: float,
                      omit_AAs: str,
                      design_chains: str,
                      max_charge: float,
                      max_attempts: int,
                      base_seed: int = 42,
                      fixed_resnums: list[int] | None = None,
                      reference_seq: str | None = None) -> list[dict]:
    """
    Collect up to n_seqs sequences with net_charge <= max_charge.
    Re-runs MPNN with a different seed each attempt if needed.

    fixed_resnums: chain A residue numbers to hold fixed (interface positions
      minus any Cys). Passed via --fixed_positions_jsonl.

    reference_seq: original sequence from redesign_audit.tsv, logged for
      comparison but not used to constrain the design.
    """
    collected = []
    seen_seqs = set()
    # Strip '_model' suffix that AF3 CIF filenames carry ({design}_model.cif)
    raw_stem  = Path(pdb_path).stem
    stem      = raw_stem[:-6] if raw_stem.endswith('_model') else raw_stem
    fixed_resnums = fixed_resnums or []

    for attempt in range(1, max_attempts + 1):
        if len(collected) >= n_seqs:
            break

        seed    = base_seed + attempt * 1000
        tmp_dir = tempfile.mkdtemp(prefix=f'mpnn_{stem}_a{attempt}_')

        try:
            # Convert CIF to PDB if needed (AF3 outputs are mmCIF)
            if pdb_path.endswith('.cif'):
                local_pdb = cif_to_pdb(pdb_path, tmp_dir, stem)
            else:
                local_pdb = pdb_path

            fixed_jsonl = None
            if fixed_resnums:
                fixed_jsonl = os.path.join(tmp_dir, 'fixed_positions.jsonl')
                write_fixed_positions_jsonl(stem, fixed_resnums, fixed_jsonl)

            fasta_path = run_mpnn(
                python_bin, mpnn_script, local_pdb,
                tmp_dir, n_seqs, temperature, omit_AAs,
                design_chains, seed, tmp_dir, fixed_jsonl
            )
            if fasta_path is None:
                continue

            records = parse_fasta(fasta_path)
            # Skip the poly-G/poly-A reference sequence (first record per backbone)
            seqs = [r for r in records
                    if not re.match(r'^[GA]+$', r['seq'])]

            for r in seqs:
                if len(collected) >= n_seqs:
                    break
                seq    = r['seq']
                charge = net_charge(seq)
                if seq in seen_seqs:
                    continue
                if charge <= max_charge:
                    seen_seqs.add(seq)
                    collected.append({
                        'header':        f"{stem}_a{attempt}_{r['header']}",
                        'seq':           seq,
                        'charge':        round(charge, 2),
                        'attempt':       attempt,
                        'n_cys':         seq.count('C'),
                        'n_met':         seq.count('M'),
                        'n_fixed':       len(fixed_resnums),
                        'reference_seq': reference_seq or '',
                    })

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if len(collected) < n_seqs:
        print(f"    [WARN] {stem}: collected {len(collected)}/{n_seqs} sequences "
              f"with charge <= {max_charge} after {max_attempts} attempts")

    return collected


# ── Mode A: new backbone PDB directory ───────────────────────────────────────

def run_mode_a(args) -> list[tuple[str, list[int], str | None]]:
    """Returns list of (pdb_path, fixed_resnums=[], reference_seq=None)."""
    pdb_files = sorted(glob.glob(os.path.join(args.pdb_dir, args.pattern)))
    if not pdb_files:
        raise ValueError(f"No PDB files in {args.pdb_dir} matching {args.pattern}")
    print(f"Mode A: designing {len(pdb_files)} backbone PDBs (no interface masking)")
    return [(pdb, [], None, 'none', []) for pdb in pdb_files]


# ── Mode B: redesign from redesign_audit.tsv ─────────────────────────────────

def run_mode_b(args) -> list[tuple[str, list[int], str | None, str, list[int]]]:
    """
    Returns list of (complex_pdb_path, fixed_resnums, reference_seq, reasons_str, surface_hydrophobic_resnums).

    redesign_audit.tsv columns used (schema from af3_design_stats.py):
      design              — design name; used to discover AF3 CIF via find_model_cif()
      needs_redesign      — bool; primary selection flag
      redesign_reasons    — list of reason strings; parsed for logging
      fixed_resnums       — pre-computed fixed positions (interface residues,
                            already excluding Cys — produced by af3_design_stats.py)
      sequence            — original sequence (logged for comparison)
      net_charge          — logged
      n_cys               — logged
      flag_charge_redesign — secondary selection flag
      flag_cys_redesign   — secondary selection flag
      pass_all            — secondary selection flag
      surface_hydrophobic_resnums — logged only; interface_fixed_resnums unchanged
    """
    audit = pd.read_csv(args.audit_tsv, sep='\t')

    # Validate required columns
    required = {
        'design', 'needs_redesign', 'redesign_reasons',
        'fixed_resnums', 'sequence', 'net_charge', 'n_cys',
        'flag_charge_redesign', 'flag_cys_redesign', 'pass_all',
    }
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(
            f"redesign_audit.tsv is missing columns: {missing}\n"
            f"Expected schema from af3_design_stats.py. Found: {list(audit.columns)}"
        )

    # Select designs
    if args.redesign_all:
        selected = audit
    elif args.redesign_failing_charge:
        selected = audit[audit['flag_charge_redesign'] == True]
    elif args.redesign_failing_cys:
        selected = audit[audit['flag_cys_redesign'] == True]
    else:  # redesign_failing_any (default) — use needs_redesign as primary flag
        selected = audit[audit['needs_redesign'] == True]

    n_needs   = int(audit['needs_redesign'].sum())
    n_charge  = int(audit['flag_charge_redesign'].sum())
    n_cys     = int(audit['flag_cys_redesign'].sum())
    n_pass_all = int((audit['pass_all'] == False).sum())

    print(f"Mode B: {len(audit)} designs in audit TSV")
    print(f"  needs_redesign: {n_needs}  |  charge fail: {n_charge}  |  "
          f"cys fail: {n_cys}  |  pass_all fail: {n_pass_all}")
    print(f"  → redesigning {len(selected)} designs")

    tasks = []
    for _, row in selected.iterrows():
        design = row['design']

        # Discover AF3 model CIF (handles timestamped subdirectory suffixes)
        cif_path = find_model_cif(args.af3_out_dir, design)
        if not cif_path:
            print(f"  [WARN] model CIF not found for '{design}' "
                  f"in {args.af3_out_dir}, skipping")
            continue
        print(f"    CIF: {cif_path}")

        # Use fixed_resnums directly — af3_design_stats.py already excludes Cys
        fixed_resnums = parse_resnums_from_audit(row.get('fixed_resnums', '[]'))

        # Parse redesign reasons for logging
        reasons     = parse_redesign_reasons(row.get('redesign_reasons', '[]'))
        reasons_str = summarize_reasons(reasons)

        ref_seq = str(row.get('sequence', '')) or None

        # Surface hydrophobic residues — logged only, interface_fixed_resnums unchanged
        surface_hyd_resnums = parse_resnums_from_audit(
            row.get('surface_hydrophobic_resnums', '[]')
        )

        tasks.append((cif_path, fixed_resnums, ref_seq, reasons_str, surface_hyd_resnums))

        hyd_note = f"  surface_hyd={len(surface_hyd_resnums)} res" if surface_hyd_resnums else ""
        print(f"  {design}: charge={row['net_charge']:+.1f}  "
              f"n_cys={row['n_cys']}  "
              f"fixed={len(fixed_resnums)} res  "
              f"reasons=[{reasons_str}]{hyd_note}")

    return tasks


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection (mutually exclusive)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pdb_dir',
                      help='Mode A: directory of backbone PDB files')
    mode.add_argument('--audit_tsv',
                      help='Mode B: redesign_audit.tsv from af3_design_stats.py')

    # Mode B options
    p.add_argument('--af3_out_dir',
                   help='Mode B: AF3 output root directory containing '                        '{design}_YYYYMMDD_HHMMSS/{design}_model.cif subdirs')

    redesign = p.add_mutually_exclusive_group()
    redesign.add_argument('--redesign_failing_any',    action='store_true', default=True,
                          help='Redesign designs where pass_all==False [default]')
    redesign.add_argument('--redesign_failing_charge', action='store_true',
                          help='Redesign only designs where flag_charge_redesign==True')
    redesign.add_argument('--redesign_failing_cys',    action='store_true',
                          help='Redesign only designs where flag_cys_redesign==True')
    redesign.add_argument('--redesign_all',            action='store_true',
                          help='Redesign all designs in the TSV')

    # MPNN settings
    p.add_argument('--out_dir',       required=True)
    p.add_argument('--mpnn_path',     required=True,
                   help='Python binary for ProteinMPNN environment')
    p.add_argument('--mpnn_script',   required=True,
                   help='Path to protein_mpnn_run.py')
    p.add_argument('--n_seqs',        type=int,   default=8,
                   help='Target number of sequences per backbone (default: 8)')
    p.add_argument('--temperature',   type=float, default=0.1,
                   help='MPNN sampling temperature (default: 0.1)')
    p.add_argument('--omit_AAs',      default='C',
                   help='AAs to omit from design (default: C — no Cys)')
    p.add_argument('--design_chains', default='A',
                   help='Chain(s) to design; all others are fixed context '
                        '(default: A)')
    p.add_argument('--max_charge',    type=float, default=-3.0,
                   help='Maximum allowed net charge (default: -3.0)')
    p.add_argument('--max_attempts',  type=int,   default=10,
                   help='Max rejection-sampling attempts per backbone (default: 10)')
    p.add_argument('--base_seed',     type=int,   default=42)
    p.add_argument('--pattern',       default='*.pdb',
                   help='Mode A: glob pattern for PDB files (default: *.pdb)')
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Build task list: [(pdb_path, fixed_resnums, reference_seq), ...]
    if args.pdb_dir:
        tasks = run_mode_a(args)
    else:
        if not args.af3_out_dir:
            raise ValueError('--af3_out_dir is required with --audit_tsv (Mode B)')
        tasks = run_mode_b(args)

    print(f"\nProteinMPNN settings:")
    print(f"  n_seqs={args.n_seqs}  T={args.temperature}  omit_AAs={args.omit_AAs}  "
          f"design_chains={args.design_chains}")
    print(f"  max_charge<={args.max_charge}  max_attempts={args.max_attempts}\n")

    all_results   = []
    combined_path = os.path.join(args.out_dir, 'all_sequences.fa')

    with open(combined_path, 'w') as combined_fa:
        for i, (pdb_path, fixed_resnums, ref_seq, reasons_str, surface_hyd_resnums) in enumerate(tasks):
            raw_stem = Path(pdb_path).stem
            stem     = raw_stem[:-6] if raw_stem.endswith('_model') else raw_stem
            hyd_note = f"  surface_hyd={len(surface_hyd_resnums)}" if surface_hyd_resnums else ""
            print(f"[{i+1}/{len(tasks)}] {stem}  "
                  f"(fixed={len(fixed_resnums)} res  reasons=[{reasons_str}]{hyd_note})")

            collected = collect_sequences(
                python_bin    = args.mpnn_path,
                mpnn_script   = args.mpnn_script,
                pdb_path      = pdb_path,
                out_dir       = args.out_dir,
                n_seqs        = args.n_seqs,
                temperature   = args.temperature,
                omit_AAs      = args.omit_AAs,
                design_chains = args.design_chains,
                max_charge    = args.max_charge,
                max_attempts  = args.max_attempts,
                base_seed     = args.base_seed,
                fixed_resnums = fixed_resnums,
                reference_seq = ref_seq,
            )

            for r in collected:
                print(f"  attempt {r['attempt']}: "
                      f"charge={r['charge']:+.1f}  "
                      f"Cys={r['n_cys']}  Met={r['n_met']}  "
                      f"{r['seq'][:40]}...")
                combined_fa.write(f">{r['header']}\n{r['seq']}\n")
                all_results.append({
                    'backbone':        stem,
                    'pdb_path':        pdb_path,
                    'header':          r['header'],
                    'seq':             r['seq'],
                    'length':          len(r['seq']),
                    'net_charge':      r['charge'],
                    'n_cys':           r['n_cys'],
                    'n_met':           r['n_met'],
                    'attempt':         r['attempt'],
                    'n_fixed':         r['n_fixed'],
                    'redesign_reasons': reasons_str,
                    'surface_hydrophobic_resnums': str(surface_hyd_resnums),
                    'reference_seq':   r['reference_seq'],
                })

    log      = pd.DataFrame(all_results)
    log_path = os.path.join(args.out_dir, 'mpnn_redesign_log.tsv')
    log.to_csv(log_path, sep='\t', index=False)

    print(f"\n{'='*60}")
    print(f"  Backbones processed:        {len(tasks)}")
    print(f"  Total sequences collected:  {len(all_results)}")
    if len(all_results):
        n_full = (log.groupby('backbone').size() == args.n_seqs).sum()
        print(f"  Backbones with full {args.n_seqs} seqs: {n_full}/{len(tasks)}")
        print(f"  Net charge — "
              f"mean={log['net_charge'].mean():.1f}  "
              f"min={log['net_charge'].min():.1f}  "
              f"max={log['net_charge'].max():.1f}")
        print(f"  Sequences needing >1 attempt: "
              f"{(log['attempt'] > 1).sum()} "
              f"({100*(log['attempt'] > 1).mean():.0f}%)")
        if log['n_cys'].sum() > 0:
            print(f"  WARNING: {(log['n_cys'] > 0).sum()} sequences still contain Cys "
                  f"— increase --max_attempts or verify --omit_AAs includes C")
        if log['n_met'].sum() > 0 and 'M' in args.omit_AAs:
            print(f"  WARNING: {(log['n_met'] > 0).sum()} sequences still contain Met "
                  f"— increase --max_attempts")
    print(f"  Combined FASTA: {combined_path}")
    print(f"  Log TSV:        {log_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
