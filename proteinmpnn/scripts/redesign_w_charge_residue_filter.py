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

  Mode B — redesign from af3_design_stats.py + collect_redesign_candidates.py
            + surface_hydrophobicity_resnums.py -> redesign_list.tsv:

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
      - Reads fixed_resnums (pre-computed by upstream scripts, already excludes Cys)
      - Discovers AF3 model CIF from timestamped subdirs in --af3_out_dir
      - Converts CIF -> PDB (Biopython), strips chains beyond ABC, runs MPNN
      - MPNN designs chain A with chains B (MHC) and C (peptide) as fixed context

    Select which designs to redesign:
      --redesign_failing_charge   flag_charge_redesign == True
      --redesign_failing_cys      flag_cys_redesign == True
      --redesign_failing_any      needs_redesign == True  [default]
      --redesign_all              all designs in the TSV

  The sequence column in redesign_list.tsv is used only for logging/comparison;
  ProteinMPNN reads the sequence from the input PDB.

ProteinMPNN pipeline (helper-script variant):
  For each backbone and each rejection-sampling attempt:
    1. parse_multiple_chains.py  -> parsed_pdbs.jsonl
    2. assign_fixed_chains.py    -> assigned_chains.jsonl
    3. make_fixed_positions_dict.py -> fixed_positions.jsonl  (only if fixed_resnums)
    4. protein_mpnn_run.py       -> seqs/{stem}.fa
  helper_scripts/ is located at os.path.dirname(--mpnn_script)/helper_scripts/

Usage -- Mode A:
    python redesign_w_charge_residue_filter.py \
        --pdb_dir   /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r3/ \
        --out_dir   /n/groups/marks/users/aaron/pmhc/proteinmpnn/outputs/r3/ \
        --mpnn_path /n/groups/marks/projects/marks_lab_and_oatml/ProteinGym2/model_envs/proteinmpnn/bin/python \
        --mpnn_script /n/groups/marks/users/aaron/enzymes/ProteinMPNN/protein_mpnn_run.py

Usage -- Mode B (r2.5 redesign):
    export PYTHONPATH="/n/groups/marks/users/aaron/pmhc_cp/post_filter/scripts:$PYTHONPATH"
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
from Bio.PDB import MMCIFParser, PDBIO


# -- Net charge ----------------------------------------------------------------

def net_charge(seq: str) -> float:
    """Physiological net charge: R + K + 0.1*H - D - E"""
    return (seq.count('R') + seq.count('K') + 0.1 * seq.count('H')
            - seq.count('D') - seq.count('E'))


# -- Sequence quality filters -------------------------------------------------

def sequence_passes_filters(seq: str, max_charge: float) -> tuple[bool, str]:
    """
    Return (passes: bool, reason: str) for a candidate sequence.

    Hard filters (any failure → discard):
      1. No Cysteine residues (independent of omit_AAs; catches edge cases
         where a fixed interface position carries a Cys from the input PDB)
      2. No homopolymer run of 4+ consecutive identical residues of any AA
         (author designs show max run length of 3; long Ala/Gly runs indicate
         low-complexity sequence recovery at T=0.1)
      3. net_charge <= max_charge

    Returns ('', '') if all filters pass, or (False, reason_string) on failure.
    """
    # Filter 1: no Cys
    if 'C' in seq:
        return False, f'cys at {[i+1 for i, aa in enumerate(seq) if aa == "C"]}'

    # Filter 2: no homopolymer >= 4
    m = re.search(r'(.)\1{3,}', seq)

    if m:
        return False, f'homopolymer {m.group(1)}x{len(m.group())} at pos {m.start()+1}'

    # Filter 3: charge
    charge = net_charge(seq)
    if charge > max_charge:
        return False, f'charge {charge:+.1f} > {max_charge}'

    return True, ''


# -- FASTA helpers -------------------------------------------------------------

def parse_fasta(fasta_path: str) -> list[dict]:
    """Parse ProteinMPNN FASTA -> [{header, seq}]."""
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


# -- AF3 CIF discovery and conversion -----------------------------------------
# find_model_cif is imported from fastrelax_designs_af3.py (identical function).
# cif_to_pdb is defined locally -- fastrelax_designs_af3.py uses load_cif()
# which returns a PyRosetta Pose; we need a plain PDB file for protein_mpnn_run.py.

try:
    from fastrelax_designs_af3 import find_model_cif
except ImportError as _e:
    raise ImportError(
        "fastrelax_designs_af3.py must be on sys.path. "
        "Add its directory to PYTHONPATH, e.g.:\n"
        "  export PYTHONPATH=/n/groups/marks/users/aaron/pmhc_cp/post_filter/scripts:$PYTHONPATH"
    ) from _e


def cif_to_pdb(cif_path: str, out_dir: str, stem: str) -> str:
    """
    Convert an mmCIF file to PDB format using Biopython and write to out_dir.
    Returns the path to the written PDB file.

    fastrelax_designs_af3.py provides load_cif() -> PyRosetta Pose, which is
    not suitable here since protein_mpnn_run.py requires a plain PDB file.
    Biopython PDBIO omits CONECT/SEQRES records that confuse protein_mpnn_run.py.
    """
    parser   = MMCIFParser(QUIET=True)
    struct   = parser.get_structure(stem, cif_path)
    out_path = os.path.join(out_dir, f'{stem}.pdb')
    io_obj   = PDBIO()
    io_obj.set_structure(struct)
    io_obj.save(out_path)
    return out_path


# -- Audit TSV parsers ---------------------------------------------------------

def parse_resnums_from_audit(cell) -> list[int]:
    """
    Parse a cell from redesign_list.tsv that looks like:
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
    Cell format: "['charge>-2.0', 'cys=[52]']" or '[]' or NaN.

    Known reason tokens and their meaning:
      'charge>-2.0'            net charge above threshold   -> rejection-sample for charge
      'cys=[N,...]'            Cys present                  -> omit_AAs C handles this
      'surface_hydrophobicity' high surface hydrophobicity  -> free redesign of non-fixed residues
      'ipsae_binder_peptide'   low peptide ipSAE            -> logged; MPNN cannot fix directly
      'packstat'               poor packing                 -> logged; free redesign may help
      'buns_delta_unsat'       buried unsatisfied H-bonds   -> logged; free redesign may help

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


# -- Chain filtering -----------------------------------------------------------

KEEP_CHAINS = {'A', 'B', 'C'}


def strip_to_abc(pdb_path: str, tmp_dir: str) -> str:
    """
    Write a copy of pdb_path retaining only ATOM/HETATM lines for chains A, B, C.
    AF3 outputs may have a 4th chain (e.g. chain D for a second peptide or
    cofactor); MPNN must not see it. The filename is preserved so MPNN's output
    FASTA key still matches the design stem.
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


# -- Stem helper ---------------------------------------------------------------

def design_stem(path: str) -> str:
    """
    Return the design name from a file path.
    AF3 CIF files are named {design}_model.cif -> strip '_model' suffix.
    Plain PDB files are named {design}.pdb -> use stem directly.
    """
    raw = Path(path).stem
    return raw[:-6] if raw.endswith('_model') else raw


# -- ProteinMPNN helper-script pipeline ----------------------------------------

def run_subprocess(cmd: list[str], label: str) -> bool:
    """Run a subprocess; print full stderr on failure. Returns True on success."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [WARN] {label} failed")
        print(f"    CMD: {' '.join(cmd)}")
        print(f"    STDERR:\n{result.stderr}")
        return False
    return True


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
             fixed_resnums: list[int] | None = None) -> str | None:
    """
    Run ProteinMPNN on a single PDB using the helper-script pipeline:
      1. parse_multiple_chains.py  -> parsed_pdbs.jsonl
      2. assign_fixed_chains.py    -> assigned_chains.jsonl
      3. make_fixed_positions_dict.py -> fixed_positions.jsonl  (if fixed_resnums)
      4. protein_mpnn_run.py       -> seqs/{stem}.fa

    helper_scripts/ is located at os.path.dirname(mpnn_script)/helper_scripts/.

    Chains beyond A/B/C are stripped before parsing. design_chains specifies
    which chain(s) to design (default 'A'); all others are fixed context.
    fixed_resnums: chain A 1-indexed residue numbers to hold fixed.
    """
    os.makedirs(out_dir, exist_ok=True)
    helpers = os.path.join(os.path.dirname(mpnn_script), 'helper_scripts')
    stem    = design_stem(pdb_path)

    # Strip chain D+ so MPNN only sees ABC.
    # Write into a dedicated subdirectory so parse_multiple_chains.py
    # sees exactly one PDB file and produces one JSONL record.
    pdb_dir      = os.path.join(tmp_dir, 'pdb')
    os.makedirs(pdb_dir, exist_ok=True)
    filtered_pdb = strip_to_abc(pdb_path, pdb_dir)

    # Intermediate JSONL paths (all in tmp_dir)
    parsed_jsonl   = os.path.join(tmp_dir, 'parsed_pdbs.jsonl')
    assigned_jsonl = os.path.join(tmp_dir, 'assigned_chains.jsonl')
    fixed_jsonl    = os.path.join(tmp_dir, 'fixed_positions.jsonl') if fixed_resnums else None

    # Step 1: parse_multiple_chains.py
    if not run_subprocess([
        python_bin,
        os.path.join(helpers, 'parse_multiple_chains.py'),
        f'--input_path={pdb_dir}',
        f'--output_path={parsed_jsonl}',
    ], 'parse_multiple_chains'):
        return None

    # Step 2: assign_fixed_chains.py -- which chain(s) to design
    if not run_subprocess([
        python_bin,
        os.path.join(helpers, 'assign_fixed_chains.py'),
        f'--input_path={parsed_jsonl}',
        f'--output_path={assigned_jsonl}',
        '--chain_list', design_chains,
    ], 'assign_fixed_chains'):
        return None

    # Step 3: make_fixed_positions_dict.py -- fix interface residues on chain A
    if fixed_resnums:
        resnums_str = ' '.join(str(r) for r in sorted(fixed_resnums))
        if not run_subprocess([
            python_bin,
            os.path.join(helpers, 'make_fixed_positions_dict.py'),
            f'--input_path={parsed_jsonl}',
            f'--output_path={fixed_jsonl}',
            '--chain_list',    'A',
            '--position_list', resnums_str,
        ], 'make_fixed_positions_dict'):
            return None

    # Step 4: protein_mpnn_run.py
    cmd = [
        python_bin, mpnn_script,
        '--jsonl_path',         parsed_jsonl,
        '--chain_id_jsonl',     assigned_jsonl,
        '--out_folder',         out_dir,
        '--num_seq_per_target', str(n_seqs),
        '--sampling_temp',      str(temperature),
        '--omit_AAs',           omit_AAs,
        '--seed',               str(seed),
        '--batch_size',         '1',
    ]
    if fixed_jsonl:
        cmd += ['--fixed_positions_jsonl', fixed_jsonl]
    if not run_subprocess(cmd, f'protein_mpnn_run (seed={seed})'):
        return None

    fasta = os.path.join(out_dir, 'seqs', f'{stem}.fa')
    return fasta if os.path.exists(fasta) else None


# -- Rejection sampling loop ---------------------------------------------------

# Temperature schedule for rejection sampling:
#   Start at T=0.1 (max_attempts attempts).
#   If still short, raise to T=0.15 (max_attempts more attempts).
#   If still short, raise to T=0.20 (max_attempts more attempts).
#   Never exceeds T=0.20.
TEMP_SCHEDULE = [0.10, 0.15, 0.20]


def _run_attempts(python_bin: str,
                  mpnn_script: str,
                  pdb_path: str,
                  local_pdb: str,
                  out_dir: str,
                  n_seqs: int,
                  temperature: float,
                  omit_AAs: str,
                  design_chains: str,
                  max_attempts: int,
                  base_seed: int,
                  attempt_offset: int,
                  max_charge: float,
                  fixed_resnums: list[int],
                  seen_seqs: set,
                  collected: list,
                  stem: str) -> None:
    """
    Run up to max_attempts MPNN calls at a fixed temperature, appending
    accepted sequences to collected and seen_seqs in place.
    attempt_offset ensures seed values don't repeat across temperature tiers.
    """
    for i in range(max_attempts):
        if len(collected) >= n_seqs:
            break
        attempt  = attempt_offset + i + 1
        seed     = base_seed + attempt * 1000
        tmp_dir  = tempfile.mkdtemp(prefix=f'mpnn_{stem}_a{attempt}_')
        try:
            fasta_path = run_mpnn(
                python_bin, mpnn_script, local_pdb,
                tmp_dir, n_seqs, temperature, omit_AAs,
                design_chains, seed, tmp_dir, fixed_resnums,
            )
            if fasta_path is None:
                continue
            records = parse_fasta(fasta_path)
            seqs    = [r for r in records
                       if not re.match(r'^[GA]+$', r['seq'])]
            for r in seqs:
                if len(collected) >= n_seqs:
                    break
                seq = r['seq']
                if seq in seen_seqs:
                    continue
                passes, reason = sequence_passes_filters(seq, max_charge)
                print(seq, passes, reason)
                if not passes:
                    continue
                seen_seqs.add(seq)
                collected.append({
                    'header':        f"{stem}_a{attempt}_T{temperature}_{r['header']}",
                    'seq':           seq,
                    'charge':        round(net_charge(seq), 2),
                    'attempt':       attempt,
                    'temperature':   temperature,
                    'n_cys':         seq.count('C'),
                    'n_met':         seq.count('M'),
                    'n_fixed':       len(fixed_resnums),
                    'reference_seq': '',
                })
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def collect_sequences(python_bin: str,
                      mpnn_script: str,
                      pdb_path: str,
                      out_dir: str,
                      n_seqs: int,
                      omit_AAs: str,
                      design_chains: str,
                      max_charge: float,
                      max_attempts: int,
                      base_seed: int = 42,
                      fixed_resnums: list[int] | None = None,
                      reference_seq: str | None = None) -> list[dict]:
    """
    Collect up to n_seqs sequences with net_charge <= max_charge using a
    three-tier temperature schedule: T=0.10 -> T=0.15 -> T=0.20.

    Each tier runs up to max_attempts MPNN calls. The next tier is only
    entered if the target count has not been reached. Never exceeds T=0.20.

    fixed_resnums: chain A residue numbers to hold fixed (interface positions,
      already excluding Cys -- passed directly from redesign_list.tsv).
    reference_seq: original sequence from redesign_list.tsv, logged only.
    """
    collected     = []
    seen_seqs     = set()
    stem          = design_stem(pdb_path)
    fixed_resnums = fixed_resnums or []

    # Convert CIF -> PDB once; reused across all temperature tiers
    if pdb_path.endswith('.cif'):
        _conv_tmp = tempfile.mkdtemp(prefix=f'mpnn_{stem}_conv_')
        local_pdb = cif_to_pdb(pdb_path, _conv_tmp, stem)
    else:
        _conv_tmp = None
        local_pdb = pdb_path

    try:
        attempt_offset = 0
        for temperature in TEMP_SCHEDULE:
            if len(collected) >= n_seqs:
                break
            n_before = len(collected)
            _run_attempts(
                python_bin, mpnn_script, pdb_path, local_pdb,
                out_dir, n_seqs, temperature, omit_AAs,
                design_chains, max_attempts, base_seed,
                attempt_offset, max_charge, fixed_resnums,
                seen_seqs, collected, stem,
            )
            n_after = len(collected)
            if n_after > n_before:
                print(f"    T={temperature}: +{n_after - n_before} seqs "
                      f"(total {n_after}/{n_seqs})")
            else:
                print(f"    T={temperature}: 0 seqs passed charge filter "
                      f"after {max_attempts} attempts")
            attempt_offset += max_attempts
    finally:
        if _conv_tmp:
            shutil.rmtree(_conv_tmp, ignore_errors=True)

    # Stamp reference_seq on collected records (set after loop to avoid closure issues)
    for r in collected:
        r['reference_seq'] = reference_seq or ''

    if len(collected) < n_seqs:
        print(f"    [WARN] {stem}: collected {len(collected)}/{n_seqs} sequences "
              f"with charge <= {max_charge} across all temperature tiers")

    return collected


# -- Mode A: new backbone PDB directory ----------------------------------------

def run_mode_a(args) -> list[tuple[str, list[int], str | None, str, list[int]]]:
    """Returns list of (pdb_path, fixed_resnums=[], reference_seq=None, reasons, hyd)."""
    pdb_files = sorted(glob.glob(os.path.join(args.pdb_dir, args.pattern)))
    if not pdb_files:
        raise ValueError(f"No PDB files in {args.pdb_dir} matching {args.pattern}")
    print(f"Mode A: designing {len(pdb_files)} backbone PDBs (no interface masking)")
    return [(pdb, [], None, 'none', []) for pdb in pdb_files]


# -- Mode B: redesign from redesign_list.tsv -----------------------------------

def run_mode_b(args) -> list[tuple[str, list[int], str | None, str, list[int]]]:
    """
    Returns list of (cif_path, fixed_resnums, reference_seq, reasons_str, surface_hyd_resnums).

    redesign_list.tsv columns used:
      design                      -- design name; used to discover AF3 CIF
      needs_redesign              -- bool; primary selection flag
      redesign_reasons            -- list of reason strings; parsed for logging
      fixed_resnums               -- pre-computed fixed positions (already excludes Cys)
      sequence                    -- original sequence (logged for comparison)
      net_charge                  -- logged
      n_cys                       -- logged
      flag_charge_redesign        -- secondary selection flag
      flag_cys_redesign           -- secondary selection flag
      pass_all                    -- secondary selection flag
      surface_hydrophobic_resnums -- logged only; fixed_resnums unchanged
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
            f"redesign_list.tsv is missing columns: {missing}\n"
            f"Found: {list(audit.columns)}"
        )

    # Select designs
    if args.redesign_all:
        selected = audit
    elif args.redesign_failing_charge:
        selected = audit[audit['flag_charge_redesign'] == True]
    elif args.redesign_failing_cys:
        selected = audit[audit['flag_cys_redesign'] == True]
    else:  # redesign_failing_any (default)
        selected = audit[audit['needs_redesign'] == True]

    n_needs    = int(audit['needs_redesign'].sum())
    n_charge   = int(audit['flag_charge_redesign'].sum())
    n_cys      = int(audit['flag_cys_redesign'].sum())
    n_pass_all = int((audit['pass_all'] == False).sum())

    print(f"Mode B: {len(audit)} designs in audit TSV")
    print(f"  needs_redesign: {n_needs}  |  charge fail: {n_charge}  |  "
          f"cys fail: {n_cys}  |  pass_all fail: {n_pass_all}")
    print(f"  -> redesigning {len(selected)} designs")

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

        # fixed_resnums already excludes Cys (produced by upstream scripts)
        fixed_resnums = parse_resnums_from_audit(row.get('fixed_resnums', '[]'))

        # Parse redesign reasons for logging
        reasons     = parse_redesign_reasons(row.get('redesign_reasons', '[]'))
        reasons_str = summarize_reasons(reasons)

        ref_seq = str(row.get('sequence', '')) or None

        # Surface hydrophobic residues -- logged only, fixed_resnums unchanged
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


# -- Argument parsing ----------------------------------------------------------

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
                      help='Mode B: redesign_list.tsv from upstream audit scripts')

    # Mode B options
    p.add_argument('--af3_out_dir',
                   help='Mode B: AF3 output root directory containing '
                        '{design}_YYYYMMDD_HHMMSS/{design}_model.cif subdirs')

    redesign = p.add_mutually_exclusive_group()
    redesign.add_argument('--redesign_failing_any',    action='store_true', default=True,
                          help='Redesign designs where needs_redesign==True [default]')
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
                   help='Path to protein_mpnn_run.py; helper_scripts/ must be '
                        'in the same directory')
    p.add_argument('--n_seqs',        type=int,   default=8,
                   help='Target number of sequences per backbone (default: 8)')
    # Temperature is now a fixed 3-tier schedule: 0.10 -> 0.15 -> 0.20
    # (see TEMP_SCHEDULE). --temperature arg removed.
    p.add_argument('--omit_AAs',      default='C',
                   help='AAs to omit from design (default: C -- no Cys)')
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


# -- Main ----------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.pdb_dir:
        tasks = run_mode_a(args)
    else:
        if not args.af3_out_dir:
            raise ValueError('--af3_out_dir is required with --audit_tsv (Mode B)')
        tasks = run_mode_b(args)

    print(f"\nProteinMPNN settings:")
    print(f"  n_seqs={args.n_seqs}  omit_AAs={args.omit_AAs}  design_chains={args.design_chains}")
    print(f"  max_charge<={args.max_charge}  max_attempts={args.max_attempts} per tier")
    print(f"  temperature schedule: {TEMP_SCHEDULE} (escalates if n_seqs not reached)\n")

    all_results   = []
    combined_path = os.path.join(args.out_dir, 'all_sequences.fa')

    with open(combined_path, 'w') as combined_fa:
        for i, (pdb_path, fixed_resnums, ref_seq, reasons_str, surface_hyd_resnums) in enumerate(tasks):
            stem     = design_stem(pdb_path)
            hyd_note = f"  surface_hyd={len(surface_hyd_resnums)}" if surface_hyd_resnums else ""
            print(f"[{i+1}/{len(tasks)}] {stem}  "
                  f"(fixed={len(fixed_resnums)} res  reasons=[{reasons_str}]{hyd_note})")

            collected = collect_sequences(
                python_bin    = args.mpnn_path,
                mpnn_script   = args.mpnn_script,
                pdb_path      = pdb_path,
                out_dir       = args.out_dir,
                n_seqs        = args.n_seqs,
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
                    'backbone':                    stem,
                    'pdb_path':                    pdb_path,
                    'header':                      r['header'],
                    'seq':                         r['seq'],
                    'length':                      len(r['seq']),
                    'net_charge':                  r['charge'],
                    'n_cys':                       r['n_cys'],
                    'n_met':                       r['n_met'],
                    'attempt':                     r['attempt'],
                    'temperature':                 r['temperature'],
                    'n_fixed':                     r['n_fixed'],
                    'redesign_reasons':            reasons_str,
                    'surface_hydrophobic_resnums': str(surface_hyd_resnums),
                    'reference_seq':               r['reference_seq'],
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
        print(f"  Net charge -- "
              f"mean={log['net_charge'].mean():.1f}  "
              f"min={log['net_charge'].min():.1f}  "
              f"max={log['net_charge'].max():.1f}")
        print(f"  Sequences needing >1 attempt: "
              f"{(log['attempt'] > 1).sum()} "
              f"({100*(log['attempt'] > 1).mean():.0f}%)")
        if log['n_cys'].sum() > 0:
            print(f"  WARNING: {(log['n_cys'] > 0).sum()} sequences still contain Cys "
                  f"-- increase --max_attempts or verify --omit_AAs includes C")
        if log['n_met'].sum() > 0 and 'M' in args.omit_AAs:
            print(f"  WARNING: {(log['n_met'] > 0).sum()} sequences still contain Met "
                  f"-- increase --max_attempts")
    print(f"  Combined FASTA: {combined_path}")
    print(f"  Log TSV:        {log_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()