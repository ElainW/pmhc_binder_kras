#!/bin/bash
#SBATCH --job-name=kras_partdiff_t2
#SBATCH --partition=gpu_quad,gpu
#SBATCH --qos=gpuquad_qos
#SBATCH --time=8:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=../slurm/kras_partdiff_t2_%j.out
#SBATCH --error=../slurm/kras_partdiff_t2_%j.err
#SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --mail-user=wangyilan0304@gmail.com
# =============================================================================
# Tier 1 Partial Diffusion — KRAS pMHC Round 2
# Backbones: orient_77, 259, 244, 88, 321, 252, 165, 43  (best pAE 6–10)
# partial_T=20  →  moderate perturbation, 100 designs per backbone
# Total trajectories: 400
#
# Chain layout in RFdiffusion output PDBs (sequential residue numbering):
#   Chain A = binder  (partially diffused, residues 1–N)
#   Chain B = MHC     (fixed, residues N+1–N+178)
#   Chain C = peptide (fixed, residues N+179–N+187)
#
# Both the contig and hotspot residue numbers are read directly from each
# PDB so they are correct regardless of binder length.
# =============================================================================
set -euo pipefail
# ── Paths ─────────────────────────────────────────────────────────────────────
PYTHON=/n/groups/marks/users/aaron/RFdiffusion/env/SE3nv/bin/python
SCRIPT=/n/groups/marks/users/aaron/RFdiffusion/scripts/run_inference.py
R1_PDB_DIR=~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/1000_filtered/orientation_filtered
R2_OUT_ROOT=~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/partial_r2
# ── Tier 1 settings ───────────────────────────────────────────────────────────
BACKBONES=(77 259 244 88 321 252 165 43)
PARTIAL_T=20
NUM_DESIGNS=50
TIER=2
# ── Shared diffusion settings ─────────────────────────────────────────────────
NOISE_CA=0
NOISE_FRAME=0
# ── Helpers ───────────────────────────────────────────────────────────────────
# Returns the raw start-end residue numbers for a chain as they appear in
# the PDB (e.g. "96-273" for chain B in a backbone with a 95-residue binder).
# RFdiffusion uses these raw numbers directly when parsing the contig map.
get_chain_range() {
    local pdb=$1 chain=$2
    local start end
    start=$(grep "^ATOM" "$pdb" | awk -v c="$chain" '$5==c {print $6+0}' | sort -n | head -1)
    end=$(  grep "^ATOM" "$pdb" | awk -v c="$chain" '$5==c {print $6+0}' | sort -n | tail -1)
    echo "${start}-${end}"
}

# Returns the Nth residue number of a given chain (1-indexed within chain).
# Used to map original hotspot positions (C6, C7) to sequential PDB numbers.
get_chain_res() {
    local pdb=$1 chain=$2 n=$3
    grep "^ATOM" "$pdb" \
        | awk -v c="$chain" '$5==c {print $6+0}' \
        | sort -un \
        | sed -n "${n}p"
}
# ── Run ───────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Tier ${TIER} | partial_T=${PARTIAL_T} | n=${NUM_DESIGNS}"
echo "  Backbones: ${BACKBONES[*]}"
echo "  SLURM job ID: ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo "  Started: $(date)"
echo "============================================================"
for bb in "${BACKBONES[@]}"; do
    input_pdb="${R1_PDB_DIR}/kras_orient_${bb}.pdb"
    out_prefix="${R2_OUT_ROOT}/tier${TIER}/orient_${bb}/orient_${bb}_pT${PARTIAL_T}_"

    if [[ ! -f "$input_pdb" ]]; then
        echo "[WARN] PDB not found: ${input_pdb} — skipping orient_${bb}"
        continue
    fi

    # Build contig from raw PDB residue numbers.
    # Chain A (binder) is specified as a length range (no chain letter) so
    # RFdiffusion treats it as hallucinated and applies partial_T noise to it.
    # Chains B and C are specified with chain+residue to remain fully fixed.
    BINDER_LEN=$(grep "^ATOM" "$input_pdb" | awk '$5=="A" {print $6+0}' | sort -un | wc -l)
    RANGE_B=$(get_chain_range "$input_pdb" "B")
    RANGE_C=$(get_chain_range "$input_pdb" "C")
    CONTIGS="[${BINDER_LEN}-${BINDER_LEN}/0 B${RANGE_B}/0 C${RANGE_C}]"

    # Map original hotspot positions C6, C7 to sequential PDB residue numbers
    HS1=$(get_chain_res "$input_pdb" "C" 6)
    HS2=$(get_chain_res "$input_pdb" "C" 7)
    HOTSPOTS="[C${HS1},C${HS2}]"

    echo ""
    echo "[$(date +%H:%M:%S)] Starting orient_${bb}"
    echo "  Binder length: ${BINDER_LEN} | Chain B: ${RANGE_B} | Chain C: ${RANGE_C}"
    echo "  Contig: ${CONTIGS} | Hotspots: ${HOTSPOTS}"
    mkdir -p "$(dirname "$out_prefix")"

    "$PYTHON" "$SCRIPT" \
        inference.output_prefix="${out_prefix}" \
        inference.input_pdb="${input_pdb}" \
        "contigmap.contigs=${CONTIGS}" \
        "ppi.hotspot_res=${HOTSPOTS}" \
        diffuser.partial_T="${PARTIAL_T}" \
        inference.num_designs="${NUM_DESIGNS}" \
        denoiser.noise_scale_ca="${NOISE_CA}" \
        denoiser.noise_scale_frame="${NOISE_FRAME}"

    echo "[$(date +%H:%M:%S)] Finished orient_${bb}"
done
echo ""
echo "============================================================"
echo "  Tier ${TIER} complete: $(date)"
echo "  Outputs: ${R2_OUT_ROOT}/tier${TIER}/"
echo "============================================================"