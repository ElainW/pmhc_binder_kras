#!/bin/bash
#SBATCH --job-name=kras_partdiff_t1
#SBATCH --partition=gpu_quad,gpu
#SBATCH --time=8:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=../slurm/kras_partdiff_t1_%j.out
#SBATCH --error=../slurm/kras_partdiff_t1_%j.err
#SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --mail-user=wangyilan0304@gmail.com
# =============================================================================
# Tier 1 Partial Diffusion — KRAS pMHC Round 2
# Backbones: orient_247, 34, 47, 255, 260, 83, 158  (best pAE < 6)
# partial_T=12  →  conservative perturbation, 100 designs per backbone
# Total trajectories: 700
# =============================================================================

set -euo pipefail


# ── Paths ─────────────────────────────────────────────────────────────────────
PYTHON=/n/groups/marks/users/aaron/RFdiffusion/env/SE3nv/bin/python
SCRIPT=/n/groups/marks/users/aaron/RFdiffusion/scripts/run_inference.py
R1_PDB_DIR=/n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/1000_filtered/orientation_filtered
R2_OUT_ROOT=/n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2

# ── Tier 1 settings ───────────────────────────────────────────────────────────
BACKBONES=(247 34 47 255 260 83 158)
PARTIAL_T=12
NUM_DESIGNS=100
TIER=1

# ── Shared diffusion settings ─────────────────────────────────────────────────
HOTSPOTS='[C6,C7]'
NOISE_CA=0
NOISE_FRAME=0

# ── Helper: get residue range for a given chain in a PDB ──────────────────────
get_chain_range() {
    local pdb=$1 chain=$2
    local start end
    start=$(grep "^ATOM" "$pdb" | awk -v c="$chain" '$5==c {print $6+0}' | sort -n | head -1)
    end=$(grep   "^ATOM" "$pdb" | awk -v c="$chain" '$5==c {print $6+0}' | sort -n | tail -1)
    echo "${start}-${end}"
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

    # Build contig dynamically from this backbone's chain lengths
    RANGE_A=$(get_chain_range "$input_pdb" "A")
    RANGE_B=$(get_chain_range "$input_pdb" "B")
    RANGE_C=$(get_chain_range "$input_pdb" "C")
    CONTIGS="[B${RANGE_B}/0 C${RANGE_C}/0 A${RANGE_A}]"

    echo ""
    echo "[$(date +%H:%M:%S)] Starting orient_${bb} (partial_T=${PARTIAL_T}, n=${NUM_DESIGNS})"
    echo "  Chain A (binder): ${RANGE_A} | Chain B: ${RANGE_B} | Chain C: ${RANGE_C}"
    echo "  Contig: ${CONTIGS}"
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
