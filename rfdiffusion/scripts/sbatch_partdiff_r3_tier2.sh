#!/bin/bash
#SBATCH --job-name=kras_partdiff_r3_t2
#SBATCH --partition=gpu_quad,gpu
#SBATCH --qos=gpuquad_qos
#SBATCH --time=8:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=../slurm/kras_partdiff_r3_tier2_%j.out
#SBATCH --error=../slurm/kras_partdiff_r3_tier2_%j.err
#SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --mail-user=wangyilan0304@gmail.com
# =============================================================================
# Partial Diffusion — KRAS pMHC Round 3, Tier 2
# Backbones with zero C4 contacts — moderate exploration toward C4
#
# orient_255_pT12__27   (0 C4 contacts)
# orient_255_pT12__53   (0 C4 contacts)
# orient_255_pT12__8    (0 C4 contacts)
# orient_255_pT12__91   (0 C4 contacts)
#
# partial_T=25, noise_scale_ca=0.5, noise_scale_frame=0.5
# Moderate backbone flexibility — allows repositioning toward C4
# without demolishing existing passing geometry
#
# Hotspots: C4, C6, C7 (peptide positions, mapped from PDB residue numbers)
# 50 designs per backbone → 200 total trajectories
#
# Chain layout in input PDBs:
#   Chain A = binder  (partially diffused)
#   Chain B = MHC     (fixed)
#   Chain C = peptide (fixed)
# =============================================================================
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PYTHON=/n/groups/marks/users/aaron/RFdiffusion/env/SE3nv/bin/python
SCRIPT=/n/groups/marks/users/aaron/RFdiffusion/scripts/run_inference.py
R2_PDB_DIR=/n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/filtered
R3_OUT_ROOT=/n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3

# ── Tier 2 settings ───────────────────────────────────────────────────────────
BACKBONES=(
    "orient_255_pT12__27"
    "orient_255_pT12__53"
    "orient_255_pT12__8"
    "orient_255_pT12__91"
)
PARTIAL_T=25
NUM_DESIGNS=50
NOISE_CA=0.5
NOISE_FRAME=0.5
TIER=2

# ── Helpers ───────────────────────────────────────────────────────────────────
get_chain_range() {
    local pdb=$1 chain=$2
    local start end
    start=$(grep "^ATOM" "$pdb" | awk -v c="$chain" '$5==c {print $6+0}' | sort -n | head -1)
    end=$(  grep "^ATOM" "$pdb" | awk -v c="$chain" '$5==c {print $6+0}' | sort -n | tail -1)
    echo "${start}-${end}"
}

get_chain_res() {
    local pdb=$1 chain=$2 n=$3
    grep "^ATOM" "$pdb" \
        | awk -v c="$chain" '$5==c {print $6+0}' \
        | sort -un \
        | sed -n "${n}p"
}

# ── Run ───────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Round 3 Tier ${TIER} | partial_T=${PARTIAL_T} | n=${NUM_DESIGNS}"
echo "  noise_scale_ca=${NOISE_CA} | noise_scale_frame=${NOISE_FRAME}"
echo "  Backbones: ${BACKBONES[*]}"
echo "  SLURM job ID: ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo "  Started: $(date)"
echo "============================================================"

for bb in "${BACKBONES[@]}"; do
    input_pdb="${R2_PDB_DIR}/${bb}.pdb"
    out_prefix="${R3_OUT_ROOT}/tier${TIER}/${bb}/${bb}_pT${PARTIAL_T}_"

    if [[ ! -f "$input_pdb" ]]; then
        echo "[WARN] PDB not found: ${input_pdb} — skipping ${bb}"
        continue
    fi

    BINDER_LEN=$(grep "^ATOM" "$input_pdb" | awk '$5=="A" {print $6+0}' | sort -un | wc -l)
    RANGE_B=$(get_chain_range "$input_pdb" "B")
    RANGE_C=$(get_chain_range "$input_pdb" "C")
    CONTIGS="[${BINDER_LEN}-${BINDER_LEN}/0 B${RANGE_B}/0 C${RANGE_C}]"

    HS1=$(get_chain_res "$input_pdb" "C" 4)
    HS2=$(get_chain_res "$input_pdb" "C" 6)
    HS3=$(get_chain_res "$input_pdb" "C" 7)
    HOTSPOTS="[C${HS1},C${HS2},C${HS3}]"

    echo ""
    echo "[$(date +%H:%M:%S)] Starting ${bb}"
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

    echo "[$(date +%H:%M:%S)] Finished ${bb}"
done

echo ""
echo "============================================================"
echo "  Tier ${TIER} complete: $(date)"
echo "  Outputs: ${R3_OUT_ROOT}/tier${TIER}/"
echo "============================================================"