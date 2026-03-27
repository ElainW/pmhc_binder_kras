#!/bin/bash
#SBATCH --job-name=kras_partdiff_t3
#SBATCH --partition=gpu_quad,gpu
#SBATCH --time=8:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=../slurm/kras_partdiff_t3_%j.out
#SBATCH --error=../slurm/kras_partdiff_t3_%j.err
#SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --mail-user=wangyilan0304@gmail.com
# =============================================================================
# Tier 3 Partial Diffusion — KRAS pMHC Round 2
# Backbones: orient_166, 324, 95, 51, 348  (best pAE 10–15)
# partial_T=25  →  aggressive perturbation, 20 designs per backbone
# Total trajectories: 100
# =============================================================================
set -euo pipefail
# ── Paths ─────────────────────────────────────────────────────────────────────
PYTHON=/n/groups/marks/users/aaron/RFdiffusion/env/SE3nv/bin/python
SCRIPT=/n/groups/marks/users/aaron/RFdiffusion/scripts/run_inference.py
R1_PDB_DIR=/n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/1000_filtered/orientation_filtered
R2_OUT_ROOT=/n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2
# ── Tier 3 settings ───────────────────────────────────────────────────────────
BACKBONES=(166 324 95 51 348)
PARTIAL_T=25
NUM_DESIGNS=20
TIER=3
# ── Shared diffusion settings ─────────────────────────────────────────────────
CONTIGS='[B3-180/0 C1-9/0 50-100]'
HOTSPOTS='[C6,C7]'
NOISE_CA=0
NOISE_FRAME=0
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
    echo ""
    echo "[$(date +%H:%M:%S)] Starting orient_${bb} (partial_T=${PARTIAL_T}, n=${NUM_DESIGNS})"
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