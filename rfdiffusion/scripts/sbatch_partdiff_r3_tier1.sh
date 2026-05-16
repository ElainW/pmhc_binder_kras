#!/bin/bash
#SBATCH --job-name=kras_partdiff_r3_t1
#SBATCH --partition=gpu_quad,gpu
#SBATCH --qos=gpuquad_qos
#SBATCH --time=8:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=../slurm/kras_partdiff_r3_tier1_%j.out
#SBATCH --error=../slurm/kras_partdiff_r3_tier1_%j.err
#SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --mail-user=wangyilan0304@gmail.com
# =============================================================================
# Partial Diffusion — KRAS pMHC Round 3, Tier 1
# Backbones with existing C4 contacts — conservative refinement
#
# orient_252_pT20__31   (1 C4 contact)
# orient_255_pT12__2    (1 C4 contact, sample08)
# orient_255_pT12__18   (1 C4 contact)
# orient_255_pT12__31   (1 C4 contact)
# orient_255_pT12__33   (1-3 C4 contacts across samples)
# orient_255_pT12__46   (1 C4 contact)
# orient_255_pT12__70   (2 C4 contacts)
# orient_34_pT12__67    (1 C4 contact)
#
# partial_T=12, noise_scale_ca=0, noise_scale_frame=0
# Near-deterministic refinement — preserves existing backbone geometry
# while biasing ProteinMPNN toward C4 engagement via hotspot
#
# Preprocessing: last 9 residues of chain B are split into chain C
# so that hotspot residues C4, C6, C7 map correctly.
# Split PDBs are written to R3_SPLIT_DIR and not kept after the job.
#
# Hotspots: C4, C6, C7 (peptide positions 4, 6, 7)
# 100 designs per backbone → 800 total trajectories
#
# Chain layout in split input PDBs:
#   Chain A = binder  (partially diffused)
#   Chain B = MHC     (fixed)
#   Chain C = peptide (fixed, renumbered 1-9)
# =============================================================================
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PYTHON=/n/groups/marks/users/aaron/RFdiffusion/env/SE3nv/bin/python
SCRIPT=/n/groups/marks/users/aaron/RFdiffusion/scripts/run_inference.py
SPLIT_SCRIPT=/n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/scripts/split_peptide_chain.py

R2_PDB_DIR=/n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/filtered
R3_SPLIT_DIR=/n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3/split_pdbs
R3_OUT_ROOT=/n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3

# ── Tier 1 settings ───────────────────────────────────────────────────────────
BACKBONES=(
    "orient_252_pT20__31"
    "orient_255_pT12__2"
    "orient_255_pT12__18"
    "orient_255_pT12__31"
    "orient_255_pT12__33"
    "orient_255_pT12__46"
    "orient_255_pT12__70"
    "orient_34_pT12__67"
)
PARTIAL_T=12
NUM_DESIGNS=100
NOISE_CA=0
NOISE_FRAME=0
TIER=1

# ── Preprocess: split peptide into chain C ────────────────────────────────────
echo "Splitting peptide chain for tier ${TIER} backbones..."
"$PYTHON" "$SPLIT_SCRIPT" \
    --input_dir  "$R2_PDB_DIR" \
    --output_dir "$R3_SPLIT_DIR" \
    --backbones  "${BACKBONES[@]}"

# ── Helpers ───────────────────────────────────────────────────────────────────
get_chain_range() {
    local pdb=$1 chain=$2
    local start end
    start=$(grep "^ATOM" "$pdb" | awk -v c="$chain" 'substr($0,22,1)==c {print substr($0,23,4)+0}' | sort -n | head -1)
    end=$(  grep "^ATOM" "$pdb" | awk -v c="$chain" 'substr($0,22,1)==c {print substr($0,23,4)+0}' | sort -n | tail -1)
    echo "${start}-${end}"
}

get_chain_res() {
    local pdb=$1 chain=$2 n=$3
    grep "^ATOM" "$pdb" \
        | awk -v c="$chain" 'substr($0,22,1)==c {print substr($0,23,4)+0}' \
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
    input_pdb="${R3_SPLIT_DIR}/${bb}.pdb"
    out_prefix="${R3_OUT_ROOT}/tier${TIER}/${bb}/${bb}_pT${PARTIAL_T}_"

    if [[ ! -f "$input_pdb" ]]; then
        echo "[WARN] Split PDB not found: ${input_pdb} — skipping ${bb}"
        continue
    fi

    BINDER_LEN=$(grep "^ATOM" "$input_pdb" | awk 'substr($0,22,1)=="A" {print substr($0,23,4)+0}' | sort -un | wc -l)
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