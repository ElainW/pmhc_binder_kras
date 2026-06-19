#!/bin/bash
#SBATCH -J af2_mono_r2.5
#SBATCH -p gpu_quad,gpu
#SBATCH --qos=gpuquad_qos
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -o ~/Desktop/pmhc_binder_kras_cp/post_filter/slurm/r2.5/af2_monomer/af2_monomer_r2.5_%a.log
#SBATCH -e ~/Desktop/pmhc_binder_kras_cp/post_filter/slurm/r2.5/af2_monomer/af2_monomer_r2.5_%a.err

# -------------------------------------------------------
# AF2 Monomer — SLURM array job (Round 2.5)
# One job per design, predicts from sequence only (no initial guess).
#
# Prepare stub PDBs first (run once on login node):
#   python ~/Desktop/pmhc_binder_kras_cp/post_filter/scripts/prepare_monomer_stubs.py \
#       --fasta   ~/Desktop/pmhc_binder_kras_cp/post_filter/inputs/r2.5/af2_monomer/af2_monomer_inputs.fasta \
#       --out_dir ~/Desktop/pmhc_binder_kras_cp/post_filter/inputs/r2.5/af2_monomer/stubs/
#
# Then submit (replace N with len(designs)-1 printed by prepare_monomer_stubs.py):
#   sbatch --array=0-N run_af2_monomer_r2.5_array.sh
# -------------------------------------------------------

# ── Environment ──────────────────────────────────────────────────────────────
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/af2_binder_design

export LD_LIBRARY_PATH=\
~/Desktop/pmhc_binder_kras/envs/af2_binder_design/lib/python3.11/site-packages/nvidia/cudnn/lib:\
~/Desktop/pmhc_binder_kras/envs/af2_binder_design/lib:\
~/Desktop/pmhc_binder_kras/envs/af2_binder_design/lib/python3.11/site-packages/nvidia/cusolver/lib:\
$LD_LIBRARY_PATH

# ── Validate GPU ─────────────────────────────────────────────────────────────
python -c "
import jax
devices = jax.devices()
print('JAX devices:', devices)
if not any('cuda' in str(d).lower() for d in devices):
    import sys; sys.exit(1)
"
if [ $? -ne 0 ]; then
    echo "ERROR: No GPU visible to JAX — aborting"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
DL_BINDER_DESIGN_DIR="~/Desktop/pmhc_binder_kras/dl_binder_design"
STUB_BASE="~/Desktop/pmhc_binder_kras_cp/post_filter/inputs/r2.5/af2_monomer/stubs"
PDB_DIR="${STUB_BASE}/pdbs"
RUNLIST_DIR="${STUB_BASE}/runlists"
OUT_DIR="~/Desktop/pmhc_binder_kras_cp/post_filter/outputs/r2.5/af2_monomer"
INDEX_FILE="${STUB_BASE}/design_index.tsv"

mkdir -p "${OUT_DIR}"

# ── Look up design for this array task ───────────────────────────────────────
# design_index.tsv columns: array_idx  design  runlist
SLURM_ARRAY_TASK_ID=0
DESIGN=$(awk -F'\t' -v idx="${SLURM_ARRAY_TASK_ID}" \
    'NR>1 && $1==idx {print $2}' "${INDEX_FILE}")
RUNLIST=$(awk -F'\t' -v idx="${SLURM_ARRAY_TASK_ID}" \
    'NR>1 && $1==idx {print $3}' "${INDEX_FILE}")

if [ -z "${DESIGN}" ]; then
    echo "ERROR: no design found for array index ${SLURM_ARRAY_TASK_ID} in ${INDEX_FILE}"
    exit 1
fi

# Per-design output dir and score file
DESIGN_OUT="${OUT_DIR}/${DESIGN}"
SCOREFILE="${DESIGN_OUT}/monomer_scores.sc"
mkdir -p "${DESIGN_OUT}"

echo "======================================================"
#echo "SLURM job:    ${SLURM_JOB_ID}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
echo "Design:       ${DESIGN}"
echo "PDB dir:      ${PDB_DIR}"
echo "Runlist:      ${RUNLIST}"
echo "Output dir:   ${DESIGN_OUT}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Start time:   $(date)"
echo "======================================================"

# ── Run AF2 monomer ───────────────────────────────────────────────────────────
# -force_monomer    fold as single chain (no binder/target split)
# -no_initial_guess predict from sequence only; ignore stub PDB coordinates
# -pdbdir           directory containing the stub PDB for this design
# -outpdbdir        write predicted monomer PDB here
# -scorefilename    per-design .sc score file
python "${DL_BINDER_DESIGN_DIR}/af2_initial_guess/predict.py" \
    -pdbdir        "${PDB_DIR}" \
    -outpdbdir     "${DESIGN_OUT}"      \
    -pae_outdir    "${DESIGN_OUT}/pae"
    -scorefilename "${SCOREFILE}"       \
    -runlist       "${RUNLIST}"         \
    -force_monomer                      \
    -no_initial_guess \
    -debug

EXIT_CODE=$?

echo "======================================================"
echo "End time:    $(date)"
echo "Exit code:   ${EXIT_CODE}"
if [ -f "${SCOREFILE}" ]; then
    echo "Score lines: $(grep -c '^SCORE:' "${SCOREFILE}" || echo 0)"
fi
echo "======================================================"

exit ${EXIT_CODE}
