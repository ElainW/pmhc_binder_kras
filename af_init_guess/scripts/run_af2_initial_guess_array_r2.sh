#!/bin/bash
#SBATCH -J af2_ig_kras
#SBATCH -p gpu_quad,gpu
#SBATCH --qos=gpuquad_qos
#SBATCH -t 3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=10G
#SBATCH -c 4
#SBATCH -o /n/groups/marks/users/aaron/pmhc/af_init_guess/logs/af2_ig_kras_%a.log
#SBATCH -e /n/groups/marks/users/aaron/pmhc/af_init_guess/logs/af2_ig_kras_%a.err

# -------------------------------------------------------
# AF2 Initial Guess — SLURM array job (KRAS)
# Submit with:
#   sbatch --array=0-N run_af2_initial_guess_array.sh
# where N = (total_chunks - 1) printed by split_silent.py
# -------------------------------------------------------

# --- Environment ---
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/af2_binder_design

# --- CUDA library paths (self-contained in conda env) ---
export LD_LIBRARY_PATH=/n/groups/marks/users/aaron/pmhc/envs/af2_binder_design/lib/python3.11/site-packages/nvidia/cudnn/lib:/n/groups/marks/users/aaron/pmhc/envs/af2_binder_design/lib:/n/groups/marks/users/aaron/pmhc/envs/af2_binder_design/lib/python3.11/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH

# --- Confirm GPU visible before running ---
python -c "
import jax
devices = jax.devices()
print('JAX devices:', devices)
if not any('cuda' in str(d).lower() for d in devices):
    import sys
    print('ERROR: No GPU detected — aborting')
    sys.exit(1)
"

SLURM_ARRAY_TASK_ID=0

if [ $? -ne 0 ]; then
    echo "ERROR: GPU not visible to JAX — check CUDA/cuDNN setup"
    exit 1
fi

# --- Paths ---
DL_BINDER_DESIGN_DIR="/n/groups/marks/users/aaron/pmhc/dl_binder_design"
CHUNK_DIR="/n/groups/marks/users/aaron/pmhc/af_init_guess/inputs/r2/kras_chunks"
OUT_DIR="/n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras"
LOG_DIR="/n/groups/marks/users/aaron/pmhc/af_init_guess/r2/logs"

# --- Per-job inputs/outputs based on array index ---
SILENT_IN="${CHUNK_DIR}/chunk_${SLURM_ARRAY_TASK_ID}.silent"
SILENT_OUT="${OUT_DIR}/af2_kras_predictions_${SLURM_ARRAY_TASK_ID}.silent"
SCOREFILEPATH="${OUT_DIR}/kras_scores_${SLURM_ARRAY_TASK_ID}.sc"
PAE_DIR="${OUT_DIR}/pae"

# --- Setup ---
mkdir -p ${OUT_DIR} ${LOG_DIR}

# --- Validate input exists ---
if [ ! -f "${SILENT_IN}" ]; then
    echo "ERROR: Input silent file not found: ${SILENT_IN}"
    exit 1
fi

echo "======================================================"
echo "SLURM job:     ${SLURM_JOB_ID}"
echo "Array task:    ${SLURM_ARRAY_TASK_ID}"
echo "Node:          $(hostname)"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Input:         ${SILENT_IN}"
echo "Output:        ${SILENT_OUT}"
echo "Scores:        ${SCOREFILEPATH}"
echo "Entries:       $(grep -c '^SCORE:' ${SILENT_IN} || echo 'unknown')"
echo "Start time:    $(date)"
echo "======================================================"

# --- Run AF2 initial guess ---
python ${DL_BINDER_DESIGN_DIR}/af2_initial_guess/predict.py \
    -silent ${SILENT_IN} \
    -outsilent ${SILENT_OUT} \
    -scorefilename ${SCOREFILEPATH} \
    -pae_outdir ${PAE_DIR} \
    -checkpoint_name 'r2.check.point'

EXIT_CODE=$?

echo "======================================================"
echo "End time:      $(date)"
echo "Exit code:     ${EXIT_CODE}"

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: predict.py failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

# --- Quick output check ---
if [ -f "${SCOREFILEPATH}" ]; then
    N_SCORED=$(grep -c '^SCORE:' ${SCOREFILEPATH} 2>/dev/null || echo 0)
    echo "Designs scored: ${N_SCORED}"
else
    echo "WARNING: Score file not created — check log for errors"
fi
echo "======================================================"
