#!/bin/bash

# --- Environment ---
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/af2_binder_design

# --- CUDA library paths (self-contained in conda env) ---
export LD_LIBRARY_PATH=~/Desktop/pmhc_binder_kras/envs/af2_binder_design/lib/python3.11/site-packages/nvidia/cudnn/lib:~/Desktop/pmhc_binder_kras/envs/af2_binder_design/lib:~/Desktop/pmhc_binder_kras/envs/af2_binder_design/lib/python3.11/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH

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

if [ $? -ne 0 ]; then
    echo "ERROR: GPU not visible to JAX — check CUDA/cuDNN setup"
    exit 1
fi

# --- Paths ---
DL_BINDER_DESIGN_DIR="~/Desktop/pmhc_binder_kras/dl_binder_design"
PDB_DIR="~/Desktop/pmhc_binder_kras/af_init_guess/inputs/examples"
OUT_DIR="~/Desktop/pmhc_binder_kras/af_init_guess/outputs/examples"
LOG_DIR="~/Desktop/pmhc_binder_kras/af_init_guess/logs_examples"

# --- Per-job inputs/outputs based on array index ---
PDB_IN="${PDB_DIR}/extracted_9O5S_reordered.pdb" # run reorder script first
PDB_OUT="${OUT_DIR}/af2_kras_predictions_extracted_9O5S.pdb"
SCOREFILEPATH="${OUT_DIR}/extracted_9O5S.sc"
PAE_DIR="${OUT_DIR}/pae"

# --- Setup ---
mkdir -p ${OUT_DIR} ${LOG_DIR}

# --- Validate input exists ---
if [ ! -f "${PDB_IN}" ]; then
    echo "ERROR: Input silent file not found: ${PDB_IN}"
    exit 1
fi

echo "======================================================"
echo "Node:          $(hostname)"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Input:         ${PDB_IN}"
echo "Output:        ${PDB_OUT}"
echo "Scores:        ${SCOREFILEPATH}"
echo "Entries:       $(grep -c '^SCORE:' ${PDB_IN} || echo 'unknown')"
echo "Start time:    $(date)"
echo "======================================================"

# --- Run AF2 initial guess ---
python ${DL_BINDER_DESIGN_DIR}/af2_initial_guess/predict.py \
    -pdbdir ${PDB_DIR} \
    -runlist 'example_run_list.txt' \
    -outpdbdir ${OUT_DIR} \
    -scorefilename ${SCOREFILEPATH} \
    -pae_outdir ${PAE_DIR} \
    -checkpoint_name 'example.check.point'

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
