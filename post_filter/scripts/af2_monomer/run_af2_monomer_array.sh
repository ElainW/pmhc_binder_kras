#!/bin/bash
#SBATCH -J af2_mono_r2
#SBATCH -p gpu_quad,gpu
#SBATCH --qos=gpuquad_qos
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -o /n/groups/marks/users/aaron/pmhc/post_filter/logs/af2_monomer_%a.log
#SBATCH -e /n/groups/marks/users/aaron/pmhc/post_filter/logs/af2_monomer_%a.err

# -------------------------------------------------------
# AF2 Monomer — SLURM array job (Round 2 post-filter)
#
# Folds each binder as a monomer (no pMHC) using
# dl_binder_design predict.py -force_monomer -no_initial_guess.
# Input: per-chunk PDB dirs from prepare_monomer_pdbs.py
# Output: predicted PDBs + score .sc file per chunk
#
# Prepare inputs first (run once on login node):
#   python prepare_monomer_pdbs.py \
#       --merged_scores .../merged_scores_ranked.tsv \
#       --split_dir  /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/ \
#       --author_csv .../science_adv0185_data_s1.csv \
#       --af2_pdb_dir .../top_pdbs/ \
#       --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/monomer_pdbs/ \
#       --chunk_size 5
#   (note the "Submit with: sbatch --array=0-N" line it prints)
#
# Then submit:
#   sbatch --array=0-N run_af2_monomer_array.sh
# -------------------------------------------------------

# ── Environment ──────────────────────────────────────────────────────────────
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/af2_binder_design

export LD_LIBRARY_PATH=\
/n/groups/marks/users/aaron/pmhc/envs/af2_binder_design/lib/python3.11/site-packages/nvidia/cudnn/lib:\
/n/groups/marks/users/aaron/pmhc/envs/af2_binder_design/lib:\
/n/groups/marks/users/aaron/pmhc/envs/af2_binder_design/lib/python3.11/site-packages/nvidia/cusolver/lib:\
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
DL_BINDER_DESIGN_DIR="/n/groups/marks/users/aaron/pmhc/dl_binder_design"
CHUNK_BASE="/n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/monomer_pdbs"
OUT_BASE="/n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer"
LOG_DIR="/n/groups/marks/users/aaron/pmhc/post_filter/logs"

CHUNK_PDBDIR="${CHUNK_BASE}/chunk_${SLURM_ARRAY_TASK_ID}"
CHUNK_OUTDIR="${OUT_BASE}/chunk_${SLURM_ARRAY_TASK_ID}"
SCOREFILE="${CHUNK_OUTDIR}/monomer_scores.sc"

mkdir -p "${CHUNK_OUTDIR}" "${LOG_DIR}"

# ── Validate input ────────────────────────────────────────────────────────────
if [ ! -d "${CHUNK_PDBDIR}" ]; then
    echo "ERROR: chunk PDB dir not found: ${CHUNK_PDBDIR}"
    exit 1
fi

N_PDBS=$(ls "${CHUNK_PDBDIR}"/*.pdb 2>/dev/null | wc -l)

echo "======================================================"
echo "SLURM job:    ${SLURM_JOB_ID}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Input PDBs:   ${CHUNK_PDBDIR}  (${N_PDBS} PDBs)"
echo "Output dir:   ${CHUNK_OUTDIR}"
echo "Score file:   ${SCOREFILE}"
echo "Start time:   $(date)"
echo "======================================================"

# ── Run AF2 monomer ───────────────────────────────────────────────────────────
# -force_monomer    treat entire input as a single-chain monomer (no binder/target split)
# -no_initial_guess fold from sequence only; skip complex-mode templating
# -pdbdir           directory of single-chain PDB inputs
# -outpdbdir        write predicted monomer PDBs here
# -scorefilename    per-chunk .sc score file
python "${DL_BINDER_DESIGN_DIR}/af2_initial_guess/predict.py" \
    -pdbdir         "${CHUNK_PDBDIR}"  \
    -outpdbdir      "${CHUNK_OUTDIR}"  \
    -scorefilename  "${SCOREFILE}"     \
    -force_monomer                     \
    -no_initial_guess

EXIT_CODE=$?

echo "======================================================"
echo "End time:    $(date)"
echo "Exit code:   ${EXIT_CODE}"
if [ -f "${SCOREFILE}" ]; then
    echo "Score lines: $(grep -c '^SCORE:' "${SCOREFILE}" || echo 0)"
fi
echo "======================================================"

exit ${EXIT_CODE}