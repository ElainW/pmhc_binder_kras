#!/bin/bash
#SBATCH --job-name=mpnn_r3
#SBATCH --partition=short
#SBATCH --time=6:00:00
#SBATCH --mem=3G
#SBATCH --ntasks=1
#SBATCH --output=../slurm/mpnn_r3_%a.out
#SBATCH --error=../slurm/mpnn_r3_%a.err
# Array range is set at submission time (see Step 2 below).
# With 349 PDBs and chunk_size=50: 7 jobs, ~4h each.

PYTHON_ENV="/n/groups/marks/users/aaron/pmhc_cp/envs/dl_binder_design/bin/python"
MPNN_ENV="/n/groups/marks/users/aaron/pmhc_cp/envs/proteinmpnn/bin/python"
PROTEINMPNN_SCRIPT_DIR="/n/groups/marks/users/aaron/enzymes/ProteinMPNN"
SCRIPT="/n/groups/marks/users/aaron/pmhc_cp/proteinmpnn/scripts/redesign_w_charge_residue_filter.py"

PDB_DIR="/n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3/clustered"
OUTPUT_DIR="/n/groups/marks/users/aaron/pmhc_cp/proteinmpnn/outputs/kras_r3"
DESIGN_LIST="${OUTPUT_DIR}/design_list.txt"

${PYTHON_ENV} ${SCRIPT} \
    --pdb_dir     ${PDB_DIR} \
    --out_dir     ${OUTPUT_DIR} \
    --mpnn_path   ${MPNN_ENV} \
    --mpnn_script ${PROTEINMPNN_SCRIPT_DIR}/protein_mpnn_run.py \
    --design_list ${DESIGN_LIST} \
    --design_idx  ${SLURM_ARRAY_TASK_ID} \
    --chunk_size  50
