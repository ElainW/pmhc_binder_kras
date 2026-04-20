#!/bin/bash
module load conda/miniforge3/24.11.3-0
mamba activate /n/groups/marks/users/aaron/pmhc/env/esm

export model_checkpoint=/n/groups/marks/users/aaron/pmhc/specificity/ESMIF/inputs/esm_if1_gvp4_t16_142M_UR50.pt

OUTPUT_DIR=/n/groups/marks/users/aaron/pmhc/specificity/ESMIF/outputs/r2/
mkdir -p ${OUTPUT_DIR}

python ${SCRIPT_DIR}/baselines/esm/compute_fitness_esm_if1.py \
    --model_location ${model_checkpoint} \
    --structure_folder ${DMS_structure_folder} \
    --DMS_index $i \
    --DMS_reference_file_path ${DMS_reference_file_path_subs} \
    --DMS_data_folder ${DMS_data_folder_subs} \
    --output_scores_folder ${OUTPUT_DIR}
