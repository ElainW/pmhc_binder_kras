#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

export PATH=/n/groups/marks/users/aaron/pmhc/silent_tools:$PATH

python analyze_af2_scores.py \
    --score_dir  /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/outputs/r2.5/af2_kras/ \
    --silent_dir /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/outputs/r2.5/af2_kras/ \
    --output_csv  /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/outputs/r2.5/af2_kras_all_scores.csv \
    --passing_csv /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/outputs/r2.5/af2_kras_passing.csv \
    --top_n 20