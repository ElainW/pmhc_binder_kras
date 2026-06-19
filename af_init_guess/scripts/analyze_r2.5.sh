#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/dl_binder_design

export PATH=~/Desktop/pmhc_binder_kras/silent_tools:$PATH

python analyze_af2_scores.py \
    --score_dir  ~/Desktop/pmhc_binder_kras_cp/af_init_guess/outputs/r2.5/af2_kras/ \
    --silent_dir ~/Desktop/pmhc_binder_kras_cp/af_init_guess/outputs/r2.5/af2_kras/ \
    --output_csv  ~/Desktop/pmhc_binder_kras_cp/af_init_guess/outputs/r2.5/af2_kras_all_scores.csv \
    --passing_csv ~/Desktop/pmhc_binder_kras_cp/af_init_guess/outputs/r2.5/af2_kras_passing.csv \
    --top_n 20