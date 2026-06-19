#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/dl_binder_design

export PATH=~/Desktop/pmhc_binder_kras/silent_tools:$PATH

# Check how many chunks finished
ls ~/Desktop/pmhc_binder_kras/af_init_guess/outputs/af2_kras/kras_scores_*.sc | wc -l

# Check for any failed chunks
# for log in ~/Desktop/pmhc_binder_kras/af_init_guess/logs/af2_ig_kras_*.log; do
#     if grep -q "ERROR\|exit code: [^0]" $log 2>/dev/null; then
#         echo "Failed: $(basename $log)"
#     fi
# done

python analyze_af2_scores.py \
    --score_dir  ~/Desktop/pmhc_binder_kras/af_init_guess/outputs/af2_kras/ \
    --silent_dir ~/Desktop/pmhc_binder_kras/af_init_guess/outputs/af2_kras/ \
    --output_csv  af2_kras_all_scores.csv \
    --passing_csv af2_kras_passing.csv \
    --top_n 20