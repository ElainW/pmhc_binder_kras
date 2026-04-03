#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

export PATH=/n/groups/marks/users/aaron/pmhc/silent_tools:$PATH

# Check how many chunks finished
ls /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/kras_scores_*.sc | wc -l

# Check for any failed chunks
for log in /n/groups/marks/users/aaron/pmhc/af_init_guess/r2/logs/af2_ig_kras_*.log; do
    if grep -q "ERROR\|exit code: [^0]" $log 2>/dev/null; then
        echo "Failed: $(basename $log)"
    fi
done

python analyze_af2_scores.py \
    --score_dir  /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/ \
    --silent_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/ \
    --output_csv  /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras_all_scores.csv \
    --passing_csv /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras_passing.csv \
    --top_n 20