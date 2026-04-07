#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

# 1. Prepare AF2 monomer FASTA
python prepare_af2_monomer_inputs.py \
    --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
    --pdb_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
    --author_csv /n/groups/marks/users/aaron/pmhc/post_filter/inputs/science.adv0185_data_s1.csv \
    --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer/