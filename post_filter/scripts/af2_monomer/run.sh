#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

DIR=/n/groups/marks/users/aaron/pmhc/post_filter/
# 1. Prepare AF2 monomer FASTA --> did this manually
# python scripts/af2_monomer/prepare_af2_monomer_inputs.py \
#     --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
#     --pdb_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
#     --author_csv inputs/science.adv0185_data_s1.csv \
#     --out_dir inputs/r2/af2_monomer/af2_monomer_inputs.fasta

# 2. Split the fasta and submit sbatch jobs
# bash run_get_AF2_structures.sh $DIR/inputs/r2/monomer_fasta/af2_monomer_inputs.fasta $DIR/inputs/r2/monomer_fasta/ $DIR/outputs/r2/af2_monomer/ $DIR/slurm/r2/af2_monomer/ '2026-04-07'

# 3. Collect the scores after the sbatch job finishes
python collect_monomer_scores.py \
    --monomer_out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer/ \
    --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer/stats/
