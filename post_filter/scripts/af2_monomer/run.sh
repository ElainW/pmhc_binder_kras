#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/dl_binder_design

DIR=~/Desktop/pmhc_binder_kras_cp/post_filter/
# 1. Prepare AF2 monomer FASTA --> did this manually
# python scripts/af2_monomer/prepare_af2_monomer_inputs.py \
#     --merged_scores ~/Desktop/pmhc_binder_kras/specificity/analysis/r2/merged_scores_ranked.tsv \
#     --pdb_dir ~/Desktop/pmhc_binder_kras/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
#     --author_csv inputs/science.adv0185_data_s1.csv \
#     --out_dir inputs/r2/af2_monomer/af2_monomer_inputs.fasta

# # 2. Split the fasta and submit sbatch jobs
# bash run_get_AF2_structures.sh $DIR/inputs/r2/monomer_fasta/af2_monomer_inputs.fasta $DIR/inputs/r2/monomer_fasta/ $DIR/outputs/r2/af2_monomer/ $DIR/slurm/r2/af2_monomer/ '2026-04-07'

# # 3. Collect the scores after the sbatch job finishes
# python collect_monomer_scores.py \
#     --monomer_out_dir $DIR/outputs/r2/af2_monomer/ \
#     --out_dir $DIR/outputs/r2/af2_monomer/stats/

# 1. Prepare AF2 monomer FASTA
python $DIR/scripts/af2_monomer/prepare_af2_monomer_inputs.py \
    --merged_scores ~/Desktop/pmhc_binder_kras_cp/specificity/analysis/r2.5/merged_scores_ranked.tsv \
    --pdb_dir ~/Desktop/pmhc_binder_kras_cp/af_init_guess/outputs/r2.5/af2_kras/top_pdbs/ \
    --out_dir $DIR/inputs/r2.5/af2_monomer/

# 2. Split the fasta and submit sbatch jobs
bash run_get_AF2_structures.sh $DIR/inputs/r2.5/af2_monomer/af2_monomer_inputs.fasta $DIR/inputs/r2.5/af2_monomer/ $DIR/outputs/r2.5/af2_monomer/ $DIR/slurm/r2.5/af2_monomer/ '2026-04-07'

# 3. Collect the scores after the sbatch job finishes
python collect_monomer_scores.py \
    --monomer_out_dir $DIR/outputs/r2.5/af2_monomer/ \
    --out_dir $DIR/outputs/r2.5/af2_monomer/stats/
