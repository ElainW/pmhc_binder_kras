#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

# 0. Split pmhc fold output pdbs into binder (chain A) and pMHC (chain B)
python split_pmhc_fold_pdbs.py \
    --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
    --af2_pdb_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
    --pmhc_fold_dir /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/outputs/r2/pmhc_fold_on_runs \
    --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/

# 1. Charge/cysteine audit (fast, CPU)
python charge_cysteine_audit.py \
    --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
    --split_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/ \
    --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/

# 2. Contact map (CPU, ~5 min for 49 designs)
python contact_map.py \
    --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
    --split_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/ \
    --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/

# 3. RMSD (CPU, fast)
python compute_rmsd.py \
    --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
    --af2_pdb_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
    --split_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/ \
    --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2

# 5. FastRelax (CPU with PyRosetta, ~2 min per design)
# Run on a CPU node — no GPU needed
python fastrelax_designs.py \
    --merged_scores /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
    --split_dir /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/ \
    --out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/fastrelax/ \
    --n_repeats 3