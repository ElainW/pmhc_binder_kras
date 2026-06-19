#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/dl_binder_design

# 0. Split pmhc fold output pdbs into binder (chain A) and pMHC (chain B)
python split_pmhc_fold_pdbs.py \
    --merged_scores ~/Desktop/pmhc_binder_kras/specificity/analysis/r2/merged_scores_ranked.tsv \
    --af2_pdb_dir ~/Desktop/pmhc_binder_kras/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
    --pmhc_fold_dir ~/Desktop/pmhc_binder_kras/specificity/pMHC_fold/outputs/r2/pmhc_fold_on_runs \
    --out_dir ~/Desktop/pmhc_binder_kras/post_filter/inputs/r2/

# # 1. Charge/cysteine audit (fast, CPU)
python charge_cysteine_audit.py \
    --merged_scores ~/Desktop/pmhc_binder_kras/specificity/analysis/r2/merged_scores_ranked.tsv \
    --split_dir ~/Desktop/pmhc_binder_kras/post_filter/inputs/r2/ \
    --out_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/

# 2. Contact map (CPU, ~5 min for 49 designs)
# unrelaxed structures
python contact_map.py \
    --merged_scores ~/Desktop/pmhc_binder_kras/specificity/analysis/r2/merged_scores_ranked.tsv \
    --split_dir ~/Desktop/pmhc_binder_kras/post_filter/inputs/r2/ \
    --out_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/contact_unrelaxed/

# 3. RMSD (CPU, fast)
python compute_rmsd.py \
    --merged_scores ~/Desktop/pmhc_binder_kras/specificity/analysis/r2/merged_scores_ranked.tsv \
    --af2_pdb_dir ~/Desktop/pmhc_binder_kras/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
    --split_dir ~/Desktop/pmhc_binder_kras/post_filter/inputs/r2/ \
    --out_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2

# 5. FastRelax (CPU with PyRosetta, ~2 min per design)
# Run on a CPU node — no GPU needed
python fastrelax_designs.py \
    --merged_scores ~/Desktop/pmhc_binder_kras/specificity/analysis/r2/merged_scores_ranked.tsv \
    --split_dir ~/Desktop/pmhc_binder_kras/post_filter/inputs/r2/ \
    --out_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/fastrelax/ \
    --n_repeats 3

# Contact map on fastrelaxed structures
python contact_map.py \
    --merged_scores ~/Desktop/pmhc_binder_kras/specificity/analysis/r2/merged_scores_ranked.tsv \
    --split_dir ~/Desktop/pmhc_binder_kras/post_filter/inputs/r2/ \
    --relaxed_pdb_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/fastrelax/relaxed_pdbs/ \
    --out_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/contact_relaxed/

# compare_contact.py is run when no hydrogen bond is incorporated, since only relaxed structures have H