#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

python cms_scoring.py \
  --pdb_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
  --out_csv  /n/groups/marks/users/aaron/pmhc/specificity/CMS/outputs/r2/cms_scores.tsv \
  --peptide_len 9 \
  --binder_chain A \
  --pmhc_chain B