#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/dl_binder_design

mkdir -p ~/Desktop/pmhc_binder_kras/specificity/CMS/outputs/r2/

python cms_scoring.py \
  --pdb_dir ~/Desktop/pmhc_binder_kras/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
  --out_csv  ~/Desktop/pmhc_binder_kras/specificity/CMS/outputs/r2/cms_scores.tsv \
  --peptide_len 9 \
  --binder_chain A \
  --pmhc_chain B