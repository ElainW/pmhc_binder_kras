#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

python generate_pmhc_fold_csvs.py \
    --pdb_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
    --wt_chainB /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/inputs/extracted_8I5E.pdb \
    --off_target_pdb_dir /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/inputs/r2/off_target_pdbs/ \
    --out_dir /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/outputs/r2/