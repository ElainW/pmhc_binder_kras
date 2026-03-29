#!/bin/bash

module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/af2_binder_design

# python calc_ipsae.py /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/af2_kras/pae/kras_orient_113_sample01_af2pred_pae.npy \
#     --pdb /n/groups/marks/users/aaron/pmhc/af_init_guess/inputs/threaded_pdbs/kras_orient_113_sample01.pdb
#     --name kras_orient_113_sample01_af2pred



python calc_ipsae.py /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/examples/pae/extracted_9O5S_reordered_af2pred_pae.npy \
    --binder 102 \
    --mhc 174 \
    --peptide 10 \
    --name extracted_9O5S_af2pred