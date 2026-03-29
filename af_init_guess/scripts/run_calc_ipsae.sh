#!/bin/bash

module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/af2_binder_design

python pmhci_ipsae.py /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/af2_kras/pae/kras_orient_0_sample01_af2pred_pae.npy \
    --binder 93 \
    --mhc 178 \
    --peptide 9 \
    --name my_design