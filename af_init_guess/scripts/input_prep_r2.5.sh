#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

mkdir -p /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/inputs/r2.5/

python thread_fasta_to_backbones.py \
    --fasta_file /n/groups/marks/users/aaron/pmhc_cp/proteinmpnn/outputs/r2.5/all_sequences.fa \
    --backbone_dir /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/filtered/ \
    --output_dir /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/inputs/r2.5/threaded_pdbs/ \
    --verify \
    --verbose

/n/groups/marks/users/aaron/pmhc/silent_tools/silentfrompdbs /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/inputs/r2.5/threaded_pdbs/*.pdb > /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/inputs/r2.5/threaded_kras_designs.silent
