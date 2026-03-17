#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

python thread_fasta_to_backbones.py \
    --fasta_dir /n/groups/marks/users/aaron/pmhc/proteinmpnn/outputs/kras/seqs/ \
    --backbone_dir /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/1000_filtered/orientation_filtered/rep/ \
    --output_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/inputs/threaded_pdbs/ \
    --verify \
    --verbose

/n/groups/marks/users/aaron/pmhc/silent_tools/silentfrompdbs /n/groups/marks/users/aaron/pmhc/af_init_guess/inputs/threaded_pdbs/*.pdb > /n/groups/marks/users/aaron/pmhc/af_init_guess/inputs/threaded_kras_designs.silent