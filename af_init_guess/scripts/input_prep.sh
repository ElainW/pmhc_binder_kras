#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

python thread_fasta_to_backbones.py \
    --fasta_dir /n/groups/marks/users/aaron/pmhc/proteinmpnn/outputs/kras/seqs/ \
    --backbone_dir /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/1000_filtered/orientation_filtered/rep/ \
    --output_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/inputs/threaded_pdbs/ \
    --verify \
    --verbose

python merge_chains_bc.py \
    --input_dir ../inputs/threaded_pdbs/ \
    --output_dir ../inputs/threaded_pdbs_merged/

/n/groups/marks/users/aaron/pmhc/silent_tools/silentfrompdbs /n/groups/marks/users/aaron/pmhc/af_init_guess/inputs/threaded_pdbs_merged/*.pdb > /n/groups/marks/users/aaron/pmhc/af_init_guess/inputs/threaded_kras_designs.silent

python split_silent.py \
    --silent ../inputs/threaded_kras_designs.silent \
    --output_dir ../inputs/kras_chunks/ \
    --chunk_size 50

# Verify chunks have 2 chains
grep "RES_NUM" \
    /n/groups/marks/users/aaron/pmhc/af_init_guess/inputs/kras_chunks/chunk_0.silent | head -3