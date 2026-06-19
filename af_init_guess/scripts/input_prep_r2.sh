#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/dl_binder_design

mkdir -p ~/Desktop/pmhc_binder_kras/af_init_guess/inputs/r2/

python thread_fasta_to_backbones.py \
    --fasta_dir ~/Desktop/pmhc_binder_kras/proteinmpnn/outputs/kras_r2/seqs/ \
    --backbone_dir ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/partial_r2/filtered/ \
    --output_dir ~/Desktop/pmhc_binder_kras/af_init_guess/inputs/r2/threaded_pdbs/ \
    --verify \
    --verbose

~/Desktop/pmhc_binder_kras/silent_tools/silentfrompdbs ~/Desktop/pmhc_binder_kras/af_init_guess/inputs/r2/threaded_pdbs/*.pdb > ~/Desktop/pmhc_binder_kras/af_init_guess/inputs/r2/threaded_kras_designs.silent

python split_silent.py \
    --silent ../inputs/r2/threaded_kras_designs.silent \
    --output_dir ../inputs/r2/kras_chunks/ \
    --chunk_size 200

# Verify chunks have 2 chains
grep "RES_NUM" \
    ~/Desktop/pmhc_binder_kras/af_init_guess/inputs/r2/kras_chunks/chunk_0.silent | head -3