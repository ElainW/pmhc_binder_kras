#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/dl_binder_design

python thread_fasta_to_backbones.py \
    --fasta_dir ~/Desktop/pmhc_binder_kras/proteinmpnn/outputs/kras/seqs/ \
    --backbone_dir ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/1000_filtered/orientation_filtered/rep/ \
    --output_dir ~/Desktop/pmhc_binder_kras/af_init_guess/inputs/threaded_pdbs/ \
    --verify \
    --verbose

python merge_chains_bc.py \
    --input_dir ../inputs/threaded_pdbs/ \
    --output_dir ../inputs/threaded_pdbs_merged/

~/Desktop/pmhc_binder_kras/silent_tools/silentfrompdbs ~/Desktop/pmhc_binder_kras/af_init_guess/inputs/threaded_pdbs_merged/*.pdb > ~/Desktop/pmhc_binder_kras/af_init_guess/inputs/threaded_kras_designs.silent

python split_silent.py \
    --silent ../inputs/threaded_kras_designs.silent \
    --output_dir ../inputs/kras_chunks/ \
    --chunk_size 50

# Verify chunks have 2 chains
grep "RES_NUM" \
    ~/Desktop/pmhc_binder_kras/af_init_guess/inputs/kras_chunks/chunk_0.silent | head -3