#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/dl_binder_design

mkdir -p ~/Desktop/pmhc_binder_kras_cp/af_init_guess/inputs/r3/

python thread_fasta_to_backbones.py \
    --fasta_file ~/Desktop/pmhc_binder_kras_cp/proteinmpnn/outputs/kras_r3/all_sequences.fa \
    --backbone_dir ~/Desktop/pmhc_binder_kras_cp/rfdiffusion/outputs/kras/partial_r3/clustered/ \
    --output_dir ~/Desktop/pmhc_binder_kras_cp/af_init_guess/inputs/r3/threaded_pdbs/ \
    --backbone_suffix_re '_a\d+_t[\d.]+$' \
    --verbose

~/Desktop/pmhc_binder_kras/silent_tools/silentfrompdbs ~/Desktop/pmhc_binder_kras_cp/af_init_guess/inputs/r3/threaded_pdbs/*.pdb > ~/Desktop/pmhc_binder_kras_cp/af_init_guess/inputs/r3/threaded_kras_designs.silent

python split_silent.py \
    --silent ../inputs/r3/threaded_kras_designs.silent \
    --output_dir ../inputs/r3/kras_chunks/ \
    --chunk_size 200

# Verify chunks have 2 chains
grep "RES_NUM" \
    ~/Desktop/pmhc_binder_kras_cp/af_init_guess/inputs/r3/kras_chunks/chunk_0.silent | head -3