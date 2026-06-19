#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/dl_binder_design

mkdir -p ~/Desktop/pmhc_binder_kras_cp/af_init_guess/inputs/r2.5/

python thread_fasta_to_backbones.py \
    --fasta_file ~/Desktop/pmhc_binder_kras_cp/proteinmpnn/outputs/r2.5/all_sequences.fa \
    --backbone_dir ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/partial_r2/filtered/ \
    --output_dir ~/Desktop/pmhc_binder_kras_cp/af_init_guess/inputs/r2.5/threaded_pdbs/ \
    --verify \
    --verbose

~/Desktop/pmhc_binder_kras/silent_tools/silentfrompdbs ~/Desktop/pmhc_binder_kras_cp/af_init_guess/inputs/r2.5/threaded_pdbs/*.pdb > ~/Desktop/pmhc_binder_kras_cp/af_init_guess/inputs/r2.5/threaded_kras_designs.silent
