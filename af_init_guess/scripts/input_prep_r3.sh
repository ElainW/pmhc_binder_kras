#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

mkdir -p /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/inputs/r3/

python thread_fasta_to_backbones.py \
    --fasta_file /n/groups/marks/users/aaron/pmhc_cp/proteinmpnn/outputs/kras_r3/all_sequences.fa \
    --backbone_dir /n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3/clustered/ \
    --output_dir /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/inputs/r3/threaded_pdbs/ \
    --backbone_suffix_re '_a\d+_t[\d.]+$' \
    --verbose

/n/groups/marks/users/aaron/pmhc/silent_tools/silentfrompdbs /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/inputs/r3/threaded_pdbs/*.pdb > /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/inputs/r3/threaded_kras_designs.silent

python split_silent.py \
    --silent ../inputs/r3/threaded_kras_designs.silent \
    --output_dir ../inputs/r3/kras_chunks/ \
    --chunk_size 200

# Verify chunks have 2 chains
grep "RES_NUM" \
    /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/inputs/r3/kras_chunks/chunk_0.silent | head -3