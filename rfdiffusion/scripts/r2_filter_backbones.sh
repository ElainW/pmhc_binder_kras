#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/dl_binder_design

cd ~/Desktop/pmhc_binder_kras/rfdiffusion/scripts/

# inspect a single PDB first to find peptide range
python r2_filter_backbones.py --inspect ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/partial_r2/tier1/orient_158/orient_158_pT12__0.pdb \
                           --peptide_len 9



# run the basic geometric filter
python r2_filter_backbones.py \
    --input_dirs \
        ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/partial_r2/tier1/ \
    --output_dir ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/partial_r2/tier1_filtered \
    --peptide_len 9 \
    --clash_dist 4.0 \
    --max_clashes 0 \
    --contact_dist 8.0 \
    --required_peptide_positions 5 \
    --log_file ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/partial_r2/tier1_filter_log.csv


python r2_filter_backbones.py \
    --input_dirs \
        ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/partial_r2/tier2/ \
    --output_dir ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/partial_r2/tier2_filtered \
    --peptide_len 9 \
    --clash_dist 4.0 \
    --max_clashes 0 \
    --contact_dist 8.0 \
    --required_peptide_positions 5 \
    --log_file ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/partial_r2/tier2_filter_log.csv

python r2_filter_backbones.py \
    --input_dirs \
        ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/partial_r2/tier3/ \
    --output_dir ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/partial_r2/tier3_filtered \
    --peptide_len 9 \
    --clash_dist 4.0 \
    --max_clashes 0 \
    --contact_dist 8.0 \
    --required_peptide_positions 5 \
    --log_file ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/partial_r2/tier3_filter_log.csv