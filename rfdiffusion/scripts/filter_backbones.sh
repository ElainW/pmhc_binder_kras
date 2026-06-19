#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate ~/Desktop/pmhc_binder_kras/envs/dl_binder_design

cd ~/Desktop/pmhc_binder_kras/rfdiffusion/scripts/

# inspect a single PDB first to find peptide range
python filter_backbones.py --inspect ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/100/_0.pdb \
                           --peptide_start 179 \
                           --peptide_end 187

# run the basic geometric filter
python filter_backbones.py \
    --input_dirs ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/100 ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/300_1 ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/300_2 ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/300_3 ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/300_4 \
    --output_dir ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/1000_filtered \
    --peptide_start 179 \
    --peptide_end 187 \
    --rg_min 13.0 \
    --rg_max 23.0 \
    --required_peptide_positions 5

# run the orientation filter
python filter_backbone_orientation.py \
    --input_dirs ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/1000_filtered \
    --output_dir ~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/1000_filtered/orientation_filtered \
