#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

# inspect a single PDB first to find peptide range
python filter_backbones.py --inspect /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/100/_0.pdb \
                           --peptide_start 179 \
                           --peptide_end 187

# run the filter
python filter_backbones.py \
    --input_dirs /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/100 /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/300_1 /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/300_2 /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/300_3 /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/300_4 \
    --output_dir /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/1000_filtered \
    --peptide_start 179 \
    --peptide_end 187 \
    --rg_min 13.0 \
    --rg_max 23.0 \
    --required_peptide_positions 5