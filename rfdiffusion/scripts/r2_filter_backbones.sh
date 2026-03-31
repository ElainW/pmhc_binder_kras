#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

cd /n/groups/marks/users/aaron/pmhc/rfdiffusion/scripts/

# inspect a single PDB first to find peptide range
# python r2_filter_backbones.py --inspect /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier1/orient_158/orient_158_pT12__0.pdb \
#                            --peptide_len 9



# run the basic geometric filter
python r2_filter_backbones.py \
    --input_dirs \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier1/orient_34 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier1/orient_47 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier1/orient_83 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier1/orient_158 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier1/orient_247 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier1/orient_255 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier1/orient_260 \
    --output_dir /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier1_filtered \
    --peptide_len 9 \
    --clash_dist 4.0 \
    --max_clashes 0 \
    --contact_dist 8.0 \
    --required_peptide_positions 5

python r2_filter_backbones.py \
    --input_dirs \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier2/orient_43 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier2/orient_77 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier2/orient_88 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier2/orient_165 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier2/orient_244 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier2/orient_252 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier2/orient_259 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier2/orient_321 \
    --output_dir /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier2_filtered \
    --peptide_len 9 \
    --clash_dist 4.0 \
    --max_clashes 0 \
    --contact_dist 8.0 \
    --required_peptide_positions 5

python r2_filter_backbones.py \
    --input_dirs \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier3/orient_166 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier3/orient_324 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier3/orient_95 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier3/orient_51 \
        /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier3/orient_348 \
    --output_dir /n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r2/tier3_filtered \
    --peptide_len 9 \
    --clash_dist 4.0 \
    --max_clashes 0 \
    --contact_dist 8.0 \
    --required_peptide_positions 5