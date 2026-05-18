#!/bin/bash
module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

cd /n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/scripts/

# inspect a single PDB first to find peptide range
python r2_filter_backbones.py --inspect /n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3/tier1/orient_252_pT20__31/orient_252_pT20__31_pT12__0.pdb \
                           --peptide_len 9



# run the basic geometric filter
python r2_filter_backbones.py \
    --input_dirs \
        /n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3/tier1/ \
    --output_dir /n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3/tier1_filtered_2 \
    --peptide_len 9 \
    --clash_dist 4.0 \
    --max_clashes 0 \
    --contact_dist 8.0 \
    --required_peptide_positions 4 5 6 7 \
    --log_file /n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3/tier1_filter_log_4567.csv

python r2_filter_backbones.py \
    --input_dirs \
        /n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3/tier2/ \
    --output_dir /n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3/tier2_filtered_2 \
    --peptide_len 9 \
    --clash_dist 4.0 \
    --max_clashes 0 \
    --contact_dist 8.0 \
    --required_peptide_positions 4 5 6 7 \
    --log_file /n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3/tier2_filter_log_4567.csv

python partial_diffusion_backbone_iRMSD.py \
    --input_dirs /n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3/tier1_filtered /n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3/tier2_filtered \
    --output_dir /n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3/clustered \
    --irmsd_cutoff 0.5 \
    --contact_cutoff 8.0 \
    --visualize_orientations \
    --superpose_orientations
