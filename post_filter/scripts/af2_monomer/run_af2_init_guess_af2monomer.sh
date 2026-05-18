module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/af2_binder_design

# Step 1 — run once on login node
python prepare_monomer_stubs.py \
    --fasta   /n/groups/marks/users/aaron/pmhc_cp/post_filter/inputs/r2.5/af2_monomer/af2_monomer_inputs.fasta \
    --out_dir /n/groups/marks/users/aaron/pmhc_cp/post_filter/inputs/r2.5/af2_monomer/stubs/
# prints: "Submit SLURM array with: sbatch --array=0-N ..."

# Step 2 — submit
sbatch --array=0-32 run_af2_monomer_r2.5_array.sh