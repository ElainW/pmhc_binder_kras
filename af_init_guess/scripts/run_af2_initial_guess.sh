mkdir -p /n/groups/marks/users/aaron/pmhc/af_init_guess/logs/
mkdir -p /n/groups/marks/users/aaron/pmhc/af_init_guess/inputs/kras_chunks
mkdir -p /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/af2_kras

# sbatch --array=0-49 run_af2_initial_guess_array.sh
sbatch --array=0 run_af2_initial_guess_array.sh