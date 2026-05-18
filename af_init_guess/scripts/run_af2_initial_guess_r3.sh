mkdir -p /n/groups/marks/users/aaron/pmhc/af_init_guess/logs/
mkdir -p /n/groups/marks/users/aaron/pmhc/af_init_guess/inputs/r3/kras_chunks
mkdir -p /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r3/af2_kras

sbatch --array=0-13 run_af2_initial_guess_array_r3.sh
