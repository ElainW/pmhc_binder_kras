mkdir -p ~/Desktop/pmhc_binder_kras/af_init_guess/logs/
mkdir -p ~/Desktop/pmhc_binder_kras/af_init_guess/inputs/r2/kras_chunks
mkdir -p ~/Desktop/pmhc_binder_kras/af_init_guess/outputs/r2/af2_kras

sbatch --array=0-47 run_af2_initial_guess_array_r2.sh
