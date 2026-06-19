mkdir -p ~/Desktop/pmhc_binder_kras/af_init_guess/logs/
mkdir -p ~/Desktop/pmhc_binder_kras/af_init_guess/inputs/kras_chunks
mkdir -p ~/Desktop/pmhc_binder_kras/af_init_guess/outputs/af2_kras

sbatch --array=0-49 run_af2_initial_guess_array.sh
