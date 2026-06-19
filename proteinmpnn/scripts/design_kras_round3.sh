#!/bin/bash
#SBATCH --job-name=proteinmpnn_r3
#SBATCH --partition=priority
#SBATCH --time=4:00:00
#SBATCH --mem=3G
#SBATCH --ntasks=1
#SBATCH --output=../slurm/kras_r3.out
#SBATCH --error=../slurm/kras_r3.err

# source activate mlfold
PYTHON="~/Desktop/pmhc_binder_kras_cp/envs/proteinmpnn/bin/python"
PROTEINMPNN_SCRIPT_DIR="/n/groups/marks/users/aaron/enzymes/ProteinMPNN"
folder_with_pdbs="~/Desktop/pmhc_binder_kras_cp/rfdiffusion/outputs/kras/partial_r3/clustered"
input_dir="../input/kras_r3"
output_dir="../outputs/kras_r3"

mkdir -p $input_dir
mkdir -p $output_dir

# mode A
python redesign_w_charge_residue_filter.py \
    --pdb_dir   ${folder_with_pdbs} \
    --out_dir   ${output_dir} \
    --mpnn_path ${PYTHON} \
    --mpnn_script ${PROTEINMPNN_SCRIPT_DIR}/protein_mpnn_run.py