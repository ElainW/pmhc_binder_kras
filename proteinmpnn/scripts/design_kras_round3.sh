#!/bin/bash
#SBATCH --job-name=proteinmpnn_r3
#SBATCH --partition=gpu_quad,gpu
#SBATCH --qos=gpuquad_qos
#SBATCH --time=4:00:00
#SBATCH --mem=3G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --mail-user=wangyilan0304@gmail.com
#SBATCH --output=../slurm/kras_r2.out
#SBATCH --error=../slurm/kras_r2.err

# source activate mlfold
PYTHON="/n/groups/marks/users/aaron/pmhc_cp/envs/proteinmpnn/bin/python"
PROTEINMPNN_SCRIPT_DIR="/n/groups/marks/users/aaron/enzymes/ProteinMPNN"
folder_with_pdbs="/n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/partial_r3/clustered"
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