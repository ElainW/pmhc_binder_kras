#!/bin/bash
#SBATCH --job-name=proteinmpnn_r2
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
#PYTHON=/n/groups/marks/software/anaconda_o2/envs/proteingym_env/bin/python
PYTHON="/n/groups/marks/projects/marks_lab_and_oatml/ProteinGym2/model_envs/proteinmpnn/bin/python"
PROTEINMPNN_SCRIPT_DIR="/n/groups/marks/users/aaron/enzymes/ProteinMPNN"
folder_with_pdbs="~/Desktop/pmhc_binder_kras/rfdiffusion/outputs/kras/partial_r2/filtered"
input_dir="../input/kras_r2"
output_dir="../outputs/kras_r2"

mkdir -p $input_dir
mkdir -p $output_dir

path_for_parsed_chains=$input_dir"/parsed_pdbs.jsonl"
path_for_assigned_chains=$input_dir"/assigned_pdbs.jsonl"

chains_to_design="A"

${PYTHON} ${PROTEINMPNN_SCRIPT_DIR}/helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

${PYTHON} ${PROTEINMPNN_SCRIPT_DIR}/helper_scripts/assign_fixed_chains.py --input_path=$path_for_parsed_chains --output_path=$path_for_assigned_chains --chain_list "$chains_to_design"

${PYTHON} ${PROTEINMPNN_SCRIPT_DIR}/protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --chain_id_jsonl $path_for_assigned_chains \
        --omit_AAs C \
        --out_folder $output_dir \
        --num_seq_per_target 8 \
        --sampling_temp "0.1" \
        --seed 31 \
        --batch_size 1
