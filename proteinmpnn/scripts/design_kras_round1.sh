#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=../slurm/kras.out

# source activate mlfold
#PYTHON=/n/groups/marks/software/anaconda_o2/envs/proteingym_env/bin/python
PYTHON="/n/groups/marks/projects/marks_lab_and_oatml/ProteinGym2/model_envs/proteinmpnn/bin/python"
PROTEINMPNN_SCRIPT_DIR="/n/groups/marks/users/aaron/enzymes/ProteinMPNN"
folder_with_pdbs="/n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/1000_filtered/orientation_filtered/rep/"
input_dir="../input/kras"
output_dir="../outputs/kras"

mkdir -p $input_dir
mkdir -p $input_dir

path_for_parsed_chains=$input_dir"/parsed_pdbs.jsonl"
path_for_assigned_chains=$input_dir"/assigned_pdbs.jsonl"

chains_to_design="A"

${PYTHON} ${PROTEINMPNN_SCRIPT_DIR}/helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

${PYTHON} ${PROTEINMPNN_SCRIPT_DIR}/helper_scripts/assign_fixed_chains.py --input_path=$path_for_parsed_chains --output_path=$path_for_assigned_chains --chain_list "$chains_to_design"

${PYTHON} ${PROTEINMPNN_SCRIPT_DIR}/protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --chain_id_jsonl $path_for_assigned_chains \
        --out_folder $output_dir \
        --num_seq_per_target 8 \
        --sampling_temp "0.1" \
        --seed 37 \
        --batch_size 1
