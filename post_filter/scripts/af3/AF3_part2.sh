#!/bin/bash
#SBATCH --partition=gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH -c 8
#SBATCH --time=2:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:a100:1

module load alphafold/3.0.1

export name=$1
export output_predction_dir=$2

run_alphafold.py \
    --output_dir=${output_predction_dir} \
    --json_path=${output_predction_dir}/${name}_data.json \
    --norun_data_pipeline