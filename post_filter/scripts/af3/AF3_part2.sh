#!/bin/bash
#SBATCH --partition=gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH -c 8
#SBATCH --time=0:20:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1,vram:20G

module load alphafold/3.0.1

export name=$1
export output_prediction_dir=$2

run_alphafold.py \
    --output_dir=${output_prediction_dir} \
    --json_path=${output_prediction_dir}/${name}/${name}_data.json \
    --norun_data_pipeline