#!/bin/bash
#SBATCH --job-name="mage-282_AF3"
#SBATCH --partition=gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH -c 8
#SBATCH --time=2:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:l40s:1
#SBATCH -o /n/groups/marks/users/aaron/pmhc/post_filter/slurm/mage-282_AF3-%j.out
#SBATCH -e /n/groups/marks/users/aaron/pmhc/post_filter/slurm/mage-282_AF3-%j.err

module load alphafold/3.0.1

run_alphafold.py \
    --output_dir=/n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3/ \
    --json_path=/n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3/mage-282/mage-282_data.json \
    --norun_data_pipeline
