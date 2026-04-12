#!/bin/bash

#SBATCH --job-name="mage-282_AF3"
#SBATCH --partition=short
#SBATCH -c 10
#SBATCH --time=2:00:00
#SBATCH --mem=5G
#SBATCH -o /n/groups/marks/users/aaron/pmhc/post_filter/slurm/mage-282_AF3-%j.out
#SBATCH -e /n/groups/marks/users/aaron/pmhc/post_filter/slurm/mage-282_AF3-%j.err

module load alphafold/3.0.1

mkdir -p /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3/

run_alphafold.py \
    --output_dir=/n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3/ \
    --json_path=/n/groups/marks/users/aaron/pmhc/post_filter/inputs/author_design_stats/af3/mage-282.json \
    --max_template_date '2026-04-07' \
    --norun_inference
