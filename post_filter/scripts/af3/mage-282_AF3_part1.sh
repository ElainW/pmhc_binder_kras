#!/bin/bash

#SBATCH --job-name="mage-282_AF3"
#SBATCH --partition=short
#SBATCH -c 10
#SBATCH --time=2:00:00
#SBATCH --mem=5G
#SBATCH -o ~/Desktop/pmhc_binder_kras/post_filter/slurm/mage-282_AF3-%j.out
#SBATCH -e ~/Desktop/pmhc_binder_kras/post_filter/slurm/mage-282_AF3-%j.err

module load alphafold/3.0.1

mkdir -p ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/af3/

run_alphafold.py \
    --output_dir=~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/af3/ \
    --json_path=~/Desktop/pmhc_binder_kras/post_filter/inputs/author_design_stats/af3/mage-282.json \
    --max_template_date '2026-04-07' \
    --norun_inference
