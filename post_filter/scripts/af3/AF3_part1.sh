#!/bin/bash

#SBATCH --partition=short
#SBATCH -c 10
#SBATCH --time=4:00:00
#SBATCH --mem=5G

module load alphafold/3.0.1

export name=$1
export output_prediction_dir=$2
export output_json_dir=$3
export max_template_date=$4

run_alphafold.py \
    --output_dir=${output_prediction_dir} \
    --json_path=${output_json_dir}/${name}.json \
    --max_template_date ${max_template_date} \
    --norun_inference