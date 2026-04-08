#!/bin/bash
module load alphafold/2.3.2-020cd6d

# for Domainome, a few jobs took more than 3h, but most finish within 2h, 25G is enough for all
export fasta_sequence=$1
export fasta_output_dir=$2
export af2_output_dir=$3
export slurm_dir=$4
export max_template_date=$5 # can put in the date when you run AF2

mkdir -p ${af2_output_dir}
mkdir -p ${slurm_dir}

python3 split_fasta_submit_af2_jobs.py ${fasta_sequence} ${fasta_output_dir} ${af2_output_dir} ${slurm_dir} ${max_template_date}
