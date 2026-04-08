#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=marks
#SBATCH -p gpu_quad,gpu
#SBATCH --qos=gpuquad_qos
#SBATCH -t 4:00:00 # usually 32 hours
#SBATCH --mem=25GB # usually 50GB
#SBATCH --exclude=compute-g-17-166,compute-g-17-167,compute-g-17-168,compute-g-17-169,compute-g-17-170,compute-g-17-171,compute-g-17-200,compute-g-17-201,compute-g-17-202,compute-g-17-203,compute-g-17-204,compute-g-17-205

# troubleshoot https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1995177985/Using+AlphaFold+2+on+O2
# find the l40s gpus in gpu_quad, sinfo --Format=nodehost,available,memory,statelong,gres:40 -p gpu_quad | grep "l40s"
module load alphafold/2.3.2-020cd6d

export fasta_sequence=$1
export max_template_date=$2
export output_dir=$3

alphafold.py \
--fasta_paths=${fasta_sequence} \
--max_template_date=${max_template_date} \
--db_preset=full_dbs \
--model_preset='monomer' \
--output_dir=${output_dir} \
--data_dir=/n/shared_db/alphafold-2.3/
