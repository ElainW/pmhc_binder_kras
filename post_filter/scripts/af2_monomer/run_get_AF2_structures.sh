#!/bin/bash
module load alphafold/2.3.1 gcc/9.2.0 cuda/11.2 hh-suite/3.3.0 python/3.10.11

# for Domainome, a few jobs took more than 3h, but most finish within 2h, 25G is enough for all
DIR=Beltran_Lehner_2025 # can use the DMS ID
export fasta_sequence=/n/groups/marks/projects/marks_lab_and_oatml/ProteinGym2/ColabFold/Beltran_Lehner_2025/beltran_domains_filtered.fa
export output_dir=$DIR/output
export max_template_date='2025-04-10' # can put in the date when you run AF2
mkdir -p $DIR/slurm
mkdir -p $DIR/fasta

# sbatch -o $DIR/slurm/AF2-%j.out -e $DIR/slurm/AF2-%j.err --job-name="domainome-AF2" get_AF2_structures.sh ${fasta_sequence} ${max_template_date} ${output_dir}
python3 split_fasta_submit_af2_jobs.py ${fasta_sequence} ${output_dir} ${max_template_date}
