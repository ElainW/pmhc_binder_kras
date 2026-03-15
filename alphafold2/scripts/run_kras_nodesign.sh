#!/bin/bash 

sbatch -o ./slurm/9UV8-%j.out -e ./slurm/9UV8-%j.err -J 9UV8_AF2 get_AF2_structures.sh ../input/kras/fastas/9UV8.fasta '2000-04-10' ../outputs/9UV8/