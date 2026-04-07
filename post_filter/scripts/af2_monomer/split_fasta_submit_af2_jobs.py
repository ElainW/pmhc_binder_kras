#!/bin/python3
import sys
import os.path
from cmd_runner import *

# split fasta with multiple entries into single entries
# print(sys.argv[1], sys.argv[2], sys.argv[3])
with open(sys.argv[1], 'r') as f_in:
	overall_dir = sys.argv[2]
	max_template_date = sys.argv[3]
	output_dir = overall_dir + '/fasta'
	i = 0
	output_file = ''
	seq_id = ''
	for lines in f_in:
		if i % 2 == 0:
			seq_id = lines.lstrip('>').rstrip()
			output_file = output_dir + "/" + seq_id + ".fa"
			with open(output_file, 'w') as f_out:
				f_out.write(lines)
			i += 1
		else:
			with open(output_file, 'a') as f_out:
				f_out.write(lines)
				pdb_f = overall_dir + "/" + seq_id + '/' + 'ranked_4.pdb'
				# print(pdb_f)
				if not os.path.isfile(pdb_f):
					cmd = f"sbatch -o {overall_dir}/slurm/{seq_id}-%j.out -e {overall_dir}/slurm/{seq_id}-%j.err -J {seq_id}_AF2 get_AF2_structures.sh {output_file} {max_template_date} {overall_dir}"
					run_cmd_small_output(cmd)
			i += 1
