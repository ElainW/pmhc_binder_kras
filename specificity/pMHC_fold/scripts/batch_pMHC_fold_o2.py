"""
activate the conda env: dl_binder_design

batch_pMHC_fold_o2.py

Submits pMHC fold jobs to O2 SLURM cluster.

On-target example:
    python batch_pMHC_fold_o2.py \
        --prefix pmhc_fold_on \
        --round r2 \
        --script /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/scripts/pmhc_fold.py \
        --alleles "A*11:01" \
        --peptides VVGADGVGK \
        --minib_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
        --structs_per_job 20 \
        --t 01:30:00

Off-target example:
    python batch_pMHC_fold_o2.py \
        --prefix pmhc_fold_off \
        --round r2 \
        --script /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/scripts/pmhc_fold.py \
        --alleles "A*11:01" \
        --peptides VVGAGGVGK \
        --minib_dir /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/inputs/r2/off_target_pdbs/ \
        --structs_per_job 20 \
        --t 01:30:00
"""

import sys, subprocess, os
from argparse import ArgumentParser
import glob
import re
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

# ── Helpers ──────────────────────────────────────────────────────────────────

def my_rstrip(string, strip):
    if string.endswith(strip):
        return string[:-len(strip)]
    return string

def cmd(command, wait=True):
    the_command = subprocess.Popen(
        command, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    if not wait:
        return
    the_stuff = the_command.communicate()
    return str(the_stuff[0]) + str(the_stuff[1])

def ceildiv(a, b):
    return -(-a // b)

def extract_protein_sequences(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    sequences = {}
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            sequence = ''
            for residue in chain:
                if residue.get_id()[0] == ' ':
                    sequence += seq1(residue.get_resname())
            sequences[chain_id] = sequence
    return sequences

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_WEIGHTS = (
    "/n/groups/marks/users/aaron/pmhc_cp/pMHCI_binder_design/pMHC_fold/"
    "datasets_alphafold_finetune/params/"
    "mixed_mhc_pae_run6_af_mhc_params_20640.pkl"
)

ALIGNMENT_CSV = (
    "/n/groups/marks/users/aaron/pmhc/pMHCI_binder_design/pMHC_fold/"
    "alignments_all_alleles_vs_pdb_June29_2024.csv"
)

CONDA_SETUP = (
    "module load conda/miniforge3/24.11.3-0 && "
    "source $(conda info --base)/etc/profile.d/conda.sh && "
    "conda activate /n/groups/marks/users/aaron/pmhc/envs/af2_binder_design"
)

LD_LIBRARY = (
    "/n/groups/marks/users/aaron/pmhc/envs/af2_binder_design/lib/python3.11/"
    "site-packages/nvidia/cudnn/lib:"
    "/n/groups/marks/users/aaron/pmhc/envs/af2_binder_design/lib:"
    "/n/groups/marks/users/aaron/pmhc/envs/af2_binder_design/lib/python3.11/"
    "site-packages/nvidia/cusolver/lib"
)

PYTHON = "/n/groups/marks/users/aaron/pmhc/envs/af2_binder_design/bin/python"

MHC_DIR = "/n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/inputs/pmhc_pdbs"

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument("--prefix",          type=str, required=True)
parser.add_argument("--round",           type=str, required=True)
parser.add_argument("--script",          type=str, required=True)
parser.add_argument("--alleles",         nargs='+', required=True)
parser.add_argument("--peptides",        nargs='+', required=True)
parser.add_argument("--minib_dir",       type=str, required=True)
parser.add_argument("--structs_per_job", type=int, default=10)
parser.add_argument("--cpus",            type=int, default=2)
parser.add_argument("--mem",             type=int, default=16)
parser.add_argument("--t",               type=str, default="01:30:00")
parser.add_argument("--jobs_per_group",  type=int, default=1)
parser.add_argument("--args",            type=str, default="")

args = parser.parse_args()

# Validate time format
if not re.compile(r'^\d{2}:\d{2}:\d{2}$').fullmatch(args.t):
    sys.exit(f"--t must be HH:MM:SS, got {args.t}")

prefix          = args.prefix
script_filename = os.path.abspath(args.script)
minib_dir       = os.path.abspath(args.minib_dir)
structs_per_job = args.structs_per_job
group_size      = args.jobs_per_group
total_path      = os.getcwd()

# ── Directory setup ───────────────────────────────────────────────────────────

runs_path     = os.path.join(os.path.dirname(total_path), "outputs", args.round, f"{prefix}_runs")
commands_path = os.path.join(total_path, f"{prefix}_commands")
slurm_path    = os.path.join(os.path.dirname(total_path), "slurm", args.round, prefix)
splits_path   = os.path.join(os.path.dirname(total_path), "inputs", args.round, f"{prefix}_splits")

os.makedirs(runs_path,   exist_ok=True)
os.makedirs(commands_path, exist_ok=True)
os.makedirs(splits_path, exist_ok=True)
os.makedirs(slurm_path, exist_ok=True)

# ── Build sample dataframe ────────────────────────────────────────────────────

alignment_df = pd.read_csv(ALIGNMENT_CSV)

# Read binder sequences from chain A of each PDB in minib_dir
binder_data = {'binder': [], 'description': []}
for f in sorted(os.listdir(minib_dir)):
    if f.endswith('.pdb'):
        seqs = extract_protein_sequences(os.path.join(minib_dir, f))
        if 'A' not in seqs:
            print(f"  WARNING: no chain A in {f}, skipping")
            continue
        binder_data['binder'].append(seqs['A'])
        binder_data['description'].append(f[:-4])

seqs_to_fold = pd.DataFrame(binder_data)
print(f"Found {len(seqs_to_fold)} binder PDBs in {minib_dir}")

data = {
    'entry_index': [], 'peptide_seq': [], 'allele_sequence': [],
    'target_chainseq': [], 'targetid': [], 'allele_msa_file_names': [],
    'description': [], 'pdb_name': []
}

for allele, peptide in zip(args.alleles, args.peptides):
    allele_seq = alignment_df.loc[
        alignment_df['allele_name_query'] == allele, 'mhc_seq_query'
    ].iloc[0]

    for i, row in seqs_to_fold.iterrows():
        data['entry_index'].append(i)
        data['peptide_seq'].append(peptide)
        data['allele_sequence'].append(allele_seq)
        data['target_chainseq'].append(f"{allele_seq}/{row['binder']}/{peptide}")
        data['targetid'].append(f"{allele}_{peptide}")
        data['allele_msa_file_names'].append(allele)
        desc = f"{row['description']}_{allele}_{peptide}".replace("*","").replace(":","")
        data['description'].append(desc)
        data['pdb_name'].append(row['description'])

df_sample = pd.DataFrame(data)
df_sample.to_csv(os.path.join(splits_path, 'sample_df.csv'), index=False)
print(f"Total rows: {len(df_sample)}")

# ── Split into chunks ─────────────────────────────────────────────────────────

chunks = [
    df_sample[i:i+structs_per_job]
    for i in range(0, len(df_sample), structs_per_job)
]
for i, chunk in enumerate(chunks):
    chunk.to_csv(os.path.join(splits_path, f'sample_df_chunk_{i}.csv'), index=False)
print(f"Split into {len(chunks)} chunks of {structs_per_job}")

# ── Build SLURM commands ──────────────────────────────────────────────────────

commands = []
for chunk_file in sorted(glob.glob(os.path.join(splits_path, "*chunk*.csv"))):
    dirname = my_rstrip(os.path.basename(chunk_file), ".csv")
    job_out_dir = os.path.join(runs_path, dirname)

    run_cmd = (
        f"mkdir -p {job_out_dir} && "
        f"{CONDA_SETUP} && "
        f"export LD_LIBRARY_PATH={LD_LIBRARY}:$LD_LIBRARY_PATH && "
        f"{PYTHON} {script_filename} "
        f"--out_dir {job_out_dir}/ "
        f"--minib_dir {minib_dir}/ "
        f"--model_weights_path {MODEL_WEIGHTS} "
        f"--model_name model_2_ptm "
        f"--seq_mode single_seq "
        f"--df {chunk_file} "
        f"--mhc_dir {MHC_DIR}"
        f"{args.args} "
        f"> {job_out_dir}/log.log 2>&1"
    )
    commands.append(run_cmd)

# Write commands list
commands_list = os.path.join(commands_path, "commands.list")
with open(commands_list, "w") as f:
    f.write("\n".join(commands) + "\n")

# ── Write SLURM array submit script ──────────────────────────────────────────

num_jobs = ceildiv(len(commands), group_size)

# Each line in commands.list becomes one array task
submit_script = os.path.join(commands_path, f"{prefix}_array.submit")
with open(submit_script, 'w') as f:
    f.write(f"""#!/bin/bash
#SBATCH -p gpu_quad,gpu
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:l40s:1
#SBATCH -t {args.t}
#SBATCH --mem={args.mem}G
#SBATCH -c {args.cpus}
#SBATCH -o {slurm_path}/slurm_%A_%a.out
#SBATCH -e {slurm_path}/slurm_%A_%a.err
#SBATCH -a 0-{num_jobs - 1}

# Read the command for this array task
CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {commands_list})
eval "$CMD"
""")

run_submit = os.path.join(total_path, f"{prefix}_run_submit.sh")
with open(run_submit, 'w') as f:
    f.write(f"#!/bin/bash\n")
    f.write(f"sbatch {submit_script}\n")
cmd(f"chmod +x {run_submit}")

# ── Test command ──────────────────────────────────────────────────────────────

test_script = os.path.join(total_path, f"{prefix}_test_command.sh")
with open(test_script, 'w') as f:
    f.write(f"#!/bin/bash\n")
    f.write(commands[0] + "\n")
cmd(f"chmod +x {test_script}")

print("\n" + "/"*80)
print("Test your setup first:")
print(f"bash  {test_script}")
print("\nWhen ready, submit all jobs:")
print(f"bash  {run_submit}")
print("/"*80 + "\n")
