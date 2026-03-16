#!/bin/bash
#SBATCH --cpus-per-task 1
#SBATCH -t 0-08:00                      # Runtime in D-HH:MM format
#SBATCH -p gpu_quad,gpu,gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --qos=gpuquad_qos
#SBATCH --mem=10G                          # Memory total in MB (for all cores)
##SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
##SBATCH --mail-user="wangyilan0304@gmail.com"
#SBATCH --output=../slurm/%x-%A_%3a-%u.out  # Nice tip: using %3a to pad to 3 characters (23 -> 023) so that filenames sort properly
#SBATCH --error=../slurm/%x-%A_%3a-%u.err
#SBATCH --job-name="kras"

# Here, we're designing binders to insulin receptor, without specifying the topology of the binder a prior
# We first provide the output path and input pdb of the target protein (insulin receptor)
# We then describe the protein we want with the contig input:
#   - residues 1-150 of the A chain of the target protein
#   - a chainbreak (as we don't want the binder fused to the target!)
#   - A 70-100 residue binder to be diffused (the exact length is sampled each iteration of diffusion)
# We tell diffusion to target three specific residues on the target, specifically residues 59, 83 and 91 of the A chain
# We make 10 designs, and reduce the noise added during inference to 0, to improve the quality of the designs

# ../scripts/run_inference.py inference.output_prefix=example_outputs/design_ppi_robo_slit inference.input_pdb=input_pdbs/2V9T.pdb 'contigmap.contigs=[A60-166/0 70-100]' 'ppi.hotspot_res=[A72,A84,A86]' inference.num_designs=10 denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0

# may be can shorten the length?

/n/groups/marks/users/aaron/RFdiffusion/env/SE3nv/bin/python /n/groups/marks/users/aaron/RFdiffusion/scripts/run_inference.py \
    inference.output_prefix=../outputs/kras/300_4/ \
    inference.input_pdb=../input_pdbs/9UV8.pdb \
    'contigmap.contigs=[A3-180/0 C1-9/0 50-100]' \
    'ppi.hotspot_res=[C6,C7]' \
    inference.num_designs=1000 \
    denoiser.noise_scale_ca=0 \
    denoiser.noise_scale_frame=0

# Here, we're making a binder, and we're specifying the course-grained topology of the binder, from a set of processed inputs
# We specify the path to the target protein, in this case the insulin receptor, along with an output path
# We tell RFdiffusion that we want to do "scaffoldguided" diffusion (i.e. we want to specify the fold of the protein)
# We tell RFdiffusion where on the (cropped) input protein we want to bind, in this case to residues 59, 83 and 91 on the A chain
# We tell RFdiffusion that we're wanting to make a binder to a target, and provide the secondary structure and block adjacency input for these. This may not be necessary
# We then provide a path to a directory of different scaffolds (we've provided some for you to use, from Cao et al., 2022)
# We generate 10 designs, and reduce the noise added during inference to 0 (which improves the quality of designs)

# ../scripts/run_inference.py scaffoldguided.target_path=input_pdbs/2V9T.pdb inference.output_prefix=example_outputs/design_ppi_robo_slit_scaffold scaffoldguided.scaffoldguided=True 'ppi.hotspot_res=[A72,A84,A86]' scaffoldguided.target_pdb=True scaffoldguided.target_ss=target_folds/insulin_target_ss.pt scaffoldguided.target_adj=target_folds/insulin_target_adj.pt scaffoldguided.scaffold_dir=./ppi_scaffolds/ inference.num_designs=10 denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0
