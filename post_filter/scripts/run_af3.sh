module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

# step 0: run fastrelax
python fastrelax_designs_af3.py \
    --af3_out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3/ \
    --out_dir     /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/fastrelax_af3/ \
    --n_repeats   3

# step 1: calculate stats
# make design_epitopes.csv
python write_design_epitope.py \
    --fasta /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/binder_pMHC_full.fasta \
    --epitope_resnum 5 \
    --epitope_notes "G12D so 5th residue in the peptide"
    --hla A*11:01 \
    --target_name KRAS \
    --output_csv /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/design_epitopes.csv

python af3_design_stats.py \
    --af3_out_dir     /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3/ \
    --relaxed_pdb_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/fastrelax_af3/relaxed_pdbs/ \
    --out_dir         /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/ \
    --targets_csv     /path/to/design_targets.csv

# step 2: visualize stats
python plot_design_stats.py \
    --stats_tsv    /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3_design_stats.tsv \
    --fastrelax_tsv /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/fastrelax_af3/fastrelax_af3_scores.tsv \
    --epitopes_csv /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/design_epitopes.csv \
    --sequences /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/monomer_fasta/af2_monomer_sequences.csv \
    --plddt_tsv    /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer/stats/monomer_scores_your_designs.tsv \
    --out_dir      /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3_plots/
