module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

# for authors' designs
slurm_dir=/n/groups/marks/users/aaron/pmhc/post_filter/slurm
# step 0: run fastrelax
sbatch -p short -c 1 -t 0-2:00 --mem=2G -o ${slurm_dir}/af3-nomsa-design-%j.out -e ${slurm_dir}/af3-nomsa-design-%j.err -J af3-nomsa-design --wrap="
python fastrelax_designs_af3.py \
    --af3_out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3_nomsa/ \
    --out_dir     /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/fastrelax_af3_nomsa/ \
    --dalphaball /n/groups/marks/users/aaron/pmhc/post_filter/scripts/DAlphaBall.gcc \
    --n_repeats   3"

sbatch -p short -c 1 -t 0-2:00 --mem=2G -o ${slurm_dir}/af3-author-%j.out -e ${slurm_dir}/af3-author-%j.err -J af3-author-design --wrap="
python fastrelax_designs_af3.py \
    --af3_out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3/ \
    --out_dir     /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/fastrelax_af3/ \
    --dalphaball /n/groups/marks/users/aaron/pmhc/post_filter/scripts/DAlphaBall.gcc \
    --n_repeats   3"

python fastrelax_designs_af3.py \
    --af3_out_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3_nomsa_all/ \
    --out_dir     /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/fastrelax_af3_nomsa_all/ \
    --dalphaball /n/groups/marks/users/aaron/pmhc/post_filter/scripts/DAlphaBall.gcc \
    --n_repeats   3

# step 1: calculate stats
python af3_design_stats.py \
    --af3_out_dir     /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3/ \
    --relaxed_pdb_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/fastrelax_af3/relaxed_pdbs/ \
    --out_dir         /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3/ \
    --targets_csv     /n/groups/marks/users/aaron/pmhc/post_filter/inputs/design_epitopes.csv \
    --fastrelax_tsv   /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/fastrelax_af3/fastrelax_af3_scores.tsv

python af3_design_stats.py \
    --af3_out_dir     /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3_nomsa/ \
    --relaxed_pdb_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/fastrelax_af3_nomsa/relaxed_pdbs/ \
    --out_dir         /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3_nomsa/ \
    --targets_csv     /n/groups/marks/users/aaron/pmhc/post_filter/inputs/design_epitopes.csv \
    --fastrelax_tsv   /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/fastrelax_af3_nomsa/fastrelax_af3_scores.tsv

python af3_design_stats_server.py \
    --af3_out_dir     ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/af3_nomsa_all/ \
    --relaxed_pdb_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/fastrelax_af3_nomsa_all/relaxed_pdbs/ \
    --out_dir         ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/af3_nomsa_all/ \
    --targets_csv     ~/Desktop/pmhc_binder_kras/post_filter/inputs/design_epitopes.csv \
    --fastrelax_tsv   ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/fastrelax_af3_nomsa_all/fastrelax_af3_scores.tsv

# step 2: visualize stats
python plot_author_stats.py \
    --stats_tsv    /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3/af3_design_stats.tsv \
    --epitopes_csv /n/groups/marks/users/aaron/pmhc/post_filter/inputs/design_epitopes.csv \
    --xlsx         /n/groups/marks/users/aaron/pmhc/post_filter/inputs/science.adv0185_data_s1.csv \
    --plddt_tsv    /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer/stats/monomer_scores_authors.tsv \
    --out_dir      /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3_plots/

python plot_author_stats.py \
    --stats_tsv    /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3_nomsa/af3_design_stats.tsv \
    --epitopes_csv /n/groups/marks/users/aaron/pmhc/post_filter/inputs/design_epitopes.csv \
    --xlsx         /n/groups/marks/users/aaron/pmhc/post_filter/inputs/science.adv0185_data_s1.csv \
    --plddt_tsv    /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer/stats/monomer_scores_authors.tsv \
    --out_dir      /n/groups/marks/users/aaron/pmhc/post_filter/outputs/author_design_stats/af3_nomsa_plots/
    
python plot_author_stats.py \
    --stats_tsv    ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/af3_nomsa_all/af3_design_stats.tsv \
    --epitopes_csv ~/Desktop/pmhc_binder_kras/post_filter/inputs/design_epitopes.csv \
    --xlsx         ~/Desktop/pmhc_binder_kras/post_filter/inputs/science.adv0185_data_s1.csv \
    --plddt_tsv    ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/af2_monomer/stats/monomer_scores_authors.tsv \
    --out_dir      ~/Desktop/pmhc_binder_kras/post_filter/outputs/author_design_stats/af3_nomsa_all_plots/
    

# step 0: run fastrelax
sbatch -p priority -c 1 -t 0-12:00 --mem=3G -o ${slurm_dir}/af3-r2-%j.out -e ${slurm_dir}/af3-r2-%j.err -J af3-r2-design --wrap="
 python fastrelax_designs_af3.py \
     --af3_out_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/af3/ \
     --out_dir     ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/fastrelax_af3/ \
     --dalphaball ~/Desktop/pmhc_binder_kras/post_filter/scripts/DAlphaBall.gcc \
     --n_repeats   3"

sbatch -p priority -c 1 -t 0-12:00 --mem=3G -o ${slurm_dir}/af3-r2-nomsa-%j.out -e ${slurm_dir}/af3-r2-nomsa-%j.err -J af3-r2-nomsa-design --wrap="
 python fastrelax_designs_af3.py \
     --af3_out_dir ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/af3_nomsa/ \
     --out_dir     ~/Desktop/pmhc_binder_kras/post_filter/outputs/r2/fastrelax_af3_nomsa/ \
     --dalphaball ~/Desktop/pmhc_binder_kras/post_filter/scripts/DAlphaBall.gcc \
     --n_repeats   3"

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
    --out_dir         /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3/ \
    --targets_csv     /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/design_epitopes.csv \
    --fastrelax_tsv   /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/fastrelax_af3/fastrelax_af3_scores.tsv

python af3_design_stats.py \
    --af3_out_dir     /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3_nomsa/ \
    --relaxed_pdb_dir /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/fastrelax_af3_nomsa/relaxed_pdbs/ \
    --out_dir         /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3_nomsa/ \
    --targets_csv     /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/design_epitopes.csv \
    --fastrelax_tsv   /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/fastrelax_af3_nomsa/fastrelax_af3_scores.tsv


# step 2: visualize stats
python plot_design_stats.py \
    --stats_tsv    /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3/af3_design_stats.tsv \
    --epitopes_csv /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/design_epitopes.csv \
    --sequences /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/monomer_fasta/af2_monomer_sequences.csv \
    --plddt_tsv    /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer/stats/monomer_scores_your_designs.tsv \
    --specificity_tsv /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
    --out_dir      /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3_plots/

python plot_design_stats.py \
    --stats_tsv    /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3_nomsa/af3_design_stats.tsv \
    --epitopes_csv /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/design_epitopes.csv \
    --sequences /n/groups/marks/users/aaron/pmhc/post_filter/inputs/r2/monomer_fasta/af2_monomer_sequences.csv \
    --plddt_tsv    /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af2_monomer/stats/monomer_scores_your_designs.tsv \
    --specificity_tsv /n/groups/marks/users/aaron/pmhc/specificity/analysis/r2/merged_scores_ranked.tsv \
    --out_dir      /n/groups/marks/users/aaron/pmhc/post_filter/outputs/r2/af3_nomsa_plots/
