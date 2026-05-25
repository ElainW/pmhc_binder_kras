module load conda/miniforge3/24.11.3-0
conda activate /n/groups/marks/users/aaron/pmhc/envs/dl_binder_design

DIR=~/Desktop/pmhc_binder_kras/post_filter
# for authors' designs
slurm_dir=$DIR/slurm

# step 0: run fastrelax
sbatch -p priority -c 1 -t 0-12:00 --mem=3G -o ${slurm_dir}/af3-r2.5-nomsa-%j.out -e ${slurm_dir}/af3-r2.5-nomsa-%j.err -J af3-r2.5-nomsa-design --wrap="
 python fastrelax_designs_af3_server.py \
     --af3_out_dir $DIR/outputs/r2.5/af3_nomsa/ \
     --out_dir     $DIR/outputs/r2.5/fastrelax_af3_nomsa/ \
     --dalphaball $DIR/scripts/DAlphaBall.gcc \
     --n_repeats   3"

# step 1: calculate stats
# make design_epitopes.csv
# generate the fasta with header consistent with AF3 output
sed 's/t0\.1/t01/g' $DIR/inputs/r2.5/binder_pMHC_full.fasta | sed 's/t0\.2/t02/g' | sed 's/__/_/g' > $DIR/inputs/r2.5/binder_pMHC_full_headerfixed.fasta

python write_design_epitope.py \
    --fasta $DIR/inputs/r2.5/binder_pMHC_full_headerfixed.fasta \
    --epitope_resnum 5 \
    --epitope_notes "G12D so 5th residue in the peptide" \
    --hla "A*11:01" \
    --target_name KRAS \
    --output_csv $DIR/inputs/r2.5/design_epitopes.csv

python af3_design_stats_server.py \
    --af3_out_dir     $DIR/outputs/r2.5/af3_nomsa/ \
    --relaxed_pdb_dir $DIR/outputs/r2.5/fastrelax_af3_nomsa/relaxed_pdbs/ \
    --out_dir         $DIR/outputs/r2.5/af3_nomsa/ \
    --targets_csv     $DIR/inputs/r2.5/design_epitopes.csv \
    --fastrelax_tsv   $DIR/outputs/r2.5/fastrelax_af3_nomsa/fastrelax_af3_scores.tsv


# step 2: visualize stats
python plot_design_stats_noaf2.py \
    --stats_tsv    $DIR/outputs/r2.5/af3_nomsa/af3_design_stats.tsv \
    --epitopes_csv $DIR/inputs/r2.5/design_epitopes.csv \
    --sequences    $DIR/inputs/r2.5/af2_monomer/af2_monomer_sequences.csv \
    --specificity_tsv ~/Desktop/pmhc_binder_kras/specificity/analysis/r2.5/merged_scores_ranked.tsv \
    --out_dir      $DIR/outputs/r2.5/af3_nomsa_plots/
