#!/bin/bash
cd ~/Desktop/pmhc_binder_kras/pMHCI_binder_design/software/mpnn/
PYTHON="~/Desktop/pmhc_binder_kras/envs/proteinmpnn/bin/python"
folder_with_pdbs="~/Desktop/pmhc_binder_kras/af_init_guess/outputs/r2/af2_kras/top_pdbs/"
pepscan_csv="~/Desktop/pmhc_binder_kras/specificity/ProteinMPNN_ala/inputs/pepscan_kras.csv"
DIR="~/Desktop/pmhc_binder_kras/specificity/ProteinMPNN_ala/"

~/Desktop/pmhc_binder_kras/envs/dl_binder_design/bin/python ${DIR}/scripts/generate_mpnn_pepscan_csv.py \
    --pdb_dir ${folder_with_pdbs} \
    --out_csv ${pepscan_csv}

${PYTHON} zero_shot_mutation_effect_prediction.py \
    --csv_file ${pepscan_csv} \
    --folder_with_pdbs ${folder_with_pdbs} \
    --output_dir ${DIR}/outputs/ \
    --use_mt_structure 0 \
    --num_seq_per_target 10 \
    --batch_size 10 \
    --mutant_column mutant \
    --mutant_chain_column mutant_chain