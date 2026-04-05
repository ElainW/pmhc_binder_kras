#!/bin/bash
cd /n/groups/marks/users/aaron/pmhc/pMHCI_binder_design/software/mpnn/
PYTHON="/n/groups/marks/users/aaron/pmhc/envs/proteinmpnn/bin/python"
folder_with_pdbs="/n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/"
pepscan_csv="/n/groups/marks/users/aaron/pmhc/specificity/ProteinMPNN_ala/inputs/pepscan_kras.csv"
DIR="/n/groups/marks/users/aaron/pmhc/specificity/ProteinMPNN_ala/"

/n/groups/marks/users/aaron/pmhc/envs/dl_binder_design/bin/python ${DIR}/scripts/generate_mpnn_pepscan_csv.py \
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