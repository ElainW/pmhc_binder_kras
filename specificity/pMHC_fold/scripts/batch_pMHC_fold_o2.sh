# On-target example:
python batch_pMHC_fold_o2.py \
    --prefix pmhc_fold_on \
    --round r2 \
    --script /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/scripts/pmhc_fold.py \
    --alleles "A*11:01" \
    --peptides VVGADGVGK \
    --minib_dir /n/groups/marks/users/aaron/pmhc/af_init_guess/outputs/r2/af2_kras/top_pdbs/ \
    --structs_per_job 20 \
    --t 01:30:00

python batch_pMHC_fold_o2.py \
    --prefix pmhc_fold_on \
    --round r2.5 \
    --script /n/groups/marks/users/aaron/pmhc_cp/specificity/pMHC_fold/scripts/pmhc_fold.py \
    --alleles "A*11:01" \
    --peptides VVGADGVGK \
    --minib_dir /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/outputs/r2.5/af2_kras/top_pdbs/ \
    --structs_per_job 20 \
    --t 01:30:00

python batch_pMHC_fold_o2.py \
    --prefix pmhc_fold_on \
    --round r3 \
    --script /n/groups/marks/users/aaron/pmhc_cp/specificity/pMHC_fold/scripts/pmhc_fold.py \
    --alleles "A*11:01" \
    --peptides VVGADGVGK \
    --minib_dir /n/groups/marks/users/aaron/pmhc_cp/af_init_guess/outputs/r3/af2_kras/top_pdbs/ \
    --structs_per_job 20 \
    --t 00:30:00

# Off-target
python batch_pMHC_fold_o2.py \
    --prefix pmhc_fold_off \
    --round r2 \
    --script /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/scripts/pmhc_fold.py \
    --alleles "A*11:01" \
    --peptides VVGAGGVGK \
    --minib_dir /n/groups/marks/users/aaron/pmhc/specificity/pMHC_fold/inputs/r2/off_target_pdbs/ \
    --structs_per_job 20 \
    --t 01:30:00

python batch_pMHC_fold_o2.py \
    --prefix pmhc_fold_off \
    --round r2.5 \
    --script /n/groups/marks/users/aaron/pmhc_cp/specificity/pMHC_fold/scripts/pmhc_fold.py \
    --alleles "A*11:01" \
    --peptides VVGAGGVGK \
    --minib_dir /n/groups/marks/users/aaron/pmhc_cp/specificity/pMHC_fold/inputs/r2.5/off_target_pdbs/ \
    --structs_per_job 20 \
    --t 01:30:00

python batch_pMHC_fold_o2.py \
    --prefix pmhc_fold_off \
    --round r3 \
    --script /n/groups/marks/users/aaron/pmhc_cp/specificity/pMHC_fold/scripts/pmhc_fold.py \
    --alleles "A*11:01" \
    --peptides VVGAGGVGK \
    --minib_dir /n/groups/marks/users/aaron/pmhc_cp/specificity/pMHC_fold/inputs/r3/off_target_pdbs/ \
    --structs_per_job 20 \
    --t 00:30:00