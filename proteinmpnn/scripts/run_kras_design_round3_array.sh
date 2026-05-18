PYTHON_ENV="/n/groups/marks/users/aaron/pmhc_cp/envs/dl_binder_design/bin/python"
MPNN_ENV="/n/groups/marks/users/aaron/pmhc_cp/envs/proteinmpnn/bin/python"
PROTEINMPNN_SCRIPT_DIR="/n/groups/marks/users/aaron/enzymes/ProteinMPNN"
SCRIPT="/n/groups/marks/users/aaron/pmhc_cp/proteinmpnn/scripts/redesign_w_charge_residue_filter_parallel.py"

PDB_DIR="/n/groups/marks/users/aaron/pmhc_cp/rfdiffusion/outputs/kras/partial_r3/clustered"
OUTPUT_DIR="/n/groups/marks/users/aaron/pmhc_cp/proteinmpnn/outputs/kras_r3"
DESIGN_LIST="${OUTPUT_DIR}/design_list.txt"

# Full workflow — run Steps 1, 3, 4 interactively; only Step 2 via sbatch.

# Step 1 — write list, get chunk count
N=$(python ${SCRIPT} --pdb_dir ${PDB_DIR} --out_dir ${OUTPUT_DIR} \
    --mpnn_path ${MPNN_ENV} --mpnn_script ${MPNN_SCRIPT} \
    --chunk_size 50 --write_design_list | tail -1)
echo "Submitting array 0-$((N-1))"   # expect: 0-6 for 349 PDBs
cat ${DESIGN_LIST}                    # verify paths look correct
# → prints: chunk_size=50  n_chunks=7  array=0-6
#            7

# Step 2 — submit
sbatch --array=0-$((N-1)) run_mpnn_array_r3.sh

# Step 3 -- check progress / resubmit failures:
#
# Count finished designs:
ls ${OUTPUT_DIR}/per_design/*.fa 2>/dev/null | wc -l # should be 349

# Check SLURM job status:
sacct -j <JOBID> --format=JobID,State,ExitCode | grep -v COMPLETED

# e.g. Resubmit specific failed chunks (e.g. chunks 3 and 5):
# sbatch --array=3,5 run_mpnn_array_r3.sh

# Step 4 — merge
python ${SCRIPT} --pdb_dir ${PDB_DIR} --out_dir ${OUTPUT_DIR} \
    --mpnn_path ${MPNN_ENV} --mpnn_script ${MPNN_SCRIPT} --merge
