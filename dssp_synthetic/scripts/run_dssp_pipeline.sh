#!/bin/bash
#SBATCH --job-name=dssp_synthetic
#SBATCH --output=logs/dssp_%j.out
#SBATCH --error=logs/dssp_%j.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wangyilan0304@gmail.com

# ============================================================
# DSSP pipeline for 102 synthetic proteins — Hýsková et al.
# Runs: fetch PDB IDs → download PDBs → PyRosetta DSSP → plot
# ============================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────
PIPELINE_DIR="/n/groups/marks/users/aaron/pmhc/dssp_synthetic"
PDB_DIR="${PIPELINE_DIR}/pdbs"
LOGS_DIR="${PIPELINE_DIR}/logs"
SCRIPTS_DIR="${PIPELINE_DIR}/scripts"

CONDA_ENV="/n/groups/marks/users/aaron/pmhc/envs/dl_binder_design"

MANIFEST="${PIPELINE_DIR}/synthetic_manifest.tsv"
OUT_PER_RESIDUE="${PIPELINE_DIR}/dssp_per_residue.tsv"
OUT_SUMMARY="${PIPELINE_DIR}/dssp_summary.tsv"
OUT_PLOT="${PIPELINE_DIR}/dssp_summary_plot.pdf"

# ── Setup ─────────────────────────────────────────────────────
mkdir -p "${PDB_DIR}" "${LOGS_DIR}"
module load conda/miniforge3/24.11.3-0
source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"

echo "=========================================="
echo "DSSP pipeline started: $(date)"
echo "Working dir: ${PIPELINE_DIR}"
echo "=========================================="

# ── Step 1: Fetch PDB IDs and download experimental structures ──
echo ""
echo "Step 1: Fetching synthetic protein PDB IDs from RCSB and downloading PDBs..."
python "${SCRIPTS_DIR}/fetch_synthetic_ids.py" \
    --outdir "${PDB_DIR}" \
    --max 500 \
    --manifest "${MANIFEST}"

N_PDB=$(wc -l < "${MANIFEST}")
echo "  Downloaded $(( N_PDB - 1 )) PDB files"

# ── Step 2: Run PyRosetta DSSP ────────────────────────────────
echo ""
echo "Step 2: Running PyRosetta DSSP analysis..."
python "${SCRIPTS_DIR}/dssp_analysis.py" \
    --pdb_dir "${PDB_DIR}" \
    --manifest "${MANIFEST}" \
    --out_per_residue "${OUT_PER_RESIDUE}" \
    --out_summary "${OUT_SUMMARY}"

N_DONE=$(wc -l < "${OUT_SUMMARY}")
echo "  Completed DSSP for $(( N_DONE - 1 )) structures"

# ── Step 3: Generate plots ─────────────────────────────────────
echo ""
echo "Step 3: Generating visualization..."
python "${SCRIPTS_DIR}/plot_dssp.py" \
    --summary "${OUT_SUMMARY}" \
    --per_residue "${OUT_PER_RESIDUE}" \
    --outfile "${OUT_PLOT}"

echo ""
echo "=========================================="
echo "Pipeline complete: $(date)"
echo "Outputs:"
echo "  Manifest:      ${MANIFEST}"
echo "  Per-residue:   ${OUT_PER_RESIDUE}"
echo "  Summary table: ${OUT_SUMMARY}"
echo "  Main plot:     ${OUT_PLOT}"
echo "  Heatmap:       ${OUT_PLOT/.pdf/_heatmap.pdf}"
echo "=========================================="
