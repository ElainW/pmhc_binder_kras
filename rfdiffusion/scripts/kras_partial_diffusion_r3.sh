# Make executable
chmod +x sbatch_partdiff_r3_tier*.sh

sbatch sbatch_partdiff_r3_tier1.sh   # 8 backbones × 100 = 800 trajectories
sbatch sbatch_partdiff_r3_tier2.sh   # 4 backbones × 50  = 200 trajectories