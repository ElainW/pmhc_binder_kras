# Make executable
chmod +x sbatch_partdiff_tier*.sh

sbatch sbatch_partdiff_tier1.sh   # 7 backbones × 100 = 700 trajectories
sbatch sbatch_partdiff_tier2.sh   # 8 backbones × 50  = 400 trajectories
sbatch sbatch_partdiff_tier3.sh   # 5 backbones × 20  = 100 trajectories