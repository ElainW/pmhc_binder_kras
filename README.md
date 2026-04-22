# pmhc_binder_kras
Designing a de novo minibinder to KRAS G12D derived peptide and MHC-I complex

- Multiple rounds of in silico design, filter, evaluate. 100k sequences, 49 (needs updates) high confidence candidates

The following is cloned from github:
- alphafold
- dl_binder_design (AF2 initial guess now outputs the PAE file as .npy to calculate ipSAE)
- pMHCI_binder_design
- silent_tools

## Main innovations:
- Geometric filter of the backbones (ROG, secondary structures: only >=2 alpha helices, angle between the principle axis of the mini binder and the pMHC complex, CoM, proximity to the neoantigen peptide residue)
    - deduplicate similar backbones, fraction of peptide solvent accessible area buried is super high > 0.8
- ProteinMPNN:
    - remove cysteines
    - only get negatively charged sequences
    - control for hydrophobicity (at least to be the same as authors’ designs: 30-50% hydrophobic residues)
    - **TO-DO**:
        - check spatial aggregation propensity
        - increase or preserve the number of hydrogen bonds, if possible
        - redesigning without affecting the interface (inspired by BindCraft)
- After AF2 initial guess, filter the set with ipSAE between the mini binder and the peptide
- Inspired by BindCraft, add a structure relaxation and energy minization step
    - Calculate interface energy, buried surface area, hydrogen bonds, packing (shape complementarity), per-residue energy
- **TO DO** ProteinMPNN alanine scan —> ESM-IF and VenusREM, full mutation scanning —> verify if the minibinders are interacting with the target in expected ways
- **TO DO** Experiment percentage of amino acids in loops, given that the current designs have very low percentage of those
- Use AF3 as a final step to validate the binding interaction. Notice in first round, discrepancy between AF3 and AF2 initial guess produced interface
    - If there are discrepancies, perhaps use docking (haddock or proclust) —> diff-dock/boltz 2 may be better for high-throughput screens, Rosetta is better for polar interactions
- Generate authors' designs structures and see where my designs fit in the distribution (**TO DO**: will be getting author's full sequence designs and predicted structures soon). See more details in post_filter/inputs/author_design_stats/README.md
- **TO DO** Predict sites for ADA binding (hard to degrade, not T cell)
