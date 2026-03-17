# Envs
## dl_binder_design env
created using dl_binder_design.yml + pyrosetta + other libraries related to plotting
1. Works on RFDiffusion, backbone filtering, ProteinMPNN
## af2_init_guess (do not use)
created to handle the issues with incompatible numpy versions between different packages (af2_init_guess.yml) and sym link to pyrosetta and rosetta in dl_binder_design
1. Remember to add `export LD_LIBRARY_PATH=/n/groups/marks/users/aaron/pmhc/envs/af2_init_guess/lib:/n/groups/marks/users/aaron/pmhc/envs/af2_init_guess/lib/python3.11/site-packages/nvidia/cusolver/lib:/n/groups/marks/users/aaron/pmhc/envs/af2_init_guess/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH` to ensure proper loading of cuda device. Verify by `python -c "import jax; print('JAX devices:', jax.devices())"` and should see `cuda(id=0)`; no need to load cuda/12.8 from gcc on O2
2. Works on AF2 initial guess, but not on tensorflow (removed to resolve version conflicts)
## af2_binder_design
modified from the original file from dl_binder_design
1. sym link to pyrosetta and rosetta in dl_binder_design
2. Remember to add `export LD_LIBRARY_PATH=/n/groups/marks/users/aaron/pmhc/envs/af2_init_guess/lib:/n/groups/marks/users/aaron/pmhc/envs/af2_init_guess/lib/python3.11/site-packages/nvidia/cusolver/lib:/n/groups/marks/users/aaron/pmhc/envs/af2_init_guess/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH` to ensure proper loading of cuda device. Verify by `python -c "import jax; print('JAX devices:', jax.devices())"` and should see `cuda(id=0)`; no need to load cuda/12.8 from gcc on O2