import os
import glob
from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.core.pose import chain_end_res, Pose
import pandas as pd
import numpy as np

# Initialize PyRosetta (silent mode to reduce output)
init("-mute all")


def get_chain_pose(pose, chain="A"):
    """Extract a single chain from a pose."""
    chain_id = ord(chain) - ord('A') + 1  # convert letter to PyRosetta chain index
    return pose.split_by_chain(chain_id)


def calc_rg_binder_only(pose, binder_chain="A"):
    binder_pose = get_chain_pose(pose, binder_chain)

    coords = []
    for i in range(1, binder_pose.total_residue() + 1):
        residue = binder_pose.residue(i)
        if residue.has("CA"):
            ca = residue.xyz("CA")
            coords.append([ca.x, ca.y, ca.z])

    coords = np.array(coords)
    centroid = coords.mean(axis=0)
    rg = np.sqrt(((coords - centroid) ** 2).sum(axis=1).mean())
    return rg

def filter_backbones_by_rg(
    input_dirs,
    output_dir,
    rg_min=None,
    rg_max=None,
    pattern="*.pdb"
):
    """
    Filter PDB backbones by radius of gyration.

    Args:
        input_dir:  Directory containing RFdiffusion output PDBs
        output_dir: Directory to copy passing structures
        rg_min:     Minimum Rg in Angstroms (None = no lower bound)
        rg_max:     Maximum Rg in Angstroms (None = no upper bound)
        pattern:    Glob pattern for PDB files
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    passed = 0
    for input_dir in input_dirs:
        pdb_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
        print(f"Found {len(pdb_files)} structures to filter in {input_dir}")

        for pdb_path in pdb_files:
            try:
                pose = pose_from_pdb(pdb_path)
                rg = calc_rg_binder_only(pose)

                passes = True
                if rg_min is not None and rg < rg_min:
                    passes = False
                if rg_max is not None and rg > rg_max:
                    passes = False

                results.append((pdb_path, rg, passes))

                if passes:
                    passed += 1
                    out_path = os.path.join(output_dir, os.path.basename(pdb_path))
                    pose.dump_pdb(out_path)

            except Exception as e:
                print(f"  Warning: failed to process {pdb_path}: {e}")

    # Summary
    print(f"\nResults: {passed}/{len(results)} structures passed filter")
    print(f"  Rg range in dataset: "
          f"{min(r for _,r,_ in results):.2f} – "
          f"{max(r for _,r,_ in results):.2f} Å")

    # Write a CSV log for all structures
    log_path = os.path.join(output_dir, "rg_filter_log.csv")
    with open(log_path, "w") as f:
        f.write("filename,rg_angstrom,passed\n")
        for path, rg, passes in results:
            f.write(f"{os.path.basename(path)},{rg:.3f},{passes}\n")
    print(f"Log written to {log_path}")

    return results


if __name__ == "__main__":
    input_dirs = ['/n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/100',
                    '/n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/300_1',
                    '/n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/300_2',
                    '/n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/300_3',
                    '/n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/300_4'
                   ]
    filter_backbones_by_rg(
        input_dirs=input_dirs,
        output_dir="/n/groups/marks/users/aaron/pmhc/rfdiffusion/outputs/kras/RG_filtered/",
        rg_min=13.0,   # adjust to your needs
        rg_max=23.0,   # adjust to your needs
    )
