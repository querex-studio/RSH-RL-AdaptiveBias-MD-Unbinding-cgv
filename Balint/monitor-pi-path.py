import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import MDAnalysis
from MDAnalysis.analysis import distances

psf_file = '../step3_input.psf'
dcd_dir = '../results_PPO/dcd_trajs/'

with open('../results_PPO/dcd_trajs/runs.txt', 'r') as file:
    for line in file:
        run_name = line.strip()
        if not run_name:
            continue

        print(run_name)

        # List and sort DCD files for this run (replaces: ls -v run_name*.dcd > .temp)
        dcd_files = glob.glob(os.path.join(dcd_dir, f"{run_name}*.dcd"))

        # Sort like "ls -v": by the _sNNN index if present, otherwise lexicographically
        def _sort_key(path):
            base = os.path.basename(path)
            if "_s" in base:
                try:
                    return int(base.split("_s")[-1].split(".")[0])
                except ValueError:
                    return base
            return base

        dcd_files = sorted(dcd_files, key=_sort_key)

        if not dcd_files:
            print(f"[warning] No DCD files found for run {run_name} in {dcd_dir}")
            continue

        # Output: Pi-<run_name>.txt, like in the original script
        output = open(f'Pi-{run_name}.txt', 'w')
        fr = 0

        for dcd_name in dcd_files:
            dcd_name = dcd_name.strip()
            if not dcd_name:
                continue

            # Load universe for this DCD
            u = MDAnalysis.Universe(psf_file, dcd_name)

            coordinating_atoms_updating_1 = u.select_atoms(
                "not segid HETC and around 2 "
                "(segid HETC and (name O2 or name O3))",
                updating=True
            )
            coordinating_atoms_updating_2 = u.select_atoms(
                "not segid HETC and around 2 "
                "(segid HETC and (name H1 or name H2))",
                updating=True
            )

            for ts in u.trajectory:
                coordinating_atoms_updating = (
                    coordinating_atoms_updating_1
                    + coordinating_atoms_updating_2
                )

                n_atoms = len(coordinating_atoms_updating)
                if n_atoms == 0:
                    output.write("\n")
                    fr += 1
                    continue

                for ik in range(n_atoms):
                    atom = coordinating_atoms_updating[ik]
                    output.write(
                        f"{atom.resid}_{atom.resname}_{atom.name}"
                    )
                    if ik < n_atoms - 1:
                        output.write(',')
                    else:
                        output.write('\n')

                print(n_atoms)
                fr += 1

        output.close()
