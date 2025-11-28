import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import MDAnalysis
from MDAnalysis.analysis import distances
import glob

psf_file = '../step3_input.psf'
dcd_dir = '../results_PPO/dcd_trajs/'

with open('../results_PPO/dcd_trajs/runs.txt', 'r') as file:
    for line in file:
        run_name = line.strip()
        print(run_name)

        # WINDOWS-SAFE REPLACEMENT FOR: ls -v run_name*.dcd > .temp
        dcd_files = sorted(
            glob.glob(os.path.join(dcd_dir, f"{run_name}*.dcd")),
            key=lambda x: int(os.path.basename(x).split("_s")[-1].split(".")[0])
        )
        with open(".temp", "w") as tempf:
            for f in dcd_files:
                tempf.write(f + "\n")

        coordinating_atoms = {}
        output = open(f'{run_name}.txt', 'w')
        fr = 0

        with open('.temp', 'r') as file2:
            for line2 in file2:
                dcd_name = line2.strip()
                u = MDAnalysis.Universe(psf_file, dcd_name)
                coordinating_atoms_updating = u.select_atoms(
                    "name O* and around 5 resname MG", updating=True
                )
                for ts in u.trajectory:
                    dist_arr = distances.distance_array(
                        u.select_atoms('resname MG'),
                        coordinating_atoms_updating
                    )
                    cutoff = (np.sort(dist_arr)[0][5] +
                              (np.sort(dist_arr)[0][6] - np.sort(dist_arr)[0][5]) / 2)
                    temp_sele = u.select_atoms(
                        f"name O* and around {cutoff} resname MG", updating=True
                    )
                    coordinating_atoms[fr] = temp_sele
                    for ik in range(6):
                        output.write(
                            f"{temp_sele[ik].resid}_{temp_sele[ik].resname}_{temp_sele[ik].name}"
                        )
                        output.write(',' if ik < 5 else '\n')
                    print(len(temp_sele))
                    fr += 1

        output.close()
