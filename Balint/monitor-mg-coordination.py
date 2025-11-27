import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import MDAnalysis
from MDAnalysis.analysis import distances

psf_file = '/mnt/data1/users/tj/WORK/RL_MFPT/1D_PMg/trajectory/explore_traj/step3_input.psf'
dcd_dir = '/mnt/data1/users/tj/WORK/RL_MFPT/1D_PMg/trajectory/explore_traj/'

with open('runs.txt', 'r') as file:
    for line in file:
      run_name = line.strip()
      print(run_name)
      os.system(f'ls -v {dcd_dir}/{run_name}*.dcd > .temp')
      coordinating_atoms = {}
      output = open(f'{run_name}.txt', 'w')
      fr = 0
      with open('.temp', 'r') as file2:
        for line2 in file2:
          dcd_name = line2.strip()
          #print(dcd_name)
          u = MDAnalysis.Universe(psf_file,dcd_name)
          coordinating_atoms_updating = u.select_atoms("name O* and around 5 resname MG", updating=True)
          for ts in u.trajectory:
            #print(coordinating_atoms_updating)
            coordinating_atoms_updating
            dist_arr = distances.distance_array(u.select_atoms('resname MG'), coordinating_atoms_updating)
            cutoff = np.sort(dist_arr)[0][5] + (np.sort(dist_arr)[0][6]-np.sort(dist_arr)[0][5])/2
            temp_sele = u.select_atoms(f"name O* and around {cutoff} resname MG", updating=True)
            coordinating_atoms[fr] = temp_sele
            for ik in range(0,6):
              output.write(f'{temp_sele[ik].resid}_{temp_sele[ik].resname}_{temp_sele[ik].name}')
              if ik < 5:
                output.write(',')
              else:
                output.write('\n')
            print(len(temp_sele))
            fr += 1
      output.close()
      