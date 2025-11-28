import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import MDAnalysis
from MDAnalysis.analysis import distances

psf_file = '../step3_input.psf'
dcd_dir = '../results_PPO/dcd_trajs/'

with open('../results_PPO/dcd_trajs/runs.txt', 'r') as file:
    for line in file:
      run_name = line.strip()
      print(run_name)
      os.system(f'ls -v {dcd_dir}/{run_name}*.dcd > .temp')
      #coordinating_atoms = {}
      output = open(f'Pi-{run_name}.txt', 'w')
      fr = 0
      with open('.temp', 'r') as file2:
        for line2 in file2:
          dcd_name = line2.strip()
          #print(dcd_name)
          u = MDAnalysis.Universe(psf_file,dcd_name)
          coordinating_atoms_updating_1 = u.select_atoms("not segid HETC and around 2 (segid HETC and (name O2 or name O3))", updating=True)
          coordinating_atoms_updating_2 = u.select_atoms("not segid HETC and around 2 (segid HETC and (name H1 or name H2))", updating=True)
          for ts in u.trajectory:
            #print(coordinating_atoms_updating)
            coordinating_atoms_updating_1
            coordinating_atoms_updating_2
            coordinating_atoms_updating = coordinating_atoms_updating_1 + coordinating_atoms_updating_2
            for ik in range(0,len(coordinating_atoms_updating)):
              output.write(f'{coordinating_atoms_updating[ik].resid}_{coordinating_atoms_updating[ik].resname}_{coordinating_atoms_updating[ik].name}')
              if ik < len(coordinating_atoms_updating)-1:
                output.write(',')
              else:
                output.write('\n')
            print(len(coordinating_atoms_updating))
            fr += 1
      output.close()
