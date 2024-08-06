from ase import io 
from sys import argv
from os import system

system(f"rm -rf {argv[2]}")

traj = io.trajectory.Trajectory(argv[1])
for i, atom in enumerate(traj):
    atom.write(argv[2], append=True)
