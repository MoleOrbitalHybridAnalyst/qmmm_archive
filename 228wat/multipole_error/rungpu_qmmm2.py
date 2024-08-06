''' use rcut_hcore=15.5 dm as ground-truth here
    input Rcut; output octupole error estimate '''
from sys import argv
from os import system, path, chdir, getcwd, environ
from pyscf import lib, gto
from pyscf.pbc.scf.addons import smearing_
from gpu4pyscf.scf import RHF
from gpu4pyscf.dft import RKS
from gpu4pyscf.qmmm.pbc.itrf import add_mm_charges
from gpu4pyscf.qmmm.pbc.tools import determine_hcore_cutoff, estimate_error

import numpy as np
from constants import *

rcut_hcore = float(argv[1])

lib.num_threads(32)
home_dir = environ['HOME']
max_memory = 40000

charge = 0
spin = 0  # n_alpha - n_beta
qm_indexes = [57, 58, 59, 258, 259, 260, 636, 637, 638, 645, 646, 647, 93, 94, 95, 657, 658, 659, 165, 166, 167]

xc = 'pbe'
basis = 'def2-SVPD'
scf_tol = 1e-12
max_scf_cycles = 100
screen_tol = 1e-14

# Chemistry - A European Journal, (2009), 186-197, 15(1)
ele2radius = {'N': 0.71, 'H': 0.32, 'C': 0.75, 'CL': 0.99, 'O': 0.63, 'CL': 0.99, 'K': 1.96, 'S': 1.03, 'FE': 1.16, 'P': 1.11, 'MG': 1.39, 'NA': 1.55}

mol = gto.Mole()
mol.charge = charge
mol.spin = spin
mol.verbose = 4
mol.max_memory = max_memory
mol.basis = basis
mol.fromfile(f"./water_new.xyz")
coords = mol.atom_coords()[qm_indexes]
box = np.eye(3) * 1.9002402289999999e+01 * A / Bohr

box_A = box * Bohr / A
coords_A = coords * Bohr / A

######################################
########### PySCF QM PART ############
######################################

mm_coords = list()
mm_charges = list()
mm_radii = list()
for i in range(mol.natm):
    if i not in qm_indexes:
        mm_coords.append(mol.atom_coord(i) * Bohr / A)
        if mol.atom_symbol(i) == 'O':
            mm_charges.append(-0.82)
        else:
            mm_charges.append(0.41)
        mm_radii.append(ele2radius[mol.atom_symbol(i).upper()])
mm_coords = np.array(mm_coords, dtype=float)
mm_charges = np.array(mm_charges, dtype=float)
mm_radii = np.array(mm_radii, dtype=float)

mol.atom = [mol.atom[i] for i in qm_indexes]
mol.build()

# make qm atoms whole
ref = coords_A
diff = coords_A - ref
n = (diff + 0.5 * np.diag(box_A)) // np.diag(box_A)
diff = diff - n * np.diag(box_A)
coords_A = diff + ref

# move qm atoms to the center of box 
ref = np.mean(coords_A, axis=0)
diff = coords_A - ref
n = (diff + 0.5 * np.diag(box_A)) // np.diag(box_A)
diff = diff - n * np.diag(box_A)
coords_A = diff 
# move mm atoms accordingly
diff = mm_coords - ref
n = (diff + 0.5 * np.diag(box_A)) // np.diag(box_A)
diff = diff - n * np.diag(box_A)
mm_coords = diff 

mol.set_geom_(coords_A, unit='Angstrom')

if xc is None:
    mf = RHF(mol)
else:
    mf = RKS(mol, xc=xc)
mf.conv_tol = scf_tol
mf.max_cycle = max_scf_cycles
mf.screen_tol = screen_tol

dm = np.load("./dm_qmmm_ewald2_15.5_10.0.npy")

mf = add_mm_charges(mf, mm_coords, box_A, mm_charges, mm_radii, rcut_hcore=15.5, rcut_ewald=10)
mf.kernel(dm0=dm)
# converged SCF energy = -534.229214821956
dm = mf.make_rdm1().get()

print(estimate_error(mol, mm_coords, box_A, mm_charges, rcut_hcore, dm, precision=1e-7))
