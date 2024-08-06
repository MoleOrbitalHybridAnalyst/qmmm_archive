''' run QM/MM-Ewald for an array of MM water positions
    input Rcut; save energy and force results '''
from sys import argv
from os import system, path, chdir, getcwd, environ
from pyscf import lib, gto
from pyscf.pbc.scf.addons import smearing_
from gpu4pyscf.scf import RHF
from gpu4pyscf.dft import RKS
from gpu4pyscf.qmmm.pbc.itrf import add_mm_charges
from gpu4pyscf.qmmm.pbc.tools import determine_hcore_cutoff, estimate_error

import numpy as np
import cupy as cp
from constants import *

rcut_hcore = float(argv[1])

lib.num_threads(32)
home_dir = environ['HOME']
max_memory = 40000

charge = 0
spin = 0  # n_alpha - n_beta
qm_indexes = list(range(3*7))

xc = 'pbe'
basis = 'def2-SVPD'
scf_tol = 1e-12
max_scf_cycles = 100
screen_tol = 1e-14

# Chemistry - A European Journal, (2009), 186-197, 15(1)
ele2radius = {'N': 0.71, 'H': 0.32, 'C': 0.75, 'CL': 0.99, 'O': 0.63, 'CL': 0.99, 'K': 1.96, 'S': 1.03, 'FE': 1.16, 'P': 1.11, 'MG': 1.39, 'NA': 1.55}

mol = gto.Mole(output="rungpu_qmmmewald5_{:.2f}.pyscflog".format(rcut_hcore))
mol.charge = charge
mol.spin = spin
mol.verbose = 3
mol.max_memory = max_memory
mol.basis = basis
mol.fromfile(f"./water_new.xyz")
coords = mol.atom_coords()[qm_indexes]
box = np.eye(3) * 2.7000000000000000e+01 * A / Bohr

# (4/3*3.14*17**3)**(1/3) = 27.39923157, meaning that rcut_hcore=17 will result into # of hcore-mm = # of mm in single-box

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

# move mm coords to box center as well
mm_coords0 = mm_coords - np.mean(mm_coords, axis=0)

qm_center = np.mean(coords_A, axis=0)

mol.set_geom_(coords_A, unit='Angstrom')

# max(qm-qm_center) = 3.8, meaning that min(qm-non-hcore-mm) dist \approx rcut_hcore - 3.8

elst = list()
gqmlst = list()
gmmlst = list()
rlst = list(np.arange(rcut_hcore*0.5, rcut_hcore*1.5, 0.5))
for r in rlst:
    print(f"r = {r}")
    vnorm = np.array([-1,-1,1])
    vnorm = vnorm / np.linalg.norm(vnorm)

    mm_coords = mm_coords0 + vnorm * r

    if xc is None:
        mf = RHF(mol)
    else:
        mf = RKS(mol, xc=xc)
    mf.conv_tol = scf_tol
    mf.max_cycle = max_scf_cycles
    mf.screen_tol = screen_tol
    
    mf = add_mm_charges(mf, mm_coords, box_A, mm_charges, mm_radii, rcut_hcore=rcut_hcore, rcut_ewald=10)
    mf.get_qm_dipoles = lambda *args: np.zeros((mol.natm, 3))
    mf.get_qm_quadrupoles = lambda *args: np.zeros((mol.natm, 3, 3))
    def get_vdiff(self, mol, ewald_pot):
        '''
        vdiff_uv = d Q_I / d dm_uv ewald_pot[0]_I
        '''
        import cupy as cp
        vdiff = np.zeros((mol.nao, mol.nao))
        ovlp = self.get_ovlp()
        aoslices = mol.aoslice_by_atom()
        for iatm in range(mol.natm):
            v0 = ewald_pot[0][iatm].get()
            p0, p1 = aoslices[iatm, 2:]
            vdiff[:,p0:p1] -= v0 * ovlp[:,p0:p1]
        vdiff = (vdiff + vdiff.T) / 2
        return cp.asarray(vdiff)
    mf.__class__.get_vdiff = get_vdiff
    e = mf.kernel()
    g = mf.nuc_grad_method()
    g.max_memory = max_memory
    g.auxbasis_response = True
    g_qm = g.kernel()

    g_mm = g.grad_hcore_mm(mf.make_rdm1()) + g.grad_nuc_mm() + g.de_ewald_mm

    elst.append(e)
    gqmlst.append(g_qm)
    gmmlst.append(g_mm)

fname = "rungpu_qmmmewald5_{:.2f}_results.npz".format(rcut_hcore)
np.savez(fname, e=elst, gqm=gqmlst, gmm=gmmlst, r=rlst)
