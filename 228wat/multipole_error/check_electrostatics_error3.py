''' check error using fixed dm
    input Rcut; output the exact error of multipole '''
from sys import argv
from os import system, path, chdir, getcwd, environ
from pyscf import lib, gto, df
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf.addons import smearing_
from pyscf.pbc.tools import fft, ifft, get_coulG
from gpu4pyscf.scf import RHF
from gpu4pyscf.dft import RKS
from gpu4pyscf.qmmm.pbc.itrf import add_mm_charges
from gpu4pyscf.qmmm.pbc.tools import determine_hcore_cutoff
from gpu4pyscf.dft import numint
import cupy as cp
#from scipy.interpolate import griddata
#from scipy.interpolate import RegularGridInterpolator
from cupyx.scipy.special import erf

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
max_scf_cycles = 10
screen_tol = 1e-14

# Chemistry - A European Journal, (2009), 186-197, 15(1)
ele2radius = {'N': 0.71, 'H': 0.32, 'C': 0.75, 'CL': 0.99, 'O': 0.63, 'CL': 0.99, 'K': 1.96, 'S': 1.03, 'FE': 1.16, 'P': 1.11, 'MG': 1.39, 'NA': 1.55}

mol = gto.Mole()
mol.charge = charge
mol.spin = spin
mol.verbose = 3
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
#mf.grids.level = 5

mf = add_mm_charges(mf, mm_coords, box_A, mm_charges, mm_radii, rcut_hcore=rcut_hcore, rcut_ewald=5)

dm = np.load("./dm_qmmm_ewald2_15.5_10.0.npy")

def energy_ewald(self, dm=None, mm_ewald_pot=None, qm_ewald_pot=None):
    # QM-QM and QM-MM pbc correction
    if dm is None:
        dm = self.make_rdm1()
    if mm_ewald_pot is None:
        if self.mm_ewald_pot is not None:
            mm_ewald_pot = self.mm_ewald_pot
        else:
            mm_ewald_pot = self.get_mm_ewald_pot(self.mol, self.mm_mol)
    if qm_ewald_pot is None:
        qm_ewald_pot = self.get_qm_ewald_pot(self.mol, dm, self.qm_ewald_hess)
    ewald_pot = mm_ewald_pot[0] #+ qm_ewald_pot[0] / 2
    e  = cp.einsum('i,i->', ewald_pot, (self.get_qm_charges(dm)-cp.asarray(mol.atom_charges())))
    ewald_pot = mm_ewald_pot[1] #+ qm_ewald_pot[1] / 2
    e += cp.einsum('ix,ix->', ewald_pot, self.get_qm_dipoles(dm))
    ewald_pot = mm_ewald_pot[2] #+ qm_ewald_pot[2] / 2
    e += cp.einsum('ixy,ixy->', ewald_pot, self.get_qm_quadrupoles(dm))
    # TODO add energy correction if sum(charges) !=0 ?
    return e

Eapprox = energy_ewald(mf, dm)
Eapprox += cp.einsum('ij,ji->', mf.get_hcore() - cp.asarray(RKS.get_hcore(mf, mol)), dm)

if xc is None:
    mf = RHF(mol)
else:
    mf = RKS(mol, xc=xc)
mf.conv_tol = scf_tol
mf.max_cycle = max_scf_cycles
mf.screen_tol = screen_tol
#mf.grids.level = 5
mf = add_mm_charges(mf, mm_coords, box_A, mm_charges, mm_radii, rcut_hcore=15.5, rcut_ewald=5)
mf.kernel(dm0=dm)
dm = mf.make_rdm1()

mm_mol = mf.mm_mol
mesh = mm_mol.mesh
ew_eta, ew_cut = mm_mol.get_ewald_params()
Gv, Gvbase, weights = mm_mol.get_Gv_weights(mesh)
absG2 = cp.einsum('gx,gx->g', Gv, Gv)
absG2[absG2==0] = 1e200

coulG = 4*cp.pi / absG2
coulG *= weights
# NOTE Gpref is actually Gpref*2
Gpref = cp.exp(-absG2/(4*ew_eta**2)) * coulG

GvR2 = cp.einsum('gx,ix->ig', Gv, mm_mol.atom_coords())
cosGvR2 = cp.cos(GvR2)
sinGvR2 = cp.sin(GvR2)

zcosGvR2 = cp.einsum("i,ig->g", mm_mol.atom_charges(), cosGvR2)
zsinGvR2 = cp.einsum("i,ig->g", mm_mol.atom_charges(), sinGvR2)

ni = mf._numint
Eref = 0
nelec = 0

mo_coeff = dm.mo_coeff
mo_occ = dm.mo_occ

opt = ni.gdftopt
optmol = opt.mol
coeff = cp.asarray(opt.coeff)
nao, nao0 = coeff.shape
dms = coeff @ dm @ coeff.T

mo_coeff = coeff @ mo_coeff
zeta = cp.sqrt(cp.asarray(mm_mol.get_zetas()))

for ao_mask, idx, weight, dft_grid_coords in ni.block_loop(optmol, mf.grids, nao, 1, blksize=256):
    mo_coeff_mask = mo_coeff[idx,:]
    rho = numint.eval_rho2(mol, ao_mask, mo_coeff_mask, mo_occ, None, 'GGA')
    nelec += (rho[0] * weight).sum()

    # ewald real-space
    Rij = dft_grid_coords[:,None,:] - cp.asarray(mm_mol.atom_coords()[None,:,:])
    Rij = cp.linalg.norm(Rij, axis=-1)
    rinv = (erf(zeta * Rij) - erf(ew_eta * Rij)) / Rij
    Eref += cp.einsum('i,ij,j->', -rho[0] * weight, rinv, mm_mol.atom_charges())

    # ewald g-space
    GvR1 = cp.einsum('gx,ix->ig', Gv, dft_grid_coords)
    cosGvR1 = cp.cos(GvR1)
    sinGvR1 = cp.sin(GvR1)
    Eref += cp.einsum('i,ig,g,g->', -rho[0] * weight, cosGvR1, zcosGvR2, Gpref)
    Eref += cp.einsum('i,ig,g,g->', -rho[0] * weight, sinGvR1, zsinGvR2, Gpref)

print(Eref - Eapprox)
