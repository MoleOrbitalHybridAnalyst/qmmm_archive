from qmmm import lammps4qmmm
from lammps import LMP_STYLE_GLOBAL, LAMMPS_INT

from pyscf.gto import Mole
from pyscf import df, lib
from gpu4pyscf.qmmm.pbc import itrf, mm_mole
from gpu4pyscf.qmmm.pbc.tools import estimate_error
from gpu4pyscf.dft import RKS

import numpy as np
import cupy as cp
from constants import *

from time import time
from sys import argv
from os import system, path, chdir, getcwd, environ

lib.num_threads(32)
home_dir = environ['HOME']


# Chemistry - A European Journal, (2009), 186-197, 15(1)
ele2radius = {'N': 0.71, 'H': 0.32, 'C': 0.75, 'CL': 0.99, 'O': 0.63, 'CL': 0.99, 'K': 1.96, 'S': 1.03, 'FE': 1.16, 'P': 1.11, 'MG': 1.39, 'NA': 1.55}

charge = -3
spin = 0

auxbasis = {'C': './6311Gss-rifit.dat', 'H': './6311Gss-rifit.dat', 'O': './6311Gss-rifit.dat', 'N': './6311Gss-rifit.dat', 'P': './6311Gss-rifit.dat', 'S': './6311Gss-rifit.dat', 'Mg': './6311Gss-rifit.dat'}
pseudo_bond_param_dir = f"{home_dir}/projects/pseudo_bond/jcp2008/refined_params/wb97x3c/separate"
aspc_nvec = 4
scf_conv_tol = 1e-10
scf_conv_tol_grad = 1e-6
max_scf_cycles = 1000
screen_tol = 1e-14
grids_level = 3
#rcut_hcore = 25           # in angstrom
rcut_ewald = 80           # in angstrom
max_memory = 80000
verbose = 5

qm_indexes = np.loadtxt("./qm_index.dat", dtype=int)       # all qm atoms, including CA
zeroq_indexes = np.loadtxt("./zeroq_index.dat", dtype=int) # mm charges invisible to qm
ca_indexes = np.loadtxt("./ca_index.dat", dtype=int)       # pseudo_bond atoms
ca_resnames = np.genfromtxt("./ca_resname.dat", dtype=str)

lmp = lammps4qmmm()
lmp.file("./cqm-eqb.in")
lmp.set_qm_atoms(qm_indexes, ca_indexes)

fp = open("./topo.xyz")
fp.readline(); fp.readline()
elements = np.array([line.split()[0] for line in fp])
fp.close()

def d4(mol):
    from dftd4.interface import DampingParam, DispersionModel
    charges = np.asarray(\
        [mol.atom_charge(i)+mol.atom_nelec_core(i) for i in range(mol.natm)])
    with lib.with_omp_threads(1):
        model = DispersionModel(
                charges,
                mol.atom_coords(),
                mol.charge)
        param = DampingParam(s6=1.0, s8=0.0, s9=1.0, a1=0.2464, a2=4.737)
        res = model.get_dispersion(param, grad=True)
    return res

def efv_scan(coords, box, rcut_hcore0, rcutlst, hexadecapole=False):
    '''
    return energy, force, virial given atom coords[N][3] and box[3][3] in Bohr
    let's ignore box
    '''
    t0 = time()

    coords_A = coords * Bohr / A
    box_A = box * Bohr / A
    natom = len(coords_A)
    mm_indexes = [i for i in range(natom) if not i in qm_indexes]

    ########################################
    ########### LAMMPS MM PART #############
    ########################################

    tags = lmp.numpy.extract_atom('id')
    order = np.argsort(tags)

    # set coordinates
    x = lmp.extract_atom('x')
    for i in range(natom):
        for k in range(3):
            x[order[i]][k] = coords_A[i][k]

    # get energy and force
    lmp.command('run 0 pre yes post no')
    tags = lmp.numpy.extract_atom('id')
    order = np.argsort(tags)
    f_mm = np.array(lmp.numpy.extract_atom('f'))
    f_mm = f_mm[order] * (kcal/A) / (hartree/Bohr)
    e_mm = lmp.extract_compute("thermo_pe", LMP_STYLE_GLOBAL, LAMMPS_INT) 
    e_mm = e_mm * kcal / hartree

    ########################################
    ########### PySCF QM PART ##############
    ########################################

    # make qm atoms whole
    ref = coords_A[qm_indexes[0]]
    diff = coords_A[qm_indexes] - ref
    n = (diff + 0.5 * np.diag(box_A)) // np.diag(box_A)
    diff = diff - n * np.diag(box_A)
    coords_A[qm_indexes] = diff + ref
    
    # move qm atoms to the center of box 
    ref = np.mean(coords_A[qm_indexes], axis=0)
    diff = coords_A - ref
    n = (diff + 0.5 * np.diag(box_A)) // np.diag(box_A)
    diff = diff - n * np.diag(box_A)
    coords_A = diff 

    t1 = time()
    print("LAMMPS time =", t1 - t0)

    assert len(coords) == len(elements)
    for i, idx in enumerate(ca_indexes):
        # change elements[CA] into special F types
        elements[idx] = f"F{i}"
    pos2str = lambda pos: " ".join([str(x) for x in pos])
    atom_str = [f"{a} {pos2str(pos)}\n" \
            for a,pos in zip(elements[qm_indexes],coords_A[qm_indexes])]
    mol = Mole()
    mol.atom = atom_str
    mol.charge = charge
    mol.spin = spin
    mol.verbose = verbose
    mol.max_memory = max_memory 
    mol.basis = {'default': f'{home_dir}/projects/wb97x-3c/jcp2023/supporting_info/vDZP_basis/basis_vDZP_NWCHEM.dat'}
    mol.ecp = {'default': f'{home_dir}/projects/wb97x-3c/jcp2023/supporting_info/vDZP_basis/ecp_vDZP_NWCHEM.dat'}
    for i, resname in enumerate(ca_resnames):
        # basis and ecp for CA pseudo_bond
        fname = f"{pseudo_bond_param_dir}/{resname.lower()}/sto-2g.dat"
        mol.basis[f"F{i}"] = fname
        mol.ecp[f"F{i}"] = fname
    mol.build()

    mm_coords = coords_A[mm_indexes]
    mm_charges = lmp.numpy.extract_atom('q')[order].copy() # all charges in tag order
    mm_charges[zeroq_indexes] = 0.0 # turn off backbone charges
    mm_charges[ca_indexes] = 0.0    # turn off CA charges in case not in zeroq_indexes
    mm_charges = mm_charges[mm_indexes]
    mm_radii = [ele2radius[e.upper()] for e in elements[mm_indexes]]

    auxbas = df.make_auxbasis(mol)
    if auxbasis is not None:
        for ele, bas in auxbasis.items():
            auxbas[ele] = bas
    mf = RKS(mol, xc='wb97xv').density_fit(auxbasis=auxbas)
    mf.grids.level = grids_level
    mf.conv_tol = scf_conv_tol
    mf.conv_tol_grad = scf_conv_tol_grad
    mf.max_cycle = max_scf_cycles
    mf.screen_tol = screen_tol
    mf.conv_check = False
    mf._numint.libxc.is_nlc = lambda *args: False  # turn off VV10

    mf = itrf.add_mm_charges(mf, mm_coords, box_A, mm_charges, mm_radii, rcut_hcore=rcut_hcore0, rcut_ewald=rcut_ewald)

    s1e = cp.asarray(mf.get_ovlp())
    dm0 = None

    e_qmmm = mf.kernel(dm0=dm0)
    dm0 = mf.make_rdm1().get()

    t2 = time()
    print("PySCF energy time =", t2 - t1)

    # D4
    d4res = d4(mf.mol)
    e_disp = d4res['energy']
    f_disp = -d4res['gradient']
    e_qmmm += e_disp

    t3 = time()
    print("PySCF grad time =", t3 - t2)
    print("efv total time =", t3 - t0)

    errlst = list()
    for rcut in rcutlst:
        errlst.append(estimate_error(mol, mm_coords, box_A, mm_charges, rcut, dm0, hexadecapole=hexadecapole))
    return errlst, e_qmmm

