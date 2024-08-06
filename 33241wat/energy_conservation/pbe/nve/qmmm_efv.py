from pyscf.gto import Mole
from pyscf import df, lib
from gpu4pyscf.qmmm.pbc import itrf, mm_mole
from gpu4pyscf.qmmm.pbc.tools import determine_hcore_cutoff
from gpu4pyscf.dft import RKS

from aspc_pred import DMPredictor

import numpy as np
import cupy as cp
from constants import *

from time import time
from sys import argv
from os import system, path, chdir, getcwd, environ

lib.num_threads(4)
home_dir = environ['HOME']


# Chemistry - A European Journal, (2009), 186-197, 15(1)
ele2radius = {'N': 0.71, 'H': 0.32, 'C': 0.75, 'CL': 0.99, 'O': 0.63, 'CL': 0.99, 'K': 1.96, 'S': 1.03, 'FE': 1.16, 'P': 1.11, 'MG': 1.39, 'NA': 1.55}

# charge and spin of the QM system
charge = 0
spin = 0

# QM SCF settings
xcstr = 'pbe'
basis = 'def2-SVPD'
aspc_nvec = 4
eval_cutoff_stride = 10
scf_conv_tol = 1e-10
scf_conv_tol_grad = 1e-6
max_scf_cycles = 1000
screen_tol = 1e-14
grids_level = 3
rcut_hcore0 = 16          # MM beyond cutoff see QM as multipoles; in angstrom
rcut_ewald = 80           # Ewald cutoff in angstrom
max_memory = 4000         # max_memory used by PySCF in MB
verbose = 3

def read_indexes(inp):
    ss = np.genfromtxt(inp, dtype=str)
    assert ss[0] == "group"
    assert ss[2] == "id"
    return np.array(ss[3:], dtype=int) - 1 # -1 cuz lmp uses serials

qm_indexes = read_indexes("./group_qm.inp")                # all qm atoms

# read a xyz to know element type of each atom
with open("./water.xyz") as fp:
    fp.readline(); fp.readline()
    elements = np.array([line.split()[0] for line in fp])

# read lammps data to know all the charges
# indeed the column for q may be different for other lmp atom_style
# may be replaced by using Pizza.py 
with open("./water.data") as fp:
    tags = list()
    charges = list()
    jline = np.inf
    for iline, line in enumerate(fp):
        if line[:5] == "Atoms":
            jline = iline
        if iline >= jline + 2:
            if line == "\n":
                break
            lsplt = line.split()
            tags.append(lsplt[0])
            charges.append(lsplt[3])
    tags = np.array(tags, dtype=int)
    charges = np.array(charges, dtype=float)
    order = np.argsort(tags)
    charges = charges[order]

def efv_scan(coords, box, init_dict):
    '''
    return energy, force, virial given atom coords[N][3] and box[3][3] in Bohr
    '''
    t0 = time()

    coords_A = coords * Bohr / A
    box_A = box * Bohr / A
    natom = len(coords_A)
    mm_indexes = [i for i in range(natom) if not i in qm_indexes]

    ########################################
    ########### PySCF QM PART ##############
    ########################################

    # make qm atoms whole
    # by moving every qm atom within half box
    # of qm atom No. 0
    ref = coords_A[qm_indexes[0]]
    diff = coords_A[qm_indexes] - ref
    n = (diff + 0.5 * np.diag(box_A)) // np.diag(box_A)
    diff = diff - n * np.diag(box_A)
    coords_A[qm_indexes] = diff + ref
    
    # move qm atoms to the center of box 
    # this is important cuz NON-periodic QM integrals in
    # PySCF will be used
    ref = np.mean(coords_A[qm_indexes], axis=0)
    diff = coords_A - ref
    n = (diff + 0.5 * np.diag(box_A)) // np.diag(box_A)
    diff = diff - n * np.diag(box_A)
    coords_A = diff 

    t1 = time()

    # build QM subsystem in PySCF
    assert len(coords) == len(elements)
    pos2str = lambda pos: " ".join([str(x) for x in pos])
    atom_str = [f"{a} {pos2str(pos)}\n" \
            for a,pos in zip(elements[qm_indexes],coords_A[qm_indexes])]
    mol = Mole()
    mol.atom = atom_str
    mol.charge = charge
    mol.spin = spin
    mol.verbose = verbose
    mol.max_memory = max_memory 
    mol.basis = basis
    mol.build()

    mm_coords = coords_A[mm_indexes]    # get MM positions in angstrom
    mm_charges = charges.copy()         # all charges in tag order
    mm_charges = mm_charges[mm_indexes] # only MM charges will be used
    mm_radii = [ele2radius[e.upper()] \
        for e in elements[mm_indexes]]  # get MM radii seen in QM calculations
    assert abs(np.sum(mm_charges) + charge) < 1e-8

    # build a Mean-field object in PySCF
    # will be the calculator to compute energy
    mf = RKS(mol, xc=xcstr).density_fit()
    mf.grids.level = grids_level
    mf.conv_tol = scf_conv_tol
    mf.conv_tol_grad = scf_conv_tol_grad
    mf.max_cycle = max_scf_cycles
    mf.screen_tol = screen_tol
    mf.conv_check = False

    s1e = cp.asarray(mf.get_ovlp())
    if 'dm_predictor' in init_dict:
        ps0 = init_dict['dm_predictor'].predict()
        mo0 = init_dict['mo0']
        mo0 = cp.dot(ps0, mo0)
        nocc = mol.nelectron // 2
        csc = cp.dot(cp.dot(mo0[:,:nocc].T, s1e), mo0[:,:nocc])
        w, v = cp.linalg.eigh(csc)
        csc_invhalf = cp.dot(v, cp.diag(w**(-0.5)) @ v.T)
        mo0[:,:nocc] = cp.dot(mo0[:,:nocc], csc_invhalf)
        mo_occ = mf.get_occ(np.arange(mol.nao), mo0)
        dm0 = mf.make_rdm1(mo0, mo_occ)
    else:
        dm0 = None

    istep = init_dict.get('istep', 0)
    if dm0 is not None and \
            istep % eval_cutoff_stride == 1:
        rcut_hcore, _ = \
            determine_hcore_cutoff(
                mol, mm_coords, box_A, mm_charges, 13,
                dm0.get(), rcut_step=1, precision=2e-4)
        init_dict['rcut_hcore'] = rcut_hcore
        print(f"istep = {istep} rcut_hcore = {rcut_hcore}")
    else:
        rcut_hcore = init_dict.get("rcut_hcore", rcut_hcore0)

    mf = itrf.add_mm_charges(
        mf, mm_coords, box_A, mm_charges, mm_radii, \
        rcut_hcore=rcut_hcore, rcut_ewald=rcut_ewald)

    # run QM and get energy
    e_qmmm = mf.kernel(dm0=dm0)

    t2 = time()
    print("PySCF energy time =", t2 - t1)

    # create the calculator to compute forces
    mf_grad = mf.nuc_grad_method()
    mf_grad.max_memory = max_memory
    f_qmmm = np.empty((natom, 3))
    # compute forces on QM atoms:
    f_qmmm[qm_indexes] = -mf_grad.kernel()
    # compute forces on MM atoms:
    dm = mf.make_rdm1()
    f_qmmm[mm_indexes] = -(mf_grad.grad_nuc_mm() + mf_grad.grad_hcore_mm(dm) + mf_grad.de_ewald_mm)

    if 'dm_predictor' not in init_dict:
        init_dict['dm_predictor'] = DMPredictor(aspc_nvec)

    init_dict['dm_predictor'].append(cp.dot(dm, s1e))
    init_dict['mo0'] = mf.mo_coeff
    init_dict['istep'] = istep + 1

    # restraint to prevent qm water diffuse away
    qm_coords = coords_A[qm_indexes[::3]]
    cen = np.mean(qm_coords, axis=0)
    r2 = np.sum((qm_coords - cen)**2)
    kappa = 0.1                          # in kcal/mol/A^2
    e_qmmm += 0.1 / hartree * r2
    g_res = 0.1 * 2 * (qm_coords - cen)  # in kcal/mol/A
    f_qmmm[qm_indexes[::3]] -= g_res / (hartree / Bohr)

    t3 = time()
    print("PySCF grad time =", t3 - t2)
    print("efv total time =", t3 - t0)

    # return the qmmm energy, forces, and virial (not computed)
    # and any information next MD step needs to know (could be nothing indeed)
    return e_qmmm, f_qmmm, None, init_dict

