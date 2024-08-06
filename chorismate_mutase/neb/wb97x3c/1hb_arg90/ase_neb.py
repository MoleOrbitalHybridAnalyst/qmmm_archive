from pyscf.gto import Mole
from pyscf import df, lib
from gpu4pyscf.qmmm.pbc import itrf, mm_mole
from gpu4pyscf.qmmm.pbc.tools import determine_hcore_cutoff
from gpu4pyscf.dft import RKS

from aspc_pred import DMPredictor

from lammps import LMP_STYLE_GLOBAL, LAMMPS_INT
import lammps

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

charge = -1
spin = 0

auxbasis = {'C': './6311Gss-rifit.dat', 'H': './6311Gss-rifit.dat', 'O': './6311Gss-rifit.dat', 'N': './6311Gss-rifit.dat', 'P': './6311Gss-rifit.dat', 'S': './6311Gss-rifit.dat', 'Mg': './6311Gss-rifit.dat'}
pseudo_bond_param_dir = f"{home_dir}/projects/pseudo_bond/jcp2008/refined_params/wb97x3c/separate"
aspc_nvec = 4
eval_cutoff_stride = 5
scf_conv_tol = 1e-10
scf_conv_tol_grad = 1e-6
max_scf_cycles = 1000
screen_tol = 1e-14
grids_level = 3
rcut_hcore0 = 16          # in angstrom
rcut_ewald = 50           # in angstrom
max_memory = 80000
verbose = 3

def read_indexes(inp):
    ss = np.genfromtxt(inp, dtype=str)
    assert ss[0] == "group"
    assert ss[2] == "id"
    return np.array(ss[3:], dtype=int) - 1 # cuz lmp uses serial

qm_indexes = read_indexes("./group_qm.inp")                # all qm atoms, including CA
zeroq_indexes = np.loadtxt("./zeroq_index.dat", dtype=int) # mm charges invisible to qm
ca_indexes = read_indexes("./group_ca.inp")                # pseudo_bond atoms
ca_resnames = ["ARG"]

lmp = lammps.lammps()
lmp.file("./lmp.in")

moving_indexes = np.loadtxt("./moving_index.dat", dtype=int) # atoms that are going to move

with open("./topo.xyz") as fp:
    fp.readline(); fp.readline()
    elements = np.array([line.split()[0] for line in fp])
    fp.seek(0,0)
    fp.readline(); fp.readline()
    full_coords0 = np.array([line.split()[1:] for line in fp], dtype=float)

    # set the coords in lammps to be topo.xyz
    # since data.system may correspond to a different configuration
    tags = lmp.numpy.extract_atom('id')
    order = np.argsort(tags)

    # set coordinates
    x = lmp.extract_atom('x')
    for i in range(len(full_coords0)):
        for k in range(3):
            x[order[i]][k] = full_coords0[i][k]


with open("./data.system") as fp:
    tags = list()
    charges = list()
    jline = np.inf
    for iline, line in enumerate(fp):
        if line[:5] == "Atoms":
            jline = iline
#            assert line.split()[-1] == "full"
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

def lammps_efv(coords_A):
    '''
    separated from efv_scan so that not pollute tags, charges, order, etc.
    '''
    tags = lmp.numpy.extract_atom('id')
    order = np.argsort(tags)

    # set coordinates
    x = lmp.extract_atom('x')
    for i, idx in enumerate(moving_indexes):
        for k in range(3):
            x[order[idx]][k] = coords_A[i][k]

    # get energy and force
    lmp.command('run 0 pre yes post no')
    tags = lmp.numpy.extract_atom('id')
    order = np.argsort(tags)
    f_mm = np.zeros_like(coords_A)
    f_full = lmp.numpy.extract_atom('f')[order]
    f_mm = f_full[moving_indexes] * (kcal/A) / (hartree/Bohr)
    e_mm = lmp.extract_compute("thermo_pe", LMP_STYLE_GLOBAL, LAMMPS_INT) 
    e_mm = e_mm * kcal / hartree
    return e_mm, f_mm, None

def efv_scan(mv_coords, box, init_dict):
    '''
    return energy, force, virial given atom coords[N][3] and box[3][3] in Bohr
    NOTE mv_coords are coords of moving atoms
    '''
    t0 = time()

    coords_A = full_coords0.copy()
    coords_A[moving_indexes] = mv_coords * Bohr / A
    box_A = box * Bohr / A
    natom = len(coords_A)
    mm_indexes = [i for i in range(natom) if not i in qm_indexes]


    ########################################
    ########### LAMMPS MM PART #############
    ########################################

    e_mm, f_mm, _ = lammps_efv(mv_coords * Bohr / A)

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

    assert len(coords_A) == len(elements)
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
    print("mol.nao =", mol.nao)

    mm_coords = coords_A[mm_indexes]
    mm_charges = charges.copy()     # all charges in tag order
    mm_charges[zeroq_indexes] = 0.0 # turn off backbone charges
    mm_charges[ca_indexes] = 0.0    # turn off CA charges in case not in zeroq_indexes
    mm_charges = mm_charges[mm_indexes]
    mm_radii = [ele2radius[e.upper()] for e in elements[mm_indexes]]
#    assert abs(np.sum(mm_charges) + charge) < 1e-8

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

    dm0 = None

    rcut_hcore = init_dict.get("rcut_hcore", rcut_hcore0)

    mf = itrf.add_mm_charges(
        mf, mm_coords, box_A, mm_charges, mm_radii, 
        rcut_hcore=rcut_hcore, rcut_ewald=rcut_ewald)
    e_qmmm = mf.kernel(dm0=dm0)

    t2 = time()
    print("PySCF energy time =", t2 - t1)

    mf_grad = mf.nuc_grad_method()
    mf_grad.max_memory = max_memory
    mf_grad.auxbasis_response = True
    f_qmmm = np.empty((natom, 3))
    f_qmmm[qm_indexes] = -mf_grad.kernel()
    dm = mf.make_rdm1()
    f_qmmm[mm_indexes] = -(mf_grad.grad_nuc_mm() + mf_grad.grad_hcore_mm(dm) + mf_grad.de_ewald_mm)

    # D4
    d4res = d4(mf.mol)
    e_disp = d4res['energy']
    f_disp = -d4res['gradient']
    e_qmmm += e_disp
    f_qmmm[qm_indexes] += f_disp

    t3 = time()
    print("PySCF grad time =", t3 - t2)
    print("efv total time =", t3 - t0)

    f_qmmm = f_qmmm[moving_indexes]

    return e_mm + e_qmmm, f_mm + f_qmmm, None, init_dict


from ase.calculators.calculator import Calculator

class PYSCF(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, box, restart=None, label='pyscf',
            atoms=None, scratch=None, **kwargs):

        Calculator.__init__(self, restart=restart, label=label, atoms=atoms, scratch=scratch, **kwargs)
        self.init_dict = dict()
        self.box = box # in Angstrom

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=['positions']):
        Calculator.calculate(self, atoms)
        coords = atoms.get_positions()
        e, f, v, self.init_dict = efv_scan(coords * A / Bohr, box * A / Bohr, init_dict=self.init_dict)
        self.results['energy'] = e * hartree / eV
        self.results['forces'] = f * (hartree/Bohr) / (eV/A)

if __name__ == "__main__":
    from sys import argv
    from ase.io import read, write
    from ase.optimize import FIRE, QuasiNewton
    from ase.neb import NEB
    import numpy as np

    box = np.diag([79.006,   79.682,   79.030])
# I haved checked this gives zero force
#    calc = PYSCF(box)
#    atoms = read("./geoms/R.xyz")
#    calc.calculate(atoms)

    NIMAGES = 10             # total number of images

    # create neb images
    images = [read(argv[1])]
    for i in range(NIMAGES-2):
        images.append(read(argv[1]))
        images[-1].calc = PYSCF(box, atoms=images[-1])
    images.append(read(argv[-1]))

    # optimize neb w/o climbing image
    neb = NEB(images)
    neb.interpolate(method='idpp')
    dyn = FIRE(neb, trajectory=f'neb.traj', logfile=f'neb.log')
    dyn.run(fmax=0.15, steps=500)

    # optimize neb w/ climbing image
    neb = NEB(images, climb=True)
    dyn = FIRE(neb, trajectory=f'nebclimb.traj', logfile=f'nebclimb.log')
    dyn.run(fmax=0.15, steps=2000)
