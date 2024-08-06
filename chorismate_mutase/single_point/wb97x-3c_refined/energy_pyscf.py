from pyscf import gto, lib
from gpu4pyscf import dft

from sys import argv

from dftd4.interface import DampingParam, DispersionModel

import numpy as np

hyb = float(argv[3])
try:
    alpha = float(argv[4])
except:
    alpha = 1.0

mol = gto.Mole()
if len(argv) == 2:
    mol.charge = -1
else:
    mol.charge = int(argv[2])
mol.basis = "/home/chhli/projects/wb97x-3c/jcp2023/supporting_info/vDZP_basis/basis_vDZP_NWCHEM.dat"
mol.ecp = "/home/chhli/projects/wb97x-3c/jcp2023/supporting_info/vDZP_basis/ecp_vDZP_NWCHEM.dat"
mol.fromfile(argv[1])
mol.verbose = 0
mol.build()

mf = dft.RKS(mol).density_fit(auxbasis='./6311Gss-rifit.dat')
mf.xc = 'wb97xv'
mf._numint.libxc.is_nlc = lambda *args: False
mf._numint.libxc.rsh_coeff = lambda *args : (0.3, alpha, hyb - alpha)

charges = np.asarray(\
    [mol.atom_charge(i)+mol.atom_nelec_core(i) for i in range(mol.natm)])
with lib.with_omp_threads(1):
    model = DispersionModel(
            charges,
            mol.atom_coords(),
            mol.charge)
    param = DampingParam(s6=1.0, s8=0.0, s9=1.0, a1=0.2464, a2=4.737)
    res = model.get_dispersion(param, grad=True)

print(f"E(wb97x-3c hyb={hyb}) =", mf.kernel() + res['energy'])
