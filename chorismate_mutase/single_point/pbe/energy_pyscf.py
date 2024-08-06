from pyscf import gto, lib
from gpu4pyscf import dft

from sys import argv

import numpy as np

mol = gto.Mole()
mol.charge = -1
mol.basis = "6-31G**"
mol.fromfile(argv[1])
mol.verbose = 5
mol.build()

mf = dft.RKS(mol)
mf.xc = 'pbe'

mf.kernel()

if mf.converged == False:
    from pyscf import dft
    dm0 = mf.make_rdm1().get()
    mf = dft.RKS(mol).density_fit()
    mf.xc = 'pbe'
    mf = mf.newton()
    mf.kernel(dm0=dm0)
