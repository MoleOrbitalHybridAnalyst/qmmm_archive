# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy
from pyscf import gto, df
from gpu4pyscf import scf
from gpu4pyscf.solvent import smd

def setUpModule():
    global mol, epsilon, lebedev_order
    mol = gto.Mole()
    mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
    mol.basis = 'def2-tzvpp'
    mol.output = '/dev/null'
    mol.build()
    lebedev_order = 29

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_cds_solvent(self):
        smdobj = smd.SMD(mol)
        smdobj.solvent = 'toluene'
        e_cds = smdobj.get_cds()
        assert numpy.abs(e_cds - -0.0013476060879874362) < 1e-8

    def test_cds_water(self):
        smdobj = smd.SMD(mol)
        smdobj.solvent = 'water'
        e_cds = smdobj.get_cds()
        assert numpy.abs(e_cds - 0.0022847142144050057) < 1e-8

    def test_smd_solvent(self):
        mf = scf.RHF(mol)
        mf = mf.SMD()
        mf.with_solvent.solvent = 'ethanol'
        e_tot = mf.kernel()
        assert numpy.abs(e_tot - -76.075066568) < 2e-4

    def test_smd_water(self):
        mf = scf.RHF(mol)
        mf = mf.SMD()
        mf.with_solvent.solvent = 'water'
        e_tot = mf.kernel()
        assert numpy.abs(e_tot - -76.0756052903) < 2e-4

if __name__ == "__main__":
    print("Full Tests for SMDs")
    unittest.main()