from pyscfad import gto, scf
from lno.ad import ccsd as lnoccsd

import argparse

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('inputxyz', default=None)
parser.add_argument('-b', '--basis', default=None)
parser.add_argument('-c', '--charge', default=-1)
parser.add_argument('--thresh_vir', default=None)
parser.add_argument('--thresh_occ', default="none")
parser.add_argument('--lo_type', default=None)
parser.add_argument('--frozen', default=None)
parser.add_argument('--max_memory', default=None)
parser.add_argument('--mfchkfile', default=None)

args = parser.parse_args()
inputxyz = args.inputxyz
basis = args.basis
charge = int(args.charge)
thresh_vir = float(args.thresh_vir)
if args.thresh_occ == "none":
    thresh_occ = 10 * thresh_vir
else:
    thresh_occ = float(args.thresh_occ)
lo_type = args.lo_type
try:
    frozen = int(args.frozen)
except:
    frozen = args.frozen
max_memory = int(args.max_memory)
mfchkfile = args.mfchkfile

mol = gto.Mole()
mol.charge = charge
mol.basis = basis
mol.verbose = 8
with open(inputxyz) as fp:
    fp.readline(); fp.readline()
    mol.atom = fp.readlines()
mol.max_memory = max_memory
mol.build(trace_exp=False, trace_ctr_coeff=False)

mf = scf.RHF(mol).density_fit()
if mfchkfile:
    from pyscf import scf as _scf
    _mol, mfdict = _scf.chkfile.load_scf(mfchkfile)
    _mf = _scf.RHF(_mol)
    _mf.mo_coeff = mfdict['mo_coeff']
    _mf.mo_occ = mfdict['mo_occ']
    _mf.mo_energy = mfdict['mo_energy']
    dm0 = _mf.make_rdm1()
    _mol = _mf = mfdict = None
else:
    dm0 = None
ehf = mf.kernel(dm0=dm0)

mfcc = lnoccsd.LNOCCSD(mf, frozen=frozen)
mfcc.thresh_occ = thresh_occ
mfcc.thresh_vir = thresh_vir
mfcc.lo_type = lo_type
mfcc.ccsd_t = True
mfcc.use_local_virt = True
mfcc.natorb_occdeg_thresh = thresh_vir / 100
mfcc.kernel()

print("HF:", ehf)
print("LNO MP2 Ecorr:", mfcc.e_corr_pt2)
print("LNO CCSD Ecorr:", mfcc.e_corr_ccsd)
print("LNO (T) Ecorr:", mfcc.e_corr_ccsd_t)
print("LNO CCSD(T) Ecorr:", mfcc.e_corr)


