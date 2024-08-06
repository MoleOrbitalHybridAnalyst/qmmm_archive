''' Octupole error estimate using dm converged at Rcut=18 '''
from qmmm_efv import efv_scan, np, Bohr

box_A = np.eye(3) * 121.5
coords_A = list()
with open("./cqm-eqb_ctrl.xyz") as fp:
    fp.readline()
    fp.readline()
    for line in fp:
        coords_A.append(line.split()[1:])
coords_A = np.array(coords_A, dtype=float)

rcutlst = list(range(15,20)) + list(range(20,30,2)) + [30, 35, 40]
errlst, _ = efv_scan(coords_A / Bohr, box_A / Bohr, 18.0, rcutlst)
np.savetxt("determine_rcut4.dat", np.array([rcutlst, errlst]).transpose())
