import numpy
from scipy.optimize import minimize
from os import system

def get_energy(params, geom):
    system(f"python energy_pyscf.py {geom} -1 {params[0]} {params[1]} > out")
    with open("out") as fp:
        last_line = fp.readlines()[-1]
        return float(last_line.split()[-1])

def loss(params):
    hyd = params[0]   # scales hf_sr
    alpha = params[1] # scales hf_lr

    ref1 = -0.027374018043246906  # forward barrier at DLPNO-CCSD(T)/CBS
    ref2 = -0.032599693900010616  # forward barrier wht R90 Flipped

    eR  = get_energy(params, "../wb97x-3c/geoms/wb97x3c_neb00.xyz")
    eTS = get_energy(params, "../wb97x-3c/geoms/wb97x3c_neb04.xyz")
    loss = (eR - eTS - ref1)**2

    eR  = get_energy(params, "../wb97x-3c/geoms/wb97x3cR90Flipped_neb00.xyz")
    eTS = get_energy(params, "../wb97x-3c/geoms/wb97x3cR90Flipped_neb04.xyz")
    loss += (eR - eTS - ref2)**2

    print(f"params, loss =", params, loss)

    return loss

if __name__ == "__main__":
    #opt = minimize(loss, [0.12, 0.75], method='Nelder-Mead', bounds=((0.1, 0.17), (0.6, 1.1)))
    opt = minimize(loss, [0.12, 0.75], method='BFGS')
    print(opt)
