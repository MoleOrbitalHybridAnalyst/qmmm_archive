import numpy as np
import re

def basis_extrap1(e, X, expnt=3.05):
    assert len(e) == len(X)
    assert len(e) >= 2
    if len(e) == 2:
        return (e[0]*X[0]**expnt-e[1]*X[1]**expnt) / (X[0]**expnt-X[1]**expnt)
    else:
        opt, _ = curve_fit(lambda x, E, A, beta: E + A * x**(-beta), X, e, p0=(min(e), 10, 3))
        return opt[0]

def basis_extrap2(e, X, alpha=5.46):
    assert len(e) == len(X)
    assert len(e) == 2
    return \
       (e[0]*np.exp(alpha*np.sqrt(X[0])) - \
        e[1]*np.exp(alpha*np.sqrt(X[1])) ) / \
       (np.exp(alpha*np.sqrt(X[0])) - \
        np.exp(alpha*np.sqrt(X[1])))

def read_energy(fname):
    with open(fname) as fp:
        for line in fp:
            if re.search("^HF:", line):
                hf = float(line.split(":")[-1])
            if re.search("^LNO MP2 Ecorr:", line):
                mp2_ecorr = float(line.split(":")[-1])
            if re.search("^LNO CCSD\(T\) Ecorr:", line):
                ccsdt_ecorr = float(line.split(":")[-1])
    return hf, mp2_ecorr, ccsdt_ecorr

def read_mp2(fname):
    with open(fname) as fp:
        for line in fp:
            if re.search("^HF:", line):
                hf = float(line.split(":")[-1])
            if re.search("^MP2 Ecorr:", line):
                mp2_ecorr = float(line.split(":")[-1])
    return hf, mp2_ecorr

def read_composite(basis, geom):
    hf, lmp2, lccsdt = read_energy(f"./{basis}/1e-5/{geom}/run.out")
    hf_, mp2 = read_mp2(f"../mp2/{basis}/{geom}/run.out")
    assert abs(hf - hf_) < 1e-5
    return hf, lccsdt - lmp2 + mp2


hf_tz_R, ecorr_tz_R = read_composite("tz", "wb97x3cR90Flipped_neb00")
hf_tz_P, ecorr_tz_P = read_composite("tz", "wb97x3cR90Flipped_neb09")
hf_tz_TS, ecorr_tz_TS = read_composite("tz", "wb97x3cR90Flipped_neb04")
hf_tz_05, ecorr_tz_05 = read_composite("tz", "wb97x3cR90Flipped_neb05")
hf_tz_03, ecorr_tz_03 = read_composite("tz", "wb97x3cR90Flipped_neb03")

hf_qz_R, ecorr_qz_R = read_composite("qz", "wb97x3cR90Flipped_neb00")
hf_qz_P, ecorr_qz_P = read_composite("qz", "wb97x3cR90Flipped_neb09")
hf_qz_TS, ecorr_qz_TS = read_composite("qz", "wb97x3cR90Flipped_neb04")
hf_qz_05, ecorr_qz_05 = read_composite("qz", "wb97x3cR90Flipped_neb05")
hf_qz_03, ecorr_qz_03 = read_composite("qz", "wb97x3cR90Flipped_neb03")

hf_extrp_R = basis_extrap2( [hf_tz_R, hf_qz_R  ], [3,4])
hf_extrp_P = basis_extrap2( [hf_tz_P, hf_qz_P  ], [3,4])
hf_extrp_TS = basis_extrap2([hf_tz_TS, hf_qz_TS], [3,4])
hf_extrp_05 = basis_extrap2([hf_tz_05, hf_qz_05], [3,4])
hf_extrp_03 = basis_extrap2([hf_tz_03, hf_qz_03], [3,4])

ecorr_extrp_R = basis_extrap1( [ecorr_tz_R, ecorr_qz_R  ], [3,4])
ecorr_extrp_P = basis_extrap1( [ecorr_tz_P, ecorr_qz_P  ], [3,4])
ecorr_extrp_TS = basis_extrap1([ecorr_tz_TS, ecorr_qz_TS], [3,4])
ecorr_extrp_05 = basis_extrap1([ecorr_tz_05, ecorr_qz_05], [3,4])
ecorr_extrp_03 = basis_extrap1([ecorr_tz_03, ecorr_qz_03], [3,4])

# P R TS 05 03
e_final = np.array([hf_extrp_P+ecorr_extrp_P, hf_extrp_R+ecorr_extrp_R, hf_extrp_TS+ecorr_extrp_TS, hf_extrp_05+ecorr_extrp_05, hf_extrp_03+ecorr_extrp_03])
res = np.transpose([[9.0,0.0,4.0,5.0,3.0],[*e_final]])
res = res[np.argsort(res[:,0])]
np.savetxt("wb97x3cR90Flipped_neb.ene.dat", res)
