import numpy as np
from sys import argv


#================= PARAMETERS ==============================

barrier = float(argv[1]) #from plumed.dat input file for OPES

T = 300 #Temperature in K

biasColumn = 5 #Column in which the bias is printed in the COLVAR file
wallColumn = 10 #Column in which the bias from wall is printed in the COLVAR file

dt = 0.001 #The frequency at which colvars is deposited (in ps)

time_unit = 10**12 # 1 s in 1 ps

#===========================================================

kT = 0.0083144621 * T # kT in kj/mol; this value is used by PLUMED; see plumed/src/tools/Tools.h

#load COLVAR file and do the analysis
cv = np.loadtxt('COLVAR')

f1 = open('time_wall','w')

a = 0
c = 0
for i in range(len(cv)):
    a += np.exp((cv[i,biasColumn-1]+barrier+cv[i,wallColumn-1])/kT)
    b = np.exp((cv[i,biasColumn-1]+barrier+cv[i,wallColumn-1])/kT)
    c += dt*b

#    print(cv[i,0],cv[i,1],c/time_unit,(cv[i,0]/time_unit)*(a/(i+1)),file=f1)
    print(cv[i,0],cv[i,3],c/time_unit, file=f1)


f1.close()


