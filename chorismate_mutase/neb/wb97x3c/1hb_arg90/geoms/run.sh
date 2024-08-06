cp ../../../qmmm_geomopt/wb97x3c/reactant/last.xyz  R.xyz
cp ../../../qmmm_geomopt/wb97x3c/product/last.xyz  P.xyz
vmd -dispdev text R.xyz -e extract_qm.tcl > /dev/null && \mv qm.xyz R.xyz
vmd -dispdev text P.xyz -e extract_qm.tcl > /dev/null && \mv qm.xyz P.xyz
