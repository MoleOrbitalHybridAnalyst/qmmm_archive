cp ../../../qmmm_geomopt/wb97x3c_R90Flipped_smd/reactant/last.xyz  R.xyz
cp ../../../qmmm_geomopt/wb97x3c_R90Flipped_smd/product/last.xyz  P.xyz
vmd -dispdev text R.xyz -e extract_qm.tcl > /dev/null && \mv qm.xyz R.xyz
vmd -dispdev text P.xyz -e extract_qm.tcl > /dev/null && \mv qm.xyz P.xyz
