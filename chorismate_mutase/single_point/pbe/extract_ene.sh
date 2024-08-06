# 1HB Arg90 neb geoms
tail -n 1 -q results/wb97x3c_neb0*out | cut -d= -f2 > wb97x3c_neb.ene.dat
# 2HB Arg90 neb geoms
tail -n 1 -q results/wb97x3cR90Flipped_neb0*.pyscf_out | cut -d= -f2 > wb97x3cR90Flipped_neb.ene.dat
