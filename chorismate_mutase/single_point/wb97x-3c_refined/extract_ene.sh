tail -n 1 -q results/wb97x3c_neb0*.hyb=0.2745.alpha=0.3716.pyscf_out | cut -d= -f3 > wb97x3c_neb.hyb=0.2745.alpha=0.3716.ene.dat
tail -n 1 -q results/wb97x3cR90Flipped_neb0*.hyb=0.2745.alpha=0.3716.pyscf_out | cut -d= -f3 > wb97x3cR90Flipped_neb.hyb=0.2745.alpha=0.3716.ene.dat