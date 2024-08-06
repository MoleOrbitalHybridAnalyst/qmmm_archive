for i in {0..9}
do
    python energy_pyscf.py ../wb97x-3c/geoms/wb97x3cR90Flipped_neb0${i}.xyz -1 0.2745 0.3716 > results/wb97x3cR90Flipped_neb0${i}.hyb=0.2745.alpha=0.3716.pyscf_out
    python energy_pyscf.py ../wb97x-3c/geoms/wb97x3c_neb0${i}.xyz -1 0.2745 0.3716 > results/wb97x3c_neb0${i}.hyb=0.2745.alpha=0.3716.pyscf_out
done
