for i in {0..9}
do
    python energy_pyscf.py geoms/wb97x3c_neb0${i}.xyz > results/wb97x3c_neb0${i}.pyscf_out
done

for i in {0..9}
do
    python energy_pyscf.py geoms/wb97x3cR90Flipped_neb0${i}.xyz > results/wb97x3cR90Flipped_neb0${i}.pyscf_out
done
