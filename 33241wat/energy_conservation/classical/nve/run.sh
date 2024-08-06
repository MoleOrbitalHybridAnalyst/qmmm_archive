export OMP_NUM_THREADS=1

i-pi nve.xml > nve.out &
echo "$!" > ipi.pid
sleep 30

~/packages/lammps-2Aug2023/build/lmp -in lmp.in > lmp.out &

wait
