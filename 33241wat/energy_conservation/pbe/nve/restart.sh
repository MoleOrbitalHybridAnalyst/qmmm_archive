export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

i-pi RESTART >> nve.out &
echo "$!" > ipi.pid
sleep 30

~/packages/lammps-2Aug2023/build/lmp -in lmp.in >> lmp.out &

env OMP_NUM_THREADS=4 python -u driver.py >> driver.out
