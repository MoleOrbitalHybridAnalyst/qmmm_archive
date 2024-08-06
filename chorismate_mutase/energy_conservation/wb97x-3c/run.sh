export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=3

i-pi nvt.xml > nvt.out &
echo "$!" > ipi.pid
sleep 60

~/packages/lammps-2Aug2023/build/lmp -in lmp.in > lmp.out &

env OMP_NUM_THREADS=20 python -u driver.py > driver.out &
