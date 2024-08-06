export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

taskset --cpu-list 0 i-pi nvt.xml > nvt.out &
echo "$!" > ipi.pid
sleep 60

taskset --cpu-list 1 ~/packages/lammps-2Aug2023/build/lmp -in lmp.in > lmp.out &

env OMP_NUM_THREADS=32 time python -u driver.py > driver.out
