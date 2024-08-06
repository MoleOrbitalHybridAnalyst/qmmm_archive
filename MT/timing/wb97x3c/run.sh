export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

taskset --cpu-list 1 i-pi nvt.xml > nvt.out &
sleep 30

taskset --cpu-list 2 ~/packages/lammps-2Aug2023/build/lmp -in cqm-eqb.in &

taskset --cpu-list 1-32 env OMP_NUM_THREADS=32 time python -u driver.py > driver.out
