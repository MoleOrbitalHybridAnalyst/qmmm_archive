export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1

taskset --cpu-list 33 i-pi nvt.xml > nvt.out &
sleep 30

taskset --cpu-list 34 ~/packages/lammps-2Aug2023/build/lmp -in cqm-eqb.in &

taskset --cpu-list 33-64 env OMP_NUM_THREADS=32 time python -u driver.py > driver.out
