export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# latest plumed that supports opes flooding
source ~/packages/plumed-2.9.0/sourceme.sh

# important to use latest ipi to work with latest plumed
export PYTHONPATH=/home/chhli/packages/i-pi-2.6.1:$PYTHONPATH
ipi=/home/chhli/packages/i-pi-2.6.1/bin/i-pi

taskset --cpu-list 0 $ipi nvt.xml > nvt.out &
echo "$!" > ipi.pid
sleep 60

taskset --cpu-list 1 ~/packages/lammps-2Aug2023/build/lmp -in lmp.in > lmp.out &

export OMP_NUM_THREADS=32
taskset --cpu-list 0-31 python -u driver.py > driver.out
