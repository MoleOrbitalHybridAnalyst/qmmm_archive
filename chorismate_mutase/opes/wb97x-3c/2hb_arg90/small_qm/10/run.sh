. ~/share/load_gpu4pyscf.sh

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# latest plumed that supports opes flooding
source ~/packages/plumed2-2.9.0/sourceme.sh

# important to use latest ipi to work with latest plumed
export PYTHONPATH=${HOME}/packages/i-pi-2.6.1:$PYTHONPATH
ipi=${HOME}/packages/i-pi-2.6.1/bin/i-pi

$ipi nvt.xml > nvt.out &
echo "$!" > ipi.pid
sleep 40

lmp -in lmp.in > lmp.out &

export OMP_NUM_THREADS=4
srun -n 1 -c 4 -G 1 --cpu-bind=cores python -u driver.py > driver.out
