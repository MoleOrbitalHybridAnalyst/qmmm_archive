export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=3

env OMP_NUM_THREADS=32 python -u ase_neb.py geoms/{R,P}.xyz > ase_neb.out
