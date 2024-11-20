compute_rate () {
    cd $1
    for i in {1..11}
    do
        cd $i
        b=`grep BARRIER plumed.dat | cut -d= -f2`
        python ../../../../rescale_time_wall.py $b
        tail -n 1 time_wall
        cd ../
    done > time_wall
    python ../../../estimate_error.py time_wall > estimate_error.out
    cd ../../../
}

compute_rate pbe/1hb_arg90/small_qm
compute_rate pbe/2hb_arg90/small_qm
compute_rate pbe/2hb_arg90/large_qm
compute_rate wb97x-3c/1hb_arg90/small_qm
compute_rate wb97x-3c/2hb_arg90/small_qm
compute_rate wb97x-3c/2hb_arg90/large_qm
compute_rate rev_wb97x-3c/1hb_arg90/small_qm
compute_rate rev_wb97x-3c/2hb_arg90/small_qm
compute_rate rev_wb97x-3c/2hb_arg90/large_qm
