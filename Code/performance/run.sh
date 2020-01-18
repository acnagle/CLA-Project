#!/bin/bash
# argument list:
#    run:       The run number. This is the index to differentiate betwee runs. Pass in as "runX", where X is the index.
#    num_iter:  The number of iterations to split the data, train, and test the models

run=$1
num_iter=$2

mkdir results/$run

python3 rfc.py $run $num_iter > results/$run/rfc.out
python3 log.py $run $num_iter > results/$run/log.out
python3 knn.py $run $num_iter > results/$run/knn.out
python3 dt-boost.py $run $num_iter > results/$run/dt-boost.out
python3 stack.py $run $num_iter > results/$run/stack.out
python3 res.py $run $num_iter > results/$run/res.out


