#!/bin/bash
# argument list:
#    run:           The run number. This is the index to differentiate betwee runs. Pass in as "runX", where X is the index.
#    num_iter:      The number of iterations to split the data, train, and test the models
#    num_aug:       The number of augmented data points to create for each data point in the data set
#    data_impute:   Boolean flag to indicate whether data imputation should be used

run=$1
num_iter=$2
num_aug=$3
data_impute=$4

mkdir results/$run

python3 rfc.py $run $num_iter $num_aug $data_impute > results/$run/rfc.out
python3 log.py $run $num_iter $num_aug $data_impute > results/$run/log.out
python3 knn.py $run $num_iter $num_aug $data_impute > results/$run/knn.out
python3 dt-boost.py $run $num_iter $num_aug $data_impute > results/$run/dt-boost.out
python3 stack.py $run $num_iter $num_aug $data_impute > results/$run/stack.out
#python3 res.py $run $num_iter $num_aug $data_impute > results/$run/res.out

echo $run. number of iterations: $num_iter. data augmentation: $num_aug. data imputation: $data_impute > results/$run/info.txt
