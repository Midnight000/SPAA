#!/bin/bash

ALGORITHM="repaint"
BASE_OUTDIR="/mnt/data/huang-lab/shipeng/imageNet/"
DATASETS="imagenet"
MASKTYPES="20 40 60 half ex64"

# CoPaint
for dataset in $DATASETS
do
    for mask in $MASKTYPES
    do
        COMMON="--dataset_name ${dataset} --n_samples 1 --config_file configs/${dataset}.yaml --device 3"
        OUT_PATH=${BASE_OUTDIR}/${ALGORITHM}/${dataset}/${mask}/
        python main.py $COMMON --outdir $OUT_PATH --mask_type $mask --algorithm $ALGORITHM
    done
done
