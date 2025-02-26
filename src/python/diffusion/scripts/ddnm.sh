#!/bin/bash

ALGORITHM="ddnm"
BASE_OUTDIR="/mnt/data/huang-lab/shipeng/imageNet_1000/"
DATASETS="imagenet"
MASKTYPES="half"

# CoPaint
for dataset in $DATASETS
do
    for mask in $MASKTYPES
    do
        COMMON="--dataset_name ${dataset} --n_samples 1 --config_file configs/${dataset}.yaml --device 0 --ddnm.schedule_jump_params.t_T 1000 --timestep_respacing 1000"
        OUT_PATH=${BASE_OUTDIR}/${ALGORITHM}/${mask}/
        python main.py $COMMON --outdir $OUT_PATH --mask_type $mask --algorithm $ALGORITHM
    done
done
