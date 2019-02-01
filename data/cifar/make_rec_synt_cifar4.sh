#!/bin/bash

# usage:  ./make_rec_synt_cifar4.sh "/mnt/ebs2/data/cifar4/synt" "cifar4"

DATA_DIR=$1
train_list_name=$2
MX_DIR=/home/ubuntu/src/mxnet

# make list for all classes
python ${MX_DIR}/tools/im2rec.py --list True --exts '.png' --recursive True ${DATA_DIR}/${train_list_name} ${DATA_DIR}

# make .rec file for all classes
python ${MX_DIR}/tools/im2rec.py --exts '.png' --quality 95 --num-thread 16 --color 1 ${DATA_DIR}/${train_list_name} ${DATA_DIR}

# declare classes
declare -a CLASS_DIR_ARR=("bicycle" "car" "motorcycle" "train")

# make list and .rec files for each class
for ((i=0;i<${#CLASS_DIR_ARR[@]};++i))
do
    train_list_name_class=${train_list_name}_${CLASS_DIR_ARR[i]}
    DATA_DIR_CLASS=${DATA_DIR}/${CLASS_DIR_ARR[i]}/

    echo ${i}
    echo ${train_list_name_class}

    python im2rec-visda-for-each-class.py --list True --exts '.png' --recursive True --label_value ${i} ${DATA_DIR}/${train_list_name_class} ${DATA_DIR_CLASS}

    python ${MX_DIR}/tools/im2rec.py --exts '.png' --quality 95 --num-thread 16 --color 1 ${DATA_DIR}/${train_list_name_class} ${DATA_DIR_CLASS}
done