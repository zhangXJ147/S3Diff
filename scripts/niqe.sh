#!/bin/bash
export NCCL_P2P_DISABLE=1

day=$(date "+%Y%m%d")

for ((i=1;i<=16;i++))

do

# Testing

python  niqe-master/niqe.py -d outputs/places50/21.jpg/train_j$i -o ./places50_21_j1-j16_niqe_master.txt

done
