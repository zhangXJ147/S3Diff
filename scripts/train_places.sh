#!/bin/bash
export NCCL_P2P_DISABLE=1

for ((i=1;i<=50;i++))

do

CUDA_VISIBLE_DEVICES=0 python main.py --task image --image_name places50/$i.jpg --run_name train_j8

done