#!/bin/bash
export NCCL_P2P_DISABLE=1

#CUDA_VISIBLE_DEVICES=1 python main.py --task image --image_name balloons.png --run_name version_1
CUDA_VISIBLE_DEVICES=1 python main.py --task image --image_name lsun50/42.jpg --run_name train_lsun50_42
