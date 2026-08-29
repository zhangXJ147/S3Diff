#!/bin/bash
export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=0 python main.py --task image --image_name balloons.png --run_name version_1
