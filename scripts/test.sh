#!/bin/bash
export NCCL_P2P_DISABLE=1


# Testing

CUDA_VISIBLE_DEVICES=0 python sample.py --task image --image_name balloons.png --run_name version_1 --sample_count 50
