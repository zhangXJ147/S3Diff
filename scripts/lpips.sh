#!/bin/bash
export NCCL_P2P_DISABLE=1

day=$(date "+%Y%m%d")

for ((i=0;i<=49;i++))

do

# Testing

python LPIPS/compute_dists_pair.py -d outputs/imagenet50/0.98_0.05_0.05/$i.jpg -o ./imagenet50_0.98_0.05_0.05_lpips.txt --all-pairs -N 10

done
