#!/bin/bash
export NCCL_P2P_DISABLE=1

day=$(date "+%Y%m%d")

for ((i=1;i<=16;i++))

do

# Testing

python SIFID/sifid_score.py --path2real images/places50 --path2fake outputs/places50/21.jpg/train_j$i

done
