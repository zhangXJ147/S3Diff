#!/bin/bash
export NCCL_P2P_DISABLE=1

day=$(date "+%Y%m%d")

for ((i=1;i<=16;i++))

do

# Testing

python  nnf_diversity-main/nnf_evaluation.py --path2real images/imagenet50_42 --path2fake outputs/imagenet50/42.jpg/train_j$i -o ./imagenet50_42_j1-j16_nnfdiv.txt

done

for ((i=1;i<=16;i++))

do

# Testing

python  nnf_diversity-main/nnf_evaluation.py --path2real images/lsun50_42 --path2fake outputs/lsun50/42.jpg/train_j$i -o ./lsun50_42_j1-j16_nnfdiv.txt

done

for ((i=1;i<=16;i++))

do

# Testing

python  nnf_diversity-main/nnf_evaluation.py --path2real images/places50_21 --path2fake outputs/places50/21.jpg/train_j$i -o ./places50_21_j1-j16_nnfdiv.txt

done

for ((i=1;i<=16;i++))

do

# Testing

python  nnf_diversity-main/nnf_evaluation.py --path2real images/sigd16_12 --path2fake outputs/sigd16/12.jpg/train_j$i -o ./sigd16_12_j1-j16_nnfdiv.txt

done

