#!/bin/bash
export NCCL_P2P_DISABLE=1

day=$(date "+%Y%m%d")

for ((i=0;i<=49;i++))

do

# Testing

python LPIPS/compute_dists_pair.py -d outputs/imagenet50/0.98_0.05_0.05/$i.jpg/train_j8_layer4 -o ./imagenet50_0.98_0.05_0.05_layer4_lpips.txt --all-pairs

done

#python LPIPS/compute_dists_pair.py -d outputs/test/2.jpg/train_test --all-pairs -N 10

#for ((i=0;i<=49;i++))
#
#do
#
## Testing
#
#python LPIPS/compute_dists_pair.py -d outputs/lsun50/0.98_0.05_0.05/$i.jpg/train_j8_layer4 -o ./lsun50_0.98_0.05_0.05_layer4_lpips.txt --all-pairs
#
#done
#
#for ((i=1;i<=50;i++))
#
#do
#
## Testing
#
#python LPIPS/compute_dists_pair.py -d outputs/places50/0.98_0.05_0.05/$i.jpg/train_j8_layer4 -o ./places50_0.98_0.05_0.05_layer4_lpips.txt --all-pairs
#
#done
#
#for ((i=1;i<=16;i++))
#
#do
#
## Testing
#
#python LPIPS/compute_dists_pair.py -d outputs/sigd16/0.98_0.05_0.05/$i.jpg/train_j8_layer4 -o ./sigd16_0.98_0.05_0.05_layer4_lpips.txt --all-pairs
#
#done
