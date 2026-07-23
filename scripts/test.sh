#!/bin/bash
export NCCL_P2P_DISABLE=1


# Testing

#CUDA_VISIBLE_DEVICES=2 python sample.py --task image --image_name balloons.png --run_name version_1 --sample_count 50

CUDA_VISIBLE_DEVICES=1 python sample.py --task image --image_name lsun50/42.jpg --run_name train_lsun50_42 --sample_count 50


#for ((i=1;i<=16;i++))
#
#do
#
#CUDA_VISIBLE_DEVICES=3 python sample.py --task image --image_name sigd16/$i.jpg --run_name train_j8_layer3 --sample_count 10
#
#
#done
#
#for ((i=1;i<=50;i++))
#
#do
#
#CUDA_VISIBLE_DEVICES=3 python sample.py --task image --image_name places50/$i.jpg --run_name train_j8_layer3 --sample_count 10
#
#
#done
#
#for ((i=0;i<=49;i++))
#
#do
#
#CUDA_VISIBLE_DEVICES=3 python sample.py --task image --image_name lsun50/$i.jpg --run_name train_j8_layer3 --sample_count 10
#
#
#done

#for ((i=0;i<=49;i++))
#
#do
#
#CUDA_VISIBLE_DEVICES=3 python sample.py --task image --image_name imagenet50/$i.jpg --run_name train_j8 --sample_count 10
#
#done

#for ((i=1;i<=16;i++))
#
#do
#
#CUDA_VISIBLE_DEVICES=6 python sample.py --task image --image_name sigd16/12.jpg --run_name train_j8 --sample_count 10
#
#
#done
#
#for ((i=1;i<=50;i++))
#
#do
#
#CUDA_VISIBLE_DEVICES=3 python sample.py --task image --image_name places50/$i.jpg --run_name train_j8 --sample_count 1
#
#
#done
#
#for ((i=0;i<=49;i++))
#
#do
#
#CUDA_VISIBLE_DEVICES=3 python sample.py --task image --image_name lsun50/$i.jpg --run_name train_j8 --sample_count 1
#
#
#done
#
#for ((i=0;i<=49;i++))
#
#do
#
#CUDA_VISIBLE_DEVICES=3 python sample.py --task image --image_name imagenet50/$i.jpg --run_name train_j8 --sample_count 1
#
#
#done
