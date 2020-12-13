#!/bin/bash

if [ "$1" = "train" ];
then
    CUDA_VISIBLE_DEVICES=0 nohup python train.py --project configs/mask/mask-m.yaml 1>log.log 2>err.log &
elif [ "$1" = "predict" ];
then
    python 1_predict_image.py --project configs/mask/mask-m.yaml --input-images /home/work/datasets/mask/yolo/images/val/ --output-images outputs --agnostic-nms --checkpoint checkpoints/mask-m/98_0.8924_0.6052_0.6339.pth
fi
