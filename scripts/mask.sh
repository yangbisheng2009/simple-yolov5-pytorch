#!/bin/bash

if [ "$1" = "train" ];
then
    CUDA_VISIBLE_DEVICES=0 nohup python train.py --project configs/mask/mask-m.yaml 1>log.log 2>err.log &
elif [ "$1" = "predict" ];
then
    python 1_predict_image.py --project configs/helmet-reflect/helmet-reflect.yaml --input-images /home/work/datasets/helmet-reflect/yolo/images/val/ --output-images outputs --agnostic-nms --checkpoint checkpoints/helmet-reflect-m/85_0.8936_0.6007_0.6300.pth
fi
