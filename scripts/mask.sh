#!/bin/bash

if [ "$1" = "train" ];
then
    CUDA_VISIBLE_DEVICES=0 nohup python train.py --project configs/mask/mask-m.yaml 1>log.log 2>err.log &
elif [ "$1" = "predict" ];
then
    if [ "$2" = "image" ];
    then
        python 1_predict_image.py --project configs/mask/mask-m.yaml --input-images /home/work/datasets/mask/yolo/images/val/ --output-images outputs --agnostic-nms --checkpoint checkpoints/mask-m/98_0.8924_0.6052_0.6339.pth
    elif [ "$2" = "video" ];
    then
        python 2_predict_video.py --checkpoint checkpoints/mask-m/98_0.8924_0.6052_0.6339.pth --input-video 192.168.1.64 --video-type camera --need-view --agnostic-nms --project configs/mask/mask-m.yaml --fps 1
    fi
fi
