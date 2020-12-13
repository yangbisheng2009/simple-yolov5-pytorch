#!/bin/bash

if [ "$1" = "train" ];
then
    CUDA_VISIBLE_DEVICES=0 nohup python train.py --project configs/car-count/car-count-m.yaml 1>log.log 2>err.log &
elif [ "$1" = "predict" ];
then
    if [ "$2" = "image" ];
    then
        python 1_predict_image.py --project configs/car-count/car-count-m.yaml --input-images /home/work/datasets/car-count/yolo/images/val/ --output-images outputs --agnostic-nms --checkpoint checkpoints/car-count-m/54_0.9694_0.6940_0.7215.pth
    elif [ "$2" = "video" ];
    then
        python 2_predict_video.py --project configs/car-count/car-count-m.yaml --input-video 192.168.1.64 --agnostic-nms --checkpoint checkpoints/car-count-m/54_0.9694_0.6940_0.7215.pth --video-type camera --need-view --fps 1
    fi
fi
