#!/bin/bash

if [ "$1" = "train" ];
then
    CUDA_VISIBLE_DEVICES=0 nohup python train.py --project configs/take-pic/take-pic-m.yaml 1>log.log 2>err.log &
elif [ "$1" = "predict" ];
then
    if [ "$2" = "image" ];
    then
        python 1_predict_image.py --project configs/take-pic/take-pic-m.yaml --input-images /home/work/datasets/take-pic/yolo/images/val/ --output-images outputs --agnostic-nms --checkpoint checkpoints/take-pic-m/99_0.9466_0.6418_0.6723.pth
    elif [ "$2" = "video" ];
    then
        python 2_predict_video.py --project configs/take-pic/take-pic-m.yaml --input-video 192.168.1.64 --agnostic-nms --checkpoint checkpoints/take-pic-m/99_0.9466_0.6418_0.6723.pth --video-type camera --need-view --fps 1
    fi
fi
