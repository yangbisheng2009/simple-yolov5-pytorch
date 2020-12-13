#!/bin/bash

if [ "$1" = "train" ];
then
    echo "No need to train."
elif [ "$1" = "predict" ];
then
    if [ "$2" = "image"];
    then
        python 1_predict_image.py --checkpoint checkpoints/helmet-reflect-m/85_0.8936_0.6007_0.6300.pth --project configs/head-count/head-count-m.yaml --input-images /home/work/datasets/helmet-reflect/yolo/images/val --output-images outputs/ --agnostic-nms
    elif [ "$2" = "video" ];
    then
        python 2_predict_video.py --checkpoint checkpoints/helmet-reflect-m/85_0.8936_0.6007_0.6300.pth --project configs/head-count/head-count-m.yaml --input-video 192.168.1.64 --agnostic-nms --video-type camera --need-view --fps 1
    fi
fi
