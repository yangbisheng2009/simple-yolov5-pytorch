#!/bin/bash

if [ "$1" = "train" ];
then
    echo "No need to train."
elif [ "$1" = "predict" ];
then
    if [ "$2" = "image"];
    then
        python 1_predict_image.py --checkpoint checkpoints/coco/coco-m.pth --input-images /home/work/datasets/helmet-reflect/yolo/images/val --output-images outputs/ --agnostic-nms --project configs/person-body/person-m.yaml
    elif [ "$2" = "video" ];
    then
        python 2_predict_video.py --checkpoint checkpoints/coco/coco-m.pth --input-video 192.168.1.64 --video-type camera --need-view --agnostic-nms --project configs/person-body/person-m.yaml --fps 1
    fi
fi
