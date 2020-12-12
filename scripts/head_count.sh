#!/bin/bash

if [ "$1" = "train" ];
then
    echo "No need to train."
elif [ "$1" = "predict" ];
then
    python 1_predict_image.py --checkpoint checkpoints/helmet-reflect-m/85_0.8936_0.6007_0.6300.pth --project configs/head-count/head-count-m.yaml --input-images /home/work/datasets/helmet-reflect/yolo/images/val --output-images outputs/ --agnostic-nms
fi
