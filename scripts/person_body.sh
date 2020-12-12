#!/bin/bash

if [ "$1" = "train" ];
then
    echo "No need to train."
elif [ "$1" = "predict" ];
then
    python 1_predict_image.py --checkpoint checkpoints/coco-m/coco-m.pth --input-images /home/work/datasets/helmet-reflect/yolo/images/val --output-images outputs/ --agnostic-nms --project configs/person-body/person-m.yaml
fi
