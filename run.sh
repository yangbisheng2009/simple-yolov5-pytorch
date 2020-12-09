#!/bin/bash

if [ "$1" = "train" ];
then
    CUDA_VISIBLE_DEVICES=0 nohup python train.py --data-cfg configs/data-cfg/helmet-reflect.yaml --model-cfg configs/model-cfg/yolov5m.yaml 1>log.log 2>err.log &
elif [ "$1" = "predict" ];
then
    CUDA_VISIBLE_DEVICES=0 python 1_predict_image.py --checkpoint checkpoints/coco/coco-s.pth --output-images output-images/ --input-images ./input-images --model-cfg configs/model-cfg/yolov5s.yaml --data-cfg configs/data-cfg/coco.yaml --agnostic-nms
fi

