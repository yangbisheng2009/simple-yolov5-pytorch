#CUDA_VISIBLE_DEVICES=0 python 1_predict_image.py --checkpoint checkpoints/yolov5s.pt --output-images output-images/
CUDA_VISIBLE_DEVICES=0 python 1_predict_image.py --checkpoint checkpoints/coco/coco-s.pth --output-images output-images/ --input-images ./input-images --model-cfg configs/model-cfg/yolov5s.yaml --data-cfg configs/data-cfg/coco.yaml --agnostic-nms
