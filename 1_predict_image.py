import argparse
import os
import yaml
import shutil
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.yolo import Model
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='./checkpoints/yolov5s.pt', help='model.pt path(s)')
parser.add_argument('--input-images', type=str, default='./input-images/', help='input images dir')
parser.add_argument('--output-images', type=str, default='./output-images/', help='output iamges dir')
parser.add_argument('--data-cfg', type=str, default='', help='data config')
parser.add_argument('--model-cfg', type=str, default='', help='model config')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
args = parser.parse_args()
print(args)


def detect():
    with open(args.data_cfg) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
        names = data_dict['names']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Model(args.model_cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.float().fuse().eval()
    imgsz = check_img_size(args.img_size, s=model.stride.max())  # check args.img_size is illleagle

    if half:
        model.half()  # to FP16

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for f in os.listdir(args.input_images):
        path = os.path.join(args.input_images, f)
        im0s = cv2.imread(path)
        img = letterbox(im0s, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=args.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in det:
                    color = colors[int(cls)]
                    label = '%s %.2f' % (names[int(cls)], conf)
                    xmin, ymin, xmax, ymax = xyxy
                    cv2.rectangle(im0, (xmin, ymin), (xmax, ymax), color=color, thickness=2)
                    cv2.putText(im0, label, (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.imwrite(os.path.join(args.output_images, f), im0)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    with torch.no_grad():
        detect()
