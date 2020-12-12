import cv2
import torch
import argparse
import random

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *
from models.yolo import Model
from utils.forward_util import forward_one


def detect():
    weights, imgsz = opt.checkpoint, opt.img_size

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(opt.output_images):
        shutil.rmtree(opt.output_images)  # delete output folder
    os.makedirs(opt.output_images)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    with open(opt.project) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    names = data_dict['names']
    colors = get_all_colors(len(names))

    model = Model(data_dict).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for f in os.listdir(opt.input_images):
        t1 = time.time()
        path = os.path.join(opt.input_images, f)
        im0s = cv2.imread(path)

        pred = forward_one(model, im0s, imgsz, device, half, opt)

        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    xmin, ymin, xmax, ymax = xyxy
                    color = colors[int(cls)]
                    cv2.rectangle(im0s, (xmin, ymin), (xmax, ymax), color=color, thickness=1)
                    cv2.putText(im0s, label, (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.imwrite(os.path.join(opt.output_images, f), im0s)
        print('%s Done. (%.3fs)' % (f, time.time() - t1))

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--project', '-p', type=str, default='configs/mhs_s.yaml', help='project')
    parser.add_argument('--input-images', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output-images', type=str, default='./output-images', help='output images dir')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        detect()
