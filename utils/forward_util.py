import cv2
import torch
import numpy as np

from utils.datasets import letterbox
from utils.utils import non_max_suppression, scale_coords
from utils import torch_utils

"""
bgr_mat: just like cv2.imread('a.jpg')
"""
def forward_one(model, bgr_mat, checked_imgsz, device, half, opt):
    img = letterbox(bgr_mat, new_shape=checked_imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = torch_utils.time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

    return pred


def forward_batch():
    pass


if __name__ == '__main__':
    forward_one()