import argparse
import random
import cv2
import torch
from collections import defaultdict

from utils.datasets import *
from utils.utils import *
from models.yolo import Model
from utils.forward_util import forward_one


def detect():
    weights, imgsz, output = opt.checkpoint, opt.img_size, opt.output_video

    # Initialize
    device = torch_utils.select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    with open(opt.project) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    names = data_dict['names']
    dest_object = data_dict['dest_object'] if 'dest_object' in data_dict else names
    if 'name_map' in data_dict:
        name_map = data_dict['name_map']
    else:
        name_map = defaultdict()
        for x in dest_object:
            name_map[x] = x
    draw_names = set()
    for k, v in name_map.items():
        draw_names.add(v)
    draw_names = list(draw_names)
    colors = get_all_colors(len(draw_names))

    model = Model(data_dict).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()
    if half:
        model.half()

    if opt.video_type == 'camera':
        rtsp = 'rtsp://admin:Admin123@' + opt.input_video + ':554/h264/chCH/sub/av_stream'
        cap = cv2.VideoCapture(rtsp)
    elif opt.video_type == 'video':
        cap = cv2.VideoCapture(opt.input_video)
    else:
        print('Input video type ERROR!')
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps, width, height)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    outstream = cv2.VideoWriter(output, fourcc, fps, (width, height))

    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    while (cap.isOpened()):
        ret, im0s = cap.read()
        if ret:
            pred = forward_one(model, im0s, imgsz, device, half, opt)

            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    for *xyxy, conf, cls in det:
                        if names[int(cls)] in dest_object:
                            draw_str = name_map[names[int(cls)]]
                            color = colors[draw_names.index(draw_str)]
                            label = '%s %.2f' % (draw_str, conf)
                            xmin, ymin, xmax, ymax = xyxy
                            cv2.rectangle(im0s, (xmin, ymin), (xmax, ymax), color=color, thickness=1)
                            cv2.putText(im0s, label, (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            if opt.need_view:
                cv2.imshow('real-time', im0s)
                key = cv2.waitKey(delay=1)
                if key == ord('q'):
                    break
            else:
                outstream.write(im0s)

    cv2.destroyAllWindows()
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--project', '-p', type=str, default='configs/mhs_s.yaml', help='project')
    parser.add_argument('--input-video', type=str, default='192.168.1.64', help='source')
    parser.add_argument('--video-type', type=str, default='camera', help='camera or video')
    parser.add_argument('--need-view', action='store_true', help='need view')
    parser.add_argument('--output-video', type=str, default='./output-images', help='output images dir')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        detect()
