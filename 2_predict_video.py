import argparse
import random
import cv2
import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *
from models.yolo import Model


def split_image(img, cell_width, cell_height, mindup_width, mindup_height):
    vertical_long_piece = []
    all_piece = []
    all_piece_loc = []
    all_piece_xy = []  # 最终每一个切片在原图的绝对坐标
    H, W, C = img.shape
    # print('height: {}, width: {}, channel: {}'.format(H, W, C))

    # 1.判断是否需要切片
    if W <= cell_width and H <= cell_height:
        all_piece.append(img)
        return vertical_long_piece, all_piece

    # 2. split image by X => get vertical long piece
    elif W > cell_width and H > cell_height:
        X_cell_cnt = math.ceil((W - cell_width) / (cell_width - mindup_width)) + 1
        vertical_xy = []  # 每一个垂直长条在原图的绝对坐标
        for cursor in range(X_cell_cnt):
            if cursor == 0:  # 第一张图
                vertical_long_piece.append(img[:, 0:cell_width])
                vertical_xy.append((0, 0, cell_width, H))
            elif cursor == X_cell_cnt - 1:  # 最后一张图
                vertical_long_piece.append(img[:, W - cell_width:W])
                vertical_xy.append((W - cell_width, 0, W, H))
            else:  # 中间图
                st = (cell_width - mindup_width) * cursor
                vertical_long_piece.append(img[:, st:st + cell_width])
                vertical_xy.append((st, 0, st + cell_width, H))

        # get all piece
        for i, _ in enumerate(vertical_long_piece):

            vp, vxy = split_Y(_, cell_width, cell_height, mindup_height)
            all_piece.extend(vp)
            all_piece_loc.extend([str(i) + '_' + str(j) for j in range(len(vp))])
            for xmin, ymin, xmax, ymax in vxy:
                xmin_long = vertical_xy[i][0]
                ymin_long = vertical_xy[i][1]
                all_piece_xy.append([xmin + xmin_long, ymin + ymin_long,
                                     xmax + xmin_long, ymax + ymin_long])

    return vertical_long_piece, all_piece, all_piece_loc, all_piece_xy


def joint_image(H, W, C, all_piece, all_piece_xy):
    img = np.zeros((H, W, C), dtype=np.uint8)
    for index, piece in enumerate(all_piece):
        xmin = all_piece_xy[index][0]
        ymin = all_piece_xy[index][1]
        xmax = all_piece_xy[index][2]
        ymax = all_piece_xy[index][3]
        # print(xmin, ymin, xmax, ymax)
        img[ymin:ymax, xmin:xmax] = piece
    return img


def split_X(img, ret_img_lst, cell_width, cell_height, mindup_width):
    H, W, C = img.shape
    if H > cell_height or W <= cell_width:
        return 'Error: W is smaller than cell width ' \
               'or H is bigger than cell height.'

    cell_cnt = math.ceil((W - cell_width) / (cell_width - mindup_width)) + 1

    for cursor in range(cell_cnt):
        if cursor == 0:  # 第一张图
            ret_img_lst.append(img[0:H, 0:cell_width])
        elif cursor == cell_cnt - 1:  # 最后一张图
            ret_img_lst.append(img[0:H, W - cell_width:W])
        else:  # 中间图
            st = (cell_width - mindup_width) * cursor
            ret_img_lst.append(img[0:H, st:st + cell_width])


def split_Y(img, cell_width, cell_height, mindup_height):
    ret_img_lst = []
    ret_xy_lst = []
    H, W, C = img.shape
    if H <= cell_height or W > cell_width:
        return 'Error: H is smaller than cell height' \
               'or W is bigger than cell width'

    cell_cnt = math.ceil((H - cell_height) / (cell_height - mindup_height)) + 1

    for cursor in range(cell_cnt):
        if cursor == 0:  # 第一张图
            ret_img_lst.append(img[0:cell_height, 0:W])
            ret_xy_lst.append((0, 0, W, cell_height))
        elif cursor == cell_cnt - 1:  # 最后一张图
            ret_img_lst.append(img[H - cell_height:H, 0:W])
            ret_xy_lst.append((0, H - cell_height, W, H))
        else:  # 中间图
            st = (cell_height - mindup_height) * cursor
            ret_img_lst.append(img[st:st + cell_height, 0:W])
            ret_xy_lst.append((0, st, W, st + cell_height))

    return ret_img_lst, ret_xy_lst


def get_all_colors(class_num, seed=1):
    color_pool = [[0, 0, 255], [0, 255, 0], [51, 253, 253], [207, 56, 248], [255, 0, 0]]
    class_colors = {}
    random.seed(seed)
    for cls in range(class_num):
        if cls < len(color_pool):
            class_colors[cls] = color_pool[cls]
        else:
            class_colors[cls] = [random.randint(0, 255) for _ in range(3)]

    return class_colors


def detect():
    source, weights, view_img, save_txt, imgsz, output = \
        opt.input_video, opt.checkpoint, opt.view_img, opt.save_txt, opt.img_size, opt.output_video
    # webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    '''
    if os.path.exists(opt.output_images):
        shutil.rmtree(opt.output_images)  # delete output folder
    os.makedirs(opt.output_images)  # make new output folder
    '''
    half = device.type != 'cpu'  # half precision only supported on CUDA

    with open(opt.project) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    model = Model(data_dict).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.names = data_dict['names']

    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = get_all_colors(len(names))

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # for path, img, im0s, vid_cap in dataset:

    cap = cv2.VideoCapture(source)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps, width, height)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # for f in os.listdir(source):
    while cap.isOpened():
        t1 = time.time()
        ret, im0s = cap.read()

        H, W, C = im0s.shape
        vertical_long_piece, all_piece, all_piece_loc, all_piece_xy = \
            split_image(im0s, opt.cell_width, opt.cell_height, opt.mindup_width, opt.mindup_height)
        all_piece_new = []
        for im0s in all_piece:
            img = letterbox(im0s, new_shape=imgsz)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            # t1 = torch_utils.time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s, im0 = '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        xmin, ymin, xmax, ymax = xyxy
                        color = colors[int(cls)]
                        cv2.rectangle(im0, (xmin, ymin), (xmax, ymax), color=color, thickness=1)
                        cv2.putText(im0, label, (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    # cv2.imwrite('abc/test.jpg', im0)

            # out.write(im0)
            all_piece_new.append(im0)
        img_joint = joint_image(H, W, C, all_piece_new, all_piece_xy)
        # cv2.imwrite('abc/testtest.jpg', img_joint)
        out.write(img_joint)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--project', '-p', type=str, default='configs/mhs_s.yaml', help='project')
    parser.add_argument('--input-video', type=str, default='inference/images',
                        help='source')  # file/folder, 0 for webcam
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

    parser.add_argument('--input-images', type=str, default='input-images', help='inputs')
    parser.add_argument('--output-images', type=str, default='output-images', help='outputs')
    parser.add_argument('--cell-width', type=int, default=640, help='')
    parser.add_argument('--cell-height', type=int, default=640, help='')
    parser.add_argument('--mindup-width', type=int, default=10, help='')
    parser.add_argument('--mindup-height', type=int, default=10, help='')

    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        detect()
