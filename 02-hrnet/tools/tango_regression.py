import os
import cv2
import time
import random
import argparse
import requests
import subprocess
import numpy as np
import _init_paths
from PIL import Image
import onnxruntime as ort
from tango_keypoints import *

names = ['soyuz']
# names = ['SwissCube']
# names = ['Tango']

np.set_printoptions(suppress=True)

class ONNX_engine():
    def __init__(self, weights, size, cuda) -> None:
        self.img_new_shape = (size, size)
        self.weights = weights
        self.device = cuda
        self.init_engine()
        self.names = names
        self.colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(self.names)}

    def init_engine(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.weights, providers=providers)

    def predict(self, im):
        outname = [i.name for i in self.session.get_outputs()]
        inname = [i.name for i in self.session.get_inputs()]
        inp = {inname[0]: im}
        outputs = self.session.run(outname, inp)[0]
        return outputs

    def max_bbox(self, bbox_all):
        bbox_tem = []
        for index, value in enumerate(bbox_all):
            # print(index, value[0][0][0], value[0][0][1], value[0][1][0], value[0][1][1])
            area = (value[0][1][0] - value[0][0][0]) * (value[0][1][1] - value[0][0][1])
            bbox_tem.append(area)
        # for iter, val in enumerate(bbox_tem):
        #     print(iter, val)

        max_b = max(bbox_tem)
        iter_max = bbox_tem.index(max_b)
        bbox_max = bbox_all[iter_max]

        # print(max_b, iter_max, bbox_max)

        return bbox_max

    def preprocess(self, src):
        self.img = src
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        image = self.img.copy()
        im, ratio, dwdh = self.letterbox(image, auto=False)
        # t1 = time.time()
        outputs = self.predict(im)
        # print("inference time", (time.time() - t1) * 1000, ' ms')
        ori_images = [self.img.copy()]
        bbox = [[(0, 0), (0, 0)]]
        bbox_all = []
        name = None
        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
            image = ori_images[int(batch_id)]
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score), 3)
            name = self.names[cls_id]
            color = self.colors[name]
            # name += ' ' + str(score)
            bbox = [[(box[0], box[1]), (box[2], box[3])]]
            # cv2.rectangle(image, box[:2], box[2:], color, 10)
            cv2.rectangle(image, box[:2], box[2:], [0, 0, 255], 10)
            # cv2.putText(image, name, (box[0], box[1] - 2), cv.FONT_HERSHEY_TRIPLEX, 4, [225, 255, 255], thickness=4)
            bbox_all.append(bbox)
        # bbox_max = self.max_bbox(bbox_all)
        # bbox = [[(-120, -120), (537, 445)]]
        return ori_images[0], bbox, name, image

    # def preprocess(self, src):
    #     self.img = src
    #     self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
    #     image = self.img.copy()
    #     im, ratio, dwdh = self.letterbox(image, auto=False)
    #     # t1 = time.time()
    #     outputs = self.predict(im)
    #     # print("inference time", (time.time() - t1) * 1000, ' ms')
    #     ori_images = [self.img.copy()]
    #     bbox = [[(0, 0), (0, 0)]]
    #     name = None
    #     for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
    #         image = ori_images[int(batch_id)]
    #         box = np.array([x0, y0, x1, y1])
    #         box -= np.array(dwdh * 2)
    #         box /= ratio
    #         box = box.round().astype(np.int32).tolist()
    #         cls_id = int(cls_id)
    #         score = round(float(score), 3)
    #         name = self.names[cls_id]
    #         color = self.colors[name]
    #         # name += ' ' + str(score)
    #         bbox = [[(box[0], box[1]), (box[2], box[3])]]
    #         # cv2.rectangle(image, box[:2], box[2:], color, 2)
    #         # cv2.putText(image, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=1)
    #
    #     return ori_images[0], bbox, name, image

    def letterbox(self, im, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        shape = im.shape[:2]
        new_shape = self.img_new_shape

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        # divide padding into 2 sides
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im)
        im = im.astype(np.float32)
        im /= 255
        return im, r, (dw, dh)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', "--i", help="images files path", type=str, required=True)
    parser.add_argument('--save_pth', "--s", help="save path", type=str, required=True)
    parser.add_argument('--save_pts', "--p", help="save pts2d", type=str, required=True)

    parser.add_argument('--yolov7', type=str, default='./tools/pth/det.onnx', help='weights path of onnx')
    parser.add_argument('--cfg', type=str, default='./tools/pth/kptn.yaml')
    parser.add_argument('--tango', type=str, default='./tools/pth/kptn.pth')

    parser.add_argument('--cuda', type=bool, default=True, help='if your pc have cuda')
    parser.add_argument('--size', type=int, default=640, help='infer the img size')

    parser.add_argument('opts', help='Modify config options using the command-line', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    return args


# python tools\tango_regression.py --i tools\01-lightbox\img --s tools\01-lightbox\save --p tools\01-lightbox\pts

if __name__ == '__main__':
    opt = parse_args()
    tango_model = init_keypoints(opt)
    onnx_engine = ONNX_engine(opt.yolov7, opt.size, opt.cuda)

    imgs_root = opt.img_path
    imgs_save = opt.save_pth
    pts_root = opt.save_pts

    image = None

    start_t = time.time()

    if not os.path.exists(imgs_save):
        os.mkdir(imgs_save)
    if not os.path.exists(pts_root):
        os.mkdir(pts_root)
    if not os.path.exists(imgs_root):
        print('no exist the file')
    else:
        src_list = os.listdir(imgs_root)
        for file_index in src_list:
            if os.path.splitext(file_index)[-1] == '.png' or os.path.splitext(file_index)[-1] == '.jpg':
                img_pth = os.path.join(imgs_root, file_index)
                sav_pth = os.path.join(imgs_save, file_index)
                image = cv2.imread(img_pth)
                detect_img, box, class_name, img_det = onnx_engine.preprocess(image)
                data_preds, img_bgr = pose_landmark(box, tango_model, image, image)

                if data_preds.all() != np.zeros([8, 3], dtype=float).all():
                    save_path = os.path.join(pts_root, file_index.split('.')[0] + '.json')
                    np.savetxt(save_path, data_preds, fmt='%0.8f', delimiter='\t')

                cv2.imwrite(sav_pth, img_bgr)
                #cv2.imwrite(sav_pth, detect_img)
                print('save img:\t', sav_pth)
    end_t = time.time()
    runtime = end_t - start_t
    print('run time/ms:\t', runtime * 1000)

    print('done')

