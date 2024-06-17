
# HRnet module (must be first)
import time

import models
import _init_paths
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

# PyTorch module
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# image module
import cv2

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# HRnet module

# # Cube
# SKELETON = [[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 5],
#             [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [4, 7]]
#
# CocoColors = [[0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [255, 255, 0], [255, 255, 0],
#               [255, 255, 0], [255, 255, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]]
#
# PointColor = [0, 0, 255]
#
# NUM_KPTS = 8


# speedplus
SKELETON = [[0, 3], [1, 4], [2, 5],
            [6, 7], [7, 8], [8, 9], [6, 9],
            [10, 11], [11, 12], [12, 13], [10, 13],
            [14, 15], [15, 16], [16, 17], [14, 17],
            [10, 14], [11, 15], [12, 16], [13, 17]]

CocoColors = [[0, 255, 0], [0, 255, 0], [0, 255, 0],
              [0, 255, 255], [255, 0, 255], [255, 255, 0], [125, 125, 125],
              [0, 125, 125], [125, 0, 125], [125, 125, 0], [125, 200, 125],
              [0, 175, 200], [175, 0, 200], [175, 200, 0], [125, 175, 200],
              [75, 125, 195], [195, 125, 75], [125, 75, 195], [125, 195, 225]]

PointColor = [0, 0, 255]

NUM_KPTS = 18

height, width, channels = 1200, 1920, 3
# height, width, channels = 1024, 1024, 3
# height, width, channels = 960, 1280, 3

# HRnet module
def draw_pose(keypoints,img):
    assert keypoints.shape == (NUM_KPTS, 2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 5)
        cv2.circle(img, (int(x_a), int(y_a)), 8, PointColor, -1)
        cv2.circle(img, (int(x_b), int(y_b)), 8, PointColor, -1)
        # cv2.circle(img, (int(x_a), int(y_a)), 20, CocoColors[i], -1)
        # cv2.circle(img, (int(x_b), int(y_b)), 20, CocoColors[i], -1)

# HRnet module
def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    model_input = transform(model_input).unsqueeze(0)

    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        output = pose_model(model_input)
        preds, maxvals = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds, maxvals

# HRnet module
def box_to_center_scale(box, model_image_width, model_image_height):
    center = np.zeros((2), dtype=np.float32)
    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25
    return center, scale

# HRnet module
def pose_landmark(box_detection, pose_model, img_bgr, img_src):
    empty_boxes = [[(0, 0), (0, 0)]]
    # image_bgr = cv2.imread(img_path)
    start = time.time()
    image = img_bgr[:, :, [2, 1, 0]]

    result = box_detection
    input = []
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(CTX)
    input.append(img_tensor)
    data_preds = []

    if result == empty_boxes:
        data_preds = np.zeros([8, 3], dtype=float)

    else:
        pred_boxes = box_detection
        if len(pred_boxes) >= 1:
            for box in pred_boxes:
                center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                image_pose = image.copy() if cfg.DATASET.COLOR_RGB else img_bgr.copy()
                pose_preds, pose_maxvals = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                pose_preds_ = np.squeeze(pose_preds)
                pose_maxvals_ = np.squeeze(pose_maxvals)
                data_preds = np.c_[pose_preds_, pose_maxvals_]

                if len(pose_preds) >= 1:
                    for kpt in pose_preds:
                        draw_pose(kpt, img_src)  # draw the poses
        fps = 1 / (time.time() - start)
        img = cv2.putText(img_src, 'fps: ' + "%.2f" % (fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                          (0, 255, 0), 2)

    return data_preds, img_src


def init_keypoints(args):

    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    update_config(cfg, args)

    tango_checkpoint = args.tango

    tango_model = eval('models.' + 'pose_hrnet' + '.get_pose_net')(cfg, is_train=False)

    tango_model.load_state_dict(torch.load(tango_checkpoint), strict=False)

    tango_model = torch.nn.DataParallel(tango_model, device_ids=cfg.GPUS)

    tango_model.to(CTX)

    return tango_model