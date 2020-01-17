"""
aioz.aiar.truongle - May 20. 2019
"""
import cv2
import os
import colorsys
import json
import numpy as np
from termcolor import colored
from PIL import Image
from sklearn.utils.linear_assignment_ import linear_assignment


def check_dir(directory, pr_flat=True):
    if os.path.isdir(directory):
        if pr_flat:
            print(colored("[INFO] {} is exist".format(directory), "cyan", attrs=['bold']))
        pass
    else:
        os.makedirs(directory)
        if pr_flat:
            print(colored("[INFO] created {}".format(directory), "cyan", attrs=['bold']))


def create_unique_color(tag, hue_step=0.41):
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    color = (int(255*r), int(255*g), int(255*b))
    return color


def draw_boxes_with_texts(img, boxes, texts, list_idx=None, color=(0, 255, 0)):
    """box format: [xmin, ymin, xmax, ymax]"""
    thickness = 2
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    for idx, bbox in enumerate(boxes):
        bbox = bbox[:4]
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        text = texts[idx]
        if list_idx is not None:
            color = create_unique_color(list_idx[idx])
        else:
            color = color
        # DRAW BOX
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness=thickness)
        # DRAW TEXT
        text_size = cv2.getTextSize(text, fontFace, fontScale, thickness)
        cv2.rectangle(img,
                      (xmin, ymin),
                      (xmin + 10 + text_size[0][0], ymin + 10 + text_size[0][1]),
                      color, -1)
        cv2.putText(img, text,
                    (xmin + 5, ymin + 5 + text_size[0][1]),
                    fontFace=fontFace,
                    fontScale=fontScale,
                    color=(255, 255, 255),
                    thickness=thickness)
    return img


def adjust_lightness(image, gamma=2.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def take_name(name_df, index, ext="jpg", len_max=6):
    len_idx = len(str(index))
    zeros = '0'*((len_max - len_idx) + 1)
    new_name = name_df + zeros + str(index) + '.' + ext
    return new_name


def multi_crop(image, boxes):
    """ boxes format [xmin, ymin, xmax, ymax] """
    list_crop = []
    if len(boxes) > 0:
        for bb in boxes:
            xmin, ymin, xmax, ymax = bb
            crop = image[ymin:ymax, xmin:xmax]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            list_crop.append(crop)
    return list_crop


def write_crop(write_dir, name, image, boxes):
    """ boxes format [xmin, ymin, xmax, ymax] """
    if len(boxes) > 0:
        for idx, bb in enumerate(boxes):
            xmin, ymin, xmax, ymax = bb
            crop = image[ymin:ymax, xmin:xmax]
            save_name = "0_" + str(name) + str(idx) + ".jpg"
            save_pth = os.path.join(write_dir, save_name)
            cv2.imwrite(save_pth, crop)


def iou(bbox, candidates):
    """[xmin, ymin, xmax, ymax]"""
    bbox_tl, bbox_br = bbox[:2], bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = (bbox[2:] - bbox[:2]).prod()
    area_candidates = (candidates[:, 2:] - candidates[:, :2]).prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def association(scores, threshold=0):
    matched_indices = linear_assignment(-scores)
    len_data = scores.shape[0]
    len_query = scores.shape[1]
    miss_ob = []
    for idm in range(len_data):
        if idm not in matched_indices[:, 0]:
            miss_ob.append(idm)
    new_ob = []
    for idn in range(len_query):
        if idn not in matched_indices[:, 1]:
            new_ob.append(idn)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if scores[m[0], m[1]] == 0:
            new_ob.append(m[1])
            miss_ob.append(m[0])
        else:
            matches.append(m)
        # matches.append(m)
    return np.asarray(matches), new_ob, miss_ob


def read_json(json_file):
    with open(json_file, "r") as fjs:
        data_json = json.load(fjs)
    return data_json


def read_txt(txt_file):
    with open(txt_file, "r") as f_txt:
        data_txt = np.loadtxt(f_txt, dtype=int, delimiter=',')
    return data_txt


def tlbr2p(boxes):
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, axis=0)
    point = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        xp = int(xmin + (xmax - xmin) / 2)
        yp = int(ymin + (ymax - ymin) / 2)
        point.append([xp, yp])
    return np.asarray(point).astype(int)


def tlbr2cb(box):
    xmin, ymin, xmax, ymax = box
    xp = int(xmin + (xmax - xmin) / 2)
    yp = int(ymax)
    return xp, yp


def check_area(box, areas):
    if np.ndim(areas) == 1:
        areas = np.expand_dims(areas, axis=0)
    xp, yp = tlbr2cb(box)
    for idx, ar in enumerate(areas):
        x1, y1, x2, y2 = ar
        # print("box: ", box)
        # print("ar: ", x1, y1, x2, y2)
        # print("xpyp: ", xp, yp)
        if (xp > x1) and (yp > y1) and (xp < x2) and (yp < y2):
            return True
        else:
            return False


def cut_video(video_in, video_out, start, stop, step=1, codex='MJPG', fps=None):
    # read video and take info
    cap = cv2.VideoCapture(video_in)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_ori = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        fps_write = fps_ori
    else:
        fps_write = fps
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    # set range
    start_frame = start * fps_ori
    stop_frame = stop * fps_ori
    # setup writer
    fourcc = cv2.VideoWriter_fourcc(*codex)
    writer = cv2.VideoWriter(video_out, fourcc, fps_write, (w, h))
    while cap.isOpened() and count < total_frame:
        ret, frame = cap.read()
        if ret:
            if (count > start_frame) and (count < stop_frame) and (count % step == 0):
                writer.write(frame)
            if count > stop_frame:
                break
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
