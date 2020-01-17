"""
truong.le - Oct 23, 2019
demo
"""
import os
import cv2
import argparse
from tqdm import tqdm
from utils import utils
from utils import yolo_utils
from src.detection_yolo3_wrapper import YoloV3Detection

parser = argparse.ArgumentParser(description="demo")
parser.add_argument("--mode", type=str, default="video",
                    help="mode: video, images")
parser.add_argument("--video_pth", type=str,
                    help='video path for process')
parser.add_argument("--im_dir", type=str,
                    help="directory store images")
parser.add_argument("--out_dir", type=str, default="data/demo/result",
                    help='directory store result')
parser.add_argument("--graph_path", type=str, default='yolov3_object_detection.pb',
                    help="pb file path")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
args = parser.parse_args()

classes = yolo_utils.read_class_names(args.class_name_path)


def process(detector, img):
    boxes, labels_idx = detector.process_prediction(img)
    # GET LABELS:
    labels = []
    for idx in labels_idx:
        labels.append(classes[idx])
    return boxes, labels, labels_idx


def main():
    detector = YoloV3Detection(graph_path=args.graph_path)
    utils.check_dir(args.out_dir)
    args.mode = "video"
    args.video_pth = "hiv00418.mp4"
    if args.mode == "video":
        cap = cv2.VideoCapture(args.video_pth)
        total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        vid_w, vid_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = os.path.join(args.out_dir, os.path.split(args.video_pth)[-1])
        out_res = cv2.VideoWriter(output_video, fourcc, fps, (int(vid_w), int(vid_h)))

        count = 0
        while cap.isOpened() and count <= total_frame:
            ret, frame = cap.read()
            if ret:
                boxes, labels, labels_idx = process(detector, frame)
                print(boxes)
                frame = utils.draw_boxes_with_texts(frame, boxes, labels, labels_idx)
            out_res.write(frame)
            cv2.imshow("Demo", frame)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif args.mode == "images":
        list_ims = os.listdir(args.im_dir)
        for im_n in tqdm(list_ims):
            if im_n.split(".")[-1] in ["jpg", "png"]:
                im_pth = os.path.join(args.im_dir, im_n)
                im = cv2.imread(im_pth)
                boxes, labels, labels_idx = process(detector, im)
                im = utils.draw_boxes_with_texts(im, boxes, labels, labels_idx)
                # save
                im_save_pth = os.path.join(args.out_dir, im_n)
                cv2.imwrite(im_save_pth, im)

    else:
        print("[ERRO] Mode is not exits, check mode pls ... ")


if __name__ == '__main__':
    main()
