"""
truong.le - Oct 23, 2019
freeze weight
"""
import cv2
import argparse
import numpy as np
import tensorflow as tf
from utils import utils
from utils import yolo_utils
from src.model import yolov3
from utils.nms_utils import gpu_nms
from termcolor import colored
from src.detection_yolo3_wrapper import YoloV3Detection

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure")
parser.add_argument("--input_image", type=str, default="data/dog.jpg",
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--save_path", type=str, default='yolov3_object_detection.pb',
                    help="path for save pb file")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
args = parser.parse_args()

anchors = yolo_utils.parse_anchors(args.anchor_path)
classes = yolo_utils.read_class_names(args.class_name_path)
num_class = len(classes)


def freeze():
    img_ori = cv2.imread(args.input_image)
    img = cv2.resize(img_ori, tuple(args.new_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    with tf.Session() as sess:
        # input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        input_data = tf.placeholder(tf.float32, [None, None, None, 3], name='image_tensor')
        yolo_model = yolov3(num_class, anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=100, score_thresh=0.4,
                                        nms_thresh=0.5)

        saver = tf.train.Saver()
        saver.restore(sess, args.restore_path)

        # op = sess.graph.get_operations()
        # for m in op:
        #     print(m.values())

        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

        # freeze model
        # list_out = ["concat_10", "concat_11", "concat_12"]
        list_out = [boxes.name.split(":")[0], scores.name.split(":")[0], labels.name.split(":")[0]]
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            list_out
        )
        with tf.gfile.GFile(args.save_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print(colored("[INFO] Freeze DONE, saved at {}".format(args.save_path), color="cyan", attrs=['bold']))
        print(colored("[INFO] Input: {}".format(input_data.name), color="white", attrs=['bold']))
        print(colored("[INFO] Outputs: {}, {}, {}".format(boxes.name, scores.name, labels.name),
                      color="white", attrs=['bold']))


def test_pb():
    detector = YoloV3Detection(graph_path=args.save_path)
    img = cv2.imread(args.input_image)
    boxes, labels_idx = detector.process_prediction(img)
    # GET LABELS:
    labels = []
    for idx in labels_idx:
        labels.append(classes[idx])
    # VIS
    img = utils.draw_boxes_with_texts(img, boxes, labels, labels_idx)
    cv2.imshow("test", img)
    cv2.waitKey()


def main():
    # freeze model
    freeze()
    # test pb
    # test_pb()


if __name__ == '__main__':
    main()
