#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import time
import seaborn as sns
from collections import Counter

from sort.sort import Sort

import argparse
from tqdm import tqdm
from utils import utils
from utils import yolo_utils
from src.detection_yolo3_wrapper import YoloV3Detection

warnings.filterwarnings('ignore')

stack_num = 15
classes = yolo_utils.read_class_names("./data/coco.names")

###
# Process: Detect person in video frame
# Input: Person detector and video frame
# Output: Bounding boxes, boxes label and labels index 
###
def process(detector, img):
    boxes, labels_idx = detector.process_prediction(img)
    # GET LABELS:
    labels = []
    boxes_tmp = []
    for i, idx in enumerate(labels_idx):
        if (idx == 0):
            boxes_tmp.append(boxes[i])
        labels.append(classes[idx])
    return np.asarray(boxes_tmp), labels, labels_idx


###
# Process: Checking if box is in ROI region of video
# Input: Box, mask that represents for ROI region
# Output: True or False
###
def check_intersect(box, mask):
    if (mask[max(0, box[1]), max(0, box[0])] != 0 or \
        mask[min(479, box[1] + box[3]), max(0, box[0])] != 0 or \
        mask[max(0, box[1]), min(639, box[0] + box[2])] != 0 or \
        mask[min(479, box[1] + box[3]), min(639, box[0] + box[2])] != 0):
        return True
    return False

###
# Process: Run detector on whole video
# Input: Detector, video number, camera folder, mask-frame's name, date, video's name
# Output: None
###
def run(detector, number, cam, mask_name, date, name):
    print('Processing video number ', number)

    video_capture = cv2.VideoCapture('/media/aioz-trung-intern/data/sml/' + cam + '/' + name)
    w = 640
    h = 480

    # Save video    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('data_res/res_' + cam + '/o_' + name, fourcc, 30, (w, h))
        
    # Draw mask region on video frame
    fps = 0.0
    mask = cv2.imread(mask_name, 0)
    contours, _ = cv2.findContours(np.expand_dims(mask,axis=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    x,y,wi,he = cv2.boundingRect(cont_sorted[0])

    # Init tracker
    tracker = Sort(use_dlib=True) #create instance of the SORT tracker
    bbox_stack = []
    avg_people = 0
    count = 0
    video_mask_frame = np.zeros(shape=[480, 640], dtype=np.float64)
    nop_list = []
    x_mask = []
    y_mask = []


    # Processing video
    while(video_capture.isOpened()):
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if (not ret):
            break
        count += 1
        org_frame = cv2.resize(frame, (w, h), interpolation = cv2.INTER_AREA)        
        mask_frame = np.zeros(shape=[480, 640], dtype=np.uint8)

        # Save first frame        
        #frame = org_frame.copy()
        #frame[mask==0] = [0,0,0]        
        #cv2.imwrite('frame.jpg', frame)        
        #break
        #if (count == 200):
        #    break

        t1 = time.time()

        # Detect
        boxs, _, _ = process(detector, org_frame)
        if (boxs.shape[0] != 0):
            boxs[:,2] = boxs[:,2] - boxs[:,0]
            boxs[:,3] = boxs[:,3] - boxs[:,1]

        # Draw bbox
        # Bounding Box rectify
        num_of_person = 0
        filtered_bbox = []
        for bbox in boxs:
            if (check_intersect(bbox, mask)):
                avg_people += 1
                filtered_bbox.append(bbox)

        if (len(bbox_stack) != stack_num):
            bbox_stack.append(filtered_bbox)
            for bbox in filtered_bbox:
                cv2.rectangle(org_frame,(int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),(255,0,0), 2)
                cv2.circle(mask_frame,(int((2*bbox[0]+bbox[2])/2), int((2*bbox[1]+bbox[3])/2)), 20, (255), -1)
                x_mask.append(int((2*bbox[0]+bbox[2])/2))
                y_mask.append(int((2*bbox[1]+bbox[3])/2))
            num_of_person = len(filtered_bbox)

        else:
            bbox_stack_len = [len(x) for x in bbox_stack]

            list_counter = Counter(bbox_stack_len)
            
            argmax = np.argmax(list(list_counter.values()))

            key = list(list_counter.keys())[argmax]
            
            index = [i for i, e in enumerate(bbox_stack_len) if e == key]
            
            if (key != len(filtered_bbox)):
                for bbox in bbox_stack[index[-1]]:
                    cv2.rectangle(org_frame,(int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),(255,0,0), 2)
                    cv2.circle(mask_frame,(int((2*bbox[0]+bbox[2])/2), int((2*bbox[1]+bbox[3])/2)), 20, (255), -1)
                    x_mask.append(int((2*bbox[0]+bbox[2])/2))
                    y_mask.append(int((2*bbox[1]+bbox[3])/2))
                num_of_person = len(bbox_stack[index[-1]])
            else:       
                for bbox in filtered_bbox:
                    cv2.rectangle(org_frame,(int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),(255,0,0), 2)
                    cv2.circle(mask_frame,(int((2*bbox[0]+bbox[2])/2), int((2*bbox[1]+bbox[3])/2)), 20, (255), -1)
                    x_mask.append(int((2*bbox[0]+bbox[2])/2))
                    y_mask.append(int((2*bbox[1]+bbox[3])/2))
                num_of_person = len(filtered_bbox)
                
            bbox_stack.append(filtered_bbox)

            del bbox_stack[0]
        
        cv2.imshow('mask'+cam, cv2.threshold(mask_frame, 1, 255, cv2.THRESH_BINARY)[1])
        
        nop_list.append(int(num_of_person))
    
        video_mask_frame += cv2.threshold(mask_frame, 1, 255, cv2.THRESH_BINARY)[1] / 255.0

        # Put number and fps
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(org_frame, 'FPS: ' + str(int(fps)), (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA) 
        cv2.putText(org_frame, 'Pps: ' + str(num_of_person), (10,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        
        # Apply transparent mask to frame
        tmp_mask = org_frame.copy()
        tmp_mask[mask==255] = 255
        alpha = 0.6
        cv2.addWeighted(org_frame, alpha, tmp_mask, 1 - alpha, 0, org_frame)

        # Write video
        out.write(org_frame)
           
        cv2.imshow('original'+cam, org_frame)
        #cv2.imshow('masking', frame)
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    np.savetxt('data_res/result/' + str(number) + '.txt', [avg_people / count])
    np.savetxt('data_res/res_' + cam + '/' + name.replace('.mp4', '') + '.txt', nop_list, fmt='%d')
    np.savetxt('data_res/cor_res_' + cam + '/x' + name.replace('.mp4', '') + '.txt', x_mask, fmt='%d')
    np.savetxt('data_res/cor_res_' + cam + '/y' + name.replace('.mp4', '') + '.txt', y_mask, fmt='%d')
    return video_mask_frame

def main(): # Argv; filename cam start end mask_name
    #list_number = [224, 225, 226,
    #   227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237]
    # 385 471
    detector = YoloV3Detection(graph_path='yolov3_object_detection.pb')

    date = sys.argv[5]
    number = int(sys.argv[2])
    cam = sys.argv[1]
    video_mask_frame = np.zeros(shape=[480, 640], dtype=np.float64)
    while (number <= int(sys.argv[3])):
        if (number >= 100):
            name = '1912' + date + '_' + str(number) + '.mp4'
        elif (number >= 10):
            name = '1912' + date + '_' + '0' + str(number) + '.mp4'
        else:
            name = '1912' + date + '_' + '00' + str(number) + '.mp4'

        # Processing    
        video_mask_frame += run(detector, number, cam, sys.argv[4], date, name)

        # Save video frequency        
        if (number % 1 == 0):
            # Save frequency result
            np.savetxt('data_res/fre_res_' + cam + '/fre_' + cam + '_' + name.replace('.mp4', '') + '.txt', video_mask_frame, fmt='%d')
            
            video_mask_frame = np.zeros(shape=[480, 640], dtype=np.float64)
        number += 1

if __name__ == '__main__':
    main()
