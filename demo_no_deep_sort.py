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


def check_intersect(box, mask):
    if (mask[max(0, box[1]), max(0, box[0])] != 0 or \
        mask[min(479, box[1] + box[3]), max(0, box[0])] != 0 or \
        mask[max(0, box[1]), min(639, box[0] + box[2])] != 0 or \
        mask[min(479, box[1] + box[3]), min(639, box[0] + box[2])] != 0):
        return True
    return False

def run(detector, number, cam, mask_name, date, name):
    print('Processing video number ', number)

    video_capture = cv2.VideoCapture('/media/aioz-trung-intern/data/sml/' + cam + '/' + name)
    w = 640
    h = 480
    
    #if writeVideo_flag:
    # Define the codec and create VideoWriter object
        #w = int(video_capture.get(3))
        #h = int(video_capture.get(4))
        
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('data_res/res_' + cam + '/o_' + name, fourcc, 30, (w, h))
        
    fps = 0.0
    mask = cv2.imread(mask_name, 0)
    contours, _ = cv2.findContours(np.expand_dims(mask,axis=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    x,y,wi,he = cv2.boundingRect(cont_sorted[0])

    #init tracker
    tracker = Sort(use_dlib=True) #create instance of the SORT tracker
    # Display init
    # colours = np.random.rand(32, 3)  # used only for display
    # plt.ion()
    # fig = plt.figure()
    bbox_stack = []
    avg_people = 0
    count = 0
    video_mask_frame = np.zeros(shape=[480, 640], dtype=np.float64)
    nop_list = []
    x_mask = []
    y_mask = []

    while(video_capture.isOpened()):
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if (not ret):
            break
        count += 1
        #org_frame = cv2.resize(frame, (w, h), interpolation = cv2.INTER_AREA)        
        mask_frame = np.zeros(shape=[480, 640], dtype=np.uint8)
                
        #frame = org_frame.copy()
        #frame[mask==0] = [0,0,0]        
        cv2.imwrite('frame.jpg', frame)        
        break
        #if (count == 200):
        #    break

        t1 = time.time()

        boxs, _, _ = process(detector, org_frame)
        if (boxs.shape[0] != 0):
            boxs[:,2] = boxs[:,2] - boxs[:,0]
            boxs[:,3] = boxs[:,3] - boxs[:,1]

        #boxs = yolo.detect_image(image)
        #print("box ", np.asarray(boxs).shape, " box: ", boxs)	
        # break

        # Draw bbox
        #print(len(boxs))
        num_of_person = 0
        filtered_bbox = []
        for bbox in boxs:
            if (check_intersect(bbox, mask)):
                avg_people += 1
                filtered_bbox.append(bbox)
                #cv2.rectangle(org_frame,(int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),(255,0,0), 2)

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


            # #index = bbox_stack_len.index(key)
            # index = [i for i, e in enumerate(bbox_stack_len) if e == key]
            del bbox_stack[0]
        
            # if (index[-1] == len(filtered_bbox)):
            #     del bbox_stack[0]
            # else:
            #     del bbox_stack[index[0]]
        # else:
        #     for bbox in filtered_bbox:
        #         cv2.rectangle(org_frame,(int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),(255,0,0), 2)
        cv2.imshow('mask'+cam, cv2.threshold(mask_frame, 1, 255, cv2.THRESH_BINARY)[1])
        
        nop_list.append(int(num_of_person))
    
        video_mask_frame += cv2.threshold(mask_frame, 1, 255, cv2.THRESH_BINARY)[1] / 255.0

        #cv2.rectangle(org_frame,(x,y),(x+wi,y+he),(0,0,255),2)
        #cv2.drawContours(org_frame, contours, 0, (0, 255, 0), 1)

        # Update tracker
        # print(org_frame.shape, boxs)
        # detections = np.array(boxs)
        # if (detections.shape[0] != 0):
        #     detections[:,2] = detections[:,2] + detections[:,0]
        #     detections[:,3] = detections[:,3] + detections[:,1]
        # trackers = tracker.update(detections, frame)

        # Put number and fps
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(org_frame, 'FPS: ' + str(int(fps)), (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA) 
        cv2.putText(org_frame, 'Pps: ' + str(num_of_person), (10,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        
        # ax1 = fig.add_subplot(111, aspect='equal')
        # ax1.imshow(org_frame)


        # for d in trackers:
        #     #f_out.write('%d,%d,%d,%d,x,x,x,x,%.3f,%.3f,%.3f,%.3f\n' % (d[4], frame, 1, 1, d[0], d[1], d[2], d[3]))
        #     d = d.astype(np.int32)
        #     ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
        #                                     ec=colours[d[4] % 32, :]))
        #     ax1.set_adjustable('box')
        #     #label
        #     ax1.annotate('id = %d' % (d[4]), xy=(d[0], d[1]), xytext=(d[0], d[1]))
        #     if detections != []:#detector is active in this frame
        #         ax1.annotate(" DETECTOR", xy=(5, 45), xytext=(5, 45))

        # plt.axis('off')
        # fig.canvas.flush_events()
        # plt.draw()
        # fig.tight_layout()
        # #save the frame with tracking boxes
        # ax1.cla()

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
    # Normalize and save
    #video_mask_frame = np.array(255 * (video_mask_frame - min(video_mask_frame.flatten())) / (max(video_mask_frame.flatten()) - min(video_mask_frame.flatten())), dtype=np.uint8)
    #print(video_mask_frame)
    #ax = sns.heatmap(video_mask_frame, vmin=0, vmax=1, cmap='jet')
    #plt.savefig('data_res/heat_res_' + cam + '/' + 'heat_' + cam + '_' + str(number) + '.png')

    #imC = cv2.applyColorMap(video_mask_frame, cv2.COLORMAP_JET)
    #cv2.imwrite('data_res/heat_res_' + cam + '/' + 'heat_' + cam + '_' + str(number) + '.jpg', imC)
    return video_mask_frame
    #splt.show()
    #ax.savefig("output.png")
    #cv2.imshow('heatmap_' + str(number) + '.jpg', ax)
    #cv2.waitKey(0)

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
    #for number in list_number:
        # Get name
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
            
            # Process video mask
            #video_mask_frame = (video_mask_frame - min(video_mask_frame.flatten())) / (max(video_mask_frame.flatten()) - min(video_mask_frame.flatten()))
            
            # Normalize and create heatmap
            #imC = None
            #imC = cv2.normalize(video_mask_frame, imC, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #imC = cv2.applyColorMap(imC, cv2.COLORMAP_JET)
            
            # Save heatmap
            # cv2.imwrite('data_res/heat_res_' + cam + '/heat_' + cam + '_' + str(number) + '.jpg', imC)
            
            # Reset video mask frame
            video_mask_frame = np.zeros(shape=[480, 640], dtype=np.float64)
        number += 1
    #np.savetxt('data_res/result/' + str(number) + '.txt', [avg_people / count])

if __name__ == '__main__':
    main()
