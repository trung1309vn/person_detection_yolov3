"""
truong.le - Oct 23, 2019
"""
import numpy as np


def parse_anchors(anchor_path):
    """
    parse anchors.
    returned data: shape [N, 2], dtype float32
    """
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
    return anchors


def read_class_names(class_name_path):
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
