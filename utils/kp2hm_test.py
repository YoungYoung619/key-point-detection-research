"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""

import os, sys, glob, cv2
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np

from utils.kp2hm_utils import *

category_index_for_generate_label = {'vehicle': 0,
                                     'light': 1,
                                     'sign': 2,
                                     'person': 3}

std_size = (128, 128) ## h, w

radius_scalar = 5

def read_one_sample(img_name, label_name):
    """read one sample
    Args:
        img_name: img name, like "/usr/img/image001.jpg"
        label_name: the label file responding the img_name, like "/usr/label/image001.xml"
    Return:
        An ndarray with the shape [img_h, img_w, img_c], bgr format
        a list of tuple, represents (obj_type_str,  [ymin, xmin, ymax, xmax])
    """
    # cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    img = cv2.imread(img_name)
    # img = io.imread(img_name)

    DOMTree = xml.dom.minidom.parse(label_name)
    collection = DOMTree.documentElement

    # objs = collection.getElementsByTagName("size")
    # for obj in objs:
    #     img_w = int(obj.getElementsByTagName('width')[0].childNodes[0].data)
    #     img_h = int(obj.getElementsByTagName('height')[0].childNodes[0].data)

    objs = collection.getElementsByTagName("object")
    labels = []
    for obj in objs:
        obj_type = obj.getElementsByTagName('name')[0].childNodes[0].data
        if obj_type == "person":
            bbox = obj.getElementsByTagName('bndbox')[0]
            ymin = bbox.getElementsByTagName('ymin')[0].childNodes[0].data
            xmin = bbox.getElementsByTagName('xmin')[0].childNodes[0].data
            ymax = bbox.getElementsByTagName('ymax')[0].childNodes[0].data
            xmax = bbox.getElementsByTagName('xmax')[0].childNodes[0].data
            label = np.array([float(ymin), float(xmin), float(ymax), float(xmax)])
            labels.append((obj_type, label))
    labels = np.stack(labels, axis=0)
    return img, labels

def generate_hm(raw_img_size, labels):
    """generate top_left, top_right, bottom_left, bottom_right, center heat map,
       embedding info and offset info
    Args:
        raw_img_size: a tuple represents img height and width
        labels: a list of corner bbox of one obj type
    """
    img_h = raw_img_size[0]
    img_w = raw_img_size[1]
    labels[:, 0] = labels[:, 0] / img_h * std_size[0]  ## ymin
    labels[:, 1] = labels[:, 1] / img_w * std_size[1]  ## xmin
    labels[:, 2] = labels[:, 2] / img_h * std_size[0]  ##ymax
    labels[:, 3] = labels[:, 3] / img_w * std_size[1]  ##xmax

    top_left_points = np.stack([labels[:, 1], labels[:, 0]], axis=-1)  ## (xmin, ymin)
    top_right_points = np.stack([labels[:, 3], labels[:, 0]], axis=-1)  ##(xmax, ymin)
    bottom_left_points = np.stack([labels[:, 1], labels[:, 2]], axis=-1)  ##(xmin, ymax)
    bottom_right_points = np.stack([labels[:, 3], labels[:, 2]], axis=-1)  ##(xmax, ymax)
    center_points = np.stack([labels[:, 3] + labels[:, 1], labels[:, 2] + labels[:, 0]], axis=-1) / 2

    top_left_points_floor = np.floor(top_left_points)
    top_right_points_floor = np.floor(top_right_points)
    bottom_left_points_floor = np.floor(bottom_left_points)
    bottom_right_points_floor = np.floor(bottom_right_points)
    center_points_floor = np.floor(center_points)

    top_left_offset = top_left_points - top_left_points_floor
    top_right_offset = top_right_points - top_right_points_floor
    bottom_left_offset = bottom_left_points - bottom_left_points_floor
    bottom_right_offset = bottom_right_points - bottom_right_points_floor
    center_offset = center_points - center_points_floor

    w_h = bottom_right_points - top_left_points
    size = np.min(w_h, axis=-1)
    ## top_left heat map
    top_left_hm = heat_map(img_size=std_size, points=top_left_points_floor, sigmas=size / radius_scalar)
    top_right_hm = heat_map(img_size=std_size, points=top_right_points_floor, sigmas=size / radius_scalar)
    bottom_left_hm = heat_map(img_size=std_size, points=bottom_left_points_floor, sigmas=size / radius_scalar)
    bottom_right_hm = heat_map(img_size=std_size, points=bottom_right_points_floor, sigmas=size / radius_scalar)
    center_hm = heat_map(img_size=std_size, points=center_points_floor, sigmas=size / radius_scalar)

    return top_left_hm, top_right_hm, bottom_left_hm, bottom_right_hm, center_hm

def generate_offset(raw_img_size, labels):
    img_h = raw_img_size[0]
    img_w = raw_img_size[1]
    labels[:, 0] = labels[:, 0] / img_h * std_size[0]  ## ymin
    labels[:, 1] = labels[:, 1] / img_w * std_size[1]  ## xmin
    labels[:, 2] = labels[:, 2] / img_h * std_size[0]  ##ymax
    labels[:, 3] = labels[:, 3] / img_w * std_size[1]  ##xmax

    top_left_points = np.stack([labels[:, 1], labels[:, 0]], axis=-1)  ## (xmin, ymin)
    top_right_points = np.stack([labels[:, 3], labels[:, 0]], axis=-1)  ##(xmax, ymin)
    bottom_left_points = np.stack([labels[:, 1], labels[:, 2]], axis=-1)  ##(xmin, ymax)
    bottom_right_points = np.stack([labels[:, 3], labels[:, 2]], axis=-1)  ##(xmax, ymax)
    center_points = np.stack([labels[:, 3] + labels[:, 1], labels[:, 2] + labels[:, 0]], axis=-1) / 2

    top_left_points_floor = np.floor(top_left_points)
    top_right_points_floor = np.floor(top_right_points)
    bottom_left_points_floor = np.floor(bottom_left_points)
    bottom_right_points_floor = np.floor(bottom_right_points)
    center_points_floor = np.floor(center_points)

    top_left_offset = top_left_points - top_left_points_floor
    top_right_offset = top_right_points - top_right_points_floor
    bottom_left_offset = bottom_left_points - bottom_left_points_floor
    bottom_right_offset = bottom_right_points - bottom_right_points_floor
    center_offset = center_points - center_points_floor

    return 0

pictures_path = 'F:\my_project\pedestrian-detection-in-hazy-weather\dataset\inria_person\PICTURES_LABELS_TRAIN\PICTURES'
anotation_path = 'F:\my_project\pedestrian-detection-in-hazy-weather\dataset\inria_person\PICTURES_LABELS_TRAIN\ANOTATION'

imgs = glob.glob(os.path.join(pictures_path, '*.jpg'))

for img_name in imgs:
    base_name = os.path.basename(img_name).split('.')[0]
    anotation_name = os.path.join(anotation_path, base_name+'.xml')
    img, labels = read_one_sample(img_name, anotation_name)

    img_h = np.shape(img)[0]
    img_w = np.shape(img)[1]

    label_dict = {}
    for label in labels:
        if label[0] in list(category_index_for_generate_label.keys()):
            if label[0] in label_dict.keys():
                label_dict[label[0]].append(label[1])
            else:
                label_dict[label[0]] = []
                label_dict[label[0]].append(label[1])

    obj_points_tl_hm = np.zeros((len(category_index_for_generate_label.values()), std_size[0], std_size[1]))
    obj_points_tr_hm = np.zeros((len(category_index_for_generate_label.values()), std_size[0], std_size[1]))
    obj_points_bl_hm = np.zeros((len(category_index_for_generate_label.values()), std_size[0], std_size[1]))
    obj_points_br_hm = np.zeros((len(category_index_for_generate_label.values()), std_size[0], std_size[1]))
    obj_points_c_hm = np.zeros((len(category_index_for_generate_label.values()), std_size[0], std_size[1]))
    for obj_type, corner_bbox in label_dict.items(): ## for each obj type, do
        obj_index = category_index_for_generate_label[obj_type]
        tl_hm, tr_hm, bl_hm, br_hm, c_hm = generate_label(raw_img_size=(img_h, img_w), labels=np.array(corner_bbox))
        obj_points_tl_hm[obj_index] = tl_hm
        obj_points_tr_hm[obj_index] = tr_hm
        obj_points_bl_hm[obj_index] = bl_hm
        obj_points_br_hm[obj_index] = br_hm
        obj_points_c_hm[obj_index] = c_hm
    pass
