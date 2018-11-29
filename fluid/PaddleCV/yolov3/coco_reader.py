#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
from skimage.transform import resize
import numpy as np
import random
from config.config import cfg


def load_label_names(path):
    label_names = []
    label_ids = []
    with open(path) as f:
        for line in f.readlines():
            data = line.strip().split()
            label_ids.append(int(data[0]))
            label_names.append(data[1])
    return label_names, label_ids

def parse_data_config(path):
    options = {}
    with open(path) as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            k, v = line.strip().split('=')[:2]
            options[k] = v
    return options

def get_img_path_list(data_cfg_path, mode):
    assert mode in ["train", "valid"]
    opts = parse_data_config(data_cfg_path)
    file_name = "/".join(data_cfg_path.split('/')[:-1] + [opts[mode]])
    with open(file_name) as f:
        img_list = map(lambda x: x.strip(), f.readlines())
    return img_list

def read_img_data(img_path, img_size):
    assert os.path.exists(img_path), "Image {} not found".format(img_path)
    img = cv2.imread(img_path).astype('float32')
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    out_img = np.pad(img, pad, 'constant', constant_values=128.0) / 255.0
    padded_h, padded_w, _ = out_img.shape
    out_img = resize(out_img, (img_size, img_size, 3), mode='reflect')
    out_img = out_img.transpose((2, 0, 1))

    return out_img, h, w, padded_h, padded_w


class CocoDataset(object):
  
    def __init__(self, data_cfg_path, mode, img_size=608, shuffle=False):
        self.data_cfg_path = data_cfg_path
        self.mode = mode
        self.img_size = img_size
        self.shuffle = shuffle
        self.img_list = get_img_path_list(data_cfg_path, mode)
        self.file_num = len(self.img_list)

    def __getitem__(self, index):
        if self.shuffle and index % self.file_num == 0:
            self.img_list = random.shuffle(self.img_list)
        img_path = self.img_list[index % self.file_num]
        label_path = img_path.replace("images", "labels")[:-4] + "txt"       

        img, h, w, padded_h, padded_w = read_img_data(img_path, self.img_size)
        img_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])

        labels = np.zeros((cfg.max_box_num))
        boxes = np.zeros((cfg.max_box_num, 4))
        if os.path.exists(label_path):
            label_data = np.loadtxt(label_path).rehsape((-1, 5))
            gt_labels = label_data[:, 0]
            gt_boxes = label_data[:, 1:].astype('flaot32')

            x1 = (gt_boxes[:, 0] - gt_boxes[:, 2] / 2.0) * w
            x2 = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2.0) * w
            y1 = (gt_boxes[:, 1] - gt_boxes[:, 3] / 2.0) * h
            y2 = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2.0) * h

            x1 += pad[1][0]
            x2 += pad[1][0]
            y1 += pad[0][0]
            y2 += pad[0][0]

            gt_boxes[:, 0] = (x1 + x2) / 2.0 / padded_w
            gt_boxes[:, 1] = (y1 + y2) / 2.0 / padded_h
            gt_boxes[:, 2] *= w / padded_w
            gt_boxes[:, 3] *= h / padded_h

            labels[: min(len(gt_labels), cfg.max_box_num)] = gt_labels[:cfg.max_box_num]
            boxes[: min(len(gt_boxes), cfg.max_box_num)] = gt_boxes[:cfg.max_box_num]

        return (img, boxes, labels, img_id, (h, w))

    def __len__(self):
        return self.file_num

