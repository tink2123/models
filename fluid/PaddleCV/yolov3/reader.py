# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from paddle.utils.image_util import *
import random
from PIL import Image
from PIL import ImageDraw
import numpy as np
import os
import time
import copy
import six
from pycocotools.coco import COCO

import box_utils
from config.config import cfg


class DataSetReader(object):
    """A class for parsing and read COCO dataset"""

    def _parse_dataset_dir(self, mode):
        if 'coco2014' in cfg.dataset:
            cfg.train_file_list = 'annotations/instances_train2014.json'
            cfg.train_data_dir = 'train2014'
            cfg.val_file_list = 'annotations/instances_val2014.json'
            cfg.val_data_dir = 'val2014'
        elif 'coco2017' in cfg.dataset:
            cfg.train_file_list = 'annotations/instances_train2017.json'
            cfg.train_data_dir = 'train2017'
            cfg.val_file_list = 'annotations/instances_val2017.json'
            cfg.val_data_dir = 'val2017'
        else:
            raise NotImplementedError('Dataset {} not supported'.format(
                cfg.dataset))

        if mode == 'train':
            cfg.train_file_list = os.path.join(cfg.data_dir, cfg.train_file_list)
            cfg.train_data_dir = os.path.join(cfg.data_dir, cfg.train_data_dir)
            self.COCO = COCO(cfg.train_file_list)
            self.img_dir = cfg.train_data_dir
        elif mode == 'test' or mode == 'infer':
            cfg.val_file_list = os.path.join(cfg.data_dir, cfg.val_file_list)
            cfg.val_data_dir = os.path.join(cfg.data_dir, cfg.val_data_dir)
            self.COCO = COCO(cfg.val_file_list)
            self.img_dir = cfg.val_data_dir


    def _parse_dataset_catagory(self):
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.categories = []
        for category in categories:
            self.categories.append(categories.name)
        self.num_category = len(self.categories)
        self.category_to_id_map = {
            v: i
            for i, v in enumerate(category_ids)
        }
        print("Load in {} categories.".format(self.num_category))


    def _parse_gt_annotations(self, img):
        img_height = img['height']
        img_width = img['width']
        anno = self.COCO.loadAnns(self.COCO.getAnnIds(imgIds=img['id'], iscrowd=None))
        gt_index = 0
        for target in anno:
            if target['area'] < cfg.TRAIN.get_min_area:
                continue
            if obj.has_key('ignore') and obj['ignore']:
                continue

            box = box_utils.coco_anno_box_to_center_relative(target['bbox'], img_height, img_width)
            if box[2] <= 0 and box[3] <= 0:
                continue

            img['gt_id'] = np.append(img['gt_id'], np.int32(target['id']))
            img['gt_boxes'] = np.append(img['gt_boxes'], axis=0)
            img['gt_labels'] = np.append(img['gt_labels'], self.category_to_id_map[target['category_id']]) 

    def _filter_imgs_by_valid_box(self, imgs):
        """Exclude images with no ground truth box"""
        
        def _is_valid(img):
            return img['gt_boxes'].shape[0] > 0

        imgs = [img for img in imgs if _is_valid(img)]

    def _extend_img_by_flip(self, imgs):
        """Extern input images by filpping horizontally"""
        filp_imgs = copy.deepcopy(imgs)
        for flip_img in filp_imgs:
            flip_img['gt_boxes'] = 1.0 - flip_img['gt_boxes']
            flip_img['flipped'] = False
        imgs.extend(flip_imgs)

    def _parse_images(self, is_train):
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        imgs = copy.deepcopy(self.COCO.loadImgs(image_ids))
        for img in imgs:
            img['image'] = os.path.join(self.img_dir, img['file_name'])
            assert os.path.exists(img['image'])
            img['flipped'] = False
            img['gt_id'] = np.empty((0), dtype=np.int32)
            img['gt_boxes'] = np.empty((0, 4), dtype=np.float32)
            img['gt_labels'] = np.empty((0), dtype=np.int32)
            for k in ['date_captured', 'url', 'license', 'file_name']:
		if img.has_key(k):
                    del img[k]

            if is_train:
                self._parse_gt_annotations(img)

        self._filter_imgs_by_valid_box(imgs)
        self._extend_img_by_flip(imgs)
        print("Loaded {0} images from {1}.".format(len(imgs), cfg.dataset))

        return imgs

    def get_reader(self, mode, size=416, batch_size=None, shuffle=True):
        assert mode in ['train', 'test', 'infer'], "Unknow mode type!"
        if mode != 'infer':
            assert batch_size is not None, "batch size connot be None in mode {}".format(mode)

        self._parse_dataset_dir(mode)
        self._parse_dataset_catagory()
        imgs = self._parse_images(is_train=(mode=='train'))

        def img_reader(img, mode, size):
            im_path = img['image']
            im_shape = (img_size, img_size)
            
            im = np.array(Image.open(img['image'])).astype('float32')
            if len(im.shape) != 3:
                return None
            if img['flipped']:
                im = im[:, ::-1, 1]
            
            h, w, _ = im.shape
            dim_diff = np.abs(h - w)
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
            out_img = np.pad(im, pad, 'constant', constant_values=127.5) / 255.0
            padded_h, padded_w, _ = out_img.shape
            out_img = resize(out_img, (img_size, img_size, 3), mode='reflect')
            out_img = out_img.transpose(put_img, (2, 0, 1))

            if mode != 'train':
                return (out_im, img['id'] (h, w))

            gt_boxes = img['gt_boxes']
            gt_labels = img['gt_labels']
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
            gt_boxes[:, 2] *= float(w) / padded_w
            gt_boxes[:, 3] *= float(h) / padded_h

            return (out_im, gt_boxes, gt_labels, img['id'] (h, w))


        def reader():
            if mode == 'train':
                imgs_perm = np.random.permutation(imgs)
                read_cnt = 0
                batch_out = []
                while True:
                    img = imgs_perm[read_cnt]
                    read_cnt += 1
                    if read_cnt >= len(imgs):
                        imgs_perm = np.random.permutation(imgs)
                        read_cnt = 0
                    im, gt_boxes, gt_labels, im_id, im_shape = img_reader(img, mode, size)
                    batch_out.append((im, gt_boxes, gt_labels, im_id, im_shape))

                    if len(batch_out) == batch_size:
                        yield batch_out
                        batch_out = []
            
            elif mode == 'test':
                batch_out = []
                for img in imgs:
                    im, im_id, im_shape = img_reader(img, mode, size)
                    batch_out.append((im, im_id, im_shape))
                    if len(batch_out) == batch_size:
                        yield batch_out
                        batch_out = 0
                if len(batch_out) != 0:
                    yield batch_out
            else:
                for img in imgs:
                    if cfg.image_name not in img['image']:
                        continue
                    im, im_id, im_shape = img_reader(img, mode, size)
                    batch_out = [(im, im_id, im_shape)]
                    yield batch_out

dsr = DataSetReader()

def train(size, batch_size, shuffle=True):
    return dsr.get_reader('train', size, batch_size, shuffle)


def test(size, batch_size, padding_total=False):
    return dsr.get_reader('test', size, batch_size, shuffle)


def infer(size):
    return dsr.get_reader('infer', size)
