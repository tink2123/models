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

import numpy as np
import os
import random
import copy
import cv2
import box_utils
import image_utils
from pycocotools.coco import COCO
from config.config import cfg


class DataSetReader(object):
    """A class for parsing and read COCO dataset"""

    def __init__(self):
        self.has_parsed_categpry = False

    def _parse_dataset_dir(self, mode):
        # cfg.data_dir = "dataset/coco"
        # cfg.train_file_list = 'annotations/instances_val2017.json'
        # cfg.train_data_dir = 'val2017'
        # cfg.dataset = "coco2017"
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
        self.categories = self.COCO.loadCats(self.COCO.getCatIds())
        self.num_category = len(self.categories)
        self.label_names = []
        self.label_ids = []
        for category in self.categories:
            self.label_names.append(category['name'])
            self.label_ids.append(int(category['id']))
        self.category_to_id_map = {
            v: i
            for i, v in enumerate(self.label_ids)
        }
        print("Load in {} categories.".format(self.num_category))
        self.has_parsed_categpry = True

    def get_label_infos(self):
        if not self.has_parsed_categpry:
            self._parse_dataset_dir("test")
            self._parse_dataset_catagory()
        return (self.label_names, self.label_ids)

    def _parse_gt_annotations(self, img):
        img_height = img['height']
        img_width = img['width']
        anno = self.COCO.loadAnns(self.COCO.getAnnIds(imgIds=img['id'], iscrowd=None))
        gt_index = 0
        for target in anno:
            if target['area'] < cfg.TRAIN.gt_min_area:
                continue
            if target.has_key('ignore') and target['ignore']:
                continue

            box = box_utils.coco_anno_box_to_center_relative(target['bbox'], img_height, img_width)
            if box[2] <= 0 and box[3] <= 0:
                continue

            img['gt_id'][gt_index] = np.int32(target['id'])
            img['gt_boxes'][gt_index] = box
            img['gt_labels'][gt_index] = self.category_to_id_map[target['category_id']]
            gt_index += 1
            if gt_index >= cfg.max_box_num:
                break

    def _parse_images(self, is_train):
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        imgs = copy.deepcopy(self.COCO.loadImgs(image_ids))
        # imgs = imgs[:8]
        for img in imgs:
            img['image'] = os.path.join(self.img_dir, img['file_name'])
            assert os.path.exists(img['image']), \
                    "image {} not found.".format(img['image'])
            box_num = cfg.max_box_num
            img['gt_id'] = np.zeros((cfg.max_box_num), dtype=np.int32)
            img['gt_boxes'] = np.zeros((cfg.max_box_num, 4), dtype=np.float32)
            img['gt_labels'] = np.zeros((cfg.max_box_num), dtype=np.int32)
            for k in ['date_captured', 'url', 'license', 'file_name']:
                if img.has_key(k):
                    del img[k]

            if is_train:
                self._parse_gt_annotations(img)

        print("Loaded {0} images from {1}.".format(len(imgs), cfg.dataset))

        return imgs

    def _parse_images_by_mode(self, mode):
        if mode == 'infer':
            return []
        else:
            return self._parse_images(is_train=(mode=='train'))

    def get_reader(self, mode, size=416, batch_size=None, shuffle=False, image=None):
        assert mode in ['train', 'test', 'infer'], "Unknow mode type!"
        if mode != 'infer':
            assert batch_size is not None, "batch size connot be None in mode {}".format(mode)
            self._parse_dataset_dir(mode)
            self._parse_dataset_catagory()

        def img_reader(img, size, mean, std):
            im_path = img['image']
            im = cv2.imread(im_path).astype('float32')
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_shape = (size, size)

            h, w, _ = im.shape
            im_scale_x = size / float(w)
            im_scale_y = size / float(h)
            out_img = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_CUBIC)
            mean = np.array(mean).reshape((1, 1, -1))
            std = np.array(std).reshape((1, 1, -1))
            out_img = (out_img / 255.0 - mean) / std
            out_img = out_img.transpose((2, 0, 1))

            return (out_img, int(img['id']), (h, w))

        def img_reader_with_augment(img, size, mean, std):
            im_path = img['image']
            im = cv2.imread(im_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_shape = (size, size)
            gt_boxes = img['gt_boxes']
            gt_labels = img['gt_labels']

            im, gt_boxes, gt_labels = image_utils.image_augment(im, gt_boxes, gt_labels, size, mean)
            
            mean = np.array(mean).reshape((1, 1, -1))
            std = np.array(std).reshape((1, 1, -1))
            out_img = (im.astype('float32') / 255.0 - mean) / std
            out_img = out_img.transpose((2, 0, 1))

            return (out_img, gt_boxes, gt_labels)

        def reader():
            if mode == 'train':
                imgs = self._parse_images_by_mode(mode)
                if shuffle:
                    imgs = np.random.permutation(imgs)
                read_cnt = 0
                batch_out = []
                img_ids = []
                while True:
                    img = imgs[read_cnt % len(imgs)]
                    read_cnt += 1
                    if read_cnt % len(imgs) == 0 and shuffle:
                        imgs = np.random.permutation(imgs)
                    im, gt_boxes, gt_labels = img_reader_with_augment(img, size, cfg.pixel_means, cfg.pixel_stds)
                    batch_out.append((im, gt_boxes, gt_labels))
                    img_ids.append(img['id'])

                    if len(batch_out) == batch_size:
                        print("img_ids: ", img_ids)
                        yield batch_out
                        batch_out = []
                        img_ids = []

            elif mode == 'test':
                imgs = self._parse_images_by_mode(mode)
                batch_out = []
                for img in imgs:
                    im, im_id, im_shape = img_reader(img, size, cfg.pixel_means, cfg.pixel_stds)
                    batch_out.append((im, im_id, im_shape))
                    if len(batch_out) == batch_size:
                        yield batch_out
                        batch_out = []
                if len(batch_out) != 0:
                    yield batch_out
            else:
                img = {}
                img['image'] = image
                img['id'] = 0
                im, im_id, im_shape = img_reader(img, size, cfg.pixel_means, cfg.pixel_stds)
                batch_out = [(im, im_id, im_shape)]
                yield batch_out

        return reader


dsr = DataSetReader()

def train(size=416, batch_size=64, shuffle=True):
    return dsr.get_reader('train', size, batch_size, shuffle)

def test(size=416, batch_size=1):
    return dsr.get_reader('test', size, batch_size)

def infer(size=416, image=None):
    return dsr.get_reader('infer', size, image=image)

def get_label_infos():
    return dsr.get_label_infos()

# def get_reader(mode, size=608, batch_size=None, shuffle=True, img_path=None):
#     assert mode in ['train', 'test', 'infer'], "Unknow mode type!"
#     if mode != 'infer':
#         assert batch_size is not None, "batch size connot be None in mode {}".format(mode)
#         data_mode = "train" if mode == 'train' else "valid"
#         dataset = CocoDataset(cfg.data_cfg_path, data_mode, size, shuffle)
#         print("Load in {} images for {}".format(len(dataset), mode))
#
#     def reader():
#         if mode == 'train':
#             batch_out = []
#             index = 0
#             while True:
#                 img, gt_boxes, gt_labels = dataset[index]
#                 batch_out.append((img, gt_boxes, gt_labels))
#                 index += 1
#                 if len(batch_out) == batch_size:
#                     yield batch_out
#                     batch_out = []
#         elif mode == 'test':
#             batch_out = []
#             for i, (img, img_id, img_shape) in enumerate(dataset):
#             # for i, (img, img_id, img_scale) in enumerate(dataset):
#                 if i >= len(dataset):
#                     break
#                 batch_out.append((img, img_id, img_shape))
#                 # batch_out.append((img, img_id, img_scale))
#                 if len(batch_out) == batch_size:
#                     yield batch_out
#                     batch_out = []
#             if len(batch_out) != 0:
#                 yield batch_out
#         else:
#             img, h, w, _, _, _ = read_img_data(img_path, size)
#             batch_out = [(img, 0, (h, w))]
#             # img, img_scale = read_img_data(img_path, size)
#             # batch_out = [(img, 0, img_scale)]
#             yield batch_out
#
#     return reader
#
#
# def train(size, batch_size, shuffle=True):
#     return get_reader('train', size, batch_size, shuffle)
#
#
# def test(size, batch_size, shuffle=False):
#     return get_reader('test', size, batch_size, shuffle)
#
#
# def infer(size, img_path):
#     return get_reader('infer', size, img_path=img_path)
