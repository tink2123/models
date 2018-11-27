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
import time
import numpy as np
import paddle
import paddle.fluid as fluid
import reader
import models
from utility import print_arguments, parse_args
import box_utils
import data_utils
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config.config import cfg


def get_pred_result(boxes, confs, labels, im_id):
    result = []
    for box, conf, label in zip(boxes, confs, labels):
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 2
        bbox = [x1, y2, w, h]
        
        res = {
                'image_id': im_id,
                'category_id': label,
                'bbox': bbox,
                'score': conf
        }
        result.append(res)
    return result

def eval():
    if '2014' in cfg.dataset:
        test_list = 'annotations/instances_val2014.json'
    elif '2017' in cfg.dataset:
        test_list = 'annotations/instances_val2017.json'

    label_names = data_utils.load_coco_names(os.path.join(cfg.data_dir, "coco.names"))
    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    print("Found {} CUDA device.".format(devices_num))

    model = models.YOLOv3(cfg.model_cfg_path, is_train=False)
    model.build_model()
    outputs = model.get_pred()
    yolo_anchors = model.get_yolo_anchors()
    yolo_classes = model.get_yolo_classes()
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if cfg.pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
    # yapf: enable
    input_size = model.get_input_size()
    test_reader = reader.test(input_size, max(devices_num, 1))
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    dts_res = []
    fetch_list = outputs
    for batch_id, batch_data in enumerate(test_reader):
        start_time = time.time()
        batch_outputs = exe.run(
            fetch_list=[v.name for v in fetch_list],
            feed=feeder.feed(batch_data),
            return_numpy=False)
        for data, outputs in zip(batch_data, batch_outputs):
            pred_boxes, pred_confs, pred_labels = box_utils.get_all_yolo_pred(
                    outputs, yolo_anchors, yolo_classes, (input_size, input_size))
            boxes, confs. labels = box_utils.calc_nms_box(pred_boxes, pred_confs, pred_labels,
                                                    im_shape, input_size, cfg.TEST.conf_thresh,
                                                    cfg.TEST.nms_thresh)
            im_shape = data[2]
            dts_res += get_pred_result(boxes, confs, labels, im_shape)

    with open("yolov3_result.json", 'w') as outfile:
        json.dump(dts_res, outfile)
    print("start evaluate detection result with coco api")
    cocoDt = coco.loadRes("yolov3_result.json")
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print("evaluate done.")


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    eval()
