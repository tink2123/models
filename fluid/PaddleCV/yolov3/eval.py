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
from eval_helper import get_nmsed_box
from eval_helper import get_dt_res
import paddle
import paddle.fluid as fluid
import reader
from utility import print_arguments, parse_args
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config.config import cfg


def eval():
    if '2014' in cfg.dataset:
        test_list = 'annotations/instances_val2014.json'
    elif '2017' in cfg.dataset:
        test_list = 'annotations/instances_val2017.json'

    coco = COCO(os.path.join(cfg.data_dir, test_list))
    category_ids = self.COCO.getCatIds()
    label_names = [c['name'] for c in self.COCO.loadCats(category_ids)]

    model = models.YOLOv3(cfg.model_cfg_path, is_train=False)
    model.build_model()
    pred_boxes, pred_confs, pred_labels = model.get_pred()
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if cfg.pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
    # yapf: enable
    infer_reader = reader.infer()
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    dts_res = []
    fetch_list = [pred_boxes, pred_confs, pred_labels]
    data = next(infer_reader())
    im_info = [data[0][1]]
    pre_boxes, pred_confs, pred_labels = exe.run(
        fetch_list=[v.name for v in fetch_list],
        feed=feeder.feed(data),
        return_numpy=False)
    boxes, labels = box_utils.calc_nms_box(pred_boxes, pred_confs, pred_labels, 
                                    cfg.TEST.conf_thresh, cfg.TEST.nms_thresh)
    path = os.path.join(cfg.image_path, cfg.image_name)

    fetch_list = [rpn_rois, confs, locs]
    for batch_id, batch_data in enumerate(test_reader()):
        start = time.time()
        im_info = []
        for data in batch_data:
            im_info.append(data[1])
        rpn_rois_v, confs_v, locs_v = exe.run(
            fetch_list=[v.name for v in fetch_list],
            feed=feeder.feed(batch_data),
            return_numpy=False)
        new_lod, nmsed_out = get_nmsed_box(rpn_rois_v, confs_v, locs_v,
                                           class_nums, im_info,
                                           numId_to_catId_map)

        dts_res += get_dt_res(total_batch_size, new_lod, nmsed_out, batch_data)
        end = time.time()
        print('batch id: {}, time: {}'.format(batch_id, end - start))

    with open("detection_result.json", 'w') as outfile:
        json.dump(dts_res, outfile)
    print("start evaluate using coco api")
    cocoDt = coco.loadRes("detection_result.json")
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    eval()
