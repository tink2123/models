import os
import time
import numpy as np
import paddle
import paddle.fluid as fluid
import reader
from utility import print_arguments, parse_args
import models
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config.config import cfg


def infer():

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
    input_size = model.get_input_size()
    infer_reader = reader.infer(input_size)
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    fetch_list = [pred_boxes, pred_confs, pred_labels]
    data = next(infer_reader())
    im_shape = data[0][2]
    pre_boxes, pred_confs, pred_labels = exe.run(
        fetch_list=[v.name for v in fetch_list],
        feed=feeder.feed(data),
        return_numpy=False)
    boxes, _, labels = box_utils.calc_nms_box(pred_boxes[0], pred_confs[0], pred_labels[0], 
                                    im_shape, input_size, cfg.TEST.conf_thresh, 
                                    cfg.TEST.nms_thresh)
    path = os.path.join(cfg.image_path, cfg.image_name)
    draw_bounding_box_on_image(path, boxes, labels, label_names)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    infer()
