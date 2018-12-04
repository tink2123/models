import os
import time
import numpy as np
import paddle
import paddle.fluid as fluid
import reader
from utility import print_arguments, parse_args
import models
import box_utils
from coco_reader import load_label_names
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config.config import cfg

# np.set_printoptions(threshold='nan')
# np.set_printoptions(suppress=True)

def infer():

    if not os.path.exists('output'):
        os.mkdir('output')

    label_names, _ = load_label_names(cfg.name_path)
    model = models.YOLOv3(cfg.model_cfg_path, is_train=False)
    model.build_model()
    outputs = model.get_pred()
    input_size = model.get_input_size()
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
    infer_reader = reader.infer(input_size, os.path.join(cfg.image_path, cfg.image_name))
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    fetch_list = outputs
    data = next(infer_reader())
    im_shape = data[0][2]
    # im_scale = data[0][2]
    outputs = exe.run(
        fetch_list=[v.name for v in fetch_list],
        feed=feeder.feed(data),
        return_numpy=True)

    pred_boxes, pred_confs, pred_labels = box_utils.get_all_yolo_pred(outputs, yolo_anchors,
                                                        yolo_classes, (input_size, input_size))
    boxes, scores, labels = box_utils.calc_nms_box(pred_boxes, pred_confs, pred_labels, 
                                    im_shape, input_size, cfg.conf_thresh, 
                                    cfg.TEST.nms_thresh)
    path = os.path.join(cfg.image_path, cfg.image_name)
    box_utils.draw_boxes_on_image(path, boxes, scores, labels, label_names)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    infer()
