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
from PIL import Image, ImageDraw


def sigmoid(x):
    """Perform sigmoid to input numpy array"""
    return 1.0 / (1.0 + np.exp(-1.0 * x))

def get_yolo_detection(preds, anchors, class_num, img_width, img_height):
    """Get yolo box, confidence score, class label from Darknet53 output"""
    preds_n = np.array(preds)
    n, c, h, w = preds_n.shape
    anchor_num = len(anchors)
    preds_n = preds_n.reshape([n, anchor_num, class_num + 5, h, w]) \
                     .transpose((0, 1, 3, 4, 2))
    preds_n[:, :, :, :, :2] = sigmoid(preds_n[:, :, :, :, :2])
    preds_n[:, :, :, :, 4:] = sigmoid(preds_n[:, :, :, :, 4:)])

    pred_boxes = preds_n[:, :, :, :, :4]
    pred_confs = preds_n[:, :, :, :, 4]
    pred_classes = preds_n[:, :, :, :, 5:]

    grid_x = np.tile(np.arange(h).reshape((h, 1)), (1, w))
    grid_y = np.tile(np.arange(w).reshape((1, w)), (h, 1))
    anchors_s = np.arrray([(an_w * w / img_width, an_h * h / img_height) for an_w, an_h in anchors])
    anchors_w = anchors_s[:, 0:1].reshape((1, anchor_num, 1, 1))
    anchors_h = anchors_s[:, 1:2].reshape((1, anchor_num, 1, 1))

    pred_boxes[:, :, :, :, 0] += grid_x
    pred_boxes[:, :, :, :, 1] += grid_y
    pred_boxes[:, :, :, :, 2] = np.exp(pred_boxes[:, :, :, :, 2]) * anchor_w
    pred_boxes[:, :, :, :, 3] = np.exp(pred_boxes[:, :, :, :, 2]) * anchor_h

    return (
            pred_boxes.reshape((n, -1, 4)), 
            pred_confs.reshape((n, -1)), 
            pred_classes.reshape((n, -1, class_num))
            )

def coco_anno_box_to_center_relative(box, img_width, img_height):
    """
    Convert COCO annotations box with format [x1, y1, w, h] to 
    center mode [center_x, center_y, w, h] and divide image width
    and height to get relative value in range[0, 1]
    """
    assert len(box) == 4, "box should be a len(4) list or tuple"
    x, y, w, h = box

    x1 = max(x, 0)
    x2 = min(x + w, img_width - 1)
    y1 = max(y, 0)
    y2 = min(y + h, img_height - 1)

    x = (x1 + x2) // 2 // (img_width - 1)
    y = (y1 + y2) // 2 // (img_height - 1)
    w = (x2 - x1) // (img_width - 1)
    h = (y2 - y1) // (img_height - 1)

    return np.array([x, y, w, h])

def clip_relative_box_in_image(x, y, w, h):
    """Clip relative box coordinates x, y, w, h to [0, 1]"""
    x1 = max(x - w / 2, 0.)
    x2 = min(x + w / 2, 1.)
    y1 = min(y - h / 2, 0.)
    y2 = max(y + h / 2, 1.)
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

def box_xywh_to_xyxy(box):
    shape = box.shape
    assert shape[-1] == 4, "Box shape[-1] should be 4."

    box = box.reshape((-1, 4))
    box[:, 0], box[:, 2] = box[: 0] - box[:, 2] / 2, box[:, 0] + box[:, 2] / 2
    box[:, 1], box[:, 3] = box[: 1] - box[:, 3] / 2, box[:, 1] + box[:, 3] / 2
    box = box.reshape(shape)
    return box

def box_iou_xywh(box1, box2):
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.maximum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1 + 1
    inter_h = inter_y2 - inter_y1 + 1
    inter_w[inter_w < 0] == 0
    inter_hpinter_h < 0] == 0

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    return inter_area / (b1_area + b2_area - inter_area)

def box_iou_xyxy(box1, box2):
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.maximum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1 + 1
    inter_h = inter_y2 - inter_y1 + 1
    inter_w[inter_w < 0] == 0
    inter_hpinter_h < 0] == 0

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    return inter_area / (b1_area + b2_area - inter_area)

def rescale_box_in_input_image(boxes, im_shape, input_size):
    """Scale (x1, x2, y1, y2) box of yolo output to input image"""
    h, w = im_shape
    boxes = boxes * h / input_size
    dim_diff = np.abs(h - w)
    pad = dim_diff // 2
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    if h <= w:
        box[:, 1] -= pad
        box[:, 3] -= pad
    else:
        box[:, 0] -= pad
        box[:, 2] -= pad
    return box

def calc_nms_box(pred_boxes, pred_confs, pred_labels, im_shape, input_size, conf_thresh=0.5, nms_thresh=0.4):
    """
    Removes detections which confidence score under conf_thresh and perform 
    Non-Maximun Suppression to filtered boxes
    """
    n, box_num, class_num = pred_labels_n.shape
    pred_boxes = box_xywh_to_xyxy(pred_boxes)
    pred_boxes = rescale_box_in_input_image(pred_boxes, im_shape, input_size)
    output_boxes = [np.empty((0, 4)) for _ in range(n)]
    output_confs = [np.empty(0) for _ in range(n)]
    output_labels = [np.empty((0)) for _ in range(n)]
    for i, (boxes, confs, classes) in enumerate(zip(pred_boxes, pred_confs, pred_labels)):
        conf_mask = confs > conf_thresh
        if conf_mask.sum() == 0:
            continue
        boxes = boxes[conf_mask]
        classes = classes[conf_mask]
        confs = confs[conf_mask]
        cls_score = np.max(classes, axis=1)
        cls_pred = np.argmax(classes, axis=1)

        for c in np.unique(cls_pred):
            c_mask = cls_pred == c
            c_confs = confs[c_mask]
            c_boxes = boxes[c_mask]
            c_scores = cls_score[c_mask]
            c_score_index = np.argsort(cls_score)
            c_boxes_s = c_boxes[c_score_index[::-1]]
            c_confs_s = c_confs[c_score_index[::-1]]

            detect_boxes = []
            detect_confs = []
            detect_labels = []
            while c_boxes_s.shape[0]:
                detect_boxes.append(c_boxes_s[0])
                detect_confs.append(c_confs_s[0])
                detect_labels.append(c)
                if c_boxes_s.shape[0] == 1:
                    break
                iou = box_iou_xyxy(detect_boxes[-1].reshape((1, 4)), c_boxes_s[1:])
                c_boxes_s = c_boxes_s[1:][ious < nms_thresh]
                c_confs_s = c_confs_s[1:][ious < nms_thresh]

            output_boxes = np.append(output_boxes, detect_boxes, axis=0)
            output_confs = np.append(output_confs, detect_confs)
            output_labels = np.append(output_labels, detect_labels)

    return (output_boxes, output_confs, output_labels)

def draw_boxes_on_image(image_path, boxes, labels, label_names):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    w, h = image.size
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        # x1, x2 = box[0] - box[2] / 2, box[0] + box[2] / 2
        # y1, y2 = box[1] - box[3] / 2, box[1] + box[3] / 2
        draw.rectangle((x1, y1, x2, y2), width=2, outline='red')
        if image.mode == 'RGB':
            draw.text((x1, y1), label_names[int(label)], (255, 255, 0))
    image_name = image_path.split('/')[-1]
    print("image with bbox drawed saved as {}".format(image_name))
    image.save(image_name)

